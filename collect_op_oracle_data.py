"""Collect a small counterfactual-oracle dataset for OP-SieveKV.

This pilot is intentionally cheaper than a full compressed-cache on-policy
oracle. For selected semantic segments, it removes the segment from the prompt
tokens and measures the change in gold-answer log probability. The resulting
oracle labels are mixed with the budget-aware v3 cheap labels from
`collect_op_distill_data.py`.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from attention_tracker import AttentionTracker
from collect_op_distill_data import (
    budget_pressure,
    build_evidence_mask_from_offsets,
    build_v2_soft_label,
    make_policy_for_features,
    segment_content_signal,
    select_teacher_support_segments,
)
from config import ExperimentConfig
from eval_multi_needle import insert_needles_at_depths, make_multi_needles
from eval_niah import NEEDLES, build_haystack_prompt
from eval_ruler_niah import build_pg_haystack
from op_policy_model import FEATURE_NAMES, load_policy_checkpoint
from run_generation import _extract_latest_user_query, load_model
from semantic_analyzer import RoleTag


def _as_1d_cpu(ids: torch.Tensor) -> torch.Tensor:
    ids = ids.detach().cpu()
    if ids.dim() == 2:
        ids = ids[0]
    return ids.to(dtype=torch.long)


@torch.no_grad()
def answer_avg_logprob(
    model,
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
) -> float:
    """Return average next-token log probability of answer_ids after prompt_ids."""
    prompt_ids = _as_1d_cpu(prompt_ids)
    answer_ids = _as_1d_cpu(answer_ids)
    if answer_ids.numel() == 0:
        return 0.0

    input_ids = torch.cat([prompt_ids, answer_ids], dim=0).unsqueeze(0).to(model.device)
    outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]

    start = max(0, int(prompt_ids.numel()) - 1)
    end = start + int(answer_ids.numel())
    answer_logits = logits[:, start:end, :]
    answer_targets = targets[:, start:end]
    log_probs = torch.log_softmax(answer_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, answer_targets.unsqueeze(-1)).squeeze(-1)
    value = float(token_log_probs.mean().item())
    del outputs, logits, targets, answer_logits, answer_targets, log_probs, token_log_probs
    return value


def drop_positions(input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Return prompt ids with positions removed."""
    ids = _as_1d_cpu(input_ids)
    keep = torch.ones(ids.numel(), dtype=torch.bool)
    valid_positions = positions[(positions >= 0) & (positions < ids.numel())].to(dtype=torch.long)
    keep[valid_positions] = False
    return ids[keep]


def oracle_label_from_delta(delta: float, args) -> float:
    """Map logprob drop delta to a keep probability."""
    centered = (float(delta) - args.oracle_positive_delta) / max(args.oracle_temperature, 1e-6)
    return float(torch.sigmoid(torch.tensor(centered)).item())


def is_protected_cheap_reason(reason: str) -> bool:
    return reason in {
        "evidence",
        "evidence_neighbor",
        "query_support",
        "system_support",
        "boundary_support",
        "pinned_support",
        "recent_support",
    }


def mix_oracle_label(
    *,
    cheap_label: float,
    cheap_weight: float,
    cheap_reason: str,
    oracle_label: float,
    budget: float,
    args,
) -> tuple[float, float, str]:
    """Mix an oracle label with a cheap label.

    Direct override is useful for probing oracle quality, but it can be too
    destructive at the lowest budgets. Conservative mode treats oracle drops as
    corrections mainly for teacher/background labels while protecting explicit
    evidence, query, and structural support.
    """
    if args.oracle_mix_mode == "override":
        label = max(cheap_label, oracle_label) if cheap_reason == "evidence" else oracle_label
        return label, max(cheap_weight, args.oracle_weight), "oracle_keep" if label >= 0.5 else "oracle_drop"

    pressure = budget_pressure(budget, args)
    keep_mix = args.oracle_keep_mix
    drop_mix = args.oracle_drop_mix * (1.0 - pressure * (1.0 - args.low_budget_oracle_drop_scale))
    weight = max(cheap_weight, args.oracle_weight)

    if oracle_label >= 0.5:
        label = max(cheap_label, cheap_label * (1.0 - keep_mix) + oracle_label * keep_mix)
        return label, weight, "oracle_keep"

    if is_protected_cheap_reason(cheap_reason):
        # Keep hard support mostly intact. This is especially important for
        # aggressive budgets where one bad oracle drop can erase the only
        # surviving answer span.
        protected_floor = args.protected_oracle_floor
        if pressure > 0:
            protected_floor = max(protected_floor, args.low_budget_protected_oracle_floor)
        label = max(protected_floor, cheap_label * (1.0 - drop_mix) + oracle_label * drop_mix)
        label = max(label, min(cheap_label, protected_floor))
        return label, cheap_weight, "oracle_protected_drop"

    label = cheap_label * (1.0 - drop_mix) + oracle_label * drop_mix
    return label, weight, "oracle_mixed_drop" if label < 0.5 else "oracle_soft_drop"


def _segment_mask_fraction(mask: torch.Tensor | None, positions: torch.Tensor, seq_len: int) -> float:
    if mask is None or len(mask) < seq_len:
        return 0.0
    return float(mask[:seq_len][positions].float().mean().item())


def is_structural_segment(policy, positions: torch.Tensor, seq_len: int, args) -> bool:
    """Avoid spending oracle calls on template/query/system segments."""
    if _segment_mask_fraction(policy.pinned_mask, positions, seq_len) >= args.structural_fraction:
        return True
    if _segment_mask_fraction(policy.question_tail_mask, positions, seq_len) >= args.structural_fraction:
        return True
    if _segment_mask_fraction(policy.question_like_mask, positions, seq_len) >= args.structural_fraction:
        return True
    if _segment_mask_fraction(policy.chat_boundary_mask, positions, seq_len) >= args.structural_fraction:
        return True
    if policy.role_tags is not None and len(policy.role_tags) >= seq_len:
        role_values = policy.role_tags[:seq_len][positions]
        system_frac = float((role_values == RoleTag.SYSTEM).float().mean().item())
        if system_frac >= args.structural_fraction:
            return True
    return False


def select_oracle_candidate_segments(
    *,
    policy,
    segment_positions: list[torch.Tensor],
    heuristic_probs: torch.Tensor,
    student_probs: torch.Tensor | None,
    evidence_segment_indices: set[int],
    seq_len: int,
    args,
) -> list[int]:
    """Select a small candidate set for expensive oracle scoring."""
    selected: set[int] = set()

    for evidence_index in evidence_segment_indices:
        for offset in range(-args.oracle_evidence_neighbors, args.oracle_evidence_neighbors + 1):
            idx = evidence_index + offset
            if 0 <= idx < len(segment_positions):
                selected.add(idx)

    mutable: list[int] = []
    for idx, positions in enumerate(segment_positions):
        if idx in selected:
            continue
        if is_structural_segment(policy, positions, seq_len, args):
            continue
        mutable.append(idx)

    def heuristic_prob(idx: int) -> float:
        return float(heuristic_probs[idx].item()) if heuristic_probs.numel() else 0.0

    def policy_prob(idx: int) -> float:
        if student_probs is not None and student_probs.numel():
            return float(student_probs[idx].item())
        return heuristic_prob(idx)

    top = sorted(mutable, key=policy_prob, reverse=True)[: args.oracle_top_k]
    bottom = sorted(mutable, key=policy_prob)[: args.oracle_bottom_k]
    uncertain = sorted(mutable, key=lambda idx: abs(policy_prob(idx) - 0.5))[: args.oracle_uncertain_k]
    for idx in [*top, *bottom, *uncertain]:
        selected.add(idx)

    if student_probs is not None and student_probs.numel():
        # Student-policy missed-keep probes: segments that the teacher/content
        # still likes but the learned policy is likely to drop.
        missed_candidates: list[tuple[float, int]] = []
        for idx in mutable:
            content_signal = segment_content_signal(policy, segment_positions[idx], seq_len)
            if content_signal < args.student_min_content_signal:
                continue
            gap = heuristic_prob(idx) - policy_prob(idx)
            missed_candidates.append((gap + 0.25 * content_signal, idx))
        missed_candidates.sort(reverse=True)
        for _score, idx in missed_candidates[: args.student_missed_k]:
            selected.add(idx)

        # Student-policy over-keep probes: high student probability with low
        # oracle/content support can reveal confident keep mistakes.
        overkeep_candidates = sorted(mutable, key=policy_prob, reverse=True)
        for idx in overkeep_candidates[: args.student_overkeep_k]:
            selected.add(idx)

    ranked = sorted(selected, key=lambda idx: (idx not in evidence_segment_indices, -policy_prob(idx), idx))
    if args.oracle_max_candidates > 0:
        ranked = ranked[: args.oracle_max_candidates]
    return ranked


@torch.no_grad()
def compute_student_probs(features: torch.Tensor, candidate_policy_bundle) -> torch.Tensor | None:
    if candidate_policy_bundle is None or features.numel() == 0:
        return None
    model, feature_mean, feature_std = candidate_policy_bundle
    normalized = (features.detach().cpu().float() - feature_mean) / feature_std
    return torch.sigmoid(model(normalized)).detach().cpu()


def build_niah_examples(args) -> list[tuple[list[dict], list[str], str, dict]]:
    examples = []
    for position in args.positions:
        for needle_index, needle in enumerate(NEEDLES):
            messages = build_haystack_prompt(
                needle,
                haystack_length=args.haystack_length,
                needle_position=position,
            )
            answer_text = needle["fact"]
            examples.append(
                (
                    messages,
                    [needle["fact"]],
                    answer_text,
                    {
                        "task": "niah",
                        "needle_index": needle_index,
                        "needle_position": position,
                    },
                )
            )
    return examples


def build_multi_needle_examples(tokenizer, args) -> list[tuple[list[dict], list[str], str, dict]]:
    rng = random.Random(args.seed)
    examples = []
    for num_needles in args.needle_counts:
        for trial in range(args.multi_trials):
            key, question, facts, values = make_multi_needles(num_needles, rng)
            depths = list(np.linspace(0.0, 1.0, num=num_needles, endpoint=True))
            haystack = build_pg_haystack(tokenizer, args.multi_target_tokens)
            text_with_needles = insert_needles_at_depths(haystack, facts, depths)
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer questions based only on the provided text.",
                },
                {"role": "user", "content": f"{text_with_needles}\n\n{question}"},
            ]
            answer_text = ", ".join(values)
            examples.append(
                (
                    messages,
                    facts,
                    answer_text,
                    {
                        "task": "multi_needle",
                        "trial": trial,
                        "num_needles": num_needles,
                        "needle_key": key,
                        "needle_values": values,
                    },
                )
            )
    return examples


@torch.no_grad()
def collect_one_example(
    model,
    tokenizer,
    messages: list[dict],
    evidence_texts: list[str],
    answer_text: str,
    budgets: list[float],
    args,
    metadata: dict,
    candidate_policy_bundle=None,
) -> tuple[list[torch.Tensor], list[float], list[float], list[dict]]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = encoding.pop("offset_mapping")[0]
    inputs = encoding.to(model.device)
    input_ids = inputs["input_ids"][0]
    seq_len = int(input_ids.numel())
    answer_ids = tokenizer(answer_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        use_cache=False,
        output_attentions=True,
        return_dict=True,
    )
    tracker = AttentionTracker(
        num_layers=getattr(model.config, "num_hidden_layers", 36),
        num_kv_heads=getattr(model.config, "num_key_value_heads", 2),
    )
    tracker.update(outputs.attentions)

    evidence_mask = build_evidence_mask_from_offsets(text, offset_mapping, evidence_texts)
    latest_query = _extract_latest_user_query(messages)
    full_logprob = answer_avg_logprob(model, input_ids, answer_ids)

    feature_rows: list[torch.Tensor] = []
    labels: list[float] = []
    weights: list[float] = []
    row_metadata: list[dict] = []
    oracle_cache: dict[tuple[int, int], dict] = {}

    for budget in budgets:
        policy = make_policy_for_features(
            tokenizer=tokenizer,
            tracker=tracker,
            input_ids=input_ids,
            latest_query=latest_query,
            budget=budget,
            args=args,
        )
        features, segment_positions, heuristic_probs = policy.compute_segment_features(seq_len)
        student_probs = compute_student_probs(features, candidate_policy_bundle)
        evidence_segment_indices = {
            segment_index
            for segment_index, positions in enumerate(segment_positions)
            if bool(evidence_mask[positions].any().item())
        }
        teacher_segment_indices = select_teacher_support_segments(
            policy=policy,
            segment_positions=segment_positions,
            heuristic_probs=heuristic_probs,
            evidence_segment_indices=evidence_segment_indices,
            budget=budget,
            seq_len=seq_len,
            args=args,
        )

        oracle_candidate_indices = select_oracle_candidate_segments(
            policy=policy,
            segment_positions=segment_positions,
            heuristic_probs=heuristic_probs,
            student_probs=student_probs,
            evidence_segment_indices=evidence_segment_indices,
            seq_len=seq_len,
            args=args,
        )

        for row_idx in oracle_candidate_indices:
            positions = segment_positions[row_idx]
            cache_key = (int(positions[0].item()), int(positions[-1].item()) + 1)
            if cache_key in oracle_cache:
                continue
            dropped_prompt_ids = drop_positions(input_ids, positions)
            dropped_logprob = answer_avg_logprob(model, dropped_prompt_ids, answer_ids)
            delta = full_logprob - dropped_logprob
            oracle_label = oracle_label_from_delta(delta, args)
            oracle_cache[cache_key] = {
                "oracle_delta": delta,
                "oracle_label": oracle_label,
                "dropped_logprob": dropped_logprob,
                "full_logprob": full_logprob,
            }

        for row_idx, positions in enumerate(segment_positions):
            heuristic_prob = float(heuristic_probs[row_idx].item()) if heuristic_probs.numel() else 0.0
            cheap_label, cheap_weight, label_reason = build_v2_soft_label(
                policy=policy,
                positions=positions,
                segment_index=row_idx,
                evidence_segment_indices=evidence_segment_indices,
                heuristic_prob=heuristic_prob,
                teacher_allowed=row_idx in teacher_segment_indices,
                budget=budget,
                seq_len=seq_len,
                args=args,
            )
            pressure = budget_pressure(budget, args)
            if pressure > 0.0:
                cheap_weight *= (1.0 + pressure * (args.low_budget_row_weight - 1.0))

            label = cheap_label
            weight = cheap_weight
            oracle_info = None
            cache_key = (int(positions[0].item()), int(positions[-1].item()) + 1)
            if cache_key in oracle_cache:
                oracle_info = oracle_cache[cache_key]
                oracle_label = float(oracle_info["oracle_label"])
                label, weight, label_reason = mix_oracle_label(
                    cheap_label=cheap_label,
                    cheap_weight=cheap_weight,
                    cheap_reason=label_reason,
                    oracle_label=oracle_label,
                    budget=budget,
                    args=args,
                )

            evidence_overlap = bool(evidence_mask[positions].any().item())
            evidence_overlap_frac = float(evidence_mask[positions].float().mean().item())
            feature_rows.append(features[row_idx])
            labels.append(label)
            weights.append(weight)
            row_metadata.append(
                {
                    **metadata,
                    "budget": budget,
                    "segment_start": int(positions[0].item()),
                    "segment_end": int(positions[-1].item()) + 1,
                    "label": label,
                    "label_reason": label_reason,
                    "cheap_label": cheap_label,
                    "evidence_overlap": evidence_overlap,
                    "evidence_overlap_frac": evidence_overlap_frac,
                    "heuristic_keep_prob": heuristic_prob,
                    "candidate_student_keep_prob": (
                        None if student_probs is None or not student_probs.numel() else float(student_probs[row_idx].item())
                    ),
                    "teacher_allowed": row_idx in teacher_segment_indices,
                    "oracle_scored": oracle_info is not None,
                    "oracle_delta": None if oracle_info is None else float(oracle_info["oracle_delta"]),
                    "oracle_label": None if oracle_info is None else float(oracle_info["oracle_label"]),
                    "full_answer_avg_logprob": full_logprob,
                    "budget_pressure": pressure,
                }
            )

    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feature_rows, labels, weights, row_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect OP-SieveKV oracle pilot data")
    parser.add_argument("--output", default="results/op_oracle_dataset_pilot.pt")
    parser.add_argument("--tasks", nargs="+", default=["niah", "multi"], choices=["niah", "multi"])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.3, 0.2, 0.1, 0.05])
    parser.add_argument("--positions", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--haystack-length", type=int, default=1200)
    parser.add_argument("--needle-counts", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--multi-trials", type=int, default=2)
    parser.add_argument("--multi-target-tokens", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--op-max-segment-tokens", type=int, default=32)
    parser.add_argument("--op-min-segment-tokens", type=int, default=4)
    parser.add_argument("--op-recent-window", type=int, default=16)
    parser.add_argument("--op-uncertainty-weight", type=float, default=0.15)

    # Budget-aware cheap-label parameters, mirrored from collect_op_distill_data.py.
    parser.add_argument("--background-label", type=float, default=0.03)
    parser.add_argument("--budget-aware-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--budget-pressure-low", type=float, default=0.05)
    parser.add_argument("--budget-pressure-high", type=float, default=0.30)
    parser.add_argument("--teacher-mix", type=float, default=0.35)
    parser.add_argument("--low-budget-teacher-mix", type=float, default=0.20)
    parser.add_argument("--teacher-threshold", type=float, default=0.70)
    parser.add_argument("--low-budget-teacher-threshold", type=float, default=0.86)
    parser.add_argument("--teacher-signal-threshold", type=float, default=0.20)
    parser.add_argument("--low-budget-teacher-signal-threshold", type=float, default=0.38)
    parser.add_argument("--teacher-top-fraction", type=float, default=0.12)
    parser.add_argument("--low-budget-teacher-top-fraction", type=float, default=0.035)
    parser.add_argument("--teacher-top-min", type=int, default=2)
    parser.add_argument("--low-budget-teacher-top-min", type=int, default=1)
    parser.add_argument("--teacher-top-max", type=int, default=24)
    parser.add_argument("--low-budget-teacher-top-max", type=int, default=6)
    parser.add_argument("--teacher-budget-token-fraction", type=float, default=0.55)
    parser.add_argument("--low-budget-teacher-budget-token-fraction", type=float, default=0.28)
    parser.add_argument("--teacher-cap", type=float, default=0.85)
    parser.add_argument("--low-budget-teacher-cap", type=float, default=0.70)
    parser.add_argument("--teacher-weight", type=float, default=2.0)
    parser.add_argument("--evidence-weight", type=float, default=8.0)
    parser.add_argument("--evidence-neighbor-segments", type=int, default=1)
    parser.add_argument("--neighbor-label", type=float, default=0.70)
    parser.add_argument("--low-budget-neighbor-label", type=float, default=0.55)
    parser.add_argument("--neighbor-weight", type=float, default=4.0)
    parser.add_argument("--pinned-label", type=float, default=0.80)
    parser.add_argument("--low-budget-pinned-label", type=float, default=0.75)
    parser.add_argument("--query-label", type=float, default=0.80)
    parser.add_argument("--low-budget-query-label", type=float, default=0.75)
    parser.add_argument("--system-label", type=float, default=0.75)
    parser.add_argument("--low-budget-system-label", type=float, default=0.70)
    parser.add_argument("--recent-label", type=float, default=0.45)
    parser.add_argument("--low-budget-recent-label", type=float, default=0.25)
    parser.add_argument("--low-budget-background-scale", type=float, default=0.35)
    parser.add_argument("--low-budget-row-weight", type=float, default=1.6)
    parser.add_argument("--support-weight", type=float, default=3.0)
    parser.add_argument("--support-min-fraction", type=float, default=0.25)

    # Oracle pilot parameters.
    parser.add_argument("--oracle-max-candidates", type=int, default=14)
    parser.add_argument("--oracle-evidence-neighbors", type=int, default=1)
    parser.add_argument("--oracle-top-k", type=int, default=5)
    parser.add_argument("--oracle-bottom-k", type=int, default=2)
    parser.add_argument("--oracle-uncertain-k", type=int, default=4)
    parser.add_argument("--oracle-positive-delta", type=float, default=0.05)
    parser.add_argument("--oracle-temperature", type=float, default=0.08)
    parser.add_argument("--oracle-weight", type=float, default=10.0)
    parser.add_argument("--oracle-mix-mode", choices=["override", "conservative"], default="conservative")
    parser.add_argument("--oracle-keep-mix", type=float, default=0.80)
    parser.add_argument("--oracle-drop-mix", type=float, default=0.65)
    parser.add_argument("--low-budget-oracle-drop-scale", type=float, default=0.35)
    parser.add_argument("--protected-oracle-floor", type=float, default=0.55)
    parser.add_argument("--low-budget-protected-oracle-floor", type=float, default=0.70)
    parser.add_argument("--structural-fraction", type=float, default=0.5)
    parser.add_argument(
        "--candidate-policy-ckpt",
        default=None,
        help="Optional learned policy checkpoint used to choose oracle candidates from student decisions.",
    )
    parser.add_argument("--student-missed-k", type=int, default=4)
    parser.add_argument("--student-overkeep-k", type=int, default=4)
    parser.add_argument("--student-min-content-signal", type=float, default=0.10)
    args = parser.parse_args()

    print(
        "Oracle collector config: "
        f"mix_mode={args.oracle_mix_mode}, "
        f"keep_mix={args.oracle_keep_mix}, drop_mix={args.oracle_drop_mix}, "
        f"low_budget_drop_scale={args.low_budget_oracle_drop_scale}, "
        f"protected_floor={args.protected_oracle_floor}, "
        f"low_budget_protected_floor={args.low_budget_protected_oracle_floor}"
    )

    config = ExperimentConfig()
    model, tokenizer = load_model(config.model)

    candidate_policy_bundle = None
    if args.candidate_policy_ckpt:
        candidate_policy, candidate_feature_mean, candidate_feature_std, candidate_metadata = load_policy_checkpoint(
            args.candidate_policy_ckpt,
            map_location="cpu",
        )
        candidate_policy_bundle = (candidate_policy, candidate_feature_mean, candidate_feature_std)
        print(
            "Candidate policy loaded: "
            f"{args.candidate_policy_ckpt}, "
            f"best_val_loss={candidate_metadata.get('best_val_loss', 'n/a')}"
        )

    examples: list[tuple[list[dict], list[str], str, dict]] = []
    if "niah" in args.tasks:
        examples.extend(build_niah_examples(args))
    if "multi" in args.tasks:
        examples.extend(build_multi_needle_examples(tokenizer, args))

    all_features: list[torch.Tensor] = []
    all_labels: list[float] = []
    all_weights: list[float] = []
    all_metadata: list[dict] = []

    for index, (messages, evidence_texts, answer_text, metadata) in enumerate(examples, start=1):
        print(f"[{index}/{len(examples)}] oracle collect {metadata}")
        rows, labels, weights, row_metadata = collect_one_example(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            evidence_texts=evidence_texts,
            answer_text=answer_text,
            budgets=args.budgets,
            args=args,
            metadata=metadata,
            candidate_policy_bundle=candidate_policy_bundle,
        )
        all_features.extend(rows)
        all_labels.extend(labels)
        all_weights.extend(weights)
        all_metadata.extend(row_metadata)

    features = torch.stack(all_features, dim=0) if all_features else torch.empty(0, len(FEATURE_NAMES))
    labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
    weights_tensor = torch.tensor(all_weights, dtype=torch.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "labels": labels_tensor,
            "weights": weights_tensor,
            "feature_names": FEATURE_NAMES,
            "metadata": all_metadata,
            "args": vars(args),
        },
        output_path,
    )

    reason_counts = Counter(item.get("label_reason", "unknown") for item in all_metadata)
    hard_reason_counts = Counter(
        item.get("label_reason", "unknown")
        for item, label in zip(all_metadata, all_labels)
        if label >= 0.5
    )
    oracle_scored = [item for item in all_metadata if item.get("oracle_scored")]
    oracle_deltas = [float(item["oracle_delta"]) for item in oracle_scored if item.get("oracle_delta") is not None]
    budget_rows: dict[float, list[float]] = {}
    budget_oracle: Counter = Counter()
    for item, label in zip(all_metadata, all_labels):
        budget = round(float(item.get("budget", 0.0)), 4)
        budget_rows.setdefault(budget, []).append(float(label))
        if item.get("oracle_scored"):
            budget_oracle[budget] += 1

    hard_positives = int((labels_tensor >= 0.5).sum().item()) if labels_tensor.numel() else 0
    label_mass = float(labels_tensor.sum().item()) if labels_tensor.numel() else 0.0
    print(
        f"Saved {features.shape[0]} rows, hard_positives={hard_positives}, "
        f"label_mass={label_mass:.1f}, oracle_rows={len(oracle_scored)}, path={output_path}"
    )
    if oracle_deltas:
        delta_tensor = torch.tensor(oracle_deltas)
        print(
            "Oracle delta stats: "
            f"min={float(delta_tensor.min().item()):.4f}, "
            f"mean={float(delta_tensor.mean().item()):.4f}, "
            f"max={float(delta_tensor.max().item()):.4f}"
        )
    print(f"Label reasons: {dict(reason_counts)}")
    print(f"Hard-positive reasons: {dict(hard_reason_counts)}")
    print("Budget label stats:")
    for budget in sorted(budget_rows):
        values = budget_rows[budget]
        print(
            f"  b={budget:g}: rows={len(values)}, oracle={budget_oracle[budget]}, "
            f"label_mass={sum(values):.1f}"
        )


if __name__ == "__main__":
    main()
