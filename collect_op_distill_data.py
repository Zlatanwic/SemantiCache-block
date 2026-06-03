"""Collect cheap-oracle segment data for OP-SieveKV policy distillation.

This MVP labels segments that overlap known NIAH / multi-needle evidence spans
as positive. It is intentionally cheaper than counterfactual drop/restore
oracle labeling, and provides the first supervised policy checkpoint.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from attention_tracker import AttentionTracker
from config import ExperimentConfig
from eval_multi_needle import insert_needles_at_depths, make_multi_needles
from eval_niah import NEEDLES, build_haystack_prompt
from eval_ruler_niah import build_pg_haystack
from eviction_policies import OPSieveKVLitePolicy
from op_policy_model import FEATURE_NAMES
from run_generation import _extract_latest_user_query, load_model
from semantic_analyzer import RoleTag, SemanticAnalyzer


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def _lerp(start: float, end: float, amount: float) -> float:
    return float(start) + (float(end) - float(start)) * _clamp(amount, 0.0, 1.0)


def budget_pressure(budget: float, args) -> float:
    """Return 0 for loose budgets and 1 for the most aggressive budget."""
    if not args.budget_aware_labels:
        return 0.0
    high = max(args.budget_pressure_high, args.budget_pressure_low + 1e-6)
    return _clamp((high - float(budget)) / (high - args.budget_pressure_low), 0.0, 1.0)


def budget_value(base: float, aggressive: float, budget: float, args) -> float:
    return _lerp(base, aggressive, budget_pressure(budget, args))


def budget_int_value(base: int, aggressive: int, budget: float, args) -> int:
    return int(round(budget_value(float(base), float(aggressive), budget, args)))


def find_text_spans(text: str, needle: str) -> list[tuple[int, int]]:
    """Return all character spans where needle occurs in text."""
    if not needle:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            break
        spans.append((index, index + len(needle)))
        start = index + max(1, len(needle))
    return spans


def build_evidence_mask_from_offsets(
    prompt_text: str,
    offset_mapping: torch.Tensor,
    evidence_texts: list[str],
) -> torch.Tensor:
    """Mark tokens whose character offsets overlap known evidence spans."""
    offsets = offset_mapping.detach().cpu()
    mask = torch.zeros(offsets.shape[0], dtype=torch.bool)
    matched_spans = 0
    for evidence_text in evidence_texts:
        for char_start, char_end in find_text_spans(prompt_text, evidence_text):
            matched_spans += 1
            token_start = offsets[:, 0]
            token_end = offsets[:, 1]
            valid = token_end > token_start
            overlaps = valid & (token_start < char_end) & (token_end > char_start)
            mask |= overlaps
    if matched_spans == 0:
        preview = evidence_texts[0][:80] if evidence_texts else ""
        print(f"  warning: evidence text not found in prompt: {preview!r}")
    return mask


def make_policy_for_features(
    tokenizer,
    tracker: AttentionTracker,
    input_ids: torch.Tensor,
    latest_query: str,
    budget: float,
    args,
) -> OPSieveKVLitePolicy:
    analyzer = SemanticAnalyzer(tokenizer)
    policy = OPSieveKVLitePolicy(
        tracker=tracker,
        analyzer=analyzer,
        pin_system=True,
        pin_latest_user=True,
        recent_window_size=args.op_recent_window,
        max_segment_tokens=args.op_max_segment_tokens,
        min_segment_tokens=args.op_min_segment_tokens,
        uncertainty_weight=args.op_uncertainty_weight,
        budget_ratio=budget,
    )
    policy.setup_semantic_signals(input_ids.detach().cpu(), latest_query_text=latest_query)
    return policy


def _segment_mask_fraction(mask: torch.Tensor | None, positions: torch.Tensor, seq_len: int) -> float:
    if mask is None or len(mask) < seq_len:
        return 0.0
    return float(mask[:seq_len][positions].float().mean().item())


def build_v2_soft_label(
    *,
    policy: OPSieveKVLitePolicy,
    positions: torch.Tensor,
    segment_index: int,
    evidence_segment_indices: set[int],
    heuristic_prob: float,
    teacher_allowed: bool,
    budget: float,
    seq_len: int,
    args,
) -> tuple[float, float, str]:
    """Build a soft oracle label that includes evidence, support, and teacher signals."""
    pressure = budget_pressure(budget, args)
    evidence_hit = segment_index in evidence_segment_indices
    neighbor_hit = (
        args.evidence_neighbor_segments > 0
        and any(abs(segment_index - evidence_index) <= args.evidence_neighbor_segments for evidence_index in evidence_segment_indices)
    )
    if evidence_hit:
        return 1.0, args.evidence_weight, "evidence"

    teacher_mix = budget_value(args.teacher_mix, args.low_budget_teacher_mix, budget, args)
    label = max(args.background_label, teacher_mix * heuristic_prob)
    weight = 1.0
    reason = "teacher" if heuristic_prob > args.background_label else "background"

    if neighbor_hit:
        label = max(label, budget_value(args.neighbor_label, args.low_budget_neighbor_label, budget, args))
        weight = max(weight, args.neighbor_weight)
        reason = "evidence_neighbor"

    pinned_frac = _segment_mask_fraction(policy.pinned_mask, positions, seq_len)
    question_tail_frac = _segment_mask_fraction(policy.question_tail_mask, positions, seq_len)
    question_like_frac = _segment_mask_fraction(policy.question_like_mask, positions, seq_len)
    boundary_frac = _segment_mask_fraction(policy.chat_boundary_mask, positions, seq_len)

    if pinned_frac > 0:
        label = max(label, budget_value(args.pinned_label, args.low_budget_pinned_label, budget, args))
        weight = max(weight, args.support_weight)
        reason = "pinned_support"
    if question_tail_frac >= args.support_min_fraction or question_like_frac >= args.support_min_fraction:
        label = max(label, budget_value(args.query_label, args.low_budget_query_label, budget, args))
        weight = max(weight, args.support_weight)
        reason = "query_support"

    if policy.role_tags is not None and len(policy.role_tags) >= seq_len:
        role_values = policy.role_tags[:seq_len][positions]
        system_frac = float((role_values == RoleTag.SYSTEM).float().mean().item())
        if system_frac > 0.5:
            label = max(label, budget_value(args.system_label, args.low_budget_system_label, budget, args))
            weight = max(weight, args.support_weight)
            reason = "system_support"

    recent_start = max(0, seq_len - max(0, args.op_recent_window))
    recent_frac = float((positions >= recent_start).float().mean().item()) if positions.numel() > 0 else 0.0
    if recent_frac >= args.support_min_fraction:
        label = max(label, budget_value(args.recent_label, args.low_budget_recent_label, budget, args))
        weight = max(weight, args.support_weight)
        reason = "recent_support"

    if boundary_frac >= args.support_min_fraction:
        boundary_label = min(budget_value(args.system_label, args.low_budget_system_label, budget, args), 0.65)
        label = max(label, boundary_label)
        weight = max(weight, 2.0)
        reason = "boundary_support"

    if teacher_allowed:
        teacher_cap = budget_value(args.teacher_cap, args.low_budget_teacher_cap, budget, args)
        label = max(label, min(teacher_cap, heuristic_prob))
        weight = max(weight, args.teacher_weight)
        reason = "heuristic_support"

    # Under aggressive budgets, force soft teacher/background labels to become
    # more selective. Hard evidence and structural support above still survive.
    if pressure > 0.0 and reason in {"teacher", "background"}:
        label *= _lerp(1.0, args.low_budget_background_scale, pressure)

    return min(max(label, 0.0), 1.0), weight, reason


def segment_content_signal(policy: OPSieveKVLitePolicy, positions: torch.Tensor, seq_len: int) -> float:
    """Return whether a segment has non-role content signal for teacher support."""
    content_signal = 0.0
    for signal in (policy.query_relevance, policy.factual_bonus, policy.authority_bonus):
        if signal is not None and len(signal) >= seq_len:
            content_signal = max(content_signal, float(signal[:seq_len][positions].max().item()))
    if policy.info_density is not None and len(policy.info_density) >= seq_len:
        content_signal = max(content_signal, 0.5 * float(policy.info_density[:seq_len][positions].max().item()))
    return content_signal


def select_teacher_support_segments(
    policy: OPSieveKVLitePolicy,
    segment_positions: list[torch.Tensor],
    heuristic_probs: torch.Tensor,
    evidence_segment_indices: set[int],
    budget: float,
    seq_len: int,
    args,
) -> set[int]:
    """Select a small top-ranked set of heuristic teacher segments."""
    teacher_threshold = budget_value(args.teacher_threshold, args.low_budget_teacher_threshold, budget, args)
    teacher_signal_threshold = budget_value(
        args.teacher_signal_threshold,
        args.low_budget_teacher_signal_threshold,
        budget,
        args,
    )
    teacher_top_fraction = budget_value(args.teacher_top_fraction, args.low_budget_teacher_top_fraction, budget, args)
    teacher_top_min = max(0, budget_int_value(args.teacher_top_min, args.low_budget_teacher_top_min, budget, args))
    teacher_top_max = max(0, budget_int_value(args.teacher_top_max, args.low_budget_teacher_top_max, budget, args))
    teacher_token_fraction = budget_value(
        args.teacher_budget_token_fraction,
        args.low_budget_teacher_budget_token_fraction,
        budget,
        args,
    )

    candidates: list[tuple[float, int]] = []
    for idx, positions in enumerate(segment_positions):
        if idx in evidence_segment_indices:
            continue
        heuristic_prob = float(heuristic_probs[idx].item()) if heuristic_probs.numel() else 0.0
        content_signal = segment_content_signal(policy, positions, seq_len)
        if heuristic_prob < teacher_threshold or content_signal < teacher_signal_threshold:
            continue
        # Rank by both teacher confidence and content signal so role-only high
        # scores no longer flood the positive labels.
        candidates.append((heuristic_prob + 0.5 * content_signal, idx))

    if not candidates:
        return set()

    candidates.sort(reverse=True)
    max_count = max(teacher_top_min, int(len(segment_positions) * teacher_top_fraction))
    max_count = min(max_count, teacher_top_max, len(candidates))
    if max_count <= 0:
        return set()

    max_teacher_tokens = max(
        1,
        int(seq_len * max(0.0, float(budget)) * max(0.0, teacher_token_fraction)),
    )
    selected: set[int] = set()
    selected_tokens = 0
    for _score, idx in candidates:
        if len(selected) >= max_count:
            break
        segment_tokens = int(segment_positions[idx].numel())
        if selected and selected_tokens + segment_tokens > max_teacher_tokens:
            continue
        selected.add(idx)
        selected_tokens += segment_tokens
        if selected_tokens >= max_teacher_tokens:
            break
    return selected


@torch.no_grad()
def collect_one_example(
    model,
    tokenizer,
    messages: list[dict],
    evidence_texts: list[str],
    budgets: list[float],
    args,
    metadata: dict,
) -> tuple[list[torch.Tensor], list[float], list[float], list[dict]]:
    """Run one prefill forward pass and emit segment feature rows."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = encoding.pop("offset_mapping")[0]
    inputs = encoding.to(model.device)
    input_ids = inputs["input_ids"][0]
    seq_len = int(input_ids.numel())

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

    feature_rows: list[torch.Tensor] = []
    labels: list[float] = []
    weights: list[float] = []
    row_metadata: list[dict] = []

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
        for row_idx, positions in enumerate(segment_positions):
            overlap = bool(evidence_mask[positions].any().item())
            heuristic_prob = float(heuristic_probs[row_idx].item()) if heuristic_probs.numel() else 0.0
            label, weight, label_reason = build_v2_soft_label(
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
                weight *= _lerp(1.0, args.low_budget_row_weight, pressure)
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
                    "evidence_overlap": overlap,
                    "evidence_overlap_frac": evidence_overlap_frac,
                    "heuristic_keep_prob": heuristic_prob,
                    "teacher_allowed": row_idx in teacher_segment_indices,
                    "budget_pressure": pressure,
                }
            )

    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feature_rows, labels, weights, row_metadata


def build_niah_examples(args) -> list[tuple[list[dict], list[str], dict]]:
    examples = []
    for position in args.positions:
        for needle_index, needle in enumerate(NEEDLES):
            messages = build_haystack_prompt(
                needle,
                haystack_length=args.haystack_length,
                needle_position=position,
            )
            examples.append(
                (
                    messages,
                    [needle["fact"]],
                    {
                        "task": "niah",
                        "needle_index": needle_index,
                        "needle_position": position,
                    },
                )
            )
    return examples


def build_multi_needle_examples(tokenizer, args) -> list[tuple[list[dict], list[str], dict]]:
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
            examples.append(
                (
                    messages,
                    facts,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect OP-SieveKV distillation data")
    parser.add_argument("--output", default="results/op_distill_dataset.pt")
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
    args = parser.parse_args()

    config = ExperimentConfig()
    model, tokenizer = load_model(config.model)

    examples: list[tuple[list[dict], list[str], dict]] = []
    if "niah" in args.tasks:
        examples.extend(build_niah_examples(args))
    if "multi" in args.tasks:
        examples.extend(build_multi_needle_examples(tokenizer, args))

    all_features: list[torch.Tensor] = []
    all_labels: list[float] = []
    all_weights: list[float] = []
    all_metadata: list[dict] = []

    for index, (messages, evidence_texts, metadata) in enumerate(examples, start=1):
        print(f"[{index}/{len(examples)}] collect {metadata}")
        rows, labels, weights, row_metadata = collect_one_example(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            evidence_texts=evidence_texts,
            budgets=args.budgets,
            args=args,
            metadata=metadata,
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
    hard_positives = int((labels_tensor >= 0.5).sum().item()) if labels_tensor.numel() else 0
    label_mass = float(labels_tensor.sum().item()) if labels_tensor.numel() else 0.0
    reason_counts = Counter(item.get("label_reason", "unknown") for item in all_metadata)
    hard_reason_counts = Counter(
        item.get("label_reason", "unknown")
        for item, label in zip(all_metadata, all_labels)
        if label >= 0.5
    )
    budget_rows: dict[float, list[float]] = {}
    budget_hard: Counter = Counter()
    budget_teacher: Counter = Counter()
    for item, label in zip(all_metadata, all_labels):
        budget = round(float(item.get("budget", 0.0)), 4)
        budget_rows.setdefault(budget, []).append(float(label))
        if label >= 0.5:
            budget_hard[budget] += 1
        if item.get("teacher_allowed"):
            budget_teacher[budget] += 1
    print(
        f"Saved {features.shape[0]} rows, hard_positives={hard_positives}, "
        f"label_mass={label_mass:.1f}, path={output_path}"
    )
    print(f"Label reasons: {dict(reason_counts)}")
    print(f"Hard-positive reasons: {dict(hard_reason_counts)}")
    print("Budget label stats:")
    for budget in sorted(budget_rows):
        values = budget_rows[budget]
        print(
            f"  b={budget:g}: rows={len(values)}, hard={budget_hard[budget]}, "
            f"teacher={budget_teacher[budget]}, label_mass={sum(values):.1f}"
        )


if __name__ == "__main__":
    main()
