"""Collect a true on-policy compressed-cache oracle dataset for OP-SieveKV.

This is the core-method collector from the research plan (RQ3, Sections 5.4-5.5).
Unlike the prompt-level pilot in ``collect_op_oracle_data.py`` -- which drops a
segment from the *full* prompt and compares against the full-context answer
log-probability -- this collector first lets the (optionally learned) retention
policy evict under its own budget to form the compressed cache state
``C^{pi_theta}``, then measures counterfactual importance *relative to that
compressed state*:

* Drop importance (for segments the policy retained):
      I_drop = logP(a | C) - logP(a | C \\ g)
* Restore importance (for segments the policy evicted):
      I_restore = logP(a | C u g) - logP(a | C)

Both reduce to the marginal value of including segment ``g`` given the policy's
own compressed context. The restore branch is what surfaces Q3
"confident-wrong" decisions: segments the policy confidently dropped that the
oracle marks as answer-critical only once other evidence is gone. The
prompt-level pilot could not discover these false negatives.

The compressed state is constructed with the exact runtime convention used by
``kv_cache_manager.py``: ``budget_tokens = max(1, int(seq_len * budget))`` and
``keep_indices = policy.select_keep_indices(policy.compute_eviction_scores(...))``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from attention_tracker import AttentionTracker
from collect_op_distill_data import (
    budget_pressure,
    build_evidence_mask_from_offsets,
    build_v2_soft_label,
    select_teacher_support_segments,
)
from collect_op_oracle_data import (
    build_multi_needle_examples,
    build_niah_examples,
    compute_student_probs,
    mix_oracle_label,
    oracle_label_from_delta,
    select_oracle_candidate_segments,
)
from config import ExperimentConfig
from eviction_policies import OPSieveKVLitePolicy
from kv_cache_manager import build_dynamic_cache_from_ddp
from op_policy_model import FEATURE_NAMES, load_policy_checkpoint
from run_generation import _extract_latest_user_query, load_model
from semantic_analyzer import SemanticAnalyzer


def make_onpolicy_policy(
    tokenizer,
    tracker: AttentionTracker,
    input_ids: torch.Tensor,
    latest_query: str,
    budget: float,
    args,
    policy_ckpt: str | None,
) -> OPSieveKVLitePolicy:
    """Build a policy whose eviction reflects the learned (on-policy) decisions.

    When ``policy_ckpt`` is provided the policy evicts using the learned MLP, so
    the compressed cache state is genuinely on-policy. Without a checkpoint the
    policy falls back to the SieveKV heuristic, which matches the research plan's
    warm-start (the heuristic is the initial pi_theta).
    """
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
        policy_ckpt=policy_ckpt,
    )
    policy.setup_semantic_signals(input_ids.detach().cpu(), latest_query_text=latest_query)
    return policy


def compressed_keep_mask(policy: OPSieveKVLitePolicy, seq_len: int, budget: float) -> torch.Tensor:
    """Return the policy's on-policy retained-token boolean mask under a budget."""
    scores = policy.compute_eviction_scores(seq_len)
    budget_tokens = max(1, int(seq_len * budget))
    keep_indices = policy.select_keep_indices(scores, budget_tokens).detach().cpu().to(dtype=torch.long)
    keep_mask = torch.zeros(seq_len, dtype=torch.bool)
    keep_mask[keep_indices[(keep_indices >= 0) & (keep_indices < seq_len)]] = True
    return keep_mask


def mask_to_indices(mask: torch.Tensor) -> torch.Tensor:
    """Return sorted CPU indices for true entries in a boolean mask."""
    return torch.nonzero(mask.detach().cpu(), as_tuple=False).flatten().to(dtype=torch.long)


def cache_from_origin_indices(past_key_values, origin_indices: torch.Tensor, model_config):
    """Assemble a DynamicCache slice using original prompt token positions.

    The selected keys/values were computed during full prefill with original
    RoPE positions. This is the cache-level analogue of runtime eviction: the
    physical cache is compacted, but each retained KV vector still encodes its
    original absolute position.
    """
    indices = origin_indices.detach().cpu().to(dtype=torch.long).unique(sorted=True)
    if indices.numel() == 0:
        return build_dynamic_cache_from_ddp([], model_config)

    if hasattr(past_key_values, "layers"):
        if len(past_key_values.layers) == 0:
            return build_dynamic_cache_from_ddp([], model_config)
        ref = past_key_values.layers[0].keys
        device_indices = indices.to(ref.device)
        ddp_cache_data = []
        for layer in past_key_values.layers:
            ddp_cache_data.append(
                (
                    torch.index_select(layer.keys, 2, device_indices),
                    torch.index_select(layer.values, 2, device_indices),
                )
            )
        return build_dynamic_cache_from_ddp(ddp_cache_data, model_config)

    if hasattr(past_key_values, "key_cache"):
        if len(past_key_values.key_cache) == 0:
            return build_dynamic_cache_from_ddp([], model_config)
        device_indices = indices.to(past_key_values.key_cache[0].device)
        ddp_cache_data = []
        for layer_idx in range(len(past_key_values.key_cache)):
            ddp_cache_data.append(
                (
                    torch.index_select(past_key_values.key_cache[layer_idx], 2, device_indices),
                    torch.index_select(past_key_values.value_cache[layer_idx], 2, device_indices),
                )
            )
        return build_dynamic_cache_from_ddp(ddp_cache_data, model_config)

    raise TypeError(f"Unsupported cache structure: {type(past_key_values)!r}")


@torch.no_grad()
def cache_answer_avg_logprob(
    *,
    model,
    full_past_key_values,
    input_ids_cpu: torch.Tensor,
    context_indices: torch.Tensor,
    answer_ids: torch.Tensor,
) -> float:
    """Score the gold answer against an assembled compressed KV cache.

    We exclude the final prompt token from the cache slice and feed it as the
    first query token, followed by answer[:-1]. This produces logits for every
    answer token while preserving the original absolute position ids used by
    runtime generation after eviction.
    """
    prompt_ids = input_ids_cpu.detach().cpu().to(dtype=torch.long)
    answer_ids = answer_ids.detach().cpu().to(dtype=torch.long)
    if prompt_ids.numel() == 0 or answer_ids.numel() == 0:
        return 0.0

    prompt_len = int(prompt_ids.numel())
    last_prompt_pos = prompt_len - 1
    context_indices = context_indices.detach().cpu().to(dtype=torch.long)
    context_indices = context_indices[(context_indices >= 0) & (context_indices < last_prompt_pos)]
    context_indices = context_indices.unique(sorted=True)

    cache = cache_from_origin_indices(full_past_key_values, context_indices, model.config)
    score_input = torch.cat([prompt_ids[-1:], answer_ids[:-1]], dim=0).unsqueeze(0).to(model.device)
    targets = answer_ids.unsqueeze(0).to(model.device)
    cache_len = int(context_indices.numel())
    cache_position = torch.arange(
        cache_len,
        cache_len + score_input.shape[1],
        device=model.device,
        dtype=torch.long,
    )
    position_ids = torch.arange(
        last_prompt_pos,
        last_prompt_pos + score_input.shape[1],
        device=model.device,
        dtype=torch.long,
    ).unsqueeze(0)

    outputs = model(
        input_ids=score_input,
        past_key_values=cache,
        use_cache=False,
        cache_position=cache_position,
        position_ids=position_ids,
        return_dict=True,
    )
    log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    value = float(token_log_probs.mean().item())
    del outputs, log_probs, token_log_probs
    return value


@torch.no_grad()
def onpolicy_segment_delta(
    *,
    model,
    full_past_key_values,
    input_ids_cpu: torch.Tensor,
    answer_ids: torch.Tensor,
    keep_indices: torch.Tensor,
    compressed_logprob: float,
    positions: torch.Tensor,
    retained: bool,
) -> float:
    """Marginal answer-logprob value of segment ``positions`` given compressed C.

    ``retained`` selects the cheaper single-forward counterfactual:
      * retained  -> drop:    delta = logP(a|C) - logP(a|C \\ g)
      * evicted   -> restore: delta = logP(a|C u g) - logP(a|C)
    Positive delta means including the segment helps the answer under the
    policy's own compressed state.
    """
    if retained:
        segment_mask = torch.isin(keep_indices, positions.detach().cpu().to(dtype=torch.long))
        counterfactual_indices = keep_indices[~segment_mask]
        without_logprob = cache_answer_avg_logprob(
            model=model,
            full_past_key_values=full_past_key_values,
            input_ids_cpu=input_ids_cpu,
            context_indices=counterfactual_indices,
            answer_ids=answer_ids,
        )
        return compressed_logprob - without_logprob

    counterfactual_indices = torch.cat([keep_indices, positions.detach().cpu().to(dtype=torch.long)], dim=0).unique(sorted=True)
    with_logprob = cache_answer_avg_logprob(
        model=model,
        full_past_key_values=full_past_key_values,
        input_ids_cpu=input_ids_cpu,
        context_indices=counterfactual_indices,
        answer_ids=answer_ids,
    )
    return with_logprob - compressed_logprob


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
    policy_bundle=None,
) -> tuple[list[torch.Tensor], list[float], list[float], list[dict]]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = encoding.pop("offset_mapping")[0]
    inputs = encoding.to(model.device)
    input_ids = inputs["input_ids"][0]
    input_ids_cpu = input_ids.detach().cpu().to(dtype=torch.long)
    seq_len = int(input_ids.numel())
    answer_ids = tokenizer(answer_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        use_cache=True,
        output_attentions=True,
        return_dict=True,
    )
    full_past_key_values = outputs.past_key_values
    tracker = AttentionTracker(
        num_layers=getattr(model.config, "num_hidden_layers", 36),
        num_kv_heads=getattr(model.config, "num_key_value_heads", 2),
    )
    tracker.update(outputs.attentions)
    del outputs

    evidence_mask = build_evidence_mask_from_offsets(text, offset_mapping, evidence_texts)
    latest_query = _extract_latest_user_query(messages)
    full_logprob = cache_answer_avg_logprob(
        model=model,
        full_past_key_values=full_past_key_values,
        input_ids_cpu=input_ids_cpu,
        context_indices=torch.arange(seq_len, dtype=torch.long),
        answer_ids=answer_ids,
    )

    feature_rows: list[torch.Tensor] = []
    labels: list[float] = []
    weights: list[float] = []
    row_metadata: list[dict] = []

    for budget in budgets:
        policy = make_onpolicy_policy(
            tokenizer=tokenizer,
            tracker=tracker,
            input_ids=input_ids,
            latest_query=latest_query,
            budget=budget,
            args=args,
            policy_ckpt=args.policy_ckpt,
        )
        features, segment_positions, heuristic_probs = policy.compute_segment_features(seq_len)

        # On-policy compressed cache state C^{pi_theta} under this budget.
        keep_mask = compressed_keep_mask(policy, seq_len, budget)
        keep_indices = mask_to_indices(keep_mask)
        compressed_logprob = cache_answer_avg_logprob(
            model=model,
            full_past_key_values=full_past_key_values,
            input_ids_cpu=input_ids_cpu,
            context_indices=keep_indices,
            answer_ids=answer_ids,
        )

        # Segment-level on-policy decision (learned policy if loaded, else heuristic).
        student_probs = compute_student_probs(features, policy_bundle)
        if student_probs is None:
            student_probs = heuristic_probs

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
        candidate_indices = select_oracle_candidate_segments(
            policy=policy,
            segment_positions=segment_positions,
            heuristic_probs=heuristic_probs,
            student_probs=student_probs,
            evidence_segment_indices=evidence_segment_indices,
            seq_len=seq_len,
            args=args,
        )

        oracle_cache: dict[int, dict] = {}
        for row_idx in candidate_indices:
            positions = segment_positions[row_idx].to(dtype=torch.long)
            retained_count = int(keep_mask[positions].sum().item()) if positions.numel() else 0
            retained_frac = retained_count / max(1, int(positions.numel()))
            if 0 < retained_count < int(positions.numel()) and not args.score_partial_segments:
                continue
            retained = retained_count > 0 if retained_count < int(positions.numel()) else True
            delta = onpolicy_segment_delta(
                model=model,
                full_past_key_values=full_past_key_values,
                input_ids_cpu=input_ids_cpu,
                answer_ids=answer_ids,
                keep_indices=keep_indices,
                compressed_logprob=compressed_logprob,
                positions=positions,
                retained=retained,
            )
            oracle_cache[row_idx] = {
                "oracle_delta": delta,
                "oracle_label": oracle_label_from_delta(delta, args),
                "oracle_mode": "drop" if retained else "restore",
                "segment_retained": retained,
                "retained_count": retained_count,
                "retained_frac": retained_frac,
            }

        for row_idx, positions in enumerate(segment_positions):
            heuristic_prob = float(heuristic_probs[row_idx].item()) if heuristic_probs.numel() else 0.0
            policy_prob = float(student_probs[row_idx].item()) if student_probs.numel() else heuristic_prob
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
                cheap_weight *= 1.0 + pressure * (args.low_budget_row_weight - 1.0)

            label = cheap_label
            weight = cheap_weight
            oracle_info = oracle_cache.get(row_idx)
            if oracle_info is not None:
                label, weight, label_reason = mix_oracle_label(
                    cheap_label=cheap_label,
                    cheap_weight=cheap_weight,
                    cheap_reason=label_reason,
                    oracle_label=float(oracle_info["oracle_label"]),
                    budget=budget,
                    args=args,
                )

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
                    "evidence_overlap": bool(evidence_mask[positions].any().item()),
                    "evidence_overlap_frac": float(evidence_mask[positions].float().mean().item()),
                    "heuristic_keep_prob": heuristic_prob,
                    "candidate_student_keep_prob": policy_prob,
                    "policy_keep_prob": policy_prob,
                    "segment_retained": bool(keep_mask[positions].float().mean().item() >= 0.5),
                    "segment_retained_frac": float(keep_mask[positions].float().mean().item()),
                    "retention_decision": (
                        "keep"
                        if float(keep_mask[positions].float().mean().item()) >= 0.5
                        else "drop"
                    ),
                    "teacher_allowed": row_idx in teacher_segment_indices,
                    "oracle_scored": oracle_info is not None,
                    "oracle_delta": None if oracle_info is None else float(oracle_info["oracle_delta"]),
                    "oracle_label": None if oracle_info is None else float(oracle_info["oracle_label"]),
                    "oracle_mode": None if oracle_info is None else oracle_info["oracle_mode"],
                    "on_policy": True,
                    "cache_state_oracle": True,
                    "oracle_state_type": "compressed_kv",
                    "compressed_cache_tokens": int(keep_indices.numel()),
                    "compressed_answer_avg_logprob": compressed_logprob,
                    "full_answer_avg_logprob": full_logprob,
                    "budget_pressure": pressure,
                }
            )

    del full_past_key_values
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feature_rows, labels, weights, row_metadata


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", default="results/op_oracle_dataset_onpolicy.pt")
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

    # Oracle candidate-selection parameters (shared with collect_op_oracle_data.py).
    parser.add_argument("--oracle-max-candidates", type=int, default=16)
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
    parser.add_argument("--student-missed-k", type=int, default=6)
    parser.add_argument("--student-overkeep-k", type=int, default=4)
    parser.add_argument("--student-min-content-signal", type=float, default=0.10)
    parser.add_argument(
        "--score-partial-segments",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Score segments that are only partially retained. Disabled by default "
            "because partial segments make drop/restore labels ambiguous."
        ),
    )

    parser.add_argument(
        "--policy-ckpt",
        default=None,
        help=(
            "Learned OP policy checkpoint. Drives BOTH the on-policy eviction "
            "(compressed cache state) and the oracle candidate selection. "
            "Without it the SieveKV heuristic acts as the warm-start pi_theta."
        ),
    )
    parser.add_argument("--max-examples", type=int, default=0, help="Cap total examples (0 = all). Useful for smoke runs.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect OP-SieveKV on-policy compressed-cache oracle data")
    add_arguments(parser)
    args = parser.parse_args()

    print(
        "On-policy oracle collector config: "
        f"policy_ckpt={args.policy_ckpt or 'heuristic-warmstart'}, "
        f"mix_mode={args.oracle_mix_mode}, budgets={args.budgets}, "
        f"max_candidates={args.oracle_max_candidates}"
    )

    config = ExperimentConfig()
    model, tokenizer = load_model(config.model)

    policy_bundle = None
    if args.policy_ckpt:
        policy_model, feature_mean, feature_std, ckpt_metadata = load_policy_checkpoint(
            args.policy_ckpt, map_location="cpu"
        )
        policy_bundle = (policy_model, feature_mean, feature_std)
        print(
            f"On-policy candidate policy loaded: {args.policy_ckpt}, "
            f"best_val_loss={ckpt_metadata.get('best_val_loss', 'n/a')}"
        )

    examples: list[tuple[list[dict], list[str], str, dict]] = []
    if "niah" in args.tasks:
        examples.extend(build_niah_examples(args))
    if "multi" in args.tasks:
        examples.extend(build_multi_needle_examples(tokenizer, args))
    if args.max_examples > 0:
        examples = examples[: args.max_examples]

    all_features: list[torch.Tensor] = []
    all_labels: list[float] = []
    all_weights: list[float] = []
    all_metadata: list[dict] = []

    for index, (messages, evidence_texts, answer_text, metadata) in enumerate(examples, start=1):
        print(f"[{index}/{len(examples)}] on-policy oracle collect {metadata}")
        rows, labels, weights, row_metadata = collect_one_example(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            evidence_texts=evidence_texts,
            answer_text=answer_text,
            budgets=args.budgets,
            args=args,
            metadata=metadata,
            policy_bundle=policy_bundle,
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

    _print_summary(all_metadata, all_labels, labels_tensor, features, output_path)


def _print_summary(all_metadata, all_labels, labels_tensor, features, output_path) -> None:
    reason_counts = Counter(item.get("label_reason", "unknown") for item in all_metadata)
    oracle_scored = [item for item in all_metadata if item.get("oracle_scored")]
    oracle_deltas = [float(item["oracle_delta"]) for item in oracle_scored if item.get("oracle_delta") is not None]
    mode_counts = Counter(item.get("oracle_mode") for item in oracle_scored if item.get("oracle_mode"))

    # Missed-keep proxy: budget selection dropped the segment, but the
    # compressed-cache restore oracle says including it helps.
    restore_rows = [item for item in oracle_scored if item.get("oracle_mode") == "restore"]
    missed_keep = [
        item
        for item in restore_rows
        if float(item.get("oracle_label", 0.0)) >= 0.5
    ]
    confident_wrong = [
        item
        for item in missed_keep
        if float(item.get("policy_keep_prob", 1.0)) < 0.5
    ]

    hard_positives = int((labels_tensor >= 0.5).sum().item()) if labels_tensor.numel() else 0
    label_mass = float(labels_tensor.sum().item()) if labels_tensor.numel() else 0.0
    print(
        f"Saved {features.shape[0]} rows, hard_positives={hard_positives}, "
        f"label_mass={label_mass:.1f}, oracle_rows={len(oracle_scored)}, path={output_path}"
    )
    print(f"Oracle modes: {dict(mode_counts)}")
    if oracle_deltas:
        delta_tensor = torch.tensor(oracle_deltas)
        print(
            "Oracle delta stats: "
            f"min={float(delta_tensor.min().item()):.4f}, "
            f"mean={float(delta_tensor.mean().item()):.4f}, "
            f"max={float(delta_tensor.max().item()):.4f}"
        )
    print(
        f"Missed-keep restore candidates: {len(missed_keep)} "
        f"of {len(restore_rows)} restore probes"
    )
    print(
        f"Q3 low-prob confident-wrong subset: {len(confident_wrong)} "
        f"of {len(missed_keep)} missed-keep probes"
    )
    print(f"Label reasons: {dict(reason_counts)}")

    budget_oracle: Counter = Counter()
    budget_rows: dict[float, list[float]] = {}
    for item, label in zip(all_metadata, all_labels):
        budget = round(float(item.get("budget", 0.0)), 4)
        budget_rows.setdefault(budget, []).append(float(label))
        if item.get("oracle_scored"):
            budget_oracle[budget] += 1
    print("Budget label stats:")
    for budget in sorted(budget_rows):
        values = budget_rows[budget]
        print(
            f"  b={budget:g}: rows={len(values)}, oracle={budget_oracle[budget]}, "
            f"label_mass={sum(values):.1f}"
        )


if __name__ == "__main__":
    main()
