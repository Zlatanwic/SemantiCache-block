"""Select informative OP-SieveKV oracle decisions for KV-TIP / Soft-OR training.

The collector intentionally writes a broad dataset: cheap support labels plus
oracle-scored counterfactual rows. For the research-plan ablations we need a
second stage that keeps the structural anchors but trains on the most
informative oracle decisions:

* entropy-only: high retention uncertainty
* divergence-only: high oracle-policy disagreement
* Q3-only: low entropy and high disagreement
* missed-keep-only: confident restore positives
* soft-or: z = h + d - h*d

The output is the same `.pt` schema consumed by `train_op_policy.py`. Use
``--selection-action filter`` for ablations and ``--selection-action reweight``
for the full method: reweight preserves the full on-policy dataset and only
upweights selected informative oracle rows.

After the TrOPD trust-region reading, the recommended full method is:

``--method-preset trust_region_reweight``

This is not a new online policy. It is a safer training-data transformation:
keep the full on-policy dataset as the trust-region support, identify oracle
outliers by Soft-OR / KV-TIP, and increase their training weight instead of
filtering away the rest of the behavioral distribution.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


SUPPORT_REASONS = {
    "boundary_support",
    "system_support",
    "pinned_support",
    "query_support",
    "recent_support",
    "heuristic_support",
    "evidence",
    "evidence_neighbor",
}


MODE_ALIASES = {
    "trust_region_soft_or": "soft_or",
}


METHOD_PRESETS = {
    "none",
    "trust_region_reweight",
}


def canonical_mode(mode: str) -> str:
    return MODE_ALIASES.get(mode, mode)


def binary_entropy(prob: float) -> float:
    p = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = min(max(q, 0.0), 1.0) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def normalize_values(values: list[float], method: str) -> list[float]:
    """Normalize metric values for cross-metric Soft-OR selection."""
    if not values:
        return []
    if method == "none":
        return [float(value) for value in values]
    if method == "minmax":
        lo = min(values)
        hi = max(values)
        denom = max(hi - lo, 1e-6)
        return [(float(value) - lo) / denom for value in values]
    if method == "zscore_sigmoid":
        mu = sum(values) / len(values)
        var = sum((float(value) - mu) ** 2 for value in values) / max(1, len(values))
        std = math.sqrt(max(var, 1e-12))
        return [sigmoid((float(value) - mu) / std) for value in values]
    raise ValueError(f"Unknown normalization method: {method}")


def policy_prob(row: dict[str, Any], field: str) -> float:
    for key in (field, "policy_keep_prob", "candidate_student_keep_prob", "heuristic_keep_prob"):
        if key in row and row.get(key) is not None:
            return float(row[key])
    return 0.0


def attach_selection_scores(
    rows: list[dict[str, Any]],
    *,
    policy_prob_field: str,
    entropy_threshold: float | None,
    divergence_threshold: float | None,
    entropy_quantile: float,
    divergence_quantile: float,
    score_normalization: str,
) -> tuple[float, float]:
    oracle_rows = [
        row
        for row in rows
        if row.get("oracle_scored") and row.get("oracle_label") is not None
    ]
    for row in oracle_rows:
        prob = policy_prob(row, policy_prob_field)
        oracle = float(row.get("oracle_label", 0.0))
        entropy = binary_entropy(prob)
        divergence = abs(oracle - prob)
        row["selection_policy_prob"] = prob
        row["retention_entropy"] = entropy
        row["oracle_policy_divergence"] = divergence
        row["decision_type"] = decision_type(prob, oracle)
        row["trust_region_probability"] = min(
            probability_ratio(oracle, prob),
            1.0,
        )

    entropy_scores = normalize_values([float(row["retention_entropy"]) for row in oracle_rows], score_normalization)
    divergence_scores = normalize_values(
        [float(row["oracle_policy_divergence"]) for row in oracle_rows],
        score_normalization,
    )
    for row, entropy_score, divergence_score in zip(oracle_rows, entropy_scores, divergence_scores):
        row["selection_entropy"] = float(entropy_score)
        row["selection_divergence"] = float(divergence_score)
        row["soft_or_score"] = float(entropy_score + divergence_score - (entropy_score * divergence_score))

    entropies = [float(row["retention_entropy"]) for row in oracle_rows]
    divergences = [float(row["oracle_policy_divergence"]) for row in oracle_rows]
    entropy_cut = entropy_threshold
    divergence_cut = divergence_threshold
    if entropy_cut is None:
        entropy_cut = quantile(entropies, entropy_quantile)
    if divergence_cut is None:
        divergence_cut = quantile(divergences, divergence_quantile)

    for row in oracle_rows:
        high_entropy = float(row["retention_entropy"]) >= entropy_cut
        high_divergence = float(row["oracle_policy_divergence"]) >= divergence_cut
        if not high_entropy and high_divergence:
            row["kvtip_quadrant"] = "Q3_confident_wrong"
        elif high_entropy and high_divergence:
            row["kvtip_quadrant"] = "Q4_uncertain_wrong"
        elif high_entropy:
            row["kvtip_quadrant"] = "Q2_uncertain_aligned"
        else:
            row["kvtip_quadrant"] = "Q1_confident_aligned"
        row["trust_region"] = not high_divergence
        row["trust_region_type"] = "trust_region" if row["trust_region"] else "oracle_outlier"
    return float(entropy_cut), float(divergence_cut)


def decision_type(policy_probability: float, oracle_label: float) -> str:
    if policy_probability >= 0.5 and oracle_label < 0.5:
        return "over_keep"
    if policy_probability < 0.5 and oracle_label >= 0.5:
        return "missed_keep"
    if policy_probability >= 0.5 and oracle_label >= 0.5:
        return "aligned_keep"
    return "aligned_drop"


def probability_ratio(oracle_label: float, policy_probability: float) -> float:
    """Approximate a TrOPD-style agreement ratio for binary keep/drop labels.

    TrOPD uses min(pi_teacher(x) / pi_student(x), 1) to identify regions where
    teacher supervision is reliable. In this retention setting the oracle is a
    binary/soft keep label, so we compare the oracle probability assigned to the
    policy-selected decision against the policy probability of that decision.
    This is only a diagnostic score; actual selection still uses entropy,
    divergence, and KV-TIP quadrants.
    """
    p = min(max(float(policy_probability), 1e-6), 1.0 - 1e-6)
    y = min(max(float(oracle_label), 1e-6), 1.0 - 1e-6)
    if p >= 0.5:
        return y / p
    return (1.0 - y) / (1.0 - p)


def oracle_score(row: dict[str, Any], mode: str) -> float:
    mode = canonical_mode(mode)
    if mode == "entropy_only":
        return float(row.get("selection_entropy", row.get("retention_entropy", 0.0)))
    if mode == "divergence_only":
        return float(row.get("selection_divergence", row.get("oracle_policy_divergence", 0.0)))
    if mode == "soft_or":
        return float(row.get("soft_or_score", 0.0))
    if mode == "q3_only":
        return float(row.get("oracle_policy_divergence", 0.0))
    if mode == "missed_keep_only":
        return float(row.get("oracle_policy_divergence", 0.0))
    return 0.0


def support_row(row: dict[str, Any], keep_reasons: set[str], keep_hard_nonoracle: bool) -> bool:
    if row.get("oracle_scored"):
        return False
    reason = str(row.get("label_reason", ""))
    if reason in keep_reasons:
        return True
    return keep_hard_nonoracle and float(row.get("label", 0.0)) >= 0.5


def select_oracle_indices(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    rho: float,
    top_k: int,
    min_per_budget: int,
) -> set[int]:
    mode = canonical_mode(mode)
    oracle = [
        row
        for row in rows
        if row.get("oracle_scored") and row.get("oracle_label") is not None
    ]
    if mode == "all_decisions":
        return set(range(len(rows)))
    if mode == "all_oracle":
        return {int(row["row_index"]) for row in oracle}
    if mode == "q3_only":
        oracle = [row for row in oracle if row.get("kvtip_quadrant") == "Q3_confident_wrong"]
    elif mode == "missed_keep_only":
        oracle = [
            row
            for row in oracle
            if row.get("decision_type") == "missed_keep" and row.get("oracle_mode") == "restore"
        ]

    by_budget: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in oracle:
        by_budget[round(float(row.get("budget", 0.0)), 6)].append(row)

    selected: set[int] = set()
    for budget_rows in by_budget.values():
        budget_rows.sort(key=lambda row: oracle_score(row, mode), reverse=True)
        count = max(min_per_budget, int(math.ceil(len(budget_rows) * rho)))
        if top_k > 0:
            count = min(count, top_k)
        count = min(count, len(budget_rows))
        selected.update(int(row["row_index"]) for row in budget_rows[:count])
    return selected


def apply_reweighting(
    rows: list[dict[str, Any]],
    weights: torch.Tensor,
    selected_indices: set[int],
    *,
    selected_weight_multiplier: float,
    q3_weight_multiplier: float,
    missed_keep_weight_multiplier: float,
    max_weight: float,
) -> torch.Tensor:
    """Return weights with multiplicative boosts for selected KV-TIP rows."""
    new_weights = weights.clone().float()
    for index in selected_indices:
        row = rows[int(index)]
        multiplier = float(selected_weight_multiplier)
        if row.get("kvtip_quadrant") == "Q3_confident_wrong":
            multiplier *= float(q3_weight_multiplier)
        if row.get("decision_type") == "missed_keep":
            multiplier *= float(missed_keep_weight_multiplier)
        new_weights[int(index)] *= multiplier
        row["selected_for_training"] = True
        row["selection_weight_multiplier"] = multiplier
        row["training_treatment"] = "oracle_outlier_reweight"

    for idx, row in enumerate(rows):
        if idx not in selected_indices:
            row["selected_for_training"] = False
            row["selection_weight_multiplier"] = 1.0
            row["training_treatment"] = "trust_region_support"

    if max_weight > 0:
        new_weights = new_weights.clamp(max=float(max_weight))
    return new_weights


def summarize(
    rows: list[dict[str, Any]],
    selected_indices: set[int],
    *,
    output_indices: torch.Tensor | None = None,
    output_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    selected_rows = [row for row in rows if int(row["row_index"]) in selected_indices]
    oracle_selected = [row for row in selected_rows if row.get("oracle_scored")]
    oracle_rows = [
        row
        for row in rows
        if row.get("oracle_scored") and row.get("oracle_label") is not None
    ]
    outlier_rows = [row for row in oracle_rows if row.get("trust_region_type") == "oracle_outlier"]
    trust_rows = [row for row in oracle_rows if row.get("trust_region_type") == "trust_region"]
    selected_outlier_rows = [
        row for row in oracle_selected if row.get("trust_region_type") == "oracle_outlier"
    ]
    output_rows = (
        [rows[int(idx)] for idx in output_indices.tolist()]
        if output_indices is not None
        else selected_rows
    )
    weights_for_summary = output_weights if output_weights is not None else torch.tensor(
        [float(row.get("weight", 0.0)) for row in output_rows],
        dtype=torch.float32,
    )
    by_budget = Counter(round(float(row.get("budget", 0.0)), 6) for row in selected_rows)
    oracle_by_budget = Counter(round(float(row.get("budget", 0.0)), 6) for row in oracle_selected)
    output_by_budget = Counter(round(float(row.get("budget", 0.0)), 6) for row in output_rows)
    return {
        "output_rows": len(output_rows),
        "selected_rows": len(selected_rows),
        "selected_oracle_rows": len(oracle_selected),
        "selected_hard_rows": sum(1 for row in selected_rows if float(row.get("label", 0.0)) >= 0.5),
        "selected_label_mass": round(sum(float(row.get("label", 0.0)) for row in selected_rows), 4),
        "selected_weight_mean": round(
            sum(float(row.get("weight", 0.0)) for row in selected_rows) / max(1, len(selected_rows)),
            4,
        ),
        "output_weight_mean": round(float(weights_for_summary.mean().item()), 4)
        if weights_for_summary.numel()
        else 0.0,
        "output_weight_max": round(float(weights_for_summary.max().item()), 4)
        if weights_for_summary.numel()
        else 0.0,
        "trust_region_oracle_rows": len(trust_rows),
        "outlier_oracle_rows": len(outlier_rows),
        "selected_outlier_rows": len(selected_outlier_rows),
        "mean_trust_region_probability": round(
            sum(float(row.get("trust_region_probability", 0.0)) for row in oracle_rows)
            / max(1, len(oracle_rows)),
            4,
        ),
        "by_budget": {str(key): by_budget[key] for key in sorted(by_budget)},
        "oracle_by_budget": {str(key): oracle_by_budget[key] for key in sorted(oracle_by_budget)},
        "output_by_budget": {str(key): output_by_budget[key] for key in sorted(output_by_budget)},
        "label_reasons": dict(Counter(str(row.get("label_reason", "unknown")) for row in selected_rows)),
        "kvtip_quadrants": dict(Counter(str(row.get("kvtip_quadrant", "not_oracle")) for row in selected_rows)),
        "decision_types": dict(Counter(str(row.get("decision_type", "not_oracle")) for row in selected_rows)),
        "trust_region_types": dict(
            Counter(str(row.get("trust_region_type", "not_oracle")) for row in selected_rows)
        ),
        "training_treatments": dict(
            Counter(str(row.get("training_treatment", "not_marked")) for row in output_rows)
        ),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    method_note = (
        "Full method: keep the full on-policy dataset and reweight selected "
        "oracle outliers."
        if report["selection_action"] == "reweight"
        else "Ablation: filter selected/support rows and train on the reduced dataset."
    )
    lines = [
        "# OP Oracle Dataset Selection",
        "",
        method_note,
        "",
        f"- Input: `{report['input']}`",
        f"- Output: `{report['output']}`",
        f"- Method preset: `{report.get('method_preset', 'none')}`",
        f"- Mode: `{report['mode']}`",
        f"- Canonical mode: `{report.get('canonical_mode', report['mode'])}`",
        f"- Selection action: `{report['selection_action']}`",
        f"- Rho: {report['rho']}",
        f"- Policy probability field: `{report['policy_prob_field']}`",
        f"- Entropy threshold: {report['entropy_threshold']:.4f}",
        f"- Divergence threshold: {report['divergence_threshold']:.4f}",
        f"- Rows: {report['input_rows']} -> {report['output_rows']} output / {report['selected_rows']} selected",
        f"- Oracle rows: {report['input_oracle_rows']} -> {report['selected_oracle_rows']}",
        f"- Label mass: {report['input_label_mass']:.1f} -> {report['selected_label_mass']:.1f}",
        f"- Output weight mean/max: {report['output_weight_mean']} / {report['output_weight_max']}",
        f"- Trust-region oracle rows: {report['trust_region_oracle_rows']}",
        f"- Oracle outlier rows: {report['outlier_oracle_rows']} -> {report['selected_outlier_rows']} selected",
        f"- Mean trust-region probability: {report['mean_trust_region_probability']}",
        "",
        "## Selected By Budget",
        "",
    ]
    for budget, count in report["by_budget"].items():
        oracle_count = report["oracle_by_budget"].get(budget, 0)
        lines.append(f"- {budget}: rows={count}, oracle={oracle_count}")
    lines.extend(["", "## KV-TIP Quadrants", ""])
    for key, value in report["kvtip_quadrants"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Decision Types", ""])
    for key, value in report["decision_types"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Trust Region Types", ""])
    for key, value in report["trust_region_types"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Training Treatments", ""])
    for key, value in report["training_treatments"].items():
        lines.append(f"- {key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select OP oracle rows for Soft-OR / KV-TIP ablations")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--method-preset",
        choices=sorted(METHOD_PRESETS),
        default="none",
        help=(
            "Use trust_region_reweight for the recommended full method: "
            "Soft-OR/KV-TIP outlier discovery with full-dataset reweighting."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=[
            "all_decisions",
            "all_oracle",
            "entropy_only",
            "divergence_only",
            "q3_only",
            "missed_keep_only",
            "soft_or",
            "trust_region_soft_or",
        ],
        default="soft_or",
    )
    parser.add_argument("--rho", type=float, default=0.30, help="Per-budget fraction of oracle rows to select")
    parser.add_argument(
        "--selection-action",
        choices=["filter", "reweight"],
        default="filter",
        help="filter keeps only selected/support rows; reweight keeps all rows and upweights selected oracle rows.",
    )
    parser.add_argument("--top-k", type=int, default=0, help="Optional per-budget cap after rho selection")
    parser.add_argument("--min-per-budget", type=int, default=1)
    parser.add_argument("--policy-prob-field", default="policy_keep_prob")
    parser.add_argument("--entropy-threshold", type=float, default=None)
    parser.add_argument("--divergence-threshold", type=float, default=None)
    parser.add_argument("--entropy-quantile", type=float, default=0.50)
    parser.add_argument("--divergence-quantile", type=float, default=0.75)
    parser.add_argument(
        "--score-normalization",
        choices=["zscore_sigmoid", "minmax", "none"],
        default="zscore_sigmoid",
        help=(
            "Normalize entropy/divergence before Soft-OR. zscore_sigmoid matches "
            "the research-plan intent of z-scored selection while keeping scores in [0, 1]."
        ),
    )
    parser.add_argument("--include-support", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-support-reasons", nargs="+", default=sorted(SUPPORT_REASONS))
    parser.add_argument("--keep-hard-nonoracle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--selected-weight-multiplier", type=float, default=2.0)
    parser.add_argument("--q3-weight-multiplier", type=float, default=1.5)
    parser.add_argument("--missed-keep-weight-multiplier", type=float, default=1.5)
    parser.add_argument("--max-weight", type=float, default=20.0, help="Clamp output weights; <=0 disables")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    if args.method_preset == "trust_region_reweight":
        args.mode = "trust_region_soft_or"
        args.selection_action = "reweight"
        args.include_support = True
        args.score_normalization = "zscore_sigmoid"

    input_path = Path(args.dataset)
    dataset = torch.load(input_path, map_location="cpu")
    metadata = [dict(item) for item in dataset.get("metadata", [])]
    if len(metadata) != int(dataset["labels"].numel()):
        raise ValueError("Dataset metadata length does not match labels.")

    labels = dataset["labels"].detach().cpu().float()
    weights = dataset["weights"].detach().cpu().float()
    for idx, row in enumerate(metadata):
        row["row_index"] = idx
        row["label"] = float(labels[idx].item())
        row["weight"] = float(weights[idx].item())

    entropy_cut, divergence_cut = attach_selection_scores(
        metadata,
        policy_prob_field=args.policy_prob_field,
        entropy_threshold=args.entropy_threshold,
        divergence_threshold=args.divergence_threshold,
        entropy_quantile=args.entropy_quantile,
        divergence_quantile=args.divergence_quantile,
        score_normalization=args.score_normalization,
    )

    selected = select_oracle_indices(
        metadata,
        mode=args.mode,
        rho=args.rho,
        top_k=args.top_k,
        min_per_budget=args.min_per_budget,
    )
    if args.selection_action == "filter" and args.include_support and args.mode != "all_decisions":
        support_reasons = set(args.keep_support_reasons)
        selected.update(
            int(row["row_index"])
            for row in metadata
            if support_row(row, support_reasons, args.keep_hard_nonoracle)
        )
    if not selected:
        raise ValueError("Selection produced zero rows.")

    if args.selection_action == "reweight":
        output_indices = torch.arange(labels.numel(), dtype=torch.long)
        output_weights = apply_reweighting(
            metadata,
            weights,
            selected,
            selected_weight_multiplier=args.selected_weight_multiplier,
            q3_weight_multiplier=args.q3_weight_multiplier,
            missed_keep_weight_multiplier=args.missed_keep_weight_multiplier,
            max_weight=args.max_weight,
        )
        output_metadata = metadata
    else:
        output_indices = torch.tensor(sorted(selected), dtype=torch.long)
        output_weights = weights[output_indices]
        output_metadata = [metadata[int(idx)] for idx in output_indices.tolist()]

    output_dataset = dict(dataset)
    output_dataset["features"] = dataset["features"].detach().cpu()[output_indices]
    output_dataset["labels"] = labels[output_indices]
    output_dataset["weights"] = output_weights
    output_dataset["metadata"] = output_metadata
    output_dataset["selection"] = {
        "source_dataset": str(input_path),
        "method_preset": args.method_preset,
        "mode": args.mode,
        "canonical_mode": canonical_mode(args.mode),
        "selection_action": args.selection_action,
        "rho": args.rho,
        "top_k": args.top_k,
        "min_per_budget": args.min_per_budget,
        "policy_prob_field": args.policy_prob_field,
        "entropy_threshold": entropy_cut,
        "divergence_threshold": divergence_cut,
        "score_normalization": args.score_normalization,
        "include_support": args.include_support,
        "keep_support_reasons": args.keep_support_reasons,
        "keep_hard_nonoracle": args.keep_hard_nonoracle,
        "selected_weight_multiplier": args.selected_weight_multiplier,
        "q3_weight_multiplier": args.q3_weight_multiplier,
        "missed_keep_weight_multiplier": args.missed_keep_weight_multiplier,
        "max_weight": args.max_weight,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_dataset, output_path)

    input_oracle_rows = sum(1 for row in metadata if row.get("oracle_scored"))
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "method_preset": args.method_preset,
        "mode": args.mode,
        "canonical_mode": canonical_mode(args.mode),
        "selection_action": args.selection_action,
        "rho": args.rho,
        "policy_prob_field": args.policy_prob_field,
        "entropy_threshold": entropy_cut,
        "divergence_threshold": divergence_cut,
        "score_normalization": args.score_normalization,
        "input_rows": len(metadata),
        "input_oracle_rows": input_oracle_rows,
        "input_label_mass": float(labels.sum().item()),
        **summarize(
            metadata,
            selected,
            output_indices=output_indices,
            output_weights=output_weights,
        ),
    }

    print("=" * 72)
    print("OP Oracle Dataset Selection")
    print("=" * 72)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(
        f"Preset: {args.method_preset}  mode={args.mode} "
        f"(canonical={canonical_mode(args.mode)})  action={args.selection_action}  rho={args.rho:g}"
    )
    print(f"Thresholds: entropy={entropy_cut:.4f}, divergence={divergence_cut:.4f}")
    print(
        f"Rows: {report['input_rows']} -> {report['output_rows']} output / "
        f"{report['selected_rows']} selected  "
        f"oracle: {report['input_oracle_rows']} -> {report['selected_oracle_rows']}  "
        f"label_mass: {report['input_label_mass']:.1f} -> {report['selected_label_mass']:.1f}"
    )
    print(f"Output weight mean/max: {report['output_weight_mean']} / {report['output_weight_max']}")
    print(
        "Trust/outlier oracle rows: "
        f"{report['trust_region_oracle_rows']} trust, "
        f"{report['outlier_oracle_rows']} outlier -> "
        f"{report['selected_outlier_rows']} selected"
    )
    print(f"By budget: {report['by_budget']}")
    print(f"Oracle by budget: {report['oracle_by_budget']}")
    print(f"KV-TIP quadrants: {report['kvtip_quadrants']}")
    print(f"Decision types: {report['decision_types']}")
    print(f"Trust region types: {report['trust_region_types']}")
    print(f"Training treatments: {report['training_treatments']}")

    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_md:
        write_markdown(Path(args.output_md), report)


if __name__ == "__main__":
    main()
