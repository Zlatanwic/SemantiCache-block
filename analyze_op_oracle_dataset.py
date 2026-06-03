"""Analyze OP-SieveKV oracle labels with KV-TIP-style diagnostics.

This script is intentionally lightweight: it reads a collected `.pt` dataset
and summarizes oracle-scored segment decisions by budget, task, label reason,
and entropy/divergence quadrant. It helps decide whether the next improvement
should target over-retention, missed evidence, or budget-specific behavior.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from op_policy_model import FEATURE_NAMES, load_policy_checkpoint


def binary_entropy(prob: float) -> float:
    p = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


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


def quadrant(entropy: float, divergence: float, entropy_threshold: float, divergence_threshold: float) -> str:
    high_entropy = entropy >= entropy_threshold
    high_divergence = divergence >= divergence_threshold
    if not high_entropy and not high_divergence:
        return "Q1_confident_aligned"
    if high_entropy and not high_divergence:
        return "Q2_uncertain_aligned"
    if not high_entropy and high_divergence:
        return "Q3_confident_wrong"
    return "Q4_uncertain_wrong"


def decision_type(policy_prob: float, oracle_label: float) -> str:
    if policy_prob >= 0.5 and oracle_label < 0.5:
        return "over_keep"
    if policy_prob < 0.5 and oracle_label >= 0.5:
        return "missed_keep"
    if policy_prob >= 0.5 and oracle_label >= 0.5:
        return "aligned_keep"
    return "aligned_drop"


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "rows": len(rows),
        "oracle_rows": sum(1 for row in rows if row.get("oracle_scored")),
        "hard_labels": sum(1 for row in rows if row["label"] >= 0.5),
        "label_mass": round(sum(float(row["label"]) for row in rows), 4),
        "mean_label": round(mean([float(row["label"]) for row in rows]), 4),
        "mean_weight": round(mean([float(row["weight"]) for row in rows]), 4),
        "mean_heuristic": round(mean([float(row.get("heuristic_keep_prob", 0.0)) for row in rows]), 4),
        "mean_oracle_label": round(
            mean([float(row["oracle_label"]) for row in rows if row.get("oracle_label") is not None]),
            4,
        ),
        "mean_oracle_delta": round(
            mean([float(row["oracle_delta"]) for row in rows if row.get("oracle_delta") is not None]),
            4,
        ),
        "reasons": dict(Counter(str(row.get("label_reason", "unknown")) for row in rows)),
    }


def build_rows(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    labels = dataset["labels"].detach().cpu().float()
    weights = dataset["weights"].detach().cpu().float()
    metadata = dataset.get("metadata", [])
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(metadata):
        row = dict(item)
        row["row_index"] = idx
        row["label"] = float(labels[idx].item())
        row["weight"] = float(weights[idx].item())
        rows.append(row)
    return rows


@torch.no_grad()
def attach_learned_policy_probs(dataset: dict[str, Any], rows: list[dict[str, Any]], policy_ckpt: str, batch_size: int) -> None:
    """Attach learned policy keep probabilities to rows."""
    if dataset.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Dataset feature schema does not match current OP policy feature schema.")
    model, feature_mean, feature_std, metadata = load_policy_checkpoint(policy_ckpt, map_location="cpu")
    features = dataset["features"].detach().cpu().float()
    probs: list[torch.Tensor] = []
    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size]
        normalized = (batch - feature_mean) / feature_std
        probs.append(torch.sigmoid(model(normalized)).detach().cpu())
    all_probs = torch.cat(probs, dim=0) if probs else torch.empty(0)
    for idx, row in enumerate(rows):
        row["learned_keep_prob"] = float(all_probs[idx].item())
    for key, value in metadata.items():
        if key in {"dataset", "best_val_loss", "hard_positive_rows", "label_mass"}:
            print(f"Policy checkpoint metadata: {key}={value}")


def attach_kvtip(rows: list[dict[str, Any]], args) -> tuple[float, float]:
    oracle_rows = [row for row in rows if row.get("oracle_scored") and row.get("oracle_label") is not None]
    for row in oracle_rows:
        policy_prob = float(row.get(args.policy_prob_field, row.get("heuristic_keep_prob", 0.0)))
        oracle_label = float(row.get("oracle_label", 0.0))
        row["retention_entropy"] = binary_entropy(policy_prob)
        row["oracle_policy_divergence"] = abs(oracle_label - policy_prob)
        row["decision_type"] = decision_type(policy_prob, oracle_label)
        row["analysis_policy_prob"] = policy_prob

    entropies = [float(row["retention_entropy"]) for row in oracle_rows]
    divergences = [float(row["oracle_policy_divergence"]) for row in oracle_rows]
    entropy_threshold = args.entropy_threshold
    divergence_threshold = args.divergence_threshold
    if entropy_threshold is None:
        entropy_threshold = quantile(entropies, args.entropy_quantile)
    if divergence_threshold is None:
        divergence_threshold = quantile(divergences, args.divergence_quantile)

    for row in oracle_rows:
        row["kvtip_quadrant"] = quadrant(
            float(row["retention_entropy"]),
            float(row["oracle_policy_divergence"]),
            entropy_threshold,
            divergence_threshold,
        )
    return float(entropy_threshold), float(divergence_threshold)


def top_q3_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    q3 = [
        row
        for row in rows
        if row.get("kvtip_quadrant") == "Q3_confident_wrong"
    ]
    q3.sort(key=lambda row: float(row.get("oracle_policy_divergence", 0.0)), reverse=True)
    keys = [
        "row_index",
        "task",
        "budget",
        "num_needles",
        "needle_position",
        "segment_start",
        "segment_end",
        "label_reason",
        "decision_type",
        "heuristic_keep_prob",
        "learned_keep_prob",
        "analysis_policy_prob",
        "oracle_label",
        "oracle_delta",
        "retention_entropy",
        "oracle_policy_divergence",
    ]
    return [{key: row.get(key) for key in keys if key in row} for row in q3[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze OP-SieveKV oracle distillation dataset")
    parser.add_argument("--dataset", default="results/op_oracle_dataset_conservative.pt")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--policy-ckpt", default=None, help="Optional learned OP policy checkpoint for student-policy Q3 analysis")
    parser.add_argument("--policy-prob-field", default="heuristic_keep_prob", help="Row field used as policy probability")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--top-q3", type=int, default=30)
    parser.add_argument("--entropy-threshold", type=float, default=None)
    parser.add_argument("--divergence-threshold", type=float, default=None)
    parser.add_argument("--entropy-quantile", type=float, default=0.5)
    parser.add_argument("--divergence-quantile", type=float, default=0.75)
    args = parser.parse_args()

    path = Path(args.dataset)
    dataset = torch.load(path, map_location="cpu")
    rows = build_rows(dataset)
    if args.policy_ckpt:
        attach_learned_policy_probs(dataset, rows, args.policy_ckpt, args.batch_size)
        args.policy_prob_field = "learned_keep_prob"
    entropy_threshold, divergence_threshold = attach_kvtip(rows, args)
    oracle_rows = [row for row in rows if row.get("oracle_scored")]

    by_budget: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_budget_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        budget = str(row.get("budget", "unknown"))
        task = str(row.get("task", "unknown"))
        by_budget[budget].append(row)
        by_task[task].append(row)
        by_budget_task[f"{budget}:{task}"].append(row)

    oracle_by_budget: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in oracle_rows:
        oracle_by_budget[str(row.get("budget", "unknown"))].append(row)

    report = {
        "dataset": str(path),
        "rows": len(rows),
        "oracle_rows": len(oracle_rows),
        "feature_names": dataset.get("feature_names", []),
        "policy_ckpt": args.policy_ckpt,
        "policy_prob_field": args.policy_prob_field,
        "entropy_threshold": round(entropy_threshold, 4),
        "divergence_threshold": round(divergence_threshold, 4),
        "overall": summarize_group(rows),
        "oracle_overall": summarize_group(oracle_rows),
        "by_budget": {key: summarize_group(value) for key, value in sorted(by_budget.items())},
        "oracle_by_budget": {key: summarize_group(value) for key, value in sorted(oracle_by_budget.items())},
        "by_task": {key: summarize_group(value) for key, value in sorted(by_task.items())},
        "by_budget_task": {key: summarize_group(value) for key, value in sorted(by_budget_task.items())},
        "kvtip_quadrants": dict(Counter(str(row.get("kvtip_quadrant", "not_oracle")) for row in rows)),
        "decision_types": dict(Counter(str(row.get("decision_type", "not_oracle")) for row in rows)),
        "top_q3": top_q3_rows(rows, args.top_q3),
    }

    print("=" * 72)
    print("OP-SieveKV Oracle Dataset Analysis")
    print("=" * 72)
    print(f"Dataset: {path}")
    print(f"Rows: {report['rows']}  Oracle rows: {report['oracle_rows']}")
    print(f"Policy probability field: {args.policy_prob_field}")
    print(f"Thresholds: entropy={entropy_threshold:.4f}, divergence={divergence_threshold:.4f}")
    print(f"Overall: {report['overall']}")
    print(f"Oracle overall: {report['oracle_overall']}")
    print(f"KV-TIP quadrants: {report['kvtip_quadrants']}")
    print(f"Decision types: {report['decision_types']}")
    print("Oracle by budget:")
    for budget, summary in report["oracle_by_budget"].items():
        print(f"  b={budget}: {summary}")
    print("Top Q3 rows:")
    for row in report["top_q3"][: min(10, len(report["top_q3"]))]:
        print(f"  {row}")

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON report to {output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# OP-SieveKV Oracle Dataset Analysis",
            "",
            f"- Dataset: `{path}`",
            f"- Rows: {report['rows']}",
            f"- Oracle rows: {report['oracle_rows']}",
            f"- Policy checkpoint: `{args.policy_ckpt}`",
            f"- Policy probability field: `{args.policy_prob_field}`",
            f"- Entropy threshold: {entropy_threshold:.4f}",
            f"- Divergence threshold: {divergence_threshold:.4f}",
            "",
            "## KV-TIP Quadrants",
            "",
            *[f"- {key}: {value}" for key, value in report["kvtip_quadrants"].items()],
            "",
            "## Decision Types",
            "",
            *[f"- {key}: {value}" for key, value in report["decision_types"].items()],
            "",
            "## Oracle By Budget",
            "",
        ]
        for budget, summary in report["oracle_by_budget"].items():
            lines.append(f"### Budget {budget}")
            lines.append("")
            lines.append(f"- Rows: {summary.get('rows', 0)}")
            lines.append(f"- Hard labels: {summary.get('hard_labels', 0)}")
            lines.append(f"- Label mass: {summary.get('label_mass', 0)}")
            lines.append(f"- Mean oracle delta: {summary.get('mean_oracle_delta', 0)}")
            lines.append(f"- Mean oracle label: {summary.get('mean_oracle_label', 0)}")
            lines.append(f"- Reasons: `{summary.get('reasons', {})}`")
            lines.append("")
        lines.extend(["## Top Q3 Confident-Wrong Rows", ""])
        for row in report["top_q3"]:
            lines.append(f"- `{row}`")
        output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved Markdown report to {output_md}")


if __name__ == "__main__":
    main()
