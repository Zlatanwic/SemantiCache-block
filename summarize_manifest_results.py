"""Summarize fixed-manifest evaluation results with confidence intervals.

This is intentionally generic for the OP-SieveKV v2 workflow. It reads JSON
results from `eval_niah.py --manifest` or `eval_multi_needle.py --manifest`,
groups by policy / budget / optional num_needles, and reports mean quality with
95% confidence intervals. When a baseline policy is supplied, it also reports a
paired delta over shared sample ids.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return Wilson interval bounds for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    spread = z * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total))) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def bootstrap_ci(values: list[float], samples: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Return percentile bootstrap CI for a mean."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(samples):
        means.append(mean([values[rng.randrange(n)] for _ in range(n)]))
    means.sort()
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return lo, hi


def quality_value(row: dict[str, Any]) -> float:
    """Return a 0..1 quality value from a result row."""
    if "score" in row:
        return float(row.get("score", 0.0)) / 100.0
    return 1.0 if bool(row.get("correct")) else 0.0


def group_key(row: dict[str, Any]) -> tuple:
    key = [str(row.get("policy")), round(float(row.get("budget", row.get("budget_ratio", 0.0))), 6)]
    if "num_needles" in row:
        key.append(int(row["num_needles"]))
    return tuple(key)


def group_label(key: tuple) -> str:
    if len(key) == 2:
        policy, budget = key
        return f"{policy} b={budget:.0%}"
    policy, budget, num_needles = key
    return f"{policy} b={budget:.0%} k={num_needles}"


def sample_key(row: dict[str, Any]) -> tuple:
    """Stable sample identity for paired comparisons."""
    return (
        row.get("sample_id"),
        row.get("needle_index"),
        row.get("needle_position", row.get("depth")),
        row.get("paragraph_offset"),
        row.get("num_needles"),
        tuple(row.get("needle_values", [])),
    )


def summarize_group(rows: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    values = [quality_value(row) for row in rows]
    n = len(values)
    avg = mean(values)
    if all(value in {0.0, 1.0} for value in values):
        lo, hi = wilson_ci(int(sum(values)), n)
    else:
        lo, hi = bootstrap_ci(values, samples=bootstrap_samples, seed=seed)
    return {
        "n": n,
        "mean": avg,
        "ci_low": lo,
        "ci_high": hi,
    }


def paired_delta(
    rows: list[dict[str, Any]],
    key: tuple,
    baseline_policy: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any] | None:
    policy = key[0]
    if policy == baseline_policy:
        return None

    target_by_sample = {
        sample_key(row): quality_value(row)
        for row in rows
        if group_key(row) == key
    }
    baseline_key = (baseline_policy, *key[1:])
    baseline_by_sample = {
        sample_key(row): quality_value(row)
        for row in rows
        if group_key(row) == baseline_key
    }
    shared = sorted(set(target_by_sample) & set(baseline_by_sample))
    if not shared:
        return None
    deltas = [target_by_sample[item] - baseline_by_sample[item] for item in shared]
    lo, hi = bootstrap_ci(deltas, samples=bootstrap_samples, seed=seed)
    return {
        "paired_n": len(shared),
        "delta": mean(deltas),
        "delta_ci_low": lo,
        "delta_ci_high": hi,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fixed-manifest eval results with CIs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--baseline-policy", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--append-to", default=None, help="Append the markdown report to an experiment log")
    parser.add_argument("--title", default=None, help="Section title when appending to an experiment log")
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    report: list[dict[str, Any]] = []
    for key in sorted(groups):
        summary = summarize_group(groups[key], args.bootstrap_samples, args.seed)
        item = {
            "group": group_label(key),
            "policy": key[0],
            "budget": key[1],
            **({"num_needles": key[2]} if len(key) == 3 else {}),
            **summary,
        }
        if args.baseline_policy:
            delta = paired_delta(rows, key, args.baseline_policy, args.bootstrap_samples, args.seed)
            if delta:
                item.update(delta)
        report.append(item)

    lines = [
        "# Manifest Evaluation Summary",
        "",
        f"- Input: `{args.input}`",
        f"- Rows: {len(rows)}",
        f"- Baseline policy: `{args.baseline_policy}`" if args.baseline_policy else "- Baseline policy: none",
        "",
        "| Group | n | Mean | 95% CI | Paired delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report:
        mean_pct = item["mean"] * 100.0
        ci = f"[{item['ci_low'] * 100.0:.1f}, {item['ci_high'] * 100.0:.1f}]"
        if "delta" in item:
            delta = (
                f"{item['delta'] * 100.0:+.1f} "
                f"[{item['delta_ci_low'] * 100.0:+.1f}, {item['delta_ci_high'] * 100.0:+.1f}] "
                f"(n={item['paired_n']})"
            )
        else:
            delta = ""
        lines.append(f"| {item['group']} | {item['n']} | {mean_pct:.1f} | {ci} | {delta} |")

    markdown = "\n".join(lines) + "\n"
    print(markdown)

    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.append_to:
        path = Path(args.append_to)
        path.parent.mkdir(parents=True, exist_ok=True)
        title = args.title or Path(args.input).stem
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        section = f"\n\n## {title}\n\nRecorded: {timestamp}\n\n{markdown}"
        with path.open("a", encoding="utf-8") as file:
            file.write(section)


if __name__ == "__main__":
    main()
