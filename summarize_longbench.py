"""Summarize LongBench results and compute per-task scores with CI.

Usage:
    python summarize_longbench.py results/longbench/longbench_results.json

Outputs:
    - Per-task mean score by policy × budget
    - 95% confidence intervals
    - Aggregated summary table
"""

import json
import numpy as np
import sys
from math import sqrt
from pathlib import Path


_T_CRIT = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    30: 2.042, 40: 2.021, 50: 2.010,
}


def t_crit(n: int) -> float:
    df = n - 1
    if df <= 0:
        return 0.0
    return _T_CRIT.get(df, 2.042)


def ci_half(arr: np.ndarray) -> float:
    n = len(arr)
    if n <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / sqrt(n)) * t_crit(n)


def summarize_longbench(json_path: str):
    with open(json_path) as f:
        data = json.load(f)

    # Group results
    groups: dict[tuple, list] = {}
    for r in data:
        if "error" in r:
            continue
        key = (r["task"], r["policy"], r["budget"])
        groups.setdefault(key, []).append(r["score"])

    tasks = sorted(set(k[0] for k in groups))
    policies = sorted(set(k[1] for k in groups))
    budgets = sorted(set(k[2] for k in groups))

    print(f"Loaded {len(data)} results, {len(groups)} groups")
    print(f"Tasks: {tasks}")
    print(f"Policies: {policies}")
    print(f"Budgets: {budgets}")
    print()

    # Per-task, per-policy, per-budget summary
    print("=" * 90)
    print("LONGBENCH RESULTS (mean ± 95% CI)")
    print("=" * 90)

    for task in tasks:
        print(f"\n--- {task} ---")
        print(f"{'Policy':<16} {'Budget':>8} {'Mean':>8} {'±CI':>8} {'n':>5}")
        print("-" * 50)
        for policy in policies:
            for budget in budgets:
                key = (task, policy, budget)
                if key not in groups:
                    continue
                scores = np.array(groups[key])
                n = len(scores)
                mean = float(scores.mean())
                ci = ci_half(scores)
                print(f"{policy:<16} {budget:>7.0%} {mean:>8.3f} {ci:>8.3f} {n:>5d}")

    # Overall summary: average across tasks for each policy × budget
    print("\n" + "=" * 90)
    print("OVERALL AVERAGE (across tasks)")
    print("=" * 90)
    print(f"\n{'Policy':<16}", end="")
    for b in budgets:
        print(f" {b:>7.0%}", end="")
    print()
    print("-" * (16 + 8 * len(budgets)))

    for policy in policies:
        print(f"{policy:<16}", end="")
        for budget in budgets:
            task_scores = []
            for task in tasks:
                key = (task, policy, budget)
                if key in groups:
                    task_scores.extend(groups[key])
            if not task_scores:
                print(f" {'N/A':>7}", end="")
                continue
            arr = np.array(task_scores)
            mean = float(arr.mean())
            print(f" {mean:>7.3f}", end="")
        print()

    # Save per-group statistics
    out_path = Path(json_path).parent / "longbench_summary.json"
    summary = {}
    for key, scores in groups.items():
        task, policy, budget = key
        arr = np.array(scores)
        n = len(arr)
        mean = float(arr.mean())
        ci = ci_half(arr)
        k = f"{task}_{policy}_b{int(budget*100)}"
        summary[k] = {
            "task": task, "policy": policy, "budget": budget,
            "mean": mean, "ci95": ci, "n": n, "std": float(arr.std(ddof=1))
        }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarize_longbench.py <results.json>")
        sys.exit(1)
    summarize_longbench(sys.argv[1])
