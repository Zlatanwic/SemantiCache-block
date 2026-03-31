"""Compute 95% confidence intervals for paper tables from raw result data.

Usage:
    python compute_paper_ci.py

Outputs printed mean ± CI for each table.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from math import sqrt


# ---------------------------------------------------------------------------
# t critical values for 95% CI (two-tailed)
# ---------------------------------------------------------------------------
# Pre-computed t_{0.975, df} for common df values
_T_CRIT = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    40: 2.021, 50: 2.010, 60: 2.000, 80: 1.990, 100: 1.984,
    200: 1.972, 500: 1.965, 1000: 1.962,
}


def t_crit(n: int) -> float:
    """Return t critical value for 95% CI with n-1 degrees of freedom."""
    df = n - 1
    if df <= 0:
        return 0.0
    if df in _T_CRIT:
        return _T_CRIT[df]
    # Extrapolate for larger n using normal approximation
    if df > 1000:
        return 1.962
    return 2.0  # fallback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def sem(arr: np.ndarray) -> float:
    """Standard error of mean."""
    n = len(arr)
    if n <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / sqrt(n))


def ci_half_width(arr: np.ndarray) -> float:
    """Half-width of 95% CI."""
    n = len(arr)
    if n <= 1:
        return 0.0
    return sem(arr) * t_crit(n)


def accuracy(data: list[dict], policy: str, budget: float | None = None) -> tuple[float, float, int]:
    """Compute mean accuracy (0-100), half-width CI, and n for a policy/budget."""
    if budget is not None:
        subset = [r for r in data if r.get("policy") == policy and r.get("budget_ratio", r.get("budget")) == budget]
    else:
        subset = [r for r in data if r.get("policy") == policy]
    if not subset:
        return 0.0, 0.0, 0
    scores = np.array([1 if (r.get("correct") or r.get("score", 0) == 100) else 0 for r in subset], dtype=float)
    n = len(scores)
    mean = float(scores.mean() * 100)
    ci = ci_half_width(scores * 100)
    return mean, ci, n


def ci_str(mean: float, ci: float, n: int) -> str:
    """Format as 'XX.X±X.X' with n."""
    return f"{mean:.1f}±{ci:.1f} (n={n})"


# ---------------------------------------------------------------------------
# Table 1: Main RULER NIAH accuracy by policy × budget
# ---------------------------------------------------------------------------

def table1_main_results():
    print("=" * 70)
    print("TABLE 1: RULER NIAH Retrieval Accuracy (%)")
    print("=" * 70)

    # Load all relevant data
    data5 = load_json("results/ruler_niah/ruler_niah_5policy.json")
    datakv = load_json("results/ruler_niah/ruler_niah_new_baselines_v2.json")

    # Policies: full, streaming, h2o, snapkv, kvzip, semantic
    # Budgets: 0.5, 0.3, 0.2
    policies = ["full", "streaming", "h2o", "snapkv", "kvzip", "semantic"]
    budgets = [0.5, 0.3, 0.2]

    # Build combined dataset, removing duplicates
    # (snapkv and streaming appear in both files)
    seen = set()
    all_data = []
    for r in data5 + datakv:
        key = (r.get("policy"), r.get("budget_ratio", r.get("budget")),
               r.get("depth"), r.get("needle_index"), r.get("trial", r.get("run", 0)))
        if key not in seen:
            seen.add(key)
            all_data.append(r)

    print(f"\n{'Policy':<20} {'b=50%':>18} {'b=30%':>18} {'b=20%':>18}")
    print("-" * 74)

    for policy in policies:
        row = f"{policy:<20}"
        for b in budgets:
            mean, ci, n = accuracy(all_data, policy, b)
            row += f" {mean:5.1f}±{ci:4.1f} (n={n:2d})"
        print(row)

    print()
    return all_data


# ---------------------------------------------------------------------------
# Table 2: Context length scaling at b=20%
# ---------------------------------------------------------------------------

def table2_scaling():
    print("=" * 70)
    print("TABLE 2: Context Length Scaling at b=20%")
    print("=" * 70)

    # Use b20 file for b=20% data
    data = load_json("results/scaling/context_scaling_qwen_b20.json")

    policies = ["full", "streaming", "h2o", "snapkv", "semantic"]
    tokens = [1800, 4000, 6000]

    print(f"\n{'Policy':<16} {'1800t':>18} {'4000t':>18} {'6000t':>18}")
    print("-" * 70)

    for policy in policies:
        row = f"{policy:<16}"
        for t in tokens:
            subset = [r for r in data if r.get("policy") == policy and r.get("target_tokens") == t]
            if not subset:
                row += f" {'N/A':>18}"
                continue
            scores = np.array([1 if (r.get("correct") or r.get("score", 0) == 100) else 0 for r in subset], dtype=float)
            n = len(scores)
            mean = scores.mean() * 100
            ci = ci_half_width(scores * 100)
            row += f" {mean:5.1f}±{ci:4.1f} (n={n:2d})"
        print(row)

    print()
    return data


# ---------------------------------------------------------------------------
# Table 3: Multi-needle NIAH at b=20%
# ---------------------------------------------------------------------------

def table3_multi_needle():
    print("=" * 70)
    print("TABLE 3: Multi-Needle NIAH at b=20%")
    print("=" * 70)

    # Multi-needle results - check if they exist, otherwise run or report missing
    path = Path("results/multi_needle/multi_needle_qwen.json")
    if path.exists():
        data = load_json(str(path))
    else:
        print("\n[NOTE] results/multi_needle/multi_needle_qwen.json not found.")
        print("Multi-needle data needs to be collected with:")
        print("  python eval_memory_and_scaling.py multi-needle-qwen --policies full streaming h2o snapkv semantic --needle-counts 1 2 4")
        print()

        # Show what paper claims
        print("Paper claims (from tex):")
        print("  k=1: Full=100, Streaming=0, H2O=0, SnapKV=100, SemantiCache=100")
        print("  k=2: Full=80, Streaming=50, H2O=45, SnapKV=70, SemantiCache=85")
        print("  k=4: Full=80, Streaming=25, H2O=20, SnapKV=70, SemantiCache=92.5")
        return None

    policies = ["full", "streaming", "h2o", "snapkv", "semantic"]
    needle_counts = [1, 2, 4]

    print(f"\n{'Policy':<16} {'k=1':>18} {'k=2':>18} {'k=4':>18}")
    print("-" * 70)

    for policy in policies:
        row = f"{policy:<16}"
        for k in needle_counts:
            subset = [r for r in data if r.get("policy") == policy and r.get("num_needles") == k]
            if not subset:
                row += f" {'N/A':>18}"
                continue
            scores = np.array([r.get("score", 0) / 100.0 for r in subset], dtype=float)
            n = len(scores)
            mean = scores.mean() * 100
            ci = ci_half_width(scores * 100)
            row += f" {mean:5.1f}±{ci:4.1f} (n={n:2d})"
        print(row)

    print()
    return data


# ---------------------------------------------------------------------------
# Table 4: Ablation study at b=20%
# ---------------------------------------------------------------------------

def table4_ablation():
    print("=" * 70)
    print("TABLE 4: Ablation Study at b=20%")
    print("=" * 70)

    data = load_json("results/ablation/ablation_results.json")

    # Map ablation name to table label
    ablation_map = {
        "full_model": "Full model",
        "no_attention": r"$-$ Attention",
        "no_info_density": r"$-$ Info density",
        "no_query_relevance": r"$-$ Query relevance",
        "no_factual_bonus": r"$-$ Factual bonus",
        "no_role_pinning": r"$-$ Role pinning",
        "attention_only": "Attention only",
    }

    # Order as in paper
    order = ["full_model", "no_attention", "no_info_density", "no_query_relevance", "no_factual_bonus", "no_role_pinning", "attention_only"]

    print(f"\n{'Variant':<30} {'Acc (%)':>12} {'Δ':>8} {'n':>5}")
    print("-" * 58)

    full_acc = None
    for abl in order:
        subset = [r for r in data if r.get("ablation") == abl]
        if not subset:
            continue
        scores = np.array([1 if (r.get("correct") or r.get("score", 0) == 100) else 0 for r in subset], dtype=float)
        n = len(scores)
        mean = float(scores.mean() * 100)
        ci = ci_half_width(scores * 100)

        if abl == "full_model":
            delta_str = "---"
            full_acc = mean
        else:
            delta_val = mean - full_acc if full_acc else 0
            delta_str = f"{delta_val:>+7.1f}"

        label = ablation_map.get(abl, abl)
        print(f"{label:<30} {mean:5.1f}±{ci:4.1f} {delta_str} {n:>5d}")

    print()
    return data


# ---------------------------------------------------------------------------
# Table 5: Systems overhead
# ---------------------------------------------------------------------------

def table5_overhead():
    print("=" * 70)
    print("TABLE 5: Systems Overhead")
    print("=" * 70)

    data_path = Path("results/overhead/overhead_profiling.json")
    if not data_path.exists():
        print("\n[NOTE] overhead_profiling.json not found.")
        return None

    data = load_json(str(data_path))

    # Full policy has budget_ratio=0.0 (no eviction); others have 0.2
    policies_order = ["full", "streaming", "h2o", "snapkv", "kvzip", "semantic", "tiered_semantic"]

    print(f"\n{'Policy':<22} {'Prefill':>8} {'Snap':>8} {'Evict/step':>11} {'Decode':>8} {'tok/s':>8} {'n':>5}")
    print("-" * 75)

    for policy in policies_order:
        budget = 0.0 if policy == "full" else 0.2
        subset = [r for r in data
                  if r.get("policy") == policy and r.get("budget_ratio") == budget]
        if not subset:
            continue
        n = len(subset)

        prefill = np.array([r.get("prefill_time", 0) for r in subset])
        snap = np.array([r.get("snapshot_time", 0) for r in subset])
        evict = np.array([r.get("eviction_time_per_step", 0) * 1000 for r in subset])
        decode = np.array([r.get("decode_time", 0) for r in subset])
        tps = np.array([r.get("tokens_per_second", 0) for r in subset])

        pf_m = float(prefill.mean())
        pf_ci = ci_half_width(prefill)
        snap_m = float(snap.mean()) if snap.max() > 0 else 0.0
        ev_m = float(evict.mean())
        ev_ci = ci_half_width(evict)
        dec_m = float(decode.mean())
        dec_ci = ci_half_width(decode)
        tps_m = float(tps.mean())
        tps_ci = ci_half_width(tps)

        snap_str = f"{snap_m:>6.2f}" if snap_m > 0 else f"{'--':>8}"
        print(f"{policy:<22} {pf_m:>6.2f}±{pf_ci:<1.2f} {snap_str} {ev_m:>9.2f}±{ev_ci:<2.2f} {dec_m:>6.2f}±{dec_ci:<1.2f} {tps_m:>6.1f}±{tps_ci:<1.1f} {n:>5d}")

    print()
    return data


# ---------------------------------------------------------------------------
# Table 6: Cross-architecture on Llama at b=20%
# ---------------------------------------------------------------------------

def table6_cross_arch():
    print("=" * 70)
    print("TABLE 6: Cross-Architecture Validation (Llama-3.2-3B-Instruct, b=20%)")
    print("=" * 70)

    data = load_json("results/ruler_niah/llama3b_1800.json")

    policies = ["full", "snapkv", "semantic"]

    print(f"\n{'Policy':<20} {'Accuracy (%)':>14} {'tok/s':>10} {'n':>5}")
    print("-" * 50)

    for policy in policies:
        subset = [r for r in data if r.get("policy") == policy]
        if not subset:
            continue
        scores = np.array([1 if (r.get("correct") or r.get("score", 0) == 100) else 0 for r in subset], dtype=float)
        tps = np.array([r.get("tokens_per_second", 0) for r in subset])
        n = len(scores)

        acc_mean = float(scores.mean() * 100)
        acc_ci = ci_half_width(scores * 100)
        tps_mean = float(tps.mean())
        tps_ci = ci_half_width(tps)

        print(f"{policy:<20} {acc_mean:5.1f}±{acc_ci:4.1f} {tps_mean:5.1f}±{tps_ci:3.1f} {n:>5d}")

    print()
    return data


# ---------------------------------------------------------------------------
# Budget sensitivity (not a main paper table but useful for figure)
# ---------------------------------------------------------------------------

def budget_sensitivity():
    print("=" * 70)
    print("BUDGET SENSITIVITY (for Figure)")
    print("=" * 70)

    # Budget sensitivity data - use dedicated file
    bs_path = Path("results/budget_sensitivity/budget_sensitivity.json")
    if bs_path.exists():
        data_bs = load_json(str(bs_path))
    else:
        data5 = load_json("results/ruler_niah/ruler_niah_5policy.json")
        datakv = load_json("results/ruler_niah/ruler_niah_new_baselines_v2.json")
        # Deduplicate
        seen = set()
        data_bs = []
        for r in data5 + datakv:
            key = (r.get("policy"), r.get("budget_ratio", r.get("budget")),
                   r.get("depth"), r.get("needle_index"), r.get("trial", r.get("run", 0)))
            if key not in seen:
                seen.add(key)
                data_bs.append(r)

    policies = ["full", "streaming", "h2o", "snapkv", "kvzip", "semantic"]
    budgets = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]

    print(f"\n{'Policy':<20}", end="")
    for b in budgets:
        print(f" {b:>5.0%}", end="")
    print()
    print("-" * (20 + 7 * len(budgets)))

    for policy in policies:
        print(f"{policy:<20}", end="")
        for b in budgets:
            subset = [r for r in data_bs
                      if r.get("policy") == policy and
                      abs(r.get("budget_ratio", r.get("budget", r.get("budget"))) - b) < 0.001]
            if not subset:
                print(f" {'--':>6}", end="")
                continue
            scores = np.array([1 if (r.get("correct") or r.get("score", 0) == 100) else 0 for r in subset], dtype=float)
            mean = float(scores.mean() * 100)
            print(f" {mean:>5.1f}", end="")
        print()

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n*** Computing 95% Confidence Intervals for Paper Tables ***\n")

    table1_main_results()
    table2_scaling()
    table3_multi_needle()
    table4_ablation()
    table5_overhead()
    table6_cross_arch()
    budget_sensitivity()

    print("\nDone. Use these values to update the paper tables.")
    print("Note: For ablation (Table 4), the ablation_results.json 'no_attention'")
    print("variant shows 70% (same as full) but paper claims no change.")
    print("Verify the ablation naming matches the paper description.")
