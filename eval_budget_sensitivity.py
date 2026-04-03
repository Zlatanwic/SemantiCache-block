"""Budget sensitivity curve for all policies on RULER NIAH.

Runs a finer grid of cache budgets to produce accuracy-vs-budget curves.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from config import ExperimentConfig
from eval_ruler_niah import (
    build_pg_haystack,
    insert_needle_at_depth,
    make_ruler_needle,
    run_ruler_eval,
    string_match_all,
)
from run_generation import load_model


def run_budget_sensitivity(
    model,
    tokenizer,
    policies: list[str],
    budgets: list[float],
    target_tokens: int = 1800,
    depths: list[float] | None = None,
    num_needles: int = 5,
    seed: int = 42,
    max_new_tokens: int = 64,
    output_path: str | None = None,
) -> list[dict]:
    """Run RULER NIAH across a fine budget grid."""
    if depths is None:
        depths = list(np.round(np.linspace(0, 1, num=5, endpoint=True), 2))

    rng = random.Random(seed)
    needles = [make_ruler_needle("numbers", rng) for _ in range(num_needles)]

    total = len(policies) * len(budgets) * len(depths) * num_needles
    print(f"Budget sensitivity: {total} total runs")
    print(f"  policies: {policies}")
    print(f"  budgets: {budgets}")
    print(f"  depths: {len(depths)} points")
    print(f"  needles: {num_needles}")
    print()

    results: list[dict] = []
    run_idx = 0
    t0 = time.time()

    for policy in policies:
        for budget in budgets:
            combo_results = []
            for depth in depths:
                for ni, needle in enumerate(needles):
                    run_idx += 1
                    elapsed = time.time() - t0
                    eta = (elapsed / run_idx) * (total - run_idx) if run_idx > 1 else 0
                    print(
                        f"[{run_idx}/{total}] {policy} b={budget:.0%} "
                        f"d={depth:.2f} n={ni+1} "
                        f"(elapsed={elapsed:.0f}s eta={eta:.0f}s)"
                    )

                    cfg = ExperimentConfig()
                    cfg.cache.policy = policy
                    cfg.cache.cache_budget = budget

                    result = run_ruler_eval(
                        model, tokenizer, cfg, needle,
                        target_tokens=target_tokens,
                        depth=depth,
                        max_new_tokens=max_new_tokens,
                    )
                    result["needle_index"] = ni + 1
                    combo_results.append(result)
                    results.append(result)
                    tag = "O" if result["correct"] else "X"
                    print(f"  [{tag}] val={needle.value} -> {result['output_text'][:60]}")

            correct = sum(1 for r in combo_results if r["correct"])
            total_combo = len(combo_results)
            print(f"  >> {policy} b={budget:.0%}: {correct}/{total_combo} ({correct/total_combo:.0%})\n")

        # Save checkpoint per policy
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved to: {out}")

    # Summary table
    print("\n" + "=" * 70)
    print("BUDGET SENSITIVITY SUMMARY")
    print("=" * 70)
    header = f"{'Policy':<16}" + "".join(f"  b={b:.0%}" for b in budgets)
    print(header)
    print("-" * len(header))
    for policy in policies:
        row = f"{policy:<16}"
        for budget in budgets:
            subset = [r for r in results if r["policy"] == policy and r["budget"] == budget]
            c = sum(1 for r in subset if r["correct"])
            t = len(subset)
            row += f"  {c:>2}/{t:<2} " if t > 0 else "   --  "
        print(row)

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    return results


def plot_budget_sensitivity(results_path: str, output_path: str = "results/budget_sensitivity/budget_curve.png"):
    """Generate budget sensitivity plot from results JSON."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(results_path) as f:
        results = json.load(f)

    policies = sorted(set(r["policy"] for r in results))
    budgets = sorted(set(r["budget"] for r in results))

    STYLE = {
        "full":        {"color": "#2ecc71", "marker": "s", "linestyle": "--", "label": "Full (upper bound)"},
        "streaming":   {"color": "#f39c12", "marker": "^", "linestyle": "-.", "label": "StreamingLLM"},
        "snapkv":      {"color": "#3498db", "marker": "D", "linestyle": "-.", "label": "SnapKV"},
        "defensivekv": {"color": "#9b59b6", "marker": "v", "linestyle": "-.", "label": "DefensiveKV"},
        "h2o":         {"color": "#e74c3c", "marker": "o", "linestyle": ":",  "label": "H2O"},
        "kvzip":       {"color": "#1abc9c", "marker": "P", "linestyle": "-.", "label": "KVzip"},
        "semantic":    {"color": "#e74c3c", "marker": "*", "linestyle": "-",  "label": "SieveKV (ours)"},
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for policy in policies:
        accs = []
        for budget in budgets:
            subset = [r for r in results if r["policy"] == policy and r["budget"] == budget]
            if subset:
                c = sum(1 for r in subset if r["correct"])
                accs.append(c / len(subset) * 100)
            else:
                accs.append(None)

        style = STYLE.get(policy, {"color": "gray", "marker": ".", "linestyle": "-", "label": policy})
        budget_pcts = [b * 100 for b in budgets]
        valid = [(b, a) for b, a in zip(budget_pcts, accs) if a is not None]
        if valid:
            bx, ay = zip(*valid)
            linewidth = 2.5 if policy == "semantic" else 1.5
            markersize = 12 if policy == "semantic" else 8
            ax.plot(
                bx, ay,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                label=style["label"],
                linewidth=linewidth,
                markersize=markersize,
            )

    ax.set_xlabel("Cache Budget (%)", fontsize=12)
    ax.set_ylabel("Retrieval Accuracy (%)", fontsize=12)
    ax.set_title("Budget Sensitivity: RULER NIAH", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(5, 55)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    pdf_path = Path(output_path).with_suffix(".pdf")
    plt.savefig(pdf_path)
    print(f"Plot saved to: {output_path} and {pdf_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Budget sensitivity curve on RULER NIAH")
    parser.add_argument(
        "--policies", nargs="+",
        default=["full", "streaming", "snapkv", "kvzip", "semantic"],
    )
    parser.add_argument(
        "--budgets", nargs="+", type=float,
        default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
        help="Cache budget fractions to evaluate",
    )
    parser.add_argument("--target-tokens", type=int, default=1800)
    parser.add_argument("--num-depths", type=int, default=5, help="Fewer depths for speed (5 vs 10)")
    parser.add_argument("--num-needles", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/budget_sensitivity/budget_sensitivity.json")
    parser.add_argument("--plot-only", type=str, default=None, help="Skip runs, just plot from existing JSON")
    args = parser.parse_args()

    if args.plot_only:
        plot_budget_sensitivity(args.plot_only, output_path="results/budget_sensitivity/budget_curve.png")
    else:
        depths = list(np.round(np.linspace(0, 1, num=args.num_depths, endpoint=True), 2))

        cfg = ExperimentConfig()
        model, tokenizer = load_model(cfg.model)

        results = run_budget_sensitivity(
            model=model,
            tokenizer=tokenizer,
            policies=args.policies,
            budgets=args.budgets,
            target_tokens=args.target_tokens,
            depths=depths,
            num_needles=args.num_needles,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            output_path=args.output,
        )

        plot_budget_sensitivity(args.output)
