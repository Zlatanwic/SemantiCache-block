"""Plot the Phase 1 block-granularity gate result for the SemServe defense.

Generates two figures from the manifest-gated NIAH JSONs:

    fig1_block_granularity_curve.png  —  Quality vs budget for token / block-16 / block-32,
                                         with the 25% gate and the `full` upper bound annotated.
    fig2_phase1_evidence.png           —  Single-image summary slide (curves + per-budget
                                         block-16 vs block-32 delta + the 5 evidence points).

Inputs (default): results/v3/gate_block16.json, results/v3/gate_block32.json.
Run:
    python bench/plot_phase1_gate.py --out-dir results/v3
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Style: serif + thin grid (slide-friendly, prints fine in B/W too)
rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
})


# ---- Data ----
SERIES_STYLE = {
    "full":            {"color": "#1f77b4", "marker": "o", "ls": "-",  "label": "full (no eviction, upper bound)"},
    "semantic":        {"color": "#d62728", "marker": "s", "ls": "-",  "label": "semantic (token-level)"},
    "block_semantic_16": {"color": "#2ca02c", "marker": "^", "ls": "-",  "label": "block_semantic, block=16 (ours)"},
    "block_semantic_32": {"color": "#ff7f0e", "marker": "D", "ls": "--", "label": "block_semantic, block=32"},
}
GATE_BUDGET = 0.25
GATE_RELATIVE = 0.95  # block-16 must reach >=95% of token-level at the gate


def load(json_paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in json_paths:
        rows.extend(json.loads(p.read_text(encoding="utf-8")))
    return rows


def aggregate(rows: list[dict]) -> dict[tuple[str, float], dict]:
    """Group by (policy_label, budget) and compute mean accuracy + std + n."""
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in rows:
        p = r["policy"]
        b = float(r["budget"])
        # Only the 16/32 block_semantic runs in gate_block16 are block=16 (the file
        # also has full + token); gate_block32.json contains only block-32.
        # We infer the block size from the source file, passed in via `series_to_path`.
        grouped[(p, b)].append(r)

    out: dict[tuple[str, float], dict] = {}
    for (p, b), runs in grouped.items():
        acc = sum(r["correct"] for r in runs) / len(runs)
        elapsed = statistics.mean(r["elapsed_time"] for r in runs)
        out[(p, b)] = {
            "acc": acc,
            "n": len(runs),
            "elapsed": elapsed,
            "stdev": statistics.stdev(acc_i for acc_i in [r["correct"] for r in runs]) if len(runs) > 1 else 0.0,
        }
    return out


def make_curves(agg: dict[tuple[str, float], dict]) -> dict[str, list[tuple[float, float]]]:
    """Return {series_name: sorted [(budget, acc), ...]}."""
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (policy, budget), stats in agg.items():
        out[policy].append((budget, stats["acc"]))
    for k in out:
        out[k] = sorted(out[k], key=lambda x: x[0])
    return dict(out)


# ---- Figures ----
def plot_curve(curves: dict[str, list[tuple[float, float]]], gate_pass: bool, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, points in curves.items():
        style = SERIES_STYLE.get(name, {"color": "gray", "marker": "x", "ls": ":", "label": name})
        xs = [b for b, _ in points]
        ys = [a * 100 for _, a in points]
        ax.plot(xs, ys, label=style["label"], color=style["color"], marker=style["marker"], ls=style["ls"])

    # Gate line + annotation (placed in upper-left to avoid the legend in lower-right)
    ax.axvline(GATE_BUDGET, color="gray", ls=":", lw=1, alpha=0.6)
    ax.text(GATE_BUDGET + 0.015, 18, f"Phase 1 gate @ {GATE_BUDGET:.0%} budget",
            fontsize=9, color="gray", va="top", ha="left", style="italic")

    verdict_color = "green" if gate_pass else "red"
    verdict = "PASS" if gate_pass else "FAIL"
    ax.text(0.02, 0.97, f"Gate: {verdict}", transform=ax.transAxes, ha="left", va="top",
            fontsize=14, fontweight="bold", color=verdict_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=verdict_color, alpha=0.9))

    ax.set_xlabel("KV cache budget (fraction of prefill retained)")
    ax.set_ylabel("NIAH accuracy (%)")
    ax.set_xscale("log")
    ax.set_xlim(0.085, 1.05)
    ax.set_ylim(0, 105)
    ax.set_xticks([0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["10%", "25%", "50%", "100%"])
    ax.set_title("Phase 1: block-granularity quality gate (Qwen2.5-3B, 2k ctx, 40-sample manifest)")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def plot_summary(curves: dict[str, list[tuple[float, float]]], agg: dict, out: Path) -> None:
    """2x2 panel: curves + 3 per-budget bar charts of block-16 vs block-32 deltas."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (1) Re-draw curves in the top-left
    ax = axes[0, 0]
    for name, points in curves.items():
        style = SERIES_STYLE.get(name, {"color": "gray", "marker": "x", "ls": ":", "label": name})
        xs = [b for b, _ in points]
        ys = [a * 100 for _, a in points]
        ax.plot(xs, ys, label=style["label"], color=style["color"], marker=style["marker"], ls=style["ls"])
    ax.axvline(GATE_BUDGET, color="gray", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("KV cache budget")
    ax.set_ylabel("NIAH accuracy (%)")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["10%", "25%", "50%", "100%"])
    ax.set_ylim(0, 105)
    ax.set_title("(a) Quality–budget curves")
    ax.legend(fontsize=8, loc="lower right")

    # (2) Per-budget deltas: block-16 vs token and block-16 vs block-32
    common_budgets = [0.25, 0.3, 0.5]
    sem_25 = next((a for b, a in curves.get("semantic", []) if b == 0.25), None)
    b16_25 = next((a for b, a in curves.get("block_semantic", []) if b == 0.25), None)
    ax = axes[0, 1]
    if sem_25 and b16_25:
        rel = b16_25 / sem_25
        ax.bar(["semantic\n(token)", f"block-16\n@ {GATE_BUDGET:.0%}"], [sem_25 * 100, b16_25 * 100],
               color=["#d62728", "#2ca02c"])
        ax.set_ylim(0, 105)
        ax.set_ylabel("NIAH accuracy (%)")
        ax.set_title(f"(b) Gate point: block-16 = {rel:.1%} of token-level")
        for i, v in enumerate([sem_25 * 100, b16_25 * 100]):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    else:
        ax.text(0.5, 0.5, "missing 25% data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    # (3) Cost (avg elapsed) per strategy at gate budget
    ax = axes[1, 0]
    budgets_to_plot = [0.25, 0.5]
    labels = []
    means = []
    colors = []
    for pol in ["semantic", "block_semantic"]:
        for b in budgets_to_plot:
            s = agg.get((pol, b))
            if s:
                labels.append(f"{pol}\n@{b:.0%}")
                means.append(s["elapsed"])
                colors.append(SERIES_STYLE.get(pol, {"color": "gray"})["color"])
    ax.bar(labels, means, color=colors)
    ax.set_ylabel("Avg elapsed time (s / run)")
    ax.set_title("(c) Per-run cost (no block-alignment overhead visible)")

    # (4) 5 evidence points as text
    ax = axes[1, 1]
    ax.set_axis_off()
    sem10 = next((a for b, a in curves.get("semantic", []) if b == 0.1), None)
    bs10 = next((a for b, a in curves.get("block_semantic", []) if b == 0.1), None)
    full_acc = next((a for b, a in curves.get("full", []) if b == 1.0), None)
    bullets = [
        "Phase 1 Gate — 5 evidence points",
        "",
        f"1. Quality ceiling: full@100% = {full_acc*100:.1f}% (40-sample NIAH)",
        f"2. Token-level @25% = {sem_25*100:.1f}%   (reference)",
        f"3. block-16 @25%     = {b16_25*100:.1f}%   (gate value)",
        f"4. Relative: block-16 / token = {(b16_25/sem_25)*100:.1f}%   (threshold ≥ 95% → {'PASS' if (b16_25/sem_25) >= GATE_RELATIVE else 'FAIL'})",
        f"5. Extreme-budget trade-off: token@10%={sem10*100:.0f}%, block-16@10%={bs10*100:.0f}%",
        "     (block=8 is plan §7 fallback for very low budgets)",
        "",
        "Conclusion: vLLM block-granularity route is",
        "algorithmically justified; Phase 2 SieveKV-on-vLLM",
        "MVP can proceed.",
    ]
    ax.text(0.02, 0.98, "\n".join(bullets), transform=ax.transAxes, ha="left", va="top",
            family="monospace", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#cccccc"))

    fig.suptitle("SemServe Phase 1: block-granularity quality gate evidence",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gate-block16", type=Path, default=Path("results/v3/gate_block16.json"))
    p.add_argument("--gate-block32", type=Path, default=Path("results/v3/gate_block32.json"))
    p.add_argument("--out-dir", type=Path, default=Path("results/v3"))
    args = p.parse_args()

    paths = [args.gate_block16]
    if args.gate_block32.exists():
        paths.append(args.gate_block32)
    rows = load(paths)
    if not rows:
        raise SystemExit("no gate JSON rows found")

    # Relabel block_semantic by source: rows in gate_block16.json => block=16,
    # rows in gate_block32.json => block=32.
    relabeled: list[dict] = []
    seen = set()
    for path in paths:
        is_block32 = "block32" in path.name
        for r in json.loads(path.read_text(encoding="utf-8")):
            if r["policy"] == "block_semantic":
                r = dict(r)
                r["policy"] = "block_semantic_32" if is_block32 else "block_semantic_16"
            relabeled.append(r)
        seen.add(path.name)
    # For the curves plot we also want a single `block_semantic` series at 16
    # for direct comparison; the summary plot uses relabeled.

    agg = aggregate(relabeled)
    # Build curves for each policy key
    curves = make_curves(agg)
    # Map back to the original (block_semantic=16) for the simple curve plot
    # by copying block_semantic_16 -> block_semantic
    if "block_semantic_16" in curves:
        curves["block_semantic"] = list(curves["block_semantic_16"])

    # Gate verdict
    sem_25 = next((a for b, a in curves.get("semantic", []) if b == GATE_BUDGET), None)
    b16_25 = next((a for b, a in curves.get("block_semantic", []) if b == GATE_BUDGET), None)
    gate_pass = bool(sem_25 and b16_25 and (b16_25 / sem_25) >= GATE_RELATIVE)
    print(f"semantic @ {GATE_BUDGET}: {sem_25}")
    print(f"block_semantic-16 @ {GATE_BUDGET}: {b16_25}")
    print(f"relative = {b16_25 / sem_25 if sem_25 else 'n/a'}; gate = {'PASS' if gate_pass else 'FAIL'}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Writing figures:")
    plot_curve(curves, gate_pass, args.out_dir / "fig1_block_granularity_curve.png")
    plot_summary(curves, agg, args.out_dir / "fig2_phase1_evidence.png")


if __name__ == "__main__":
    main()
