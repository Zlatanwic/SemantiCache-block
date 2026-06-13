"""Plot prior-work evidence figures for the SemServe defense deck.

These reuse already-committed HF-testbed results (no GPU needed) to back the
claim that SemServe's *semantic scoring engine* is validated. They are NOT
SemServe system results — keep that distinction in the talk.

Generates:
    fig_prior_budget_curve.png  — RULER NIAH accuracy vs budget, 5 policies
                                  (semantic = ours, vs snapkv/kvzip/streaming/full).
    fig_prior_lowbudget_bars.png — Single-needle NIAH @ 20% budget bar contrast
                                  (semantic = full = 100%, H2O = 47%).

Inputs (committed):
    results/budget_sensitivity/budget_sensitivity.json   (RULER, 5 policies)
    results/main_table/main_table_all.json               (single-needle, full/h2o/semantic)
Run:
    uv run python bench/plot_prior_evidence.py --out-dir results/v3
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
})

# Slide-consistent palette (semantic = green = "ours", matching fig1).
STYLE = {
    "semantic":  {"color": "#2ca02c", "marker": "^", "ls": "-",  "lw": 3.0, "label": "semantic (ours)"},
    "full":      {"color": "#555555", "marker": "o", "ls": "--", "lw": 1.8, "label": "full (no eviction)"},
    "snapkv":    {"color": "#1f77b4", "marker": "s", "ls": "-",  "lw": 2.0, "label": "SnapKV"},
    "kvzip":     {"color": "#ff7f0e", "marker": "D", "ls": "-",  "lw": 2.0, "label": "KVzip"},
    "streaming": {"color": "#d62728", "marker": "v", "ls": "-",  "lw": 2.0, "label": "StreamingLLM"},
    "h2o":       {"color": "#9467bd", "marker": "P", "ls": "-",  "lw": 2.0, "label": "H2O"},
}
ORDER = ["semantic", "full", "snapkv", "h2o", "kvzip", "streaming"]


def accuracy_by_budget(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Return {policy: sorted [(budget, accuracy_fraction), ...]} from a results JSON."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    grouped: dict[tuple[str, float], list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        key = (r["policy"], float(r["budget"]))
        grouped[key][0] += 1 if r.get("correct") else 0
        grouped[key][1] += 1
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (pol, b), (c, n) in grouped.items():
        out[pol].append((b, c / n))
    return {p: sorted(v) for p, v in out.items()}


def plot_budget_curve(curves: dict[str, list[tuple[float, float]]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for pol in [p for p in ORDER if p in curves] + [p for p in curves if p not in ORDER]:
        s = STYLE.get(pol, {"color": "gray", "marker": "x", "ls": ":", "lw": 1.5, "label": pol})
        xs = [b * 100 for b, _ in curves[pol]]
        ys = [a * 100 for _, a in curves[pol]]
        ax.plot(xs, ys, label=s["label"], color=s["color"], marker=s["marker"],
                ls=s["ls"], lw=s["lw"])

    # Beneficial-filtering annotation: semantic@10% above full.
    sem = dict(curves.get("semantic", []))
    full = dict(curves.get("full", []))
    if 0.1 in sem and 0.1 in full and sem[0.1] > full[0.1]:
        ax.annotate(f"semantic @10% = {sem[0.1]*100:.0f}%\n> full {full[0.1]*100:.0f}% (filtering effect)",
                    xy=(10, sem[0.1] * 100), xytext=(16, 92),
                    fontsize=9, color="#2ca02c",
                    arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2))

    ax.set_xlabel("KV cache budget (% of prefill retained)")
    ax.set_ylabel("RULER NIAH accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Content-aware retention vs baselines (RULER NIAH, Qwen2.5-3B)")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def plot_lowbudget_bars(curves: dict[str, list[tuple[float, float]]], budget: float, out: Path) -> None:
    order = [p for p in ["full", "semantic", "h2o"] if p in curves]
    vals, labels, colors = [], [], []
    for p in order:
        acc = dict(curves[p]).get(budget)
        if acc is None:
            continue
        vals.append(acc * 100)
        labels.append(STYLE[p]["label"])
        colors.append(STYLE[p]["color"])

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("NIAH accuracy (%)")
    ax.set_title(f"Single-needle NIAH @ {budget:.0%} budget (~80% KV saved)")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ruler", type=Path, default=Path("results/budget_sensitivity/budget_sensitivity.json"))
    p.add_argument("--single", type=Path, default=Path("results/main_table/main_table_all.json"))
    p.add_argument("--out-dir", type=Path, default=Path("results/v3"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Writing prior-evidence figures:")
    plot_budget_curve(accuracy_by_budget(args.ruler), args.out_dir / "fig_prior_budget_curve.png")
    plot_lowbudget_bars(accuracy_by_budget(args.single), 0.2, args.out_dir / "fig_prior_lowbudget_bars.png")


if __name__ == "__main__":
    main()
