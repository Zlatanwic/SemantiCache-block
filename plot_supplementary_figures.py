"""Generate supplementary figures for the paper.

1. Multi-needle bar chart
2. Context scaling line chart
3. Signal distribution comparison (needle vs haystack)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

POLICY_LABELS = {
    "full": "Full KV",
    "streaming": "StreamingLLM",
    "h2o": "H2O",
    "snapkv": "SnapKV",
    "semantic": "SieveKV",
}

POLICY_COLORS = {
    "full": "#888888",
    "streaming": "#E8734A",
    "h2o": "#F5D76E",
    "snapkv": "#4A90D9",
    "semantic": "#5CB85C",
}


# -----------------------------------------------------------------------
# Figure 1: Multi-needle bar chart
# -----------------------------------------------------------------------

def plot_multi_needle(data_path: str, output_path: str):
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))

    policies_order = ["full", "streaming", "h2o", "snapkv", "semantic"]
    needle_counts = [1, 2, 4]

    # Compute mean scores
    scores = {}
    for policy in policies_order:
        scores[policy] = {}
        for k in needle_counts:
            subset = [r for r in data if r["policy"] == policy and r["num_needles"] == k]
            scores[policy][k] = np.mean([r["score"] for r in subset]) if subset else 0

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(policies_order))
    width = 0.22
    offsets = [-width, 0, width]

    for i, k in enumerate(needle_counts):
        vals = [scores[p][k] for p in policies_order]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=f"k = {k}",
                      color=[POLICY_COLORS[p] for p in policies_order],
                      alpha=0.5 + 0.2 * i,
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in policies_order], rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Multi-Needle NIAH at b = 20%")
    ax.legend(title="Needles", loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 2: Context scaling line chart
# -----------------------------------------------------------------------

def plot_context_scaling(data_paths: list[str], output_path: str):
    all_data = []
    for p in data_paths:
        all_data.extend(json.loads(Path(p).read_text(encoding="utf-8")))

    # Exclude full KV (no eviction, trivially ~0 ms)
    policies_order = ["streaming", "h2o", "snapkv", "semantic"]
    target_tokens = sorted(set(r["target_tokens"] for r in all_data))

    POLICY_LINESTYLES = {
        "streaming": "--",
        "h2o": "-.",
        "snapkv": ":",
        "semantic": "-",
    }

    fig, ax = plt.subplots(figsize=(6, 4))

    for policy in policies_order:
        latencies = []
        for t in target_tokens:
            subset = [r for r in all_data if r["policy"] == policy and r["target_tokens"] == t]
            latencies.append(np.mean([r["eviction_time_per_step_ms"] for r in subset]) if subset else 0)
        ax.plot(target_tokens, latencies,
                marker="o", markersize=6,
                color=POLICY_COLORS[policy],
                linestyle=POLICY_LINESTYLES[policy],
                label=POLICY_LABELS[policy],
                linewidth=2)
        # Annotate last point
        ax.annotate(f"{latencies[-1]:.1f}",
                    (target_tokens[-1], latencies[-1]),
                    textcoords="offset points", xytext=(8, -2),
                    fontsize=8, color=POLICY_COLORS[policy])

    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Eviction latency per step (ms)")
    ax.set_xticks(target_tokens)
    ax.set_xticklabels([str(t) for t in target_tokens], rotation=45, ha='right')
    ax.set_title("Eviction Overhead Scaling at b = 20%")
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 3: Signal distribution (needle vs haystack)
# -----------------------------------------------------------------------

def plot_signal_distribution(data_path: str, output_path: str):
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    tokens = data["tokens"]

    needle_tokens = [t for t in tokens if t["is_needle"]]
    haystack_tokens = [t for t in tokens if not t["is_needle"] and not t["is_query"]]

    signals = ["query_relevance", "factual", "density"]
    signal_labels = ["Query Relevance", "Factual Likelihood", "Info Density"]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))

    for ax, signal, label in zip(axes, signals, signal_labels):
        needle_vals = [t[signal] for t in needle_tokens]
        haystack_vals = [t[signal] for t in haystack_tokens]

        # Box plot
        bp = ax.boxplot(
            [haystack_vals, needle_vals],
            labels=["Haystack\n(n=3653)", "Needle\n(n=30)"],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=1.5),
        )
        bp["boxes"][0].set_facecolor("#D0D0D0")
        bp["boxes"][1].set_facecolor("#5CB85C")

        needle_mean = np.mean(needle_vals)
        haystack_mean = np.mean(haystack_vals)
        ratio = needle_mean / max(haystack_mean, 1e-6)

        ax.set_title(f"{label}\n({ratio:.1f}× ratio)")
        ax.set_ylabel("Score" if ax == axes[0] else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Signal Scores: Needle vs. Haystack Tokens", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    # Figure 1: Multi-needle
    mn_path = "results/multi_needle/multi_needle_qwen.json"
    if Path(mn_path).exists():
        plot_multi_needle(mn_path, "multi_needle_bar.pdf")
        plot_multi_needle(mn_path, "multi_needle_bar.png")

    # Figure 2: Context scaling
    scaling_paths = []
    for p in [
        "results/scaling/context_scaling_qwen_b20.json",
        "results/scaling/context_scaling_16k.json",
        "results/scaling/context_scaling_32k.json",
    ]:
        if Path(p).exists():
            scaling_paths.append(p)
    if scaling_paths:
        plot_context_scaling(scaling_paths, "context_scaling.pdf")
        plot_context_scaling(scaling_paths, "context_scaling.png")

    # Figure 3: Signal distribution
    cs_path = "results/case_study/token_retention.json"
    if Path(cs_path).exists():
        plot_signal_distribution(cs_path, "signal_distribution.pdf")
        plot_signal_distribution(cs_path, "signal_distribution.png")

    print("\nDone. Generated figures in current directory.")
