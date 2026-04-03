#!/usr/bin/env python3
"""Depth heatmap: accuracy by (policy, depth) for each budget level."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

# ── Load data ──────────────────────────────────────────────────────────
with open("results/ruler_niah/ruler_niah_5policy.json") as f:
    data_5p = json.load(f)
with open("results/ruler_niah/ruler_niah_new_baselines_v2.json") as f:
    data_nb = json.load(f)

# Merge: 5policy has full/h2o/streaming/snapkv/semantic; new_baselines has kvzip
# For streaming/snapkv, 5policy is the canonical source (new_baselines may duplicate)
all_data = data_5p + [r for r in data_nb if r["policy"] == "kvzip"]

# ── Aggregate: mean accuracy per (policy, budget, depth) ──────────────
acc = defaultdict(list)
for r in all_data:
    key = (r["policy"], r["budget"], r["depth"])
    acc[key].append(100.0 if r["correct"] else 0.0)

mean_acc = {k: np.mean(v) for k, v in acc.items()}

# ── Config ─────────────────────────────────────────────────────────────
policies = ["full", "streaming", "h2o", "snapkv", "kvzip", "semantic"]
policy_labels = [
    "Full KV",
    "StreamingLLM",
    "H2O",
    "SnapKV",
    "KVzip",
    "SieveKV (ours)",
]
budgets = [0.5, 0.3, 0.2]
budget_labels = ["b = 50%", "b = 30%", "b = 20%"]
depths = sorted({r["depth"] for r in all_data})
depth_labels = [f"{d:.0%}" for d in depths]

# ── Plot ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
cmap = plt.cm.RdYlGn
norm = mcolors.Normalize(vmin=0, vmax=100)

for ax, budget, blabel in zip(axes, budgets, budget_labels):
    matrix = np.zeros((len(policies), len(depths)))
    for i, pol in enumerate(policies):
        for j, d in enumerate(depths):
            matrix[i, j] = mean_acc.get((pol, budget, d), 0.0)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    for i in range(len(policies)):
        for j in range(len(depths)):
            val = matrix[i, j]
            color = "white" if val < 30 or val > 85 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depth_labels, fontsize=13, rotation=45)
    ax.set_xlabel("Needle Depth", fontsize=14)
    ax.set_title(blabel, fontsize=15, fontweight="bold")

    if ax == axes[0]:
        ax.set_yticks(range(len(policies)))
        ax.set_yticklabels(policy_labels, fontsize=14)

fig.suptitle("RULER NIAH Retrieval Accuracy by Needle Depth",
             fontsize=16, fontweight="bold", y=1.0)

fig.subplots_adjust(left=0.13, right=0.88, top=0.85, bottom=0.2,wspace=0.15, hspace=0.1)

# 单独放 colorbar
cax = fig.add_axes([0.905, 0.18, 0.012, 0.58])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Accuracy (%)", fontsize=14, labelpad=8)
cbar.ax.tick_params(labelsize=13)

plt.savefig("depth_heatmap.png", dpi=200)
plt.savefig("depth_heatmap.pdf")

# ── Print summary table for verification ───────────────────────────────
print(f"\n{'Policy':<20} {'Budget':>6}  " +
      "  ".join(f"d={d:.0%}" for d in depths))
for pol, plabel in zip(policies, policy_labels):
    for budget in budgets:
        row = [mean_acc.get((pol, budget, d), 0.0) for d in depths]
        print(f"{plabel:<20} {budget:>5.0%}  " +
              "  ".join(f"{v:5.0f}" for v in row))
