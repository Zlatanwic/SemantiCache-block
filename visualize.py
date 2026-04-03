"""
visualize.py: 可视化工具。

功能:
1. Attention heatmap: 标注哪些 token 被保留/驱逐
2. 衰减曲线: 不同 budget 下各策略的质量对比
3. 角色分布: 不同策略保留的 token 角色组成
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans']  # 通用字体
import seaborn as sns
from pathlib import Path


def plot_eviction_heatmap(
    tokens: list[str],
    eviction_scores: np.ndarray,
    keep_mask: np.ndarray,
    role_tags: np.ndarray = None,
    title: str = "KV Cache Eviction Map",
    save_path: str = None,
):
    """
    绘制 token 级别的驱逐热力图。

    每个 token 显示为一个小方块:
    - 颜色: 驱逐分数 (红=高分/被驱逐, 蓝=低分/被保留)
    - 边框: 绿色=被保留, 红色=被驱逐

    Args:
        tokens: token 文本列表
        eviction_scores: 驱逐分数, shape [seq_len]
        keep_mask: bool, True=保留
        role_tags: 可选, 角色标签 (用于标注颜色)
        title: 图标题
        save_path: 保存路径
    """
    seq_len = len(tokens)
    tokens_per_row = 40
    num_rows = (seq_len + tokens_per_row - 1) // tokens_per_row

    fig, axes = plt.subplots(num_rows, 1, figsize=(20, 2 * num_rows))
    if num_rows == 1:
        axes = [axes]

    # 角色颜色映射
    role_colors = {
        0: "#cccccc",  # FILLER
        1: "#a8d8ea",  # ASSISTANT
        2: "#f8e8a0",  # CONTEXT
        3: "#b8e6b8",  # USER_HISTORY
        4: "#4CAF50",  # USER_LATEST
        5: "#FF5722",  # SYSTEM
    }

    for row_idx, ax in enumerate(axes):
        start = row_idx * tokens_per_row
        end = min(start + tokens_per_row, seq_len)
        row_len = end - start

        for i, pos in enumerate(range(start, end)):
            # 背景色: 基于驱逐分数
            score = eviction_scores[pos]
            if np.isinf(score) and score > 0:
                color = "#ff4444"
            elif np.isinf(score) and score < 0:
                color = "#4444ff"
            else:
                # normalize to [0, 1]
                finite_scores = eviction_scores[np.isfinite(eviction_scores)]
                if len(finite_scores) > 0:
                    vmin, vmax = finite_scores.min(), finite_scores.max()
                    if vmax > vmin:
                        norm = (score - vmin) / (vmax - vmin)
                    else:
                        norm = 0.5
                else:
                    norm = 0.5
                # Red (evict) to Blue (keep)
                color = plt.cm.RdYlBu(1 - norm)

            # 边框: 保留=绿, 驱逐=红
            edge_color = "#22cc22" if keep_mask[pos] else "#cc2222"
            edge_width = 2 if not keep_mask[pos] else 1

            rect = plt.Rectangle(
                (i, 0), 1, 1,
                facecolor=color, edgecolor=edge_color, linewidth=edge_width,
            )
            ax.add_patch(rect)

            # token 文本
            token_text = tokens[pos][:6]  # 截断长 token
            ax.text(i + 0.5, 0.5, token_text, ha='center', va='center',
                    fontsize=6, rotation=45)

        ax.set_xlim(0, tokens_per_row)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        if row_idx == 0:
            ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_budget_degradation_curve(
    results_path: str = "results/niah_results.json",
    save_path: str = "results/degradation_curve.png",
):
    """
    绘制不同 budget 下各策略的准确率衰减曲线。

    X轴: cache budget (1.0, 0.5, 0.3, 0.1)
    Y轴: accuracy
    每条线: 一个策略
    """
    with open(results_path) as f:
        results = json.load(f)

    # 按 (policy, budget) 分组计算准确率
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        key = (r["policy"], r["budget"])
        grouped[key].append(r["correct"])

    policies = sorted(set(r["policy"] for r in results))
    budgets = sorted(set(r["budget"] for r in results), reverse=True)

    # 策略样式
    style_map = {
        "full": {"color": "#666666", "marker": "s", "linestyle": "--", "label": "Full (no eviction)"},
        "window": {"color": "#e74c3c", "marker": "o", "linestyle": "-", "label": "Local Window"},
        "streaming": {"color": "#f39c12", "marker": "^", "linestyle": "-", "label": "StreamingLLM"},
        "h2o": {"color": "#3498db", "marker": "D", "linestyle": "-", "label": "H2O"},
        "semantic": {"color": "#2ecc71", "marker": "*", "linestyle": "-", "label": "SieveKV (ours)"},
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for policy in policies:
        accs = []
        valid_budgets = []
        for budget in budgets:
            key = (policy, budget)
            if key in grouped:
                acc = sum(grouped[key]) / len(grouped[key])
                accs.append(acc)
                valid_budgets.append(budget)

        if not accs:
            continue

        style = style_map.get(policy, {"color": "gray", "marker": "x", "linestyle": "-", "label": policy})
        ax.plot(valid_budgets, accs,
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], label=style["label"],
                linewidth=2, markersize=8)

    ax.set_xlabel("Cache Budget Ratio", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("NIAH Accuracy vs. Cache Budget", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # 从大到小

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_role_distribution(
    results: list[dict],
    save_path: str = "results/role_distribution.png",
):
    """
    绘制不同策略保留的 token 中各角色的占比。

    这个需要在实验中额外记录每个策略保留的 token 的角色分布。
    这里提供框架, 具体数据从实验中采集。
    """
    # 示例数据格式
    # results = [
    #   {"policy": "h2o", "budget": 0.5,
    #    "role_counts": {"SYSTEM": 20, "USER_LATEST": 15, "ASSISTANT": 80, ...}},
    #   ...
    # ]

    role_names = ["SYSTEM", "USER_LATEST", "USER_HISTORY", "ASSISTANT", "CONTEXT", "FILLER"]
    role_colors = ["#FF5722", "#4CAF50", "#8BC34A", "#2196F3", "#FFC107", "#9E9E9E"]

    policies = list(set(r["policy"] for r in results))
    x = np.arange(len(policies))
    width = 0.12

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for i, role in enumerate(role_names):
        values = []
        for policy in policies:
            r = next((r for r in results if r["policy"] == policy), None)
            if r and "role_counts" in r:
                total = sum(r["role_counts"].values())
                values.append(r["role_counts"].get(role, 0) / total if total > 0 else 0)
            else:
                values.append(0)

        ax.bar(x + i * width, values, width, label=role, color=role_colors[i])

    ax.set_xlabel("Eviction Policy")
    ax.set_ylabel("Proportion of Retained Tokens")
    ax.set_title("Role Distribution of Retained Tokens by Policy")
    ax.set_xticks(x + width * (len(role_names) - 1) / 2)
    ax.set_xticklabels(policies)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    # 如果有结果文件, 画衰减曲线
    results_file = Path("results/niah_results.json")
    if results_file.exists():
        plot_budget_degradation_curve(str(results_file))
    else:
        print("No results found. Run eval_niah.py --sweep first.")
        print("Generating example plots with dummy data...")

        # 生成示例图
        dummy_results = []
        for policy in ["full", "streaming", "h2o", "semantic"]:
            for budget in [1.0, 0.5, 0.3, 0.1]:
                if policy == "full" and budget != 1.0:
                    continue
                # 模拟不同策略的表现
                base_acc = {"full": 1.0, "streaming": 0.6, "h2o": 0.75, "semantic": 0.85}
                degradation = {"full": 0, "streaming": 0.4, "h2o": 0.25, "semantic": 0.1}
                acc = max(0, base_acc[policy] - degradation[policy] * (1 - budget))
                n = 10
                correct = int(acc * n)
                for i in range(n):
                    dummy_results.append({
                        "policy": policy,
                        "budget": budget,
                        "correct": i < correct,
                    })

        os.makedirs("results", exist_ok=True)
        with open("results/niah_results.json", "w") as f:
            json.dump(dummy_results, f)
        plot_budget_degradation_curve()
        print("Example plot generated with dummy data.")
