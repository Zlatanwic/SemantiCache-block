"""
AttentionTracker: 追踪每个 token 的累积 attention score 及其在 head 维度的分布特征。

核心数据:
  - cumulative_scores: shape [seq_len], 每个 token 被后续 token attend 的累积分数
  - head_entropy: shape [seq_len], 每个 token 的 attention score 在 head 维度的熵
    (高熵 = 被多 head 均匀关注 = "语义枢纽", 应保留)
"""

import torch
import numpy as np
from typing import Optional


class AttentionTracker:
    """追踪 attention score 的累积统计信息"""

    def __init__(self, num_layers: int, num_kv_heads: int):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads

        # 每个 token 在每个 layer-head 上被 attend 的累积分数
        # shape: [num_layers, num_kv_heads, seq_len]
        self.per_head_scores: Optional[torch.Tensor] = None

        # 已追踪的 decode 步数
        self.num_steps = 0

    def reset(self):
        self.per_head_scores = None
        self.num_steps = 0

    @torch.no_grad()
    def update(self, attentions: tuple[torch.Tensor, ...]):
        """
        从模型输出的 attention weights 更新累积分数。

        Args:
            attentions: tuple of tensors, 每层一个
                每个 tensor shape: [batch, num_heads, q_len, kv_len]
                注意: Qwen 用 GQA, num_heads=28, num_kv_heads=4
                      所以 attention 里是 28 个 Q head, 需要按 KV group 聚合

        对于 decode 阶段 (q_len=1):
            我们取最新 query token 对所有 KV 位置的 attention 分布,
            累加到 per_head_scores 上。
        """
        if not attentions or attentions[0] is None:
            return

        device = attentions[0].device
        kv_len = attentions[0].shape[-1]

        # 初始化（第一次调用时）
        if self.per_head_scores is None:
            self.per_head_scores = torch.zeros(
                self.num_layers, self.num_kv_heads, kv_len,
                device=device, dtype=torch.float32
            )

        # 如果 kv_len 增长了（新 token 加入），扩展 tensor
        if kv_len > self.per_head_scores.shape[-1]:
            pad = torch.zeros(
                self.num_layers, self.num_kv_heads,
                kv_len - self.per_head_scores.shape[-1],
                device=device, dtype=torch.float32
            )
            self.per_head_scores = torch.cat([self.per_head_scores, pad], dim=-1)

        for layer_idx, attn in enumerate(attentions):
            # attn shape: [1, num_q_heads, q_len, kv_len]
            # 取最后一个 query position 的 attention
            attn_last = attn[0, :, -1, :kv_len]  # [num_q_heads, kv_len]

            # GQA: 将 Q heads 按 KV group 聚合 (平均)
            # Qwen2.5-3B: num_q_heads=16, num_kv_heads=2 => group_size=8
            group_size = attn_last.shape[0] // self.num_kv_heads
            # reshape -> [num_kv_heads, group_size, kv_len] -> mean over group
            attn_grouped = attn_last.reshape(
                self.num_kv_heads, group_size, kv_len
            ).mean(dim=1)  # [num_kv_heads, kv_len]

            self.per_head_scores[layer_idx, :, :kv_len] += attn_grouped.float()

        self.num_steps += 1

    def get_cumulative_scores(self) -> torch.Tensor:
        """
        返回每个 token 的累积 attention score (在所有 layer 和 head 上求和)。
        Returns: shape [seq_len], 值越高表示该 token 越"重要"
        """
        if self.per_head_scores is None:
            return torch.tensor([])
        # sum over layers and heads
        return self.per_head_scores.sum(dim=(0, 1))  # [seq_len]

    def get_head_entropy(self) -> torch.Tensor:
        """
        返回每个 token 在 head 维度的熵。

        直觉: 如果一个 token 被所有 head 均匀关注 -> 高熵 -> 语义枢纽 -> 应保留
              如果只被少数 head 偶尔关注 -> 低熵 -> 可能不重要 -> 可以驱逐

        Returns: shape [seq_len]
        """
        if self.per_head_scores is None:
            return torch.tensor([])

        # 在所有 layer 上聚合, 得到 [num_kv_heads, seq_len]
        head_scores = self.per_head_scores.sum(dim=0)  # [num_kv_heads, seq_len]

        # 归一化为概率分布 (沿 head 维度)
        total = head_scores.sum(dim=0, keepdim=True).clamp(min=1e-10)
        probs = head_scores / total  # [num_kv_heads, seq_len]

        # 计算熵: H = -sum(p * log(p))
        log_probs = torch.log(probs.clamp(min=1e-10))
        entropy = -(probs * log_probs).sum(dim=0)  # [seq_len]

        return entropy

    def evict_positions(self, keep_mask: torch.Tensor):
        """
        驱逐后更新内部状态: 只保留 keep_mask 为 True 的位置。
        Args:
            keep_mask: bool tensor, shape [seq_len]
        """
        if self.per_head_scores is not None:
            self.per_head_scores = self.per_head_scores[:, :, keep_mask]
