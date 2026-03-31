"""Attention accumulation utilities for KV-cache policies."""

from __future__ import annotations

from typing import Optional

import torch


class AttentionTracker:
    """Track cumulative attention and head entropy over logical cache positions."""

    def __init__(self, num_layers: int, num_kv_heads: int):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.per_head_scores: Optional[torch.Tensor] = None
        self.last_step_scores: Optional[torch.Tensor] = None
        self.num_steps = 0

    def reset(self):
        self.per_head_scores = None
        self.last_step_scores = None
        self.num_steps = 0

    @torch.no_grad()
    def update(
        self,
        attentions: tuple[torch.Tensor, ...],
        kv_positions: Optional[torch.Tensor] = None,
        new_token_position: Optional[int] = None,
    ):
        """
        Update cumulative attention scores from model attention outputs.

        If `kv_positions` is provided, scores are accumulated into those logical
        cache positions instead of assuming a contiguous `[0, ..., kv_len-1]`
        layout. This is required for tiered caches where the model only sees a
        materialized subset of the logical cache.
        """
        if not attentions or attentions[0] is None:
            return

        device = attentions[0].device
        kv_len = attentions[0].shape[-1]
        logical_positions = None
        logical_seq_len = kv_len

        if kv_positions is not None:
            logical_positions = kv_positions.to(device=device, dtype=torch.long)
            if logical_positions.numel() == kv_len - 1 and new_token_position is not None:
                logical_positions = torch.cat(
                    [
                        logical_positions,
                        torch.tensor([new_token_position], device=device, dtype=torch.long),
                    ]
                )
            if logical_positions.numel() != kv_len:
                raise ValueError(
                    f"kv_positions length {logical_positions.numel()} does not match attention kv_len {kv_len}"
                )
            logical_seq_len = int(logical_positions.max().item()) + 1 if logical_positions.numel() > 0 else 0

        if self.per_head_scores is None:
            self.per_head_scores = torch.zeros(
                self.num_layers,
                self.num_kv_heads,
                logical_seq_len,
                device=device,
                dtype=torch.float32,
            )

        if logical_seq_len > self.per_head_scores.shape[-1]:
            pad = torch.zeros(
                self.num_layers,
                self.num_kv_heads,
                logical_seq_len - self.per_head_scores.shape[-1],
                device=device,
                dtype=torch.float32,
            )
            self.per_head_scores = torch.cat([self.per_head_scores, pad], dim=-1)
        if self.last_step_scores is None:
            self.last_step_scores = torch.zeros(logical_seq_len, device=device, dtype=torch.float32)
        elif logical_seq_len > self.last_step_scores.shape[-1]:
            pad = torch.zeros(
                logical_seq_len - self.last_step_scores.shape[-1],
                device=device,
                dtype=torch.float32,
            )
            self.last_step_scores = torch.cat([self.last_step_scores, pad], dim=-1)

        current_step_scores = torch.zeros(logical_seq_len, device=device, dtype=torch.float32)

        for layer_idx, attn in enumerate(attentions):
            attn_last = attn[0, :, -1, :kv_len]
            group_size = attn_last.shape[0] // self.num_kv_heads
            attn_grouped = attn_last.reshape(
                self.num_kv_heads,
                group_size,
                kv_len,
            ).mean(dim=1)

            if logical_positions is None:
                self.per_head_scores[layer_idx, :, :kv_len] += attn_grouped.float()
                current_step_scores[:kv_len] += attn_grouped.float().sum(dim=0)
            else:
                self.per_head_scores[layer_idx].index_add_(1, logical_positions, attn_grouped.float())
                current_step_scores.index_add_(0, logical_positions, attn_grouped.float().sum(dim=0))

        self.last_step_scores = current_step_scores
        self.num_steps += 1

    def get_cumulative_scores(self) -> torch.Tensor:
        """Return cumulative attention per token summed across layers and heads."""
        if self.per_head_scores is None:
            return torch.tensor([])
        return self.per_head_scores.sum(dim=(0, 1))

    def get_head_entropy(self) -> torch.Tensor:
        """Return entropy of per-head attention mass for each token."""
        if self.per_head_scores is None:
            return torch.tensor([])

        head_scores = self.per_head_scores.sum(dim=0)
        total = head_scores.sum(dim=0, keepdim=True).clamp(min=1e-10)
        probs = head_scores / total
        log_probs = torch.log(probs.clamp(min=1e-10))
        entropy = -(probs * log_probs).sum(dim=0)
        return entropy

    def get_last_step_scores(self) -> torch.Tensor:
        """Return attention mass assigned during the most recent decode step."""
        if self.last_step_scores is None:
            return torch.tensor([])
        return self.last_step_scores

    def evict_positions(self, keep_mask: torch.Tensor):
        """Keep tracker state aligned with the logical cache after eviction."""
        if self.per_head_scores is not None:
            self.per_head_scores = self.per_head_scores[:, :, keep_mask]
        if self.last_step_scores is not None:
            self.last_step_scores = self.last_step_scores[keep_mask]
