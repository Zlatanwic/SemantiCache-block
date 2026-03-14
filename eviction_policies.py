"""Eviction policy implementations for KV-cache pruning."""

from abc import ABC, abstractmethod
from typing import Optional

import torch

from attention_tracker import AttentionTracker
from semantic_analyzer import RoleTag, SemanticAnalyzer


class EvictionPolicy(ABC):
    """Base class for all cache-eviction policies."""

    @abstractmethod
    def compute_eviction_scores(self, seq_len: int, **kwargs) -> torch.Tensor:
        """
        Return per-token eviction scores.

        Higher scores mean a token is more likely to be evicted.
        `torch.inf` means evict first. `-torch.inf` means never evict.
        """

    def select_eviction_indices(self, scores: torch.Tensor, num_to_evict: int) -> torch.Tensor:
        """Select the token indices with the largest eviction scores."""
        _, indices = scores.topk(num_to_evict, largest=True)
        return indices.sort().values

    def select_keep_indices(self, scores: torch.Tensor, budget: int) -> torch.Tensor:
        """Select the token indices to keep under a fixed budget."""
        _, keep_indices = scores.topk(budget, largest=False)
        return keep_indices.sort().values


class FullCachePolicy(EvictionPolicy):
    """Upper-bound baseline: never evict any token."""

    def compute_eviction_scores(self, seq_len: int, **kwargs) -> torch.Tensor:
        return torch.full((seq_len,), -torch.inf)


class LocalWindowPolicy(EvictionPolicy):
    """Keep only the most recent `window_size` tokens."""

    def __init__(self, window_size: int):
        self.window_size = window_size

    def compute_eviction_scores(self, seq_len: int, **kwargs) -> torch.Tensor:
        scores = torch.zeros(seq_len)
        if seq_len > self.window_size:
            scores[: seq_len - self.window_size] = torch.inf
        return scores


class StreamingLLMPolicy(EvictionPolicy):
    """Keep sink tokens plus a recent sliding window."""

    def __init__(self, sink_tokens: int = 4, window_size: int = 512):
        self.sink_tokens = sink_tokens
        self.window_size = window_size

    def compute_eviction_scores(self, seq_len: int, **kwargs) -> torch.Tensor:
        scores = torch.zeros(seq_len)
        keep_count = self.sink_tokens + self.window_size
        if seq_len <= keep_count:
            return scores

        scores[: self.sink_tokens] = -torch.inf
        scores[seq_len - self.window_size :] = -torch.inf
        scores[self.sink_tokens : seq_len - self.window_size] = torch.inf
        return scores


class H2OPolicy(EvictionPolicy):
    """Heavy-Hitter Oracle baseline based on cumulative attention scores."""

    def __init__(self, tracker: AttentionTracker):
        self.tracker = tracker

    def compute_eviction_scores(self, seq_len: int, **kwargs) -> torch.Tensor:
        cumulative = self.tracker.get_cumulative_scores()
        if len(cumulative) == 0 or len(cumulative) < seq_len:
            return torch.zeros(seq_len)

        cumulative = cumulative[:seq_len]
        max_score = cumulative.max().clamp(min=1e-10)
        return 1.0 - (cumulative / max_score)


class SemantiCachePolicy(EvictionPolicy):
    """
    SemantiCache: combine attention, information density, and head-entropy signals.

    On top of semantic role protection, this policy keeps a recent decode window
    to avoid collapsing generation quality when the cache budget is capped.
    """

    def __init__(
        self,
        tracker: AttentionTracker,
        analyzer: SemanticAnalyzer,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
        query_weight: float = 0.25,
        factual_weight: float = 0.2,
        pin_system: bool = True,
        pin_latest_user: bool = True,
        recent_window_size: int = 64,
        block_size: int = 16,
        latest_user_tail_tokens: int = 64,
    ):
        self.tracker = tracker
        self.analyzer = analyzer
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.query_weight = query_weight
        self.factual_weight = max(factual_weight, 0.35)
        self.pin_system = pin_system
        self.pin_latest_user = pin_latest_user
        self.recent_window_size = recent_window_size
        self.block_size = min(block_size, 8)
        self.latest_user_tail_tokens = latest_user_tail_tokens

        self.role_tags: Optional[torch.Tensor] = None
        self.info_density: Optional[torch.Tensor] = None
        self.query_relevance: Optional[torch.Tensor] = None
        self.factual_bonus: Optional[torch.Tensor] = None
        self.pinned_mask: Optional[torch.Tensor] = None

    @staticmethod
    def _inverse_normalize(values: Optional[torch.Tensor], seq_len: int) -> Optional[torch.Tensor]:
        """Map a signal to eviction pressure where larger source values are safer to keep."""
        if values is None or len(values) < seq_len:
            return None

        window = values[:seq_len]
        scale = window.max().clamp(min=1e-10)
        return 1.0 - (window / scale)

    def setup_semantic_signals(self, input_ids: torch.Tensor, latest_query_text: str = "") -> None:
        """Compute semantic signals once after prefill."""
        self.role_tags = self.analyzer.compute_role_tags(input_ids)
        self.info_density = self.analyzer.compute_info_density(input_ids)
        self.query_relevance = self.analyzer.compute_query_relevance(input_ids, latest_query_text)
        self.factual_bonus = self.analyzer.compute_factual_bonus(input_ids)
        self.pinned_mask = self.analyzer.get_pinned_mask(
            self.role_tags,
            self.pin_system,
            self.pin_latest_user,
            latest_user_tail_tokens=self.latest_user_tail_tokens,
        )

    def extend_signals(self, num_new_tokens: int = 1) -> None:
        """Append semantic defaults for newly generated assistant tokens."""
        if self.role_tags is not None:
            new_roles = torch.full((num_new_tokens,), RoleTag.ASSISTANT, dtype=torch.long)
            self.role_tags = torch.cat([self.role_tags, new_roles])

        if self.info_density is not None:
            new_density = torch.full((num_new_tokens,), 0.5, dtype=torch.float32)
            self.info_density = torch.cat([self.info_density, new_density])

        if self.query_relevance is not None:
            new_query = torch.zeros(num_new_tokens, dtype=torch.float32)
            self.query_relevance = torch.cat([self.query_relevance, new_query])

        if self.factual_bonus is not None:
            new_factual = torch.zeros(num_new_tokens, dtype=torch.float32)
            self.factual_bonus = torch.cat([self.factual_bonus, new_factual])

        if self.pinned_mask is not None:
            new_pinned = torch.zeros(num_new_tokens, dtype=torch.bool)
            self.pinned_mask = torch.cat([self.pinned_mask, new_pinned])

    def compute_eviction_scores(self, seq_len: int, **kwargs) -> torch.Tensor:
        scores = torch.zeros(seq_len, dtype=torch.float32)

        attn_pressure = self._inverse_normalize(self.tracker.get_cumulative_scores(), seq_len)
        if attn_pressure is not None:
            scores += self.alpha * attn_pressure.cpu()

        density_pressure = self._inverse_normalize(self.info_density, seq_len)
        if density_pressure is not None:
            scores += self.beta * density_pressure

        entropy_pressure = self._inverse_normalize(self.tracker.get_head_entropy(), seq_len)
        if entropy_pressure is not None:
            scores += self.gamma * entropy_pressure.cpu()

        query_pressure = self._inverse_normalize(self.query_relevance, seq_len)
        if query_pressure is not None:
            scores += self.query_weight * query_pressure

        factual_pressure = self._inverse_normalize(self.factual_bonus, seq_len)
        if factual_pressure is not None:
            scores += self.factual_weight * factual_pressure

        if self.pinned_mask is not None and len(self.pinned_mask) >= seq_len:
            scores[self.pinned_mask[:seq_len]] = -torch.inf

        if self.recent_window_size > 0:
            recent_start = max(0, seq_len - self.recent_window_size)
            scores[recent_start:] = -torch.inf

        return scores

    def select_keep_indices(self, scores: torch.Tensor, budget: int) -> torch.Tensor:
        """Keep protected tokens plus the best-scoring contiguous blocks."""
        seq_len = scores.shape[0]
        if budget >= seq_len:
            return torch.arange(seq_len, device=scores.device)

        protected_mask = torch.zeros(seq_len, dtype=torch.bool, device=scores.device)
        if self.pinned_mask is not None and len(self.pinned_mask) >= seq_len:
            protected_mask |= self.pinned_mask[:seq_len].to(device=scores.device)
        if self.recent_window_size > 0:
            protected_mask[max(0, seq_len - self.recent_window_size) :] = True

        protected_indices = torch.nonzero(protected_mask, as_tuple=False).flatten()
        if protected_indices.numel() >= budget:
            return protected_indices[-budget:].sort().values

        keep_indices = protected_indices.tolist()
        remaining_budget = budget - len(keep_indices)

        candidate_blocks: list[tuple[float, list[int]]] = []
        block_size = max(1, self.block_size)
        for block_start in range(0, seq_len, block_size):
            block_end = min(seq_len, block_start + block_size)
            block_positions = torch.arange(block_start, block_end, device=scores.device)
            candidate_mask = ~protected_mask[block_start:block_end]
            candidate_positions = block_positions[candidate_mask]
            if candidate_positions.numel() == 0:
                continue

            block_score = scores[candidate_positions].mean().item()
            candidate_blocks.append((block_score, candidate_positions.tolist()))

        candidate_blocks.sort(key=lambda item: item[0])

        for _, block_positions in candidate_blocks:
            if remaining_budget <= 0:
                break

            if len(block_positions) <= remaining_budget:
                keep_indices.extend(block_positions)
                remaining_budget -= len(block_positions)
            else:
                keep_indices.extend(block_positions[-remaining_budget:])
                remaining_budget = 0

        if remaining_budget > 0:
            all_indices = torch.arange(seq_len, device=scores.device)
            already_kept = torch.zeros(seq_len, dtype=torch.bool, device=scores.device)
            already_kept[torch.tensor(keep_indices, device=scores.device, dtype=torch.long)] = True
            fallback_candidates = all_indices[~already_kept]
            fallback_scores = scores[fallback_candidates]
            _, extra_order = fallback_scores.topk(min(remaining_budget, fallback_candidates.numel()), largest=False)
            keep_indices.extend(fallback_candidates[extra_order].tolist())

        keep_tensor = torch.tensor(keep_indices, device=scores.device, dtype=torch.long).unique(sorted=True)
        if keep_tensor.numel() > budget:
            keep_tensor = keep_tensor[-budget:]
        return keep_tensor.sort().values

    def evict_positions(self, keep_mask: torch.Tensor) -> None:
        """Keep semantic signal tensors aligned with the pruned cache."""
        if self.role_tags is not None:
            self.role_tags = self.role_tags[keep_mask]
        if self.info_density is not None:
            self.info_density = self.info_density[keep_mask]
        if self.query_relevance is not None:
            self.query_relevance = self.query_relevance[keep_mask]
        if self.factual_bonus is not None:
            self.factual_bonus = self.factual_bonus[keep_mask]
        if self.pinned_mask is not None:
            self.pinned_mask = self.pinned_mask[keep_mask]
