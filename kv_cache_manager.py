"""KV cache management utilities, including top-k warm promotion."""

from __future__ import annotations

from typing import Optional

import torch
from transformers.cache_utils import DynamicCache

from attention_tracker import AttentionTracker
from config import CacheConfig
from eviction_policies import EvictionPolicy, SemantiCachePolicy, TieredSemantiCachePolicy
from quantized_tier_cache import QuantizedWarmTier


def get_cache_layer_count(past_key_values: DynamicCache) -> int:
    """Return the number of layers currently present in the cache."""
    if hasattr(past_key_values, "layers"):
        return len(past_key_values.layers)
    if hasattr(past_key_values, "key_cache"):
        return len(past_key_values.key_cache)
    return 0


def get_cache_seq_len(past_key_values: DynamicCache) -> int:
    """Return the cached sequence length from the first layer."""
    if hasattr(past_key_values, "layers"):
        if len(past_key_values.layers) == 0:
            return 0
        first_layer = past_key_values.layers[0]
        if hasattr(first_layer, "get_seq_length"):
            return first_layer.get_seq_length()
        if getattr(first_layer, "keys", None) is None or first_layer.keys.numel() == 0:
            return 0
        return first_layer.keys.shape[-2]

    if hasattr(past_key_values, "key_cache"):
        if len(past_key_values.key_cache) == 0:
            return 0
        return past_key_values.key_cache[0].shape[2]

    return 0


def get_first_layer_cache_shape(past_key_values: DynamicCache):
    """Return the shape of the first-layer key cache."""
    if hasattr(past_key_values, "layers"):
        if len(past_key_values.layers) == 0:
            return None
        first_layer = past_key_values.layers[0]
        keys = getattr(first_layer, "keys", None)
        if keys is None or keys.numel() == 0:
            return None
        return keys.shape

    if hasattr(past_key_values, "key_cache"):
        if len(past_key_values.key_cache) == 0:
            return None
        return past_key_values.key_cache[0].shape

    return None


def prune_dynamic_cache(past_key_values: DynamicCache, keep_indices: torch.Tensor) -> DynamicCache:
    """Prune a DynamicCache in place to only the selected sequence positions."""
    if hasattr(past_key_values, "layers"):
        if len(past_key_values.layers) == 0:
            return past_key_values

        first_layer = past_key_values.layers[0]
        keys = getattr(first_layer, "keys", None)
        if keys is None or keys.numel() == 0:
            return past_key_values

        keep_idx_device = keep_indices.to(keys.device)
        for layer in past_key_values.layers:
            if getattr(layer, "keys", None) is None or layer.keys.numel() == 0:
                continue
            layer.keys = torch.index_select(layer.keys, 2, keep_idx_device)
            layer.values = torch.index_select(layer.values, 2, keep_idx_device)
        return past_key_values

    if hasattr(past_key_values, "key_cache"):
        if len(past_key_values.key_cache) == 0:
            return past_key_values
        keep_idx_device = keep_indices.to(past_key_values.key_cache[0].device)
        for layer_idx in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[layer_idx] = torch.index_select(
                past_key_values.key_cache[layer_idx], 2, keep_idx_device
            )
            past_key_values.value_cache[layer_idx] = torch.index_select(
                past_key_values.value_cache[layer_idx], 2, keep_idx_device
            )
        return past_key_values

    raise TypeError(f"Unsupported cache structure: {type(past_key_values)!r}")


def build_dynamic_cache_from_ddp(ddp_cache_data: list[tuple[torch.Tensor, torch.Tensor]], model_config) -> DynamicCache:
    """Create a DynamicCache from per-layer key/value tensors."""
    return DynamicCache(ddp_cache_data=ddp_cache_data, config=model_config)


def append_last_token_from_output(hot_cache: DynamicCache, decode_output_cache: DynamicCache) -> DynamicCache:
    """Append the newly generated token from the model output back into the hot cache."""
    ddp_cache_data: list[tuple[torch.Tensor, torch.Tensor]] = []
    for hot_layer, output_layer in zip(hot_cache.layers, decode_output_cache.layers):
        new_key = output_layer.keys[:, :, -1:, :]
        new_value = output_layer.values[:, :, -1:, :]
        ddp_cache_data.append(
            (
                torch.cat([hot_layer.keys, new_key], dim=2),
                torch.cat([hot_layer.values, new_value], dim=2),
            )
        )
    return build_dynamic_cache_from_ddp(ddp_cache_data, decode_output_cache.layers[0].__dict__.get("config", None))


class KVCacheManager:
    """Manage KV cache pruning and tiered semantic cache promotion."""

    def __init__(
        self,
        config: CacheConfig,
        policy: EvictionPolicy,
        tracker: AttentionTracker,
        num_layers: int,
        model_config,
    ):
        self.config = config
        self.policy = policy
        self.tracker = tracker
        self.num_layers = num_layers
        self.model_config = model_config

        self.initial_seq_len: Optional[int] = None
        self.current_cache_len: int = 0
        self.total_evicted = 0
        self.eviction_steps = 0

        self.hot_cache_len = 0
        self.hot_positions = torch.empty(0, dtype=torch.long)
        self.last_prepared_positions = torch.empty(0, dtype=torch.long)
        self.last_promoted_positions = torch.empty(0, dtype=torch.long)
        self.origin_positions = torch.empty(0, dtype=torch.long)
        self.last_prepared_origin_positions = torch.empty(0, dtype=torch.long)
        self.last_promoted_origin_positions = torch.empty(0, dtype=torch.long)
        self.next_origin_position = 0
        self.warm_tier = QuantizedWarmTier(storage_device=self._warm_storage_device)
        self.materialization_steps = 0
        self.promotion_steps = 0
        self.last_promoted_warm_count = 0
        self.peak_promoted_warm_count = 0

        self.last_cold_cache_len = 0
        self.peak_hot_cache_len = 0
        self.peak_warm_cache_len = 0
        self.peak_cold_cache_len = 0

    @property
    def _warm_storage_device(self) -> str:
        return "cpu" if self.config.semantic_warm_device == "cpu" else "cpu"

    @property
    def uses_tiered_cache(self) -> bool:
        return isinstance(self.policy, TieredSemantiCachePolicy)

    def set_initial_seq_len(self, seq_len: int):
        """Record the prefill length that defines later cache budgets."""
        self.initial_seq_len = seq_len
        self.current_cache_len = seq_len
        self.hot_cache_len = seq_len
        self.hot_positions = torch.arange(seq_len, dtype=torch.long)
        self.last_prepared_positions = self.hot_positions.clone()
        self.origin_positions = torch.arange(seq_len, dtype=torch.long)
        self.last_prepared_origin_positions = self.origin_positions.clone()
        self.next_origin_position = seq_len
        self.last_cold_cache_len = 0
        self.peak_hot_cache_len = max(self.peak_hot_cache_len, seq_len)

    @property
    def cache_budget_tokens(self) -> int:
        """Logical retained-token budget across hot and warm tiers."""
        if self.initial_seq_len is None:
            return 999999
        return max(1, int(self.initial_seq_len * self.config.cache_budget))

    @property
    def hot_budget_tokens(self) -> int:
        """Full-precision budget for the hot tier."""
        total_budget = self.cache_budget_tokens
        if self.uses_tiered_cache:
            hot_budget = max(1, int(total_budget * self.config.semantic_hot_ratio))
            generated_count = 0
            generated_mask = getattr(self.policy, "generated_assistant_mask", None)
            generated_window = getattr(self.policy, "generated_retention_window", 0)
            if generated_mask is not None and generated_window > 0:
                generated_count = min(int(generated_mask.sum().item()), int(generated_window))
            return min(max(hot_budget, generated_count), total_budget)
        return total_budget

    def should_evict(self, current_len: int) -> bool:
        """Return whether the current logical cache exceeds the policy budget."""
        if self.config.policy == "full":
            return False
        if self.uses_tiered_cache:
            return current_len > self.hot_budget_tokens or current_len > self.cache_budget_tokens
        return current_len > self.cache_budget_tokens

    def _tracker_matches_logical_cache(self, current_len: int) -> bool:
        """Return whether tracker state aligns with the full logical cache length."""
        per_head_scores = getattr(self.tracker, "per_head_scores", None)
        return per_head_scores is None or per_head_scores.shape[-1] == current_len

    def _build_cache_data_for_positions(
        self,
        hot_cache: DynamicCache,
        requested_positions: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Assemble a cache slice from hot full-precision data and warm quantized data."""
        requested_positions = requested_positions.to(dtype=torch.long).cpu()
        if requested_positions.numel() == 0:
            return []

        hot_mask = torch.isin(requested_positions, self.hot_positions)
        hot_requested = requested_positions[hot_mask]
        warm_requested = requested_positions[~hot_mask]

        hot_lookup = None
        if hot_requested.numel() > 0:
            hot_lookup = torch.searchsorted(self.hot_positions, hot_requested)

        warm_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        if warm_requested.numel() > 0 and self.warm_tier.size > 0:
            target_device = hot_cache.layers[0].keys.device
            warm_layers = self.warm_tier.materialize_positions(warm_requested, target_device)

        ddp_cache_data: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, hot_layer in enumerate(hot_cache.layers):
            hot_keys_sel = None
            hot_values_sel = None
            if hot_lookup is not None:
                hot_keys_sel = torch.index_select(hot_layer.keys, 2, hot_lookup.to(hot_layer.keys.device))
                hot_values_sel = torch.index_select(hot_layer.values, 2, hot_lookup.to(hot_layer.values.device))

            if hot_keys_sel is not None:
                out_keys = torch.empty(
                    hot_layer.keys.shape[0],
                    hot_layer.keys.shape[1],
                    requested_positions.numel(),
                    hot_layer.keys.shape[3],
                    device=hot_layer.keys.device,
                    dtype=hot_layer.keys.dtype,
                )
                out_values = torch.empty_like(out_keys)
            else:
                warm_keys_ref, warm_values_ref = warm_layers[layer_idx]
                out_keys = torch.empty(
                    warm_keys_ref.shape[0],
                    warm_keys_ref.shape[1],
                    requested_positions.numel(),
                    warm_keys_ref.shape[3],
                    device=warm_keys_ref.device,
                    dtype=warm_keys_ref.dtype,
                )
                out_values = torch.empty_like(out_keys)

            if hot_keys_sel is not None:
                out_keys[:, :, hot_mask, :] = hot_keys_sel
                out_values[:, :, hot_mask, :] = hot_values_sel
            if warm_requested.numel() > 0:
                warm_keys_sel, warm_values_sel = warm_layers[layer_idx]
                out_keys[:, :, ~hot_mask, :] = warm_keys_sel
                out_values[:, :, ~hot_mask, :] = warm_values_sel

            ddp_cache_data.append((out_keys, out_values))

        return ddp_cache_data

    def prepare_past_key_values(self, hot_cache: DynamicCache) -> DynamicCache:
        """Promote only top-k warm tokens into a temporary cache for the next decode step."""
        if not self.uses_tiered_cache or self.warm_tier.size == 0:
            self.last_promoted_warm_count = 0
            self.last_prepared_positions = self.hot_positions.clone()
            self.last_promoted_positions = torch.empty(0, dtype=torch.long)
            self.last_prepared_origin_positions = self.origin_positions[self.hot_positions].clone()
            self.last_promoted_origin_positions = torch.empty(0, dtype=torch.long)
            return hot_cache

        promotion_k = min(max(0, self.config.semantic_warm_top_k), self.warm_tier.size)
        if promotion_k <= 0:
            self.last_promoted_warm_count = 0
            self.last_prepared_positions = self.hot_positions.clone()
            self.last_promoted_positions = torch.empty(0, dtype=torch.long)
            self.last_prepared_origin_positions = self.origin_positions[self.hot_positions].clone()
            self.last_promoted_origin_positions = torch.empty(0, dtype=torch.long)
            return hot_cache

        scores = self.policy.compute_promotion_scores(
            self.current_cache_len,
            hot_positions=self.hot_positions,
        )
        warm_positions = self.warm_tier.positions
        warm_scores = scores[warm_positions]
        eligible_mask = torch.isfinite(warm_scores) & ~torch.isneginf(warm_scores)
        eligible_positions = warm_positions[eligible_mask]
        eligible_scores = warm_scores[eligible_mask]
        if eligible_positions.numel() == 0:
            self.last_promoted_warm_count = 0
            self.last_prepared_positions = self.hot_positions.clone()
            self.last_promoted_positions = torch.empty(0, dtype=torch.long)
            self.last_prepared_origin_positions = self.origin_positions[self.hot_positions].clone()
            self.last_promoted_origin_positions = torch.empty(0, dtype=torch.long)
            return hot_cache

        promotion_k = min(promotion_k, eligible_positions.numel())
        if promotion_k <= 0:
            self.last_promoted_warm_count = 0
            self.last_prepared_positions = self.hot_positions.clone()
            self.last_promoted_positions = torch.empty(0, dtype=torch.long)
            self.last_prepared_origin_positions = self.origin_positions[self.hot_positions].clone()
            self.last_promoted_origin_positions = torch.empty(0, dtype=torch.long)
            return hot_cache

        ranked_promoted_positions = self._select_block_promotions(
            eligible_positions,
            eligible_scores,
            promotion_k,
        )
        promoted_positions = ranked_promoted_positions.sort().values
        requested_positions = self._build_prepared_positions(
            self.hot_positions,
            ranked_promoted_positions,
        )

        self.materialization_steps += 1
        self.promotion_steps += 1
        self.last_promoted_warm_count = promoted_positions.numel()
        self.peak_promoted_warm_count = max(self.peak_promoted_warm_count, self.last_promoted_warm_count)
        self.last_prepared_positions = requested_positions.clone()
        self.last_promoted_positions = ranked_promoted_positions.clone()
        self.last_prepared_origin_positions = self.origin_positions[requested_positions].clone()
        self.last_promoted_origin_positions = self.origin_positions[ranked_promoted_positions].clone()

        ddp_cache_data = self._build_cache_data_for_positions(hot_cache, requested_positions)
        return build_dynamic_cache_from_ddp(ddp_cache_data, self.model_config)

    def _build_prepared_positions(
        self,
        hot_positions: torch.Tensor,
        ranked_promoted_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the materialized decode cache order.

        We keep stable hot ordering first, then append promoted warm positions in
        priority order. This preserves the active hot cache as the backbone while
        letting promoted spans act like a recency-biased semantic patch for the
        next decode step.
        """
        ordered_positions = torch.cat(
            [
                hot_positions.to(dtype=torch.long),
                ranked_promoted_positions.to(dtype=torch.long),
            ]
        )
        if ordered_positions.numel() <= 1:
            return ordered_positions

        seen: set[int] = set()
        unique_positions: list[int] = []
        for logical_position in ordered_positions.tolist():
            if logical_position in seen:
                continue
            seen.add(logical_position)
            unique_positions.append(logical_position)

        return torch.tensor(unique_positions, dtype=torch.long)

    def _select_diverse_promotions(
        self,
        ranked_candidates: torch.Tensor,
        promotion_k: int,
    ) -> torch.Tensor:
        """Pick top-ranked warm positions while suppressing near-duplicate neighbors."""
        if ranked_candidates.numel() <= 1 or promotion_k <= 1:
            return ranked_candidates[:promotion_k]

        min_gap = max(0, int(getattr(self.config, "semantic_promotion_min_gap", 0)))
        if min_gap <= 0:
            return ranked_candidates[:promotion_k]

        selected: list[int] = []
        selected_origins: list[int] = []
        for logical_position in ranked_candidates.tolist():
            origin_position = int(self.origin_positions[logical_position].item())
            if all(abs(origin_position - existing) >= min_gap for existing in selected_origins):
                selected.append(logical_position)
                selected_origins.append(origin_position)
                if len(selected) >= promotion_k:
                    break

        if len(selected) < promotion_k:
            for logical_position in ranked_candidates.tolist():
                if logical_position not in selected:
                    selected.append(logical_position)
                    if len(selected) >= promotion_k:
                        break

        return torch.tensor(
            selected[:promotion_k],
            dtype=torch.long,
        )

    def _select_block_promotions(
        self,
        eligible_positions: torch.Tensor,
        eligible_scores: torch.Tensor,
        promotion_k: int,
    ) -> torch.Tensor:
        """Promote contiguous warm spans instead of isolated token positions."""
        if eligible_positions.numel() == 0 or promotion_k <= 0:
            return torch.empty(0, dtype=torch.long)

        block_size = max(1, int(getattr(self.config, "semantic_promotion_block_size", 1)))
        if eligible_positions.numel() <= 1 or block_size <= 1:
            _, topk_order = eligible_scores.topk(min(promotion_k, eligible_positions.numel()), largest=True)
            ranked_candidates = eligible_positions[topk_order]
            return self._select_diverse_promotions(ranked_candidates, promotion_k)

        blocks: list[dict] = []
        current_positions: list[int] = []
        current_scores: list[float] = []
        current_origins: list[int] = []
        previous_origin: int | None = None

        for logical_position, score in zip(eligible_positions.tolist(), eligible_scores.tolist()):
            origin_position = int(self.origin_positions[logical_position].item())
            should_flush = (
                current_positions
                and (
                    origin_position != (previous_origin + 1 if previous_origin is not None else origin_position)
                    or len(current_positions) >= block_size
                )
            )
            if should_flush:
                blocks.append(
                    {
                        "positions": current_positions.copy(),
                        "origins": current_origins.copy(),
                        "score": max(current_scores) + 0.25 * (sum(current_scores) / len(current_scores)),
                    }
                )
                current_positions.clear()
                current_scores.clear()
                current_origins.clear()

            current_positions.append(logical_position)
            current_scores.append(float(score))
            current_origins.append(origin_position)
            previous_origin = origin_position

        if current_positions:
            blocks.append(
                {
                    "positions": current_positions.copy(),
                    "origins": current_origins.copy(),
                    "score": max(current_scores) + 0.25 * (sum(current_scores) / len(current_scores)),
                }
            )

        blocks.sort(key=lambda block: block["score"], reverse=True)
        min_gap = max(0, int(getattr(self.config, "semantic_promotion_min_gap", 0)))

        selected_positions: list[int] = []
        selected_ranges: list[tuple[int, int]] = []
        for block in blocks:
            block_origins = block["origins"]
            block_start, block_end = block_origins[0], block_origins[-1]
            if any(
                not (block_end + min_gap < start or block_start - min_gap > end)
                for start, end in selected_ranges
            ):
                continue

            remaining_budget = promotion_k - len(selected_positions)
            if remaining_budget <= 0:
                break

            block_positions = block["positions"]
            if len(block_positions) <= remaining_budget:
                selected_positions.extend(block_positions)
                selected_ranges.append((block_start, block_end))
                continue

            block_score_lookup = {
                int(logical_position): float(score)
                for logical_position, score in zip(eligible_positions.tolist(), eligible_scores.tolist())
            }
            best_window_sum: float | None = None
            window_start = 0
            max_window_start = len(block_positions) - remaining_budget
            for candidate_start in range(max_window_start + 1):
                candidate_end = candidate_start + remaining_budget
                candidate_sum = sum(
                    block_score_lookup[int(logical_position)]
                    for logical_position in block_positions[candidate_start:candidate_end]
                )
                if (
                    best_window_sum is None
                    or candidate_sum > best_window_sum
                    or (
                        abs(candidate_sum - best_window_sum) < 1e-6
                        and candidate_start > window_start
                    )
                ):
                    best_window_sum = candidate_sum
                    window_start = candidate_start
            window_end = window_start + remaining_budget
            selected_positions.extend(block_positions[window_start:window_end])
            selected_ranges.append((block_origins[window_start], block_origins[window_end - 1]))
            break

        if len(selected_positions) < promotion_k:
            _, topk_order = eligible_scores.topk(min(promotion_k, eligible_positions.numel()), largest=True)
            fallback_ranked = eligible_positions[topk_order]
            for logical_position in fallback_ranked.tolist():
                if logical_position not in selected_positions:
                    selected_positions.append(logical_position)
                    if len(selected_positions) >= promotion_k:
                        break

        return torch.tensor(selected_positions[:promotion_k], dtype=torch.long)

    @torch.no_grad()
    def finalize_decode_step(
        self,
        hot_cache: DynamicCache,
        decode_output_cache: DynamicCache,
    ) -> DynamicCache:
        """Persist only the newly generated token back into the hot tier after a decode step."""
        if not self.uses_tiered_cache:
            return decode_output_cache

        ddp_cache_data: list[tuple[torch.Tensor, torch.Tensor]] = []
        for hot_layer, out_layer in zip(hot_cache.layers, decode_output_cache.layers):
            new_key = out_layer.keys[:, :, -1:, :]
            new_value = out_layer.values[:, :, -1:, :]
            ddp_cache_data.append(
                (
                    torch.cat([hot_layer.keys, new_key], dim=2),
                    torch.cat([hot_layer.values, new_value], dim=2),
                )
            )

        updated_hot_cache = build_dynamic_cache_from_ddp(ddp_cache_data, self.model_config)
        new_position = torch.tensor([self.current_cache_len], dtype=torch.long)
        self.hot_positions = torch.cat([self.hot_positions, new_position], dim=0)
        new_origin_position = torch.tensor([self.next_origin_position], dtype=torch.long)
        self.origin_positions = torch.cat([self.origin_positions, new_origin_position], dim=0)
        self.next_origin_position += 1
        self.hot_cache_len += 1
        self.current_cache_len += 1
        self.peak_hot_cache_len = max(self.peak_hot_cache_len, self.hot_cache_len)
        if self.should_evict(self.current_cache_len):
            return self.evict(updated_hot_cache)
        return updated_hot_cache

    @torch.no_grad()
    def evict(self, hot_cache: DynamicCache) -> DynamicCache:
        """Rebalance the logical cache into hot, warm, and cold tiers."""
        if get_cache_layer_count(hot_cache) == 0:
            return hot_cache

        current_len = self.current_cache_len if self.uses_tiered_cache else get_cache_seq_len(hot_cache)
        if not self.should_evict(current_len):
            if self.uses_tiered_cache and self.warm_tier.size > 0:
                all_positions = torch.arange(current_len, dtype=torch.long)
                ddp_cache_data = self._build_cache_data_for_positions(hot_cache, all_positions)
                hot_cache = build_dynamic_cache_from_ddp(ddp_cache_data, self.model_config)
                self.hot_positions = all_positions
                self.hot_cache_len = current_len
                self.warm_tier.clear()
            else:
                if current_len > self.origin_positions.numel():
                    missing = current_len - self.origin_positions.numel()
                    start = self.next_origin_position
                    new_origins = torch.arange(start, start + missing, dtype=torch.long)
                    self.origin_positions = torch.cat([self.origin_positions, new_origins], dim=0)
                    self.next_origin_position += missing
                self.hot_cache_len = get_cache_seq_len(hot_cache)
                self.hot_positions = torch.arange(self.hot_cache_len, dtype=torch.long)

            self.current_cache_len = current_len
            self.last_cold_cache_len = 0
            self.peak_hot_cache_len = max(self.peak_hot_cache_len, self.hot_cache_len)
            self.last_prepared_positions = self.hot_positions.clone()
            self.last_prepared_origin_positions = self.origin_positions[self.hot_positions].clone()
            return hot_cache

        eviction_scores = self.policy.compute_eviction_scores(current_len)

        if self.uses_tiered_cache:
            hot_indices_old, warm_indices_old = self.policy.select_tier_indices(
                eviction_scores,
                total_budget=self.cache_budget_tokens,
            )
            retained_indices_old = torch.cat([hot_indices_old, warm_indices_old]).sort().values
            retained_mask = torch.zeros(current_len, dtype=torch.bool)
            retained_mask[retained_indices_old] = True

            hot_cache_data = self._build_cache_data_for_positions(hot_cache, hot_indices_old)
            new_hot_cache = build_dynamic_cache_from_ddp(hot_cache_data, self.model_config)

            if warm_indices_old.numel() > 0:
                warm_cache_data = self._build_cache_data_for_positions(hot_cache, warm_indices_old)
                warm_positions_new = torch.searchsorted(retained_indices_old, warm_indices_old)
                self.warm_tier.rebuild_from_ddp_cache_data(warm_cache_data, warm_positions_new)
            else:
                self.warm_tier.clear()

            if self._tracker_matches_logical_cache(current_len):
                self.tracker.evict_positions(retained_mask)
            if isinstance(self.policy, SemantiCachePolicy):
                self.policy.evict_positions(retained_mask)
            self.origin_positions = self.origin_positions[retained_indices_old]

            hot_positions_new = torch.searchsorted(retained_indices_old, hot_indices_old)
            self.hot_positions = hot_positions_new.cpu()
            self.hot_cache_len = hot_indices_old.numel()
            self.current_cache_len = retained_indices_old.numel()
            self.last_cold_cache_len = current_len - retained_indices_old.numel()
            self.total_evicted += self.last_cold_cache_len
            self.eviction_steps += 1
            self.peak_hot_cache_len = max(self.peak_hot_cache_len, self.hot_cache_len)
            self.peak_warm_cache_len = max(self.peak_warm_cache_len, self.warm_tier.size)
            self.peak_cold_cache_len = max(self.peak_cold_cache_len, self.last_cold_cache_len)
            return new_hot_cache

        budget = self.cache_budget_tokens
        keep_indices = self.policy.select_keep_indices(eviction_scores, budget)
        keep_count = keep_indices.numel()
        if keep_count == 0:
            return hot_cache

        keep_mask = torch.zeros(current_len, dtype=torch.bool)
        keep_mask[keep_indices] = True
        prune_dynamic_cache(hot_cache, keep_indices)
        if self._tracker_matches_logical_cache(current_len):
            self.tracker.evict_positions(keep_mask)
        if isinstance(self.policy, SemantiCachePolicy):
            self.policy.evict_positions(keep_mask)

        actual_evicted = current_len - keep_count
        self.total_evicted += actual_evicted
        self.eviction_steps += 1
        self.current_cache_len = keep_count
        self.hot_cache_len = keep_count
        self.hot_positions = torch.arange(keep_count, dtype=torch.long)
        self.warm_tier.clear()
        self.last_cold_cache_len = actual_evicted
        self.peak_hot_cache_len = max(self.peak_hot_cache_len, keep_count)
        self.peak_cold_cache_len = max(self.peak_cold_cache_len, actual_evicted)
        return hot_cache

    def get_stats(self) -> dict:
        stats = {
            "initial_seq_len": self.initial_seq_len,
            "cache_budget_tokens": self.cache_budget_tokens,
            "cache_budget_ratio": self.config.cache_budget,
            "current_cache_len": self.current_cache_len,
            "hot_cache_len": self.hot_cache_len,
            "warm_cache_len": self.warm_tier.size,
            "cold_cache_len": self.last_cold_cache_len,
            "peak_hot_cache_len": self.peak_hot_cache_len,
            "peak_warm_cache_len": max(self.peak_warm_cache_len, self.warm_tier.size),
            "peak_cold_cache_len": self.peak_cold_cache_len,
            "hot_budget_tokens": self.hot_budget_tokens,
            "hot_ratio": self.config.semantic_hot_ratio,
            "warm_top_k": self.config.semantic_warm_top_k,
            "last_promoted_warm_count": self.last_promoted_warm_count,
            "peak_promoted_warm_count": self.peak_promoted_warm_count,
            "promotion_steps": self.promotion_steps,
            "total_evicted": self.total_evicted,
            "eviction_steps": self.eviction_steps,
            "materialization_steps": self.materialization_steps,
        }
        stats.update(self.warm_tier.get_stats())
        return stats
