"""
KVCacheManager: KV Cache 驱逐的核心执行器。

职责:
1. 在每步 decode 后检查 cache 是否超出预算
2. 调用 EvictionPolicy 计算驱逐分数
3. 对 past_key_values 执行 index_select 裁剪
4. 同步更新 AttentionTracker 和 SemantiCachePolicy 的内部状态
"""

import torch
from typing import Optional
from transformers.cache_utils import DynamicCache

from config import CacheConfig
from eviction_policies import EvictionPolicy, SemantiCachePolicy
from attention_tracker import AttentionTracker


def get_cache_layer_count(past_key_values: DynamicCache) -> int:
    """兼容旧版 `key_cache` 和新版 `layers` 结构。"""
    if hasattr(past_key_values, "layers"):
        return len(past_key_values.layers)
    if hasattr(past_key_values, "key_cache"):
        return len(past_key_values.key_cache)
    return 0


def get_cache_seq_len(past_key_values: DynamicCache) -> int:
    """返回第一层 KV cache 的序列长度。"""
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
    """返回第一层 key cache 的 shape。"""
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
    """原地裁剪 DynamicCache，兼容 Transformers 旧版和 5.x 新版结构。"""
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


class KVCacheManager:
    """管理 KV Cache 的驱逐操作"""

    def __init__(
        self,
        config: CacheConfig,
        policy: EvictionPolicy,
        tracker: AttentionTracker,
        num_layers: int,
    ):
        self.config = config
        self.policy = policy
        self.tracker = tracker
        self.num_layers = num_layers

        # prefill 后的初始 sequence length (用于计算预算)
        self.initial_seq_len: Optional[int] = None
        # 当前实际保留的 cache 大小
        self.current_cache_len: int = 0
        # 驱逐统计
        self.total_evicted = 0
        self.eviction_steps = 0

    def set_initial_seq_len(self, seq_len: int):
        """prefill 后调用, 设定基准 cache 长度"""
        self.initial_seq_len = seq_len
        self.current_cache_len = seq_len

    @property
    def cache_budget_tokens(self) -> int:
        """根据 budget 比例计算允许的最大 cache token 数"""
        if self.initial_seq_len is None:
            return 999999
        return int(self.initial_seq_len * self.config.cache_budget)

    def should_evict(self, current_len: int) -> bool:
        """判断当前 cache 是否需要驱逐"""
        if self.config.policy == "full":
            return False
        return current_len > self.cache_budget_tokens

    @torch.no_grad()
    def evict(self, past_key_values: DynamicCache) -> DynamicCache:
        """
        对 DynamicCache 执行驱逐操作。

        HuggingFace Transformers 的 DynamicCache 内部结构:
        - past_key_values.key_cache: list of tensors, 每层一个
          shape: [batch, num_kv_heads, seq_len, head_dim]
        - past_key_values.value_cache: 同上

        驱逐 = 沿 seq_len 维度 (dim=2) 做 index_select, 只保留高分 token。
        """
        if get_cache_layer_count(past_key_values) == 0:
            return past_key_values

        current_len = get_cache_seq_len(past_key_values)

        if not self.should_evict(current_len):
            self.current_cache_len = current_len
            return past_key_values

        budget = self.cache_budget_tokens
        num_to_evict = current_len - budget

        if num_to_evict <= 0:
            return past_key_values

        # 计算驱逐分数
        eviction_scores = self.policy.compute_eviction_scores(current_len)

        # 选择要保留的 token
        keep_indices = self.policy.select_keep_indices(eviction_scores, budget)
        keep_count = keep_indices.numel()
        if keep_count == 0:
            return past_key_values

        # 创建 keep mask (用于更新 tracker 等)
        keep_mask = torch.zeros(current_len, dtype=torch.bool)
        keep_mask[keep_indices] = True

        # 对每层的 KV cache 执行裁剪
        prune_dynamic_cache(past_key_values, keep_indices)

        # 同步更新 tracker 的内部状态
        self.tracker.evict_positions(keep_mask)

        # 如果是 SemantiCache, 也更新其语义信号
        if isinstance(self.policy, SemantiCachePolicy):
            self.policy.evict_positions(keep_mask)

        # 更新统计
        actual_evicted = current_len - keep_count
        self.total_evicted += actual_evicted
        self.eviction_steps += 1
        self.current_cache_len = keep_count

        return past_key_values

    def get_stats(self) -> dict:
        return {
            "initial_seq_len": self.initial_seq_len,
            "cache_budget_tokens": self.cache_budget_tokens,
            "cache_budget_ratio": self.config.cache_budget,
            "current_cache_len": self.current_cache_len,
            "total_evicted": self.total_evicted,
            "eviction_steps": self.eviction_steps,
        }
