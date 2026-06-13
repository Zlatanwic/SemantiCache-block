"""Global configuration for SemantiCache experiments."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    torch_dtype: str = "float16"
    device_map: str = "auto"
    attn_implementation: str = "eager"

    # BitsAndBytes 4-bit quantization
    use_bnb_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Model architecture (auto-detected from model.config if set to 0)
    num_layers: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    cache_dir: str | None = None

    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True
    mask_polite_openers: bool = False
    demo_stop_on_repeat: bool = False
    stop_when_output_contains: list[str] = field(default_factory=list)
    show_progress_bar: bool = False


@dataclass
class CacheConfig:
    # Fraction of the prefill prompt length retained in the KV cache
    cache_budget: float = 0.5
    policy: Literal[
        "full",
        "window",
        "streaming",
        "h2o",
        "snapkv",
        "kvzip",
        "defensivekv",
        "semantic",
        "block_semantic",
        "tiered_semantic",
        "op_sievekv_lite",
    ] = "full"

    # StreamingLLM
    sink_tokens: int = 4

    # Eviction cadence
    evict_every_n_steps: int = 1

    # SemantiCache weights
    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3
    query_weight: float = 0.25
    factual_weight: float = 0.2

    # Always keep the newest generated tokens to preserve decode stability
    semantic_recent_window: int = 64
    semantic_block_size: int = 16
    # BlockSemantiCache: paged-KV realistic whole-block eviction granularity
    eviction_block_size: int = 16
    semantic_latest_user_tail_tokens: int = 16
    semantic_hot_ratio: float = 0.5
    semantic_hot_block_size: int = 6
    semantic_warm_device: Literal["cpu", "same"] = "cpu"
    semantic_warm_bits: Literal[8] = 8
    semantic_warm_top_k: int = 16
    semantic_warm_promotable_reserve: int = 8
    semantic_generated_retention_window: int = 12
    semantic_hot_recent_window: int = 8
    semantic_promotion_block_size: int = 4
    semantic_promotion_min_gap: int = 4
    semantic_debug_promotion: bool = False
    semantic_debug_promotion_every: int = 25
    semantic_debug_promotion_top_n: int = 4
    semantic_debug_tiers: bool = False
    semantic_debug_tiers_top_n: int = 8
    semantic_debug_logits_top_n: int = 0
    semantic_follow_token_bias: float = 4.0

    # OP-SieveKV-Lite semantic segment policy
    op_max_segment_tokens: int = 32
    op_min_segment_tokens: int = 4
    op_recent_window: int = 16
    op_uncertainty_weight: float = 0.15
    op_policy_ckpt: str | None = None

    # Hard protection for semantic roles
    pin_system: bool = True
    pin_latest_user: bool = True


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    seed: int = 42
    output_dir: str = "./results"
