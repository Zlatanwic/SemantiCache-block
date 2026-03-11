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

    # Qwen2.5-3B architecture
    num_layers: int = 36
    num_attention_heads: int = 16
    num_kv_heads: int = 2
    head_dim: int = 128

    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True


@dataclass
class CacheConfig:
    # Fraction of the prefill prompt length retained in the KV cache
    cache_budget: float = 0.5
    policy: Literal["full", "window", "streaming", "h2o", "semantic"] = "full"

    # StreamingLLM
    sink_tokens: int = 4

    # Eviction cadence
    evict_every_n_steps: int = 1

    # SemantiCache weights
    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3

    # Always keep the newest generated tokens to preserve decode stability
    semantic_recent_window: int = 64
    semantic_block_size: int = 16
    semantic_latest_user_tail_tokens: int = 64

    # Hard protection for semantic roles
    pin_system: bool = True
    pin_latest_user: bool = True


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    seed: int = 42
    output_dir: str = "./results"
