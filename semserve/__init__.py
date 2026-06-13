"""SemServe: semantic-aware multi-tenant KV cache allocation for vLLM.

This package is intentionally vLLM-independent so the allocation/scoring logic
can be unit-tested on CPU and reused by both the HF testbed and the vLLM fork.
"""

__all__ = ["block_mapping", "priority", "allocator", "sim"]
