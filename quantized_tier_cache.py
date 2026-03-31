"""Utilities for managing a quantized warm tier beside the active DynamicCache."""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from transformers.cache_utils import DynamicCache


@dataclass
class QuantizedTensor:
    """A simple symmetric per-vector int8 quantization container."""

    q: torch.Tensor
    scale: torch.Tensor
    orig_dtype: torch.dtype

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, storage_device: str = "cpu") -> "QuantizedTensor":
        """Quantize along the head-dimension so each token-head vector gets its own scale."""
        if tensor.numel() == 0:
            empty_scale = torch.empty(*tensor.shape[:-1], 1, dtype=torch.float32, device=storage_device)
            return cls(
                q=torch.empty_like(tensor, dtype=torch.int8, device=storage_device),
                scale=empty_scale,
                orig_dtype=tensor.dtype,
            )

        working = tensor.detach().to(torch.float32)
        scale = working.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
        q = torch.round(working / scale).clamp(-127, 127).to(torch.int8)
        return cls(
            q=q.to(storage_device),
            scale=scale.to(storage_device),
            orig_dtype=tensor.dtype,
        )

    def dequantize(self, device: torch.device) -> torch.Tensor:
        if self.q.numel() == 0:
            return self.q.to(device=device, dtype=self.orig_dtype)
        restored = self.q.to(device=device, dtype=torch.float32) * self.scale.to(device=device)
        return restored.to(dtype=self.orig_dtype)


@dataclass
class WarmLayerCache:
    """One layer of quantized warm-tier KV state."""

    key: QuantizedTensor
    value: QuantizedTensor


class QuantizedWarmTier:
    """Quantized cache tier stored outside the active DynamicCache."""

    def __init__(self, storage_device: str = "cpu"):
        self.storage_device = storage_device
        self.positions = torch.empty(0, dtype=torch.long)
        self.layers: list[WarmLayerCache] = []
        self.quantize_time_s = 0.0
        self.dequantize_time_s = 0.0
        self.quantize_ops = 0
        self.dequantize_ops = 0
        self.peak_size = 0

    def clear(self) -> None:
        self.positions = torch.empty(0, dtype=torch.long)
        self.layers = []

    @property
    def size(self) -> int:
        return int(self.positions.numel())

    def rebuild_from_cache(self, past_key_values: DynamicCache, warm_indices: torch.Tensor) -> None:
        """Quantize the selected positions from a materialized cache into the warm tier."""
        if warm_indices.numel() == 0:
            self.clear()
            return

        start = time.perf_counter()
        warm_indices = warm_indices.to(dtype=torch.long)
        self.positions = warm_indices.cpu()
        self.layers = []
        for layer in past_key_values.layers:
            keys = torch.index_select(layer.keys, 2, warm_indices.to(layer.keys.device))
            values = torch.index_select(layer.values, 2, warm_indices.to(layer.values.device))
            self.layers.append(
                WarmLayerCache(
                    key=QuantizedTensor.from_tensor(keys, storage_device=self.storage_device),
                    value=QuantizedTensor.from_tensor(values, storage_device=self.storage_device),
                )
            )
        self.quantize_time_s += time.perf_counter() - start
        self.quantize_ops += 1
        self.peak_size = max(self.peak_size, self.size)

    def layer_tensors(
        self,
        layer_idx: int,
        target_device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = time.perf_counter()
        layer = self.layers[layer_idx]
        keys = layer.key.dequantize(target_device)
        values = layer.value.dequantize(target_device)
        self.dequantize_time_s += time.perf_counter() - start
        self.dequantize_ops += 1
        return keys, values

    def materialize_positions(
        self,
        requested_positions: torch.Tensor,
        target_device: torch.device,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Dequantize only a requested subset of warm positions, preserving the requested order."""
        if requested_positions.numel() == 0:
            return [
                (
                    torch.empty(0, device=target_device),
                    torch.empty(0, device=target_device),
                )
                for _ in self.layers
            ]

        requested_positions = requested_positions.to(dtype=torch.long).cpu()
        lookup = torch.searchsorted(self.positions, requested_positions)
        result: list[tuple[torch.Tensor, torch.Tensor]] = []

        for layer in self.layers:
            start = time.perf_counter()
            qk = torch.index_select(layer.key.q, 2, lookup.to(layer.key.q.device))
            sk = torch.index_select(layer.key.scale, 2, lookup.to(layer.key.scale.device))
            qv = torch.index_select(layer.value.q, 2, lookup.to(layer.value.q.device))
            sv = torch.index_select(layer.value.scale, 2, lookup.to(layer.value.scale.device))

            keys = (qk.to(device=target_device, dtype=torch.float32) * sk.to(device=target_device)).to(
                dtype=layer.key.orig_dtype
            )
            values = (qv.to(device=target_device, dtype=torch.float32) * sv.to(device=target_device)).to(
                dtype=layer.value.orig_dtype
            )
            self.dequantize_time_s += time.perf_counter() - start
            self.dequantize_ops += 1
            result.append((keys, values))

        return result

    def rebuild_from_ddp_cache_data(
        self,
        ddp_cache_data: list[tuple[torch.Tensor, torch.Tensor]],
        logical_positions: torch.Tensor,
    ) -> None:
        """Replace the warm tier with already-sliced tensors and their logical positions."""
        if logical_positions.numel() == 0:
            self.clear()
            return

        start = time.perf_counter()
        self.positions = logical_positions.to(dtype=torch.long).cpu()
        self.layers = []
        for keys, values in ddp_cache_data:
            self.layers.append(
                WarmLayerCache(
                    key=QuantizedTensor.from_tensor(keys, storage_device=self.storage_device),
                    value=QuantizedTensor.from_tensor(values, storage_device=self.storage_device),
                )
            )
        self.quantize_time_s += time.perf_counter() - start
        self.quantize_ops += 1
        self.peak_size = max(self.peak_size, self.size)

    def materialize(
        self,
        hot_cache: DynamicCache,
        hot_positions: torch.Tensor,
        model_config,
    ) -> DynamicCache:
        """Merge full-precision hot cache with dequantized warm cache in original order."""
        if self.size == 0:
            return hot_cache

        hot_positions = hot_positions.to(dtype=torch.long, device=self.positions.device)
        merged_positions = torch.cat([hot_positions.cpu(), self.positions], dim=0)
        order = torch.argsort(merged_positions)

        ddp_cache_data: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, hot_layer in enumerate(hot_cache.layers):
            warm_keys, warm_values = self.layer_tensors(layer_idx, hot_layer.keys.device)
            merged_keys = torch.cat([hot_layer.keys, warm_keys], dim=2)
            merged_values = torch.cat([hot_layer.values, warm_values], dim=2)
            merged_keys = torch.index_select(merged_keys, 2, order.to(merged_keys.device))
            merged_values = torch.index_select(merged_values, 2, order.to(merged_values.device))
            ddp_cache_data.append((merged_keys, merged_values))

        return DynamicCache(ddp_cache_data=ddp_cache_data, config=model_config)

    def get_stats(self) -> dict:
        return {
            "warm_quantize_time_s": self.quantize_time_s,
            "warm_dequantize_time_s": self.dequantize_time_s,
            "warm_quantize_ops": self.quantize_ops,
            "warm_dequantize_ops": self.dequantize_ops,
            "warm_peak_size": self.peak_size,
        }
