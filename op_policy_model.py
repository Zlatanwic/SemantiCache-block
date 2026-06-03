"""Lightweight learned retention policy for OP-SieveKV distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


SIGNAL_NAMES = ["attn", "entropy", "density", "query", "factual", "authority", "recency"]
POOL_NAMES = ["max", "mean", "std"]
ROLE_NAMES = ["filler", "assistant", "context", "user_history", "user_latest", "system"]

FEATURE_NAMES = [
    *[f"{signal}_{pool}" for signal in SIGNAL_NAMES for pool in POOL_NAMES],
    "start_norm",
    "end_norm",
    "center_norm",
    "length_norm",
    "budget_ratio",
    *[f"role_{role}" for role in ROLE_NAMES],
    "pinned_frac",
    "question_tail_frac",
    "question_like_frac",
    "template_frac",
    "boundary_frac",
    "heuristic_keep_prob",
]
FEATURE_DIM = len(FEATURE_NAMES)


class SegmentRetentionMLP(nn.Module):
    """Small MLP that maps segment features to a keep logit."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def save_policy_checkpoint(
    path: str | Path,
    model: SegmentRetentionMLP,
    *,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a trained policy with feature normalization statistics."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "architecture": {
                "input_dim": model.input_dim,
                "hidden_dim": model.hidden_dim,
                "dropout": model.dropout,
            },
            "feature_mean": feature_mean.detach().cpu(),
            "feature_std": feature_std.detach().cpu().clamp(min=1e-6),
            "metadata": metadata or {},
        },
        output_path,
    )


def load_policy_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[SegmentRetentionMLP, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Load a trained segment retention policy checkpoint."""
    checkpoint = torch.load(Path(path), map_location=map_location)
    feature_names = checkpoint.get("feature_names")
    if feature_names != FEATURE_NAMES:
        raise ValueError(
            "OP policy checkpoint feature schema mismatch. "
            f"Expected {len(FEATURE_NAMES)} features, got {len(feature_names or [])}."
        )

    architecture = checkpoint.get("architecture", {})
    model = SegmentRetentionMLP(
        input_dim=int(architecture.get("input_dim", FEATURE_DIM)),
        hidden_dim=int(architecture.get("hidden_dim", 64)),
        dropout=float(architecture.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    feature_mean = checkpoint["feature_mean"].to(map_location)
    feature_std = checkpoint["feature_std"].to(map_location).clamp(min=1e-6)
    metadata = checkpoint.get("metadata", {})
    return model, feature_mean, feature_std, metadata
