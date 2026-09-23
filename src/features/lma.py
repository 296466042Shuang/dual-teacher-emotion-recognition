"""Utilities for deployment-time 33-D LMA descriptors.

The paper uses a 33-D Laban Movement Analysis representation. This public
repository deliberately does not recreate dataset-specific extraction logic
without the original preprocessing pipeline. Instead, it exposes a clean
interface for validated precomputed descriptors.
"""

from __future__ import annotations

import torch

LMA_DIM = 33


def validate_lma(lma: torch.Tensor) -> torch.Tensor:
    """Validate and sanitise a batch of 33-D LMA descriptors."""
    if lma.ndim != 2 or lma.shape[-1] != LMA_DIM:
        raise ValueError(f"expected LMA tensor with shape [B, {LMA_DIM}]")
    return torch.nan_to_num(lma.float(), nan=0.0, posinf=0.0, neginf=0.0)


def standardize_lma(
    lma: torch.Tensor,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Standardise LMA descriptors using supplied or batch statistics."""
    lma = validate_lma(lma)
    if mean is None:
        mean = lma.mean(dim=0, keepdim=True)
    if std is None:
        std = lma.std(dim=0, keepdim=True, unbiased=False)
    return (lma - mean) / std.clamp_min(eps)
