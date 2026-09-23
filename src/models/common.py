"""Shared PyTorch building blocks for the public DT-KD reference implementation.

This code is intentionally compact. It illustrates the model interfaces used by
the paper without reproducing private experiment infrastructure or datasets.
"""

from __future__ import annotations

import torch
from torch import nn


class TemporalPoseEncoder(nn.Module):
    """Encode a skeleton sequence shaped [B, T, J, C]."""

    def __init__(
        self,
        num_joints: int,
        coord_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        in_channels = num_joints * coord_dim
        self.num_joints = num_joints
        self.coord_dim = coord_dim

        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        if pose.ndim != 4:
            raise ValueError("pose must have shape [B, T, J, C]")

        b, t, j, c = pose.shape
        if j != self.num_joints or c != self.coord_dim:
            raise ValueError(
                f"expected J={self.num_joints}, C={self.coord_dim}; got J={j}, C={c}"
            )

        x = pose.reshape(b, t, j * c).transpose(1, 2)
        x = self.temporal(x)
        x = x.mean(dim=-1)
        return self.proj(x)


class LMAEncoder(nn.Module):
    """Encode the 33-D deployment-time LMA descriptor."""

    def __init__(self, in_dim: int = 33, hidden_dim: int = 64, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, lma: torch.Tensor) -> torch.Tensor:
        if lma.ndim != 2:
            raise ValueError("lma must have shape [B, 33]")
        return self.net(lma)


class FusionHead(nn.Module):
    """Fuse temporal pose and LMA features into affect predictions."""

    def __init__(
        self,
        pose_dim: int,
        lma_dim: int,
        fused_dim: int,
        num_fine_classes: int = 26,
        num_coarse_classes: int = 7,
    ) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(pose_dim + lma_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.fine_head = nn.Linear(fused_dim, num_fine_classes)
        self.coarse_head = nn.Linear(fused_dim, num_coarse_classes)

    def forward(self, pose_feature: torch.Tensor, lma_feature: torch.Tensor) -> dict:
        feature = self.fuse(torch.cat([pose_feature, lma_feature], dim=-1))
        return {
            "feature": feature,
            "logits_26": self.fine_head(feature),
            "logits_7": self.coarse_head(feature),
        }
