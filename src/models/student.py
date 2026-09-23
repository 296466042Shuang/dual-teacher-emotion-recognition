"""Deployable student model.

At inference time this model requires only 2D pose sequences and 33-D LMA
features. Teacher models are not needed.
"""

from __future__ import annotations

from torch import nn

from .common import FusionHead, LMAEncoder, TemporalPoseEncoder


class StudentModel(nn.Module):
    """Compact 2D-pose student used for deployment."""

    def __init__(
        self,
        num_joints: int = 17,
        lma_dim: int = 33,
        feature_dim: int = 96,
    ) -> None:
        super().__init__()
        self.pose_encoder = TemporalPoseEncoder(
            num_joints=num_joints,
            coord_dim=2,
            hidden_dim=96,
            out_dim=feature_dim,
        )
        self.lma_encoder = LMAEncoder(lma_dim, hidden_dim=48, out_dim=48)
        self.head = FusionHead(feature_dim, 48, feature_dim)

    def forward(self, pose_2d, lma):
        pose_feature = self.pose_encoder(pose_2d)
        lma_feature = self.lma_encoder(lma)
        return self.head(pose_feature, lma_feature)
