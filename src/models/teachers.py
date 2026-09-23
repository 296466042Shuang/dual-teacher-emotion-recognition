"""Teacher models for the compact public DT-KD reference implementation."""

from __future__ import annotations

from torch import nn

from .common import FusionHead, LMAEncoder, TemporalPoseEncoder


class Privileged3DTeacher(nn.Module):
    """Teacher operating on privileged 3D kinematics available during training."""

    def __init__(
        self,
        num_joints: int = 17,
        lma_dim: int = 33,
        feature_dim: int = 128,
    ) -> None:
        super().__init__()
        self.pose_encoder = TemporalPoseEncoder(
            num_joints=num_joints,
            coord_dim=3,
            hidden_dim=160,
            out_dim=feature_dim,
        )
        self.lma_encoder = LMAEncoder(lma_dim, hidden_dim=80, out_dim=64)
        self.head = FusionHead(feature_dim, 64, feature_dim)

    def forward(self, pose_3d, lma):
        pose_feature = self.pose_encoder(pose_3d)
        lma_feature = self.lma_encoder(lma)
        return self.head(pose_feature, lma_feature)


class ModalityAligned2DTeacher(nn.Module):
    """Teacher aligned with noisy 2D skeleton observations used at deployment."""

    def __init__(
        self,
        num_joints: int = 17,
        lma_dim: int = 33,
        feature_dim: int = 128,
    ) -> None:
        super().__init__()
        self.pose_encoder = TemporalPoseEncoder(
            num_joints=num_joints,
            coord_dim=2,
            hidden_dim=144,
            out_dim=feature_dim,
        )
        self.lma_encoder = LMAEncoder(lma_dim, hidden_dim=72, out_dim=64)
        self.head = FusionHead(feature_dim, 64, feature_dim)

    def forward(self, pose_2d, lma):
        pose_feature = self.pose_encoder(pose_2d)
        lma_feature = self.lma_encoder(lma)
        return self.head(pose_feature, lma_feature)
