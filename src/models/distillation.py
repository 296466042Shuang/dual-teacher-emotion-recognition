"""Distillation utilities for the compact public DT-KD reference implementation.

The original paper combines complementary teacher signals through sample-wise
entropy routing and additional structured regularisation. This file provides a
lightweight public implementation of those ideas rather than the exact private
training code used for the reported experiments.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def binary_entropy_from_logits(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return mean binary entropy per sample for multi-label logits."""
    p = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    h = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
    return h.mean(dim=-1)


def entropy_routing_weights(
    teacher_a_logits: torch.Tensor,
    teacher_b_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sample-wise teacher weights from predictive confidence."""
    max_h = math.log(2.0)
    conf_a = 1.0 - binary_entropy_from_logits(teacher_a_logits) / max_h
    conf_b = 1.0 - binary_entropy_from_logits(teacher_b_logits) / max_h

    conf = torch.stack([conf_a, conf_b], dim=-1).clamp_min(1e-4)
    weight = conf / conf.sum(dim=-1, keepdim=True)
    return weight[:, 0], weight[:, 1]


def routed_teacher_probability(
    teacher_a_logits: torch.Tensor,
    teacher_b_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Fuse both teachers after sample-wise entropy routing."""
    wa, wb = entropy_routing_weights(teacher_a_logits, teacher_b_logits)
    pa = torch.sigmoid(teacher_a_logits / temperature)
    pb = torch.sigmoid(teacher_b_logits / temperature)
    return wa[:, None] * pa + wb[:, None] * pb


def multi_label_kd_loss(
    student_logits: torch.Tensor,
    target_probability: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Soft-target distillation for a multi-label affective output space."""
    loss = F.binary_cross_entropy_with_logits(
        student_logits / temperature,
        target_probability,
    )
    return loss * (temperature ** 2)


def prototype_alignment_loss(
    feature: torch.Tensor,
    fine_targets: torch.Tensor,
    prototypes: torch.Tensor | None,
) -> torch.Tensor:
    """Optional prototype regulariser.

    Prototypes should have shape [26, D]. This repository intentionally does
    not ship paper-specific psychological prototype values.
    """
    if prototypes is None:
        return feature.new_zeros(())

    if prototypes.ndim != 2 or prototypes.shape[0] != fine_targets.shape[-1]:
        raise ValueError("prototypes must have shape [num_fine_classes, feature_dim]")

    weights = fine_targets.float()
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    target = weights @ prototypes / denom
    return 1.0 - F.cosine_similarity(feature, target, dim=-1).mean()


class DTKDLoss(nn.Module):
    """Compact composite objective reflecting the public paper description."""

    def __init__(
        self,
        student_feature_dim: int = 96,
        teacher_feature_dim: int = 128,
        temperature: float = 2.0,
        alpha_kd: float = 1.0,
        alpha_coarse: float = 0.25,
        alpha_feature: float = 0.10,
        alpha_proto: float = 0.05,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha_kd = alpha_kd
        self.alpha_coarse = alpha_coarse
        self.alpha_feature = alpha_feature
        self.alpha_proto = alpha_proto
        self.student_to_teacher = nn.Linear(student_feature_dim, teacher_feature_dim)

    def forward(
        self,
        student: dict,
        teacher_3d: dict,
        teacher_2d: dict,
        fine_targets: torch.Tensor,
        coarse_targets: torch.Tensor | None = None,
        prototypes: torch.Tensor | None = None,
    ) -> dict:
        hard = F.binary_cross_entropy_with_logits(
            student["logits_26"],
            fine_targets.float(),
        )

        routed_prob = routed_teacher_probability(
            teacher_3d["logits_26"],
            teacher_2d["logits_26"],
            temperature=self.temperature,
        )
        kd = multi_label_kd_loss(
            student["logits_26"],
            routed_prob.detach(),
            temperature=self.temperature,
        )

        coarse = student["logits_26"].new_zeros(())
        if coarse_targets is not None:
            coarse = F.binary_cross_entropy_with_logits(
                student["logits_7"],
                coarse_targets.float(),
            )

        wa, wb = entropy_routing_weights(
            teacher_3d["logits_26"],
            teacher_2d["logits_26"],
        )
        teacher_feature = (
            wa[:, None] * teacher_3d["feature"]
            + wb[:, None] * teacher_2d["feature"]
        ).detach()

        projected_student = self.student_to_teacher(student["feature"])
        feature = F.mse_loss(
            F.normalize(projected_student, dim=-1),
            F.normalize(teacher_feature, dim=-1),
        )

        proto = prototype_alignment_loss(
            student["feature"],
            fine_targets,
            prototypes,
        )

        total = (
            hard
            + self.alpha_kd * kd
            + self.alpha_coarse * coarse
            + self.alpha_feature * feature
            + self.alpha_proto * proto
        )

        return {
            "total": total,
            "hard": hard,
            "kd": kd,
            "coarse": coarse,
            "feature": feature,
            "prototype": proto,
        }
