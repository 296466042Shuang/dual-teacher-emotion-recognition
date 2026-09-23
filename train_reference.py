"""One-step DT-KD smoke test with synthetic data.

This script demonstrates the public model interfaces and composite loss. It is
not the paper's full training pipeline and is not intended to reproduce the
reported benchmark numbers.
"""

import torch

from src.models import ModalityAligned2DTeacher, Privileged3DTeacher, StudentModel
from src.models.distillation import DTKDLoss


def freeze(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def main() -> None:
    torch.manual_seed(42)

    batch_size = 4
    frames = 60
    joints = 17

    teacher_3d = Privileged3DTeacher(num_joints=joints)
    teacher_2d = ModalityAligned2DTeacher(num_joints=joints)
    student = StudentModel(num_joints=joints)

    freeze(teacher_3d)
    freeze(teacher_2d)
    student.train()

    criterion = DTKDLoss(student_feature_dim=96, teacher_feature_dim=128)
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(criterion.parameters()),
        lr=3e-4,
        weight_decay=1e-4,
    )

    pose_3d = torch.randn(batch_size, frames, joints, 3)
    pose_2d = torch.randn(batch_size, frames, joints, 2)
    lma = torch.randn(batch_size, 33)

    fine_target = torch.randint(0, 2, (batch_size, 26)).float()
    coarse_target = torch.randint(0, 2, (batch_size, 7)).float()

    with torch.no_grad():
        out_3d = teacher_3d(pose_3d, lma)
        out_2d = teacher_2d(pose_2d, lma)

    out_student = student(pose_2d, lma)
    losses = criterion(
        out_student,
        out_3d,
        out_2d,
        fine_target,
        coarse_target,
    )

    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()

    printable = {name: float(value.detach()) for name, value in losses.items()}
    print(printable)


if __name__ == "__main__":
    main()
