"""Public DT-KD model components."""

from .student import StudentModel
from .teachers import ModalityAligned2DTeacher, Privileged3DTeacher

__all__ = [
    "StudentModel",
    "Privileged3DTeacher",
    "ModalityAligned2DTeacher",
]
