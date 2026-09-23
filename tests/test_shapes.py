"""Minimal public interface tests."""

import unittest

import torch

from src.models import ModalityAligned2DTeacher, Privileged3DTeacher, StudentModel
from src.models.distillation import entropy_routing_weights


class ShapeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.batch = 2
        self.frames = 16
        self.joints = 17
        self.lma = torch.randn(self.batch, 33)
        self.pose_2d = torch.randn(self.batch, self.frames, self.joints, 2)
        self.pose_3d = torch.randn(self.batch, self.frames, self.joints, 3)

    def test_student_output_shapes(self):
        model = StudentModel(num_joints=self.joints)
        out = model(self.pose_2d, self.lma)
        self.assertEqual(tuple(out["logits_26"].shape), (self.batch, 26))
        self.assertEqual(tuple(out["logits_7"].shape), (self.batch, 7))
        self.assertEqual(tuple(out["feature"].shape), (self.batch, 96))

    def test_teacher_output_shapes(self):
        t3 = Privileged3DTeacher(num_joints=self.joints)
        t2 = ModalityAligned2DTeacher(num_joints=self.joints)

        out3 = t3(self.pose_3d, self.lma)
        out2 = t2(self.pose_2d, self.lma)

        self.assertEqual(tuple(out3["logits_26"].shape), (self.batch, 26))
        self.assertEqual(tuple(out2["logits_26"].shape), (self.batch, 26))
        self.assertEqual(tuple(out3["feature"].shape), (self.batch, 128))
        self.assertEqual(tuple(out2["feature"].shape), (self.batch, 128))

    def test_entropy_weights_are_normalised(self):
        a = torch.randn(self.batch, 26)
        b = torch.randn(self.batch, 26)
        wa, wb = entropy_routing_weights(a, b)
        self.assertTrue(torch.allclose(wa + wb, torch.ones_like(wa), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
