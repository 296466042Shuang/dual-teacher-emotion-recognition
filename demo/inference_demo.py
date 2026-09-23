"""Minimal deployment-time smoke test using synthetic inputs."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.lma import standardize_lma
from src.models import StudentModel


def main() -> None:
    torch.manual_seed(7)

    model = StudentModel(num_joints=17, lma_dim=33, feature_dim=96)
    model.eval()

    # Example 2-second-like skeleton clip represented as T frames.
    pose_2d = torch.randn(2, 60, 17, 2)
    lma_33d = standardize_lma(torch.randn(2, 33))

    with torch.no_grad():
        out = model(pose_2d, lma_33d)
        probability = torch.sigmoid(out["logits_26"])

    print("2D pose:", tuple(pose_2d.shape))
    print("LMA:", tuple(lma_33d.shape))
    print("26-way probabilities:", tuple(probability.shape))
    print("First sample, first five probabilities:", probability[0, :5])


if __name__ == "__main__":
    main()
