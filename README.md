# Privacy-Preserving Bodily Emotion Recognition

**Dual-teacher knowledge distillation for in-the-wild skeleton-based emotion recognition**

Research project accompanying:

**Shuang Wu and Daniela M. Romano**  
*Privacy-preserving in-the-wild bodily expressed emotion recognition: A dual-teacher distillation framework with psychological priors*  
**AI Open (2026)**  
DOI: https://doi.org/10.1016/j.aiopen.2026.08.001

> **Public-code note:** this repository contains a compact reference implementation reconstructed around the published method and deployment interface. It is intended to communicate the modelling design clearly; it is **not** the exact internal training code used to produce the paper's reported benchmark numbers. Dataset-specific preprocessing, trained weights, private experiment orchestration, and some research artefacts are not released here.

## Overview

Skeleton-based affect recognition provides a privacy-conscious alternative to appearance-heavy RGB pipelines, but real-world deployment remains difficult because 2D pose observations can be noisy, incomplete, viewpoint-dependent, and affected by occlusion.

DT-KD uses complementary training-time supervision to narrow this lab-to-field gap:

1. a **privileged 3D teacher** for cleaner kinematic guidance;
2. a **modality-aligned 2D teacher** closer to the deployment domain;
3. a compact **2D + 33-D LMA student** retained at inference.

The public reference code also illustrates sample-wise entropy routing between teachers, fine/coarse prediction heads corresponding to the psychology-grounded 7→26 structure, feature alignment, and optional prototype regularisation.

A broader research question behind this work is:

> **Which latent information remains recoverable when sensing observations are noisy, incomplete, or constrained by privacy requirements?**

## Public reference implementation

The repository intentionally exposes the core research interface rather than the complete experimental stack.

### Included

- PyTorch temporal pose encoder for 2D/3D skeleton sequences.
- Privileged 3D teacher.
- Modality-aligned 2D teacher.
- Compact deployment-time student.
- 33-D LMA feature interface.
- Sample-wise entropy routing.
- Multi-label knowledge-distillation loss.
- Coarse 7-way and fine 26-way prediction heads.
- Lightweight feature-alignment and prototype-loss hooks.
- Synthetic inference and one-step training smoke tests.

### Not included

- Raw BoLD, EMOTIC, or other restricted datasets.
- Dataset-specific research preprocessing pipelines.
- Trained checkpoints used for the paper.
- Exact paper-specific LMA/VAD prototype values.
- Full experiment orchestration and ablation infrastructure.
- Private paths, cluster scripts, or internal research utilities.

## Repository structure

```text
.
├── assets/
├── configs/
│   └── example.yaml
├── demo/
│   └── inference_demo.py
├── examples/
├── src/
│   ├── datasets/
│   ├── features/
│   │   └── lma.py
│   ├── models/
│   │   ├── common.py
│   │   ├── distillation.py
│   │   ├── student.py
│   │   └── teachers.py
│   └── utils/
│       └── metrics.py
├── train_reference.py
├── requirements.txt
└── README.md
```

## Installation

Python 3.10+ and PyTorch are recommended.

```bash
git clone https://github.com/296466042Shuang/dual-teacher-emotion-recognition.git
cd dual-teacher-emotion-recognition
pip install -r requirements.txt
```

## Quick start

Run the deployment-time student on synthetic 2D pose + 33-D LMA input:

```bash
python demo/inference_demo.py
```

Run a single synthetic DT-KD training step to inspect the teacher/student interfaces and composite objective:

```bash
python train_reference.py
```

These scripts are smoke tests for the public reference implementation and do not reproduce the paper's benchmark results.

## Core model interface

```python
from src.models import StudentModel

student = StudentModel(
    num_joints=17,
    lma_dim=33,
    feature_dim=96,
)

output = student(pose_2d, lma_33d)
fine_logits = output["logits_26"]
coarse_logits = output["logits_7"]
```

At deployment, teacher networks and privileged 3D observations are not required.

## Data

The original experiments use research datasets subject to their respective licences and access conditions. Raw datasets are therefore **not redistributed through this repository**.

Only synthetic or non-sensitive examples are used in the public smoke tests.

## Citation

```bibtex
@article{wu2026privacy,
  title={Privacy-preserving in-the-wild bodily expressed emotion recognition: A dual-teacher distillation framework with psychological priors},
  author={Wu, Shuang and Romano, Daniela M.},
  journal={AI Open},
  volume={7},
  pages={248--275},
  year={2026},
  doi={10.1016/j.aiopen.2026.08.001}
}
```

## Research context

This project sits within broader work on temporal human-behaviour modelling, privacy-preserving computer vision, multimodal sensing, knowledge distillation, representation learning under noisy or partial observations, and deployment-aware human-centred AI.

## Authors

**Shuang Wu** — University College London (UCL)  
**Daniela M. Romano**

## Licence

No blanket software licence is granted yet. A licence will be added after the public release has been checked for third-party code and data dependencies.
