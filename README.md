# Privacy-Preserving Bodily Emotion Recognition

**Dual-teacher knowledge distillation for in-the-wild skeleton-based emotion recognition**

Research code and project overview accompanying:

**Shuang Wu and Daniela M. Romano**  
*Privacy-preserving in-the-wild bodily expressed emotion recognition: A dual-teacher distillation framework with psychological priors*  
**AI Open (2026)**  
DOI: https://doi.org/10.1016/j.aiopen.2026.08.001

> **Repository status:** the public research-code release is being cleaned and documented. This repository currently provides the project overview and will host the core modelling, inference, and reproducibility components.

## Overview

Skeleton-based affect recognition provides a privacy-conscious alternative to appearance-heavy RGB pipelines, but real-world deployment remains difficult because 2D pose observations can be noisy, incomplete, viewpoint-dependent, and affected by occlusion.

This project studies how richer training-time information can be transferred into a deployable model operating on lightweight body-motion representations. The framework combines complementary teacher signals, psychologically informed motion features, and knowledge distillation.

A broader research question behind this work is:

> **Which latent information remains recoverable when sensing observations are noisy, incomplete, or constrained by privacy requirements?**

## Method

The framework contains three main components:

1. **Privileged 3D teacher** — learns from richer 3D body-motion information and provides structured supervision unavailable at deployment time.
2. **Modality-aligned 2D teacher** — provides supervision closer to the deployment setting and helps reduce the gap between privileged training information and noisy 2D observations.
3. **Deployable student** — learns from complementary teacher signals and is designed for inference using privacy-preserving pose-derived representations.

Psychologically informed motion descriptors, including Laban Movement Analysis (LMA)-related features, are incorporated as interpretable affective priors.

## Research highlights

- Dual-teacher knowledge distillation with complementary privileged and deployment-aligned supervision.
- Temporal modelling of human body motion rather than static appearance alone.
- Privacy-conscious inference based on pose-derived representations.
- Robustness-oriented learning under noisy and incomplete observations.
- Psychologically informed motion representations using LMA/VAD-related priors.
- Deployment-aware design: richer information can be used during training without requiring it at inference.

## Repository structure

```text
.
├── assets/
├── configs/
├── demo/
├── examples/
├── src/
│   ├── datasets/
│   ├── features/
│   ├── models/
│   └── utils/
├── requirements.txt
└── README.md
```

## Installation

The cleaned implementation is being prepared for public release. The intended environment is Python 3.10+ with PyTorch.

```bash
git clone https://github.com/296466042Shuang/dual-teacher-emotion-recognition.git
cd dual-teacher-emotion-recognition
pip install -r requirements.txt
```

## Data

The original experiments use research datasets subject to their respective licences and access conditions. Raw datasets are therefore **not redistributed through this repository**.

Small synthetic or non-sensitive examples may be included to demonstrate expected input formats without exposing identifiable imagery or restricted data.

## Reproducibility

The public release will focus on the core modelling and inference components needed to understand the dual-teacher framework. Dataset-specific preprocessing and experiment paths will be documented separately where redistribution is permitted.

## Citation

```bibtex
@article{wu2026privacy,
  title={Privacy-preserving in-the-wild bodily expressed emotion recognition: A dual-teacher distillation framework with psychological priors},
  author={Wu, Shuang and Romano, Daniela M.},
  journal={AI Open},
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

Licence information will be added after the cleaned public code release is finalised and third-party code/data dependencies have been checked.
