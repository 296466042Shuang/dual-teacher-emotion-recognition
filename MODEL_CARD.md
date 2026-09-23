# Model Card

## Model summary

This repository provides a compact public reference implementation of a dual-teacher knowledge-distillation framework for privacy-preserving bodily emotion recognition.

The deployment model consumes:

- 2D skeleton sequences with shape `[B, T, J, 2]`
- 33-D Laban Movement Analysis (LMA)-related descriptors with shape `[B, 33]`

and produces:

- 26-D fine affect logits
- 7-D coarse affect logits

Teacher models are used only during training in the reference design.

## Intended use

This implementation is intended for:

- research and educational use;
- studying privileged-information learning and teacher–student distillation;
- prototyping skeleton-based temporal representation learning;
- testing deployment interfaces that avoid raw RGB appearance at inference.

It is not intended as a clinical, diagnostic, surveillance, or high-stakes decision system.

## Training-time information

The public design includes:

- a privileged 3D teacher;
- a modality-aligned 2D teacher;
- sample-wise confidence / entropy routing;
- a deployable 2D + LMA student;
- optional feature and prototype alignment.

The public repository does not include the original restricted datasets, private experimental infrastructure, or paper checkpoints.

## Privacy characteristics

The deployment interface is designed around pose-derived representations rather than raw RGB appearance.

This can reduce exposure to identity-bearing visual information, but skeleton data are **not inherently anonymous**. Motion patterns may still contain identifying or sensitive information. Any real deployment should therefore apply appropriate data minimisation, access control, retention, and consent procedures.

## Limitations

The compact public implementation is designed to communicate the research mechanism, not to reproduce all benchmark results.

Important limitations include:

- synthetic inputs are used in the public smoke tests;
- dataset-specific preprocessing is omitted;
- no trained paper checkpoints are distributed;
- the reference encoders are intentionally compact;
- psychological prototype values used in internal experiments are not distributed;
- affect labels are context-dependent and should not be interpreted as direct measurements of internal mental state.

## Out-of-scope use

This code should not be used to:

- infer sensitive personal characteristics;
- make employment, medical, legal, insurance, or other consequential decisions;
- identify individuals from motion;
- claim ground-truth emotional state from pose alone.

## Evaluation

The repository includes interface and shape tests plus synthetic inference / optimisation smoke tests.

These validate that the public implementation is executable and internally consistent. They are not substitutes for dataset-level evaluation, cross-subject validation, robustness testing, or calibration analysis.

## Relationship to the paper

Associated paper:

**Shuang Wu and Daniela M. Romano**  
*Privacy-preserving in-the-wild bodily expressed emotion recognition: A dual-teacher distillation framework with psychological priors*  
AI Open, 2026  
DOI: 10.1016/j.aiopen.2026.08.001

The code in this repository is a compact public reference implementation reconstructed around the published method and deployment interface. It is not represented as the exact internal codebase used for all experiments reported in the paper.
