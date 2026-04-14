---
type: idea
node_id: idea:001
title: "TC-ATD: Timestep-Conditioned Adaptive Token Dictionary"
stage: implemented
outcome: pending
origin_skill: idea-creator
created_at: 2026-04-11
updated_at: 2026-04-11
target_gaps: [G1, G2, G3, G4]
---

# One-line thesis

Make ATD's Token Dictionary timestep-aware via AdaLN-Zero modulation + t-gated dual-branch shallow fusion for flow matching SR.

## Innovation 1: Timestep-Conditioned Token Dictionary

- Apply AdaLN-Zero to Token Dictionary before cross-attention
- td_mod = AdaLN(td, temb) → ATD_CA uses modulated TD → attention patterns adapt to t
- AC_MSA categories auto-adapt because tk_id derives from modulated sim_atd
- Additive temb injection per layer

## Innovation 2: Timestep-Adaptive Shallow Feature Extraction

- Dual-branch conv: conv_xt (x_t branch) + conv_lr (lq_bicubic branch)
- PixelUnshuffle(downscale) on each branch for spatial reduction
- t-gated fusion: gate = sigmoid(Linear(temb)), t→0 focus on LR, t→1 focus on x_t
- AdaLN-Zero on fused features

## Architecture

- MPv2_arch.py, 4.586M params (downscale=4)
- JiTModel wrapper, upscale=1, DF2K training, 4x SR
- Forward signature: (x, c=None, temp=None)

## Inspired By

- paper:atd2024 (base architecture)
- V-HMN (hierarchical Hopfield memory)
- MEMO (memory-guided diffusion)
- SMNet (staged memory for denoising)
- M2ADL (dynamic memory augmented dictionary learning)

## Experiments

exp:001 (planned)

## Failure Notes

(Not yet tested)
