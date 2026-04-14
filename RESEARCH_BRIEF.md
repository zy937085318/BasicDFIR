# Research Brief: Timestep-Adaptive Architectures for Flow Matching Super-Resolution

## Problem Statement

When adapting SR architectures (SwinIR, ATD) to flow matching methods, there is a fundamental structural mismatch:
- **Flow matching requires equal input/output resolution** (iterative refinement in pixel space)
- **SR tasks need LR→HR transformation** (e.g., 4x upscaling)
- **Naive solution (upsample LR→HR then process)** is computationally expensive and produces suboptimal results

## Baseline Architecture

- **ATD (Adaptive Token Dictionary)**: Uses learnable Token Dictionary + ATD_CA cross-attention + AC_MSA category attention + SW-MSA window attention. Has built-in texture memory mechanism.
- **Current flow matching pipeline**: `FlowMatchingModel` uses `AffineProbPath(CondOTScheduler)`, trains velocity prediction with `x_t` + bicubic LR as input at HR resolution.
- **JiTModel**: Uses x_1 prediction (endpoint prediction) instead of velocity prediction.
- **MPv1_1_arch**: Current attempt using PixelUnshuffle + SE + grouped Conv2d for shallow feature extraction at reduced resolution.

## Research Direction 1: Timestep-Adaptive Multi-Scale Feature Extraction

**Core insight**: In flow matching, the information content of `x_t` varies with timestep `t`:
- When `t → 0`: `x_t ≈ x_0` (pure noise) → less spatial information, more downsampling is acceptable → larger `group_num`
- When `t → 1`: `x_t ≈ x_1` (clean HR image) → rich spatial information, need finer processing → smaller `group_num`

**Current implementation (MPv1_1_arch.py)**:
- Uses `PixelUnshuffle(upscale)` to reduce resolution
- SE module for channel-wise recalibration
- Grouped Conv2d (`groups=upscale`) for shallow features
- **Limitation**: group_num is fixed, not adaptive to `t`

**Goal**: Design a module that dynamically adjusts feature extraction granularity based on `(t, x_t)`:
- Adaptive group convolution where `group_num` is a function of `t`
- Or: learnable routing that selects different resolution branches based on `t`
- Key constraint: must be differentiable and efficient

## Research Direction 2: Timestep-Aware Texture Memory for Flow Matching SR

**Core insight**: ATD's Token Dictionary serves as a fixed texture memory. In flow matching, the texture information needed changes with `t`:
- Early steps (`t ≈ 0`): need coarse structure/macro texture from memory
- Later steps (`t ≈ 1`): need fine detail/micro texture from memory

**Goal**: Design a timestep-conditioned memory module:
- Memory tokens or attention patterns that adapt to `t`
- Could be: `t`-gated memory retrieval, `t`-dependent token selection, or `t`-modulated similarity computation
- Must integrate cleanly with ATD's existing ATD_CA + AC_MSA framework
- Should be more parameter-efficient than simply scaling up the Token Dictionary

## Experimental Setup

- **Scale**: 4x SR (flow matching with upscale=1, LR upsampled to HR resolution via bicubic first)
- **Datasets**: DF2K (train), Set5/Set14/B100/Urban100/Manga109 (test)
- **Metrics**: PSNR, SSIM (Y channel)
- **Baseline comparison**: ATD, SwinIR, existing flow matching methods (FlowMatching, JiT)
- **GPU**: NVIDIA 4090 (remote server, SSH: `ssh -p 5122 ybb@10.16.104.29`)
- **Framework**: BasicSR with custom flow matching extensions

## Constraints

- Architecture uses `upscale=1` only (flow matching mode) — LR is bicubic-upsampled to HR resolution before entering the network
- Architecture must accept `(x, c=None, temp=None)` forward signature for flow matching compatibility
- Should not dramatically increase parameter count vs ATD baseline
- Training should complete within ~24 hours on single 4090
