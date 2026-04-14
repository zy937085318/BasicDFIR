# Research Idea Report

**Direction**: Timestep-adaptive architectures for flow matching super-resolution
**Generated**: 2026-04-11
**Ideas evaluated**: 10 generated → 5 survived filtering → 3 top ranked → recommended

## Landscape Summary

Flow matching has emerged as an efficient alternative to diffusion models for image super-resolution, offering straight-line transport paths that enable single-step or few-step inference. The core challenge in adapting flow matching to SR is the **resolution mismatch**: flow matching requires input/output at the same resolution, but SR needs LR→HR transformation. Current approaches handle this via (1) operating at HR resolution (expensive), (2) PixelUnshuffle into rearranged space (CAFlow), or (3) latent space processing (TADM, FluxSR).

ATD (CVPR 2024) introduced a learnable Token Dictionary with cross-attention for texture memory in SR, achieving SOTA results. However, it was designed for standard SR (not flow matching) and uses static dictionary tokens regardless of input characteristics.

The timestep-adaptive paradigm is emerging: TFDSR (IJCAI 2025) showed that different timesteps need different frequency-domain processing (early: phase/low-freq, late: amplitude/high-freq). TASR (arXiv 2024) demonstrated timestep-aware feature integration in ControlNet+SD. CAFlow showed that PixelUnshuffle reduces spatial compute by 16×. But **no one has combined timestep-adaptive spatial processing with texture memory in flow matching SR**.

### Key Gaps
1. **No timestep-conditioned spatial resolution adaptation** — TFDSR adapts frequency domain, CAFlow adapts per-image depth, but no one adapts spatial resolution based on timestep
2. **No timestep-aware memory in flow matching SR** — ATD's Token Dictionary is static; RestoreFormer/DMDNet have key-value memory but no timestep conditioning
3. **ATD has never been adapted for flow matching** — it exists only in standard SR and denoising contexts
4. **No coupling between spatial resolution and memory retrieval** — these are treated as independent design choices

## Recommended Ideas (ranked)

### Idea 1: Coupled Resolution-Memory Routing (CRMR) — RECOMMENDED
- **Hypothesis**: Spatial resolution and memory retrieval should be jointly adapted to timestep — at early timesteps (t→0, noise), low-res features query coarse memory tokens; at late timesteps (t→1, signal), high-res features query fine tokens
- **Method**: Multi-branch PixelUnshuffle (groups=1,4,16) with timestep-gated soft routing + AdaLN-modulated Token Dictionary, sharing the same timestep embedding
- **Minimum experiment**: Modify ATDTransformerLayer to accept timestep, add multi-branch shallow extraction with timestep gates, modulate td with (γ,β) from timestep. Train 100K iters on DF2K, compare PSNR/SSIM vs ATD baseline and vs FlowMatching+UNet
- **Expected outcome**: +0.3-0.8 dB PSNR over ATD baseline in flow matching mode; more significant gains at fewer ODE steps
- **Novelty**: 9/10 — closest work: TFDSR (frequency-domain timestep adaptation, different mechanism), CAFlow (per-image routing, not per-timestep), ATD (static dictionary). No prior work couples spatial resolution and memory based on timestep.
- **Feasibility**: ~18-24h on single 4090 for full training; pilot in 2-4h on DF2K subset
- **Risk**: MEDIUM — coupling two adaptive mechanisms may require careful tuning, but each component individually is proven
- **Contribution type**: New method + empirical finding (coupling effect)
- **Pilot result**: NEEDS PILOT — design below
- **Reviewer's likely objection**: "Is the coupling necessary, or would independent timestep adaptation suffice?" → Ablation study needed: compare coupled vs independent vs static
- **Why we should do this**: This is the most complete and novel idea — it addresses both research directions simultaneously and the coupling creates a natural story

### Idea 2: Timestep-Modulated Token Dictionary (TMTD)
- **Hypothesis**: ATD's static Token Dictionary provides identical texture priors regardless of timestep. Modulating dictionary tokens with timestep via AdaLN allows the memory to provide structure priors at early steps and texture priors at later steps
- **Method**: `td_t = td * (1 + γ(t)) + β(t)` where γ,β come from a 2-layer MLP on sinusoidal timestep embedding. Applied to all `self.td` parameters in ATD's BasicBlock
- **Minimum experiment**: Modify BasicBlock.forward to accept timestep, apply AdaLN modulation to td before cross-attention. Train 100K iters on DF2K in flow matching mode
- **Expected outcome**: +0.2-0.5 dB PSNR over static dictionary baseline; improved visual quality at fewer ODE steps
- **Novelty**: 8/10 — closest: ATD (static dict), AdaLN-Zero in DiT (modulates features, not memory tokens). No prior timestep-modulated memory tokens.
- **Feasibility**: Minimal code change (~20 lines in atd_arch.py), ~18h on 4090
- **Risk**: LOW — AdaLN modulation is well-established; applying to memory tokens is a natural extension
- **Contribution type**: New method
- **Pilot result**: NEEDS PILOT
- **Reviewer's likely objection**: "Why not just make the dictionary larger?" → Timestep modulation is parameter-efficient (adds ~0.1% params vs 2× dictionary)
- **Why we should do this**: Simplest path to a publishable result; low risk, clear contribution

### Idea 3: Timestep-Gated Resolution Routing (TGR)
- **Hypothesis**: Processing noisy x_t at full spatial resolution wastes computation. Timestep-conditioned soft routing between PixelUnshuffle branches (group=1,4,16) allocates computation proportional to information content
- **Method**: Replace fixed conv_first in MPv1_1 with 3 parallel grouped convolutions. Timestep embedding → MLP → softmax weights → weighted sum of branch outputs
- **Minimum experiment**: Replace shallow feature extraction in MPv1_1_arch, add TimestepEmbedder + routing MLP. Train 100K iters on DF2K
- **Expected outcome**: 2-4× FLOPs reduction at early timesteps with <0.1 dB quality loss; improved efficiency-quality tradeoff
- **Novelty**: 7/10 — closest: CAFlow (per-image early exit, not per-timestep spatial routing), MoE routing (similar mechanism but for experts not resolution). Timestep-based spatial routing is new.
- **Feasibility**: Moderate code change, ~18h on 4090
- **Risk**: LOW-MEDIUM — routing needs to learn meaningful patterns; may converge to single branch
- **Contribution type**: New method + efficiency improvement
- **Pilot result**: NEEDS PILOT
- **Reviewer's likely objection**: "Does the routing actually learn timestep-dependent patterns or collapse to a fixed branch?" → Visualize routing weights across timesteps
- **Why we should do this**: Addresses the core efficiency problem of flow matching SR

## Backup Ideas

### Idea 4: Timestep-Adaptive Category Size in AC_MSA (TACA)
- **Hypothesis**: ATD's fixed category_size=128 in AC_MSA is suboptimal across timesteps
- **Risk**: MEDIUM — soft grouping harder to optimize
- **Note**: Best explored as an ablation within Idea 1 (CRMR) rather than standalone

### Idea 5: Unified Timestep-Adaptive ATD Flow (UTA-ATD)
- The full system combining Ideas 1+2+3+4
- **Risk**: MEDIUM — many moving parts
- **Note**: This is the aspiration if individual components show promise

## Eliminated Ideas (for reference)
| Idea | Reason eliminated |
|------|-------------------|
| Flow Velocity-Guided Memory Retrieval (FVGMR) | Too speculative; velocity is noisy early in training; HIGH risk |
| Dynamic Group Conv via Timestep (DGCT) | Subsumed by Idea 3 (TGR) which provides a cleaner formulation |
| Timestep-Conditioned ConvFFN (TCC) | Too incremental; subsumed by Idea 2 (TMTD) |
| Timestep-Adaptive SE Gating (TASE) | Minor modification; low impact as standalone |

## Pilot Experiment Design

### Pilot 1: TMTD (Idea 2) — Simplest, lowest risk
- **Setup**: Modify ATD's BasicBlock to accept timestep, apply AdaLN to td
- **Scale**: 50K iterations on DF2K 128×128 patches, batch=8
- **Baseline**: ATD without timestep conditioning (static td)
- **Metric**: PSNR on Set5 4x SR (flow matching, 5 ODE steps)
- **Success criterion**: +0.15 dB over baseline
- **Estimated time**: 2-3 hours on 4090

### Pilot 2: TGR (Idea 3) — Medium complexity
- **Setup**: Replace conv_first with 3-branch grouped conv + timestep routing
- **Scale**: 50K iterations on DF2K 128×128 patches, batch=8
- **Baseline**: MPv1_1 with fixed groups=4
- **Metric**: PSNR + FLOPs on Set5
- **Success criterion**: ≥ baseline PSNR with <80% FLOPs, or +0.1 dB at same FLOPs
- **Estimated time**: 2-3 hours on 4090

### Pilot 3: CRMR (Idea 1) — Full concept
- **Setup**: Combine TMTD + TGR in single architecture
- **Scale**: 50K iterations on DF2K 128×128 patches, batch=8
- **Baseline**: Best of Pilot 1 and Pilot 2
- **Metric**: PSNR on Set5
- **Success criterion**: +0.1 dB over best individual component
- **Estimated time**: 3-4 hours on 4090

## Suggested Execution Order
1. **Pilot TMTD first** (Idea 2) — simplest, fastest to implement, validates timestep-memory interaction
2. **Pilot TGR** (Idea 3) — validates timestep-spatial resolution interaction
3. **Pilot CRMR** (Idea 1) — validates coupling; only if individual components show promise
4. If CRMR shows positive signal → scale up to full experiment → /auto-review-loop

## Next Steps
- [ ] Implement Pilot 1 (TMTD) — modify ATD's BasicBlock with timestep-modulated td
- [ ] Implement Pilot 2 (TGR) — modify MPv1_1 shallow extraction with timestep routing
- [ ] Deploy pilots to 4090 server
- [ ] Analyze results, select top idea
- [ ] Scale to full experiment (DF2K, 300K+ iters)
- [ ] Invoke /auto-review-loop for submission-ready iteration
