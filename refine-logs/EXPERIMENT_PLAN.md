# Experiment Plan

**Problem**: Flow matching SR requires processing at HR resolution with x_t of varying information content across timesteps, but ATD's Token Dictionary is static — all timesteps share identical memory retrieval.
**Method Thesis**: Timestep-conditioned modulation of Token Dictionary (via AdaLN-Zero) + t-gated dual-branch shallow fusion enables adaptive memory retrieval and spatial processing for flow matching SR, improving quality without significant parameter overhead.
**Date**: 2026-04-11
**Architecture**: MPv2_arch (TC-ATD), 4.586M params, JiTModel, downscale=4 (256→64)

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| **C1 (Primary)**: Timestep-conditioned Token Dictionary (AdaLN on TD) improves flow matching SR | Core novelty — ATD从未被timestep条件化 | Ablation: remove AdaLN on TD → performance drops | B1, B3 |
| **C2 (Supporting)**: T-gated dual-branch shallow fusion provides complementary benefit | x_t和lq在不同timestep有不同信息价值 | Ablation: replace t-gate with simple add → drops | B1, B4 |
| **A1 ruled out**: Gain not from extra params alone | Reviewer会质疑是参数增加而非机制有效 | Parameter-matched variant without t-conditioning | B3 |
| **A2 ruled out**: Gain not from dual-branch alone | 需要证明t-gating本身有用 | Dual-branch without t-gate (constant 0.5 fusion) | B4 |

## Paper Storyline

**Main paper must prove:**
1. TC-ATD 在 4x SR 上优于 baseline 方法 (Table 1)
2. Timestep conditioning on TD 是提升的核心来源 (Table 2 ablation)
3. T-gated fusion 有互补贡献 (Table 2 ablation)

**Appendix can support:**
- 可视化：不同 timestep 下 Token Dictionary 的 attention pattern 变化
- 不同 downscale 倍率的效率分析
- 更多测试集结果

**Experiments intentionally cut:**
- Perceptual loss (LPIPS/FID) 作为 primary metric — 先关注 PSNR/SSIM
- 极大模型变体 — 单卡 4090 限制
- 视频SR/其他退化任务扩展 — 超出scope

## Experiment Blocks

### Block 1: Main Anchor Result (B1)
- **Claim tested**: C1 + C2 — full TC-ATD 优于所有 baseline
- **Why this block exists**: 论文核心，证明方法有效
- **Dataset / split / task**: DF2K train, Set5/Set14/Urban100/Manga109/B100 test, 4x SR
- **Compared systems**:
  - **ATD** (standard SR, no flow matching, pixelshuffledirect)
  - **MPv1_1** (ATD + PixelUnshuffle + SE, no timestep, same embed_dim/depths)
  - **JiT + UNet** (existing flow matching baseline)
  - **SwinIR** (strong non-generative SR baseline)
  - **TC-ATD (ours)** — MPv2_arch with temb_ch=256, downscale=4
- **Metrics**: PSNR, SSIM (Y channel), 参数量, FLOPs
- **Setup details**:
  - Single 4090, batch_size=4, patch=256×256 (HR)
  - 500K iter, AdamW lr=2e-4, MultiStepLR
  - L1 loss, ema_decay=0.999, fp16
  - ODE sampling: 5 steps (midpoint)
- **Success criterion**: TC-ATD PSNR ≥ baseline best + 0.1dB on ≥3/5 test sets
- **Failure interpretation**: 如果 TC-ATD 没有超过 baseline，说明 timestep conditioning 不够有效或训练不充分
- **Table / figure target**: Table 1 (main results)
- **Priority**: MUST-RUN

### Block 2: Sanity Check — Overfit on Small Split (B2)
- **Claim tested**: 模型能收敛、前向传播正确、loss 下降
- **Why this block exists**: 快速验证 pipeline，避免浪费 GPU 时间
- **Dataset / split / task**: DF2K 前 100 张图，1000 iter
- **Compared systems**: 仅 TC-ATD
- **Metrics**: train L1 loss
- **Setup details**: 同上，但 total_iter=1000
- **Success criterion**: L1 loss 稳定下降到 < 0.05
- **Failure interpretation**: 实现 bug（前向/损失/梯度）
- **Priority**: MUST-RUN

### Block 3: TC-ATD Ablation — Remove AdaLN on TD (B3)
- **Claim tested**: C1 — timestep conditioning on Token Dictionary 是核心贡献
- **Why this block exists**: 证明 AdaLN-Zero 调制 TD 有效，而非单纯参数增加
- **Dataset / split / task**: DF2K train, Set5 + Urban100 test
- **Compared systems**:
  - **TC-ATD (full)**: temb_ch=256, all components enabled
  - **w/o TC-ATD (temb_ch=0)**: 移除 timestep embedding，保留 dual-branch conv（无 t-gating）+ ATD with static TD
  - **w/ temb-add-only**: 保留 temb_proj (additive injection) 但移除 td_adaln (AdaLN on TD)
- **Metrics**: PSNR, SSIM (Y channel)
- **Setup details**: 同 B1，但在 2 个 test set 上评估即可
- **Success criterion**: w/o TC-ATD 下降 ≥ 0.15dB; w/ temb-add-only 介于两者之间
- **Failure interpretation**: 如果 temb-add-only 等于 full → AdaLN 不必要，additive 即可；如果 w/o 等于 full → timestep conditioning 无效
- **Table / figure target**: Table 2 (ablation)
- **Priority**: MUST-RUN

### Block 4: T-Gated Fusion Ablation (B4)
- **Claim tested**: C2 — t-gated dual-branch fusion 有互补贡献
- **Why this block exists**: 证明 t-dependent fusion 有意义，不是简单 concat 就够
- **Dataset / split / task**: DF2K train, Set5 + Urban100 test
- **Compared systems**:
  - **TC-ATD (full)**: t-gated fusion
  - **w/ simple-add**: feat_xt + feat_lr (no gate, no AdaLN on shallow)
  - **w/ concat**: cat(feat_xt, feat_lr) → Conv2d(2*embed_dim, embed_dim)
  - **w/ constant-gate**: gate=0.5 (fixed, learnable disabled)
- **Metrics**: PSNR, SSIM (Y channel)
- **Setup details**: 同 B3，在 2 个 test set 上
- **Success criterion**: Full > simple-add by ≥ 0.05dB; constant-gate 介于 full 和 simple-add 之间
- **Failure interpretation**: 如果 concat 等于 full → dual-branch 有用但 gating 不必要
- **Table / figure target**: Table 2 (ablation, second section)
- **Priority**: MUST-RUN

### Block 5: Downscale Factor Comparison (B5)
- **Claim tested**: downscale=4 的选择是合理的效率-性能权衡
- **Why this block exists**: 验证 spatial reduction 倍率的影响
- **Dataset / split / task**: DF2K train, Set5 test
- **Compared systems**: downscale ∈ {1, 2, 4, 8}
- **Metrics**: PSNR, SSIM, 推理时间, GPU 内存
- **Setup details**: 同 B1，单 test set
- **Success criterion**: downscale=4 接近最优效率-性能比
- **Failure interpretation**: downscale=2 显著更好 → 4x 压缩太多
- **Table / figure target**: Table 3 或 Appendix (efficiency analysis)
- **Priority**: NICE-TO-HAVE

### Block 6: Timestep Attention Visualization (B6)
- **Claim tested**: C1 — Token Dictionary 的 attention pattern 确实随 t 变化
- **Why this block exists**: 定性证据，让 reviewer 信服 TC-ATD 机制确实在工作
- **Dataset / split / task**: 几张 test 图，不同 t 值
- **Compared systems**: N/A (可视化分析)
- **Metrics**: sim_atd attention map 可视化
- **Setup details**: 选 2-3 张图，t ∈ {0.1, 0.3, 0.5, 0.7, 0.9}，可视化 ATD_CA 的 attention weights
- **Success criterion**: 不同 t 下 attention 分布有明显差异（t→0 偏向粗结构 token，t→1 偏向精细 token）
- **Failure interpretation**: attention 无差异 → TD modulation 可能被其他通路补偿了
- **Table / figure target**: Figure (qualitative)
- **Priority**: NICE-TO-HAVE

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost (GPU-h) | Risk |
|-----------|------|------|---------------|--------------|------|
| **M0** Sanity | 验证 pipeline 能跑通 | B2 (1000 iter overfit) | loss 下降？ | 0.5h | 低 |
| **M1** Main baseline | 跑 MPv1_1 baseline (no timestep) | B3 variant: temb_ch=0 | 结果记录作为 ablation 下界 | 20h | 中 |
| **M2** Full method | 跑 TC-ATD full | B1 (our method) | PSNR > baseline? | 24h | 中 |
| **M3** Ablation: TC-ATD | 移除 AdaLN on TD | B3 (w/o TC-ATD, w/ temb-add-only) | 证明 AdaLN 是关键 | 20h×2=40h | 低 |
| **M4** Ablation: Fusion | 移除 t-gating | B4 variants | 证明 t-gate 有贡献 | 20h×3=60h | 低 |
| **M5** Full test | 所有 test set 评估 | B1 补完所有 test set | 完成主表 | 额外 4h | 低 |
| **M6** Visualization | Attention 可视化 | B6 | 论文 figure | 1h | 低 |
| **M7** Downscale | 效率分析 | B5 (downscale sweep) | Appendix | 24h×3=72h | 可选 |

## Compute and Data Budget
- **Total estimated GPU-hours**: ~220h (must-run ~85h, nice-to-have ~135h)
- **Data**: DF2K (train), Set5/Set14/Urban100/Manga109/B100 (test) — 已在服务器上
- **Biggest bottleneck**: M3+M4 ablation 并行需多 GPU，但只有单卡 4090 → 串行执行

## Risks and Mitigations
- **R1: TC-ATD 没有超过 baseline**
  - Mitigation: 检查是否训练充分（增加 iter 或调 lr）；检查 temb_ch 是否足够
- **R2: AdaLN on TD 效果不明显**
  - Mitigation: 尝试更强的调制方式（FiLM 或 6-parameter DiT-style modulation）
- **R3: t-gated fusion 贡献微小**
  - Mitigation: 作为 supporting claim，不强求显著；论文可弱化此部分
- **R4: 训练不稳定（NaN/爆炸）**
  - Mitigation: AdaLN-Zero 已 zero-init；检查 fp16 数值范围

## Final Checklist
- [x] Main paper tables are covered (B1 main result, B3+B4 ablation)
- [x] Novelty is isolated (B3: AdaLN on TD ablation)
- [x] Simplicity is defended (B3: temb-add-only vs full AdaLN)
- [x] Dual-branch contribution is justified (B4: fusion variants)
- [x] Nice-to-have runs are separated (B5 downscale, B6 visualization)
