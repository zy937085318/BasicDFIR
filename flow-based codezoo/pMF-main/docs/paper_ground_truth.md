# pMF 论文 Ground Truth（Lu et al., 2026）

参考论文：Lu 等，*One-step Latent-free Image Generation with Pixel Mean Flows*（arXiv:2601.22158v1）。

本文件只记录论文中**明确写出**的公式/算法/实现细节，作为代码逐行核对的“唯一标尺”。论文未明确之处统一标为 TODO（并在代码侧同步加 TODO 注释）。

## 1. 基本记号与时间约定
- 时间：t∈[0,1]，t=1 为噪声端，t=0 为数据端。
- 采样：0≤r≤t≤1。

## 2. 线性插值与目标速度（Flow Matching 背景）
- 线性插值：z_t = (1 - t) x + t ε
- 条件速度：v = ε - x

## 3. pMF 的核心重参数化（x-prediction + v-loss）
### 3.1 网络输出空间（x 预测）
- 网络直接输出 x_θ(z_t, r, t)（“denoised-image-like quantity”）。
- 由网络输出构造平均速度场：
  - u_θ(z_t, r, t) = (z_t - x_θ(z_t, r, t)) / t

### 3.2 iMF / MeanFlow identity 与 stop-grad JVP
- MeanFlow identity（用于把 u 映射到 v 空间）：
  - v(z_t, t) = u(z_t, r, t) + (t - r) * d/dt u(z_t, r, t)
- iMF 的训练目标（论文将其作为实现参考）：
  - V_θ = u_θ + (t - r) * JVP_sg
  - 其中 JVP_sg 是对 d/dt u_θ 的 Jacobian-vector product 估计，并对该项 stop-gradient。

## 4. 训练算法（Algorithm 1）
### 4.1 训练伪代码（无 CFG）
论文 Algorithm 1 关键步骤（顺序与符号保持一致）：
- 采样 t, r = sample_t_r()
- 采样 ε ~ N(0, I)（shape 同 x）
- z = (1 - t) * x + t * ε
- 定义：
  - u_fn(z, r, t) = (z - net(z, r, t)) / t
- 计算：
  - v = u_fn(z, t, t)
  - (u, dudt) = jvp(u_fn, (z, r, t), (v, 0, 1))
  - V = u + (t - r) * stopgrad(dudt)
- 损失：
  - loss = metric(V, ε - x)

### 4.2 关键实现约束（用于逐行核对）
- jvp 的 tangents 中，z 方向的向量必须是 v = u_fn(z, t, t)，而不是直接用 (ε - x)。
- stop-gradient 应落在 dudt 上，而不是整个 V。

## 5. 训练算法（Algorithm 2：CFG 训练）
### 5.1 训练伪代码（带 CFG）
论文 Algorithm 2 关键步骤：
- 采样 t, r, w = sample_t_r_cfg()
- ε ~ N(0,I)，z = (1 - t) * x + t * ε
- u_fn(z, r, t) = (z - net(z, r, t)) / t
- v_c = u_fn(z, t, t, w, c)
- v_u = u_fn(z, t, t, w, None)
- v_g = (ε - x) + (1 - 1/w) * (v_c - v_u)
- (u, dudt) = jvp(u_fn, (z, r, t, w, c), (v_g, 0, 1, 0, 0))
- V = u + (t - r) * stopgrad(dudt)
- loss = metric(V, v_g)

### 5.2 关键实现约束（用于逐行核对）
- 网络输入必须支持 w 与 guidance interval conditioning（论文 Appendix A 明确要求“严格跟随 iMF 的 CFG 实现”）。
- jvp 的 z 方向 tangents 使用 v_g（而不是 v 或 ε-x）。

## 6. 感知损失（Perceptual Loss）
论文写法：
- 总损失：L = L_pMF + λ * L_perc
- L_perc 直接作用于 x_θ（网络输出的“denoised image”）与 ground-truth x。
- 仅在 t ≤ t_thr 时启用（避免过模糊）。

论文 Appendix A 额外实现细节：
- 在计算 perceptual loss 前，对生成图与 GT 都做 random crop，并 resize 到 224×224。
- 论文讨论了 VGG-LPIPS 与 ConvNeXt-V2 变体（具体实现细节在 Appendix A 与引用实现中）。

## 7. 时间采样器（time sampler）
论文主体强调在 (r, t) 平面覆盖 0≤r≤t 的区域；Appendix A 给出“长训练”设置：
- 使用 logit-normal(0.0, 0.8) 的时间采样（加大噪声尺度）。
- 为更平滑分布，以 10% 概率从 [0,1]×[0,1] 均匀采样 (t, r)（替代默认采样器的一部分）。
- 将感知损失阈值 t_thr 调整为 0.6。

## 8. EMA（Exponential Moving Average）
论文 Appendix A：
- EMA 跟随 EDM 的实现。
- 维护多个 EMA decay，并在推理时选择最佳 decay 对应的权重。

## 9. 推理与评估（与实现相关的明确点）
- 1-NFE（一步生成）是论文主张的核心属性之一。
- Appendix B 给出一组评估示例：CFG scale ω=7.0，CFG interval [0.1, 0.7]（用于某个报告的 FID/IS）。\n+
## 10. 论文未完全公开的实现细节（必须在代码侧加 TODO）
- Table 8 的完整超参表（例如精确的 warmup、LR schedule 细节、EMA decay 列表等）：HTML 版本在当前抓取中未包含表格内容，需从 PDF 或官方实现中精确抄录后再对齐。\n+
