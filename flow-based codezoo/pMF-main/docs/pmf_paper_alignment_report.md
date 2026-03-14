# pMF 论文一致性对比报告（逐模块）

论文：Lu 等，*One-step Latent-free Image Generation with Pixel Mean Flows*（2026, arXiv:2601.22158v1）  
仓库：/data2/private/huangcheng/pMF

本报告以 [paper_ground_truth.md](file:///data2/private/huangcheng/pMF/docs/paper_ground_truth.md) 为准，按模块列出：已验证一致项 / TODO 待确认项 / 偏差与修正建议（含代码定位）。

## 0. 审查范围（本轮覆盖的关键文件）
- 模型： [dit.py](file:///data2/private/huangcheng/pMF/src/pmf/dit.py)
- pMF 损失与采样： [pixel_mean_flow.py](file:///data2/private/huangcheng/pMF/src/pmf/pixel_mean_flow.py)
- 优化器： [optimizer.py](file:///data2/private/huangcheng/pMF/src/pmf/optimizer.py)
- 训练： [train.py](file:///data2/private/huangcheng/pMF/scripts/train.py)
- 推理： [eval.py](file:///data2/private/huangcheng/pMF/scripts/eval.py)
- 配置加载： [config.py](file:///data2/private/huangcheng/pMF/src/pmf/config.py)
- 配置： [pMF-B-16.yaml](file:///data2/private/huangcheng/pMF/configs/pMF-B-16.yaml)
- 数据： [dataset.py](file:///data2/private/huangcheng/pMF/src/pmf/dataset.py)

## 1. 模型结构（DiT / iMF-style）
### 1.1 已验证一致（与论文明确点对齐）
- **像素空间 ViT/DiT 结构**：patchify → Transformer blocks → unpatchify（[dit.py:L170-L283](file:///data2/private/huangcheng/pMF/src/pmf/dit.py#L170-L283)）
- **条件输入包含 t 与 r**：分别做 timestep embedding 后融合（[dit.py:L171-L277](file:///data2/private/huangcheng/pMF/src/pmf/dit.py#L171-L277)）

### 1.2 TODO 待确认（论文未明确或需对齐 iMF 官方实现）
- **CFG scale interval 的嵌入细节**：论文 Appendix A 仅说“网络条件输入 CFG scale interval”，未给出 interval 的 embedding 形式与融合方式；当前用 2→D MLP 占位实现（[dit.py:L173-L179](file:///data2/private/huangcheng/pMF/src/pmf/dit.py#L173-L179)）。
- TODO 需要从 iMF 实现/论文 Table 8/PDF 补齐：interval 的定义（[a,b] 还是离散桶）、采样分布、是否与 w 独立、是否需要额外投影/归一化。

## 2. 损失函数与 pMF 算法（Algorithm 1 / 2）
### 2.1 已验证一致（与论文明确点对齐）
- **线性插值**：z=(1-t)x+tε（[pixel_mean_flow.py:L134-L140](file:///data2/private/huangcheng/pMF/src/pmf/pixel_mean_flow.py#L134-L140)）
- **u_fn 定义**：u=(z-net)/t（[pixel_mean_flow.py:L150-L153](file:///data2/private/huangcheng/pMF/src/pmf/pixel_mean_flow.py#L150-L153)）
- **Algorithm 1 的 JVP 方向**：使用 v=u_fn(z,t,t) 作为 jvp 的 z 方向 tangents（[pixel_mean_flow.py:L166-L171](file:///data2/private/huangcheng/pMF/src/pmf/pixel_mean_flow.py#L166-L171)）
- **MeanFlow identity 近似**：V = u + (t-r)*stopgrad(dudt)（[pixel_mean_flow.py:L173-L177](file:///data2/private/huangcheng/pMF/src/pmf/pixel_mean_flow.py#L173-L177)）
- **主损失**：MSE(V, target)（target 根据是否 CFG 训练选择 v_target 或 v_g）（[pixel_mean_flow.py:L176-L178](file:///data2/private/huangcheng/pMF/src/pmf/pixel_mean_flow.py#L176-L178)）
- **感知损失阈值**：仅在 t≤t_thr 时启用（[pixel_mean_flow.py:L184-L196](file:///data2/private/huangcheng/pMF/src/pmf/pixel_mean_flow.py#L184-L196)）
- **感知损失的 crop+resize**：实现了 random crop + resize 到 224（[pixel_mean_flow.py:L27-L50](file:///data2/private/huangcheng/pMF/src/pmf/pixel_mean_flow.py#L27-L50)）

### 2.2 偏差（已修正）
- **关键偏差：JVP tangents 误用 ε-x**：原实现把 v_target=ε-x 当作 jvp 的 z 方向 tangents，这与论文 Algorithm 1/2 不符；已改为 Algorithm 1：v=u_fn(z,t,t)，Algorithm 2：v_g（见上方“已验证一致”条目）。

### 2.3 TODO 待确认（论文未明确或需对齐 iMF 官方实现）
- **(t,r) 的 10% uniform 采样细节**：论文 Appendix A 表述“以 10% 概率均匀采样 (t,r)”，未明确是否为三角域 0≤r≤t 的均匀；当前实现为 t 的 mixture + r|t~U[0,t]（[pixel_mean_flow.py:L61-L88](file:///data2/private/huangcheng/pMF/src/pmf/pixel_mean_flow.py#L61-L88)）。
- **CFG interval 与 w 的训练采样分布**：论文未给出 sample_t_r_cfg 的具体实现；当前实现为 w~U[w_min,w_max]，interval 在三角域采样（[pixel_mean_flow.py:L90-L103](file:///data2/private/huangcheng/pMF/src/pmf/pixel_mean_flow.py#L90-L103)）。
- **ConvNeXt-V2 perceptual loss**：论文 Appendix A 讨论了变体；当前仅实现 VGG-LPIPS（[pixel_mean_flow.py:L17-L17](file:///data2/private/huangcheng/pMF/src/pmf/pixel_mean_flow.py#L17-L17)）。

## 3. 训练循环（优化器 / 调度 / EMA / batch）
### 3.1 已验证一致（与论文明确点对齐）
- **Muon + AdamW 混合优化**：实现了 Muon 与 AdamW 分组并同时 step（[optimizer.py:L91-L141](file:///data2/private/huangcheng/pMF/src/pmf/optimizer.py#L91-L141)，[train.py:L112-L132](file:///data2/private/huangcheng/pMF/scripts/train.py#L112-L132)）
- **Cosine + warmup 结构**：训练脚本为两个 optimizer 各自创建 cosine schedule + warmup（[train.py:L179-L185](file:///data2/private/huangcheng/pMF/scripts/train.py#L179-L185)）
- **多 EMA decay 的维护与保存**：按配置维护多个 EMA，并在 checkpoint 时分别保存（[train.py:L150-L155](file:///data2/private/huangcheng/pMF/scripts/train.py#L150-L155)，[train.py:L223-L234](file:///data2/private/huangcheng/pMF/scripts/train.py#L223-L234)）

### 3.2 偏差（已修正）
- **配置文件未被实际加载**：默认找不到 configs/config.yaml 会回退默认值，导致训练超参与 YAML 不一致；已改为默认优先加载 configs/pMF-B-16.yaml，并支持 PMF_CONFIG / CLI 指定（[config.py:L68-L87](file:///data2/private/huangcheng/pMF/src/pmf/config.py#L68-L87)，[train.py:L77-L86](file:///data2/private/huangcheng/pMF/scripts/train.py#L77-L86)）。
- **梯度累积与 global batch**：训练脚本现在会根据 WORLD_SIZE、micro_batch_size 与 global_batch_size 计算/提示需要的 gradient_accumulation_steps（[train.py:L80-L102](file:///data2/private/huangcheng/pMF/scripts/train.py#L80-L102)）。

### 3.3 TODO 待确认（论文未明确或需从 Table 8 精确抄录）
- **warmup_steps 与调度细节**：论文 Table 8 的精确值需从 PDF/官方实现抄录；当前仍使用配置中的占位值（[config.py:L49-L53](file:///data2/private/huangcheng/pMF/src/pmf/config.py#L49-L53)，[train.py:L157-L165](file:///data2/private/huangcheng/pMF/scripts/train.py#L157-L165)）。
- **EMA decay 列表**：论文仅说明“维护多个 EMA decay 并选择最佳”，未给出 decay 列表；当前在配置中给出候选值（[pMF-B-16.yaml:L17-L25](file:///data2/private/huangcheng/pMF/configs/pMF-B-16.yaml#L17-L25)）。
## 4. 推理（1-NFE + CFG）
### 4.1 已验证一致（与论文明确点对齐）
- **1-NFE 一步生成**：固定 t=1,r=0 生成 x(z1,0,1)（[pixel_mean_flow.py:L205-L246](file:///data2/private/huangcheng/pMF/src/pmf/pixel_mean_flow.py#L205-L246)）
- **CFG interval 输入通路**：eval 脚本支持传入 cfg_interval 并喂给模型（[eval.py:L43-L63](file:///data2/private/huangcheng/pMF/scripts/eval.py#L43-L63)）

### 4.2 TODO 待确认（论文未明确或需对齐 iMF 官方实现）
- **CFG 的推理组合公式**：论文 Appendix A 强调“严格跟随 iMF 的 CFG 实现”，但推理侧的精确组合/interval 语义在本文中未完整展开；当前推理实现为两次前向（cond/uncond）并在 x 空间线性合成，同时把 w 与 interval 作为条件输入（[pixel_mean_flow.py:L228-L245](file:///data2/private/huangcheng/pMF/src/pmf/pixel_mean_flow.py#L228-L245)）。
- TODO 需要从 iMF 推理代码核对：是否应在 v/u 空间做 guidance、interval 在 1-NFE 时是否仍有作用、以及最佳 w/interval 的选择流程（论文 Appendix B 给出示例 w=7.0, interval=[0.1,0.7]）。
## 5. 数据预处理 / 后处理
### 5.1 已验证一致（与论文“像素空间训练”常见做法相容）
- **归一化到 [-1,1]**：Normalize(mean=0.5,std=0.5)（[dataset.py:L63-L69](file:///data2/private/huangcheng/pMF/src/pmf/dataset.py#L63-L69)）
- **训练增强**：RandomResizedCrop + HorizontalFlip（[dataset.py:L63-L69](file:///data2/private/huangcheng/pMF/src/pmf/dataset.py#L63-L69)）
- **推理输出后处理**：[-1,1]→[0,1] 并 clamp（[eval.py:L58-L61](file:///data2/private/huangcheng/pMF/scripts/eval.py#L58-L61)）

### 5.2 TODO 待确认
- **额外增强（如 ColorJitter）**：论文未在正文/Appendix A 明确；当前未启用（[dataset.py:L62-L69](file:///data2/private/huangcheng/pMF/src/pmf/dataset.py#L62-L69)）。
## 6. 总结与下一步建议
- 当前实现已在 **pMF 核心数学（Algorithm 1/2 的 jvp 方向、V 构造、perceptual crop+resize）**、**CFG 条件输入通路**、**多 EMA 保存**、**配置加载与训练脚本可复现性** 等关键处对齐论文明确表述。
- 剩余必须精确对齐且论文文本不足的部分主要集中在：**Table 8 超参表** 与 **iMF 官方 CFG/interval 语义**。建议优先补齐 Table 8 的原文数值（或对齐官方代码），再把本仓库的 TODO 一次性逐项消除。
## 7. 全部 TODO(pMF) 清单（需进一步确认）
- [pixel_mean_flow.py:L22](file:///data2/private/huangcheng/pMF/src/pmf/pixel_mean_flow.py#L22)：ConvNeXt-V2 perceptual loss 变体未实现
- [pixel_mean_flow.py:L72](file:///data2/private/huangcheng/pMF/src/pmf/pixel_mean_flow.py#L72)：(t,r) 的 10% uniform 采样是否为三角域均匀
- [pixel_mean_flow.py:L106](file:///data2/private/huangcheng/pMF/src/pmf/pixel_mean_flow.py#L106)：CFG interval 训练采样分布未公开
- [config.py:L54-L56](file:///data2/private/huangcheng/pMF/src/pmf/config.py#L54-L56)：EMA decay 列表与 warmup/schedule 需从 Table 8/官方实现抄录
- [dit.py:L174](file:///data2/private/huangcheng/pMF/src/pmf/dit.py#L174)：CFG scale interval 的 embedding 形式占位实现
