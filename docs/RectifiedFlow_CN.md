# Rectified Flow 超分辨率方法

## 简介

Rectified Flow 是一种基于流匹配（Flow Matching）的图像生成和超分辨率方法。本项目将 Rectified Flow 集成到 BasicDFIR 框架中，用于图像超分辨率任务。

## 安装依赖

首先安装必要的依赖包：

```bash
pip install torchdiffeq einops scipy
```

可选依赖：
- `hyper-connections`: 用于多流残差连接（如果可用）

## 使用方法

### 训练

使用以下命令训练 Rectified Flow 模型：

```bash
python -m basicsr.train -opt options/train/RectifiedFlow/train_RectifiedFlow_x4.yml
```

### 测试

使用以下命令测试训练好的模型：

```bash
python -m basicsr.test -opt options/test/RectifiedFlow/test_RectifiedFlow_x4.yml
```

## 配置说明

### 网络架构配置

在配置文件的 `network_g` 部分：

```yaml
network_g:
  type: FlowUNet
  dim: 64                    # 基础维度
  init_dim: 64              # 初始维度
  channels: 3               # 输入通道数
  dim_mults: [1, 2, 4, 8]   # 维度倍数
  mean_variance_net: false  # 是否使用均值-方差网络
  dropout: 0.0              # Dropout 率
  attn_dim_head: 32         # 注意力头维度
  attn_heads: 4             # 注意力头数
  num_residual_streams: 2   # 残差流数量
```

### Rectified Flow 配置

在配置文件的 `rectified_flow` 部分：

```yaml
rectified_flow:
  time_cond_kwarg: 'times'           # 时间条件参数名
  predict: 'flow'                    # 预测目标：'flow' 或 'noise'
  noise_schedule: 'cosmap'           # 噪声调度：'cosmap' 或自定义函数
  use_consistency: false             # 是否使用一致性流匹配
  consistency_decay: 0.9999          # 一致性衰减率
  odeint_kwargs:                     # ODE 求解器参数
    atol: 1e-5
    rtol: 1e-5
    method: 'midpoint'
```

### 损失函数配置

支持以下损失函数：

1. **MSELoss** (默认)
```yaml
flow_opt:
  type: MSELoss
  loss_weight: 1.0
```

2. **PseudoHuberLoss**
```yaml
flow_opt:
  type: PseudoHuberLoss
  data_dim: 3
  loss_weight: 1.0
```

3. **PseudoHuberLossWithLPIPS** (需要 torchvision)
```yaml
flow_opt:
  type: PseudoHuberLossWithLPIPS
  data_dim: 3
  lpips_kwargs: {}
  loss_weight: 1.0
```

## 特性

### 1. 流匹配 (Flow Matching)

Rectified Flow 使用流匹配方法，通过 ODE 求解器从噪声生成高分辨率图像。

### 2. 一致性流匹配 (Consistency Flow Matching)

通过设置 `use_consistency: true` 启用一致性流匹配，可以提高生成质量和训练稳定性。

### 3. 多种噪声调度

支持余弦映射（cosmap）噪声调度，也可以自定义噪声调度函数。

### 4. 灵活的损失函数

支持多种损失函数，包括 MSE、Pseudo Huber 和结合 LPIPS 的损失。

## 注意事项

1. **内存使用**：Rectified Flow 模型通常需要较大的内存，建议使用较小的 batch size。

2. **采样时间**：使用 ODE 求解器进行采样可能需要较长时间，可以通过调整 `steps` 参数平衡质量和速度。

3. **依赖项**：
   - `torchdiffeq` 是必需的，用于 ODE 求解
   - `einops` 是必需的，用于张量操作
   - `scipy` 用于某些优化操作（可选）

4. **超参数调优**：
   - `dim` 和 `dim_mults` 控制网络容量
   - `odeint_kwargs` 中的 `atol` 和 `rtol` 影响采样精度
   - `consistency_decay` 影响一致性训练的稳定性

## 示例

### 基本训练示例

```yaml
# 使用默认配置训练 4x 超分辨率模型
model_type: RectifiedFlowModel
network_g:
  type: FlowUNet
  dim: 64
  channels: 3
rectified_flow:
  predict: 'flow'
  noise_schedule: 'cosmap'
```

### 使用一致性流匹配

```yaml
rectified_flow:
  use_consistency: true
  consistency_decay: 0.9999
  consistency_loss_weight: 1.0
```

## 参考文献

- Rectified Flow: A Marginal Preserving Approach to Optimal Transport
- Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion

## 故障排除

### 问题：导入错误

如果遇到 `einops` 或 `torchdiffeq` 导入错误，请确保已安装：

```bash
pip install einops torchdiffeq
```

### 问题：采样失败

如果采样失败，模型会自动回退到直接前向传播模式。确保 `data_shape` 在训练时被正确设置。

### 问题：内存不足

尝试：
- 减小 batch size
- 减小网络维度 `dim`
- 使用梯度累积




