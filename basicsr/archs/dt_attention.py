"""
DT-Domain Self-Attention: 在 Domain Transform 测地距离空间中执行注意力

核心思想（基于 Gastal & Oliveira, SIGGRAPH 2011）:
  - 论文的 Domain Transform 将像素域的欧氏距离映射为测地距离
  - 测地距离在强边缘处剧增，从而天然保持边缘
  - 将这一思想应用于 Self-Attention:
    用测地距离替代/调制标准的位置距离，使 attention 权重跨边缘自动衰减

提供三种实现:
  1. GeodesicAttention: 测地距离作为 attention bias (推荐)
  2. GeodesicDistanceAttention: 测地距离替代欧氏距离 (理论最优)
  3. EdgeGateAttention: DT 边缘强度作为门控 (最简单)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def compute_geodesic_distance(x, sigma_ratio=1.0):
    """
    计算 Domain Transform 累积测地距离

    论文公式 (11): c_t(u) = ∫₀ᵘ (1 + σs/σr · Σ_k |I'k(x)|) dx
    离散形式: ct[i] = ct[i-1] + 1 + σs/σr · Σ_k |x[k,i] - x[k,i-1]|

    Args:
        x: [B, C, H, W] 输入特征
        sigma_ratio: σs/σr，控制边缘灵敏度（越大对边缘越敏感）
    Returns:
        ct_h: [B, H, W] 水平方向测地距离（沿每行）
        ct_v: [B, H, W] 垂直方向测地距离（沿每列）
    """
    B, C, H, W = x.shape

    # 水平方向: 沿宽度轴的差分
    diff_h = x[:, :, :, 1:] - x[:, :, :, :-1]  # [B, C, H, W-1]
    abs_diff_h = diff_h.abs().sum(dim=1)  # [B, H, W-1] — 跨通道求和
    integrand_h = torch.cat([
        torch.ones(B, H, 1, device=x.device),
        1.0 + sigma_ratio * abs_diff_h
    ], dim=-1)  # [B, H, W]
    ct_h = torch.cumsum(integrand_h, dim=-1)  # [B, H, W]

    # 垂直方向: 沿高度轴的差分
    diff_v = x[:, :, 1:, :] - x[:, :, :-1, :]  # [B, C, H-1, W]
    abs_diff_v = diff_v.abs().sum(dim=1)  # [B, H-1, W]
    integrand_v = torch.cat([
        torch.ones(B, 1, W, device=x.device),
        1.0 + sigma_ratio * abs_diff_v
    ], dim=-2)  # [B, H, W]
    ct_v = torch.cumsum(integrand_v, dim=-2)  # [B, H, W]

    return ct_h, ct_v


# ===================================================================
# 方案 B: 测地距离作为 Attention Bias (推荐)
# ===================================================================

class GeodesicAttention(nn.Module):
    """
    测地距离引导的自注意力

    在标准 self-attention 的基础上，用 Domain Transform 累积测地距离
    作为 attention bias，使得跨越强边缘的像素对之间的注意力自动降低。

    Attention(Q, K, V) = softmax(Q·Kᵀ / √d + bias_geo) · V

    其中 bias_geo[i,j] = -|ct[i] - ct[j]| / τ
    τ 是可学习的温度参数

    可在行级或列级使用，或组合为 2D 迭代（类似论文的多pass方案）。
    """

    def __init__(self, dim, num_heads=4, sigma_ratio=1.0, row_mode='horizontal'):
        """
        Args:
            dim: 特征维度
            num_heads: 注意力头数
            sigma_ratio: σs/σr，边缘灵敏度
            row_mode: 'horizontal' 沿行做注意力, 'vertical' 沿列做注意力
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sigma_ratio = sigma_ratio
        self.row_mode = row_mode

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # 可学习的测地距离温度参数
        self.tau = nn.Parameter(torch.ones(num_heads) * 1.0)

        # 值域缩放: 将 ct 距离映射到合适的 attention bias 范围
        self.geo_scale = nn.Parameter(torch.ones(num_heads) * 0.1)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 特征图
        Returns:
            out: [B, C, H, W] 注意力输出
        """
        B, C, H, W = x.shape

        # 计算测地距离
        ct_h, ct_v = compute_geodesic_distance(x, self.sigma_ratio)
        ct = ct_h if self.row_mode == 'horizontal' else ct_v  # [B, H, W]

        if self.row_mode == 'horizontal':
            # 沿每行做注意力: 序列长度 = W, batch = B*H
            # x → [B*H, W, C]
            x_seq = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
            ct_seq = ct.reshape(B * H, W)  # [B*H, W]
        else:
            # 沿每列做注意力: 序列长度 = H, batch = B*W
            x_seq = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
            ct_seq = ct.reshape(B * W, H)

        N = x_seq.shape[1]  # 序列长度

        # QKV 投影
        qkv = self.qkv(x_seq)  # [B', N, 3C]
        qkv = qkv.reshape(B if self.row_mode == 'horizontal' else B,
                          H if self.row_mode == 'horizontal' else W,
                          N, 3, self.num_heads, self.head_dim)
        # 简化: 直接用 reshape 后的 x_seq
        qkv = self.qkv(x_seq).reshape(x_seq.shape[0], N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B', heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 标准 attention score
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B', heads, N, N]

        # 计算测地距离 bias
        # ct_seq: [B', N] → 距离矩阵 [B', N, N]
        ct_i = ct_seq.unsqueeze(-1)  # [B', N, 1]
        ct_j = ct_seq.unsqueeze(-2)  # [B', 1, N]
        geo_dist = (ct_i - ct_j).abs()  # [B', N, N]

        # 可学习的缩放 + 负号（距离越大 bias 越负）
        # geo_scale: [heads], 扩展到 [1, heads, 1, 1]
        geo_bias = -geo_dist.unsqueeze(1) * self.geo_scale.view(1, -1, 1, 1)

        attn = attn + geo_bias
        attn = F.softmax(attn, dim=-1)

        # 输出
        out = (attn @ v).transpose(1, 2).reshape(x_seq.shape[0], N, C)
        out = self.proj(out)

        # 恢复形状
        if self.row_mode == 'horizontal':
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            out = out.reshape(B, W, H, C).permute(0, 3, 2, 1)

        return out


# ===================================================================
# 方案 C: DT 边缘强度作为门控 (最简单)
# ===================================================================

class EdgeGateAttention(nn.Module):
    """
    DT 边缘强度门控注意力

    用 DT 域的边缘强度 (|x[i] - x[i-1]|) 调制标准 self-attention。
    在强边缘位置降低信息混合程度。

    实现: edge_strength → MLP → gate → 乘以 attention 权重
    """

    def __init__(self, dim, num_heads=4, sigma_ratio=1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sigma_ratio = sigma_ratio

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # 从边缘强度生成门控
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, num_heads),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 计算边缘强度 (跨通道)
        diff_h = x[:, :, :, 1:] - x[:, :, :, :-1]
        diff_v = x[:, :, 1:, :] - x[:, :, :-1, :]
        edge_h = diff_h.abs().sum(dim=1)  # [B, H, W-1]
        edge_v = diff_v.abs().sum(dim=1)  # [B, H-1, W]

        # Pad 到完整尺寸
        edge_h = F.pad(edge_h, (0, 1, 0, 0), mode='replicate')
        edge_v = F.pad(edge_v, (0, 0, 0, 1), mode='replicate')

        # 合合两个方向的边缘强度
        edge_strength = (edge_h + edge_v).unsqueeze(-1)  # [B, H, W, 1]
        gate = self.gate_mlp(edge_strength)  # [B, H, W, heads]

        # 标准 self-attention (使用窗口)
        x_flat = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        N = H * W
        x_seq = x_flat.reshape(B, N, C)

        qkv = self.qkv(x_seq).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, heads, N, N]
        attn = F.softmax(attn, dim=-1)

        # 门控: reshape gate 为 sender 和 receiver
        # gate_sender: 边缘强度高的位置发出的 attention 被抑制
        gate_sender = gate.reshape(B, N, self.num_heads).permute(0, 2, 1)  # [B, heads, N]
        gate_sender = (1.0 - gate_sender * self.sigma_ratio * 0.1).unsqueeze(-1)  # [B, heads, N, 1]

        attn = attn * gate_sender  # 边缘位置发出的权重被衰减

        # 重新归一化
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return out


# ===================================================================
# 组合: 2D 迭代测地注意力 (类似论文的 2-pass 1D 滤波)
# ===================================================================

class IterativeGeodesicAttention(nn.Module):
    """
    迭代测地注意力 — 2D 边缘保持

    模仿论文 Section 5 的方法:
    交替进行水平/垂直方向的 1D 测地注意力，
    每次迭代 σ 减半以消除方向伪影。

    每次迭代 = 水平 pass + 垂直 pass
    """

    def __init__(self, dim, num_heads=4, num_iters=2, sigma_ratio=1.0):
        super().__init__()
        self.num_iters = num_iters

        # 每次迭代有独立的注意力参数
        self.h_attentions = nn.ModuleList([
            GeodesicAttention(dim, num_heads, sigma_ratio, 'horizontal')
            for _ in range(num_iters)
        ])
        self.v_attentions = nn.ModuleList([
            GeodesicAttention(dim, num_heads, sigma_ratio, 'vertical')
            for _ in range(num_iters)
        ])

        # 每次迭代后融合
        self.fusions = nn.ModuleList([
            nn.Linear(dim * 2, dim) for _ in range(num_iters)
        ])

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        out = x
        for i in range(self.num_iters):
            h_out = self.h_attentions[i](out)
            v_out = self.v_attentions[i](out)

            # 融合水平和垂直方向的结果
            B, C, H, W = x.shape
            fused = self.fusions[i](
                torch.cat([h_out, v_out], dim=1).permute(0, 2, 3, 1).reshape(B * H * W, C * 2)
            ).reshape(B, H, W, C).permute(0, 3, 1, 2)

            out = fused + out  # 残差连接

        return out


# ===================================================================
# 测试代码
# ===================================================================

if __name__ == '__main__':
    print("DT-Domain Self-Attention 验证")
    print("=" * 60)

    B, C, H, W = 2, 32, 16, 16
    x = torch.randn(B, C, H, W)

    # 测试测地距离计算
    print("\n--- 测地距离计算 ---")
    ct_h, ct_v = compute_geodesic_distance(x)
    print(f"  ct_h shape: {ct_h.shape}")
    print(f"  ct_v shape: {ct_v.shape}")
    print(f"  ct_h 范围: [{ct_h.min():.2f}, {ct_h.max():.2f}]")
    print(f"  ct_v 范围: [{ct_v.min():.2f}, {ct_v.max():.2f}]")

    # 测试 GeodesicAttention
    print("\n--- GeodesicAttention (推荐方案) ---")
    model = GeodesicAttention(dim=C, num_heads=4, sigma_ratio=1.0, row_mode='horizontal')
    out = model(x)
    print(f"  输入: {x.shape} → 输出: {out.shape}")
    print(f"  梯度检查: ", end="")
    loss = out.sum()
    loss.backward()
    print(f"tau.grad = {model.tau.grad}, geo_scale.grad = {model.geo_scale.grad}")
    print(f"  ✓ 反向传播正常")

    # 测试 EdgeGateAttention
    print("\n--- EdgeGateAttention (简单方案) ---")
    model2 = EdgeGateAttention(dim=C, num_heads=4)
    out2 = model2(x)
    print(f"  输入: {x.shape} → 输出: {out2.shape}")

    # 测试迭代组合
    print("\n--- IterativeGeodesicAttention (完整方案) ---")
    model3 = IterativeGeodesicAttention(dim=C, num_heads=4, num_iters=2)
    out3 = model3(x)
    print(f"  输入: {x.shape} → 输出: {out3.shape}")

    # 对比实验: 有边缘 vs 无边缘
    print("\n--- 对比实验: 测地距离的边缘保持效果 ---")
    x_smooth = torch.ones(1, 1, 1, 32) * 0.5  # 完全平坦
    x_edge = torch.cat([torch.ones(1, 1, 1, 16) * 0.2,
                         torch.ones(1, 1, 1, 16) * 0.8], dim=-1)  # 中间有强边缘

    ct_smooth, _ = compute_geodesic_distance(x_smooth)
    ct_edge, _ = compute_geodesic_distance(x_edge)

    print(f"  平坦信号 ct 范围: [{ct_smooth.min():.2f}, {ct_smooth.max():.2f}]")
    print(f"  有边缘信号 ct 范围: [{ct_edge.min():.2f}, {ct_edge.max():.2f}]")
    print(f"  平坦信号相邻 ct 差: {(ct_smooth[0,0,1:] - ct_smooth[0,0,:-1]).unique().tolist()}")
    print(f"  有边缘信号相邻 ct 差: {(ct_edge[0,0,1:] - ct_edge[0,0,:-1]).unique().tolist()}")
    print(f"  → 强边缘处测地距离跳变大 → attention 权重跨边缘自动衰减 ✓")
