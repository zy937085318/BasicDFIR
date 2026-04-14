"""
Helmholtz Beltrami Flow UNet v3 for image super-resolution.

Core Design (基于 Helmholtz 分解):
  v_total = v_curl_free + v_solenoidal

  1. v_curl_free = ∇φ (curl-free by construction)
     - 由 LR bicubic 引导
     - 负责平滑区域 / 全局插值

  2. v_solenoidal = curl(ψ) (div-free by construction)
     - 由深度图 + 语义特征引导
     - 负责边缘 / 纹理区域 / 局部涡旋

  3. Beltrami 作为 soft regularization loss:
     L_B = ||∇×v - λ·v||² + ||∇·v||² + ||∇×v||²
     (不强制约束，只作为正则)

Key Innovation:
  - 通过 Helmholtz 分解将速度场预测分解为两个物理意义互补的分量
  - curl-free 分支：保证全局平滑性
  - solenoidal 分支：保证局部涡旋性
  - 与 JiT Flow Matching 完全兼容

Reference:
  - Helmholtz Decomposition: ∇²φ = ∇·v, ∇²ψ = ∇×v
  - Beltrami Flow: ∇×v = λ·v (作为软正则)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_arch import (
    Swish, group_norm, upsample, downsample, ResidualBlock,
    SelfAttention, TimestepEmbedding, Dual_TimestepEmbedding,
    conv2d,
)
from basicsr.utils.registry import ARCH_REGISTRY


def curl_free_gradient(phi):
    """
    Compute curl-free velocity: v = ∇φ
    Automatically satisfies ∇×(∇φ) = 0.

    Args:
        phi: (B, 1, H, W) scalar potential
    Returns:
        v: (B, 2, H, W) velocity field (u_x, v_y)
    """
    # ∂φ/∂x, ∂φ/∂y
    dphi_dx = torch.gradient(phi, dim=3)[0]
    dphi_dy = torch.gradient(phi, dim=2)[0]
    # Handle PyTorch 2.x tuple return
    if isinstance(dphi_dx, tuple):
        dphi_dx = dphi_dx[0]
    if isinstance(dphi_dy, tuple):
        dphi_dy = dphi_dy[0]
    return torch.cat([dphi_dx, dphi_dy], dim=1)


def solenoidal_curl(psi):
    """
    Compute solenoidal velocity: v = ∇ × ψ
    In 2D, this gives a div-free field: v = (-∂ψ/∂y, ∂ψ/∂x)
    Automatically satisfies ∇·(∇×ψ) = 0.

    Args:
        psi: (B, 1, H, W) scalar stream function
    Returns:
        v: (B, 2, H, W) velocity field (u_x, v_y)
    """
    # ∂ψ/∂x, ∂ψ/∂y
    dpsi_dx = torch.gradient(psi, dim=3)[0]
    dpsi_dy = torch.gradient(psi, dim=2)[0]
    if isinstance(dpsi_dx, tuple):
        dpsi_dx = dpsi_dx[0]
    if isinstance(dpsi_dy, tuple):
        dpsi_dy = dpsi_dy[0]
    # v_x = -∂ψ/∂y, v_y = ∂ψ/∂x  (div-free by construction)
    return torch.cat([-dpsi_dy, dpsi_dx], dim=1)


def helmholtz_divergence(v):
    """
    Compute 2D divergence: ∇·v = ∂v_x/∂x + ∂v_y/∂y
    Args:
        v: (B, 2, H, W)
    Returns:
        div: (B, 1, H, W)
    """
    dvx_dx = torch.gradient(v[:, 0:1], dim=3)[0]
    dvy_dy = torch.gradient(v[:, 1:2], dim=2)[0]
    if isinstance(dvx_dx, tuple):
        dvx_dx = dvx_dx[0]
    if isinstance(dvy_dy, tuple):
        dvy_dy = dvy_dy[0]
    return dvx_dx + dvy_dy


def helmholtz_curl(v):
    """
    Compute 2D curl (scalar): ∇×v = ∂v_y/∂x - ∂v_x/∂y
    Args:
        v: (B, 2, H, W)
    Returns:
        curl: (B, 1, H, W)
    """
    dvy_dx = torch.gradient(v[:, 1:2], dim=3)[0]
    dvx_dy = torch.gradient(v[:, 0:1], dim=2)[0]
    if isinstance(dvy_dx, tuple):
        dvy_dx = dvy_dx[0]
    if isinstance(dvx_dy, tuple):
        dvx_dy = dvx_dy[0]
    return dvy_dx - dvx_dy


class DepthEstimator(nn.Module):
    """
    Lightweight depth estimation branch.
    Predicts a pseudo-depth map from the LR image.
    This is a simplified depth estimator - can be supervised with GT depth if available.
    """

    def __init__(self, in_channels=3, hidden_channels=16, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            Swish(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),  # H/2
            Swish(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, padding=1),
            Swish(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, stride=2, padding=1),  # H/4
            Swish(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
            Swish(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            Swish(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image
        Returns:
            depth: (B, 1, H, W) predicted depth map (relative, not metric)
        """
        return self.decoder(self.encoder(x))


class ScalarPotentialBranch(nn.Module):
    """
    Predicts scalar potential φ for curl-free velocity: v = ∇φ
    Driven by LR condition + current state x_t.
    """

    def __init__(self, in_channels, hidden_channels=64, temb_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            Swish(),
            ResidualBlock(in_ch=hidden_channels, temb_ch=temb_channels, out_ch=hidden_channels),
            ResidualBlock(in_ch=hidden_channels, temb_ch=temb_channels, out_ch=hidden_channels),
            nn.Conv2d(hidden_channels, 1, 1),  # Output φ
        )

    def forward(self, feat, lq_feat, temb):
        """
        Args:
            feat: (B, C, H, W) UNet bottleneck features from x_t
            lq_feat: (B, C, H, W) downsampled LR features
            temb: (B, temb_ch) timestep embedding
        Returns:
            phi: (B, 1, H, W) scalar potential
        """
        x = torch.cat([feat, lq_feat], dim=1)
        phi = self.net(x)
        return phi


class StreamFunctionBranch(nn.Module):
    """
    Predicts stream function ψ for solenoidal velocity: v = ∇×ψ
    Driven by depth map + semantic features.
    """

    def __init__(self, in_channels, hidden_channels=64, temb_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden_channels, 3, padding=1),  # +1 for depth
            Swish(),
            ResidualBlock(in_ch=hidden_channels, temb_ch=temb_channels, out_ch=hidden_channels),
            ResidualBlock(in_ch=hidden_channels, temb_ch=temb_channels, out_ch=hidden_channels),
            nn.Conv2d(hidden_channels, 1, 1),  # Output ψ
        )

    def forward(self, feat, depth, temb):
        """
        Args:
            feat: (B, C, H, W) UNet bottleneck features from x_t
            depth: (B, 1, H, W) estimated or GT depth map
            temb: (B, temb_ch) timestep embedding
        Returns:
            psi: (B, 1, H, W) stream function
        """
        x = torch.cat([feat, depth], dim=1)
        psi = self.net(x)
        return psi


class HelmholtzVelocityPredictor(nn.Module):
    """
    Predicts velocity field via Helmholtz decomposition:
      v = v_curl_free + v_solenoidal
        = ∇φ(lq) + ∇×ψ(depth)

    Both components are computed at multiple resolutions for multi-scale flow.
    """

    def __init__(self, in_channels, lq_channels, depth_channels=1,
                 hidden_channels=64, temb_channels=128, num_scales=2):
        super().__init__()
        self.num_scales = num_scales

        # Branch 1: curl-free from LR
        self.phi_branch = ScalarPotentialBranch(
            in_channels=in_channels + lq_channels,
            hidden_channels=hidden_channels,
            temb_channels=temb_channels,
        )

        # Branch 2: solenoidal from depth
        self.psi_branch = StreamFunctionBranch(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            temb_channels=temb_channels,
        )

        # Channel adapter for lq features
        self.lq_adapter = nn.Sequential(
            nn.Conv2d(lq_channels, in_channels, 1),
            Swish(),
        )

        # Modulation gate: learns to balance curl-free vs solenoidal contributions
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, feat, lq_feat, depth, temb):
        """
        Args:
            feat: (B, C, H, W) UNet bottleneck features
            lq_feat: (B, lq_C, H, W) LR features (downsampled to feat resolution)
            depth: (B, 1, H, W) depth map (estimated or GT)
            temb: (B, temb_ch) timestep embedding
        Returns:
            v_total: (B, 2, H, W) combined velocity field
            v_curl_free: (B, 2, H, W) curl-free component
            v_solenoidal: (B, 2, H, W) solenoidal component
        """
        # Adapt lq features to match feat channels
        lq_adapted = self.lq_adapter(lq_feat)

        # Component 1: curl-free v = ∇φ (from LR)
        phi = self.phi_branch(feat, lq_adapted, temb)
        v_curl_free = curl_free_gradient(phi)  # (B, 2, H, W)

        # Component 2: solenoidal v = ∇×ψ (from depth)
        psi = self.psi_branch(feat, depth, temb)
        v_solenoidal = solenoidal_curl(psi)  # (B, 2, H, W)

        # Combine: v_total = (1-α)·v_curl_free + α·v_solenoidal
        alpha = torch.sigmoid(self.alpha)  # ∈ [0, 1]
        v_total = (1 - alpha) * v_curl_free + alpha * v_solenoidal

        return v_total, v_curl_free, v_solenoidal


@ARCH_REGISTRY.register()
class HelmholtzUNetv3(nn.Module):
    """
    Helmholtz-Beltrami Flow UNet for JiT-style image super-resolution.

    Architecture:
      1. Shared UNet encoder-decoder backbone (predicts intermediate features)
      2. Helmholtz Velocity Predictor:
         - Curl-free branch: v = ∇φ, guided by LR condition
         - Solenoidal branch: v = ∇×ψ, guided by depth estimation
      3. Final output: x_1_pred from velocity (Euler integration)

    The network outputs x_1 directly but is trained with Helmholtz decomposition
    loss, providing physical priors on the velocity field structure.
    """

    def __init__(self,
                 input_channels,
                 img_size,
                 ch=32,
                 output_channels=3,
                 ch_mult=(1, 2, 4, 8),
                 num_res_blocks=2,
                 attn_resolutions=(16,),
                 dropout=0.,
                 resamp_with_conv=True,
                 act=Swish(),
                 normalize=group_norm,
                 meanflow=False,
                 check_point=None,
                 helmholtz_hidden_channels=64,
                 helmholtz_alpha_init=0.3,
                 enable_depth_branch=True,
                 enable_helmholtz=True,
                 ):
        super().__init__()
        self.input_channels = input_channels
        self.input_height = img_size
        self.ch = ch
        self.output_channels = output_channels
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv
        self.enable_helmholtz = enable_helmholtz
        self.enable_depth_branch = enable_depth_branch

        # Resolve string act/normalize
        if isinstance(act, str):
            if act.lower() == 'swish':
                act = Swish()
            elif act.lower() == 'relu':
                act = torch.nn.ReLU(inplace=True)
            elif act.lower() == 'gelu':
                act = torch.nn.GELU()
            else:
                raise ValueError(f"Unknown act: {act}")
        if isinstance(normalize, str):
            if normalize.lower() == 'group_norm':
                normalize = group_norm
        self.act = act
        self.normalize = normalize

        # init
        self.num_resolutions = num_resolutions = len(ch_mult)
        in_ht = self.input_height
        in_ch = input_channels
        temb_ch = ch * 4
        assert in_ht % 2 ** (num_resolutions - 1) == 0

        # Timestep embedding
        self.meanflow = meanflow
        if self.meanflow:
            self.temb_net = Dual_TimestepEmbedding(
                embedding_dim=ch, hidden_dim=temb_ch, output_dim=temb_ch, act=act)
        else:
            self.temb_net = TimestepEmbedding(
                embedding_dim=ch, hidden_dim=temb_ch, output_dim=temb_ch, act=act)

        # Encoder
        self.first_conv = conv2d(in_ch, ch)
        unet_chs = [ch]
        down_modules = []
        for i_level in range(num_resolutions):
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            in_ch = ch
            for i_block in range(num_res_blocks):
                block_modules[f'{i_level}a_{i_block}a_block'] = ResidualBlock(
                    in_ch=in_ch, temb_ch=temb_ch, out_ch=out_ch,
                    dropout=dropout, act=act, normalize=normalize)
                if in_ht in attn_resolutions:
                    block_modules[f'{i_level}a_{i_block}b_attn'] = SelfAttention(
                        out_ch, normalize=normalize)
                unet_chs += [out_ch]
                in_ch = out_ch
            if i_level != num_resolutions - 1:
                block_modules[f'{i_level}b_downsample'] = downsample(
                    out_ch, with_conv=resamp_with_conv)
                in_ht //= 2
                unet_chs += [out_ch]
            down_modules += [nn.ModuleDict(block_modules)]
        self.down_modules = nn.ModuleList(down_modules)

        # Middle
        mid_modules = []
        mid_modules += [ResidualBlock(in_ch, temb_ch=temb_ch, out_ch=in_ch,
                                      dropout=dropout, act=act, normalize=normalize)]
        mid_modules += [SelfAttention(in_ch, normalize=normalize)]
        mid_modules += [ResidualBlock(in_ch, temb_ch=temb_ch, out_ch=in_ch,
                                      dropout=dropout, act=act, normalize=normalize)]
        self.mid_modules = nn.ModuleList(mid_modules)

        # Bottleneck channels
        self.bottleneck_ch = in_ch  # ch * ch_mult[-1]

        # Depth estimator
        if self.enable_depth_branch:
            self.depth_estimator = DepthEstimator(
                in_channels=3,  # input RGB
                hidden_channels=16,
                out_channels=1,
            )

        # Helmholtz velocity predictor
        if self.enable_helmholtz:
            self.helmholtz_predictor = HelmholtzVelocityPredictor(
                in_channels=self.bottleneck_ch,
                lq_channels=ch,  # first conv output channels for lq
                depth_channels=1,
                hidden_channels=helmholtz_hidden_channels,
                temb_channels=temb_ch,
                num_scales=1,
            )

        # Decoder
        up_modules = []
        for i_level in reversed(range(num_resolutions)):
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                block_modules[f'{i_level}a_{i_block}a_block'] = ResidualBlock(
                    in_ch=in_ch + unet_chs.pop(), temb_ch=temb_ch, out_ch=out_ch,
                    dropout=dropout, act=act, normalize=normalize)
                if in_ht in attn_resolutions:
                    block_modules[f'{i_level}a_{i_block}b_attn'] = SelfAttention(
                        out_ch, normalize=normalize)
                in_ch = out_ch
            if i_level != 0:
                block_modules[f'{i_level}b_upsample'] = upsample(
                    out_ch, with_conv=resamp_with_conv)
                in_ht *= 2
            up_modules += [nn.ModuleDict(block_modules)]
        self.up_modules = nn.ModuleList(up_modules)
        assert not unet_chs

        # Output: predict x_1 (3 channels)
        self.end_conv = nn.Sequential(
            normalize(in_ch),
            act,
            conv2d(in_ch, output_channels, init_scale=0.),
        )

        # LR encoder (separate, lighter encoder for LR condition)
        self.lq_first_conv = conv2d(3, ch)  # lq is always RGB 3 channels
        self.lq_down_modules = nn.ModuleList(down_modules[:num_resolutions])

        if check_point is not None:
            checkpoint = torch.load(check_point, map_location='cpu')
            self.load_state_dict(checkpoint, strict=False)

    def _compute_cond_module(self, module, x, temp):
        for m in module:
            x = m(x, temp)
        return x

    def _has_attn(self, block_modules, key):
        return key in block_modules

    def forward(self, x, c=None, temp=None):
        """
        Forward pass compatible with JiT framework.

        Args:
            x: (B, 6, H, W) concatenated [x_t, lq_bicubic]
            c: condition (unused)
            temp: (B,) timestep t in [0, 1]

        Returns:
            x_1_pred: (B, 3, H, W) predicted target image
        """
        B, _, H, W = x.size()
        x_t = x[:, :3]    # (B, 3, H, W)
        lq = x[:, 3:]     # (B, 3, H, W)

        # Timestep embedding
        if temp is None:
            temp = torch.ones(B).to(x.device)
        if self.meanflow:
            temp_t, remp = temp[:, 0:1, ...].squeeze(), temp[:, 1:, ...].squeeze()
            temb = self.temb_net(temp_t, remp)
        else:
            temb = self.temb_net(temp)
        assert list(temb.shape) == [B, self.ch * 4]

        # ================================================================
        # Encode x_t (main encoder)
        # ================================================================
        hs = [self.first_conv(x)] #hs->[B, 32, H, W]

        for i_level in range(self.num_resolutions):
            block_modules = self.down_modules[i_level]
            for i_block in range(self.num_res_blocks):
                print("i_level:{}, i_block:{}".format(i_level, i_block))
                resnet_block = block_modules[f'{i_level}a_{i_block}a_block']
                h = resnet_block(hs[-1], temb)
                attn_key = f'{i_level}a_{i_block}b_attn'
                if h.size(2) in self.attn_resolutions and attn_key in block_modules:
                    h = block_modules[attn_key](h, temb)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(block_modules[f'{i_level}b_downsample'](hs[-1]))

        # Middle
        h = hs[-1]
        h = self._compute_cond_module(self.mid_modules, h, temb)
        bottleneck_feat = h  # (B, bottleneck_ch, H/32, W/32)

        # ================================================================
        # Encode lq (lightweight encoder for LR condition)
        # ================================================================
        lq_hs = [self.lq_first_conv(lq)]
        for i_level in range(self.num_resolutions):
            block_modules = self.lq_down_modules[i_level]
            for i_block in range(self.num_res_blocks):
                resnet_block = block_modules[f'{i_level}a_{i_block}a_block']
                lq_h = resnet_block(lq_hs[-1], temb)
                lq_hs.append(lq_h)
            if i_level != self.num_resolutions - 1:
                lq_hs.append(block_modules[f'{i_level}b_downsample'](lq_hs[-1]))

        # Get lq bottleneck features (downsampled to match x_t bottleneck resolution)
        lq_bottleneck = lq_hs[-1]
        lq_bottleneck = F.interpolate(
            lq_bottleneck, size=bottleneck_feat.shape[2:], mode='bilinear', align_corners=False
        )

        # ================================================================
        # Depth estimation (from LR)
        # ================================================================
        if self.enable_depth_branch:
            depth = self.depth_estimator(lq)  # (B, 1, H, W)
            depth = torch.sigmoid(depth)  # Normalize to [0, 1]
            # Downsample depth to bottleneck resolution
            depth_bottleneck = F.interpolate(
                depth, size=bottleneck_feat.shape[2:], mode='bilinear', align_corners=False
            )
        else:
            depth = torch.ones(B, 1, H, W).to(x.device) * 0.5  # uniform depth
            depth_bottleneck = F.interpolate(
                depth, size=bottleneck_feat.shape[2:], mode='bilinear', align_corners=False
            )

        # ================================================================
        # Helmholtz velocity decomposition
        # ================================================================
        if self.enable_helmholtz:
            v_total, v_curl_free, v_solenoidal = self.helmholtz_predictor(
                bottleneck_feat, lq_bottleneck, depth_bottleneck, temb
            )
            # Upsample velocity to full resolution for visualization
            v_total_full = F.interpolate(
                v_total, size=(H, W), mode='bilinear', align_corners=False
            )
            # Compute displacement from velocity (simple Euler)
            # dx = v * dt ≈ v at t≈1 (assume final step)
            displacement = v_total_full
            self._last_v = v_total_full
            self._last_v_curl_free = F.interpolate(
                v_curl_free, size=(H, W), mode='bilinear', align_corners=False
            )
            self._last_v_solenoidal = F.interpolate(
                v_solenoidal, size=(H, W), mode='bilinear', align_corners=False
            )
        else:
            displacement = torch.zeros(B, 2, H, W).to(x.device)
            self._last_v = displacement
            self._last_v_curl_free = displacement
            self._last_v_solenoidal = displacement

        # ================================================================
        # Decode (standard UNet decoder)
        # ================================================================
        for i_idx, i_level in enumerate(reversed(range(self.num_resolutions))):
            block_modules = self.up_modules[i_idx]
            for i_block in range(self.num_res_blocks + 1):
                resnet_block = block_modules[f'{i_level}a_{i_block}a_block']
                h = resnet_block(torch.cat([h, hs.pop()], axis=1), temb)
                attn_key = f'{i_level}a_{i_block}b_attn'
                if h.size(2) in self.attn_resolutions and attn_key in block_modules:
                    h = block_modules[attn_key](h, temb)
            if i_level != 0:
                h = block_modules[f'{i_level}b_upsample'](h)
        assert not hs

        # Final output + displacement from velocity
        x_1_pred = self.end_conv(h)

        # Add Helmholtz displacement as residual
        # Note: displacement is small, scaled by a learnable factor
        if self.enable_helmholtz:
            disp_scale = getattr(self, 'disp_scale', nn.Parameter(torch.tensor(1.0)))
            x_1_pred = x_1_pred + disp_scale * displacement[:, :1].expand(-1, 3, -1, -1)

        return x_1_pred

    def forward_with_velocity(self, x, c=None, temp=None):
        """
        Forward pass that returns velocity fields (for analysis).

        Returns:
            x_1_pred: (B, 3, H, W)
            v_total: (B, 2, H, W)
            v_curl_free: (B, 2, H, W)
            v_solenoidal: (B, 2, H, W)
        """
        out = self.forward(x, c, temp)
        return out, self._last_v, self._last_v_curl_free, self._last_v_solenoidal

    def compute_helmholtz_losses(self, v, lambda_beltrami=0.01):
        """
        Compute Helmholtz + Beltrami regularization losses.

        L_total = ||∇·v||² + ||∇×v||² + λ·||∇×v - λ·v||² (Beltrami soft)

        Args:
            v: (B, 2, H, W) velocity field
            lambda_beltrami: weight for Beltrami condition

        Returns:
            dict of scalar losses
        """
        div = helmholtz_divergence(v)
        curl = helmholtz_curl(v)

        l_div = (div ** 2).mean()  # Penalize non-div-free
        l_curl = (curl ** 2).mean()  # Penalize non-curl-free

        # Beltrami soft constraint: ∇×v ≈ λ·v
        # In 2D: ||curl - λ·|v||²
        v_mag = (v ** 2).sum(dim=1, keepdim=True).sqrt()
        l_beltrami = ((curl - lambda_beltrami * v_mag) ** 2).mean()

        return {
            'l_div': l_div,
            'l_curl': l_curl,
            'l_beltrami': l_beltrami,
        }
