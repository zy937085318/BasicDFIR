"""
ABCUNetv2: ABC Flow enhanced UNet for image super-resolution.

Key design: ABC flow is used as a content-aware feature modulation mechanism
embedded within the JiT (Flow Matching) framework, NOT as a replacement for
the time-path.

Architecture:
  - Standard JiT backbone (UNet) for velocity/x_1 prediction
  - ABC flow field generator: predicts local ABC parameters from bottleneck features
  - Content-aware flow modulation: uses image gradients + ABC params to modulate
    features at each resolution level
  - ABC flow is computed at bottleneck, then spatially interpolated to each level
  - Output remains 3 channels (x_1 prediction), compatible with JiT

Reference:
  - ABC Flow: Arnold-Beltrami-Celebi 3D fluid dynamics equations
  - JiT: Just-in-Time adaptive kernel transformers (flow matching backbone)
"""

from torch.nn.init import _calculate_fan_in_and_fan_out
import torch.nn as nn
from .unet_arch import (
    Swish, group_norm, upsample, downsample, ResidualBlock,
    SelfAttention, TimestepEmbedding, Dual_TimestepEmbedding,
    conv2d, variance_scaling_init_,
)
from basicsr.utils.registry import ARCH_REGISTRY
import torch
import torch.nn.functional as F


@ARCH_REGISTRY.register()
class ABCFlowFieldGenerator(nn.Module):
    """
    Predicts local ABC (Arnold-Beltrami-Celebi) flow parameters from bottleneck features.

    Takes high-level semantic features (bottleneck of UNet) and predicts
    spatially-varying ABC parameters. These are then used to compute a
    2D ABC flow field for feature modulation.

    Uses a lightweight 3-layer CNN to predict 3 parameter maps (A, B, C)
    from the bottleneck features.
    """

    def __init__(self, in_channels, hidden_channels=32, out_channels=3):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            Swish(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            Swish(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, features):
        """
        Args:
            features: (B, C, H, W) bottleneck features
        Returns:
            abc_params: (B, 3, H, W) predicted ABC parameter fields
        """
        return self.predictor(features)


@ARCH_REGISTRY.register()
class ABCFlowFeatureModulator(nn.Module):
    """
    Computes content-aware 2D ABC flow field and applies it as feature modulation.

    The 2D ABC flow field approximation (Arnold-Beltrami-Celebi):
      u = A(x,y) * sin(y) + C(x,y) * cos(y)   (x-direction flow)
      v = B(x,y) * sin(x) + A(x,y) * cos(y)   (y-direction flow)

    Modulation: F' = F + α * (u_flow * grad_x + v_flow * grad_y)

    The modulation factor is channel-independent (same u_flow, v_flow for all
    channels) but modulated by the channel's local gradient direction, acting
    as a structure-aware enhancement that preserves edges and textures.
    """

    def __init__(self, modulation_strength=0.1):
        super().__init__()
        self.modulation_strength = modulation_strength

    def forward(self, x, abc_params):
        """
        Args:
            x: (B, C, H, W) input features to be modulated
            abc_params: (B, 3, H, W) ABC parameter fields (A, B, C).
                        Will be spatially interpolated to match x's H, W.
        Returns:
            x_enhanced: (B, C, H, W) flow-enhanced features
        """
        B, C, H, W = x.shape

        # Interpolate ABC params to match feature spatial dimensions
        abc_params = F.interpolate(
            abc_params, size=(H, W), mode='bilinear', align_corners=False
        )

        # Extract ABC parameters
        A = abc_params[:, 0:1]   # (B, 1, H, W)
        B_param = abc_params[:, 1:2]  # (B, 1, H, W)
        C = abc_params[:, 2:3]   # (B, 1, H, W)

        # Coordinate grids (normalized to [0, 2π])
        # With indexing='xy': X[i,j]=x_coords[j], Y[i,j]=y_coords[i] => shape (H, W)
        x_coords = torch.linspace(0, 2 * torch.pi, W, device=x.device, dtype=x.dtype)
        y_coords = torch.linspace(0, 2 * torch.pi, H, device=x.device, dtype=x.dtype)
        X, Y = torch.meshgrid(x_coords, y_coords, indexing='xy')

        # 2D ABC flow field (spatially varying via A, B, C)
        u_flow = A * torch.sin(Y) + C * torch.cos(Y)   # (B, 1, H, W)
        v_flow = B_param * torch.sin(X) + A * torch.cos(Y)  # (B, 1, H, W)

        # Normalize input features for gradient computation
        # Use tensor.view with explicit tuple
        x_min_vals = x.flatten(2).min(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        x_max_vals = x.flatten(2).max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        x_norm = (x - x_min_vals) / (x_max_vals - x_min_vals + 1e-8)

        # Spatial gradients (Sobel-like, per-channel)
        # torch.gradient returns tuple in older PyTorch, tensor in newer (2.x+)
        grad_x_raw = torch.gradient(x_norm, dim=3)
        grad_y_raw = torch.gradient(x_norm, dim=2)
        grad_x = grad_x_raw[0] if isinstance(grad_x_raw, tuple) else grad_x_raw
        grad_y = grad_y_raw[0] if isinstance(grad_y_raw, tuple) else grad_y_raw

        # Normalize gradients channel-wise
        grad_x = grad_x / (grad_x.std(dim=(2, 3), keepdim=True) + 1e-8)
        grad_y = grad_y / (grad_y.std(dim=(2, 3), keepdim=True) + 1e-8)

        # Flow-modulated gradient enhancement
        # Each channel gets: α * (u_flow * grad_x_ch + v_flow * grad_y_ch)
        u_mod = u_flow * grad_x  # (B, C, H, W)
        v_mod = v_flow * grad_y  # (B, C, H, W)
        flow_mod = u_mod + v_mod

        # Residual-style enhancement: preserves original features
        x_enhanced = x + self.modulation_strength * flow_mod
        return x_enhanced


@ARCH_REGISTRY.register()
class ABCUNetv2(nn.Module):
    """
    ABC Flow enhanced UNet for JiT-style image super-resolution.

    Combines standard flow-matching UNet with content-aware ABC flow feature modulation.
    The ABC flow acts as a spatial attention mechanism that modulates intermediate
    features based on image structure (edges, textures), while the UNet backbone
    handles the temporal flow matching path.

    Key design:
      1. Standard UNet encoder-decoder (same as PnPFlowUNet / JiT backbone)
      2. ABC flow field generator: predicts ABC params from bottleneck features
      3. ABC modulation applied at each resolution level (via spatial interpolation)
      4. Output: 3 channels (x_1 prediction), fully compatible with JiT framework
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
                 abc_hidden_channels=32,
                 abc_modulation_strength=0.1,
                 enable_abc_flow=True,
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
        self.enable_abc_flow = enable_abc_flow

        # Resolve string act/normalize to actual modules (from yml config)
        if isinstance(act, str):
            if act.lower() == 'swish':
                act = Swish()
            elif act.lower() == 'relu':
                act = torch.nn.ReLU(inplace=True)
            elif act.lower() == 'gelu':
                act = torch.nn.GELU()
            else:
                raise ValueError(f"Unknown act type: {act}")
        if isinstance(normalize, str):
            if normalize.lower() == 'group_norm':
                normalize = group_norm
            else:
                raise ValueError(f"Unknown normalize type: {normalize}")
        self.act = act
        self.normalize = normalize

        # init
        self.num_resolutions = num_resolutions = len(ch_mult)
        in_ht = self.input_height
        in_ch = input_channels
        temb_ch = ch * 4
        assert in_ht % 2 ** (num_resolutions - 1) == 0, \
            "input_height doesn't satisfy the condition"

        # Timestep embedding
        self.meanflow = meanflow
        if self.meanflow:
            self.temb_net = Dual_TimestepEmbedding(
                embedding_dim=ch,
                hidden_dim=temb_ch,
                output_dim=temb_ch,
                act=act,
            )
        else:
            self.temb_net = TimestepEmbedding(
                embedding_dim=ch,
                hidden_dim=temb_ch,
                output_dim=temb_ch,
                act=act,
            )

        # Downsampling
        self.first_conv = conv2d(in_ch, ch)
        unet_chs = [ch]
        in_ht = in_ht
        in_ch = ch
        down_modules = []
        for i_level in range(num_resolutions):
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block_modules[f'{i_level}a_{i_block}a_block'] = ResidualBlock(
                    in_ch=in_ch,
                    temb_ch=temb_ch,
                    out_ch=out_ch,
                    dropout=dropout,
                    act=act,
                    normalize=normalize,
                )
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
        mid_modules += [ResidualBlock(in_ch,
                                      temb_ch=temb_ch,
                                      out_ch=in_ch,
                                      dropout=dropout,
                                      act=act,
                                      normalize=normalize)]
        mid_modules += [SelfAttention(in_ch, normalize=normalize)]
        mid_modules += [ResidualBlock(in_ch,
                                      temb_ch=temb_ch,
                                      out_ch=in_ch,
                                      dropout=dropout,
                                      act=act,
                                      normalize=normalize)]
        self.mid_modules = nn.ModuleList(mid_modules)

        # ABC flow components
        if self.enable_abc_flow:
            self.bottleneck_ch = in_ch  # channels at bottleneck = ch * ch_mult[-1]
            self.abc_predictor = ABCFlowFieldGenerator(
                in_channels=self.bottleneck_ch,
                hidden_channels=abc_hidden_channels,
                out_channels=3,
            )
            self.abc_modulator = ABCFlowFeatureModulator(
                modulation_strength=abc_modulation_strength,
            )

        # Upsampling
        up_modules = []
        for i_level in reversed(range(num_resolutions)):
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                block_modules[f'{i_level}a_{i_block}a_block'] = ResidualBlock(
                    in_ch=in_ch + unet_chs.pop(),
                    temb_ch=temb_ch,
                    out_ch=out_ch,
                    dropout=dropout,
                    act=act,
                    normalize=normalize)
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

        # End: predict x_1 (3 channels, same as JiT)
        self.end_conv = nn.Sequential(
            normalize(in_ch),
            self.act,
            conv2d(in_ch, output_channels, init_scale=0.),
        )

        if check_point is not None:
            checkpoint = torch.load(check_point, map_location=torch.device('cpu'))
            self.load_state_dict(checkpoint, strict=False)

    def _compute_cond_module(self, module, x, temp):
        for m in module:
            x = m(x, temp)
        return x

    def _apply_abc_to_features(self, features):
        """Apply ABC flow modulation to features.

        ABC params are predicted once at bottleneck, then spatially interpolated
        to match the target feature resolution.
        Only callable after bottleneck ABC params have been computed.
        """
        if not self.enable_abc_flow:
            return features
        if not hasattr(self, '_abc_params') or self._abc_params is None:
            return features
        return self.abc_modulator(features, self._abc_params)

    def forward(self, x, c=None, temp=None):
        """
        Forward pass compatible with JiT framework.

        Args:
            x: (B, 6, H, W) concatenated [x_t, lq_bicubic]
            c: condition (unused, for compatibility)
            temp: (B,) timestep t in [0, 1]

        Returns:
            x_1_pred: (B, 3, H, W) predicted target image
        """
        B, _, H, W = x.size()

        # Timestep embedding
        if temp is None:
            temp = torch.ones(B).to(x.device)
        if self.meanflow:
            temp_t, remp = temp[:, 0:1, ...].squeeze(), temp[:, 1:, ...].squeeze()
            temb = self.temb_net(temp_t, remp)
        else:
            temb = self.temb_net(temp)
        assert list(temb.shape) == [B, self.ch * 4]

        # Downsampling (standard UNet)
        hs = [self.first_conv(x)]

        for i_level in range(self.num_resolutions):
            block_modules = self.down_modules[i_level]
            for i_block in range(self.num_res_blocks):
                resnet_block = block_modules[f'{i_level}a_{i_block}a_block']
                h = resnet_block(hs[-1], temb)
                attn_key = f'{i_level}a_{i_block}b_attn'
                if h.size(2) in self.attn_resolutions and attn_key in block_modules:
                    h = block_modules[attn_key](h, temb)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                downsample_block = block_modules[f'{i_level}b_downsample']
                hs.append(downsample_block(hs[-1]))

        # Middle
        h = hs[-1]
        h = self._compute_cond_module(self.mid_modules, h, temb)

        # Predict ABC params at bottleneck, then apply to bottleneck and decoder
        if self.enable_abc_flow:
            self._abc_params = self.abc_predictor(h)  # (B, 3, h_H, h_W)
            h = self._apply_abc_to_features(h)  # modulate bottleneck

        # Upsampling: apply ABC modulation after each resblock in decoder
        for i_idx, i_level in enumerate(reversed(range(self.num_resolutions))):
            block_modules = self.up_modules[i_idx]
            for i_block in range(self.num_res_blocks + 1):
                resnet_block = block_modules[f'{i_level}a_{i_block}a_block']
                h = resnet_block(torch.cat([h, hs.pop()], axis=1), temb)
                if h.size(2) in self.attn_resolutions and f'{i_level}a_{i_block}b_attn' in block_modules:
                    h = block_modules[f'{i_level}a_{i_block}b_attn'](h, temb)
                if self.enable_abc_flow:
                    h = self._apply_abc_to_features(h)  # modulate decoder features
            if i_level != 0:
                upsample_block = block_modules[f'{i_level}b_upsample']
                h = upsample_block(h)
        assert not hs

        # Final output: x_1 prediction (3 channels, compatible with JiT)
        h = self.end_conv(h)
        return h

    def forward_feature(self, x, c=None, temp=None):
        """Alias for forward, for compatibility with some model wrappers."""
        return self.forward(x, c, temp)

    def compute_abc_flow_2d_from_image(self, x, features=None):
        """
        Utility: compute ABC flow field for visualization/analysis.

        Args:
            x: (B, C, H, W) input image
            features: optional pre-computed bottleneck features
        Returns:
            flow: (B, 2, H, W) 2D ABC flow field (u, v)
        """
        B, _, H, W = x.shape
        x_coords = torch.linspace(0, 2 * torch.pi, W, device=x.device, dtype=x.dtype)
        y_coords = torch.linspace(0, 2 * torch.pi, H, device=x.device, dtype=x.dtype)
        X, Y = torch.meshgrid(x_coords, y_coords, indexing='xy')

        # Default global ABC parameters for visualization
        A = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        B_param = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        C = torch.tensor(1.0, device=x.device, dtype=x.dtype)

        u = A * torch.sin(Y) + C * torch.cos(Y)
        v = B_param * torch.sin(X) + A * torch.cos(Y)

        u = u.unsqueeze(0).expand(B, -1, -1)
        v = v.unsqueeze(0).expand(B, -1, -1)

        return torch.stack([u, v], dim=1)
