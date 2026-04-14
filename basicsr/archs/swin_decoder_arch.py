"""
Decoder architecture that uses the Middle and Upsampling parts of PnPFlowUNet.

Takes intermediate features from ConvNeXt backbone (get_intermediate_layers(x, n=4))
and produces decoder outputs corresponding to PnPFlowUNet's Downsampling block outputs.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.swinir_arch import RSTB
from basicsr.archs.unet_arch import (
    Swish, upsample,
    ResidualBlock, SelfAttention, TimestepEmbedding,
    Dual_TimestepEmbedding, conv2d
)


def group_norm(out_ch, num_groups=30):
    return nn.GroupNorm(
        num_groups=num_groups,
        num_channels=out_ch,
        eps=1e-6,
        affine=True)
    

@ARCH_REGISTRY.register()
class SwinUNetDecoder(nn.Module):
    """
    UNet-style Decoder using the Middle and Upsampling parts of PnPFlowUNet.

    Takes intermediate features from ConvNeXt backbone (get_intermediate_layers(x, n=4))
    and produces 4 decoder outputs corresponding to PnPFlowUNet's Downsampling block outputs.

    Input: list of 4 feature tensors from ConvNeXt
           feat[0]: [B, 96, H/4, W/4] (stage 0)
           feat[1]: [B, 192, H/4, W/4] (stage 1)
           feat[2]: [B, 384, H/4, W/4] (stage 2)
           feat[3]: [B, 768, H/4, W/4] (stage 3)

    Output: list of 4 tensors [x[0], x[1], x[2], x[3]]
            x[0]: finest resolution output (32 ch, corresponds to Downsampling level 0)
            x[1]: second level output (64 ch, corresponds to Downsampling level 1)
            x[2]: third level output (128 ch, corresponds to Downsampling level 2)
            x[3]: coarsest resolution output (256 ch, corresponds to Downsampling level 3)
    """

    def __init__(
        self,
        encoder_dims=[96, 192, 384, 768],
        img_size=256,
        ch=30,
        output_channels=3,
        ch_mult=(1, 2, 4, 6),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.,
        resamp_with_conv=True,
        act=Swish(),
        normalize=group_norm,
        meanflow=False,
        swin_depths=[6, 6, 6, 6, 6, 6],
        swin_num_heads=[6, 6, 6, 6, 6, 6],  # 256 / 8 = 32, must be divisible
        window_size=8,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        resi_connection='1conv',
    ):
        super().__init__()
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.act = act
        self.normalize = normalize
        self.resamp_with_conv = resamp_with_conv

        num_resolutions = len(ch_mult)
        temb_ch = ch * 4
        in_ch = ch * ch_mult[-1]  # 256

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

        # Project encoder features to decoder channels
        # encoder_dims = [96, 192, 384, 768]
        # decoder channels = ch * ch_mult = [32, 64, 128, 256]
        self.proj_modules = nn.ModuleList([
            nn.Conv2d(encoder_dims[i], ch * ch_mult[i], 1, 1, 0)
            for i in range(num_resolutions)
        ])

        # Middle modules: RSTB blocks (Residual Swin Transformer Block)
        # RSTB expects input_resolution, depth, num_heads per layer
        self.mid_num_layers = len(swin_depths)
        self.mid_input_resolution = (img_size // 4, img_size // 4)  # H/4, W/4 since encoder features are at H/4
        self.mid_drop_path = [x.item() for x in torch.linspace(0, drop_path_rate, sum(swin_depths))]

        mid_modules = []
        for i_layer in range(self.mid_num_layers):
            layer = RSTB(
                dim=in_ch,
                input_resolution=self.mid_input_resolution,
                depth=swin_depths[i_layer],
                num_heads=swin_num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=self.mid_drop_path[sum(swin_depths[:i_layer]):sum(swin_depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=1,
                resi_connection=resi_connection)
            mid_modules.append(layer)
        self.mid_modules = nn.ModuleList(mid_modules)

        # Upsampling modules (mirrors PnPFlowUNet's up_modules)
        up_modules = []
        for i_level in reversed(range(num_resolutions)):
            out_ch = ch * ch_mult[i_level]

            # Determine input channels for this level
            # Level 3 (coarsest): middle output (256ch) -> first ResBlock -> upsample
            # Level 2: upsample (256ch) + skip (128ch) = 384ch -> first ResBlock -> 128ch
            # Level 1: upsample (128ch) + skip (64ch) = 192ch -> first ResBlock -> 64ch
            # Level 0: upsample (64ch) + skip (32ch) = 96ch -> first ResBlock -> 32ch
            if i_level == num_resolutions - 1:
                in_ch_level = in_ch  # 256 for level 3
            else:
                prev_out_ch = ch * ch_mult[i_level + 1]  # output ch of previous (coarser) level
                in_ch_level = prev_out_ch + out_ch  # upsample + skip connection

            block_modules = {}
            for i_block in range(num_res_blocks + 1):
                block_modules['{}a_{}a_block'.format(i_level, i_block)] = \
                    ResidualBlock(
                        in_ch=in_ch_level if i_block == 0 else out_ch,
                        temb_ch=temb_ch,
                        out_ch=out_ch,
                        dropout=dropout,
                        act=act,
                        normalize=normalize)

            if i_level != 0:
                block_modules['{}b_upsample'.format(i_level)] = upsample(
                    out_ch, with_conv=resamp_with_conv)

            up_modules += [nn.ModuleDict(block_modules)]

        self.up_modules = nn.ModuleList(up_modules)
        self.end_conv = nn.Sequential(
            normalize(out_ch),
            self.act,
            conv2d(out_ch, output_channels, init_scale=0.),
        )

    def _compute_cond_module(self, module, x, temp):
        for m in module:
            x = m(x, temp)
        return x

    def forward(self, encoder_features, temp=None):
        """
        Args:
            encoder_features: list of 4 [B, C, H/4, W/4] tensors
                              channels: [96, 192, 384, 768]
            temp: timestep embedding

        Returns:
            list of 4 tensors [x[0], x[1], x[2], x[3]]
            x[0]: level 0 output (32 ch, H resolution)
            x[1]: level 1 output (64 ch, H/2 resolution)
            x[2]: level 2 output (128 ch, H/4 resolution)
            x[3]: level 3 output (256 ch, H/8 resolution)
        """
        B = encoder_features[0].shape[0]

        # Timestep embedding
        if temp is None:
            temp = torch.ones(B, device=encoder_features[0].device)
        if self.meanflow:
            temp, remp = temp[:, 0:1].squeeze(), temp[:, 1:].squeeze()
            temb = self.temb_net(temp, remp)
        else:
            temb = self.temb_net(temp)

        # Project encoder features to decoder channel dimensions
        proj_feats = [proj(feat) for proj, feat in zip(self.proj_modules, encoder_features)]
        # proj_feats[0]: [B, 32, H/4, W/4]
        # proj_feats[1]: [B, 64, H/4, W/4]
        # proj_feats[2]: [B, 128, H/4, W/4]
        # proj_feats[3]: [B, 256, H/4, W/4]

        # Level 3 output (coarsest, H/8): downsample deepest encoder feature
        x_3 = F.avg_pool2d(proj_feats[3], 2)  # [B, 256, H/8, W/8]

        # Level 2: Process proj_feats[3] through middle (RSTB), concat with proj_feats[2] at H/4
        # RSTB expects (B, N, C) format where N = H*W
        x_size = (proj_feats[3].shape[2], proj_feats[3].shape[3])
        # Convert (B, C, H, W) to (B, N, C) for RSTB
        x_for_rstb = proj_feats[3].flatten(2).transpose(1, 2)  # (B, H*W, C)
        for m in self.mid_modules:
            x_for_rstb = m(x_for_rstb, x_size)
        # Convert back to (B, C, H, W)
        h_mid = x_for_rstb.transpose(1, 2).view(x_for_rstb.shape[0], -1, x_size[0], x_size[1])
        h = torch.cat([h_mid, proj_feats[2]], dim=1)  # [B, 384, H/4, W/4]
        block_modules = self.up_modules[1]  # level index 1 -> level 2
        for i_block in range(self.num_res_blocks + 1):
            h = block_modules['2a_{}a_block'.format(i_block)](h, temb)
        x_2 = h  # [B, 128, H/4, W/4]

        # Level 1: upsample x_2 (H/4->H/2), concat with proj_feats[1] (H/4->H/2 upsampled)
        h = F.interpolate(x_2, scale_factor=2, mode='nearest')  # [B, 128, H/2, W/2]
        h = torch.cat([h, F.interpolate(proj_feats[1], scale_factor=2, mode='nearest')], dim=1)  # [B, 128+128, H/2, W/2]
        block_modules = self.up_modules[2]  # level index 2 -> level 1
        for i_block in range(self.num_res_blocks + 1):
            h = block_modules['1a_{}a_block'.format(i_block)](h, temb)
        x_1 = h  # [B, 64, H/2, W/2]

        # Level 0: upsample x_1 (H/2->H), concat with proj_feats[0] (H/4->H upsampled by 4x)
        h = F.interpolate(x_1, scale_factor=2, mode='nearest')  # [B, 64, H, W]
        h = torch.cat([h, F.interpolate(proj_feats[0], scale_factor=4, mode='nearest')], dim=1)  # [B, 64+128, H, W]
        block_modules = self.up_modules[3]  # level index 3 -> level 0
        for i_block in range(self.num_res_blocks + 1):
            h = block_modules['0a_{}a_block'.format(i_block)](h, temb)
        x_0 = h  # [B, 32, H, W]
        x_0 = self.end_conv(x_0)  # [B, output_channels, H, W]
        return x_0 #[x_0, x_1, x_2, x_3]
