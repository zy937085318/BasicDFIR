# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
# DINoUNet with 7 Memory Enhancement Innovation Points for Image Super-Resolution

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.dinov3.convnext import ConvNeXt
from basicsr.archs.unet_arch import (
    Swish, group_norm, upsample, ResidualBlock,
    SelfAttention, TimestepEmbedding, Dual_TimestepEmbedding, conv2d,
)
from basicsr.utils.registry import ARCH_REGISTRY


# =============================================================================
# Innovation Point 1: Hopfield Memory Block (V-HMN style)
# Source: Vision Hopfield Memory Networks (arXiv 2026)
# =============================================================================

class HopfieldMemoryBlock(nn.Module):
    """
    Hopfield-style associative memory block.
    Uses cosine similarity to retrieve prototypes from a learnable memory bank.
    Iterative correction rule: z^{(t+1)} = z^{(t)} + β · (m - z^{(t)})
    """
    def __init__(self, channels, memory_size=64, num_heads=4, beta=0.5, num_iterations=3):
        super().__init__()
        self.channels = channels
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.beta = beta
        self.num_iterations = num_iterations
        
        self.memory_bank = nn.Parameter(torch.randn(memory_size, channels))
        nn.init.normal_(self.memory_bank, std=0.02)
        
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.query_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.key_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.value_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query_proj(x)
        
        # Reshape to [B, H*W, C]
        q_flat = q.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        q_norm = F.normalize(q_flat, p=2, dim=-1)
        mem_norm = F.normalize(self.memory_bank, p=2, dim=-1)  # [M, C]
        
        # Transpose memory to [B, C, M] for proper matmul
        mem_expanded = mem_norm.unsqueeze(0).expand(B, -1, -1).transpose(1, 2)  # [B, C, M]
        
        # Attention: q_norm [B, H*W, C] @ mem_expanded [B, C, M] -> [B, H*W, M]
        attn_mem = torch.bmm(q_norm, mem_expanded)
        attn_mem = F.softmax(attn_mem, dim=-1)
        
        # Retrieve: attn_mem [B, H*W, M] @ mem_norm [B, M, C] transposed from [B, C, M] 
        # Actually need attn_mem [B, H*W, M] @ mem_norm.T [B, M, C] -> [B, H*W, C]
        mem_retrieved = torch.bmm(attn_mem, mem_expanded.transpose(1, 2))  # [B, H*W, C]
        
        # Iterative correction
        z = q_norm
        for _ in range(self.num_iterations):
            z_norm = F.normalize(z, p=2, dim=-1)
            attn_iter = torch.bmm(z_norm, mem_expanded)
            attn_iter = F.softmax(attn_iter, dim=-1)
            m_agg = torch.bmm(attn_iter, mem_expanded.transpose(1, 2))
            z = z + self.beta * (m_agg - z)
        
        # Reshape back to spatial
        out = z.permute(0, 2, 1).view(B, C, H, W)
        out = self.out_proj(out)
        return x + out


# =============================================================================
# Innovation Point 2: ConvLSTM Cell (MADUNet style)
# Source: Memory-Augmented Deep Unfolding Network (IJCV 2023)
# =============================================================================

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        
        self.W_xi = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.W_hi = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.b_i = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        
        self.W_xf = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.W_hf = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.b_f = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        
        self.W_xo = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.W_ho = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.b_o = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        
        self.W_xc = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.W_hc = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.b_c = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        
    def forward(self, x, h, c):
        i = torch.sigmoid(self.W_xi(x) + self.W_hi(h) + self.b_i)
        f = torch.sigmoid(self.W_xf(x) + self.W_hf(h) + self.b_f)
        o = torch.sigmoid(self.W_xo(x) + self.W_ho(h) + self.b_o)
        c_tilde = torch.tanh(self.W_xc(x) + self.W_hc(h) + self.b_c)
        c_new = f * c + i * c_tilde
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTMModule(nn.Module):
    def __init__(self, channels, num_layers=1):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        cells = []
        for i in range(num_layers):
            cells.append(ConvLSTMCell(channels, channels, kernel_size=3))
        self.cells = nn.ModuleList(cells)
        self.h_init = nn.Parameter(torch.randn(1, channels, 1, 1))
        self.c_init = nn.Parameter(torch.randn(1, channels, 1, 1))
        
    def forward(self, x, states=None):
        B, C, H, W = x.shape
        if states is None:
            h = self.h_init.expand(B, -1, H, W)
            c = self.c_init.expand(B, -1, H, W)
        else:
            h, c = states
        for cell in self.cells:
            h, c = cell(x, h, c)
            x = h
        return h, (h, c)


# =============================================================================
# Innovation Point 3: Token Dictionary Attention (ATD style)
# Source: Adaptive Token Dictionary (arXiv 2026)
# =============================================================================

class TokenDictionaryAttention(nn.Module):
    def __init__(self, channels, num_tokens=64, temperature=0.1, num_categories=4):
        super().__init__()
        self.channels = channels
        self.num_tokens = num_tokens
        self.temperature = temperature
        self.num_categories = num_categories
        
        self.token_dict = nn.Parameter(torch.randn(num_tokens, channels))
        nn.init.normal_(self.token_dict, std=0.02)
        
        self.category_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, num_categories, kernel_size=1),
        )
        
        self.query_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.key_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.value_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.adjusted_temp = 1 + temperature * (num_tokens ** 0.5)
        
    def forward(self, x, temp=None):
        B, C, H, W = x.shape
        cat_weights = self.category_proj(x)
        cat_weights = F.softmax(cat_weights, dim=1)
        
        q = self.query_proj(x)
        
        # Reshape to [B, H*W, C]
        q_flat = q.view(B, C, -1).permute(0, 2, 1)
        
        q_norm = F.normalize(q_flat, p=2, dim=-1)
        d_norm = F.normalize(self.token_dict, p=2, dim=-1)  # [M, C]
        
        # Transpose dictionary for proper matmul
        d_expanded = d_norm.unsqueeze(0).expand(B, -1, -1).transpose(1, 2)  # [B, C, M]
        
        # Cross-attention with dictionary
        attn = torch.bmm(q_norm, d_expanded)  # [B, H*W, M]
        attn = attn / self.adjusted_temp
        attn = F.softmax(attn, dim=-1)
        mem_out = torch.bmm(attn, d_expanded.transpose(1, 2))  # [B, H*W, C]
        
        # Reshape back
        out = mem_out.permute(0, 2, 1).view(B, C, H, W)
        out = self.out_proj(out)
        return x + out


# =============================================================================
# Innovation Point 4: Memory Recursive Module (MRNet style)
# Source: Memory Recursive Network (ACM MM 2020)
# =============================================================================

class MemoryRecursiveBlock(nn.Module):
    def __init__(self, channels, num_iterations=3):
        super().__init__()
        self.channels = channels
        self.num_iterations = num_iterations
        
        self.input_encoder = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        
        self.memory_updater = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        
        self.residual_encoder = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        
        self.shuffle_conv = nn.Sequential(
            nn.Conv2d(channels * (num_iterations + 1), channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        
    def forward(self, x, h_prev=None):
        if h_prev is None:
            h_prev = torch.zeros_like(x)
        h = h_prev
        iter_features = [x]
        for i in range(self.num_iterations):
            ie_input = torch.cat([x, h], dim=1)
            ie_out = self.input_encoder(ie_input)
            r_i = self.residual_encoder(x)
            mu_input = torch.cat([r_i, h], dim=1)
            h = self.memory_updater(mu_input)
            iter_features.append(h)
        concat_features = torch.cat(iter_features, dim=1)
        h = self.shuffle_conv(concat_features)
        out = self.out_proj(h)
        return x + out


# =============================================================================
# Innovation Point 5: Corgi Memory Bank
# Source: Cached Memory Guided Video Generation (arXiv 2024)
# =============================================================================

class CorgiMemoryBank(nn.Module):
    def __init__(self, channels, num_slots=8, lambda_mem=0.3):
        super().__init__()
        self.channels = channels
        self.num_slots = num_slots
        self.lambda_mem = lambda_mem
        self.register_buffer("memory_bank", torch.randn(num_slots, channels))
        self.register_buffer("centroid", torch.randn(1, channels))
        self.centroid_momentum = 0.99
        
        self.agg_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels // 4, num_slots, kernel_size=1),
        )
        
    def forward(self, x, update=True):
        B, C, H, W = x.shape
        x_flat = F.adaptive_avg_pool2d(x, 1)
        z_new = x_flat.squeeze(-1).squeeze(-1)
        centroid_aligned = self.centroid.expand(B, -1)
        coverage_scores = torch.norm(z_new - centroid_aligned, p=2, dim=1)
        
        if update and self.training:
            with torch.no_grad():
                new_centroid = z_new.mean(dim=0, keepdim=True)
                self.centroid = self.centroid_momentum * self.centroid +                                 (1 - self.centroid_momentum) * new_centroid
        
        mem_bank = self.memory_bank
        attn_logits = x_flat.view(B, C) @ mem_bank.T / (C ** 0.5)
        attn_weights = F.softmax(attn_logits, dim=-1)
        mem_bank_expanded = mem_bank.unsqueeze(0).expand(B, -1, -1)
        m_agg = torch.bmm(attn_weights.unsqueeze(1), mem_bank_expanded).squeeze(1)
        m_agg_spatial = m_agg.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        out = x + self.lambda_mem * m_agg_spatial
        
        if update and self.training:
            with torch.no_grad():
                k = min(2, self.num_slots, B)
                if k > 0:
                    topk_vals, topk_idx = torch.topk(coverage_scores, k=k, largest=True)
                    for i, idx in enumerate(topk_idx):
                        self.memory_bank[idx] = z_new[i].detach()
        return out


# =============================================================================
# Innovation Point 7: Dictionary Memory (MemNet + CDLNet style)
# Source: MemNet + CDLNet
# =============================================================================

class DictionaryMemoryBlock(nn.Module):
    def __init__(self, channels, dict_size=64, sparsity=4):
        super().__init__()
        self.channels = channels
        self.dict_size = dict_size
        self.sparsity = sparsity
        
        self.texture_dict = nn.Parameter(torch.randn(dict_size, channels))
        nn.init.kaiming_normal_(self.texture_dict, mode="fan_out")
        self.structure_dict = nn.Parameter(torch.randn(dict_size, channels))
        nn.init.kaiming_normal_(self.structure_dict, mode="fan_out")
        
        self.domain_comp = nn.Parameter(torch.eye(channels)[:, :channels])
        
        self.sparse_proj = nn.Sequential(
            nn.Conv2d(channels, dict_size, kernel_size=1),
            nn.GELU(),
        )
        
        self.texture_recon = nn.Conv2d(dict_size, channels, kernel_size=1)
        self.structure_recon = nn.Conv2d(dict_size, channels, kernel_size=1)
        
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, 2, kernel_size=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Sparse coding
        x_norm = F.normalize(x, p=2, dim=1)
        tex_code = self.sparse_proj(x_norm)  # [B, M, H, W]
        
        # Reconstruct from texture dictionary
        tex_code_flat = tex_code.flatten(2)  # [B, M, H*W]
        tex_dict = F.normalize(self.texture_dict, p=2, dim=-1)  # [M, C]
        
        # tex_code_flat is [B, M, H*W], need [B, H*W, M] for matmul with [M, C]
        tex_code_T = tex_code_flat.transpose(1, 2)  # [B, H*W, M]
        tex_dict_expanded = tex_dict.unsqueeze(0).expand(B, -1, -1)  # [B, M, C]
        tex_recon = torch.bmm(tex_code_T, tex_dict_expanded)  # [B, H*W, C]
        tex_recon = tex_recon.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        
        # Reconstruct from structure dictionary  
        struct_dict = F.normalize(self.structure_dict, p=2, dim=-1)  # [M, C]
        struct_dict_expanded = struct_dict.unsqueeze(0).expand(B, -1, -1)  # [B, M, C]
        struct_recon = torch.bmm(tex_code_T, struct_dict_expanded)
        struct_recon = struct_recon.transpose(1, 2).reshape(B, C, H, W)
        
        # Apply domain compensation
        tex_recon = tex_recon + 0.1 * struct_recon
        
        # Fusion gate
        combined = torch.cat([tex_recon, struct_recon], dim=1)
        gate = self.fusion_gate(combined)
        out = gate[:, 0:1] * tex_recon + gate[:, 1:2] * struct_recon
        return x + out


# =============================================================================
# Innovation Point 8: Two-Stage Coarse-to-Fine (SMNet style)
# Source: Staged Memory Network (ACM MM 2020)
# =============================================================================

class CoarseToFineStage(nn.Module):
    def __init__(self, channels, num_stages=2):
        super().__init__()
        self.channels = channels
        
        self.coarse_stage = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        
        dilations = [1, 2, 4, 8]
        self.context_block = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // 4, kernel_size=3, padding=d, dilation=d),
                nn.BatchNorm2d(channels // 4),
                nn.GELU(),
            ) for d in dilations
        ])
        
        self.context_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        
        self.fine_stage = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        
        self.memory_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        
    def forward(self, x):
        coarse_feat = self.coarse_stage(x)
        context_feats = []
        for ctx_conv in self.context_block:
            context_feats.append(ctx_conv(coarse_feat))
        context_concat = torch.cat(context_feats, dim=1)
        context_out = self.context_fusion(context_concat)
        coarse_out = coarse_feat + context_out
        fine_input = torch.cat([x, coarse_out], dim=1)
        fine_feat = self.fine_stage(fine_input)
        mem_attn = self.memory_attention(fine_feat)
        fine_feat = fine_feat * mem_attn
        out = self.out_proj(fine_feat)
        return x + out


# =============================================================================
# ConvNeXt Encoder (from dinounet_arch.py)
# =============================================================================

class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        in_chans=3,
        pretrained=None,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        freeze_backbone=False,
        use_multiscale=True,
        scale_factor_first_layer=2
    ):
        super().__init__()
        self.dims = dims
        self.depths = depths
        self.num_resolutions = 4

        self.backbone = ConvNeXt(
            in_chans=in_chans,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            scale_factor=scale_factor_first_layer,
        )

        if pretrained is not None:
            self._load_pretrained(pretrained)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            stem = self.backbone.downsample_layers[0]
            for i in stem:
                i.weight.requires_grad = True
                i.bias.requires_grad = True

        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.down_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                )
                for i in range(3)
            ])

    def _load_pretrained(self, pretrained_path):
        skip_layers = ["downsample_layers.0.0.weight", "downsample_layers.0.0.bias"]
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model_dict = self.backbone.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items()
                         if k in model_dict and k not in skip_layers}
        self.backbone.load_state_dict(filtered_dict, strict=False)

    def forward(self, x):
        features = []
        for i in range(4):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)
            features.append(x)

        if self.use_multiscale:
            scaled_feats = [features[0]]
            for i in range(3):
                scaled_feats.append(self.down_proj[i](scaled_feats[-1]))
            return scaled_feats
        return features


# =============================================================================
# UNet Decoder with Memory Modules
# =============================================================================

class UNetDecoderMemory(nn.Module):
    def __init__(
        self,
        encoder_dims=[96, 192, 384, 768],
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        act=Swish(),
        normalize=group_norm,
        meanflow=False,
        enable_hopfield=False,
        enable_convlstm=False,
        enable_token_dict=False,
        enable_mrm=False,
        enable_memory_bank=False,
        enable_dict_memory=False,
        enable_coarse_to_fine=False,
    ):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.num_resolutions = len(encoder_dims)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv
        self.act = act
        self.normalize = normalize
        self.meanflow = meanflow
        
        self.enable_hopfield = enable_hopfield
        self.enable_convlstm = enable_convlstm
        self.enable_token_dict = enable_token_dict
        self.enable_mrm = enable_mrm
        self.enable_memory_bank = enable_memory_bank
        self.enable_dict_memory = enable_dict_memory
        self.enable_coarse_to_fine = enable_coarse_to_fine

        ch = encoder_dims[-1]
        temb_ch = ch * 4
        if meanflow:
            self.temb_net = Dual_TimestepEmbedding(
                embedding_dim=ch, hidden_dim=temb_ch, output_dim=temb_ch, act=act)
        else:
            self.temb_net = TimestepEmbedding(
                embedding_dim=ch, hidden_dim=temb_ch, output_dim=temb_ch, act=act)

        up_modules = []
        in_ch = encoder_dims[-1]

        for i_level in range(self.num_resolutions):
            block_modules = {}
            enc_stage_idx = self.num_resolutions - 1 - i_level
            skip_ch = encoder_dims[enc_stage_idx]
            n_blocks = num_res_blocks + 1 if i_level > 0 else num_res_blocks
            out_ch = skip_ch

            if i_level == 0:
                block_modules["bottleneck_proj"] = nn.Sequential(
                    normalize(in_ch),
                    act,
                    conv2d(in_ch, skip_ch),
                )
                in_ch = skip_ch

            # Memory modules are inserted at each decoder level
            # They process features at the skip_ch channel dimension
            if enable_hopfield:
                block_modules["hopfield_mem"] = HopfieldMemoryBlock(skip_ch)
            if enable_convlstm:
                block_modules["conv_lstm"] = ConvLSTMModule(skip_ch)
            if enable_token_dict:
                block_modules["token_dict"] = TokenDictionaryAttention(skip_ch)
            if enable_mrm:
                block_modules["mrm"] = MemoryRecursiveBlock(skip_ch)
            if enable_memory_bank:
                block_modules["memory_bank"] = CorgiMemoryBank(skip_ch)
            if enable_dict_memory:
                block_modules["dict_mem"] = DictionaryMemoryBlock(skip_ch)
            if enable_coarse_to_fine:
                block_modules["coarse_to_fine"] = CoarseToFineStage(skip_ch)

            for i_block in range(n_blocks):
                if i_block == 0 and i_level > 0:
                    block_in_ch = in_ch + skip_ch
                elif i_block == 0 and i_level == 0:
                    block_in_ch = in_ch
                else:
                    block_in_ch = out_ch

                block_modules[f"{i_level}a_{i_block}a_block"] = ResidualBlock(
                    in_ch=block_in_ch,
                    temb_ch=temb_ch,
                    out_ch=out_ch,
                    dropout=dropout,
                    act=act,
                    normalize=normalize,
                )

            if i_level < self.num_resolutions - 1:
                block_modules[f"{i_level}b_upsample"] = upsample(
                    out_ch, with_conv=resamp_with_conv)

            up_modules.append(nn.ModuleDict(block_modules))
            in_ch = out_ch

        self.up_modules = nn.ModuleList(up_modules)

        self.end_conv = nn.Sequential(
            normalize(encoder_dims[0]),
            self.act,
            conv2d(encoder_dims[0], encoder_dims[0], init_scale=0.),
        )

    def forward(self, encoder_features, temb):
        enc_feats = list(reversed(encoder_features))
        h = enc_feats[0]

        for i_level in range(self.num_resolutions):
            block_modules = self.up_modules[i_level]
            n_blocks = self.num_res_blocks + 1 if i_level > 0 else self.num_res_blocks

            if i_level == 0:
                h = block_modules["bottleneck_proj"](h)

            # Residual blocks (concat happens inside first block at i_level > 0)
            for i_block in range(n_blocks):
                resnet_block = block_modules[f"{i_level}a_{i_block}a_block"]
                h = resnet_block(h, temb)

            # Memory modules at the output of each level (before upsample/concat)
            # These operate at the skip_ch channel dimension
            if self.enable_hopfield and "hopfield_mem" in block_modules:
                h = block_modules["hopfield_mem"](h)
            if self.enable_token_dict and "token_dict" in block_modules:
                h = block_modules["token_dict"](h, temb)
            if self.enable_mrm and "mrm" in block_modules:
                h = block_modules["mrm"](h)
            if self.enable_dict_memory and "dict_mem" in block_modules:
                h = block_modules["dict_mem"](h)
            if self.enable_coarse_to_fine and "coarse_to_fine" in block_modules:
                h = block_modules["coarse_to_fine"](h)

            # ConvLSTM and MemoryBank after other memory modules
            if self.enable_convlstm and "conv_lstm" in block_modules:
                h, _ = block_modules["conv_lstm"](h)
            if self.enable_memory_bank and "memory_bank" in block_modules:
                h = block_modules["memory_bank"](h)

            # Upsample + concatenate (except for last level)
            if i_level < self.num_resolutions - 1:
                h = self.up_modules[i_level][f"{i_level}b_upsample"](h)
                h = torch.cat([h, enc_feats[i_level + 1]], dim=1)

        h = self.end_conv(h)
        return h


dinov3 = {
    "tiny": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768], "pretrained": None},
    "small": {"depths": [3, 3, 27, 3], "dims": [96, 192, 384, 768], "pretrained": None},
    "base": {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024], "pretrained": None},
    "large": {"depths": [3, 3, 27, 3], "dims": [192, 384, 768, 1536], "pretrained": None},
}


@ARCH_REGISTRY.register()
class ConvNeXtUNetMemory(nn.Module):
    """
    ConvNeXt-based UNet with 7 Memory Enhancement Innovation Points.
    """
    def __init__(
        self,
        input_channels=6,
        img_size=256,
        output_channels=3,
        encoder_type="tiny",
        drop_path_rate=0.0,
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        scale_factor_first_layer=2,
        resamp_with_conv=True,
        act=Swish(),
        normalize=group_norm,
        meanflow=False,
        freeze_backbone=False,
        check_point=None,
        enable_hopfield=False,
        enable_convlstm=False,
        enable_token_dict=False,
        enable_mrm=False,
        enable_memory_bank=False,
        enable_dict_memory=False,
        enable_coarse_to_fine=False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.img_size = img_size
        self.output_channels = output_channels
        self.depths = dinov3[encoder_type]["depths"]
        self.dims = dinov3[encoder_type]["dims"]
        self.pretrained = dinov3[encoder_type]["pretrained"]
        dims = self.dims
        depths = self.depths
        pretrained = self.pretrained
        self.num_res_blocks = num_res_blocks
        self.meanflow = meanflow

        if isinstance(act, str):
            if act.lower() == "swish":
                act = Swish()
            elif act.lower() == "relu":
                act = nn.ReLU(inplace=True)
            elif act.lower() == "gelu":
                act = nn.GELU()
        if isinstance(normalize, str):
            if normalize.lower() == "group_norm":
                normalize = group_norm
        self.act = act
        self.normalize = normalize

        self.encoder = ConvNeXtEncoder(
            in_chans=input_channels,
            pretrained=pretrained,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            freeze_backbone=freeze_backbone,
        )

        self.decoder = UNetDecoderMemory(
            encoder_dims=dims,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            act=act,
            normalize=normalize,
            meanflow=meanflow,
            enable_hopfield=enable_hopfield,
            enable_convlstm=enable_convlstm,
            enable_token_dict=enable_token_dict,
            enable_mrm=enable_mrm,
            enable_memory_bank=enable_memory_bank,
            enable_dict_memory=enable_dict_memory,
            enable_coarse_to_fine=enable_coarse_to_fine,
        )

        if hasattr(self.encoder, "use_multiscale") and self.encoder.use_multiscale:
            self.output_proj = nn.Sequential(
                normalize(dims[0]),
                act,
                nn.Upsample(scale_factor=scale_factor_first_layer, mode="bilinear", align_corners=False),
                conv2d(dims[0], output_channels, init_scale=0.),
            )
        else:
            self.output_proj = nn.Sequential(
                normalize(dims[0]),
                act,
                conv2d(dims[0], output_channels, init_scale=0.),
            )

        if check_point is not None:
            checkpoint = torch.load(check_point, map_location="cpu")
            self.load_state_dict(checkpoint, strict=False)

    def forward(self, x, c=None, temp=None):
        B, C, H, W = x.size()

        if temp is None:
            temp = torch.ones(B, device=x.device)
        elif temp.dim() > 1 and temp.size(1) > 1:
            temp = temp.squeeze()
        if self.meanflow:
            if temp.dim() == 1:
                temp_t = temp
                temp_r = temp
            else:
                temp_t = temp[:, 0]
                temp_r = temp[:, 1]
            temb = self.decoder.temb_net(temp_t, temp_r)
        else:
            if temp.dim() > 1:
                temp = temp.squeeze()
            temb = self.decoder.temb_net(temp)

        encoder_features = self.encoder(x)
        decoded = self.decoder(encoder_features, temb)
        out = self.output_proj(decoded)
        return out
