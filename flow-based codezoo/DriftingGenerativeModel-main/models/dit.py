"""DiT-style Transformer Generator for Drifting Models.

Implements a Diffusion Transformer (DiT) style architecture with:
- SwiGLU activation in MLP blocks
- Rotary Position Embeddings (RoPE)
- RMSNorm normalization
- QK-Norm in attention
- Adaptive Layer Norm for conditioning
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute position encodings
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build the sin/cos cache for the given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors."""
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)


class SwiGLU(nn.Module):
    """SwiGLU activation function with linear projections."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, in_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    """Multi-head self-attention with optional QK-Norm and RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        use_qk_norm: bool = True,
        use_rope: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Apply QK-Norm
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE
        if self.use_rope:
            q, k = self.rope(q, k, N)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class AdaLN(nn.Module):
    """Adaptive Layer Normalization for conditioning.

    Modulates the normalized features using learned scale and shift
    parameters conditioned on class embeddings and CFG scale.
    """

    def __init__(self, dim: int, num_modulations: int = 6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, num_modulations * dim),
        )

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Compute modulation parameters.

        Returns: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        """
        modulations = self.adaLN_modulation(c).chunk(6, dim=-1)
        return modulations


class DiTBlock(nn.Module):
    """DiT Transformer block with adaptive layer norm."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Normalization
        norm_cls = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm1 = norm_cls(hidden_size)
        self.norm2 = norm_cls(hidden_size)

        # Attention
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            use_qk_norm=use_qk_norm,
            use_rope=use_rope,
        )

        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        if use_swiglu:
            # SwiGLU uses 2/3 of the hidden size for efficiency
            self.mlp = SwiGLU(hidden_size, int(mlp_hidden * 2 / 3))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, hidden_size),
            )

        # Adaptive Layer Norm
        self.adaLN = AdaLN(hidden_size)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass with conditioning.

        Args:
            x: Input tensor (B, N, C)
            c: Conditioning tensor (B, C)

        Returns:
            Output tensor (B, N, C)
        """
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN(c)
        )

        # Attention block with modulation
        x_norm = self.norm1(x)
        x_mod = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_mod)

        # MLP block with modulation
        x_norm = self.norm2(x)
        x_mod = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mod)

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class FinalLayer(nn.Module):
    """Final layer with adaptive layer norm and linear projection."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """Diffusion Transformer (DiT) style generator for Drifting Models.

    This is a single-pass, non-iterative network that maps noise to data.
    The architecture follows DiT with SwiGLU, RoPE, RMSNorm, and QK-Norm.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        out_channels: int = 4,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        self.num_patches = self.patch_embed.num_patches

        # Class embedding (includes null class for CFG)
        self.class_embed = nn.Embedding(num_classes + 1, hidden_size)

        # CFG scale embedding (encodes alpha value)
        self.cfg_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Position embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_swiglu=use_swiglu,
                use_rope=use_rope,
                use_rmsnorm=use_rmsnorm,
                use_qk_norm=use_qk_norm,
            )
            for _ in range(depth)
        ])

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Initialize position embedding
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize class embedding
        nn.init.normal_(self.class_embed.weight, std=0.02)

        # Initialize all linear layers and embeddings
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        # Zero-out final linear layer
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patch tokens back to image.

        Args:
            x: (B, num_patches, patch_size**2 * out_channels)

        Returns:
            imgs: (B, out_channels, H, W)
        """
        p = self.patch_size
        h = w = int(self.num_patches**0.5)
        c = self.out_channels

        x = x.reshape(-1, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
        x = x.reshape(-1, c, h * p, w * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input noise tensor (B, C, H, W)
            y: Class labels (B,) with values in [0, num_classes-1]
               Use num_classes for unconditional (null class)
            cfg_scale: CFG scale values (B,) or None

        Returns:
            Output tensor (B, out_channels, H, W)
        """
        # Patch embed
        x = self.patch_embed(x) + self.pos_embed

        # Class conditioning with dropout for CFG training
        if self.training and self.class_dropout_prob > 0:
            drop_mask = torch.rand(y.shape[0], device=y.device) < self.class_dropout_prob
            y = torch.where(drop_mask, torch.full_like(y, self.num_classes), y)

        c = self.class_embed(y)

        # Add CFG scale embedding if provided
        if cfg_scale is not None:
            cfg_scale = cfg_scale.view(-1, 1).float()
            c = c + self.cfg_embed(cfg_scale)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Final layer
        x = self.final_layer(x, c)

        # Unpatchify
        x = self.unpatchify(x)

        return x


def DiT_L_2(**kwargs) -> DiT:
    """DiT-L/2 model for latent space (256x256 -> 32x32 latents)."""
    return DiT(
        img_size=32,
        patch_size=2,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        **kwargs,
    )


def DiT_XL_2(**kwargs) -> DiT:
    """DiT-XL/2 model for latent space."""
    return DiT(
        img_size=32,
        patch_size=2,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        **kwargs,
    )


def DiT_L_16(**kwargs) -> DiT:
    """DiT-L/16 model for pixel space (256x256)."""
    return DiT(
        img_size=256,
        patch_size=16,
        in_channels=3,
        out_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        **kwargs,
    )
