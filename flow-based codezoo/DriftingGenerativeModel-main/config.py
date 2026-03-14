"""Configuration for Drifting Model training and inference."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ModelConfig:
    """Configuration for the generator model (DiT-style)."""
    # Model architecture
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0

    # Input/Output dimensions
    in_channels: int = 4  # Latent space channels (VAE)
    out_channels: int = 4
    patch_size: int = 2  # For latent space
    image_size: int = 32  # Latent space size (256/8 for VAE)

    # Conditioning
    num_classes: int = 1000  # ImageNet classes
    class_dropout_prob: float = 0.1  # For CFG training

    # Architecture details
    use_swiglu: bool = True
    use_rope: bool = True
    use_rmsnorm: bool = True
    use_qk_norm: bool = True


@dataclass
class EncoderConfig:
    """Configuration for the feature encoder (ResNet-MAE style)."""
    # Architecture
    base_channels: int = 64
    channel_multipliers: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    num_blocks_per_stage: int = 2

    # Input dimensions
    in_channels: int = 4

    # Multi-scale feature extraction
    feature_scales: List[int] = field(default_factory=lambda: [1, 2, 4, 8])

    # Normalization
    use_group_norm: bool = True
    num_groups: int = 32


@dataclass
class DriftingConfig:
    """Configuration for the drifting field computation."""
    # Kernel temperature (official: 0.05)
    temperature: float = 0.05

    # Multi-temperature option (default: False for official behavior)
    use_multi_temp: bool = False
    temperatures: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.2])

    # Sample counts
    n_classes: int = 64  # Number of classes per batch
    n_pos: int = 32  # Positive samples per class
    n_neg: int = 64  # Generated samples per class
    n_uncond: int = 16  # Unconditional samples for CFG

    # CFG training
    cfg_alpha_min: float = 1.0
    cfg_alpha_max: float = 4.0
    cfg_power: float = 3.0  # For sampling p(alpha) ~ alpha^(-power)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    gradient_clip: float = 2.0

    # Training schedule
    num_epochs: int = 100
    warmup_steps: int = 5000

    # Batch size
    batch_size: int = 4096  # Total = n_classes * n_neg

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_update_every: int = 1

    # Logging and checkpointing
    log_every: int = 100
    save_every: int = 10000
    sample_every: int = 5000

    # Mixed precision
    use_amp: bool = True

    # Paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"


@dataclass
class SamplingConfig:
    """Configuration for sampling/inference."""
    # CFG
    cfg_scale: float = 1.0  # Guidance scale at inference

    # Sampling
    num_samples: int = 50000  # For FID evaluation
    batch_size: int = 256

    # Output
    output_dir: str = "./samples"


@dataclass
class Config:
    """Full configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    drifting: DriftingConfig = field(default_factory=DriftingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    # General
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 8

    # Dataset
    data_path: str = "./data/imagenet"
    latent_path: Optional[str] = None  # Pre-computed latents

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create config from dictionary."""
        model_cfg = ModelConfig(**config_dict.get("model", {}))
        encoder_cfg = EncoderConfig(**config_dict.get("encoder", {}))
        drifting_cfg = DriftingConfig(**config_dict.get("drifting", {}))
        training_cfg = TrainingConfig(**config_dict.get("training", {}))
        sampling_cfg = SamplingConfig(**config_dict.get("sampling", {}))

        return cls(
            model=model_cfg,
            encoder=encoder_cfg,
            drifting=drifting_cfg,
            training=training_cfg,
            sampling=sampling_cfg,
            seed=config_dict.get("seed", 42),
            device=config_dict.get("device", "cuda"),
            num_workers=config_dict.get("num_workers", 8),
            data_path=config_dict.get("data_path", "./data/imagenet"),
            latent_path=config_dict.get("latent_path", None),
        )
