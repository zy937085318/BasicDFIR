"""Training script for Drifting Models.

Implements the training loop as described in Algorithm 1-2 of the paper.
"""

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from config import Config, ModelConfig, EncoderConfig, DriftingConfig, TrainingConfig
from models import DiT, FeatureEncoder
from drifting import DriftingLoss, SimpleDriftingLoss
from cfg import sample_cfg_alpha, GuidedDriftingField
from dataset import (
    ToyDataset2D,
    LatentDataset,
    ImageNetLatentDataset,
    create_dataloader,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_every: int = 1,
    ):
        self.model = model
        self.decay = decay
        self.update_every = update_every
        self.step = 0

        # Create shadow parameters
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        self.step += 1
        if self.step % self.update_every != 0:
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def apply_shadow(self):
        """Apply shadow parameters to model."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict:
        return {"shadow": self.shadow, "step": self.step}

    def load_state_dict(self, state_dict: Dict):
        self.shadow = state_dict["shadow"]
        self.step = state_dict["step"]


class Trainer:
    """Trainer for Drifting Models."""

    def __init__(
        self,
        config: Config,
        generator: nn.Module,
        feature_encoder: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)

        # Models
        self.generator = generator.to(self.device)
        self.feature_encoder = feature_encoder.to(self.device)

        # Freeze feature encoder (typically pretrained)
        for param in self.feature_encoder.parameters():
            param.requires_grad = False
        self.feature_encoder.eval()

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss (using official implementation)
        self.drift_loss = DriftingLoss(
            feature_encoder=self.feature_encoder,
            temperature=config.drifting.temperature if hasattr(config.drifting, 'temperature') else 0.05,
            use_multi_temp=config.drifting.use_multi_temp if hasattr(config.drifting, 'use_multi_temp') else False,
        ).to(self.device)

        # CFG-aware loss wrapper
        self.guided_drift = GuidedDriftingField(
            drift_loss=self.drift_loss,
            n_uncond=config.drifting.n_uncond,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=config.training.learning_rate,
            betas=(config.training.beta1, config.training.beta2),
            weight_decay=config.training.weight_decay,
        )

        # Learning rate scheduler (cosine with warmup)
        self.scheduler = self._create_scheduler()

        # EMA
        if config.training.use_ema:
            self.ema = EMA(
                self.generator,
                decay=config.training.ema_decay,
                update_every=config.training.ema_update_every,
            )
        else:
            self.ema = None

        # Mixed precision
        self.scaler = GradScaler() if config.training.use_amp else None

        # Logging
        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.output_dir / "logs")

        # Training state
        self.global_step = 0
        self.epoch = 0

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler with linear warmup and cosine decay."""
        warmup_steps = self.config.training.warmup_steps
        total_steps = (
            len(self.train_loader) * self.config.training.num_epochs
        )

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step.

        Following Algorithm 1 from the paper:
        1. Sample class labels and CFG alpha
        2. Generate samples
        3. Sample positive samples (same class)
        4. Extract features
        5. Compute drifting field and loss
        6. Update parameters
        """
        self.generator.train()
        cfg = self.config
        drifting_cfg = cfg.drifting

        positive_samples, labels = batch
        positive_samples = positive_samples.to(self.device)
        labels = labels.to(self.device)

        B = positive_samples.shape[0]

        # Sample CFG alpha values
        cfg_alpha = sample_cfg_alpha(
            batch_size=B,
            alpha_min=drifting_cfg.cfg_alpha_min,
            alpha_max=drifting_cfg.cfg_alpha_max,
            power=drifting_cfg.cfg_power,
            device=self.device,
        )

        # Sample noise
        noise = torch.randn_like(positive_samples)

        # Forward pass
        with autocast(enabled=cfg.training.use_amp):
            # Generate samples
            generated = self.generator(noise, labels, cfg_alpha)

            # Get unconditional positive samples (for CFG)
            # In a full implementation, these would be sampled from the dataset
            # without class conditioning
            uncond_positive = positive_samples[
                torch.randperm(B, device=self.device)[:drifting_cfg.n_uncond]
            ]

            # Compute loss
            loss, metrics = self.guided_drift.compute_loss(
                generated=generated,
                positive=positive_samples,
                uncond_positive=uncond_positive,
                cfg_alpha=cfg_alpha,
                update_stats=True,
            )

        # Backward pass
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                cfg.training.gradient_clip,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                cfg.training.gradient_clip,
            )
            self.optimizer.step()

        # Update scheduler
        self.scheduler.step()

        # Update EMA
        if self.ema is not None:
            self.ema.update()

        metrics["lr"] = self.scheduler.get_last_lr()[0]
        return metrics

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {}

        for batch_idx, batch in enumerate(self.train_loader):
            metrics = self.train_step(batch)

            # Accumulate metrics
            for k, v in metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)

            # Logging
            if self.global_step % self.config.training.log_every == 0:
                log_str = f"Step {self.global_step} | "
                log_str += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                logger.info(log_str)

                # TensorBoard
                for k, v in metrics.items():
                    self.writer.add_scalar(f"train/{k}", v, self.global_step)

            # Checkpointing
            if self.global_step % self.config.training.save_every == 0:
                self.save_checkpoint()

            # Sampling
            if self.global_step % self.config.training.sample_every == 0:
                self.sample_and_log()

            self.global_step += 1

        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        return avg_metrics

    def train(self):
        """Full training loop."""
        logger.info("Starting training...")
        logger.info(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")

        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.config.training.num_epochs}")

            metrics = self.train_epoch()

            log_str = f"Epoch {epoch + 1} | "
            log_str += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(log_str)

        # Final save
        self.save_checkpoint(final=True)
        logger.info("Training complete!")

    @torch.no_grad()
    def sample_and_log(self, num_samples: int = 16):
        """Generate samples and log to TensorBoard."""
        self.generator.eval()

        if self.ema is not None:
            self.ema.apply_shadow()

        # Sample random classes
        classes = torch.randint(
            0, self.config.model.num_classes,
            (num_samples,), device=self.device
        )

        # Sample noise
        noise = torch.randn(
            num_samples,
            self.config.model.in_channels,
            self.config.model.image_size,
            self.config.model.image_size,
            device=self.device,
        )

        # CFG scale
        cfg_alpha = torch.ones(num_samples, device=self.device) * self.config.sampling.cfg_scale

        # Generate
        samples = self.generator(noise, classes, cfg_alpha)

        # Log
        # Normalize to [0, 1] for visualization
        samples = (samples + 1) / 2
        samples = samples.clamp(0, 1)

        # Add to TensorBoard (as grid)
        from torchvision.utils import make_grid
        grid = make_grid(samples, nrow=int(num_samples**0.5), normalize=False)
        self.writer.add_image("samples", grid, self.global_step)

        if self.ema is not None:
            self.ema.restore()

        self.generator.train()

    def save_checkpoint(self, final: bool = False):
        """Save training checkpoint."""
        name = "final.pt" if final else f"step_{self.global_step}.pt"
        path = self.checkpoint_dir / name

        state = {
            "generator": self.generator.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config,
        }

        if self.ema is not None:
            state["ema"] = self.ema.state_dict()

        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()

        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        state = torch.load(path, map_location=self.device)

        self.generator.load_state_dict(state["generator"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]

        if self.ema is not None and "ema" in state:
            self.ema.load_state_dict(state["ema"])

        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

        logger.info(f"Loaded checkpoint from {path}")


class ToyTrainer:
    """Simplified trainer for 2D toy examples."""

    def __init__(
        self,
        dataset: ToyDataset2D,
        hidden_dim: int = 128,
        num_layers: int = 3,
        lr: float = 1e-3,
        temperature: float = 0.1,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.dataset = dataset

        # Simple MLP generator
        layers = [nn.Linear(2, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 2))
        self.generator = nn.Sequential(*layers).to(self.device)

        # Loss
        self.drift_loss = SimpleDriftingLoss(temperature=temperature)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr)

        self.step = 0

    def train_step(self, batch_size: int = 256) -> Dict[str, float]:
        """Single training step."""
        self.generator.train()

        # Sample positive data
        idx = torch.randint(0, len(self.dataset), (batch_size,))
        positive, _ = self.dataset[idx]
        positive = positive.to(self.device)

        # Sample noise and generate
        noise = torch.randn(batch_size, 2, device=self.device)
        generated = self.generator(noise)

        # Compute loss
        loss, metrics = self.drift_loss(generated, positive)

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1
        return metrics

    def train(self, num_steps: int = 2000, log_every: int = 100):
        """Train for specified number of steps."""
        for step in range(num_steps):
            metrics = self.train_step()

            if step % log_every == 0:
                log_str = f"Step {step} | "
                log_str += " | ".join([f"{k}: {v:.4e}" for k, v in metrics.items()])
                logger.info(log_str)

    @torch.no_grad()
    def sample(self, num_samples: int = 1000) -> torch.Tensor:
        """Generate samples."""
        self.generator.eval()
        noise = torch.randn(num_samples, 2, device=self.device)
        samples = self.generator(noise)
        return samples.cpu()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train Drifting Model")
    parser.add_argument("--config", type=str, default=None, help="Config file")
    parser.add_argument("--data_path", type=str, default="./data", help="Data path")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output dir")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--toy", action="store_true", help="Run toy 2D example")
    parser.add_argument("--toy_dataset", type=str, default="swiss_roll", help="Toy dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.toy:
        # Toy 2D example
        logger.info(f"Training on toy dataset: {args.toy_dataset}")
        dataset = ToyDataset2D(name=args.toy_dataset, n_samples=10000)
        trainer = ToyTrainer(dataset, device=args.device)
        trainer.train(num_steps=2000)

        # Sample and save
        samples = trainer.sample(1000)
        logger.info(f"Final samples shape: {samples.shape}")
        logger.info(f"Sample mean: {samples.mean():.4f}, std: {samples.std():.4f}")

    else:
        # Full training
        config = Config()
        config.device = args.device
        config.training.output_dir = args.output_dir
        config.training.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        config.data_path = args.data_path

        # Create models
        generator = DiT(
            img_size=config.model.image_size,
            patch_size=config.model.patch_size,
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            hidden_size=config.model.hidden_size,
            depth=config.model.depth,
            num_heads=config.model.num_heads,
            num_classes=config.model.num_classes,
        )

        feature_encoder = FeatureEncoder(
            in_channels=config.encoder.in_channels,
            base_channels=config.encoder.base_channels,
            channel_multipliers=config.encoder.channel_multipliers,
            num_blocks_per_stage=config.encoder.num_blocks_per_stage,
        )

        # Create dataset
        # For now, use toy data for testing
        dataset = ToyDataset2D(n_samples=10000)
        train_loader = create_dataloader(
            dataset,
            batch_size=config.training.batch_size,
            num_workers=config.num_workers,
        )

        # Create trainer
        trainer = Trainer(
            config=config,
            generator=generator,
            feature_encoder=feature_encoder,
            train_loader=train_loader,
        )

        # Resume if checkpoint provided
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)

        # Train
        trainer.train()


if __name__ == "__main__":
    main()
