import cv2
import torch
import os
import shutil
import glob
import logging
import math
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append('./pmf')
from accelerate import Accelerator
from tqdm import tqdm
from pmf.config import get_config
from pmf.dit import DiT
from pmf.pixel_mean_flow import PixelMeanFlow
from pmf.optimizer import configure_optimizers, CombinedOptimizer
from pmf.dataset import get_srloader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from timm.utils import ModelEmaV2
from transformers import get_cosine_schedule_with_warmup
import argparse
from basicsr.utils import yaml_load

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, config, accelerator, keep_last=3, keep_best=1):
        self.config = config
        self.accelerator = accelerator
        self.keep_last = keep_last
        self.keep_best = keep_best
        self.best_loss = float('inf')
        self.best_checkpoints = [] # List of (loss, path)
        self.recent_checkpoints = [] # List of path

    def save(self, model, step, loss=None):
        # Save current checkpoint
        # Standardize naming: pmf-imagenet-xl-step-{step}.pt
        # We don't have model size in config explicitly as a string tag, but we can infer or just use generic.
        # User requested "Appendix A" naming. Let's assume "pmf-step-{step}.pt" for now.
        filename = f"pmf-step-{step}"
        if loss is not None:
            filename += f"-loss-{loss:.4f}"
        filename += ".pt"

        save_path = os.path.join(self.config.checkpoint_dir, filename)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        self.accelerator.save(self.accelerator.unwrap_model(model).state_dict(), save_path)

        # Manage recent
        self.recent_checkpoints.append(save_path)
        if len(self.recent_checkpoints) > self.keep_last:
            to_remove = self.recent_checkpoints.pop(0)
            # Only remove if not in best list
            is_best = any(to_remove == p for _, p in self.best_checkpoints)
            if not is_best and os.path.exists(to_remove):
                os.remove(to_remove)

        # Manage best
        if loss is not None:
            if loss < self.best_loss:
                # This is a new best candidate
                self.best_checkpoints.append((loss, save_path))
                self.best_checkpoints.sort(key=lambda x: x[0])

                # Keep top k
                if len(self.best_checkpoints) > self.keep_best:
                    _, path_to_remove = self.best_checkpoints.pop(-1) # Remove worst of best
                    # Check if it is in recent list
                    is_recent = path_to_remove in self.recent_checkpoints
                    if not is_recent and os.path.exists(path_to_remove):
                        os.remove(path_to_remove)

                self.best_loss = self.best_checkpoints[0][0]

def train(config_path):
    config = get_config(config_path)
    dataset_opt = yaml_load('/home/ybb/Project/BasicDFIR/options/train/PixelMeanFlow/train_PFM_DiT_B8_SRx2_P256_github_pfm_source.yml')['datasets']
    num_processes = int(os.environ.get("WORLD_SIZE", "1"))
    micro = getattr(config, "micro_batch_size", None)
    global_bs = getattr(config, "global_batch_size", None)
    if global_bs is not None and micro is not None and micro > 0:
        computed_accum = math.ceil(global_bs / (micro * num_processes))
        if getattr(config, "gradient_accumulation_steps", None) in (None, 0, 1) and computed_accum > 1:
            config.gradient_accumulation_steps = computed_accum
        if micro * num_processes * config.gradient_accumulation_steps != global_bs:
            logger.warning(
                f"global_batch_size mismatch: micro_batch_size({micro})*world_size({num_processes})*grad_accum({config.gradient_accumulation_steps})"
                f"={micro * num_processes * config.gradient_accumulation_steps} != global_batch_size({global_bs})."
            )

    accelerator = Accelerator(
        mixed_precision=getattr(config, "mixed_precision", "fp16"),
        gradient_accumulation_steps=getattr(config, "gradient_accumulation_steps", 1),
    )

    device = accelerator.device

    # Dataset
    train_opt = dataset_opt['train']
    train_opt['phase'] = 'train'
    train_loader = get_srloader(config, train_opt, shuffle=True)
    # Validation Dataset
    val_opt = dataset_opt['val_1']
    val_opt['phase'] = 'val'
    val_loader = get_srloader(config, val_opt, shuffle=False)

    # Logging
    if accelerator.is_local_main_process:
        os.makedirs(config.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=config.log_dir)
    else:
        writer = None

    # Model
    model = DiT(
        input_size=config.image_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        class_dropout_prob=config.class_dropout_prob,
        num_classes=config.num_classes,
        learn_sigma=config.learn_sigma
    )

    # pMF Wrapper
    pmf_model = PixelMeanFlow(model, config)

    # Optimizer
    # We pass the underlying model to optimizer configuration
    optimizers_list = configure_optimizers(model, config)
    # optimizer = CombinedOptimizer(optimizers_list)
    # Accelerator needs list of optimizers, but CombinedOptimizer is a wrapper.
    # It's better to let accelerator handle optimizers if possible.
    # But we have two optimizers. Accelerator can handle list.

    # Prepare
    # We unpack optimizers to prepare them individualy
    model, *optimizers_list, train_loader, val_loader = accelerator.prepare(
        model, *optimizers_list, train_loader, val_loader
    )

    # Re-wrap optimizer
    optimizer = CombinedOptimizer(optimizers_list)

    # Re-wrap model in pmf (careful with DDP)
    pmf_model.model = model
    pmf_model.to(device)

    # Checkpoint Manager
    if accelerator.is_local_main_process:
        ckpt_manager = CheckpointManager(config, accelerator, keep_last=3, keep_best=1)

    # Random Seed
    if config.seed is not None:
        import random
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        logger.info(f"Random seed set to {config.seed}")

    ema_decays = getattr(config, "ema_decays", [0.9999])
    if not isinstance(ema_decays, (list, tuple)) or len(ema_decays) == 0:
        raise ValueError("ema_decays must be a non-empty list.")
    ema_models = []
    for d in ema_decays:
        ema_m = ModelEmaV2(model, decay=float(d))
        ema_m.to(device)
        ema_models.append(ema_m)
    logger.info(f"Initialized EMA decays: {ema_decays}")

    # Scheduler
    # Convert epochs to steps for scheduler
    # We need len(train_loader) which might be approximate if using IterableDataset,
    # but get_loader returns standard DataLoader (or we assume len is available).
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.num_epochs
    warmup_ratio = getattr(config, "warmup_ratio", 0.03)
    warmup_steps = int(total_steps * warmup_ratio)
    logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps} (ratio={warmup_ratio})")

    # Optimizer is CombinedOptimizer, which wraps a list.
    # get_cosine_schedule_with_warmup expects a single optimizer usually.
    # We need to handle this. CombinedOptimizer doesn't have param_groups directly exposed in a way scheduler likes?
    # Actually, CombinedOptimizer has `step` but not `param_groups`.
    # We should apply scheduler to the underlying optimizers.

    # Unpack optimizers from CombinedOptimizer (hacky, assuming structure)
    # The accelerator wrapped them, so we need to access them.
    # But wait, we wrapped them in CombinedOptimizer *after* prepare?
    # In previous code:
    # model, *optimizers_list, train_loader, val_loader = accelerator.prepare(...)
    # optimizer = CombinedOptimizer(optimizers_list)

    schedulers = []
    for opt in optimizers_list:
        sched = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        schedulers.append(sched)

    logger.info(f"Initialized Schedulers: {len(schedulers)} (Warmup: {warmup_steps}, Total: {total_steps})")

    # Training Loop
    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        running_loss = 0.0
        epoch_loss_sum = 0.0
        num_batches = 0

        for lq, gt in progress_bar:
            with accelerator.accumulate(model):
                # x is (N, C, H, W), y is (N,)
                loss, loss_dict, x_pred = pmf_model.forward_loss(lq, gt)
                # print('x_pred:', x_pred.shape, x_pred.detach().cpu().min(), '-',
                #       x_pred.detach().cpu().max(),x_pred.detach().cpu().mean())
                # print('lq:', lq.shape, lq.detach().cpu().min(), lq.detach().cpu().max())
                # print('gt:', gt.shape, gt.detach().cpu().min(), gt.detach().cpu().max())
                # print(pmf_model.model.final_layer.linear.weight.sum())
                optimizer.zero_grad()
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_value_(pmf_model.parameters(), clip_value=1)
                optimizer.step()
                for sched in schedulers:
                    sched.step()
                for ema_m in ema_models:
                    ema_m.update(model)

            global_step += 1
            if global_step % 100 == 0:
                save_path = os.path.join(config.output_dir, f"sample_globalstep_{global_step}.npy")
                # os.makedirs(config.output_dir, exist_ok=True)
                # cv2.imwrite(x_pred, save_path)
                np.save(save_path, x_pred[0:1,...].detach().cpu().numpy())
            running_loss = loss.item() # Approximation for progress bar
            epoch_loss_sum += running_loss
            num_batches += 1
            progress_bar.set_postfix(**loss_dict)

            # Logging
            if accelerator.is_local_main_process:
                for k, v in loss_dict.items():
                    writer.add_scalar(f"train/{k}", v, global_step)
                writer.add_scalar("train/epoch", epoch, global_step)
                # Log LR (use first group of first optimizer)
                current_lr = optimizers_list[0].param_groups[0]['lr']
                writer.add_scalar("train/lr", current_lr, global_step)

        # Save checkpoint at the end of each epoch
        if accelerator.is_local_main_process:
            epoch_avg_loss = epoch_loss_sum / max(num_batches, 1)
            # Use epoch avg loss as metric for best model
            ckpt_manager.save(model, f"epoch_{epoch}", loss=epoch_avg_loss)
            for d, ema_m in zip(ema_decays, ema_models):
                ema_path = os.path.join(config.checkpoint_dir, f"ema_decay_{d}_epoch_{epoch}.pt")
                torch.save(ema_m.module.state_dict(), ema_path)

            # Validation Loop
            model.eval()
            val_loss_total = 0.0
            val_steps = 0

            # Limit validation batches to save time during training
            max_val_batches = 50

            logger.info(f"Running validation at end of epoch {epoch}...")

            with torch.no_grad():
                for val_x, val_y in val_loader:
                    if val_steps >= max_val_batches:
                        break

                    loss, loss_dict, x_pred = pmf_model.forward_loss(val_x, val_y)
                    val_loss_total += loss_dict['loss_total']
                    val_steps += 1

            avg_val_loss = val_loss_total / max(val_steps, 1)

            writer.add_scalar("val/loss_total", avg_val_loss, global_step)
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")
            '''
            val during training
            '''
            with torch.no_grad():
                # Sample some images
                progress_bar = tqdm(val_loader, disable=not accelerator.is_local_main_process)
                progress_bar.set_description(f"Epoch {epoch}")
                for lq, gt in progress_bar:
                    with accelerator.accumulate(model):
                        z = lq
                        samples = pmf_model.sample(z, None, cfg_scale=1.0)
                        # Save images
                        from torchvision.utils import save_image
                        # Denormalize: [-1, 1] -> [0, 1]
                        samples_norm = samples
                        save_path = os.path.join(config.output_dir, f"sample_epoch_{epoch}.png")
                        os.makedirs(config.output_dir, exist_ok=True)
                        save_image(samples_norm, save_path)

                # Log images to TensorBoard
                writer.add_images("val/samples", samples_norm, global_step)

            model.train()



    if accelerator.is_local_main_process:
        print("Training complete.")
        # Save final
        ckpt_manager.save(model, "final", loss=running_loss)
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default='/home/ybb/Project/BasicDFIR/pMF-main/configs/pMFSR-B-16.yaml',
                        help="Path to YAML config. Overrides PMF_CONFIG.")
    args = parser.parse_args()
    train(args.config)
