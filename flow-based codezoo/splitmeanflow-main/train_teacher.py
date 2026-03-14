from dit_a import DiT
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict

from model import RectifiedFlow, LogitNormalCosineScheduler
# from fid_evaluation import FIDEvaluation
# import moviepy.editor as mpy
# from comet_ml import Experiment
import os
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from basicsr.utils import yaml_load
from basicsr.data import build_dataset


def get_srloader(dataset_opt, shuffle=True, batch_size=64):
    dataset = build_dataset(dataset_opt)
    if shuffle is True:
        batch_size = batch_size
        prefetch_factor = 4
    else:
        batch_size = 1
        prefetch_factor =1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        num_workers=batch_size,
        pin_memory=True,
        drop_last=True
    )
    return loader


def main():
    n_steps = 400000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    
    image_size = 256
    image_channels = 3
    num_classes = None
    
    # Create directories
    os.makedirs('./images_DF2K', exist_ok=True)
    os.makedirs('./results_DF2K', exist_ok=True)
    checkpoint_root_path = './checkpoint/dit_DF2K/'
    os.makedirs(checkpoint_root_path, exist_ok=True)

    # # Initialize Comet ML experiment
    # experiment = Experiment(
    #     project_name="dit-flow-matching-DF2K-SRx8",
    # )
    
   
    # experiment.log_parameters({
    #     "dataset": "DF2Kx8",
    #     "n_steps": n_steps,
    #     "batch_size": batch_size,
    #     "learning_rate": 1e-4,
    #     "model": "DiT-DF2K",
    #     "dim": 384,
    #     "depth": 12,
    #     "num_heads": 6,
    #     "patch_size": 2,
    #     "dropout_prob": 0.1,
    #     "optimizer": "Adam",
    #     "mixed_precision": "bfloat16",
    #     "fid_subset_size": 1000,
    #     "image_size": image_size,
    #     "image_channels": image_channels,
    #     "training_cfg_rate": 0.2,
    #     "lambda_weight": 0.05,
    #     "sigma_min": 1e-06,
    #     "ema_decay": 0.9999,
    # })

    
    transform = T.Compose([
        T.ToTensor(),  # Converts to [0, 1]
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  
    ])
    dataset_opt = yaml_load('/home/ybb/Project/BasicDFIR/options/train/PixelMeanFlow/train_PFM_DiT_B8_SRx2_P256_github_pfm_source.yml')['datasets']
    # Dataset
    train_opt = dataset_opt['train']
    train_opt['phase'] = 'train'
    val_opt = dataset_opt['val_1']
    val_opt['phase'] = 'val'
    train_dataloader = get_srloader(train_opt, shuffle=True, batch_size=batch_size)
    val_dataloader = get_srloader(val_opt, shuffle=False, batch_size=1)

   
    timestep_scheduler = LogitNormalCosineScheduler(
        loc=0.0,        
        scale=1.0,      
        min_t=1e-8,    
        max_t=1.0-1e-8 
    )

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i

    train_dataloader = cycle(train_dataloader)
    # Create model for CIFAR-10 generation
    model = DiT(
        input_size=256,
        patch_size=64,
        in_channels=3,
        hidden_size=512,
        depth=24,
        num_heads=16,
        num_classes=None,
        learn_sigma=False,
        class_dropout_prob=0.1,
    ).to(device)
    
    # Create EMA model
    ema_model = deepcopy(model).eval()
    ema_decay = 0.9999
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0001)
    
    # Create sampler (use EMA model for sampling)
    sampler = RectifiedFlow(
        ema_model,  # Use EMA model
        device=device,
        channels=image_channels,
        image_size=image_size,
        num_classes=num_classes,
        use_logit_normal_cosine=True,
        logit_normal_loc=0.0,
        logit_normal_scale=1.0,
        timestep_min=1e-8,
        timestep_max=1.0-1e-8,
    )
    
    scaler = torch.cuda.amp.GradScaler()

    
    # fid_subset_size = 1000
    #
    def limited_cycle(iterable, max_batches):
        batch_count = 0
        while batch_count < max_batches:
            for i in iterable:
                yield i
                batch_count += 1
                if batch_count >= max_batches:
                    break
    #
    # fid_batches_needed = (fid_subset_size + batch_size - 1) // batch_size
    #
    # fid_dataset = torchvision.datasets.CIFAR10(
    #     root="/mnt/nvme",
    #     train=True,
    #     download=False,
    #     transform=transform,
    # )
    
    # fid_dataloader = torch.utils.data.DataLoader(
    #     fid_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     drop_last=False,
    #     num_workers=4
    # )
    
    # val_dataloader_limited = limited_cycle(val_dataloader, 1)
    # fid_eval = FIDEvaluation(batch_size, fid_dataloader_limited, sampler, num_fid_samples=100)
    
    def update_ema(ema_model, model, decay):
        """Update EMA model parameters"""
        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.lerp_(p.data, 1 - decay)
    
    def sample_and_log_images():
        """Sample images from model"""
        ema_model.eval()
        with torch.no_grad():
            sample_list = []
            for i, sample in enumerate(val_dataloader):
                lq, gt = sample[0].to(device), sample[1].to(device)
                samples = sampler.sample_img(lq, 10, return_all_steps=False)
                # print((lq-samples).sum())
            sample_list.append(lq)
            sample_list.append(samples)
            sample_list.append(gt)
            sample_list = torch.cat(sample_list, dim=0)
            log_img = make_grid(sample_list, nrow=3, padding=5, normalize=False, value_range=(0, 1))
            img_save_path = f"./results_DF2K/step{step}.png"
            save_image(log_img, img_save_path)

            # experiment.log_image(
            #     img_save_path,
            #     name=f"cfg_{cfg_scale}",
            #     step=step
            # )


            # selected_indices = list(range(0, len(trajectory), 5))
            # images_list = []
            # for idx in selected_indices:
            #     frame = trajectory[idx]
            #
            #     frame = (frame + 1) / 2
            #     frame = frame.clamp(0, 1)
            #
            #     grid = make_grid(frame, nrow=10)
            #     images_list.append(
            #         grid.permute(1, 2, 0).cpu().numpy() * 255
            #     )
            #
            # clip = mpy.ImageSequenceClip(images_list, fps=10)
            # gif_path = f"/mnt/nvme/images_cifar10/step{step}_cfg{cfg_scale}.gif"
            # clip.write_gif(gif_path)
            #
            # experiment.log_image(
            #     gif_path,
            #     name=f"trajectory_cfg_{cfg_scale}",
            #     step=step,
            #     image_format="gif")
    
    losses = []
    sigma_min = 1e-06
    training_cfg_rate = 0.2
    lambda_weight = 0.001
    use_immiscible = False
    gradient_clip = 1.0
    
    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training DiT on DF2K")
        for step in pbar:
            lq, gt = next(train_dataloader)
            optimizer.zero_grad()
            lq = lq.to(device)
            gt = gt.to(device)
            # lq = data[0].to(device) #lq
            # gt = data[1].to(device) #gt
            b = lq.shape[0]

           
            t = torch.rand(b, device=device)

            alpha_t = t
            sigma_t = 1 - (1 - sigma_min) * t

           
            alpha_t = alpha_t.view(b, 1, 1, 1)
            sigma_t = sigma_t.view(b, 1, 1, 1)

          
            if use_immiscible:
                k = 4
               
                z_candidates = torch.randn(b, k, image_channels, image_size, image_size, device=lq.device, dtype=lq.dtype)
                
                x1_flat = lq.flatten(start_dim=1)  # [b, c*h*w]
                z_candidates_flat = z_candidates.flatten(start_dim=2)  # [b, k, c*h*w]
                
                
                distances = torch.norm(x1_flat.unsqueeze(1) - z_candidates_flat, dim=2)  # [b, k]
                
                
                min_distances, min_indices = torch.min(distances, dim=1)  # [b]
                
               
                batch_indices = torch.arange(b, device=lq.device)
                z = z_candidates[batch_indices, min_indices]  # [b, c, h, w]
                
            else:
                # Standard noise sampling
                z = lq
                
           
            x_t = sigma_t * z + alpha_t * gt

           
            u_positive = gt - (1 - sigma_min) * z
            
         
            if b > 1:
                perm = torch.randperm(b, device=lq.device)
                # Ensure no self-pairing
                for i in range(b):
                    if perm[i] == i:
                        perm[i] = (i + 1) % b
                u_negative = u_positive[perm]
            else:
                u_negative = u_positive

            # if training_cfg_rate > 0:
            #     drop_mask = torch.rand(b, device=x1.device) < training_cfg_rate
            #     y = torch.where(drop_mask, num_classes, y)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # print(x_t.shape)
                # print(t.shape)
                pred = model(x_t, t, None)
                # print(model.final_layer.linear.weight.sum())
                
                # Compute losses
                positive_loss = F.mse_loss(pred, u_positive)
                negative_loss = F.mse_loss(pred, u_negative)
                loss = positive_loss - lambda_weight * negative_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA
            update_ema(ema_model, model, ema_decay)

            # Logging
            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item(), "grad_norm": grad_norm.item()})
            # experiment.log_metric("loss", loss.item(), step=step)
            # experiment.log_metric("grad_norm", grad_norm.item(), step=step)
            # experiment.log_metric("positive_loss", positive_loss.item(), step=step)
            # experiment.log_metric("negative_loss", negative_loss.item(), step=step)

            if (step+1) % 100 == 0:
                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                # experiment.log_metric("avg_loss_100", avg_loss, step=step)
                print("avg loss: ", avg_loss)
            if (step+1) % 2000 == 0 or step == n_steps - 1:
                # avg_loss = sum(losses) / len(losses) if losses else 0
                # print(f"\nStep: {step+1}/{n_steps} | avg_loss: {avg_loss:.4f}")

                losses.clear()
                
                sample_and_log_images()

            # if step % 10000 == 0 or step == n_steps - 1:
            #     # FID evaluation
            #     try:
            #         print(f"Running FID evaluation on {fid_subset_size} samples...")
            #         with torch.autocast(device_type='cuda', dtype=torch.float32):
            #             fid_score = fid_eval.fid_score()
            #         print(f"FID score at step {step}: {fid_score}")
            #         experiment.log_metric("FID", fid_score, step=step)
            #     except Exception as e:
            #         print(f"FID evaluation failed: {e}")
                
                # Save checkpoint
                if (step + 1) % 5000 == 0 or step == n_steps - 1:
                    checkpoint_path = os.path.join(checkpoint_root_path, f"step_{step}.pth")
                    state_dict = {
                        "model": model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "config": {
                            "dataset": "DF2K",
                            "image_size": image_size,
                            "image_channels": image_channels,
                            "sigma_min": sigma_min,
                            "lambda_weight": lambda_weight,
                            "training_cfg_rate": training_cfg_rate,
                            "gradient_clip": gradient_clip,
                            "ema_decay": ema_decay,
                        }
                    }
                    torch.save(state_dict, checkpoint_path)
                
                    # experiment.log_model(
                    #     name=f"checkpoint_step_{step}",
                    #     file_or_folder=checkpoint_path
                    # )

    checkpoint_path = os.path.join(checkpoint_root_path, "model_DF2K10x8_final.pth")
    state_dict = {
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "config": {
            "dataset": "DF2K",
            "image_size": image_size,
            "image_channels": image_channels,
            "sigma_min": sigma_min,
            "lambda_weight": lambda_weight,
            "training_cfg_rate": training_cfg_rate,
            "gradient_clip": gradient_clip,
            "ema_decay": ema_decay,
        }
    }
    torch.save(state_dict, checkpoint_path)
    
    # experiment.log_model(
    #     name="final_model",
    #     file_or_folder=checkpoint_path
    # )
    #
    # experiment.end()


if __name__ == "__main__":
    main()