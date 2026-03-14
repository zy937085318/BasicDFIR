# Adapted from https://github.com/annegnx/PnP-Flow

import lpips
import torch

from ignite.metrics import SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn_alex = lpips.LPIPS(net='alex').to(DEVICE)  # best forward scores


def compute_psnr(clean_img, noisy_img, rec_img, problem, H_adj):
    # Ensure images are in the appropriate range and format for PSNR calculation
    H_adj_noisy_img = H_adj(noisy_img).cpu()
    clean_img = clean_img.cpu()
    noisy_img = noisy_img.cpu()
    rec_img = rec_img.cpu()

    clean_img = clean_img.permute(0, 2, 3, 1).cpu().data.numpy()
    if problem == 'superresolution':
        noisy_img = H_adj_noisy_img.permute(0, 2, 3, 1).cpu().data.numpy()
    else:
        noisy_img = noisy_img.permute(0, 2, 3, 1).cpu().data.numpy()
    rec_img = rec_img.permute(0, 2, 3, 1).cpu().data.numpy()

    # Compute PSNR values
    psnr_rec = PSNR(clean_img, rec_img, data_range=1.0)
    psnr_noisy = PSNR(clean_img, noisy_img, data_range=1.0)
    return psnr_rec, psnr_noisy


def compute_lpips(clean_img, noisy_img, rec_img, problem, H_adj):
    # Ensure images are in the appropriate range and format for LPIPS calculation
    H_adj_noisy_img = H_adj(noisy_img).cpu()
    clean_img = clean_img.cpu()
    noisy_img = noisy_img.cpu()
    rec_img = rec_img.cpu()

    # Permute images to NCHW format and move to the correct device
    clean_img = clean_img.to(DEVICE)
    rec_img = rec_img.to(DEVICE)

    if problem == 'superresolution':
        noisy_img = H_adj_noisy_img.to(DEVICE)
    else:
        noisy_img = noisy_img.to(DEVICE)

    # Ensure images are in the expected format (N, C, H, W) and range [-1, 1] for LPIPS
    clean_img = 2 * clean_img - 1
    rec_img = 2 * rec_img - 1
    noisy_img = 2 * noisy_img - 1

    # Compute LPIPS values
    lpips_rec = loss_fn_alex(clean_img, rec_img, normalize=True).mean().item()
    lpips_noisy = loss_fn_alex(
        clean_img, noisy_img, normalize=True).mean().item()

    return lpips_rec, lpips_noisy


def compute_ssim(clean_img, noisy_img, rec_img, problem, H_adj):
    # Ensure images are in the appropriate range and format for SSIM calculation
    H_adj_noisy_img = H_adj(noisy_img).cpu()
    clean_img = clean_img.cpu()
    noisy_img = noisy_img.cpu()
    rec_img = rec_img.cpu()

    # Convert images to the appropriate format for SSIM calculation
    if problem == 'superresolution':
        noisy_img = H_adj_noisy_img
    else:
        noisy_img = noisy_img

    # Initialize SSIM metric for restored and noisy images
    ssim_metric = SSIM(data_range=1.0)
    ssim_metric_noisy = SSIM(data_range=1.0)

    # Compute SSIM values
    ssim_metric.update((rec_img, clean_img))
    ssim_rec = ssim_metric.compute()
    ssim_metric_noisy.update((noisy_img, clean_img))
    ssim_noisy = ssim_metric_noisy.compute()

    return ssim_rec, ssim_noisy
