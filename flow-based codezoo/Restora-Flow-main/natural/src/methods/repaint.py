import os
import time
import shutil

import torch

import helpers as helpers
from utils.image_metrics import compute_psnr, compute_ssim, compute_lpips


class RePaint(object):
    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.method = args.method

    def solve_ip(self, test_loader, degradation):
        H, H_adj = degradation.H, degradation.H_adj
        psnrs, ssims, lpips = [], [], []

        loader = iter(test_loader)
        for batch in range(self.args.max_batch):
            self.args.batch = batch

            (clean_img, labels) = next(loader)
            noisy_img = H(clean_img.clone().to(self.device))
            torch.manual_seed(batch)
            noisy_img += torch.randn_like(noisy_img) * self.args.sigma_noise
            noisy_img, clean_img = noisy_img.to(self.device), clean_img.to('cpu')

            # initialize the image with the adjoint operator (i.e. create a mask)
            x = H_adj(torch.ones_like(noisy_img)).to(self.device)

            with torch.no_grad():
                if self.args.problem == 'superresolution':
                    # Superresolution input for RePaint
                    superresolution_input = clean_img.to(self.device) * x + torch.randn_like(
                        clean_img.to(self.device)) * self.args.sigma_noise
                    # helpers.save_image(superresolution_input, self.args.save_path_ip, 'superresolution_input.png')
                    output = self.model.p_sample_loop_progressive_repaint(
                        x0=superresolution_input,
                        mask=x,
                        shape=superresolution_input.shape,
                        output_folder=self.args.save_path_ip,
                        jump_length=self.args.jump_length,
                        jump_n_sample=self.args.jump_n_sample,
                        device="cuda",
                        progress=False,
                        clip_denoised=False)
                else:
                    output = self.model.p_sample_loop_progressive_repaint(
                        x0=noisy_img,
                        mask=x,
                        shape=noisy_img.shape,
                        output_folder=self.args.save_path_ip,
                        jump_length=self.args.jump_length,
                        jump_n_sample=self.args.jump_n_sample,
                        device="cuda",
                        progress=True,
                        clip_denoised=False
                    )
            restored_img = output.detach().clone()

            if self.args.compute_metrics:
                psnr_rec, psnr_noisy = compute_psnr(clean_img, noisy_img, restored_img, self.args, H_adj)
                print(f"Batch {batch}: psnr_rec={psnr_rec}, psnr_noisy={psnr_noisy}")
                psnrs.append(psnr_rec)
                ssim_rec, ssim_noisy = compute_ssim(clean_img, noisy_img, restored_img, self.args, H_adj)
                print(f"Batch {batch}: ssim_rec={ssim_rec}, ssim_noisy={ssim_noisy}")
                ssims.append(ssim_rec)
                lpip_rec, lpip_noisy = compute_lpips(clean_img, noisy_img, restored_img, self.args, H_adj)
                lpips.append(lpip_rec)

            helpers.save_images(clean_img, noisy_img, restored_img, self.args, H_adj)

        return psnrs, ssims, lpips

    def run_method(self, data_loaders, degradation, sigma_noise):
        print(f'Params: timesteps={self.args.timesteps}, '
              f'jump_length={self.args.jump_length}, '
              f'jump_n_sample={self.args.jump_n_sample}\n')

        self.args.sigma_noise = sigma_noise

        # Copy configuration and run files
        files_to_copy = [
            os.path.join(self.args.root, 'config', 'method_config', f'{self.args.method}.yaml'),
            os.path.join(self.args.root, 'src', 'methods', 'repaint.py'),
        ]

        for f in files_to_copy:
            if os.path.isfile(f):
                shutil.copy2(f, self.args.save_path_ip)

        # Solve inverse problem
        start = time.time()
        psnrs, ssims, lpips = self.solve_ip(data_loaders[self.args.eval_split], degradation)
        total_time = round(time.time() - start, 4)

        # Compute metrics
        if self.args.compute_metrics:
            avg_psnr = sum(psnrs) / len(psnrs)
            avg_ssim = sum(ssims) / len(ssims)
            avg_lpips = sum(lpips) / len(lpips)

            print(f"Total time = {total_time:.4f}")
            print(f"Average PSNR = {avg_psnr:.4f}")
            print(f"Average SSIM = {avg_ssim:.4f}")
            print(f"Average LPIPS = {avg_lpips:.4f}")

            # Save evaluation results
            eval_file = os.path.join(self.args.save_path_ip, 'eval.txt')
            with open(eval_file, 'a') as file:
                file.write(
                    f'Params: timesteps={self.args.timesteps}, '
                    f'jump_length={self.args.jump_length}, '
                    f'jump_n_sample={self.args.jump_n_sample}\n'
                    f'---------------------------------------------------------\n'
                )

                for idx, (psnr, ssim, lpip) in enumerate(zip(psnrs, ssims, lpips)):
                    file.write(f'Batch {idx}: PSNR = {psnr:.4f}, SSIM = {ssim:.4f}, LPIPS = {lpip:.4f}\n')

                file.write(f'---------------------------------------------------------\n')
                file.write(f'Average PSNR = {avg_psnr:.4f}\n')
                file.write(f'Average SSIM = {avg_ssim:.4f}\n')
                file.write(f'Average LPIPS = {avg_lpips:.4f}\n')
                file.write(f'Total time = {total_time}\n')
