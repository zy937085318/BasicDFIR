import os
import shutil
import time

import torch
from tqdm.auto import tqdm

import helpers as helpers
from utils import scheduler
from utils.image_metrics import compute_psnr, compute_ssim, compute_lpips


class RestoraFlow(object):
    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model.to(device)

    def model_forward(self, x, t):
        return self.model(x, t)

    def sample_denoising(self, input_img):
        steps_ode = self.args.steps_ode
        device = input_img.device

        x = torch.randn_like(input_img, device=device)  # initialize x with noise
        x_obs = input_img * (1 - self.args.sigma_noise)

        torch_linspace = torch.linspace(0, 1, int(steps_ode), device=device)
        delta_t = 1 / len(torch_linspace)

        for t in torch_linspace:
            mask = torch.ones(input_img.shape, device=device)

            if t < (1 - self.args.sigma_noise):
                x = mask * x_obs + (1 - mask) * x
            else:
                x = x + delta_t * self.model(x, torch.tensor(t, device=device).repeat(x.shape[0]))

        return x

    def sample_mask_based(self, input_img, mask, sample_id=0, progress=False, debug=False):
        batch_size = input_img.shape[0]
        output_folder = self.args.save_path_ip

        if debug:
            helpers.save_image(input_img, output_folder, 'input_img.png')
            helpers.save_image(mask, output_folder, 'mask.png')

        x = torch.randn_like(input_img, device=self.device)  # initialize x with noise

        pred_x_start = None
        steps_ode = self.args.steps_ode
        correction_steps = self.args.correction_steps

        if correction_steps < 1:
            raise ValueError("Number of correction steps must be ≥ 1.")

        times = scheduler.get_schedule_jump(
            t_T=steps_ode,
            n_sample=1,
            jump_length=1,
            jump_n_sample=correction_steps+1
        )

        # normalize
        times = [((x - min(times)) / (max(times) - min(times))) for x in times]
        times.reverse()
        time_pairs = list(zip(times[:-1], times[1:]))

        if progress:
            time_pairs = tqdm(time_pairs)

        for t_last, t_cur in time_pairs:
            if debug:
                print("t_last, t_cur: ", t_last, t_cur)

            t_last_t = torch.tensor([t_last] * batch_size, device=self.device).view(batch_size, 1, 1, 1)
            t_cur_t = torch.tensor([t_cur] * batch_size, device=self.device).view(batch_size, 1, 1, 1)

            if t_last < t_cur:
                with torch.no_grad():
                    if pred_x_start is not None:
                        # mask-based update
                        eps = torch.randn_like(x)
                        z_prim = t_last_t * input_img + (1 - t_last_t) * eps
                        x = mask * z_prim + (1 - mask) * x

                        if debug:
                            known = (mask * z_prim)
                            helpers.save_image(known, output_folder, f'{sample_id}_known.png')

                            unknown = (1 - mask) * x
                            helpers.save_image(unknown, output_folder, f'{sample_id}_unknown.png')

                    # flow update
                    delta_t = (t_cur_t - t_last_t)
                    x = x + delta_t * self.model(x, torch.tensor(t_last, device=self.device).repeat(batch_size))
                    out_sample = x.clone()

                    if debug:
                        helpers.save_image(out_sample, output_folder, f'out_sample.png')

                    pred_x_start = True
            else:
                # trajectory correction
                x_1_prim = x + (1 - t_last_t) * self.model(x, torch.tensor(t_last, device=self.device).repeat(batch_size))
                x = t_cur_t * x_1_prim + (1 - t_cur_t) * torch.randn_like(x)

        return out_sample

    def solve_ip(self, test_loader, degradation):
        H, H_adj = degradation.H, degradation.H_adj

        loader = iter(test_loader)
        psnrs, ssims, lpips = [], [], []

        for batch in range(self.args.max_batch):
            self.args.batch = batch

            (clean_img, labels) = next(loader)
            noisy_img = H(clean_img.clone().to(self.device))
            torch.manual_seed(batch)
            noisy_img += torch.randn_like(noisy_img) * self.args.sigma_noise
            noisy_img, clean_img = noisy_img.to(self.device), clean_img.to('cpu')

            # initialize the image with the adjoint operator (i.e. create a mask)
            x = H_adj(torch.ones_like(noisy_img)).to(self.device)  # ones_like

            with torch.no_grad():
                if self.args.problem == 'denoising':
                    output = self.sample_denoising(input_img=noisy_img)
                elif self.args.problem == 'superresolution':
                    superresolution_input = clean_img.to(self.device) * x + torch.randn_like(
                        clean_img.to(self.device)) * self.args.sigma_noise
                    # helpers.save_image(superresolution_input, output_folder, 'superresolution_input.png')
                    output = self.sample_mask_based(input_img=superresolution_input, mask=x)
                else:   # inpainting
                    output = self.sample_mask_based(input_img=noisy_img, mask=x)

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

            # Save results
            helpers.save_images(clean_img, noisy_img, restored_img, self.args, H_adj)

        return psnrs, ssims, lpips

    def run_method(self, data_loaders, degradation, sigma_noise):
        print(f'Params: steps_ode={self.args.steps_ode}, '
                f'correction_steps={self.args.correction_steps}\n')

        self.args.sigma_noise = sigma_noise

        # Copy configuration and run files
        files_to_copy = [
            os.path.join(self.args.root, 'config', 'method_config', f'{self.args.method}.yaml'),
            os.path.join(self.args.root, 'src', 'methods', 'restora_flow.py'),
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
                    f'Params: steps_ode={self.args.steps_ode}, '
                    f'correction_steps={self.args.correction_steps}\n'
                    f'---------------------------------------------------------\n'
                )

                for idx, (psnr, ssim, lpip) in enumerate(zip(psnrs, ssims, lpips)):
                    file.write(f'Batch {idx}: PSNR = {psnr:.4f}, SSIM = {ssim:.4f}, LPIPS = {lpip:.4f}\n')

                file.write(f'---------------------------------------------------------\n')
                file.write(f'Average PSNR = {avg_psnr:.4f}\n')
                file.write(f'Average SSIM = {avg_ssim:.4f}\n')
                file.write(f'Average LPIPS = {avg_lpips:.4f}\n')
                file.write(f'Total time = {total_time}\n')
