import argparse
import yaml
import os
import time

from runners.hand_restoration import HandRestoration
from utils import degradations


def load_config(path="medical/configs/restoration_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_degradation(problem, problem_cfg, input_res):
    method_params = problem_cfg.get("args", [])

    if problem == "denoising":
        return degradations.Denoising()

    elif problem == "box_inpainting":
        mask_size = method_params
        return degradations.BoxInpainting(mask_size)

    elif problem == "superresolution":
        sf = method_params[0]
        return degradations.Superresolution(sf, input_res)

    elif problem == "occlusion_removal":
        return degradations.OcclusionsSimulation()

    else:
        raise ValueError(f"Unknown problem type '{problem}'")


def main():
    parser = argparse.ArgumentParser(description="Run hand restoration experiments.")

    parser.add_argument("--model_type", choices=["flow", "ddpm"], required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--config", default="configs/restoration_config.yaml")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.problem not in cfg["problems"]:
        raise ValueError(f"Unknown problem: {args.problem}")

    # Degradation operator
    degradation = build_degradation(args.problem, cfg["problems"][args.problem], cfg["input_resolution"])

    # Input noise
    sigma_y = cfg["sigma_overrides"].get(args.problem, cfg["default_sigma_y"])

    # Method-specific parameters
    method_cfg = cfg["methods"][args.method]
    params = method_cfg["overrides"].get(args.problem, method_cfg["default_params"])

    # Create output folder
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_folder = os.path.join(
        "exports/hand_samples",
        args.model_type,
        args.problem,
        args.method,
        timestamp,
    )
    os.makedirs(output_folder, exist_ok=True)

    # Run restoration
    restorer = HandRestoration(args.model_type)
    restorer.load_model()

    print(f"Running {args.problem} using {args.method}...")
    print(f"Params: {params}")
    start = time.time()
    psnrs, ssims, lpips = restorer.run(
        args.problem,
        args.method,
        degradation,
        sigma_y,
        params,
        output_folder,
    )
    total_time = time.time() - start

    # Log results
    def compute_average(values, default=0.0):
        if args.problem == 'occlusion_removal' and not values:
            print('No occlusions added to images.')
            exit()
        return sum(values) / len(values) if values else default

    avg_psnr = compute_average(psnrs)
    avg_ssim = compute_average(ssims)
    avg_lpip = compute_average(lpips)

    with open(os.path.join(output_folder, "eval.txt"), "w") as f:
        for i, (p, s, l) in enumerate(zip(psnrs, ssims, lpips)):
            f.write(f"Batch {i}: psnr={p:.4f}, ssim={s:.4f}, lpip={l:.4f}\n")
        f.write(f"Params: {params}\n")
        f.write(f"Avg PSNR: {avg_psnr:.4f}\n")
        f.write(f"Avg SSIM: {avg_ssim:.4f}\n")
        f.write(f"Avg LPIPS: {avg_lpip:.4f}\n")
        f.write(f"Total time: {total_time:.4f}\n")

    print("Output saved to:", output_folder)


if __name__ == "__main__":
    main()
