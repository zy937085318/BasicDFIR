import torch
import os
from tqdm import tqdm
from pmf.config import get_config
from pmf.dit import DiT
from pmf.pixel_mean_flow import PixelMeanFlow
from torchvision.utils import save_image
import argparse

def evaluate(args):
    config = get_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    pmf_model = PixelMeanFlow(model, config)
    pmf_model.to(device) # Just for consistency, though wrapper doesn't have much state

    os.makedirs(args.output_dir, exist_ok=True)

    num_samples = args.num_samples
    batch_size = args.batch_size

    print(f"Generating {num_samples} samples...")

    if args.cfg_interval is None:
        cfg_interval = None
    else:
        a, b = args.cfg_interval
        cfg_interval = torch.tensor([[float(a), float(b)]], device=device)

    for i in tqdm(range(0, num_samples, batch_size)):
        curr_batch = min(batch_size, num_samples - i)

        z = torch.randn(curr_batch, config.in_channels, config.image_size, config.image_size, device=device)

        if args.class_idx is not None:
            y = torch.full((curr_batch,), args.class_idx, device=device, dtype=torch.long)
        else:
            y = torch.randint(0, config.num_classes-1, (curr_batch,), device=device)

        with torch.no_grad():
            if cfg_interval is None:
                interval_in = None
            else:
                interval_in = cfg_interval.repeat(curr_batch, 1)
            samples = pmf_model.sample(z, y, cfg_scale=args.cfg_scale, cfg_interval=interval_in)

        # Denormalize
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)

        for j in range(curr_batch):
            save_image(samples[j], os.path.join(args.output_dir, f"sample_{i+j}.png"))

    print("Done.")
    print(f"Samples saved to {args.output_dir}")
    print("To compute FID, use: fidelity --gpu 0 --fid --input1 {} --input2 /path/to/dataset".format(args.output_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config. Overrides PMF_CONFIG.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="eval_samples")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--cfg_interval", type=float, nargs=2, default=None, metavar=("A", "B"))
    parser.add_argument("--class_idx", type=int, default=None, help="Generate specific class")
    args = parser.parse_args()

    evaluate(args)
