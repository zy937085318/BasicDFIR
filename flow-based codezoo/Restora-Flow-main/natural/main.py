# Parts of this file are adapted from https://github.com/annegnx/PnP-Flow

import argparse
import os
import random
import time

import torch.backends.cudnn as cudnn

from utils.degradations import *
from helpers import define_unet, load_cfg_from_cfg_file, load_model, merge_cfg_from_list
from src.dataloaders import DataLoaders
from src.methods.ddnm import DDNM
from src.methods.d_flow import DFlow
from src.methods.flow_priors import FlowPriors
from src.methods.ot_ode import OTOde
from src.methods.pnp_flow import PnPFlow
from src.methods.repaint import RePaint
from src.methods.restora_flow import RestoraFlow
from src.train_flow_matching import FLOW_MATCHING


torch.cuda.empty_cache()


def parse_args(method_name=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Main')

    cfg = load_cfg_from_cfg_file('config/main_config.yaml')

    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)

    dataset_config = cfg.root + 'config/dataset_config/{}.yaml'.format(cfg.dataset)
    cfg.update(load_cfg_from_cfg_file(dataset_config))

    if method_name is None:
        method_name = cfg.method
    method_config_file = cfg.root + 'config/method_config/{}.yaml'.format(method_name)
    cfg.update(load_cfg_from_cfg_file(method_config_file))

    if args.opts is not None:
        # override config with command line input
        cfg = merge_cfg_from_list(cfg, args.opts)

    # for all keys in the method config file, create a dictionary {key: value} in the cfg object cfg.dict_cfg_method
    method_cfg = load_cfg_from_cfg_file(method_config_file)
    cfg.dict_cfg_method = {}
    for key in method_cfg.keys():
        cfg.dict_cfg_method[key] = cfg[key]
    return cfg


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    if args.train:
        print('Training...')
        data_loaders = DataLoaders(
            args.dataset, args.batch_size_train, args.batch_size_train, args.dim_image, train=True).load_data()
        unet = define_unet(args, device)

        if args.model_type == "ot":
            generative_model = FLOW_MATCHING(unet, device, args)
        elif args.model_type == "ddpm":
            raise Exception("DDPM training currently not implemented.")
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")

        generative_model.train(data_loaders)
        print('Training done!')

    if args.eval:
        print('Starting evaluation...')
        # Load model
        model_checkpoint_path = os.path.join(args.root, 'model_checkpoints', args.dataset, args.latent, args.model_type,
                                             'model_final.pt')
        unet = define_unet(args, device)
        model = load_model(unet, args.model_type, model_checkpoint_path, device)

        # Input noise
        sigma_noise = args.sigma_y

        if args.problem == "denoising":
            sigma_noise = 0.2
            additional_dir_name = f"sigma_{sigma_noise}"
            degradation = Denoising()

        elif args.problem == "box_inpainting":
            additional_dir_name = f"size_{args.mask_size_x}x{args.mask_size_y}"
            # degradation = DiverseMaskInpainting(mask_type='diagonal')
            degradation = BoxInpainting((args.mask_size_x, args.mask_size_y))

        elif args.problem == "random_inpainting":
            additional_dir_name = f'p_{args.p_value}'
            degradation = RandomInpainting(args.p_value)

        elif args.problem == "superresolution":
            additional_dir_name = f'sf_{args.sf}'
            degradation = Superresolution(args.sf, args.dim_image)

        else:
            raise Exception("Problem not supported.")

        print(f'Solving {args.problem} with {args.method}...')

        # Load data
        data_loaders = DataLoaders(args.dataset, args.batch_size_ip, args.batch_size_ip, args.dim_image).load_data()

        # Create output directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.save_path_ip = os.path.join(args.root, 'results', args.dataset, args.model_type, args.problem,
                                         additional_dir_name, args.method, args.eval_split, timestamp)
        os.makedirs(args.save_path_ip, exist_ok=True)
        print("Output will be saved to ", args.save_path_ip)

        # Run restoration method
        if args.method == 'pnp_flow':
            method = PnPFlow(model, device, args)
        elif args.method == 'd_flow':
            method = DFlow(model, device, args)
        elif args.method == 'ot_ode':
            method = OTOde(model, device, args)
        elif args.method == 'flow_priors':
            method = FlowPriors(model, device, args)
        elif args.method == 'restora_flow':
            method = RestoraFlow(model, device, args)
        elif args.method == 'repaint':
            method = RePaint(model, device, args)
        elif args.method == 'ddnm':
            method = DDNM(model, device, args)
        else:
            raise ValueError(f"Unsupported method: {args.method}")

        return method.run_method(data_loaders, degradation, sigma_noise)


if __name__ == "__main__":
    main()
