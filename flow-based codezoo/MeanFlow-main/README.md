# MeanFlow: Pytorch Implementation
This repository contains a minimalist PyTorch implementation of MeanFlow, a novel single-step flow matching model for high-quality image generation.

## Reproduced ImageNet Results

| Model | Epoch | FID(NFE=1), our results| FID(NFE=1), results in paper|
|---------------|---------------|----------------|----------------|
|SiT-B/4(wo cfg)| 80 |58.74|61.06, Table 1f|
|SiT-B/4 | 80 |15.43|15.53, Table 1f|
|SiT-B/2 | 240 |6.06|6.17, Table 2|
|SiT-L/2 | 240 |3.94(140/240)|3.84, Table 2|
|SiT-XL/2 | 240 |3.39(200/240)|3.43, Table 2|

**Note**: **All the weights trained on ImageNet256 are availavle at [here](https://drive.google.com/drive/folders/1oWt6tdm5WIeVaZnBuUVheKIG3cNDffl9?usp=drive_link)**.

## DO NOT overlook the pretrained flow matching model：Fine-tuning Pretrained Flow Matching Models with MeanFlow
| Model | FID(NFE=1), our results| FID(NFE=2), our results|FID(NFE=2), results in paper|
|---------------|---------------|----------------|----------------|
|SiT-XL/2(w cfg) + [pretrained weights](https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0) (1400 epoch)|4.52|2.81 (1400+20+40)|2.93, 240 epoch, Table 2|
|SiT-XL/2(w cfg) + [pretrained weights](https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0) (1400 epoch)|15.50|2.55 (1400+20+110)|2.20, 1000 epoch, Table 2|

**Tips**: Direct fine-tuning using MeanFlow with classifier-free guidance (CFG) exhibits training instability. To address this issue, we adopt a staged training strategy: initially fine-tuning with MeanFlow without CFG for 20 epochs, followed by continued fine-tuning with CFG-enabled MeanFlow.


**Notes**: 
1. When evaluating models trained with CFG , the --cfg-scale parameter must be set to 1.0 during inference, as the CFG guidance has been incorporated into the model during training and is no longer controllable at sampling time.
2. We currently use [sd-vae-ft-ema](https://huggingface.co/stabilityai/sd-vae-ft-mse), which is not the suggested tokenizer in original paper ([sd-vae-ft-mse](https://huggingface.co/pcuenq/sd-vae-ft-mse-flax)). **Maybe replacing with ```sd-vae-ft-mse``` would yield better results**.

## Installation

```bash
# Clone this repository
git clone https://github.com/zhuyu-cs/MeanFlow.git
cd MeanFlow

# Install dependencies
pip install -r requirements.txt
```

## Usage

### ImageNet 256

**Preparing Data**

This implementation requires [ImageNet ILSVRC2012](https://image-net.org/challenges/LSVRC/2012/) (1000 classes). The preprocessing converts images to VAE-encoded latents in LMDB format.

Expected directory structure:
```
ImageNet/
└── train/
    ├── n01440764/
    │   ├── n01440764_10026.JPEG
    │   └── ...
    ├── n01443537/
    └── ... (1000 class folders)
```

**Step 1:** Convert ImageNet to LMDB format using `preprocess_imagenet/image2lmdb.py`:
```bash
# Edit the paths in image2lmdb.py, then run:
cd ./preprocess_imagenet
python image2lmdb.py
```

**Step 2:** Encode images to VAE latents:
```bash
cd ./preprocess_imagenet
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    main_cache.py \
    --source_lmdb /path/to/imagenet_train.lmdb \
    --target_lmdb /path/to/train_vae_latents_lmdb \
    --img_size 256 \
    --batch_size 1024 \
    --lmdb_size_gb 400
```

**Training**

We provide training configurations for different model scales (B, L, XL) based on the hyperparameters from the original paper::

```bash

accelerate launch --multi_gpu \
    train.py \
    --exp-name "meanflow_b_4" \
    --output-dir "work_dir" \
    --data-dir "/data/train_vae_latents_lmdb" \
    --model "SiT-B/4" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 80\
    --path-type "linear" \
    --weighting "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0 \
    --cfg-omega 3.0 \ #1.0 for no cfg
    --cfg-kappa 0.\
    --cfg-min-t 0.0\
    --cfg-max-t 1.0

accelerate launch --multi_gpu \
    train.py \
    --exp-name "meanflow_b_2" \
    --output-dir "exp" \
    --data-dir "/data/train_vae_latents_lmdb" \
    --model "SiT-B/2" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 240\
    --path-type "linear" \
    --weighting "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0 \
    --cfg-omega 1.0 \
    --cfg-kappa 0.5\
    --cfg-min-t 0.0\
    --cfg-max-t 1.0

accelerate launch --multi_gpu \
    train.py \
    --exp-name "meanflow_l_2" \
    --output-dir "exp" \
    --data-dir "/data/train_vae_latents_lmdb" \
    --model "SiT-L/2" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 240\
    --path-type "linear" \
    --weighting "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0 \
    --cfg-omega 0.2 \
    --cfg-kappa 0.92\
    --cfg-min-t 0.0\
    --cfg-max-t 0.8

accelerate launch --multi_gpu \
    train.py \
    --exp-name "meanflow_xl_2" \
    --output-dir "exp" \
    --data-dir "/data/train_vae_latents_lmdb" \
    --model "SiT-XL/2" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 240\
    --path-type "linear" \
    --weighting "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0 \
    --cfg-omega 0.2 \
    --cfg-kappa 0.92\
    --cfg-min-t 0.0\
    --cfg-max-t 0.75

accelerate launch --multi_gpu \
    train.py \
    --exp-name "meanflow_xl_2_plus" \
    --output-dir "exp" \
    --data-dir "/data/train_vae_latents_lmdb" \
    --model "SiT-XL/2" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 1000\
    --path-type "linear" \
    --weighting "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0 \
    --cfg-omega 1.0 \
    --cfg-kappa 0.5\
    --cfg-min-t 0.3\
    --cfg-max-t 0.8
```
Each configuration is optimized for different model sizes according to the original paper's settings.

**Finetuning with pretrained flow matching model**

For finetuning explorations, please use the `finetune_fm` branch.
```bash
git checkout finetune_fm
```

**Sampling and Evaluation**

For large-scale sampling and quantitative evaluation (FID, IS), we provide a distributed evaluation framework:

```bash
torchrun --nproc_per_node=8 --nnodes=1 evaluate.py \
    --ckpt "/path/to/the/weights" \
    --model "SiT-L/2" \
    --resolution 256 \
    --cfg-scale 1.0 \
    --per-proc-batch-size 128 \
    --num-fid-samples 50000 \
    --sample-dir "./fid_dir" \
    --compute-metrics \
    --num-steps 1\
    --fid-statistics-file "./fid_stats/adm_in256_stats.npz"
```
This evaluation performs distributed sampling across 8 GPUs to generate 50,000 high-quality samples for robust FID computation. The framework validates MeanFlow's single-step generation capability (num-steps=1) and computes FID scores against pre-computed ImageNet statistics.

### CIFAR10

**Requirements**
- NVIDIA A100/H100 80GB GPU recommended for optimal performance

*Note: The UNet architecture needs higher memory consumption compared to Diffusion Transformer (DiT) models*

**Training**

1. Switch to the CIFAR-10 experimental branch:
```bash
git checkout cifar10
```

2. Standard Training (High Memory)
```bash
accelerate launch --num_processes=8 \
    train.py \
    --exp-name "cifar_unet" \
    --output-dir "work_dir" \
    --data-dir "/data/dataset/train_sdvae_latents_lmdb" \
    --resolution 32 \
    --batch-size 1024 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 19200\ # about 800k iters.
    --path-type "linear" \
    --weighting "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -2.0 \
    --time-sigma 2.0 \
    --ratio-r-not-equal-t 0.75 \
    --adaptive-p 0.75
```

2. Memory-Efficient Training (Lower GPU Memory)
```bash
accelerate launch --num_processes=8 \
      train.py \
      --exp-name "cifar_unet" \
      --output-dir "work_dir" \
      --data-dir "/data/dataset/train_sdvae_latents_lmdb" \
      --resolution 32 \
      --batch-size 512 \
      --gradient-accumulation-steps 2 \
      --allow-tf32 \
      --mixed-precision "bf16" \
      --epochs 19200\ 
      --path-type "linear" \
      --weighting "adaptive" \
      --time-sampler "logit_normal" \
      --time-mu -2.0 \
      --time-sigma 2.0 \
      --ratio-r-not-equal-t 0.75 \
      --adaptive-p 0.75
```

3. Evaluation 
```bash
torchrun --nproc_per_node=8 evaluate.py \
    --ckpt "./work_dir/cifar_unet/checkpoints/0200000.pt" \
    --per-proc-batch-size 128 \
    --num-fid-samples 50000 \
    --sample-dir "./fid_dir" \
    --compute-metrics \
    --num-steps 1\
    --fid-ref "train"
```
**Results**

| Iters | FID(NFE=1)|
|---------------|----------------|
| 50k|210.36|
| 100k|6.35|

## Acknowledgements

This implementation builds upon:
- [SiT](https://github.com/willisma/SiT/tree/main) (model architecture)
- [REPA](https://github.com/sihyun-yu/REPA/tree/main) (training pipeline)
- [MAR](https://github.com/LTH14/mar/tree/main) (data preprocessing)

## Official MeanFlow Repos
See also:
- Official [MeanFlow JAX repo](https://github.com/Gsunshine/meanflow) with ImageNet experiments.
- Official [MeanFlow PyTorch repo](https://github.com/Gsunshine/py-meanflow) with CIFAR10 experiments.

## Citation
If you find this implementation useful in your research, please cite the original work and this repo:
```
@article{geng2025mean,
  title={Mean Flows for One-step Generative Modeling},
  author={Geng, Zhengyang and Deng, Mingyang and Bai, Xingjian and Kolter, J Zico and He, Kaiming},
  journal={arXiv preprint arXiv:2505.13447},
  year={2025}
}

@misc{meanflow_pytorch,
  title={MeanFlow: PyTorch Implementation},
  author={Zhu, Yu},
  year={2025},
  howpublished={\url{https://github.com/zhuyu-cs/MeanFlow}},
  note={PyTorch implementation of Mean Flows for One-step Generative Modeling}
}
```
## License

[MIT License](LICENSE)
