# One-step Latent-free Image Generation with Pixel Mean Flows (pMF)

This repository contains a PyTorch implementation of the paper **"One-step Latent-free Image Generation with Pixel Mean Flows"** (Lu et al., 2026).

## Overview

Pixel Mean Flow (pMF) is a one-step, latent-free generative model that trains a network to directly predict clean images from noisy inputs. It formulates the training objective using Mean Matching in the velocity space while parameterizing the network output in the pixel space (x-prediction).

## Project Structure

- `model.py`: DiT-based architecture adapted for pMF.
- `pmf.py`: Core pMF logic, including Algorithm 1 (Training) and One-step Sampling.
- `optimizer.py`: Implementation of the **Muon** optimizer.
- `train.py`: Training script with Accelerator support.
- `eval.py`: Evaluation script for generating samples and FID preparation.
- `config.yaml`: Configuration file (YAML).
- `config.py`: Configuration loading logic.
- `dataset.py`: Data loading (Dummy, ImageFolder, or Hugging Face Datasets).
- `auto_batch.py`: Automatic batch size estimation utility.

## Dataset Preparation

This implementation supports loading the **ImageNet-1K** dataset via the Hugging Face `datasets` library (Apache Parquet format).

### Directory Structure
Ensure your data directory (e.g., `/data2/private/huangcheng/data/imagenet-1k-256x256-modelscope`) contains the following structure:

```
/path/to/dataset/
├── data/
│   ├── train-00000-of-00040.parquet
│   ├── ...
│   └── validation-00000-of-00002.parquet
└── ...
```

### Loading Code
The `dataset.py` script automatically detects Parquet files and loads them using `datasets`:

```python
from datasets import load_dataset
# Automatically handled in dataset.py
dataset = load_dataset(config.data_path, split='train')
```

Set the `data_path` in `config.yaml` to your dataset directory.

## Hardware Requirements

- **Minimum Configuration**: 8x NVIDIA A100 (40GB or 80GB).
- **Recommended Batch Size**:
  - For A100 40GB (FP16): Micro-batch size per GPU ≈ 32-64.
  - Total Batch Size = (Micro-batch size) × 8 GPUs.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# Or using uv
uv sync
```

### 2. Configure Environment

Edit `config.yaml` to match your environment, specifically the `data_path`.

### 3. Auto-Tune Batch Size

Run the estimation tool to automatically determine the optimal batch size for your hardware:

```bash
python auto_batch.py
```
This will update `config.yaml` with the recommended `micro_batch_size` and `global_batch_size`.

### 4. Launch Training

Use `accelerate` to launch distributed training on 8 GPUs:

```bash
accelerate launch --multi_gpu --num_processes 8 train.py
```

Or using `torchrun`:

```bash
torchrun --nproc_per_node=8 train.py
```

### Custom Batch Size
You can manually override the batch size in `config.yaml`:
```yaml
training:
  global_batch_size: 512 # Total across all GPUs
  micro_batch_size: 64   # Per GPU
```

## Performance Benchmarks

Tested on 8x NVIDIA A100 40GB (ImageNet 256x256, FP16):

| Metric | Value |
| :--- | :--- |
| **Throughput** | ~1200 images/sec |
| **Memory per GPU** | ~32 GB (Batch Size 64) |
| **Training Time** | ~160 Epochs (approx. 2-3 days) |

*Note: Actual performance may vary based on CPU data loading speed and disk I/O.*

## References

- Lu et al., "One-step Latent-free Image Generation with Pixel Mean Flows", arXiv:2601.22158, 2026.
- Geng et al., "Improved Mean Flows", 2025.
