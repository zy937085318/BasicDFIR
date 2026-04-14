# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BasicDFIR is a super-resolution research framework extending [BasicSR](https://github.com/xinntao/BasicSR) with flow matching methods: FlowMatching, RectifyFlow, MeanFlow, and JiT (Jigsaw Transformer). The active research area is `basicsr/archs/MyProposal/` and its associated models.

## Remote Server

GPU training is performed on a remote server:
- **SSH**: `ssh -p 5122 ybb@10.16.104.29` (alias: `ssh 40908`)
- **GPU**: NVIDIA 4090
- **Conda**: `/home/ybb/miniconda3/bin/conda` (env: `base`)
- **Python**: `/home/ybb/miniconda3/bin/python`
- **Server project path**: `/home/ybb/Project/BasicDFIR`

Sync files to server:
```bash
scp -P 5122 <file> ybb@10.16.104.29:/home/ybb/Project/BasicDFIR/
```

## Quick Start

### Install
```bash
pip install -e .                # basic install
BASICSR_EXT=True pip install -e .  # with CUDA extensions (deform_conv, fused_act, upfirdn2d)
```

### Train
```bash
# Single GPU
python basicsr/train.py -opt options/train/<folder>/<config>.yml

# Distributed (4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt <config> --launcher pytorch

# Or use the helper script
bash scripts/train_script/dist_train.sh
```

### Test
```bash
python basicsr/test.py -opt options/test/<folder>/<config>.yml
```

## Architecture

### Directory Structure

- `basicsr/` - Core framework
  - `archs/` - Network architectures (`*_arch.py` files, auto-discovered recursively)
  - `models/` - Model wrappers (registry: `*_model.py`)
  - `data/` - Dataset classes (registry: `*_dataset.py`)
  - `losses/` - Loss functions (registry: `*_loss.py`)
  - `metrics/` - Evaluation metrics
  - `utils/` - Utilities including registry system
  - `models/flow_matching/` - Flow matching library (path, scheduler, solver) used by JiT/Flow models
- `options/train/` - YAML configs organized by model type
- `scripts/train_script/` - Training launch scripts
- `flow-based codezoo/` - Cloned reference repos (NOT part of the package)

### Registry Pattern

Auto-registration via decorators. All `__init__.py` files use `find_files()` for **recursive** discovery (unlike upstream BasicSR which uses flat `scandir`):

| Registry | Decorator | Discovery |
|---|---|---|
| `MODEL_REGISTRY` | `@MODEL_REGISTRY.register()` | `basicsr/models/*_model.py` (recursive) |
| `ARCH_REGISTRY` | `@ARCH_REGISTRY.register()` | `basicsr/archs/**/*_arch.py` (recursive) |
| `DATASET_REGISTRY` | `@DATASET_REGISTRY.register()` | `basicsr/data/**/*_dataset.py` (recursive) |
| `LOSS_REGISTRY` | `@LOSS_REGISTRY.register()` | `basicsr/losses/**/*_loss.py` (recursive) |

**Important**: `ARCH_REGISTRY.get(name)` in `build_network()` first tries `name`, then falls back to `name_basicsr`. So the registered class name (e.g., `MPv1_arch`) must match `network_g.type` in the YAML config, OR config must use the `_basicsr` suffix.

Two architecture files registering the same class name will conflict. If creating a new architecture file in `MyProposal/`, use a unique class name (e.g., `MPv1_2_arch` with class `MPv1_2_arch`).

## Model Inheritance Hierarchy

```
BaseModel
  ├─ SRModel                        # Standard SR base
  │   ├─ JiTModel                   # cat([x_t, lq]) → 6ch input, ODE solver, tile split
  │   │   ├─ ABCFlowv2Model
  │   │   └─ ABCFlowv3Model
  │   ├─ JiTv2Model                 # separate (x_t, lq, t), configurable tile_mode
  │   ├─ FlowModel → RectifiedFlowModel
  │   ├─ FlowMatchingModel
  │   ├─ MeanFlowModel, Meanflow_minimax_Model
  │   ├─ MPv1Model, ATDModel, MambaIR*
  │   └─ PixelMeanFlow_Model
  ├─ BaseSRModel                    # Has image splitting for inference
  │   └─ MemIRModel
  ├─ SRUNetModel → SRUFixedNetModel
  ├─ MemIRv1/v2/v3Model             # Independent BaseModel subclasses
  └─ BaseAMPModel → SRAMPModel
```

## JiT vs JiTv2 Forward Signatures

| Mode | Forward Signature | Input Channels | upscale |
|---|---|---|---|
| JiT | `forward(x, c=None, temp=None)` | 6 (x_t + lq concat) | 1 |
| JiTv2 | `forward(x, lq, t)` | 3+3 (separate) | 1 |
| Standard SR | `forward(x)` | 3 | 4 |

JiT networks require sinusoidal timestep embedding: `get_sinusoidal_positional_embedding(temp, embedding_dim)` → `TimestepEmbedding` MLP → inject `temb` via `temb_proj` linear in each transformer layer.

Reference: `basicsr/archs/unet_arch.py` (PnPFlowUNet), `basicsr/archs/MyProposal/MPv1_1_JiT_arch.py`.

## Config File Structure (YAML)

Key fields in `options/train/<folder>/<config>.yml`:
- `model_type`: matches `@MODEL_REGISTRY.register()` class name (e.g., `JiTModel`, `JiTv2Model`, `SRModel`)
- `network_g.type`: matches `@ARCH_REGISTRY.register()` class name
- `network_g.upscale`: `1` for flow/JiT models, `4` for standard SR
- `network_g.temb_ch`: timestep embedding channels (>0 for JiT)
- `network_g.embedding_dim`: base dimension for sinusoidal embedding
- `train.pixel_opt`: loss config (L1Loss, MSELoss, etc.)
- `val.tile_mode`: JiTv2Model only — `'split'`, `'none'`, `'auto'`
- `val.sample_step`: ODE solver steps for flow/JiT models
- Dataset paths use server paths (`/8T1/dataset/SRDataset/...`)

## Experiment Output

Training creates `experiments/<name>_<timestamp>_<us>/` with `models/`, `training_states/`, `visualization/`, logs. Renamed to `<name>_finished` when complete. Best checkpoints are pruned automatically.

## Key Skills Available

Skills are invoked with `/<skill-name>` (e.g., `/inno-idea-generation`):
- Research pipeline: `/inno-pipeline-planner`, `/inno-idea-generation`, `/inno-experiment-dev`
- Paper writing: `/inno-paper-writing`, `/inno-paper-reviewer`, `/inno-rebuttal`
- Experiment: `/inno-experiment-analysis`, `/inno-deep-research`, `/experiment-plan`
- Training: `/training-check`, `/monitor-experiment`
