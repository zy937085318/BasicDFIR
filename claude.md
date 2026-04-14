# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Remote Server

GPU training is performed on a remote server:
- **SSH**: `ssh -p 5122 ybb@10.16.104.29` (alias: `ssh 40908`)
- **GPU**: NVIDIA 4090
- **Conda**: `/home/ybb/miniconda3/bin/conda` (env: `base`)
- **Python**: `/home/ybb/miniconda3/bin/python`
- **Server project path**: `/home/ybb/Project/BasicDFIR`

Sync files to server before testing:
```bash
scp -P 5122 <file> ybb@10.16.104.29:/home/ybb/Project/BasicDFIR/
```

## Project Overview

BasicDFIR is a super-resolution research framework extending [BasicSR](https://github.com/xinntao/BasicSR) with flow matching methods: FlowMatching, RectifyFlow, MeanFlow, and JiT (Jigsaw Transformer).

### Directory Structure

- `basicsr/` - Core framework (data, losses, metrics, models, train.py)
- `basicsr/models/` - Model wrappers (SRModel, JiTModel, FlowModel, ABCFlow, MemIR, etc.)
- `basicsr/archs/` - Network architectures (ATD, SwinIR, DINOv3, UNet, MyProposal/*)
- `options/train/` - YAML configs organized by model type (DINOv3, JiT, MyProposal, etc.)
- `scripts/train_script/` - Shell scripts for launching training
- `flow_matching/` - Flow matching utilities (path, scheduler, solver)

### Registry Pattern

Models and architectures use auto-registration via decorators:
- Models ending in `_model.py`: auto-imported by `basicsr/models/__init__.py`
- Architectures ending in `_arch.py`: auto-imported by `basicsr/archs/__init__.py`
- Use `@MODEL_REGISTRY.register()` for models, `@ARCH_REGISTRY.register()` for architectures

## Training Commands

### Single GPU Training
```bash
python basicsr/train.py -opt options/train/<folder>/<config>.yml
```

### Distributed Training
Use scripts in `scripts/train_script/`:
```bash
bash scripts/train_script/dist_train.sh
```

### Testing on Server
```bash
ssh -p 5122 ybb@10.16.104.29 '/home/ybb/miniconda3/bin/python -c "import torch; ..."'
```

## JiTModel Architecture Requirements

JiTModel (Jigsaw Transformer for flow matching) requires networks with specific forward signature:

### Forward Signature
```python
def forward(self, x, c=None, temp=None):
    # x: [B, C, H, W] - Input tensor
    # c: Reference condition (unused, reserved)
    # temp: [B] - Timestep tensor for flow matching
```

### Input Channel Handling
- **Standard SR**: 3 channels (LR image)
- **JiT Mode**: 6 channels (concat of `x_t` + `lq_bicubic`)
- Networks should auto-expand 3→6 channels when `upscale=1`

### Timestep Embedding
JiT networks require sinusoidal timestep embedding:
1. `get_sinusoidal_positional_embedding(temp, embedding_dim)`
2. `TimestepEmbedding` MLP to project to `temb_ch`
3. Inject `temb` into each transformer layer (typically via `temb_proj` linear layer)

Reference: `basicsr/archs/unet_arch.py` (PnPFlowUNet)

## Config File Structure (YAML)

Key sections in `options/train/<folder>/<config>.yml`:
- `model_type`: SRUNetModel, JiTModel, FlowModel, etc.
- `network_g`: Network architecture config
  - `type`: Architecture name (must match registered @ARCH_REGISTRY)
  - `upscale`: 1 for JiT, 4 for SR
  - `in_chans`: 3 for SR, 6 or 3 (auto-expand) for JiT
  - `temb_ch`: Timestep embedding channels (required for JiT)
  - `embedding_dim`: Base dimension for timestep embedding
- `datasets`: Train/validation data paths
- `train`: Optimizer, scheduler, total_iter, losses
- `val`: Validation frequency, metrics
- `logger`: TensorBoard, WandB settings

## Architecture Implementation Guidelines

### Adding New Models for JiT

1. **Inherit registration pattern** - Use `@ARCH_REGISTRY.register()`
2. **Add timestep embedding support**:
   - Add `temb_ch` and `embedding_dim` to `__init__`
   - Create `TimestepEmbedding` module
   - Pass `temb_ch` to all transformer/attention layers
3. **Modify forward signature** to `(x, c=None, temp=None)`
4. **Handle 3→6 channel expansion** when `upscale=1`

Example: `basicsr/archs/MyProposal/PsAtdv1_1_arch.py`

## Common Model Types

- `SRUNetModel` - Basic SR with UNet
- `SRModel` - Standard SR model
- `JiTModel` - Flow matching with ODE solver
- `FlowModel` - RectifiedFlow/MeanFlow variants
- `ABCFlow*Model` - ABCFlow variants (v2, v3)
- `MemIR*Model` - Memory-based SR models

## MyProposal Architectures

Located in `basicsr/archs/MyProposal/`:
- `PsAtd_arch.py` - Base PsATD architecture
- `PsAtdv1_1_arch.py` - JiT-compatible version with timestep embedding
- Configs: `options/train/MyProposal/MyProposalv1_*.yml`

Key params for PsATD:
- `embed_dim`: Feature dimension
- `depths`: Number of layers per stage
- `num_heads`: Attention heads
- `window_size`: Window attention size
- `temb_ch`: Set >0 for JiT, 0 for SR
