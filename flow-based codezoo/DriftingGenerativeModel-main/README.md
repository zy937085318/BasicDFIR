# Drifting Models

**Unofficial Implementation** of "Generative Modeling via Drifting" (arXiv:2602.04770).

## Core Idea

Generator directly outputs samples, trained by drifting field V:
```
V = attraction_to_real - repulsion_from_generated
Loss = ||G(z) - stopgrad(G(z) + V)||²
```

## Key Insight

**Drifting requires meaningful distance metric.** High-dimensional pixel space suffers from curse of dimensionality. Solution: project to lower-dim semantic space.

## Experiments

### 2D Toy (run_toy.py)
Direct drifting in 2D space. Works perfectly.
```bash
python run_toy.py --dataset moons --save_dir ./toy_moons
python run_toy.py --dataset swiss_roll --save_dir ./toy_swiss
python run_toy.py --dataset checker --save_dir ./toy_checker
```

**Results (Swiss Roll, 2000 steps):**
- Final loss: 2.07e-05
- Final drift norm: 0.0052
- Outputs: `samples.png`, `progress.png`, `drift_field.png`, `losses.png`

### MNIST
| Script | Feature Space | Status | Final Loss |
|--------|---------------|--------|------------|
| `run_mnist.py` | 784D pixel | Needs re-run | - |
| `run_mnist_encoder.py` | 64D CNN encoder | **Done** | 1.15e-04 |

```bash
python run_mnist.py --save_dir ./mnist_outputs
python run_mnist_encoder.py --save_dir ./mnist_encoder_outputs
```

**Results (MNIST Encoder, AE 2000 steps + Generator 8000 steps):**
- AE pretraining: 2000 steps for encoder/decoder
- Generator training: 8000 steps in 64D feature space
- Outputs: `samples.png`, `comparison.png`, `ae_reconstruction.png`, `losses.png`, `models.pt` (4.7MB)

### CIFAR-10
| Script | Encoder | Feature Dim | Result |
|--------|---------|-------------|--------|
| `train_cifar10_drifting.py` | ResNet18 | 512 | **Best** |
| `train_cifar10_resnet50_v2.py` | ResNet50 | 2048 | OK, some artifacts |
| `train_cifar10_multiscale.py` | ResNet18 multi-scale | 64+128+256+512 | OK, color issues |
| `train_cifar10_simclr.py` | SimCLR | 512 | **Failed** - color artifacts |

```bash
python train_cifar10_drifting.py --save_dir ./cifar10_drifting --drift_steps 100000
```

## Temperature

Multi-temperature strategy: small temp (0.02-0.04) prevents mode collapse, large temp (0.2-0.4) for global distribution.

High-dim features need larger temperatures (distances scale with sqrt(D)).

## File Structure

```
├── drifting.py              # Core drift computation
├── run_toy.py               # 2D toy examples
├── run_mnist.py             # MNIST pixel space
├── run_mnist_encoder.py     # MNIST with encoder
├── train_cifar10_*.py       # CIFAR-10 experiments
├── pretrained_models/       # -> $SCRATCH/Driftmodel/pretrained_models
└── cifar10_*/               # Output directories
```
