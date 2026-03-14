# Discrepancy Log

| Section | Item | Code Location | Discrepancy Description | Correction Plan | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 3.2 | Data Preprocessing | `src/pmf/dataset.py` | Training uses `Resize` + `CenterCrop` (validation style) instead of `RandomResizedCrop` + `RandomHorizontalFlip` + `ColorJitter`. | Implement conditional transforms for `train=True`. | **P0** |
| Table 1 | Training Hyperparams | `scripts/train.py` | Missing Learning Rate Scheduler (Cosine Decay with Warmup). | Implement `get_cosine_schedule_with_warmup`. | **P0** |
| Table 1 | Training Hyperparams | `scripts/train.py` | Missing EMA (Exponential Moving Average) for model weights. | Integrate `timm.utils.ModelEma` or `accelerate` EMA. | **P0** |
| 7 | Random Seed | `scripts/train.py` | Random seed is configured but not globally set via `torch.manual_seed`, etc. | Add `set_seed` function call at start. | **P1** |
| 6 | Evaluation Indicators | `scripts/eval.py` | Script only generates samples; missing FID/IS/LPIPS calculation logic using standard libraries. | Add `calc_metrics.py` or integrate `torch-fidelity` calls. | **P1** |
| Appendix A | File Naming | `scripts/train.py` | Checkpoint naming (`checkpoint_step_...`) may not match specific Appendix A conventions. | Standardize to `pmf-imagenet-xl-step-{step}.pt` (or similar) and confirm. | **P2** |
| 8 | Dependencies | `requirements.txt` | Versions are unpinned. | Pin key dependencies (`torch`, `torchvision`, `timm`) to stable versions. | **P2** |
