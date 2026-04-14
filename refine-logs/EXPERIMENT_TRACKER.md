# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Test Set | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|----------|---------|----------|--------|-------|
| R001 | M0 | sanity | TC-ATD full, 1000 iter | train split | L1 loss | MUST | TODO | Verify pipeline |
| R002 | M1 | baseline | MPv2 temb_ch=0 (no timestep) | Set5+Urban100 | PSNR/SSIM | MUST | TODO | Ablation lower bound |
| R003 | M2 | main | TC-ATD full (temb=256, ds=4) | Set5+Urban100 | PSNR/SSIM | MUST | TODO | Our method |
| R004 | M3 | ablation | w/ temb-add-only (no td_adaln) | Set5+Urban100 | PSNR/SSIM | MUST | TODO | Isolate AdaLN on TD |
| R005 | M4 | ablation | w/ simple-add (no t-gate) | Set5+Urban100 | PSNR/SSIM | MUST | TODO | Isolate t-gating |
| R006 | M4 | ablation | w/ constant-gate (gate=0.5) | Set5+Urban100 | PSNR/SSIM | MUST | TODO | Gating vs averaging |
| R007 | M4 | ablation | w/ concat fusion | Set5+Urban100 | PSNR/SSIM | MUST | TODO | Fusion strategy |
| R008 | M5 | full-test | TC-ATD full | All 5 test sets | PSNR/SSIM | MUST | TODO | Complete Table 1 |
| R009 | M6 | viz | TC-ATD full, save attn maps | 3 images | Attention viz | NICE | TODO | Qualitative figure |
| R010 | M7 | efficiency | downscale=1 | Set5 | PSNR/time/mem | NICE | TODO | No spatial reduction |
| R011 | M7 | efficiency | downscale=2 | Set5 | PSNR/time/mem | NICE | TODO | 2x reduction |
| R012 | M7 | efficiency | downscale=8 | Set5 | PSNR/time/mem | NICE | TODO | 8x reduction |
