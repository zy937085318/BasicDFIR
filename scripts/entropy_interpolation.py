"""
绘制随机噪声 → HR图像插值过程中的信息熵变化曲线
对指定目录下所有图像分别绘制，最终汇总到一张图。

Flow matching 插值路径: x_t = (1-t)·ε + t·x_hr  (RGB)
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# ============================================================
# 熵计算函数
# ============================================================

def shannon_entropy(gray):
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))


def local_avg_entropy(gray, win=7):
    h, w = gray.shape
    hc, wc = (h // win) * win, (w // win) * win
    img = gray[:hc, :wc]
    entropies = []
    for i in range(0, hc, win):
        for j in range(0, wc, win):
            patch = img[i:i+win, j:j+win]
            hist, _ = np.histogram(patch.flatten(), bins=64, range=(0, 256))
            prob = hist / hist.sum()
            prob = prob[prob > 0]
            entropies.append(-np.sum(prob * np.log2(prob)))
    return np.mean(entropies)


def differential_entropy(gray):
    diff = gray[:, 1:].astype(np.float32) - gray[:, :-1].astype(np.float32)
    diff_int = np.clip(np.round(diff), -128, 127).astype(np.int16) + 128
    hist, _ = np.histogram(diff_int.flatten(), bins=257, range=(0, 257))
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))


def rgb_entropy(img_rgb, fn, **kwargs):
    return np.mean([fn(img_rgb[:, :, c], **kwargs) for c in range(3)])


# ============================================================
# 主流程
# ============================================================

hr_dir = Path('/Users/ybb/dataset/1. SR_test_dataset/Set5/HR/')
images = sorted(hr_dir.glob('*.png'))
print(f"Found {len(images)} images in {hr_dir}")

num_steps = 50
t_values = np.linspace(0, 1, num_steps)

# 收集所有图像的熵曲线
all_results = {}

for img_path in images:
    name = img_path.stem
    x_hr = np.array(Image.open(img_path)).astype(np.float32)
    H, W, C = x_hr.shape
    print(f"\n[{name}] shape=({H}, {W}, {C})")

    np.random.seed(42)
    epsilon = np.random.randn(H, W, C).astype(np.float32) * 80 + 128
    epsilon = np.clip(epsilon, 0, 255)

    shannon_e, local_e, diff_e = [], [], []

    for t in t_values:
        x_t = (1 - t) * epsilon + t * x_hr
        x_t_uint8 = np.clip(x_t, 0, 255).astype(np.uint8)

        shannon_e.append(rgb_entropy(x_t_uint8, shannon_entropy))
        local_e.append(rgb_entropy(x_t_uint8, local_avg_entropy, win=7))
        diff_e.append(rgb_entropy(x_t_uint8, differential_entropy))

    # LR: bicubic 下采样 4x
    x_lr = np.array(Image.fromarray(x_hr.astype(np.uint8)).resize(
        (W // 4, H // 4), Image.BICUBIC)).astype(np.float32)

    lr_shannon = rgb_entropy(x_lr.astype(np.uint8), shannon_entropy)
    lr_local = rgb_entropy(x_lr.astype(np.uint8), local_avg_entropy, win=7)
    lr_diff = rgb_entropy(x_lr.astype(np.uint8), differential_entropy)

    all_results[name] = {
        'shannon': shannon_e,
        'local': local_e,
        'diff': diff_e,
        'shape': (H, W),
        'hr': x_hr,
        'epsilon': epsilon,
        'lr': x_lr,
        'lr_metrics': {'shannon': lr_shannon, 'local': lr_local, 'diff': lr_diff},
    }
    print(f"  Shannon: {shannon_e[0]:.2f} → {shannon_e[-1]:.2f}  "
          f"Local: {local_e[0]:.2f} → {local_e[-1]:.2f}  "
          f"LR Shannon: {lr_shannon:.2f}  LR Local: {lr_local:.2f}")

# ============================================================
# 汇总曲线图
# ============================================================

n_img = len(all_results)
names = list(all_results.keys())
colors = plt.cm.Set2(np.linspace(0, 1, max(n_img, 8)))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for metric_name, ax in zip(['shannon', 'local', 'diff'], axes):
    for i, name in enumerate(names):
        ax.plot(t_values, all_results[name][metric_name], 'o-', markersize=2,
                linewidth=1.8, color=colors[i], label=name)
        # LR 参考线
        lr_val = all_results[name]['lr_metrics'][metric_name]
        ax.axhline(lr_val, color=colors[i], linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlabel('Interpolation t  (0=noise, 1=HR)', fontsize=11)
    ax.set_ylabel('Entropy (bits)', fontsize=11)
    ax.set_title(f'{metric_name.capitalize()} Entropy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)

fig.suptitle('Information Entropy along Noise → HR Interpolation  (Set5, RGB averaged)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/Users/ybb/code zoo/BasicDFIR/scripts/entropy_curves_all.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("\nSaved: scripts/entropy_curves_all.png")

# ============================================================
# 每张图像单独的详细图 (含插值样本)
# ============================================================

for name in names:
    r = all_results[name]
    H, W = r['shape']

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.32)

    # 上: 熵曲线
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_values, r['shannon'], 'o-', color='#e74c3c', markersize=3, linewidth=2,
             label='Shannon Entropy')
    ax1.plot(t_values, r['local'], 's-', color='#3498db', markersize=3, linewidth=2,
             label='Local Entropy (7×7)')
    ax1.plot(t_values, r['diff'], '^-', color='#2ecc71', markersize=3, linewidth=2,
             label='Differential Entropy')

    # LR 参考线 (下采样 4x 的熵)
    lr = r['lr_metrics']
    ax1.axhline(lr['shannon'], color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.6,
                label=f'LR Shannon={lr["shannon"]:.2f}')
    ax1.axhline(lr['local'], color='#3498db', linestyle='--', linewidth=1.2, alpha=0.6,
                label=f'LR Local={lr["local"]:.2f}')
    ax1.axhline(lr['diff'], color='#2ecc71', linestyle='--', linewidth=1.2, alpha=0.6,
                label=f'LR Diff={lr["diff"]:.2f}')

    ax1.set_xlabel('t  (0=noise, 1=HR)', fontsize=11)
    ax1.set_ylabel('Entropy (bits)', fontsize=11)
    ax1.set_title(f'Entropy: Noise → HR  [{name}]', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=8, loc='center right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.02, 1.02)

    # 下: 插值样本 (完整图像，缩放显示)
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    sample_indices = [0, 10, 20, 30, 40, 49]
    thumb_h = 0.22  # 缩略图高度 (figure 比例)
    # 按宽高比计算宽度
    aspect = W / H
    thumb_w = thumb_h * aspect

    total_w = len(sample_indices) * (thumb_w + 0.01) - 0.01
    x_start = (1 - total_w) / 2  # 居中

    for i, si in enumerate(sample_indices):
        t = t_values[si]
        x_t = (1 - t) * r['epsilon'] + t * r['hr']
        x_t_uint8 = np.clip(x_t, 0, 255).astype(np.uint8)

        x0 = x_start + i * (thumb_w + 0.01)
        sub_ax = fig.add_axes([x0, 0.05, thumb_w, thumb_h])
        sub_ax.imshow(x_t_uint8)
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
        sub_ax.set_title(f't={t:.1f}', fontsize=9, pad=2)
        for spine in sub_ax.spines.values():
            spine.set_edgecolor('#999')
            spine.set_linewidth(0.5)

    out_path = f'/Users/ybb/code zoo/BasicDFIR/scripts/entropy_{name}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_path}")
    plt.close(fig)

print("\nAll done.")
