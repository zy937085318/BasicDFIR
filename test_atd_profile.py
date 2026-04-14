"""
Performance profiling script for ATD model components.
Tests each component's time consumption to identify bottlenecks.
"""

import torch
import sys
import time
from functools import partial
sys.path.insert(0, '/home/ybb/Project/BasicDFIR')

from basicsr.archs.atd_arch import (
    ATD, ATDTransformerLayer, ATD_CA, AC_MSA, WindowAttention,
    BasicBlock, ATDB, dwconv, ConvFFN_td, index_reverse, feature_shuffle
)


def timer(func, *args, warmup=3, repeat=10, **kwargs):
    """Time a function execution with warmup."""
    for _ in range(warmup):
        func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(repeat):
        result = func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = (time.time() - start) / repeat
    return elapsed, result


def test_dynamic_operations():
    """Test time cost of dynamic operations (index_reverse, sort, etc)."""
    print("\n" + "="*60)
    print("1. Dynamic Operations (CPU-bound bottlenecks)")
    print("="*60)

    batch_size = 4
    n_tokens = 4096  # Typical: 64x64 feature map
    num_classes = 64

    # Random token assignments
    tk_id = torch.randint(0, num_classes, (batch_size, n_tokens))
    if torch.cuda.is_available():
        tk_id = tk_id.cuda()

    # Test index_reverse
    def run_index_reverse():
        return index_reverse(tk_id)

    t, _ = timer(run_index_reverse)
    print(f"  index_reverse:     {t*1000:.3f} ms  (loop over batch)")

    # Test torch.sort
    def run_sort():
        return torch.sort(tk_id, dim=-1, stable=False)

    t, _ = timer(run_sort)
    print(f"  torch.sort:        {t*1000:.3f} ms")

    # Test feature_shuffle (uses gather)
    qkv = torch.randn(batch_size, n_tokens, 180)
    if torch.cuda.is_available():
        qkv = qkv.cuda()

    def run_shuffle():
        _, indices = torch.sort(tk_id, dim=-1)
        return feature_shuffle(qkv, indices)

    t, _ = timer(run_shuffle)
    print(f"  feature_shuffle:   {t*1000:.3f} ms  (gather op)")

    # Test argmax
    sim_atd = torch.randn(batch_size, n_tokens, num_classes)
    if torch.cuda.is_available():
        sim_atd = sim_atd.cuda()

    def run_argmax():
        return torch.argmax(sim_atd, dim=-1)

    t, _ = timer(run_argmax)
    print(f"  torch.argmax:      {t*1000:.3f} ms")

    # Test gather with dynamic indices
    td = torch.randn(batch_size, num_classes, 180)
    if torch.cuda.is_available():
        td = td.cuda()

    def run_gather():
        tk_id_out = torch.argmax(sim_atd, dim=-1)
        index = tk_id_out.reshape(batch_size, n_tokens, 1).expand(-1, -1, 16)
        return torch.gather(td, dim=1, index=index)

    t, _ = timer(run_gather)
    print(f"  torch.gather:      {t*1000:.3f} ms  (dynamic indexing)")


def test_attention_modules():
    """Test different attention mechanisms."""
    print("\n" + "="*60)
    print("2. Attention Modules Comparison")
    print("="*60)

    dim = 180
    batch_size = 4
    n_tokens = 4096
    num_heads = 6
    window_size = 8
    num_tokens_td = 64

    # Input tensors
    x = torch.randn(batch_size, n_tokens, dim)
    td = torch.randn(batch_size, num_tokens_td, dim)
    qkv = torch.randn(batch_size, n_tokens, dim * 3)

    if torch.cuda.is_available():
        x, td, qkv = x.cuda(), td.cuda(), qkv.cuda()

    x_size = (64, 64)

    # ATD_CA (Cross-Attention with Token Dictionary)
    atd_ca = ATD_CA(dim=dim, num_tokens=num_tokens_td, reducted_dim=10)
    if torch.cuda.is_available():
        atd_ca = atd_ca.cuda()
    atd_ca.eval()

    def run_atd_ca():
        with torch.no_grad():
            return atd_ca(x, td, x_size)

    t, _ = timer(run_atd_ca)
    print(f"  ATD_CA:            {t*1000:.3f} ms  (Cross-Attn with TD)")

    # AC_MSA (Adaptive Category Self-Attention)
    ac_msa = AC_MSA(dim=dim, num_heads=num_heads, category_size=128)
    if torch.cuda.is_available():
        ac_msa = ac_msa.cuda()
    ac_msa.eval()

    def run_ac_msa():
        with torch.no_grad():
            tk_id = torch.randint(0, num_tokens_td, (batch_size, n_tokens))
            if torch.cuda.is_available():
                tk_id = tk_id.cuda()
            return ac_msa(qkv, tk_id, x_size)

    t, _ = timer(run_ac_msa)
    print(f"  AC_MSA:            {t*1000:.3f} ms  (Grouped Self-Attn)")

    # WindowAttention (SW-MSA)
    win_attn = WindowAttention(dim=dim, window_size=(window_size, window_size), num_heads=num_heads)
    if torch.cuda.is_available():
        win_attn = win_attn.cuda()
    win_attn.eval()

    # Prepare window input
    nw = (64 // window_size) ** 2
    x_windows = torch.randn(batch_size * nw, window_size * window_size, dim * 3)
    if torch.cuda.is_available():
        x_windows = x_windows.cuda()
    rpi = torch.randint(0, (2*window_size-1)**2, (window_size*window_size, window_size*window_size))
    if torch.cuda.is_available():
        rpi = rpi.cuda()

    def run_window_attn():
        with torch.no_grad():
            return win_attn(x_windows, rpi=rpi, mask=None)

    t, _ = timer(run_window_attn)
    print(f"  WindowAttention:   {t*1000:.3f} ms  (SW-MSA per window)")


def test_conv_ffn():
    """Test ConvFFN with dwconv overhead."""
    print("\n" + "="*60)
    print("3. ConvFFN (Depthwise Conv + Reshape Overhead)")
    print("="*60)

    dim = 180
    batch_size = 4
    n_tokens = 4096
    hidden_dim = 360
    td_features = 16

    x = torch.randn(batch_size, n_tokens, dim)
    x_td = torch.randn(batch_size, n_tokens, td_features)
    x_size = (64, 64)

    if torch.cuda.is_available():
        x, x_td = x.cuda(), x_td.cuda()

    convffn = ConvFFN_td(in_features=dim, hidden_features=hidden_dim, td_features=td_features)
    if torch.cuda.is_available():
        convffn = convffn.cuda()
    convffn.eval()

    def run_convffn():
        with torch.no_grad():
            return convffn(x, x_td, x_size)

    t, _ = timer(run_convffn)
    print(f"  ConvFFN_td:        {t*1000:.3f} ms  (fc1+dwconv+fc2)")
    print(f"    Includes 4x reshape: BNC <-> BCHW")


def test_single_layer():
    """Test complete ATDTransformerLayer with all 3 attention branches."""
    print("\n" + "="*60)
    print("4. Complete ATDTransformerLayer (3 Attn branches)")
    print("="*60)

    dim = 180
    batch_size = 4
    n_tokens = 4096
    x_size = (64, 64)

    x = torch.randn(batch_size, n_tokens, dim)
    td = torch.randn(batch_size, 64, dim)

    if torch.cuda.is_available():
        x, td = x.cuda(), td.cuda()

    layer = ATDTransformerLayer(
        dim=dim, idx=0, input_resolution=x_size,
        num_heads=6, window_size=8, shift_size=0,
        dim_ffn_td=16, category_size=128, num_tokens=64,
        reducted_dim=4, convffn_kernel_size=5, mlp_ratio=2.0
    )
    if torch.cuda.is_available():
        layer = layer.cuda()
    layer.eval()

    params = {
        'attn_mask': None,
        'rpi_sa': torch.randint(0, 64, (64, 64)).cuda() if torch.cuda.is_available() else torch.randint(0, 64, (64, 64))
    }

    def run_layer():
        with torch.no_grad():
            return layer(x, td, x_size, params)

    t, _ = timer(run_layer)
    print(f"  ATDTransformerLayer:  {t*1000:.3f} ms")
    print(f"    Contains: ATD_CA + AC_MSA + SW-MSA + ConvFFN")


def test_atdb_block():
    """Test ATDB block (stack of layers)."""
    print("\n" + "="*60)
    print("5. ATDB Block (Multiple Layers)")
    print("="*60)

    dim = 180
    depth = 6
    batch_size = 4
    img_size = 64
    n_tokens = img_size * img_size

    x = torch.randn(batch_size, n_tokens, dim)
    if torch.cuda.is_available():
        x = x.cuda()

    x_size = (img_size, img_size)

    atdb = ATDB(
        dim=dim, idx=0, input_resolution=x_size,
        depth=depth, num_heads=6, window_size=8,
        dim_ffn_td=16, category_size=128, num_tokens=64,
        reducted_dim=4, convffn_kernel_size=5, mlp_ratio=2.0,
        img_size=img_size, patch_size=1
    )
    if torch.cuda.is_available():
        atdb = atdb.cuda()
    atdb.eval()

    params = {
        'attn_mask': None,
        'rpi_sa': torch.randint(0, 64, (64, 64)).cuda() if torch.cuda.is_available() else torch.randint(0, 64, (64, 64))
    }

    def run_atdb():
        with torch.no_grad():
            return atdb(x, x_size, params)

    t, _ = timer(run_atdb)
    print(f"  ATDB (depth={depth}):      {t*1000:.3f} ms")
    print(f"  Per layer avg:            {t/depth*1000:.3f} ms")


def test_full_atd_model():
    """Test full ATD model forward pass."""
    print("\n" + "="*60)
    print("6. Full ATD Model (Small Config)")
    print("="*60)

    img_size = 64
    batch_size = 4

    x = torch.randn(batch_size, 3, img_size, img_size)
    if torch.cuda.is_available():
        x = x.cuda()

    model = ATD(
        img_size=img_size, patch_size=1, in_chans=3,
        embed_dim=90, depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6),
        window_size=8, dim_ffn_td=16, category_size=128,
        num_tokens=64, reducted_dim=4, convffn_kernel_size=5,
        mlp_ratio=2., upscale=4, upsampler='pixelshuffle'
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters:  {total_params/1e6:.2f}M")

    def run_forward():
        with torch.no_grad():
            return model(x)

    t, out = timer(run_forward)
    print(f"  Forward pass:      {t*1000:.3f} ms")
    print(f"  Output shape:      {out.shape}")


def main():
    print("="*60)
    print("ATD Model Performance Profiling")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device:      {torch.cuda.get_device_name(0)}")
    print("="*60)

    test_dynamic_operations()
    test_attention_modules()
    test_conv_ffn()
    test_single_layer()
    test_atdb_block()
    test_full_atd_model()

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("Key findings:")
    print("  1. Dynamic operations (index_reverse, sort, gather) are CPU-bound")
    print("  2. AC_MSA has high overhead due to sorting + shuffling")
    print("  3. Each layer runs 3 attention branches (ATD_CA + AC_MSA + SW-MSA)")
    print("  4. ConvFFN has reshape overhead (BNC <-> BCHW conversions)")
    print("="*60)


if __name__ == '__main__':
    main()
