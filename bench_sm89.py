#!/usr/bin/env python3
"""Benchmark LiteAttention on SM89 (RTX 4090).

Measures:
1. Flash Attention baseline (no skip lists)
2. Skip list overhead (all tiles computed, skip machinery active)
3. Skip list with actual skipping (various skip ratios)
4. PyTorch SDPA baseline
"""
import math
import torch
import torch.nn.functional as F
import lite_attention

fa = lite_attention._internal.flash_attn_interface.flash_attn_func


def tflops(B, S, H, D, causal, ms):
    """Compute TFLOPS for attention."""
    flops = 4 * B * H * S * S * D  # 2 matmuls (QK^T and PV), each 2*S*S*D
    if causal:
        flops //= 2
    return flops / (ms * 1e-3) / 1e12


def bench_kernel(fn, warmup=10, iters=50):
    """Benchmark a kernel, return median ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[len(times) // 2]  # median


def make_skip_list(B, H, S, kBlockN, kBlockM=128, skip_ratio=0.0):
    """Create skip lists. skip_ratio=0 means compute all tiles (no skipping)."""
    q_blocks = (S + kBlockM - 1) // kBlockM
    k_blocks = (S + kBlockN - 1) // kBlockN
    num_k_blocks = k_blocks + 2

    read_list = torch.full((B, H, q_blocks, num_k_blocks), -1, dtype=torch.int16, device='cuda')
    write_list = torch.full((B, H, q_blocks, num_k_blocks), -1, dtype=torch.int16, device='cuda')

    # Number of K blocks to actually compute
    n_compute = max(1, int(k_blocks * (1.0 - skip_ratio)))

    for b in range(B):
        for h in range(H):
            for qb in range(q_blocks):
                read_list[b, h, qb, 0] = 2  # length = 2 (one range: start, end)
                read_list[b, h, qb, 1] = 0  # start
                read_list[b, h, qb, 2] = n_compute  # end

    return read_list, write_list


def bench_flash_attn(B, S, H, D, causal=False):
    """Benchmark vanilla flash attention."""
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')

    ms = bench_kernel(lambda: fa(q, k, v, causal=causal))
    tf = tflops(B, S, H, D, causal, ms)
    return ms, tf


def bench_skip_list(B, S, H, D, skip_ratio=0.0, phase=True, causal=False):
    """Benchmark flash attention with skip lists."""
    kBlockN = 128  # SM89 hdim128
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')

    read_list, write_list = make_skip_list(B, H, S, kBlockN, kBlockM=128, skip_ratio=skip_ratio)

    ms = bench_kernel(lambda: fa(q, k, v, attn_read_list=read_list, attn_write_list=write_list,
                                  thr=-10.0, phase=phase, causal=causal))
    # effective TFLOPS based on tiles actually computed
    effective_S = int(S * (1.0 - skip_ratio))
    effective_S = max(effective_S, 1)
    tf = tflops(B, S, H, D, causal, ms)  # theoretical (full attention)
    tf_eff = tflops(B, effective_S, H, D, causal, ms)  # effective (computed tiles only -- approximate)
    return ms, tf, tf_eff


def bench_sdpa(B, S, H, D, causal=False):
    """Benchmark PyTorch SDPA."""
    # SDPA expects (B, H, S, D)
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')

    ms = bench_kernel(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
    tf = tflops(B, S, H, D, causal, ms)
    return ms, tf


if __name__ == "__main__":
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f"GPU: {name}, SM{cap[0]}{cap[1]}")
    print(f"PyTorch: {torch.__version__}")
    print()

    B, D = 1, 128
    H = 32

    # === Part 1: Flash Attention vs SDPA baseline ===
    print("=" * 90)
    print(f"Part 1: Flash Attention vs PyTorch SDPA  (B={B}, H={H}, D={D}, bf16)")
    print("=" * 90)
    print(f"{'S':>6}  {'Causal':>6}  {'FA ms':>8}  {'FA TF':>8}  {'SDPA ms':>8}  {'SDPA TF':>8}  {'Speedup':>8}")
    print("-" * 90)

    for causal in [False, True]:
        for S in [512, 1024, 2048, 4096, 8192]:
            fa_ms, fa_tf = bench_flash_attn(B, S, H, D, causal)
            sdpa_ms, sdpa_tf = bench_sdpa(B, S, H, D, causal)
            speedup = sdpa_ms / fa_ms
            tag = "yes" if causal else "no"
            print(f"{S:>6}  {tag:>6}  {fa_ms:>8.3f}  {fa_tf:>7.1f}T  {sdpa_ms:>8.3f}  {sdpa_tf:>7.1f}T  {speedup:>7.2f}x")

    # === Part 2: Skip list overhead ===
    print()
    print("=" * 90)
    print(f"Part 2: Skip list overhead (all tiles computed)  (B={B}, H={H}, D={D})")
    print("=" * 90)
    print(f"{'S':>6}  {'Phase':>6}  {'No-skip ms':>10}  {'Skip ms':>10}  {'Overhead':>10}")
    print("-" * 90)

    for S in [512, 1024, 2048, 4096, 8192]:
        fa_ms, _ = bench_flash_attn(B, S, H, D)
        for phase in [True, False]:
            skip_ms, _, _ = bench_skip_list(B, S, H, D, skip_ratio=0.0, phase=phase)
            overhead = (skip_ms - fa_ms) / fa_ms * 100
            tag = "A" if phase else "B"
            print(f"{S:>6}  {tag:>6}  {fa_ms:>10.3f}  {skip_ms:>10.3f}  {overhead:>+9.1f}%")

    # === Part 3: Skip ratio sweep ===
    print()
    print("=" * 90)
    print(f"Part 3: Skip ratio sweep  (B={B}, H={H}, D={D}, S=4096)")
    print("=" * 90)
    print(f"{'Skip%':>6}  {'Phase':>6}  {'ms':>8}  {'Theor TF':>9}  {'Eff TF':>9}  {'vs baseline':>11}")
    print("-" * 90)

    S = 4096
    fa_ms_base, _ = bench_flash_attn(B, S, H, D)
    print(f"{'0%':>6}  {'base':>6}  {fa_ms_base:>8.3f}  {'':>9}  {'':>9}  {'1.00x':>11}")

    for skip_pct in [0, 25, 50, 75, 90]:
        ratio = skip_pct / 100.0
        for phase in [True]:
            ms, tf, tf_eff = bench_skip_list(B, S, H, D, skip_ratio=ratio, phase=phase)
            speedup = fa_ms_base / ms
            print(f"{skip_pct:>5}%  {'A':>6}  {ms:>8.3f}  {tf:>8.1f}T  {tf_eff:>8.1f}T  {speedup:>10.2f}x")

    # === Part 4: Batch size scaling ===
    print()
    print("=" * 90)
    print(f"Part 4: Batch size scaling  (S=2048, H={H}, D={D})")
    print("=" * 90)
    print(f"{'B':>4}  {'FA ms':>8}  {'FA TF':>8}  {'SDPA ms':>8}  {'SDPA TF':>8}")
    print("-" * 60)

    S = 2048
    for B_test in [1, 2, 4, 8]:
        fa_ms, fa_tf = bench_flash_attn(B_test, S, H, D)
        sdpa_ms, sdpa_tf = bench_sdpa(B_test, S, H, D)
        print(f"{B_test:>4}  {fa_ms:>8.3f}  {fa_tf:>7.1f}T  {sdpa_ms:>8.3f}  {sdpa_tf:>7.1f}T")

    # === Part 5: Head count sweep ===
    print()
    print("=" * 90)
    print(f"Part 5: Head count sweep  (B=1, S=4096, D={D})")
    print("=" * 90)
    print(f"{'H':>4}  {'FA ms':>8}  {'FA TF':>8}  {'SDPA ms':>8}  {'SDPA TF':>8}")
    print("-" * 60)

    S = 4096
    for H_test in [1, 4, 8, 16, 32]:
        fa_ms, fa_tf = bench_flash_attn(1, S, H_test, D)
        sdpa_ms, sdpa_tf = bench_sdpa(1, S, H_test, D)
        print(f"{H_test:>4}  {fa_ms:>8.3f}  {fa_tf:>7.1f}T  {sdpa_ms:>8.3f}  {sdpa_tf:>7.1f}T")

    print()
    print("Done.")
