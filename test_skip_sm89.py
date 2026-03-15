"""Comprehensive skip list test for SM89."""
import torch
import lite_attention

fa = lite_attention._internal.flash_attn_interface.flash_attn_func

def test_correctness(S, H, phase, causal=False):
    """Verify skip list output matches reference (no skip)."""
    B, D = 1, 128
    kBlockN = 128  # SM89 hdim128
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')

    q_blocks = (S + kBlockN - 1) // kBlockN
    k_blocks = (S + kBlockN - 1) // kBlockN
    num_k_blocks = k_blocks + 2

    # Read list: all blocks non-skipped
    read_list = torch.full((B, H, q_blocks, num_k_blocks), -1, dtype=torch.int16, device='cuda')
    for b in range(B):
        for h in range(H):
            for qb in range(q_blocks):
                read_list[b, h, qb, 0] = 2
                read_list[b, h, qb, 1] = 0
                read_list[b, h, qb, 2] = k_blocks

    write_list = torch.full((B, H, q_blocks, num_k_blocks), -1, dtype=torch.int16, device='cuda')

    # Reference (no skip list)
    ref = fa(q, k, v, causal=causal)
    torch.cuda.synchronize()

    # With skip list
    out = fa(q, k, v, attn_read_list=read_list, attn_write_list=write_list,
             thr=-10.0, phase=phase, causal=causal)
    torch.cuda.synchronize()

    max_diff = (out - ref).abs().max().item()
    status = "PASS" if max_diff < 0.01 else "FAIL"
    tag = f"S={S:4d} H={H} phase={phase!s:5s} causal={causal!s:5s}"
    print(f"  {status}: {tag}  max_diff={max_diff:.6f}")
    return max_diff < 0.01

def test_multi_range(S, H, phase):
    """Test with multiple ranges in the skip list."""
    B, D = 1, 128
    kBlockN = 128
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')

    q_blocks = (S + kBlockN - 1) // kBlockN
    k_blocks = (S + kBlockN - 1) // kBlockN
    num_k_blocks = k_blocks + 2

    if k_blocks < 4:
        print(f"  SKIP: S={S} too small for multi-range test")
        return True

    # Read list: two non-contiguous ranges [0,2) and [3,k_blocks)
    read_list = torch.full((B, H, q_blocks, num_k_blocks), -1, dtype=torch.int16, device='cuda')
    for b in range(B):
        for h in range(H):
            for qb in range(q_blocks):
                read_list[b, h, qb, 0] = 4      # length (4 values = 2 ranges)
                read_list[b, h, qb, 1] = 0       # range 1 start
                read_list[b, h, qb, 2] = 2       # range 1 end
                read_list[b, h, qb, 3] = 3       # range 2 start
                read_list[b, h, qb, 4] = k_blocks  # range 2 end

    write_list = torch.full((B, H, q_blocks, num_k_blocks), -1, dtype=torch.int16, device='cuda')

    try:
        out = fa(q, k, v, attn_read_list=read_list, attn_write_list=write_list,
                 thr=-10.0, phase=phase)
        torch.cuda.synchronize()
        tag = f"S={S:4d} H={H} phase={phase!s:5s} multi_range"
        print(f"  PASS: {tag}  out_max={out.abs().max():.4f}")
        print(f"    write[0,0,0]={write_list[0,0,0].tolist()}")
        return True
    except RuntimeError as e:
        print(f"  FAIL multi_range S={S} phase={phase}: {e}")
        return False

def test_unaligned_seqlen():
    """Test with sequence length not aligned to kBlockN."""
    B, H, D = 1, 4, 128
    kBlockN = 128
    S = 500  # not a multiple of 128

    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')

    q_blocks = (S + kBlockN - 1) // kBlockN  # 4
    k_blocks = (S + kBlockN - 1) // kBlockN  # 4
    num_k_blocks = k_blocks + 2  # 6

    read_list = torch.full((B, H, q_blocks, num_k_blocks), -1, dtype=torch.int16, device='cuda')
    for b in range(B):
        for h in range(H):
            for qb in range(q_blocks):
                read_list[b, h, qb, 0] = 2
                read_list[b, h, qb, 1] = 0
                read_list[b, h, qb, 2] = k_blocks

    write_list = torch.full((B, H, q_blocks, num_k_blocks), -1, dtype=torch.int16, device='cuda')

    ref = fa(q, k, v)
    torch.cuda.synchronize()

    for phase in [True, False]:
        try:
            out = fa(q, k, v, attn_read_list=read_list, attn_write_list=write_list,
                     thr=-10.0, phase=phase)
            torch.cuda.synchronize()
            max_diff = (out - ref).abs().max().item()
            status = "PASS" if max_diff < 0.01 else "FAIL"
            print(f"  {status}: S={S} (unaligned) phase={phase!s:5s}  max_diff={max_diff:.6f}")
        except RuntimeError as e:
            print(f"  FAIL: S={S} (unaligned) phase={phase}: {e}")

if __name__ == "__main__":
    print(f"CUDA: {torch.cuda.get_device_name()}, cap={torch.cuda.get_device_capability()}")
    print()

    print("=== Correctness (skip-all = no skip) ===")
    all_pass = True
    for S in [128, 256, 512, 1024]:
        for H in [1, 4, 32]:
            for phase in [True, False]:
                all_pass &= test_correctness(S, H, phase)
    for S in [512]:
        for phase in [True, False]:
            all_pass &= test_correctness(S, 4, phase, causal=True)
    print(f"\nAll correctness: {'PASS' if all_pass else 'FAIL'}")
    print()

    print("=== Multi-range ===")
    for S in [512, 1024]:
        for phase in [True, False]:
            test_multi_range(S, 4, phase)
    print()

    print("=== Unaligned sequence length ===")
    test_unaligned_seqlen()
