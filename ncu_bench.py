#!/usr/bin/env python3
"""Minimal script for ncu profiling — runs one FA call with and without skip lists."""
import torch
import lite_attention

fa = lite_attention._internal.flash_attn_interface.flash_attn_func

B, S, H, D = 1, 4096, 32, 128
kBlockN = 128

q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')

q_blocks = (S + kBlockN - 1) // kBlockN
k_blocks = (S + kBlockN - 1) // kBlockN
num_k_blocks = k_blocks + 2

read_list = torch.full((B, H, q_blocks, num_k_blocks), -1, dtype=torch.int16, device='cuda')
write_list = torch.full((B, H, q_blocks, num_k_blocks), -1, dtype=torch.int16, device='cuda')
for b in range(B):
    for h in range(H):
        for qb in range(q_blocks):
            read_list[b, h, qb, 0] = 2
            read_list[b, h, qb, 1] = 0
            read_list[b, h, qb, 2] = k_blocks

# Warmup
for _ in range(3):
    fa(q, k, v)
    fa(q, k, v, attn_read_list=read_list, attn_write_list=write_list, thr=-10.0, phase=True)
torch.cuda.synchronize()

# Profiled calls
torch.cuda.nvtx.range_push("FA_baseline")
fa(q, k, v)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("FA_skiplist")
fa(q, k, v, attn_read_list=read_list, attn_write_list=write_list, thr=-10.0, phase=True)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
