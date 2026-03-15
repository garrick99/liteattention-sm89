# BackLite Code Review

**Date:** 2026-03-15
**Reviewer:** Garrick (with Claude)
**Codebase:** moonmath-ai/BackLite (1,062 commits)
**Hardware tested on:** None (SM90 Hopper required; RTX 4090 is SM89)
**Repository:** https://github.com/moonmath-ai/BackLite

---

## What BackLite Does

BackLite wraps FA3 to add **sparse backward passes** for transformer attention training. The forward pass is exact FA3 — no approximation. During the forward, per-tile log-sum-exp (LSE) statistics are recorded. A fused Triton kernel then converts these LSE stats into a bitmask identifying which tiles contribute negligible probability mass (below a configurable `negl_prob` threshold). The backward pass skips masked tiles entirely — no memory reads, no FLOPs.

**Key insight:** In many workloads (especially diffusion models), a large fraction of attention tiles have near-zero probability mass. Skipping them in the backward pass gives a significant speedup with negligible gradient error.

---

## Architecture

### Forward Pass
- Standard FA3 forward kernel with an additional `tile_stats` output buffer
- `tile_stats[B, H, seq_q, N_k]` stores per-tile LSE in log2 domain
- The tile_stats tensor is allocated with a transposed stride (`permute(0, 1, 3, 2)`) to give `stride_m=1` — the FA3 kernel requires this layout
- **Critical comment in code:** "do NOT call .contiguous() here" — doing so relayouts to `stride_m=N`, causing `cudaErrorMisalignedAddress` in the kernel

### Mask Generation (Triton)
- `mask_from_stats_fused()` in `tile_stats_reduce.py`
- Grid = `(BH, N)`: one CTA per (batch-head, KV-block)
- Each CTA loops over all `Tm` Q-tiles, computes tile_mass, normalizes, sorts, thresholds
- Output: `int32[B, H, N, NUM_WORDS]` packed bitmask
- Uses `tl.sort` + `tl.cumsum` for threshold computation — elegant but limits `Tm ≤ 2048` (max seq_len ~256K at kBlockN=128)

### Backward Pass (CUDA, SM90)
- Modified FA3 backward kernel in `mainloop_bwd_sm90_tma_gmma_ws.hpp`
- `MaskBits` struct loaded in `get_mask_words()` — reads bitmask and unpacks active m-block indices into shared memory
- Inner loop iterates only over active m-blocks via `next_active()`

---

## Bug 1: `get_mask_words` only processes first 32 m-blocks

**File:** `hopper/_internal/cpp/mainloop_bwd_sm90_tma_gmma_ws.hpp`, `get_mask_words()` (~line 494)
**Severity:** HIGH

The function reads `first_word = __ldg(ptr)` and uses `__popc(first_word)` to count active blocks. It then unpacks bit positions into `shared_storage.compatition_m_blocks[]` using a warp-wide prefix-popcount pattern — but only for the **first 32-bit word**.

The Triton mask kernel produces `NUM_WORDS = TM_PAD // 32` words per (bh, n) pair. When `Tm > 32` (i.e., `seq_q > 32 * kBlockM_bwd`, which is basically any non-trivial sequence length), the remaining words are never read. **All m-blocks with index ≥ 32 are silently dropped.**

For example, with kBlockM_bwd=64 and seq_q=4096: Tm=64, NUM_WORDS=2. The second word (m-blocks 32-63) is completely ignored. Half the Q-tiles are never processed in the backward pass.

```cpp
// CURRENT: only reads word 0
uint32_t first_word = (num_words > 0) ? __ldg(ptr) : 0u;
int length = __popc(first_word);

// ... only unpacks bits from first_word into shared memory ...
```

**Fix:** Must iterate over all `num_words` words, accumulating active m-block indices. The shared storage array `compatition_m_blocks[32]` is also undersized — it can hold at most 32 active indices, but there could be up to `Tm` (up to 2048) active blocks.

---

## Bug 2: `compatition_m_blocks` array too small

**File:** `hopper/_internal/cpp/flash_bwd_kernel_sm90.h`, line 87
**Severity:** HIGH (related to Bug 1)

```cpp
alignas(16) int compatition_m_blocks[32];
```

This shared memory array holds the unpacked list of active m-block indices. It's hardcoded to 32 entries. When `Tm > 32` and more than 32 m-blocks are active, writing past index 31 is an out-of-bounds shared memory write — undefined behavior, likely silent data corruption.

**Fix:** Size should be `max(32, Tm)` or dynamically bounded. Since `Tm` can be up to 2048, this needs to be either:
- A dynamic allocation in shared memory, or
- Bounded by the actual max Tm for the kernel configuration

---

## Bug 3: Typo — `compatition_m_blocks` should be `compaction_m_blocks`

**File:** `mainloop_bwd_sm90_tma_gmma_ws.hpp` (lines 516, 522, 526) and `flash_bwd_kernel_sm90.h` (line 87)
**Severity:** Low (cosmetic, but confusing)

`compatition` is not a word. Based on context (bit unpacking / prefix-popcount stream compaction), this should be `compaction_m_blocks`.

---

## Bug 4: Missing `__syncthreads()` after shared memory population

**File:** `mainloop_bwd_sm90_tma_gmma_ws.hpp`, `get_mask_words()` (~line 527)
**Severity:** MEDIUM

The function writes to `shared_storage.compatition_m_blocks[]` using only threads in `threadIdx.x < 32` (first warp), then immediately returns the `MaskBits` struct pointing to that shared memory. There is no `__syncthreads()` between the write and subsequent reads by other warps/warpgroups.

The backward mainloop has multiple warpgroups (NumMmaWarpGroups=2-3 + 1 producer). If any warp outside the first warp reads from `compatition_m_blocks` before the first warp's writes are visible, you get a race condition.

**Fix:** Add `__syncthreads()` after the shared memory population block:
```cpp
if (threadIdx.x < 32) {
    // ... existing bit-unpack code ...
}
__syncthreads();  // ensure all warps see the compacted indices
return {shared_storage.compatition_m_blocks, length, first_word, -1};
```

---

## Bug 5: `back_lite.py` `get_MN()` hardcodes SM90 tile sizes

**File:** `hopper/back_lite.py`, `get_MN()` static method
**Severity:** MEDIUM (same bug as LiteAttention)

```python
result = _back_lite_ops.get_tile_size_fwd_sm90(
    head_dim, head_dim, False, False, element_size,
    v_colmajor, False, False, is_int8,
)
```

Unconditionally calls `get_tile_size_fwd_sm90()` regardless of actual GPU architecture. On SM89 (RTX 4090) or future SM120 (Blackwell), this returns wrong tile sizes. The SM80/SM89 kernel uses different block dimensions (e.g., kBlockN=128 vs SM90's kBlockN=176 for hdim=128).

This is the exact same bug as LiteAttention Bug #1. When used on SM89, the mask dimensions won't match the kernel's actual tile decomposition.

**Fix:** Same as LiteAttention — detect compute capability and dispatch:
```python
arch = torch.cuda.get_device_capability()
sm_version = arch[0] * 10 + arch[1]
if sm_version >= 90:
    result = _back_lite_ops.get_tile_size_fwd_sm90(...)
else:
    result = _back_lite_ops.get_tile_size_fwd_sm8x(...)
```

---

## Bug 6: `seq_q % kBlockM_bwd != 0` raises RuntimeError instead of padding

**File:** `hopper/_internal/flash_attn_interface.py`, `_generate_mask_from_stats_mass()`, ~line 57
**Severity:** LOW

```python
if seq_q % kBlockM_bwd != 0:
    raise RuntimeError(f"Mask generation requires seq_q divisible by kBlockM_bwd ...")
```

This is overly restrictive. The Triton bitmask kernel already handles partial tiles correctly via `valid_r = row_offs < seq_q` masking. The restriction should be removed or relaxed to a warning. Many practical sequence lengths (e.g., 2048 with kBlockM_bwd=192) won't be divisible.

---

## Observation: Triton Kernel Quality

The `_fused_bitmask_kernel` in `tile_stats_reduce.py` is well-written:
- Uses `range()` instead of `tl.static_range()` to avoid O(Tm) PTX unrolling — good call for large Tm
- Scatter-via-`tl.where` is an acceptable pattern for Triton (compiles to vectorized compare+select)
- The sort+cumsum+threshold approach for determining which tiles to ignore is mathematically clean
- The tie-breaking logic (`n_ties_to_ignore`, `tie_cumcount`) is correct — handles cases where multiple tiles have the same probability mass

**Performance note:** At large Tm (>512), the `for m in range(Tm)` loop will dominate kernel runtime. Each iteration does a full `tl.where` over `TM_PAD` elements to scatter one value. For Tm=2048, that's 2048 × 2048 = 4M compare+select operations. A register-packed approach (one thread per Q-tile) would be faster but would require a different kernel structure.

---

## Observation: tile_stats Tensor Layout

The forward pass allocates tile_stats with a specific transposed layout:
```python
tile_stats = torch.full(
    (batch, heads, num_n_blocks, seq_q), float('-inf'),
    ...
).permute(0, 1, 3, 2)  # [B,H,T,N] with stride_m=1
```

The comment warns: "do NOT call .contiguous()". This is because the FA3 kernel writes tile_stats with stride-1 along the seq_q dimension. If someone adds `.contiguous()` during refactoring, it silently changes the stride and the kernel produces `cudaErrorMisalignedAddress`. This should be an assertion or a more prominent guard.

---

## Observation: No SM80/SM89 Backward Sparsity

The SM80 backward mainloop (`mainloop_bwd_sm80.hpp`) has no block sparsity support — no `ptr_block_mask`, no `MaskBits`, no tile skipping. BackLite's sparsity features are SM90-only.

This means:
- On RTX 4090 (SM89): Forward works (FA3), but backward sparsity is not available. `negl_prob > 0` will generate the bitmask but the SM80 backward kernel will ignore it.
- The code silently falls back to dense backward, which is correct but defeats the purpose of BackLite.

---

## Observation: INT8 Quantization Path

The `_quantize_query_key` method in `back_lite.py` implements SageAttention-style quantization:
- Q: per-block quantization with shared scale per head
- K: smooth by subtracting channel-wise mean, then per-block quantization
- Both go through the C++ `quantize_qk` operator

This is independent of the sparsity path and can be used without `negl_prob`. The composition (INT8 + sparse backward) should work but is likely undertested given the bugs above.

---

## Findings Summary

| # | Finding | Severity | File | Line(s) |
|---|---------|----------|------|---------|
| 1 | `get_mask_words` only reads first 32-bit word — drops m-blocks ≥ 32 | **HIGH** | `mainloop_bwd_sm90_tma_gmma_ws.hpp` | 506-526 |
| 2 | `compatition_m_blocks[32]` too small for Tm > 32 | **HIGH** | `flash_bwd_kernel_sm90.h` | 87 |
| 3 | Typo: `compatition` → `compaction` | Low | multiple | — |
| 4 | Missing `__syncthreads()` after shared memory writes | **MEDIUM** | `mainloop_bwd_sm90_tma_gmma_ws.hpp` | ~527 |
| 5 | `get_MN()` hardcodes SM90 tile sizes | Medium | `back_lite.py` | `get_MN()` |
| 6 | `seq_q % kBlockM_bwd != 0` overly restrictive | Low | `flash_attn_interface.py` | ~57 |
| — | No SM80/SM89 backward sparsity support | Info | — | — |
| — | tile_stats layout fragility (.contiguous() trap) | Info | `flash_attn_interface.py` | ~309 |

---

## Recommendations

1. **Fix bugs 1+2 together** — the `get_mask_words` function needs to iterate over all bitmask words and the shared memory array needs to be large enough. This is the critical path bug — with seq_q > 32 * kBlockM_bwd, BackLite silently produces incorrect gradients.

2. **Add `__syncthreads()`** (bug 4) — race condition between first warp's writes and other warpgroups' reads of the compaction array.

3. **Fix `get_MN()` arch dispatch** (bug 5) — same fix as LiteAttention, already proven to work.

4. **Add a correctness test with seq_q > 2048** — current tests may pass because kBlockM_bwd is large enough that Tm ≤ 32. The multi-word bug only manifests at longer sequences.

5. **Consider making the tile_stats layout less fragile** — either assert the stride or use a dedicated allocation function that guarantees the correct layout.
