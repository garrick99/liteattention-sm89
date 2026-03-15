# GitHub Issue: Dao-AILab/flash-attention

## Title
SM90 forward: local (sliding window) mask left-side not implemented

## Labels
bug, SM90, hopper, sliding-window

## Body

### Description

The SM90 forward mainloop in `flash_attn/cute/flash_fwd.py` (around line 979) has a `# TODO: local` comment where the left-side local attention mask iterations should be. The `is_local` flag propagates through `BlockInfo` and `AttentionMask`, but the mainloop never generates separate masked iterations for the left boundary of the sliding window.

### Expected Behavior

The SM100 path (`flash_fwd_sm100.py`, around lines 1529-1541) properly handles this:

```python
# SM100 — correctly implements local mask left-side:
if const_expr(self.is_local and block_info.window_size_left is not None):
    n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
    for n_tile in cutlass.range(0, n_block_max - n_block_min, unroll=1):
        ...
```

The SM90 path should have equivalent logic but instead has only `# TODO: local`.

### Impact

- **Sliding window attention (`window_size_left`) produces incorrect results on SM90 (Hopper)**
- Tiles near the left boundary of the attention window will include tokens that should be masked
- Standard causal and non-causal attention are unaffected
- SM100 (Blackwell) is unaffected — the implementation is correct there

### Workaround

Use full causal attention instead of sliding window on Hopper, or run on Blackwell where the implementation is complete.

### How Found

Code review comparing SM90 and SM100 forward paths side by side.

### Environment

- flash-attention fa4 branch (CuTeDSL implementation)
- Affected: All SM90 GPUs (H100, H200) using sliding window / local attention
- Not affected: SM100 (Blackwell), non-local attention
