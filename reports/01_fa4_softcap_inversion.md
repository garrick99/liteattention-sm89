# GitHub Issue: Dao-AILab/flash-attention

## Title
Bug: SM90 forward softcap condition inverted in flash_fwd.py (breaks Gemma-2, Command-R+)

## Labels
bug, SM90, hopper

## Body

### Description

The softcap condition in `FlashAttentionForwardSm90.__call__` (`flash_attn/cute/flash_fwd.py`, around line 670) is inverted. When `softcap` is provided, the code discards it; when `softcap` is `None`, the else branch attempts to use `None` as a float.

### Bug

```python
# flash_fwd.py, ~line 670 (SM90 forward):
if const_expr(softcap is not None):    # <-- WRONG: should be "softcap is None"
    softmax_scale_log2 = softmax_scale * LOG2_E
    softcap_val = None
else:
    softmax_scale_log2 = softcap * LOG2_E
    softcap_val = Float32(softmax_scale / softcap)
```

The SM100 path has the correct condition:

```python
# flash_fwd_sm100.py, ~line 592 (SM100 forward):
if const_expr(softcap is None):        # <-- CORRECT
    softmax_scale_log2 = softmax_scale * LOG2_E
    softcap_val = None
else:
    softmax_scale_log2 = softcap * LOG2_E
    softcap_val = Float32(softmax_scale / softcap)
```

### Impact

- **Softcap is completely broken on SM90 (Hopper: H100, H200)**
- Models using softcap (Gemma-2, Command-R+, others) will produce incorrect attention outputs on Hopper GPUs
- Non-softcap usage (the common case) is unaffected because `softcap` defaults to `None` and the buggy `if` branch happens to do the right thing for that case — this is likely why the bug hasn't been caught
- SM100 (Blackwell) is unaffected — the condition is correct there

### Fix

One-character change: `is not None` → `is None`

```python
if const_expr(softcap is None):
    softmax_scale_log2 = softmax_scale * LOG2_E
    softcap_val = None
else:
    softmax_scale_log2 = softcap * LOG2_E
    softcap_val = Float32(softmax_scale / softcap)
```

### How Found

Line-by-line code review of the FA4 CuTeDSL codebase comparing SM90 and SM100 forward paths. The SM100 path was used as the reference implementation.

### Environment

- flash-attention fa4 branch (CuTeDSL implementation)
- Affected: All SM90 GPUs (H100, H200) when softcap is enabled
- Not affected: SM100 (Blackwell), non-softcap attention
