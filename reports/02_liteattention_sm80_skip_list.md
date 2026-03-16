# GitHub Issue: moonmath-ai/LiteAttention

## Title
Bug: 6 bugs in SM80/SM89 skip list forward path (v0.4.0)

## Labels
bug, SM80, SM89

## Body

### Description

Six bugs were found in the SM80/SM89 skip list implementation during integration testing on RTX 4090 (SM89, CUDA 12.4). All bugs are in the forward-pass skip list path (`Is_skipable=true`). The non-skip path is unaffected.

Bugs 3 and 4 together make the `Phase=false` path completely non-functional, causing oscillating bad results across consecutive forward passes.

### Bug 1: `get_MN()` always calls SM90 tile size function

**File:** `hopper/lite_attention.py`, `get_MN()` static method

`get_MN()` unconditionally calls `get_tile_size_fwd_sm90()`, which returns `kBlockN=176` for hdim=128. On SM89, the kernel uses `tile_size_fwd_sm8x()` with `kBlockN=128`. The skip list is sized for `ceil(S/176)` k-tiles but the kernel expects `ceil(S/128)` k-tiles, causing silent data corruption or OOB access.

**Fix:** Detect compute capability and dispatch:
```python
arch = torch.cuda.get_device_capability()
sm_version = arch[0] * 10 + arch[1]
if sm_version >= 90:
    result = _lite_attention_ops.get_tile_size_fwd_sm90(...)
else:
    result = _lite_attention_ops.get_tile_size_fwd_sm8x(...)
```

This also requires adding a `get_tile_size_fwd_sm8x` C++ wrapper in `flash_api.cpp` with TORCH_LIBRARY registration.

### Bug 2: Skip list writer never closes last range

**File:** `hopper/_internal/cpp/mainloop_fwd_sm80.hpp`

When the reader has no more ranges (`has_more() == false`), the main loop exits without closing the last range. The writer's `finalize()` only writes the length field — it does not close open ranges. Result: `calc_percentage()` returns garbage (e.g. 52476%).

The SM90 kernel avoids this via `DelayedSkipListWriter` which flushes its delay buffer in `finalize()`. The SM80 port uses the basic `SkipListWriter` and needs an explicit range closure.

**Fix:** After the while loop:
```cpp
if (thread_idx == 0) {
    skip_writer.record_range_end(false, skip_reader.end_idx);
    skip_writer.finalize();
}
```

### Bug 3: `range_first` returns wrong tile for Phase=false

**File:** `hopper/_internal/cpp/mainloop_fwd_sm80.hpp`

```cpp
auto range_first = [](int start, int end) {
    return Phase ? start : end - 1;  // BUG
};
```

For the Reverse=True reader, `start_idx` is the HIGH bound and `end_idx` is the LOW bound (already adjusted by `+step` during `load_range()`). Backward iteration should start at `start_idx`, not `end_idx - 1` (which underflows by 2).

Example: Read list `[2, 0, 16]`. Phase=false: `start_idx = 15`, `end_idx = -1`. `range_first(15, -1)` returns `-2` instead of `15`.

**Fix:**
```cpp
auto range_first = [](int start, int /*end*/) {
    return start;
};
```

### Bug 4: `at_range_end` checks wrong bound for Phase=false

**File:** `hopper/_internal/cpp/mainloop_fwd_sm80.hpp`

```cpp
if constexpr (Phase) {
    at_range_end = (n_block + 1 >= skip_reader.end_idx);  // Correct
} else {
    at_range_end = (n_block - 1 < skip_reader.start_idx); // BUG: checks high bound
}
```

For Phase=false, `start_idx` is the HIGH bound. This condition is true for almost every tile, causing the loop to exit after one tile.

**Fix:**
```cpp
} else {
    at_range_end = (n_block - 1 <= skip_reader.end_idx);  // Check low bound
}
```

### Bug 5: `record_range_end` called before current tile's skip decision

**File:** `hopper/_internal/cpp/mainloop_fwd_sm80.hpp`

When `at_range_end` is true and the reader has more ranges, `record_range_end()` is called before `fwd_step_skip()` processes the current tile. This inserts the range-end marker before the tile's transition entry, producing incorrect skip list sequences.

Example: Compute ranges [0,35) and [72,143) produce `[4, 0, 35, 34, 143]` instead of `[4, 0, 35, 72, 143]`.

**Fix:** Defer `record_range_end()` until after `fwd_step_skip()`:
```cpp
bool pending_range_end = false;
int pending_range_end_idx = 0;

// In the at_range_end branch:
pending_range_end = true;
pending_range_end_idx = skip_reader.end_idx;

// After fwd_step_skip:
if (pending_range_end && thread_idx == 0) {
    skip_writer.record_range_end(false, pending_range_end_idx);
    pending_range_end = false;
}
```

### Bug 6: Must-skip list initialization off-by-one

**File:** `hopper/lite_attention.py`, `init_skip_list()` static method

The normal "compute all" initialization stores `[2, ktiles-1, -1]` (pre-adjusted by -1 because the reader adds `+step`). The must-skip initialization stores unadjusted tile indices, causing the reader to compute `end = ktiles + 1` (one past valid), resulting in CUDA illegal memory access.

**Fix:**
```python
tile_indices = [x - 1 for x in tile_indices]
tile_indices = [len(tile_indices)] + list(reversed(tile_indices))
```

### Test Results After Fixes

RTX 4090 (SM89), hdim=128, bf16:

| Test | Status |
|------|--------|
| test_skip_all | PASS |
| test_skip_nothing | PASS |
| test_softmax_lse_correctness | PASS |
| test_rectangular_attention_correctness | PASS |
| test_rectangular_attention_skipping_twice | PASS |
| test_must_skip_list (6 cases) | PASS |
| test_stress | PASS |

Integration test: 10 consecutive forward passes, structured attention (first 512 tokens correlated), threshold=-1.0:
```
Fwd  1: compute=100.0%, cos_sim=0.99999613
Fwd  2: compute= 81.2%, cos_sim=1.00000000
Fwd  3: compute= 81.2%, cos_sim=0.99999613
...stable through Fwd 10...
```

### Environment

- LiteAttention v0.4.0
- RTX 4090 (SM89), CUDA 12.4
- High-core-count CPU, Ubuntu
- bf16, hdim=128
