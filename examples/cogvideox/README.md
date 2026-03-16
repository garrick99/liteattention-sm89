# CogVideoX Integration — SM89 (RTX 4090)

Real video model integration of LiteAttention with CogVideoX-2b on RTX 4090.

## Results

### CogVideoX-2b on RTX 4090 (SM89)
| Config | Time | Speed | Speedup |
|--------|------|-------|---------|
| FA2 baseline | 17.2s | 2.77 it/s | 1.0x |
| LA kernel only | 17.7s | 2.73 it/s | ~1.0x |
| LA skip list (thr=-2.0) | 14.9s | 3.61 it/s | 1.30x |

### Quality
- LA kernel only: visually identical to baseline (same math, cos_sim=1.0)
- LA skip list: more detail than baseline in some areas, 30% faster
- 900 SDPA calls replaced, 0 fallbacks
- Skip list settles by step 2, ramps from 3.24s/it to 3.61 it/s

### Key Finding: Per-Layer Skip List Isolation
Each attention layer must have its own skip list instance. Sharing skip lists
across layers with the same tensor shape produces garbled output because each
layer has different attention patterns. Key by call order within a denoising
step, not by tensor shape.

## Scripts

- `baseline_fa2.py` — Standard CogVideoX with PyTorch SDPA
- `liteattention_kernel_only.py` — LA kernel swap, no skip lists
- `liteattention_skiplist.py` — Full LA with per-layer skip lists
