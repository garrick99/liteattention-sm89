import torch
import torch.nn.functional as F
import time
import os
import sys
import types
import importlib.util

dtype = torch.bfloat16
device = "cuda"
LA_ROOT = os.path.expanduser("~/projects/moonmath/LiteAttention")

# ── Bootstrap LiteAttention module ──
import torch  # must be first

pkg = types.ModuleType("lite_attention")
pkg.__path__ = [os.path.join(LA_ROOT, "hopper")]
sys.modules["lite_attention"] = pkg

spec = importlib.util.spec_from_file_location("lite_attention._C", os.path.join(LA_ROOT, "hopper/_C.abi3.so"))
cmod = importlib.util.module_from_spec(spec)
sys.modules["lite_attention._C"] = cmod
spec.loader.exec_module(cmod)
pkg._C = cmod

# Execute hopper/__init__.py in the package namespace
exec(open(os.path.join(LA_ROOT, "hopper/__init__.py")).read(), pkg.__dict__)
LiteAttention = pkg.LiteAttention
print("LiteAttention loaded")

# ── Create per-shape LA instances ──
la_instances = {}  # shape_key -> LiteAttention instance

def get_la(B, H, S, D):
    key = (B, H, S, D)
    if key not in la_instances:
        la_instances[key] = LiteAttention(threshold=-2.0).to(device)
    return la_instances[key]

la_calls = 0
la_time = 0.0
fallback_calls = 0
step_idx = [0]

_orig_sdpa = F.scaled_dot_product_attention

def la_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    global la_calls, la_time, fallback_calls
    
    if query.dim() != 4:
        fallback_calls += 1
        return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    
    B, H, S, D = query.shape

    if (query.dtype != torch.bfloat16 or D > 256 or D < 64
        or attn_mask is not None or not query.is_cuda):
        fallback_calls += 1
        return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    la = get_la(B, H, S, D)

    # [B,H,S,D] -> [B,S,H,D]
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()

    t0 = time.perf_counter()
    try:
        out = la(q, k, v)
        la_time += time.perf_counter() - t0
        la_calls += 1
        return out.transpose(1, 2)
    except Exception as e:
        fallback_calls += 1
        if fallback_calls <= 5:
            print(f"LA fallback: {type(e).__name__}: {e}")
        return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal, scale=scale)

# ── Load pipeline ──
print("Loading CogVideoX-2b...")
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=dtype)
pipe.enable_model_cpu_offload()
print("Pipeline loaded")

F.scaled_dot_product_attention = la_sdpa
print("SDPA patched with LiteAttention (skip lists active)")

prompt = "A golden retriever running through a sunlit meadow with wildflowers, cinematic quality, 4K"
print(f"Generating: {prompt}")

t0 = time.time()
with torch.no_grad():
    video = pipe(
        prompt=prompt,
        num_frames=16,
        guidance_scale=6.0,
        num_inference_steps=30,
        generator=torch.Generator("cpu").manual_seed(42),
    ).frames[0]

gen_time = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1024**3

print(f"Done in {gen_time:.1f}s | Peak VRAM: {peak:.1f} GB")
print(f"LA calls: {la_calls} | Fallback: {fallback_calls}")
print(f"LA instances: {len(la_instances)} unique shapes")
for k, la in la_instances.items():
    print(f"  Shape {k}")
if la_calls > 0:
    print(f"LA avg: {la_time/la_calls*1000:.2f}ms | Total: {la_time:.2f}s")

os.makedirs(os.path.expanduser("~/cogvideo_output"), exist_ok=True)
export_to_video(video, os.path.expanduser("~/cogvideo_output/skiplist.mp4"), fps=8)
print("Saved to ~/cogvideo_output/skiplist.mp4")

F.scaled_dot_product_attention = _orig_sdpa
