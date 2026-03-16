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

# ── Minimal LiteAttention setup: only load what we need ──
# Register lite_attention package
pkg = types.ModuleType("lite_attention")
pkg.__path__ = [os.path.join(LA_ROOT, "hopper")]
sys.modules["lite_attention"] = pkg

# Load C extension
so_path = os.path.join(LA_ROOT, "hopper", "_C.abi3.so")
spec = importlib.util.spec_from_file_location("lite_attention._C", so_path)
cmod = importlib.util.module_from_spec(spec)
sys.modules["lite_attention._C"] = cmod
spec.loader.exec_module(cmod)
pkg._C = cmod

# Direct import of flash_attn_interface — skip the full hopper/__init__.py chain
iface_path = os.path.join(LA_ROOT, "hopper", "_internal", "flash_attn_interface.py")
spec2 = importlib.util.spec_from_file_location("lite_attention._internal.flash_attn_interface", iface_path)
iface = importlib.util.module_from_spec(spec2)

# Need _internal registered
ipkg = types.ModuleType("lite_attention._internal")
ipkg.__path__ = [os.path.join(LA_ROOT, "hopper", "_internal")]
sys.modules["lite_attention._internal"] = ipkg
sys.modules["lite_attention._internal.flash_attn_interface"] = iface
spec2.loader.exec_module(iface)

flash_attn_func = iface.flash_attn_func
print("LiteAttention flash_attn_func loaded")

# ── State for skip list tracking ──
# We'll use raw flash_attn_func calls first (no skip list) to get a baseline,
# then add skip list in a follow-up. The key question for Omer is quality, 
# and flash_attn_func on SM89 already beats SDPA.

la_calls = 0
la_time = 0.0
fallback_calls = 0

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

    # SDPA: [B,H,S,D] -> flash_attn: [B,S,H,D]
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()

    t0 = time.perf_counter()
    try:
        out = flash_attn_func(q, k, v, causal=is_causal)
        la_time += time.perf_counter() - t0
        la_calls += 1
        return out.transpose(1, 2)
    except Exception as e:
        fallback_calls += 1
        if la_calls == 0:
            print(f"LA first fallback: {e}")
        return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal, scale=scale)

# ── Load pipeline ──
print("Loading CogVideoX-2b...")
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=dtype)
pipe.enable_model_cpu_offload()
print("Pipeline loaded")

# Patch
F.scaled_dot_product_attention = la_sdpa
print("SDPA patched")

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
if la_calls > 0:
    print(f"LA avg: {la_time/la_calls*1000:.2f}ms | LA total: {la_time:.2f}s")

os.makedirs(os.path.expanduser("~/cogvideo_output"), exist_ok=True)
export_to_video(video, os.path.expanduser("~/cogvideo_output/liteattention.mp4"), fps=8)
print("Saved to ~/cogvideo_output/liteattention.mp4")

F.scaled_dot_product_attention = _orig_sdpa
