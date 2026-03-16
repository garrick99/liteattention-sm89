"""
CogVideoX baseline: generate a video with standard FA2 attention.
Uses CPU offloading to fit in 24GB VRAM.
"""
import torch
import time
import os

dtype = torch.bfloat16
device = "cuda"

print("Loading CogVideoX-2b pipeline...")
t0 = time.time()

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=dtype,
)
# CPU offload moves modules to GPU only when needed
pipe.enable_model_cpu_offload()

load_time = time.time() - t0
print(f"Pipeline loaded in {load_time:.1f}s")

prompt = "A golden retriever running through a sunlit meadow with wildflowers, cinematic quality, 4K"

print(f"Generating video with FA2 baseline...")
print(f"Prompt: {prompt}")

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
print(f"Generation complete in {gen_time:.1f}s")

peak_mem = torch.cuda.max_memory_allocated() / 1024**3
print(f"Peak VRAM: {peak_mem:.1f} GB")

os.makedirs(os.path.expanduser("~/cogvideo_output"), exist_ok=True)
output_path = os.path.expanduser("~/cogvideo_output/baseline_fa2.mp4")
export_to_video(video, output_path, fps=8)
print(f"Saved to {output_path}")
