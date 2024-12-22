import os
import torch
from diffusers import StableDiffusionPipeline

print(torch.cuda.is_available())

output_folder = "AI generated Videos"

def clear_cuda_cache():
    torch.cuda.empty_cache()  # Frees the GPU memory cache
    print("CUDA cache cleared.")


def report_cuda_memory():
    print("CUDA Memory Summary:")
    print(torch.cuda.memory_summary(device=None, abbreviated=False))  # Summarizes GPU memory usage

def generate_frames(name,prompt,context, guidance_scale=12.5, num_inference_steps=150, seed=None, height=512, width=512,):
    model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4').to('cuda')
    report_cuda_memory()
    if seed is not None:
        torch.manual_seed(seed)

    seed = 42

    print(f"Prompt: {prompt}"
          f"Context: {context}"
          f"Guidance Scale: {guidance_scale}"
          f"Num Inference Steps: {num_inference_steps}"
          f"Seed: {seed}"
          f"Height: {height}"
          f"Width: {width}")

    frame = model(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]
    frame.save(f"{output_folder}/{name}/{frame}.png")
    return frame
