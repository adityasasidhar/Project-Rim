from diffusers import StableDiffusionPipeline
import os
import torch

# Folder to save generated images
output_folder = "static/images"

def clear_cuda_cache():
    torch.cuda.empty_cache()  # Frees the GPU memory cache
    print("CUDA cache cleared.")


def report_cuda_memory():
    print("CUDA Memory Summary:")
    print(torch.cuda.memory_summary(device=None, abbreviated=False))  # Summarizes GPU memory usage

# Function to generate high-quality images
def stablediffusion_generate_image(prompt, guidance_scale=10, num_inference_steps=150,
                                                seed=None, height=2160, width=3840):
    if seed is not None:
        torch.manual_seed(seed)

    # Load the Stable Diffusion model pipeline
    model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4').to('cuda')
    report_cuda_memory()

    # Generate the image
    image = model(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]

    return image



os.makedirs(output_folder, exist_ok=True)

prompt = "An Astronaut riding a horse on mars"

# Generate 4K image
print(f"Generating 4K image for prompt: \"{prompt}\"")
image = stablediffusion_generate_image(
    prompt=prompt,
    height=512,
    width=512,
    guidance_scale= 13,
    num_inference_steps=300,
    seed=42
)

# Save the image to the output directory
output_path = os.path.join(output_folder, "astronaut.png")
image.save(output_path)
print(f"4K Image saved at: {output_path}")
clear_cuda_cache()
