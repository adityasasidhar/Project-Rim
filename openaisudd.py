import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

print(torch.cuda.is_available())

output_folder = "AI generated Videos"


def clear_cuda_cache():
    """Frees the GPU memory cache."""
    torch.cuda.empty_cache()
    print("CUDA cache cleared.")


def report_cuda_memory():
    """Prints a summary of GPU memory usage."""
    print("CUDA Memory Summary:")
    print(torch.cuda.memory_summary(device=None, abbreviated=False))


def generate_frames(name, prompt, context, guidance_scale=12.5, num_inference_steps=150, seed=None, height=512,
                    width=512):
    """
    Generate a single frame using the Stable Diffusion model.

    Args:
        name (str): Name of the output subfolder.
        prompt (str): Text prompt for the model.
        context (str): Additional context for the prompt.
        guidance_scale (float): The classifier-free guidance scale.
        num_inference_steps (int): Number of inference steps.
        seed (int, optional): Random seed for reproducibility.
        height (int): Height of the generated image.
        width (int): Width of the generated image.

    Returns:
        PIL.Image: The generated frame.
    """
    # Ensure the output folder exists
    os.makedirs(f"{output_folder}/{name}", exist_ok=True)

    # Load the Stable Diffusion model
    model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4').to('cuda')

    report_cuda_memory()

    # Set the random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    print(f"Prompt: {prompt}\n"
          f"Context: {context}\n"
          f"Guidance Scale: {guidance_scale}\n"
          f"Num Inference Steps: {num_inference_steps}\n"
          f"Seed: {seed}\n"
          f"Height: {height}\n"
          f"Width: {width}")

    # Generate the image
    frame = model(
        prompt=f"{prompt}, {context}",
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]

    # Save the image
    output_path = f"{output_folder}/{name}/frame_{seed if seed is not None else 'default'}.png"
    frame.save(output_path)
    print(f"Frame saved at: {output_path}")
    return frame


def generate_video(name, prompt, context, seconds, fps, guidance_scale=12.5, num_inference_steps=150, height=512,
                   width=512):
    """
    Generate a video by creating multiple frames using the Stable Diffusion model.

    Args:
        name (str): Name of the output subfolder.
        prompt (str): Text prompt for the model.
        context (str): Additional context for the prompt.
        seconds (int): Length of the video in seconds.
        fps (int): Frames per second.
        guidance_scale (float): The classifier-free guidance scale.
        num_inference_steps (int): Number of inference steps.
        height (int): Height of the generated images.
        width (int): Width of the generated images.

    Returns:
        None
    """
    total_frames = seconds * fps
    os.makedirs(f"{output_folder}/{name}", exist_ok=True)

    for frame_num in range(total_frames):
        seed = frame_num  # Unique seed for each frame
        print(f"Generating frame {frame_num + 1}/{total_frames}")
        frame = generate_frames(name, prompt, context, guidance_scale, num_inference_steps, seed, height, width)
        frame.save(f"{output_folder}/{name}/frame_{frame_num:04d}.png")

    print(f"Video frames saved in {output_folder}/{name}")


# Example usage
if __name__ == "__main__":
    clear_cuda_cache()

    name = "Example_Video"
    prompt = "A futuristic cityscape at sunset"
    context = "with flying cars and neon lights"
    guidance_scale = 12.5
    num_inference_steps = 50
    seconds = 5  # Length of the video in seconds
    fps = 3  # Frames per second
    height = 512
    width = 512

    generate_video(name, prompt, context, seconds, fps, guidance_scale, num_inference_steps, height, width)
