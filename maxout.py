from diffusers import StableDiffusionPipeline
import os
import torch
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"GPU is active: {torch.cuda.is_available()}")

# Output folder for saving generated images
output_folder = "AI generate images"


def clear_cuda_cache():
    """Frees the GPU memory cache."""
    torch.cuda.empty_cache()
    logging.info("CUDA cache cleared.")


def report_cuda_memory():
    """Logs CUDA memory stats."""
    logging.info("CUDA Memory Summary:")
    logging.info(torch.cuda.memory_summary(device=None, abbreviated=False))


# Clear the CUDA cache before starting
clear_cuda_cache()


# Function to generate high-quality images
def stablediffusion_generate_image(prompt, guidance_scale=15, num_inference_steps=500,
                                   seed=None, height=4320, width=7680,
                                   model_name='CompVis/stable-diffusion-v1-4',
                                   report_memory=False, batch_size=1):
    """
    Generate high-quality images using Stable Diffusion pipeline.
    Optimized for high-end GPUs (e.g., NVIDIA H100).
    """
    # Set the random seed for reproducibility
    seed = seed or 1024
    torch.manual_seed(seed)

    try:
        # Load the Stable Diffusion model with mixed precision for efficiency
        model = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda')

        # Enable memory optimization with attention slicing for larger resolutions
        model.enable_attention_slicing()

        # Report GPU memory usage if required
        if report_memory:
            report_cuda_memory()

        # Generate the image(s)
        images = []
        for batch in range(batch_size):  # Support for batch generation
            image = model(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images[0]  # Only one image generated per batch here
            images.append(image)

        # Free GPU resources
        model.to('cpu')
        del model
        clear_cuda_cache()

        return images[0] if batch_size == 1 else images
    except Exception as e:
        logging.error(f"Error during image generation: {e}")
        return None


if __name__ == "__main__":
    # Create the output folder if it doesn't already exist
    os.makedirs(output_folder, exist_ok=True)

    # Define the prompt for high-quality image generation
    prompt = (
        "An ultra-detailed 8K photorealistic painting of a Tulip field that stretches to infinity, "
        "with butterflies flying among the flowers under a vivid blue sky full of soft white clouds, glowing in majestic sunlight. "
        "Hyper-realistic, cinematic lighting, vibrant colors, surreal painterly style."
    )

    # Settings for max quality
    height = 2160  # 4K height
    width = 3840  # 4K width
    guidance_scale = 15  # Strong guidance for the model to follow the prompt
    num_inference_steps = 500  # High number of inference steps for image sharpness
    seed = 42  # Seed for reproducible results
    model_name = 'stabilityai/stable-diffusion-2-1'  # Latest high-quality model
    batch_size = 1  # Can increase for parallel image generations

    # Log task start
    logging.info(f"Generating 8K image for prompt: \"{prompt}\" using H100 GPU.")

    # Generate the image
    image = stablediffusion_generate_image(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        model_name=model_name,
        report_memory=True,
        batch_size=batch_size
    )

    # Save the generated image
    if image:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(output_folder, f"{prompt[:50].replace(' ', '_')}_{timestamp}.png")
        image.save(output_path)
        logging.info(f"Image saved at: {output_path}")
    else:
        logging.error("Image generation failed. No image saved.")