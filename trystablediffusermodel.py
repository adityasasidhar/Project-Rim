from diffusers import StableDiffusionPipeline
import os
import torch
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"GPU is active: {torch.cuda.is_available()}")

output_folder = "AI generate images"


def clear_cuda_cache():
    """Frees the GPU memory cache."""
    torch.cuda.empty_cache()
    logging.info("CUDA cache cleared.")


def report_cuda_memory():
    """Logs CUDA memory stats."""
    logging.info("CUDA Memory Summary:")
    logging.info(torch.cuda.memory_summary(device=None, abbreviated=False))

clear_cuda_cache()

# Function to generate high-quality images
def stablediffusion_generate_image(prompt, guidance_scale=10, num_inference_steps=150,
                                   seed=None, height=2160, width=3840, model_name='CompVis/stable-diffusion-v1-4',
                                   report_memory=False):
    # Set the seed
    seed = 42

    try:
        # Load the Stable Diffusion model pipeline
        model = StableDiffusionPipeline.from_pretrained(model_name).to('cuda')
        if report_memory:
            report_cuda_memory()

        # Generate the image
        image = model(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]

        # Free resources
        model.to('cpu')
        del model
        clear_cuda_cache()

        return image
    except Exception as e:
        logging.error(f"Error during image generation: {e}")
        return None


if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)

    # Prompt & Settings
    prompt = (
        "A stunning ultra-detailed 8K photorealistic digital painting of a scenic tulip field extending to the horizon,"
        "with butterflies, a vivid clear blue sky, soft white scattered clouds, and warm afternoon sunlight. "
        "Hyper-realistic, cinematic soft lighting, vibrant and surreal colors with painterly texture."
    )
    height = 2160
    width = 3840
    guidance_scale = 15
    num_inference_steps = 500
    seed = 42

    # Log the task start
    logging.info(f"Generating image for prompt: \"{prompt}\"")

    # Generate the image
    image = stablediffusion_generate_image(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed
    )

    # Handle image save
    # Handle image save
    if image:
        output_path = os.path.join(output_folder, "poppy.png")  # Save as 'poppy.png'
        image.save(output_path)
        logging.info(f"Image saved at: {output_path}")
    else:
        logging.error("Image generation failed. No image was saved.")
