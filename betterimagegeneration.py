from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion v2 model from Hugging Face
model_id = "stabilityai/stable-diffusion-2-1-base"  # Use the base version for low VRAM
# Alternatively: "stabilityai/stable-diffusion-2-1" for the larger version (needs ~10GB VRAM)

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Mixed precision to save VRAM
    variant='fp16'            # Use fp16 version for efficiency
)  # Use "cpu" if no GPU is available

# Enable VRAM optimizations (useful for low-VRAM GPUs)
pipe.enable_attention_slicing()

# Define the text prompt
prompt = "A futuristic cityscape with flying cars during sunset, highly detailed, ultra-realistic, 4K"

# Generate the image
print("Generating image...")
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

# Save the image
output_path = "generated_image.png"
image.save(output_path)
print(f"Image saved to {output_path}")
image.show()
