import torch
import os
from diffusers import StableDiffusionPipeline

# Create output directory if it doesn't exist
os.makedirs("data/generated", exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
        "danyalmalik/stable-diffusion-chest-xray",
        torch_dtype=torch.float16
)

pipe.to("cuda")
pipe.enable_attention_slicing()

prompt = "a realistic chest x-ray showing a patient with Pneumonia"

# Generate 50 images
for i in range(50):
    print(f"Generating image {i+1}/50...")
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save(f"data/generated/chest_x_ray_{i+1:03d}.png")

print("All 50 images generated successfully!")
