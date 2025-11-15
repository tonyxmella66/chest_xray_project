"""
Generate Chest X-Rays using LoRA Fine-tuned Stable Diffusion

This script loads a LoRA fine-tuned model and generates chest X-ray images.
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
import argparse
import os
import json


def load_lora_model(base_model_path, lora_weights_path, device='cuda'):
    """
    Load Stable Diffusion pipeline with LoRA weights
    
    Args:
        base_model_path: Path or HuggingFace ID of base model
        lora_weights_path: Path to saved LoRA weights directory
        device: Device to load model on
    
    Returns:
        Configured pipeline ready for inference
    """
    print(f"Loading base model: {base_model_path}")
    
    # Load base pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        safety_checker=None,  # Disable for medical images
    )
    
    print(f"Loading LoRA weights: {lora_weights_path}")
    
    # Load metadata if available
    metadata_path = os.path.join(lora_weights_path, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            print(f"LoRA configuration:")
            print(f"  Rank: {metadata.get('lora_rank', 'unknown')}")
            print(f"  Alpha: {metadata.get('lora_alpha', 'unknown')}")
            print(f"  Training samples: {metadata.get('training_samples', 'unknown')}")
            print(f"  Epochs: {metadata.get('epochs', 'unknown')}")
    
    # Load LoRA weights into UNet
    pipeline.unet = PeftModel.from_pretrained(
        pipeline.unet,
        lora_weights_path
    )
    
    # Use faster scheduler for inference
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    
    # Enable optimizations
    pipeline.to(device)
    pipeline.enable_attention_slicing()
    
    # Enable xformers if available
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        print("✓ xformers memory efficient attention enabled")
    except:
        print("⚠ xformers not available, using standard attention")
    
    print("✓ Model loaded successfully!")
    return pipeline


def generate_images(
    pipeline,
    prompt,
    negative_prompt=None,
    num_images=4,
    num_inference_steps=30,
    guidance_scale=7.5,
    height=512,
    width=512,
    output_dir='generated_xrays',
    seed=None
):
    """
    Generate images using the loaded pipeline
    
    Args:
        pipeline: Loaded StableDiffusionPipeline with LoRA
        prompt: Text prompt for generation
        negative_prompt: Negative prompt (what to avoid in generation)
        num_images: Number of images to generate
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        height: Image height
        width: Image width
        output_dir: Directory to save generated images
        seed: Random seed (None for random)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed if provided
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    else:
        generator = None
    
    print(f"\nGenerating {num_images} images...")
    print(f"Prompt: '{prompt}'")
    if negative_prompt:
        print(f"Negative prompt: '{negative_prompt}'")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    
    for i in range(num_images):
        print(f"Generating image {i+1}/{num_images}...")
        
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        # Save image
        output_path = os.path.join(output_dir, f'generated_xray_{i+1:03d}.png')
        image.save(output_path)
        print(f"✓ Saved: {output_path}")
    
    print(f"\n✓ All {num_images} images generated successfully!")
    print(f"✓ Images saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate chest X-rays using LoRA fine-tuned Stable Diffusion'
    )
    
    parser.add_argument('--lora_weights', type=str, required=True,
                       help='Path to LoRA weights directory (e.g., finetuned_lora/final_lora)')
    parser.add_argument('--base_model', type=str,
                       default='runwayml/stable-diffusion-v1-5',
                       help='Base model (must match training base model)')
    parser.add_argument('--prompt', type=str,
                       default='a chest x-ray showing pleural effusion',
                       help='Text prompt for generation')
    parser.add_argument('--negative_prompt', type=str, default=None,
                       help='Negative prompt (what to avoid, e.g., "blurry, low quality, distorted")')
    parser.add_argument('--num_images', type=int, default=4,
                       help='Number of images to generate')
    parser.add_argument('--steps', type=int, default=30,
                       help='Number of inference steps (20-50 recommended)')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                       help='Guidance scale (7.0-9.0 recommended)')
    parser.add_argument('--height', type=int, default=512,
                       help='Image height')
    parser.add_argument('--width', type=int, default=512,
                       help='Image width')
    parser.add_argument('--output_dir', type=str, default='generated_xrays',
                       help='Output directory for generated images')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (None for random)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Load model
    pipeline = load_lora_model(
        base_model_path=args.base_model,
        lora_weights_path=args.lora_weights,
        device=args.device
    )
    
    # Generate images
    generate_images(
        pipeline=pipeline,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_images=args.num_images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == '__main__':
    main()