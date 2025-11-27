"""
Diffusion-based image generation for chest X-rays.
"""
import os
import logging
import torch
import pandas as pd
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
from codecarbon import EmissionsTracker
from datetime import datetime

from config import (
    BASE_DIFFUSION_MODEL,
    GENERATION_PROMPT,
    GENERATION_NEGATIVE_PROMPT,
    GENERATION_NUM_INFERENCE_STEPS,
    GENERATION_GUIDANCE_SCALE,
    GENERATION_IMAGE_HEIGHT,
    GENERATION_IMAGE_WIDTH
)

logger = logging.getLogger(__name__)


def load_diffusion_model(base_model_path=BASE_DIFFUSION_MODEL, device=None):
    """
    Load Stable Diffusion pipeline for image generation

    Args:
        base_model_path (str): Path or HuggingFace ID of base model
        device: Device to load model on (if None, will auto-detect)

    Returns:
        StableDiffusionPipeline: Configured pipeline ready for inference
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Loading diffusion model: {base_model_path}")

    # Load base pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        safety_checker=None,  # Disable for medical images
    )

    # Use faster scheduler for inference
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    # Move to device and enable optimizations
    pipeline.to(device)
    pipeline.enable_attention_slicing()

    # Enable xformers if available
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        logger.info("xformers memory efficient attention enabled")
    except:
        logger.info("xformers not available, using standard attention")

    logger.info("Diffusion model loaded successfully")
    return pipeline


def load_diffusion_model_with_lora(lora_weights_path, base_model_path=BASE_DIFFUSION_MODEL, device=None):
    """
    Load Stable Diffusion pipeline with LoRA weights

    Args:
        lora_weights_path (str): Path to LoRA weights directory
        base_model_path (str): Path or HuggingFace ID of base model
        device: Device to load model on (if None, will auto-detect)

    Returns:
        StableDiffusionPipeline: Configured pipeline with LoRA weights
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Loading base model: {base_model_path}")

    # Load base pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        safety_checker=None,
    )

    logger.info(f"Loading LoRA weights: {lora_weights_path}")

    # Load metadata if available
    metadata_path = os.path.join(lora_weights_path, 'metadata.json')
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            logger.info("LoRA configuration:")
            logger.info(f"  Rank: {metadata.get('lora_rank', 'unknown')}")
            logger.info(f"  Alpha: {metadata.get('lora_alpha', 'unknown')}")
            logger.info(f"  Training samples: {metadata.get('training_samples', 'unknown')}")
            logger.info(f"  Epochs: {metadata.get('epochs', 'unknown')}")

    # Load LoRA weights into UNet
    pipeline.unet = PeftModel.from_pretrained(
        pipeline.unet,
        lora_weights_path
    )

    # Use faster scheduler for inference
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    # Move to device and enable optimizations
    pipeline.to(device)
    pipeline.enable_attention_slicing()

    # Enable xformers if available
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        logger.info("xformers memory efficient attention enabled")
    except:
        logger.info("xformers not available, using standard attention")

    logger.info("Diffusion model with LoRA weights loaded successfully")
    return pipeline


def generate_synthetic_images(
    num_images,
    output_dir,
    pipeline=None,
    prompt=GENERATION_PROMPT,
    negative_prompt=GENERATION_NEGATIVE_PROMPT,
    num_inference_steps=GENERATION_NUM_INFERENCE_STEPS,
    guidance_scale=GENERATION_GUIDANCE_SCALE,
    height=GENERATION_IMAGE_HEIGHT,
    width=GENERATION_IMAGE_WIDTH,
    seed=None,
    log_dir='./log',
    device=None
):
    """
    Generate synthetic chest X-ray images using diffusion model

    Args:
        num_images (int): Number of images to generate
        output_dir (str): Directory to save generated images
        pipeline: Pre-loaded StableDiffusionPipeline (if None, will load)
        prompt (str): Text prompt for generation
        negative_prompt (str): Negative prompt (what to avoid)
        num_inference_steps (int): Number of denoising steps
        guidance_scale (float): Classifier-free guidance scale
        height (int): Image height
        width (int): Image width
        seed (int): Random seed (None for random)
        log_dir (str): Directory to save emissions data
        device: Device to run on

    Returns:
        tuple: (list of image paths, CO2 emissions in kg)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Load pipeline if not provided
    pipeline_provided = pipeline is not None
    if not pipeline_provided:
        pipeline = load_diffusion_model(device=device)

    # Initialize emission tracker
    tracker = EmissionsTracker(
        project_name="Diffusion_Image_Generation",
        output_dir=log_dir,
        output_file=f'diffusion_generation_emissions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        log_level='warning'
    )

    logger.info("="*70)
    logger.info("Starting CO2 emission tracking for image generation...")
    logger.info("="*70)
    tracker.start()

    # Set seed if provided
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    else:
        generator = None

    logger.info(f"Generating {num_images} synthetic images...")
    logger.info(f"Prompt: '{prompt}'")
    if negative_prompt:
        logger.info(f"Negative prompt: '{negative_prompt}'")
    logger.info(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")

    generated_paths = []

    for i in range(num_images):
        logger.info(f"Generating image {i+1}/{num_images}...")

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
        output_path = os.path.join(output_dir, f'synthetic_{i+1:04d}.png')
        image.save(output_path)
        generated_paths.append(output_path)
        logger.info(f"Saved: {output_path}")

    # Stop emission tracking
    generation_emissions = tracker.stop()
    logger.info("="*70)
    logger.info(f"Image Generation CO2 Emissions: {generation_emissions:.6f} kg")
    logger.info("="*70)

    logger.info(f"All {num_images} images generated successfully!")
    logger.info(f"Images saved to: {output_dir}")

    # Clean up pipeline if we loaded it
    if not pipeline_provided:
        del pipeline
        torch.cuda.empty_cache()

    return generated_paths, generation_emissions


def create_synthetic_csv(generated_paths, target_feature, target_value, output_csv, data_root=None):
    """
    Create a CSV file for generated images with target labels

    Args:
        generated_paths (list): List of paths to generated images
        target_feature (str): Name of the target feature
        target_value (float): Value for the target feature (0.0 or 1.0)
        output_csv (str): Path to save the CSV file
        data_root (str): Root directory to make paths relative to (if None, uses absolute paths)

    Returns:
        pd.DataFrame: DataFrame with generated image paths and labels
    """
    # Make paths relative to data_root if provided
    if data_root is not None:
        # Normalize data_root path
        data_root = os.path.normpath(data_root)

        # Make each path relative to data_root
        relative_paths = []
        for path in generated_paths:
            norm_path = os.path.normpath(path)
            # Remove data_root prefix if present
            if norm_path.startswith(data_root + os.sep):
                rel_path = norm_path[len(data_root) + len(os.sep):]
            elif norm_path.startswith(data_root):
                rel_path = norm_path[len(data_root):].lstrip(os.sep)
            else:
                rel_path = norm_path
            relative_paths.append(rel_path)

        paths_to_save = relative_paths
    else:
        paths_to_save = generated_paths

    # Create DataFrame with paths and label
    data = {
        'Path': paths_to_save,
        target_feature: [target_value] * len(paths_to_save)
    }

    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    logger.info(f"Created synthetic data CSV: {output_csv}")
    logger.info(f"  {len(df)} images with {target_feature}={target_value}")

    return df


def augment_training_data(original_train_csv, synthetic_csv, output_csv):
    """
    Combine original training data with synthetic data

    Args:
        original_train_csv (str): Path to original training CSV
        synthetic_csv (str): Path to synthetic data CSV
        output_csv (str): Path to save augmented CSV

    Returns:
        pd.DataFrame: Combined DataFrame
    """
    # Load both datasets
    original_df = pd.read_csv(original_train_csv)
    synthetic_df = pd.read_csv(synthetic_csv)

    # Combine
    augmented_df = pd.concat([original_df, synthetic_df], ignore_index=True)

    # Save
    augmented_df.to_csv(output_csv, index=False)

    logger.info(f"Created augmented training dataset: {output_csv}")
    logger.info(f"  Original samples: {len(original_df)}")
    logger.info(f"  Synthetic samples: {len(synthetic_df)}")
    logger.info(f"  Total samples: {len(augmented_df)}")

    return augmented_df
