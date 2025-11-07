import torch
import os
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import pandas as pd
import argparse

def generate_images(num_images_per_prompt=20, output_dir="data/generated/pleural_effusion"):
    """
    Generate chest X-ray images with pleural effusion using multiple prompts
    
    Args:
        num_images_per_prompt: Number of images to generate for each prompt
        output_dir: Directory to save generated images
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Negative prompt to exclude unwanted elements
    negative_prompt = (
        "text, labels, watermark, logo, hospital name, patient info, annotation, "
        "numbers, letters, arrows, grids, borders, frames, CT, MRI, ultrasound, "
        "PET, color, artifacts, distortion, unrealistic anatomy"
    )
    
    # Define all prompts
    prompts = [
        "high-resolution posterior-anterior chest X-ray showing right pleural effusion, realistic grayscale tone, clear rib detail, no text or labels",
        "diagnostic-quality chest X-ray showing left pleural effusion, balanced exposure, full chest view, no artifacts or writing",
        "posterior-anterior chest X-ray showing bilateral pleural effusions, uniform lighting, natural contrast, no labels",
        "realistic grayscale chest radiograph showing right pleural effusion, proper exposure and lung markings, no text",
        "chest X-ray showing left pleural effusion, clean diagnostic style, soft lighting, clear diaphragm contour",
        "frontal chest X-ray showing right pleural effusion, accurate rib spacing, smooth gradient shading, no watermark",
        "high-clarity chest radiograph showing bilateral pleural effusions, full thoracic frame, natural grayscale balance",
        "diagnostic chest X-ray showing pleural effusion, clear heart border and diaphragm, no text or arrows",
        "chest X-ray showing right pleural effusion, crisp lung edges, no artifacts, true grayscale radiographic look",
        "realistic chest X-ray showing left pleural effusion, correct exposure, even lighting, no letters or symbols",
        "high-detail chest radiograph showing pleural effusion, balanced tones, visible ribs and vertebral shadows",
        "posterior-anterior chest X-ray showing right pleural effusion, accurate anatomy, clean background, no labels",
        "frontal chest X-ray showing bilateral pleural effusions, diagnostic contrast, no patient info or watermark",
        "chest radiograph showing left pleural effusion, realistic grayscale and texture, no annotations",
        "high-resolution chest X-ray showing right pleural effusion, balanced exposure, no overlay or gridlines",
        "diagnostic-style chest X-ray showing pleural effusion, clear mediastinal outline, smooth lighting, no text",
        "chest X-ray showing bilateral pleural effusions, realistic tonal range, sharp anatomical borders",
        "posterior-anterior chest X-ray showing left pleural effusion, soft grayscale lighting, no frame or numbers",
        "chest radiograph showing pleural effusion, even exposure, full chest composition, realistic X-ray tone",
        "high-detail chest X-ray showing right pleural effusion, crisp ribs, soft contrast, no medical labels",
        "diagnostic chest X-ray showing left pleural effusion, clear costophrenic angle, no text or markings",
        "posterior-anterior chest radiograph showing bilateral pleural effusions, realistic radiographic depth, no artifacts",
        "realistic chest X-ray showing pleural effusion, uniform lighting, accurate grayscale range, no symbols",
        "chest X-ray showing right pleural effusion, clean diaphragm line, even tone, no numbers or letters",
        "diagnostic-quality frontal chest X-ray showing left pleural effusion, true radiograph texture, no overlays",
        "high-resolution chest X-ray showing bilateral pleural effusions, clear lung detail, smooth grayscale balance",
        "realistic PA chest X-ray showing pleural effusion, no text, no watermark, even diagnostic lighting",
        "chest X-ray showing pleural effusion, natural grayscale tones, sharp anatomy, no artifacts or grids",
        "diagnostic chest radiograph showing right pleural effusion, balanced contrast and clarity, no border",
        "posterior-anterior chest X-ray showing left pleural effusion, uniform tone mapping, no text or symbols"
    ]
    
    print(f"Loading Stable Diffusion model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "danyalmalik/stable-diffusion-chest-xray",
        torch_dtype=torch.float16
    )
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    
    # Store mapping data
    mapping_data = []
    
    total_images = len(prompts) * num_images_per_prompt
    print(f"\nGenerating {total_images} images ({num_images_per_prompt} per prompt)...")
    print(f"Output directory: {output_dir}\n")
    
    image_counter = 0
    
    # Generate images for each prompt
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n{'='*80}")
        print(f"Prompt {prompt_idx + 1}/{len(prompts)}")
        print(f"{'='*80}")
        print(f"{prompt[:100]}...")
        print()
        
        for img_idx in tqdm(range(num_images_per_prompt), desc=f"Generating images"):
            image_counter += 1
            
            # Vary parameters for diversity
            guidance_scale = 7.0 + (img_idx % 4) * 0.5  # 7.0, 7.5, 8.0, 8.5
            num_steps = 30 + (img_idx % 4) * 5  # 30, 35, 40, 45
            seed = 42 + prompt_idx * 1000 + img_idx  # Unique seed for each image
            
            generator = torch.Generator(device="cuda").manual_seed(seed)
            
            # Generate image
            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            
            # Save image with structured filename
            filename = f"pleural_effusion_p{prompt_idx+1:02d}_img{img_idx+1:03d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            
            # Add to mapping data
            mapping_data.append({
                'image_id': image_counter,
                'filename': filename,
                'prompt_number': prompt_idx + 1,
                'image_number': img_idx + 1,
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'guidance_scale': guidance_scale,
                'num_inference_steps': num_steps,
                'seed': seed
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(mapping_data)
    csv_path = os.path.join(output_dir, 'image_mapping.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"Generation Complete!")
    print(f"{'='*80}")
    print(f"Total images generated: {image_counter}")
    print(f"Images saved to: {output_dir}")
    print(f"Mapping CSV saved to: {csv_path}")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst few rows of mapping:")
    print(df.head())
    print(f"\nMapping columns: {list(df.columns)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate chest X-ray images with pleural effusion using multiple prompts'
    )
    parser.add_argument(
        '--num_images_per_prompt',
        type=int,
        default=20,
        help='Number of images to generate for each prompt (default: 20)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/generated/pleural_effusion',
        help='Directory to save generated images (default: data/generated/pleural_effusion)'
    )
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Number of prompts: 30")
    print(f"  Images per prompt: {args.num_images_per_prompt}")
    print(f"  Total images to generate: {30 * args.num_images_per_prompt}")
    print(f"  Output directory: {args.output_dir}")
    print(f"\n")
    
    generate_images(
        num_images_per_prompt=args.num_images_per_prompt,
        output_dir=args.output_dir
    )