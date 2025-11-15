"""
Fine-tune Stable Diffusion for Chest X-Ray Generation with LoRA (Pleural Effusion)

This script fine-tunes a diffusion model on CheXpert pleural effusion data using LoRA.
LoRA provides 60-70% faster training with comparable quality to full fine-tuning.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
from torchvision import transforms
from accelerate import Accelerator
from codecarbon import EmissionsTracker
from datetime import datetime
import sys
import argparse


class LogFileWriter:
    """Redirect stdout/stderr to both console and log file"""
    def __init__(self, log_file, mode='a'):
        self.terminal = sys.stdout
        self.log = open(log_file, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def setup_logging(log_dir, script_name):
    """Setup logging to file"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{script_name}_{timestamp}.log')

    # Redirect stdout and stderr
    log_writer = LogFileWriter(log_file, 'w')
    sys.stdout = log_writer
    sys.stderr = log_writer

    return log_file, log_writer


class ChestXRayDataset(Dataset):
    """Dataset for chest X-ray diffusion fine-tuning"""
    
    def __init__(self, csv_file, image_root, size=512, max_samples=None):
        """
        Args:
            csv_file: Path to CSV with image annotations
            image_root: Root directory containing images
            size: Image size (512 for Stable Diffusion)
            max_samples: Maximum number of samples to use (None for all)
        """
        self.df = pd.read_csv(csv_file)
        
        # Limit samples if specified
        if max_samples is not None and len(self.df) > max_samples:
            self.df = self.df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
        self.image_root = image_root
        self.size = size
        
        print(f"Loaded {len(self.df)} training samples")
        
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_root, self.df.iloc[idx]['Path'])
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = torch.zeros(3, self.size, self.size)
        
        # Caption for pleural effusion positive cases
        caption = "a chest x-ray showing pleural effusion"
        
        return {
            'pixel_values': image,
            'caption': caption
        }


def collate_fn(examples):
    """Collate batch"""
    pixel_values = torch.stack([ex['pixel_values'] for ex in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    captions = [ex['caption'] for ex in examples]
    return {'pixel_values': pixel_values, 'captions': captions}


def train(
    csv_file,
    image_root,
    output_dir,
    base_model='runwayml/stable-diffusion-v1-5',
    num_epochs=5,
    batch_size=8,
    learning_rate=1e-4,
    gradient_accumulation_steps=4,
    mixed_precision='fp16',
    save_steps=500,
    image_size=512,
    max_samples=10000,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.1,
    log_dir='.'
):
    """
    Fine-tune diffusion model for chest X-ray generation using LoRA

    Args:
        csv_file: Path to diffusion_finetune.csv
        image_root: Root directory containing organized images
        output_dir: Directory to save fine-tuned model
        base_model: Base Stable Diffusion model
        num_epochs: Number of training epochs (default 5 for LoRA)
        batch_size: Training batch size (can be larger with LoRA)
        learning_rate: Learning rate (higher for LoRA, typically 1e-4)
        gradient_accumulation_steps: Gradient accumulation steps
        mixed_precision: 'fp16', 'bf16', or 'no'
        save_steps: Save checkpoint every N steps
        image_size: Input image size (512 recommended)
        max_samples: Maximum training samples (None for all)
        lora_rank: LoRA rank (8 or 16 recommended)
        lora_alpha: LoRA alpha (typically 2*rank)
        lora_dropout: LoRA dropout probability
        log_dir: Directory to save log files and emissions data
    """

    # Setup logging
    log_file, log_writer = setup_logging(log_dir, 'diffusion_lora_finetuning')
    print(f"Logging to: {log_file}")

    # Initialize emission tracker
    tracker = EmissionsTracker(
        project_name="Diffusion_LoRA_Finetuning",
        output_dir=log_dir,
        output_file=f'diffusion_finetuning_emissions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        log_level='warning'
    )

    print("\n" + "="*70)
    print("Starting CO2 emission tracking for diffusion fine-tuning...")
    print("="*70)
    tracker.start()

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    
    print(f"Device: {accelerator.device}")
    print(f"Mixed precision: {mixed_precision}")
    
    # Load model components using pipeline (more reliable)
    print(f"Loading base model: {base_model}")
    print("This may take a few minutes on first run (downloading model)...")
    
    # Load the full pipeline
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float32  # Load in FP32, let Accelerate handle FP16 casting
        )
        print(f"Model loaded successfully: {base_model}")
    except Exception as e:
        print(f"Error loading {base_model}: {e}")
        print("\nPlease check:")
        print("1. Internet connection")
        print("2. HuggingFace Hub access")
        print("3. Try: pip install --upgrade diffusers transformers")
        raise
    
    # Extract components
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
    unet = pipeline.unet
    
    # Use DDPM scheduler for training (better than the pipeline's default)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon"
    )
    
    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Configure LoRA for UNet
    print("\n" + "="*60)
    print("Configuring LoRA for UNet...")
    print("="*60)
    
    # Target all cross-attention and self-attention layers in the UNet
    # These are the key layers for adapting to new visual domains
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            "to_q",
            "to_k", 
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2"
        ],
        lora_dropout=lora_dropout,
    )
    
    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()  # Show how many parameters are being trained
    
    # Create dataset
    print(f"\nLoading dataset from {csv_file}")
    dataset = ChestXRayDataset(
        csv_file=csv_file,
        image_root=image_root,
        size=image_size,
        max_samples=max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Optimizer - only optimize LoRA parameters
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Prepare with accelerator
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # Calculate total training steps
    num_update_steps_per_epoch = len(dataloader)
    total_steps = num_epochs * num_update_steps_per_epoch
    
    # Training configuration
    print("\n" + "="*60)
    print("Training Configuration (LoRA)")
    print("="*60)
    print(f"  Base model: {base_model}")
    print(f"  Training samples: {len(dataset)}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Total training steps: {total_steps}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  LoRA alpha: {lora_alpha}")
    print(f"  LoRA dropout: {lora_dropout}")
    print(f"  Output: {output_dir}")
    print("="*60 + "\n")
    
    # Training loop
    global_step = 0
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        progress_bar = tqdm(dataloader, desc="Training")
        epoch_loss = 0.0
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(unet):
                # Encode images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch['pixel_values'].to(accelerator.device)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=latents.device
                ).long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                with torch.no_grad():
                    text_inputs = tokenizer(
                        batch['captions'],
                        padding='max_length',
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors='pt'
                    )
                    text_embeddings = text_encoder(
                        text_inputs.input_ids.to(accelerator.device)
                    )[0]
                
                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                
                # Calculate loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction='mean')
                
                # Backpropagation
                accelerator.backward(loss)
                
                optimizer.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                epoch_loss += loss.detach().item()
                progress_bar.set_postfix({
                    'loss': f'{loss.detach().item():.4f}',
                    'avg_loss': f'{epoch_loss / (step + 1):.4f}'
                })
                global_step += 1
                
                # Save checkpoint
                if global_step % save_steps == 0 and accelerator.is_main_process:
                    checkpoint_dir = os.path.join(output_dir, f'checkpoint-{global_step}')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    accelerator.wait_for_everyone()
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    
                    # Save only LoRA weights (much smaller!)
                    unwrapped_unet.save_pretrained(checkpoint_dir)
                    
                    # Save metadata
                    metadata = {
                        'base_model': base_model,
                        'lora_rank': lora_rank,
                        'lora_alpha': lora_alpha,
                        'global_step': global_step,
                        'epoch': epoch + 1
                    }
                    import json
                    with open(os.path.join(checkpoint_dir, 'metadata.json'), 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    print(f"\n✓ Checkpoint saved: {checkpoint_dir}")
        
        # End of epoch summary
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed - Average loss: {avg_epoch_loss:.4f}")
    
    # Save final model
    print("\n" + "="*60)
    print("Saving final LoRA model...")
    print("="*60)
    
    if accelerator.is_main_process:
        accelerator.wait_for_everyone()
        unwrapped_unet = accelerator.unwrap_model(unet)
        
        final_path = os.path.join(output_dir, 'final_lora')
        os.makedirs(final_path, exist_ok=True)
        
        # Save LoRA weights
        unwrapped_unet.save_pretrained(final_path)
        
        # Save metadata
        metadata = {
            'base_model': base_model,
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'training_samples': len(dataset),
            'epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'final_step': global_step
        }
        import json
        with open(os.path.join(final_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Final LoRA model saved: {final_path}")
        print(f"✓ Model size: ~{lora_rank * 2}MB (vs ~4GB for full model)")
        print("\nTo use this model for inference:")
        print(f"  1. Load base model: StableDiffusionPipeline.from_pretrained('{base_model}')")
        print(f"  2. Load LoRA weights: PeftModel.from_pretrained(unet, '{final_path}')")

    # Stop emission tracking
    finetuning_emissions = tracker.stop()
    print("\n" + "="*70)
    print(f"Fine-tuning CO2 Emissions: {finetuning_emissions:.6f} kg")
    print("="*70)

    # Save emissions to final metadata
    if accelerator.is_main_process:
        metadata_path = os.path.join(final_path, 'metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata['finetuning_co2_kg'] = float(finetuning_emissions)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    # Close log writer
    log_writer.close()


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune Stable Diffusion with LoRA for chest X-ray generation'
    )
    
    parser.add_argument('--csv_file', type=str, required=True,
                       help='Path to diffusion_finetune.csv')
    parser.add_argument('--image_root', type=str, required=True,
                       help='Root directory containing organized images')
    parser.add_argument('--output_dir', type=str, default='finetuned_lora',
                       help='Output directory for fine-tuned LoRA model')
    parser.add_argument('--base_model', type=str, 
                       default='runwayml/stable-diffusion-v1-5',
                       help='Base Stable Diffusion model')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs (default: 5 for LoRA)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size (can be larger with LoRA)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4 for LoRA)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                       choices=['no', 'fp16', 'bf16'],
                       help='Mixed precision training')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Save checkpoint every N steps')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Input image size')
    parser.add_argument('--max_samples', type=int, default=10000,
                       help='Maximum training samples (None for all)')
    parser.add_argument('--lora_rank', type=int, default=8,
                       help='LoRA rank (8 or 16 recommended, higher = more capacity)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha (typically 2*rank)')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout probability')
    parser.add_argument('--log_dir', type=str, default='.',
                       help='Directory to save log files and emissions data (default: current directory)')

    args = parser.parse_args()
    
    # Convert 0 to None for max_samples
    max_samples = None if args.max_samples == 0 else args.max_samples
    
    train(
        csv_file=args.csv_file,
        image_root=args.image_root,
        output_dir=args.output_dir,
        base_model=args.base_model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        save_steps=args.save_steps,
        image_size=args.image_size,
        max_samples=max_samples,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        log_dir=args.log_dir
    )


if __name__ == '__main__':
    main()
