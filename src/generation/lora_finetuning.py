"""
LoRA fine-tuning for diffusion models.
"""
import os
import logging
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from accelerate import Accelerator
from codecarbon import EmissionsTracker
from datetime import datetime

from config import (
    BASE_DIFFUSION_MODEL,
    LORA_FINETUNE_CAPTION,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_NUM_EPOCHS,
    LORA_BATCH_SIZE,
    LORA_LEARNING_RATE,
    LORA_GRADIENT_ACCUMULATION_STEPS,
    LORA_MIXED_PRECISION,
    LORA_SAVE_STEPS,
    GENERATION_IMAGE_HEIGHT,
    GENERATION_IMAGE_WIDTH
)

logger = logging.getLogger(__name__)


class ChestXRayFinetuneDataset(Dataset):
    """Dataset for chest X-ray diffusion fine-tuning"""

    def __init__(self, csv_file, image_root, size=512, max_samples=None):
        """
        Args:
            csv_file (str): Path to CSV with image annotations
            image_root (str): Root directory containing images
            size (int): Image size (512 for Stable Diffusion)
            max_samples (int): Maximum number of samples to use (None for all)
        """
        self.df = pd.read_csv(csv_file)

        # Limit samples if specified
        if max_samples is not None and len(self.df) > max_samples:
            self.df = self.df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        self.image_root = image_root
        self.size = size

        logger.info(f"Loaded {len(self.df)} fine-tuning samples")

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
            logger.error(f"Error loading {img_path}: {e}")
            image = torch.zeros(3, self.size, self.size)

        # Caption for pleural effusion positive cases
        caption = LORA_FINETUNE_CAPTION

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


def finetune_diffusion_lora(
    finetune_csv,
    image_root,
    output_dir,
    log_dir='./log',
    base_model=BASE_DIFFUSION_MODEL,
    num_epochs=LORA_NUM_EPOCHS,
    batch_size=LORA_BATCH_SIZE,
    learning_rate=LORA_LEARNING_RATE,
    gradient_accumulation_steps=LORA_GRADIENT_ACCUMULATION_STEPS,
    mixed_precision=LORA_MIXED_PRECISION,
    lora_rank=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    image_size=GENERATION_IMAGE_HEIGHT
):
    """
    Fine-tune diffusion model with LoRA

    Args:
        finetune_csv (str): Path to CSV with fine-tuning images
        image_root (str): Root directory containing images
        output_dir (str): Directory to save fine-tuned model
        log_dir (str): Directory to save logs and emissions
        base_model (str): Base Stable Diffusion model
        num_epochs (int): Number of training epochs
        batch_size (int): Training batch size
        learning_rate (float): Learning rate
        gradient_accumulation_steps (int): Gradient accumulation steps
        mixed_precision (str): Mixed precision mode
        lora_rank (int): LoRA rank
        lora_alpha (int): LoRA alpha
        lora_dropout (float): LoRA dropout
        image_size (int): Input image size

    Returns:
        tuple: (output_path, CO2 emissions in kg)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup logger
    from utils.logging import setup_logger
    finetune_logger = setup_logger(
        f'lora_finetuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        log_dir=log_dir
    )

    # Initialize emission tracker
    tracker = EmissionsTracker(
        project_name="Diffusion_LoRA_Finetuning",
        output_dir=log_dir,
        output_file=f'diffusion_finetuning_emissions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        log_level='warning'
    )

    finetune_logger.info("="*70)
    finetune_logger.info("Starting CO2 emission tracking for diffusion fine-tuning...")
    finetune_logger.info("="*70)
    tracker.start()

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    finetune_logger.info(f"Device: {accelerator.device}")
    finetune_logger.info(f"Mixed precision: {mixed_precision}")

    # Load model components
    finetune_logger.info(f"Loading base model: {base_model}")

    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float32  # Load in FP32, let Accelerate handle FP16
    )

    # Extract components
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
    unet = pipeline.unet

    # Use DDPM scheduler for training
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
    finetune_logger.info("="*60)
    finetune_logger.info("Configuring LoRA for UNet...")
    finetune_logger.info("="*60)

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

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    finetune_logger.info(f"Trainable params: {trainable_params:,} || Total params: {total_params:,} || Trainable%: {100 * trainable_params / total_params:.2f}")

    # Create dataset
    finetune_logger.info(f"Loading dataset from {finetune_csv}")
    dataset = ChestXRayFinetuneDataset(
        csv_file=finetune_csv,
        image_root=image_root,
        size=image_size,
        max_samples=None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Optimizer
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

    # Training configuration
    finetune_logger.info("="*60)
    finetune_logger.info("Training Configuration (LoRA)")
    finetune_logger.info("="*60)
    finetune_logger.info(f"  Base model: {base_model}")
    finetune_logger.info(f"  Training samples: {len(dataset)}")
    finetune_logger.info(f"  Epochs: {num_epochs}")
    finetune_logger.info(f"  Batch size: {batch_size}")
    finetune_logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
    finetune_logger.info(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    finetune_logger.info(f"  Learning rate: {learning_rate}")
    finetune_logger.info(f"  LoRA rank: {lora_rank}")
    finetune_logger.info(f"  LoRA alpha: {lora_alpha}")
    finetune_logger.info(f"  Output: {output_dir}")
    finetune_logger.info("="*60)

    # Training loop
    global_step = 0

    for epoch in range(num_epochs):
        finetune_logger.info(f"Epoch {epoch + 1}/{num_epochs}")
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

        # End of epoch summary
        avg_epoch_loss = epoch_loss / len(dataloader)
        finetune_logger.info(f"Epoch {epoch + 1} completed - Average loss: {avg_epoch_loss:.4f}")

    # Save final model
    finetune_logger.info("="*60)
    finetune_logger.info("Saving final LoRA model...")
    finetune_logger.info("="*60)

    if accelerator.is_main_process:
        accelerator.wait_for_everyone()
        unwrapped_unet = accelerator.unwrap_model(unet)

        final_path = os.path.join(output_dir, 'lora_weights')
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

        finetune_logger.info(f"Final LoRA model saved: {final_path}")
    else:
        final_path = os.path.join(output_dir, 'lora_weights')

    # Stop emission tracking
    finetuning_emissions = tracker.stop()
    finetune_logger.info("="*70)
    finetune_logger.info(f"Fine-tuning CO2 Emissions: {finetuning_emissions:.6f} kg")
    finetune_logger.info("="*70)

    # Save emissions to metadata
    if accelerator.is_main_process:
        metadata_path = os.path.join(final_path, 'metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata['finetuning_co2_kg'] = float(finetuning_emissions)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    finetune_logger.info("="*60)
    finetune_logger.info("Training complete!")
    finetune_logger.info("="*60)

    # Clean up
    del pipeline, unet, vae, text_encoder, tokenizer
    torch.cuda.empty_cache()

    return final_path, finetuning_emissions
