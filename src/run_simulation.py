"""
Main simulation script for chest X-ray classification experiments.

This script runs multiple training experiments with different dataset sizes
and random seeds, tracking performance metrics and carbon emissions.
"""
import os
import json
import logging
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from config import (
    RANDOM_SEEDS,
    DATASET_SIZES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    CLASSIFIER_DEV_CSV,
    TEST_CSV,
    BASE_OUTPUT_DIR,
    TARGET_FEATURE,
    DATA_ROOT,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    IMAGE_GEN_RATIOS,
    GENERATION_OUTPUT_DIR
)
from training.trainer import train_model
from generation.diffusion import (
    load_diffusion_model,
    load_diffusion_model_with_lora,
    generate_synthetic_images,
    create_synthetic_csv,
    augment_training_data
)
from generation.lora_finetuning import finetune_diffusion_lora
from utils.logging import (
    setup_logger,
    create_summary_log,
    log_run_result,
    finalize_summary_log
)


def create_data_splits(data_size, random_seed):
    """
    Create train/val/test splits for a given data size and random seed

    Args:
        data_size (int): Total dataset size
        random_seed (int): Random seed for sampling

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Calculate split sizes
    train_size = int(data_size * TRAIN_RATIO)
    val_size = int(data_size * VAL_RATIO)
    test_size = int(data_size * TEST_RATIO)

    # Load the full development dataset
    classifier_dev_full = pd.read_csv(CLASSIFIER_DEV_CSV)

    # Sample training and validation data
    # Sample together then split to ensure no overlap
    dev_sample = classifier_dev_full.sample(
        n=train_size + val_size,
        random_state=random_seed
    )

    # Split the sample into training and validation
    train_df = dev_sample.iloc[:train_size]
    val_df = dev_sample.iloc[train_size:train_size + val_size]

    # Load test dataset and sample from it
    test_full = pd.read_csv(TEST_CSV)
    test_df = test_full.sample(n=test_size, random_state=random_seed)

    # Verify no overlap between train and val
    train_paths = set(train_df['Path'])
    val_paths = set(val_df['Path'])
    overlap = train_paths.intersection(val_paths)

    assert len(overlap) == 0, (
        f"Found {len(overlap)} overlapping paths between "
        f"training and validation sets!"
    )

    return train_df, val_df, test_df


def create_data_splits_with_finetune(data_size, random_seed, finetune_ratio=0.2):
    """
    Create train/val/test splits with a portion reserved for diffusion fine-tuning

    Args:
        data_size (int): Total dataset size
        random_seed (int): Random seed for sampling
        finetune_ratio (float): Ratio of data to reserve for fine-tuning (default: 0.2)

    Returns:
        tuple: (finetune_df, train_df, val_df, test_df)
    """
    # Calculate split sizes
    finetune_size = int(data_size * finetune_ratio)
    remaining_size = data_size - finetune_size
    train_size = int(remaining_size * TRAIN_RATIO)
    val_size = int(remaining_size * VAL_RATIO)
    test_size = int(remaining_size * TEST_RATIO)

    # Load the full development dataset
    classifier_dev_full = pd.read_csv(CLASSIFIER_DEV_CSV)

    # First, sample the fine-tuning set
    finetune_df = classifier_dev_full.sample(n=finetune_size, random_state=random_seed)

    # Remove fine-tuning samples from the pool
    remaining_pool = classifier_dev_full[~classifier_dev_full.index.isin(finetune_df.index)]

    # Sample training and validation data from remaining pool
    dev_sample = remaining_pool.sample(n=train_size + val_size, random_state=random_seed)

    # Split into training and validation
    train_df = dev_sample.iloc[:train_size]
    val_df = dev_sample.iloc[train_size:train_size + val_size]

    # Load test dataset and sample from it
    test_full = pd.read_csv(TEST_CSV)
    test_df = test_full.sample(n=test_size, random_state=random_seed)

    # Verify no overlap
    finetune_paths = set(finetune_df['Path'])
    train_paths = set(train_df['Path'])
    val_paths = set(val_df['Path'])

    assert len(finetune_paths.intersection(train_paths)) == 0, "Overlap between finetune and train!"
    assert len(finetune_paths.intersection(val_paths)) == 0, "Overlap between finetune and val!"
    assert len(train_paths.intersection(val_paths)) == 0, "Overlap between train and val!"

    return finetune_df, train_df, val_df, test_df


def save_data_splits(train_df, val_df, test_df, run_dir):
    """
    Save data splits to CSV files

    Args:
        train_df (DataFrame): Training data
        val_df (DataFrame): Validation data
        test_df (DataFrame): Test data
        run_dir (str): Directory to save the splits

    Returns:
        tuple: (train_csv_path, val_csv_path, test_csv_path)
    """
    os.makedirs(run_dir, exist_ok=True)

    train_csv = os.path.join(run_dir, "train.csv")
    val_csv = os.path.join(run_dir, "val.csv")
    test_csv = os.path.join(run_dir, "test.csv")

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return train_csv, val_csv, test_csv


def run_single_experiment(data_size, random_seed, run_number, total_runs, device, logger):
    """
    Run a single experiment with given data size and random seed

    Args:
        data_size (int): Dataset size
        random_seed (int): Random seed
        run_number (int): Current run number
        total_runs (int): Total number of runs
        device: Device to train on
        logger: Logger instance

    Returns:
        dict: Results dictionary or None if failed
    """
    logger.info("="*80)
    logger.info(f"RUN {run_number}/{total_runs}")
    logger.info(f"Data size: {data_size}, Random seed: {random_seed}")
    logger.info("="*80)

    # Create data splits
    train_df, val_df, test_df = create_data_splits(data_size, random_seed)

    logger.info("Dataset shapes:")
    logger.info(f"  Training set: {train_df.shape}")
    logger.info(f"  Validation set: {val_df.shape}")
    logger.info(f"  Test set: {test_df.shape}")

    # Create directory for this run
    run_dir = os.path.join(BASE_OUTPUT_DIR, f"size_{data_size}_seed_{random_seed}")

    # Save splits
    train_csv, val_csv, test_csv = save_data_splits(train_df, val_df, test_df, run_dir)
    logger.info(f"Saved splits to {run_dir}")

    # Create subdirectories
    output_dir = os.path.join(run_dir, "training_outputs")
    log_dir = os.path.join(run_dir, "logs")

    # Train the model
    logger.info("Starting model training...")
    results = train_model(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        output_dir=output_dir,
        log_dir=log_dir,
        target_feature=TARGET_FEATURE,
        data_root=DATA_ROOT,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=device
    )

    # Add run information to results
    results['data_size'] = data_size
    results['random_seed'] = random_seed
    results['run_number'] = run_number
    results['run_dir'] = run_dir

    # Save individual run results
    results_file = os.path.join(run_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Run {run_number}/{total_runs} completed successfully!")

    return results


def run_augmented_experiments(data_size, random_seed, run_number, total_runs,
                              train_csv, val_csv, test_csv, run_dir,
                              device, logger, diffusion_pipeline=None):
    """
    Run experiments with synthetic data augmentation

    Args:
        data_size (int): Dataset size
        random_seed (int): Random seed
        run_number (int): Current run number
        total_runs (int): Total number of runs
        train_csv (str): Path to original training CSV
        val_csv (str): Path to validation CSV
        test_csv (str): Path to test CSV
        run_dir (str): Directory for this run
        device: Device to train on
        logger: Logger instance
        diffusion_pipeline: Pre-loaded diffusion model (if None, will load)

    Returns:
        list: List of results dictionaries for each augmentation ratio
    """
    augmented_results = []

    # Load diffusion model once for all ratios (if not provided)
    pipeline_provided = diffusion_pipeline is not None
    if not pipeline_provided:
        logger.info("Loading diffusion model for image generation...")
        diffusion_pipeline = load_diffusion_model(device=device)

    # Read original training data to get the number of samples
    train_df = pd.read_csv(train_csv)
    num_train_samples = len(train_df)

    for gen_ratio in IMAGE_GEN_RATIOS:
        logger.info("="*80)
        logger.info(f"AUGMENTATION EXPERIMENT - Ratio: {gen_ratio}")
        logger.info("="*80)

        # Calculate number of synthetic images to generate
        num_synthetic = int(num_train_samples * gen_ratio)
        logger.info(f"Generating {num_synthetic} synthetic images ({gen_ratio}x training set size)")

        # Create directory for this augmentation ratio
        gen_dir = os.path.join(
            GENERATION_OUTPUT_DIR,
            f"seed_{random_seed}",
            f"size_{data_size}",
            f"ratio_{gen_ratio}"
        )
        os.makedirs(gen_dir, exist_ok=True)

        # Generate synthetic images
        try:
            generated_paths, gen_emissions = generate_synthetic_images(
                num_images=num_synthetic,
                output_dir=gen_dir,
                pipeline=diffusion_pipeline,
                seed=random_seed,
                log_dir=os.path.join(run_dir, "logs"),
                device=device
            )

            # Create CSV for synthetic data (all positive for pleural effusion)
            synthetic_csv = os.path.join(gen_dir, "synthetic_data.csv")
            create_synthetic_csv(
                generated_paths=generated_paths,
                target_feature=TARGET_FEATURE,
                target_value=1.0,  # Positive pleural effusion
                output_csv=synthetic_csv,
                data_root=DATA_ROOT
            )

            # Create augmented training set
            augmented_train_csv = os.path.join(gen_dir, "augmented_train.csv")
            augment_training_data(
                original_train_csv=train_csv,
                synthetic_csv=synthetic_csv,
                output_csv=augmented_train_csv
            )

            # Create output directory for this augmented experiment
            aug_output_dir = os.path.join(run_dir, f"augmented_ratio_{gen_ratio}")
            aug_log_dir = os.path.join(aug_output_dir, "logs")

            # Train model with augmented data
            logger.info(f"Training classifier with augmented data (ratio {gen_ratio})...")
            aug_results = train_model(
                train_csv=augmented_train_csv,
                val_csv=val_csv,
                test_csv=test_csv,
                output_dir=aug_output_dir,
                log_dir=aug_log_dir,
                target_feature=TARGET_FEATURE,
                data_root=DATA_ROOT,
                num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                device=device
            )

            # Add experiment metadata
            aug_results['data_size'] = data_size
            aug_results['random_seed'] = random_seed
            aug_results['run_number'] = run_number
            aug_results['gen_ratio'] = gen_ratio
            aug_results['num_synthetic'] = num_synthetic
            aug_results['generation_co2_kg'] = float(gen_emissions)
            aug_results['augmented'] = True
            aug_results['synthetic_dir'] = gen_dir

            # Save individual augmented results
            results_file = os.path.join(aug_output_dir, "results.json")
            with open(results_file, 'w') as f:
                json.dump(aug_results, f, indent=2)

            augmented_results.append(aug_results)

            logger.info(f"Augmented experiment (ratio {gen_ratio}) completed successfully!")

        except Exception as e:
            logger.error(f"Error in augmented experiment (ratio {gen_ratio}): {str(e)}")
            # Continue with next ratio
            continue

    # Clean up diffusion model if we loaded it
    if not pipeline_provided and diffusion_pipeline is not None:
        del diffusion_pipeline
        torch.cuda.empty_cache()

    return augmented_results


def run_finetuned_experiments(data_size, random_seed, run_number, total_runs,
                               run_dir, device, logger):
    """
    Run experiments with fine-tuned diffusion model

    Args:
        data_size (int): Dataset size
        random_seed (int): Random seed
        run_number (int): Current run number
        total_runs (int): Total number of runs
        run_dir (str): Directory for this run
        device: Device to train on
        logger: Logger instance

    Returns:
        list: List of results dictionaries for each augmentation ratio
    """
    finetuned_results = []

    logger.info("")
    logger.info("="*80)
    logger.info("FINE-TUNED DIFFUSION EXPERIMENTS")
    logger.info("="*80)

    # Create data splits with fine-tuning set (20% for fine-tuning, 80% for classifier)
    logger.info(f"Creating data splits with 20% reserved for diffusion fine-tuning...")
    finetune_df, train_df, val_df, test_df = create_data_splits_with_finetune(
        data_size, random_seed, finetune_ratio=0.2
    )

    logger.info(f"Data split sizes:")
    logger.info(f"  Fine-tuning set: {finetune_df.shape}")
    logger.info(f"  Training set: {train_df.shape}")
    logger.info(f"  Validation set: {val_df.shape}")
    logger.info(f"  Test set: {test_df.shape}")

    # Save fine-tuning data
    finetune_dir = os.path.join(run_dir, "finetuning_data")
    os.makedirs(finetune_dir, exist_ok=True)

    finetune_csv = os.path.join(finetune_dir, "finetune.csv")
    train_csv = os.path.join(finetune_dir, "train.csv")
    val_csv = os.path.join(finetune_dir, "val.csv")
    test_csv = os.path.join(finetune_dir, "test.csv")

    finetune_df.to_csv(finetune_csv, index=False)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    logger.info(f"Saved splits to {finetune_dir}")

    # Fine-tune diffusion model with LoRA
    logger.info("")
    logger.info("="*80)
    logger.info("FINE-TUNING DIFFUSION MODEL WITH LORA")
    logger.info("="*80)

    try:
        from config import LORA_OUTPUT_DIR
        lora_output_dir = os.path.join(
            LORA_OUTPUT_DIR,
            f"seed_{random_seed}_size_{data_size}"
        )

        lora_weights_path, finetune_co2 = finetune_diffusion_lora(
            finetune_csv=finetune_csv,
            image_root=DATA_ROOT,
            output_dir=lora_output_dir,
            log_dir=os.path.join(run_dir, "logs")
        )

        logger.info(f"LoRA fine-tuning completed!")
        logger.info(f"LoRA weights saved to: {lora_weights_path}")
        logger.info(f"Fine-tuning CO2 emissions: {finetune_co2:.6f} kg")

        # Load fine-tuned diffusion model
        logger.info("")
        logger.info("Loading fine-tuned diffusion model...")
        finetuned_pipeline = load_diffusion_model_with_lora(
            lora_weights_path=lora_weights_path,
            device=device
        )

        # Get number of training samples for calculating synthetic image count
        num_train_samples = len(train_df)

        # Run augmented experiments with fine-tuned model
        for gen_ratio in IMAGE_GEN_RATIOS:
            logger.info("")
            logger.info("="*80)
            logger.info(f"FINE-TUNED AUGMENTATION - Ratio: {gen_ratio}")
            logger.info("="*80)

            # Calculate number of synthetic images
            num_synthetic = int(num_train_samples * gen_ratio)
            logger.info(f"Generating {num_synthetic} synthetic images from fine-tuned model...")

            # Create directory for this ratio
            gen_dir = os.path.join(
                GENERATION_OUTPUT_DIR,
                f"finetuned_seed_{random_seed}",
                f"size_{data_size}",
                f"ratio_{gen_ratio}"
            )
            os.makedirs(gen_dir, exist_ok=True)

            try:
                # Generate synthetic images
                generated_paths, gen_emissions = generate_synthetic_images(
                    num_images=num_synthetic,
                    output_dir=gen_dir,
                    pipeline=finetuned_pipeline,
                    seed=random_seed,
                    log_dir=os.path.join(run_dir, "logs"),
                    device=device
                )

                # Create CSV for synthetic data
                synthetic_csv = os.path.join(gen_dir, "synthetic_data.csv")
                create_synthetic_csv(
                    generated_paths=generated_paths,
                    target_feature=TARGET_FEATURE,
                    target_value=1.0,
                    output_csv=synthetic_csv,
                    data_root=DATA_ROOT
                )

                # Create augmented training set
                augmented_train_csv = os.path.join(gen_dir, "augmented_train.csv")
                augment_training_data(
                    original_train_csv=train_csv,
                    synthetic_csv=synthetic_csv,
                    output_csv=augmented_train_csv
                )

                # Create output directory for this experiment
                aug_output_dir = os.path.join(run_dir, f"finetuned_ratio_{gen_ratio}")
                aug_log_dir = os.path.join(aug_output_dir, "logs")

                # Train classifier with augmented data
                logger.info(f"Training classifier with fine-tuned augmented data (ratio {gen_ratio})...")
                aug_results = train_model(
                    train_csv=augmented_train_csv,
                    val_csv=val_csv,
                    test_csv=test_csv,
                    output_dir=aug_output_dir,
                    log_dir=aug_log_dir,
                    target_feature=TARGET_FEATURE,
                    data_root=DATA_ROOT,
                    num_epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    learning_rate=LEARNING_RATE,
                    device=device
                )

                # Add experiment metadata
                aug_results['data_size'] = data_size
                aug_results['random_seed'] = random_seed
                aug_results['run_number'] = run_number
                aug_results['gen_ratio'] = gen_ratio
                aug_results['num_synthetic'] = num_synthetic
                aug_results['generation_co2_kg'] = float(gen_emissions)
                aug_results['finetune_co2_kg'] = float(finetune_co2)
                aug_results['augmented'] = True
                aug_results['finetuned'] = True
                aug_results['synthetic_dir'] = gen_dir
                aug_results['lora_weights_path'] = lora_weights_path
                aug_results['finetune_samples'] = len(finetune_df)

                # Save results
                results_file = os.path.join(aug_output_dir, "results.json")
                with open(results_file, 'w') as f:
                    json.dump(aug_results, f, indent=2)

                finetuned_results.append(aug_results)

                logger.info(f"Fine-tuned augmentation (ratio {gen_ratio}) completed successfully!")

            except Exception as e:
                logger.error(f"Error in fine-tuned augmentation (ratio {gen_ratio}): {str(e)}")
                continue

        # Clean up fine-tuned pipeline
        del finetuned_pipeline
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error in LoRA fine-tuning: {str(e)}")
        logger.error("Skipping fine-tuned experiments for this run")
        return finetuned_results

    return finetuned_results


def print_summary_statistics(all_results, total_runs, logger):
    """
    Print summary statistics for all completed runs

    Args:
        all_results (list): List of result dictionaries
        total_runs (int): Total number of runs
        logger: Logger instance
    """
    if not all_results:
        return

    logger.info("="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)

    # Separate baseline, augmented, and fine-tuned results
    baseline_results = [r for r in all_results if not r.get('augmented', False)]
    augmented_results = [r for r in all_results if r.get('augmented', False) and not r.get('finetuned', False)]
    finetuned_results = [r for r in all_results if r.get('finetuned', False)]

    logger.info(f"Total baseline runs: {len(baseline_results)}")
    logger.info(f"Total augmented runs (non-finetuned): {len(augmented_results)}")
    logger.info(f"Total fine-tuned runs: {len(finetuned_results)}")
    logger.info("")

    # Baseline statistics
    logger.info("-"*80)
    logger.info("BASELINE RESULTS (No Synthetic Data)")
    logger.info("-"*80)
    for data_size in DATASET_SIZES:
        size_results = [r for r in baseline_results if r['data_size'] == data_size]
        if size_results:
            avg_val_acc = np.mean([r['final_metrics']['accuracy'] for r in size_results])
            avg_val_auc = np.mean([r['final_metrics']['auc'] for r in size_results])

            logger.info(f"Dataset size: {data_size}")
            logger.info(f"  Avg validation accuracy: {avg_val_acc:.4f}")
            logger.info(f"  Avg validation AUC: {avg_val_auc:.4f}")

            if 'test_metrics' in size_results[0]:
                avg_test_acc = np.mean([r['test_metrics']['accuracy'] for r in size_results])
                avg_test_auc = np.mean([r['test_metrics']['auc'] for r in size_results])
                logger.info(f"  Avg test accuracy: {avg_test_acc:.4f}")
                logger.info(f"  Avg test AUC: {avg_test_auc:.4f}")

    # Augmented statistics (non-finetuned)
    logger.info("")
    logger.info("-"*80)
    logger.info("AUGMENTED RESULTS (Non-Finetuned Diffusion Model)")
    logger.info("-"*80)
    for data_size in DATASET_SIZES:
        for gen_ratio in IMAGE_GEN_RATIOS:
            ratio_results = [
                r for r in augmented_results
                if r['data_size'] == data_size and r.get('gen_ratio') == gen_ratio
            ]
            if ratio_results:
                avg_val_acc = np.mean([r['final_metrics']['accuracy'] for r in ratio_results])
                avg_val_auc = np.mean([r['final_metrics']['auc'] for r in ratio_results])

                logger.info(f"Dataset size: {data_size}, Ratio: {gen_ratio}")
                logger.info(f"  Avg validation accuracy: {avg_val_acc:.4f}")
                logger.info(f"  Avg validation AUC: {avg_val_auc:.4f}")

                if 'test_metrics' in ratio_results[0]:
                    avg_test_acc = np.mean([r['test_metrics']['accuracy'] for r in ratio_results])
                    avg_test_auc = np.mean([r['test_metrics']['auc'] for r in ratio_results])
                    logger.info(f"  Avg test accuracy: {avg_test_acc:.4f}")
                    logger.info(f"  Avg test AUC: {avg_test_auc:.4f}")

    # Fine-tuned statistics
    logger.info("")
    logger.info("-"*80)
    logger.info("FINE-TUNED RESULTS (Fine-Tuned Diffusion Model)")
    logger.info("-"*80)
    for data_size in DATASET_SIZES:
        for gen_ratio in IMAGE_GEN_RATIOS:
            ratio_results = [
                r for r in finetuned_results
                if r['data_size'] == data_size and r.get('gen_ratio') == gen_ratio
            ]
            if ratio_results:
                avg_val_acc = np.mean([r['final_metrics']['accuracy'] for r in ratio_results])
                avg_val_auc = np.mean([r['final_metrics']['auc'] for r in ratio_results])

                logger.info(f"Dataset size: {data_size}, Ratio: {gen_ratio}")
                logger.info(f"  Avg validation accuracy: {avg_val_acc:.4f}")
                logger.info(f"  Avg validation AUC: {avg_val_auc:.4f}")

                if 'test_metrics' in ratio_results[0]:
                    avg_test_acc = np.mean([r['test_metrics']['accuracy'] for r in ratio_results])
                    avg_test_auc = np.mean([r['test_metrics']['auc'] for r in ratio_results])
                    logger.info(f"  Avg test accuracy: {avg_test_acc:.4f}")
                    logger.info(f"  Avg test AUC: {avg_test_auc:.4f}")

    # Overall CO2 statistics
    logger.info("")
    logger.info("-"*80)
    logger.info("CARBON EMISSIONS")
    logger.info("-"*80)

    total_training_co2 = sum([r['training_co2_kg'] for r in all_results])
    total_generation_co2 = sum([r.get('generation_co2_kg', 0) for r in augmented_results + finetuned_results])
    total_finetune_co2 = sum([r.get('finetune_co2_kg', 0) for r in finetuned_results])

    logger.info(f"Total classifier training CO2: {total_training_co2:.6f} kg")
    logger.info(f"Total image generation CO2: {total_generation_co2:.6f} kg")
    logger.info(f"Total diffusion fine-tuning CO2: {total_finetune_co2:.6f} kg")

    if all_results and 'test_co2_kg' in all_results[0]:
        total_test_co2 = sum([r.get('test_co2_kg', 0) for r in all_results])
        logger.info(f"Total test CO2: {total_test_co2:.6f} kg")
        grand_total = total_training_co2 + total_generation_co2 + total_finetune_co2 + total_test_co2
        logger.info(f"Grand total CO2: {grand_total:.6f} kg")
    else:
        grand_total = total_training_co2 + total_generation_co2 + total_finetune_co2
        logger.info(f"Grand total CO2: {grand_total:.6f} kg")


def main():
    """Run the full simulation"""
    # Setup main logger
    logger = setup_logger('simulation', log_dir='./log')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # Store all results
    all_results = []

    # Create summary log
    config_dict = {
        'Target feature': TARGET_FEATURE,
        'Data root': DATA_ROOT,
        'Dataset sizes': DATASET_SIZES,
        'Random seeds': RANDOM_SEEDS,
        'Total runs': len(DATASET_SIZES) * len(RANDOM_SEEDS)
    }
    summary_log = create_summary_log(BASE_OUTPUT_DIR, config_dict, log_dir='./log')

    logger.info("="*80)
    logger.info("STARTING SIMULATION")
    logger.info("="*80)
    logger.info(f"Dataset sizes: {DATASET_SIZES}")
    logger.info(f"Random seeds: {RANDOM_SEEDS}")
    logger.info(f"Total runs: {len(DATASET_SIZES) * len(RANDOM_SEEDS)}")
    logger.info("="*80)

    run_number = 0
    total_runs = len(DATASET_SIZES) * len(RANDOM_SEEDS)

    # Run experiments
    for data_size in DATASET_SIZES:
        for random_seed in RANDOM_SEEDS:
            run_number += 1

            try:
                # Run baseline experiment (no synthetic data)
                logger.info("="*80)
                logger.info("BASELINE EXPERIMENT (No Synthetic Data)")
                logger.info("="*80)

                results = run_single_experiment(
                    data_size, random_seed, run_number, total_runs, device, logger
                )
                results['augmented'] = False  # Mark as baseline
                all_results.append(results)

                # Log baseline success
                log_run_result(
                    summary_log, run_number, total_runs,
                    data_size, random_seed, results=results
                )

                # Now run augmented experiments with synthetic data
                logger.info("")
                logger.info("="*80)
                logger.info("STARTING AUGMENTED EXPERIMENTS")
                logger.info("="*80)

                # Get the paths from the baseline run
                run_dir = results['run_dir']
                train_csv = os.path.join(run_dir, "train.csv")
                val_csv = os.path.join(run_dir, "val.csv")
                test_csv = os.path.join(run_dir, "test.csv")

                # Run augmented experiments
                augmented_results = run_augmented_experiments(
                    data_size=data_size,
                    random_seed=random_seed,
                    run_number=run_number,
                    total_runs=total_runs,
                    train_csv=train_csv,
                    val_csv=val_csv,
                    test_csv=test_csv,
                    run_dir=run_dir,
                    device=device,
                    logger=logger
                )

                # Add augmented results to all results
                all_results.extend(augmented_results)

                # Now run fine-tuned experiments
                logger.info("")
                logger.info("="*80)
                logger.info("STARTING FINE-TUNED EXPERIMENTS")
                logger.info("="*80)

                finetuned_results = run_finetuned_experiments(
                    data_size=data_size,
                    random_seed=random_seed,
                    run_number=run_number,
                    total_runs=total_runs,
                    run_dir=run_dir,
                    device=device,
                    logger=logger
                )

                # Add fine-tuned results to all results
                all_results.extend(finetuned_results)

                logger.info(f"Run {run_number}/{total_runs} completed with all experiments!")

            except Exception as e:
                error_msg = f"Error in run {run_number}: {str(e)}"
                logger.error(error_msg)

                # Log failure
                log_run_result(
                    summary_log, run_number, total_runs,
                    data_size, random_seed, error=e
                )

                # Continue with next run
                continue

    # Save aggregated results
    logger.info("="*80)
    logger.info("SIMULATION COMPLETE")
    logger.info("="*80)

    all_results_file = os.path.join(BASE_OUTPUT_DIR, "all_results.json")
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"All results saved to: {all_results_file}")

    # Print summary statistics
    print_summary_statistics(all_results, total_runs, logger)

    # Finalize summary log
    finalize_summary_log(summary_log, len(all_results), total_runs)

    logger.info(f"Summary log saved to: {summary_log}")
    logger.info("Simulation complete!")


if __name__ == "__main__":
    main()
