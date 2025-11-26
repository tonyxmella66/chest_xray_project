"""
Training and evaluation utilities for chest X-ray classification.
"""
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from codecarbon import EmissionsTracker

from config import (
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    LR_SCHEDULER_STEP_SIZE,
    LR_SCHEDULER_GAMMA,
    NUM_WORKERS,
    SHUFFLE_TRAIN,
    TARGET_FEATURE,
    DATA_ROOT,
    TORCH_SEED,
    NUMPY_SEED
)
from data.dataset import ChestXRayDataset
from data.transforms import get_transforms
from models.densenet import create_densenet121
from utils.metrics import calculate_metrics
from utils.logging import setup_logger

logger = logging.getLogger(__name__)


def set_random_seeds():
    """Set random seeds for reproducibility"""
    torch.manual_seed(TORCH_SEED)
    np.random.seed(NUMPY_SEED)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch

    Args:
        model (torch.nn.Module): Model to train
        dataloader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """
    Validate the model for one epoch

    Args:
        model (torch.nn.Module): Model to validate
        dataloader (DataLoader): Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        tuple: (epoch_loss, epoch_accuracy, predictions, labels, probabilities)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions and probabilities for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            probabilities = torch.softmax(outputs, dim=1)
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc, all_predictions, all_labels, all_probabilities


def train_model(train_csv, val_csv, test_csv, output_dir, log_dir,
                target_feature=TARGET_FEATURE, data_root=DATA_ROOT,
                num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE, device=None):
    """
    Train a single model with the given train/val/test split

    Args:
        train_csv (str): Path to training CSV
        val_csv (str): Path to validation CSV
        test_csv (str): Path to test CSV
        output_dir (str): Directory to save outputs
        log_dir (str): Directory to save logs
        target_feature (str): Target feature to classify
        data_root (str): Root directory for image data
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        device: Device to train on (if None, will auto-detect)

    Returns:
        dict: Training results including metrics and emissions
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    # Setup logger for this training run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_logger = setup_logger(
        f'training_{timestamp}',
        log_dir=log_dir,
        log_file=f'training_{timestamp}.log'
    )

    run_logger.info(f"Using device: {device}")

    # Load data
    run_logger.info(f"Loading training data from {train_csv}...")
    train_dataframe = pd.read_csv(train_csv)

    run_logger.info(f"Loading validation data from {val_csv}...")
    val_dataframe = pd.read_csv(val_csv)

    run_logger.info("="*70)
    run_logger.info("Training Configuration")
    run_logger.info("="*70)
    run_logger.info(f"Target feature: {target_feature}")
    run_logger.info(f"Training samples (before filtering): {len(train_dataframe)}")
    run_logger.info(f"Validation samples (before filtering): {len(val_dataframe)}")

    # Set random seeds
    set_random_seeds()

    # Get transforms
    train_transform, val_transform = get_transforms()

    # Initialize emission tracker
    tracker = EmissionsTracker(
        project_name="DenseNet121_Training",
        output_dir=log_dir,
        output_file=f'training_emissions_{timestamp}.csv',
        log_level='warning'
    )

    run_logger.info("="*70)
    run_logger.info("Starting CO2 emission tracking for training...")
    run_logger.info("="*70)
    tracker.start()

    # Create datasets
    run_logger.info(f"Loading training data for target '{target_feature}'...")
    train_dataset = ChestXRayDataset(
        train_dataframe, data_root, target_feature,
        transform=train_transform, balance_classes=False, max_samples=None
    )

    run_logger.info(f"Loading validation data for target '{target_feature}'...")
    val_dataset = ChestXRayDataset(
        val_dataframe, data_root, target_feature,
        transform=val_transform, balance_classes=False, max_samples=None
    )

    run_logger.info(f"Training samples: {len(train_dataset)}")
    run_logger.info(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=SHUFFLE_TRAIN, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=NUM_WORKERS
    )

    # Create model
    run_logger.info("Creating DenseNet121 model...")
    model = create_densenet121()
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_SCHEDULER_STEP_SIZE, gamma=LR_SCHEDULER_GAMMA
    )

    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    patience_counter = 0

    run_logger.info("="*70)
    run_logger.info("Starting training...")
    run_logger.info("="*70)

    for epoch in range(num_epochs):
        run_logger.info(f"Epoch {epoch+1}/{num_epochs}")
        run_logger.info("-" * 50)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_predictions, val_labels, val_probabilities = validate_epoch(
            model, val_loader, criterion, device
        )

        # Calculate metrics
        metrics = calculate_metrics(val_predictions, val_labels, val_probabilities)

        # Update learning rate
        scheduler.step()

        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Log results
        run_logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        run_logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        run_logger.info(f"Val Precision: {metrics['precision']:.4f}")
        run_logger.info(f"Val Recall: {metrics['recall']:.4f}")
        run_logger.info(f"Val F1: {metrics['f1']:.4f}")
        run_logger.info(f"Val AUC: {metrics['auc']:.4f}")
        run_logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model and implement early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = metrics
            patience_counter = 0
            model_filename = os.path.join(
                output_dir,
                f'best_densenet_{target_feature.lower().replace(" ", "_")}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'metrics': metrics,
                'target_feature': target_feature
            }, model_filename)
            run_logger.info("New best model saved!")
        else:
            patience_counter += 1
            run_logger.info(f"No improvement for {patience_counter} epochs")

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            run_logger.info(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement")
            break

    # Stop training emissions tracking
    training_emissions = tracker.stop()
    run_logger.info("="*70)
    run_logger.info(f"Training CO2 Emissions: {training_emissions:.6f} kg")
    run_logger.info("="*70)

    run_logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")

    # Load best model
    model_filename = os.path.join(
        output_dir,
        f'best_densenet_{target_feature.lower().replace(" ", "_")}.pth'
    )
    checkpoint = torch.load(model_filename)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation on validation set
    val_loss, val_acc, val_predictions, val_labels, val_probabilities = validate_epoch(
        model, val_loader, criterion, device
    )
    final_metrics = calculate_metrics(val_predictions, val_labels, val_probabilities)

    run_logger.info("="*70)
    run_logger.info(f"Final Validation Results - {target_feature}")
    run_logger.info("="*70)
    run_logger.info(f"Validation Accuracy: {final_metrics['accuracy']:.4f}")
    run_logger.info(f"Validation Precision: {final_metrics['precision']:.4f}")
    run_logger.info(f"Validation Recall: {final_metrics['recall']:.4f}")
    run_logger.info(f"Validation F1-Score: {final_metrics['f1']:.4f}")
    run_logger.info(f"Validation AUC: {final_metrics['auc']:.4f}")

    # Initialize results dictionary
    results = {
        'target_feature': target_feature,
        'best_val_acc': float(best_val_acc),
        'final_metrics': {k: float(v) for k, v in final_metrics.items()},
        'training_co2_kg': float(training_emissions),
        'num_epochs_trained': len(train_losses),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset)
    }

    # Evaluate on test set
    if test_csv is not None and os.path.exists(test_csv):
        run_logger.info("="*70)
        run_logger.info("Test Set Evaluation")
        run_logger.info("="*70)

        # Load test data
        run_logger.info(f"Loading test data from {test_csv}...")
        test_dataframe = pd.read_csv(test_csv)

        # Start test emissions tracking
        test_tracker = EmissionsTracker(
            project_name="DenseNet121_Test_Evaluation",
            output_dir=log_dir,
            output_file=f'test_emissions_{timestamp}.csv',
            log_level='warning'
        )
        run_logger.info("Starting CO2 emission tracking for test evaluation...")
        test_tracker.start()

        # Create test dataset
        test_dataset = ChestXRayDataset(
            test_dataframe, data_root, target_feature,
            transform=val_transform, balance_classes=False, max_samples=None
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=NUM_WORKERS
        )

        # Evaluate
        test_loss, test_acc, test_predictions, test_labels, test_probabilities = validate_epoch(
            model, test_loader, criterion, device
        )
        test_metrics = calculate_metrics(test_predictions, test_labels, test_probabilities)

        # Stop test emissions tracking
        test_emissions = test_tracker.stop()
        run_logger.info("="*70)
        run_logger.info(f"Test Evaluation CO2 Emissions: {test_emissions:.6f} kg")
        run_logger.info("="*70)

        run_logger.info(f"Test Set Results for {target_feature}:")
        run_logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        run_logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        run_logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
        run_logger.info(f"Test F1-Score: {test_metrics['f1']:.4f}")
        run_logger.info(f"Test AUC: {test_metrics['auc']:.4f}")

        # Add test results
        results['test_metrics'] = {k: float(v) for k, v in test_metrics.items()}
        results['test_co2_kg'] = float(test_emissions)
        results['test_samples'] = len(test_dataset)

    return results
