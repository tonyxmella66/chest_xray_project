import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import warnings
import json
from datetime import datetime
from codecarbon import EmissionsTracker
import sys
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
    def __init__(self, dataframe, root_dir, target_feature, transform=None, balance_classes=True, max_samples=None):
        """
        Args:
            dataframe (DataFrame): Pandas dataframe with annotations.
            root_dir (string): Directory with all the images.
            target_feature (string): Name of the target feature column to use for classification.
            transform (callable, optional): Optional transform to be applied on a sample.
            balance_classes (bool): Whether to balance positive and negative classes.
            max_samples (int, optional): Maximum number of samples to load. If None, loads all samples.
        """
        self.data_frame = dataframe.copy()
        self.root_dir = root_dir
        self.transform = transform
        self.target_feature = target_feature

        # Check if target feature exists in the dataset
        if target_feature not in self.data_frame.columns:
            available_features = [col for col in self.data_frame.columns if col not in ['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA']]
            raise ValueError(f"Target feature '{target_feature}' not found in dataset. Available features: {available_features}")

        # Filter out rows with missing target feature values
        self.data_frame = self.data_frame.dropna(subset=[target_feature])

        # Create binary labels: 1 for target = 1.0, 0 for target = 0.0
        self.data_frame = self.data_frame[self.data_frame[target_feature].isin([0.0, 1.0])]
        self.data_frame['label'] = self.data_frame[target_feature].astype(int)

        if balance_classes:
            # Balance the dataset by sampling equal numbers of positive and negative cases
            pos_cases = self.data_frame[self.data_frame['label'] == 1]
            neg_cases = self.data_frame[self.data_frame['label'] == 0]

            min_samples = min(len(pos_cases), len(neg_cases))
            if min_samples > 0:
                pos_sampled = pos_cases.sample(n=min_samples, random_state=42)
                neg_sampled = neg_cases.sample(n=min_samples, random_state=42)
                self.data_frame = pd.concat([pos_sampled, neg_sampled]).reset_index(drop=True)

        # Limit the number of samples if max_samples is specified
        if max_samples is not None and len(self.data_frame) > max_samples:
            # If we have balanced classes, maintain the balance when limiting samples
            if balance_classes:
                pos_cases = self.data_frame[self.data_frame['label'] == 1]
                neg_cases = self.data_frame[self.data_frame['label'] == 0]

                # Calculate how many samples per class we can take
                samples_per_class = max_samples // 2
                if samples_per_class > 0:
                    pos_limited = pos_cases.sample(n=min(samples_per_class, len(pos_cases)), random_state=42)
                    neg_limited = neg_cases.sample(n=min(samples_per_class, len(neg_cases)), random_state=42)
                    self.data_frame = pd.concat([pos_limited, neg_limited]).reset_index(drop=True)
                else:
                    # If max_samples is very small, just take the first max_samples
                    self.data_frame = self.data_frame.head(max_samples)
            else:
                # If not balancing, just take the first max_samples
                self.data_frame = self.data_frame.head(max_samples)

        print(f"Dataset created with {len(self.data_frame)} samples for target: {target_feature}")
        print(f"Positive cases ({target_feature}=1): {len(self.data_frame[self.data_frame['label'] == 1])}")
        print(f"Negative cases ({target_feature}=0): {len(self.data_frame[self.data_frame['label'] == 0])}")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])

        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # Get the binary label
        label = self.data_frame.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms():
    """Define data augmentation and preprocessing transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def create_densenet_model(num_classes=2, freeze_layers=True):
    """Create a DenseNet model for binary classification"""
    model = models.densenet121(pretrained=True)

    # Freeze early layers for faster fine-tuning
    if freeze_layers:
        for name, param in model.named_parameters():
            # Freeze all layers except the last denseblock and classifier
            if 'denseblock4' not in name and 'classifier' not in name:
                param.requires_grad = False

    # Modify the classifier for binary classification
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch"""
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
    """Validate the model for one epoch"""
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
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of class 1

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc, all_predictions, all_labels, all_probabilities

def calculate_metrics(predictions, labels, probabilities):
    """Calculate comprehensive metrics"""
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    auc = roc_auc_score(labels, probabilities)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def main(target_feature="Pleural Effusion", data_root="data",
         train_csv="data/organized_images/train.csv",
         val_csv="data/organized_images/val.csv",
         test_csv=None,
         max_train_samples=None,
         max_val_samples=None,
         num_epochs=20,
         batch_size=32,
         learning_rate=0.001,
         output_dir="training_outputs",
         log_dir="."):
    """
    Main training function

    Args:
        target_feature (str): Name of the target feature to classify
        data_root (str): Root directory containing the image data
        train_csv (str): Path to the CSV file with training data
        val_csv (str): Path to the CSV file with validation data
        test_csv (str): Path to test CSV file (optional)
        max_train_samples (int): Maximum number of training samples to load
        max_val_samples (int): Maximum number of validation samples to load
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        output_dir (str): Directory to save all outputs
        log_dir (str): Directory to save log files
    """
    # Setup logging
    log_file, log_writer = setup_logging(log_dir, 'densenet121_training')
    print(f"Logging to: {log_file}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the training data
    print(f"\nLoading training data from {train_csv}...")
    train_dataframe = pd.read_csv(train_csv)

    # Load the validation data
    print(f"Loading validation data from {val_csv}...")
    val_dataframe = pd.read_csv(val_csv)

    print(f"\n{'='*70}")
    print(f"Training Configuration")
    print(f"{'='*70}")
    print(f"Target feature: {target_feature}")
    print(f"Training samples (before filtering): {len(train_dataframe)}")
    print(f"Validation samples (before filtering): {len(val_dataframe)}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Get transforms
    train_transform, val_transform = get_transforms()

    # Initialize emission tracker for training
    tracker = EmissionsTracker(
        project_name="DenseNet121_Training",
        output_dir=log_dir,
        output_file=f'densenet121_training_emissions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        log_level='warning'
    )

    print("\n" + "="*70)
    print("Starting CO2 emission tracking for training...")
    print("="*70)
    tracker.start()

    # Create datasets
    print(f"\nLoading training data for target '{target_feature}'...")
    train_dataset = ChestXRayDataset(train_dataframe, data_root, target_feature,
                                     transform=train_transform, balance_classes=False, max_samples=None)
    print(f"\nLoading validation data for target '{target_feature}'...")
    val_dataset = ChestXRayDataset(val_dataframe, data_root, target_feature,
                                   transform=val_transform, balance_classes=False, max_samples=None)

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Create model
    print("\nCreating DenseNet121 model...")
    model = create_densenet_model(num_classes=2)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # Early stopping parameters
    patience = 5
    patience_counter = 0

    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)

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

        # Print results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Val Precision: {metrics['precision']:.4f}")
        print(f"Val Recall: {metrics['recall']:.4f}")
        print(f"Val F1: {metrics['f1']:.4f}")
        print(f"Val AUC: {metrics['auc']:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model and implement early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = metrics
            patience_counter = 0
            model_filename = os.path.join(output_dir,
                f'best_densenet_{target_feature.lower().replace(" ", "_")}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'metrics': metrics,
                'target_feature': target_feature
            }, model_filename)
            print("New best model saved!")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break

    # Stop training emissions tracking
    training_emissions = tracker.stop()
    print("\n" + "="*70)
    print(f"Training CO2 Emissions: {training_emissions:.6f} kg")
    print("="*70)

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")

    # Plot training curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot([optimizer.param_groups[0]['lr']] * len(train_losses))
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')

    plt.tight_layout()
    plot_filename = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Load best model and evaluate
    print(f"\nLoading best model for final evaluation...")
    model_filename = os.path.join(output_dir,
        f'best_densenet_{target_feature.lower().replace(" ", "_")}.pth')
    checkpoint = torch.load(model_filename)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation on validation set
    val_loss, val_acc, val_predictions, val_labels, val_probabilities = validate_epoch(
        model, val_loader, criterion, device
    )
    final_metrics = calculate_metrics(val_predictions, val_labels, val_probabilities)

    print(f"\n{'='*70}")
    print(f"Final Validation Results - {target_feature}")
    print(f"{'='*70}")
    print(f"Validation Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Validation Precision: {final_metrics['precision']:.4f}")
    print(f"Validation Recall: {final_metrics['recall']:.4f}")
    print(f"Validation F1-Score: {final_metrics['f1']:.4f}")
    print(f"Validation AUC: {final_metrics['auc']:.4f}")

    # Save training summary
    training_summary = {
        'target_feature': target_feature,
        'best_val_acc': float(best_val_acc),
        'final_metrics': {k: float(v) for k, v in final_metrics.items()},
        'training_co2_kg': float(training_emissions),
        'num_epochs_trained': len(train_losses),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset)
    }

    # Evaluate on test set if provided
    if test_csv is not None and os.path.exists(test_csv):
        print(f"\n{'='*70}")
        print(f"Test Set Evaluation")
        print(f"{'='*70}")

        # Load test data
        print(f"\nLoading test data from {test_csv}...")
        test_dataframe = pd.read_csv(test_csv)

        # Start test emissions tracking
        test_tracker = EmissionsTracker(
            project_name="DenseNet121_Test_Evaluation",
            output_dir=log_dir,
            output_file=f'densenet121_test_emissions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            log_level='warning'
        )
        print("\nStarting CO2 emission tracking for test evaluation...")
        test_tracker.start()

        # Create test dataset (using all available test data)
        _, test_transform = get_transforms()
        test_dataset = ChestXRayDataset(test_dataframe, data_root, target_feature,
                                       transform=test_transform, balance_classes=False, max_samples=None)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Evaluate
        test_loss, test_acc, test_predictions, test_labels, test_probabilities = validate_epoch(
            model, test_loader, criterion, device
        )
        test_metrics = calculate_metrics(test_predictions, test_labels, test_probabilities)

        # Stop test emissions tracking
        test_emissions = test_tracker.stop()
        print("\n" + "="*70)
        print(f"Test Evaluation CO2 Emissions: {test_emissions:.6f} kg")
        print("="*70)

        print(f"\nTest Set Results for {target_feature}:")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test F1-Score: {test_metrics['f1']:.4f}")
        print(f"Test AUC: {test_metrics['auc']:.4f}")

        # Add test results to summary
        training_summary['test_metrics'] = {k: float(v) for k, v in test_metrics.items()}
        training_summary['test_co2_kg'] = float(test_emissions)
        training_summary['test_samples'] = len(test_dataset)
    else:
        if test_csv is None:
            print(f"\nNo test CSV provided, skipping test evaluation")
        else:
            print(f"\nTest CSV not found at {test_csv}, skipping test evaluation")

    # Save summary to JSON
    summary_file = os.path.join(output_dir, f'training_summary_{target_feature.lower().replace(" ", "_")}.json')
    with open(summary_file, 'w') as f:
        json.dump(training_summary, f, indent=2)

    # Create metrics visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Training Results for {target_feature}', fontsize=16)

    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_values = [final_metrics[m] for m in metrics_names]

    ax.bar(metrics_names, metric_values, alpha=0.7, color='steelblue')
    ax.set_ylabel('Score')
    ax.set_xlabel('Metric')
    ax.set_title('Validation Metrics')
    ax.set_ylim([0, 1])

    # Add value labels on bars
    for i, v in enumerate(metric_values):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    summary_plot = os.path.join(output_dir, f'metrics_plot_{target_feature.lower().replace(" ", "_")}.png')
    plt.savefig(summary_plot, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"All outputs saved to: {output_dir}")
    print(f"Summary file: {summary_file}")
    print(f"Training curves: {plot_filename}")
    print(f"Metrics plot: {summary_plot}")

    # Close log writer
    log_writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DenseNet121 for Chest X-Ray Classification')
    parser.add_argument('--target', type=str, default='Pleural Effusion',
                       help='Target feature to classify (default: "Pleural Effusion")')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Root directory containing the image data (default: "data")')
    parser.add_argument('--train_csv', type=str, required=True,
                       help='Path to CSV file with training data')
    parser.add_argument('--val_csv', type=str, required=True,
                       help='Path to CSV file with validation data')
    parser.add_argument('--test_csv', type=str, default=None,
                       help='Path to test CSV file (optional, default: None)')
    parser.add_argument('--max_train_samples', type=int, default=None,
                       help='Maximum number of training samples (default: None for all)')
    parser.add_argument('--max_val_samples', type=int, default=None,
                       help='Maximum number of validation samples (default: None for all)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--output_dir', type=str, default='training_outputs',
                       help='Directory to save all outputs (default: "training_outputs")')
    parser.add_argument('--log_dir', type=str, default='.',
                       help='Directory to save log files and emissions data (default: current directory)')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Training Configuration")
    print(f"{'='*70}")
    print(f"  Target feature: {args.target}")
    print(f"  Data root: {args.data_root}")
    print(f"  Training CSV: {args.train_csv}")
    print(f"  Validation CSV: {args.val_csv}")
    print(f"  Test CSV: {args.test_csv if args.test_csv else 'None'}")
    print(f"  Max training samples: {args.max_train_samples if args.max_train_samples else 'All available'}")
    print(f"  Max validation samples: {args.max_val_samples if args.max_val_samples else 'All available'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*70}\n")

    main(target_feature=args.target,
         data_root=args.data_root,
         train_csv=args.train_csv,
         val_csv=args.val_csv,
         test_csv=args.test_csv,
         max_train_samples=args.max_train_samples,
         max_val_samples=args.max_val_samples,
         num_epochs=args.epochs,
         batch_size=args.batch_size,
         learning_rate=args.learning_rate,
         output_dir=args.output_dir,
         log_dir=args.log_dir)
