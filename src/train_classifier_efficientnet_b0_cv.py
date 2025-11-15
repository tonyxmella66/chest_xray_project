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
    def __init__(self, csv_file, root_dir, target_feature, transform=None, balance_classes=True, max_samples=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            target_feature (string): Name of the target feature column to use for classification.
            transform (callable, optional): Optional transform to be applied on a sample.
            balance_classes (bool): Whether to balance positive and negative classes.
            max_samples (int, optional): Maximum number of samples to load. If None, loads all samples.
        """
        self.data_frame = pd.read_csv(csv_file)
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
    """Define data augmentation and preprocessing transforms for EfficientNet"""
    # EfficientNet-B0 uses 224x224 input size
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

def create_efficientnet_b0_model(num_classes=2):
    """Create an EfficientNet-B0 model for binary classification"""
    model = models.efficientnet_b0(pretrained=True)
    
    # Modify the classifier for binary classification
    # EfficientNet has a classifier with in_features
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
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

def train_single_fold(fold_num, target_feature, data_root, cv_splits_dir, 
                     max_train_samples, num_epochs, batch_size, learning_rate, output_dir):
    """
    Train model on a single fold
    
    Args:
        fold_num (int): Fold number (1-5)
        target_feature (str): Name of the target feature to classify
        data_root (str): Root directory containing the image data
        cv_splits_dir (str): Directory containing the CV split CSV files
        max_train_samples (int): Maximum number of training samples to load
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        output_dir (str): Directory to save outputs
    
    Returns:
        dict: Dictionary containing training results and metrics
    """
    print(f"\n{'='*70}")
    print(f"Training Fold {fold_num}")
    print(f"{'='*70}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42 + fold_num)
    np.random.seed(42 + fold_num)
    
    # Data paths
    train_csv = os.path.join(cv_splits_dir, f"fold_{fold_num}_train.csv")
    val_csv = os.path.join(cv_splits_dir, f"fold_{fold_num}_val.csv")
    
    # Verify files exist
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    print(f"\nLoading training data for target '{target_feature}' (max {max_train_samples} samples)...")
    train_dataset = ChestXRayDataset(train_csv, data_root, target_feature, transform=train_transform, 
                                   balance_classes=True, max_samples=max_train_samples)
    print(f"Loading validation data for target '{target_feature}'...")
    val_dataset = ChestXRayDataset(val_csv, data_root, target_feature, transform=val_transform, 
                                 balance_classes=False, max_samples=None)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = create_efficientnet_b0_model(num_classes=2)
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
    
    print("Starting training...")
    
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
            patience_counter = 0
            model_filename = os.path.join(output_dir, 
                f'best_efficientnet_b0_fold{fold_num}_{target_feature.lower().replace(" ", "_")}.pth')
            torch.save({
                'epoch': epoch,
                'fold': fold_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'metrics': metrics,
                'target_feature': target_feature,
                'model_type': 'efficientnet_b0'
            }, model_filename)
            print("New best model saved!")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    print(f"\nTraining completed for Fold {fold_num}! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training curves for this fold
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'Fold {fold_num}: Training and Validation Loss (EfficientNet-B0)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title(f'Fold {fold_num}: Training and Validation Accuracy (EfficientNet-B0)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot([optimizer.param_groups[0]['lr']] * len(train_losses))
    plt.title(f'Fold {fold_num}: Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'training_curves_fold{fold_num}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Load best model and evaluate
    print(f"\nLoading best model for fold {fold_num} final evaluation...")
    model_filename = os.path.join(output_dir, 
        f'best_efficientnet_b0_fold{fold_num}_{target_feature.lower().replace(" ", "_")}.pth')
    checkpoint = torch.load(model_filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    val_loss, val_acc, val_predictions, val_labels, val_probabilities = validate_epoch(
        model, val_loader, criterion, device
    )
    final_metrics = calculate_metrics(val_predictions, val_labels, val_probabilities)
    
    print(f"\nFinal Evaluation Results for Fold {fold_num} - {target_feature}:")
    print(f"Validation Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Validation Precision: {final_metrics['precision']:.4f}")
    print(f"Validation Recall: {final_metrics['recall']:.4f}")
    print(f"Validation F1-Score: {final_metrics['f1']:.4f}")
    print(f"Validation AUC: {final_metrics['auc']:.4f}")
    
    return {
        'fold': fold_num,
        'best_val_acc': best_val_acc,
        'final_metrics': final_metrics,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'model_filename': model_filename
    }

def evaluate_on_test(model, test_csv, data_root, target_feature, batch_size, device):
    """Evaluate model on test set"""
    print(f"\n{'='*70}")
    print(f"Evaluating on Test Set")
    print(f"{'='*70}")
    
    # Get transform
    _, test_transform = get_transforms()
    
    # Create test dataset
    test_dataset = ChestXRayDataset(test_csv, data_root, target_feature, 
                                   transform=test_transform, balance_classes=False, max_samples=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate
    test_loss, test_acc, test_predictions, test_labels, test_probabilities = validate_epoch(
        model, test_loader, criterion, device
    )
    test_metrics = calculate_metrics(test_predictions, test_labels, test_probabilities)
    
    print(f"\nTest Set Results for {target_feature}:")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1']:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    
    return test_metrics

def main(target_feature="Pleural Effusion", folds=[1, 2, 3, 4, 5], data_root="data",
         cv_splits_dir="cv_splits", test_csv="test.csv", max_train_samples=200,
         num_epochs=20, batch_size=32, learning_rate=0.001, output_dir="cv_outputs_efficientnet_b0", log_dir="."):
    """
    Main training function with cross-validation using EfficientNet-B0

    Args:
        target_feature (str): Name of the target feature to classify
        folds (list): List of fold numbers to train on
        data_root (str): Root directory containing the image data
        cv_splits_dir (str): Directory containing the CV split CSV files
        test_csv (str): Path to test CSV file
        max_train_samples (int): Maximum number of training samples to load per fold
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        output_dir (str): Directory to save all outputs
        log_dir (str): Directory to save log files
    """
    # Setup logging
    log_file, log_writer = setup_logging(log_dir, 'efficientnet_b0_training')
    print(f"Logging to: {log_file}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize emission tracker for training
    tracker = EmissionsTracker(
        project_name="EfficientNet_B0_Training",
        output_dir=log_dir,
        output_file=f'efficientnet_b0_training_emissions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        log_level='warning'
    )

    print("\n" + "="*70)
    print("Starting CO2 emission tracking for training...")
    print("="*70)
    tracker.start()

    # Store results for all folds
    all_fold_results = []

    # Train on each fold
    for fold_num in folds:
        fold_results = train_single_fold(
            fold_num=fold_num,
            target_feature=target_feature,
            data_root=data_root,
            cv_splits_dir=cv_splits_dir,
            max_train_samples=max_train_samples,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir
        )
        all_fold_results.append(fold_results)

    # Stop training emissions tracking
    training_emissions = tracker.stop()
    print("\n" + "="*70)
    print(f"Training CO2 Emissions: {training_emissions:.6f} kg")
    print("="*70)

    # Calculate average metrics across folds
    print(f"\n{'='*70}")
    print(f"Cross-Validation Summary for {target_feature} (EfficientNet-B0)")
    print(f"{'='*70}")
    
    avg_accuracy = np.mean([r['final_metrics']['accuracy'] for r in all_fold_results])
    avg_precision = np.mean([r['final_metrics']['precision'] for r in all_fold_results])
    avg_recall = np.mean([r['final_metrics']['recall'] for r in all_fold_results])
    avg_f1 = np.mean([r['final_metrics']['f1'] for r in all_fold_results])
    avg_auc = np.mean([r['final_metrics']['auc'] for r in all_fold_results])
    
    std_accuracy = np.std([r['final_metrics']['accuracy'] for r in all_fold_results])
    std_precision = np.std([r['final_metrics']['precision'] for r in all_fold_results])
    std_recall = np.std([r['final_metrics']['recall'] for r in all_fold_results])
    std_f1 = np.std([r['final_metrics']['f1'] for r in all_fold_results])
    std_auc = np.std([r['final_metrics']['auc'] for r in all_fold_results])
    
    print(f"\nAverage Metrics Across {len(folds)} Folds:")
    print(f"Accuracy:  {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Precision: {avg_precision:.4f} ± {std_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f} ± {std_recall:.4f}")
    print(f"F1-Score:  {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"AUC:       {avg_auc:.4f} ± {std_auc:.4f}")
    
    # Save cross-validation results
    cv_summary = {
        'model_type': 'efficientnet_b0',
        'target_feature': target_feature,
        'num_folds': len(folds),
        'folds': folds,
        'avg_metrics': {
            'accuracy': float(avg_accuracy),
            'precision': float(avg_precision),
            'recall': float(avg_recall),
            'f1': float(avg_f1),
            'auc': float(avg_auc)
        },
        'std_metrics': {
            'accuracy': float(std_accuracy),
            'precision': float(std_precision),
            'recall': float(std_recall),
            'f1': float(std_f1),
            'auc': float(std_auc)
        },
        'fold_results': [
            {
                'fold': r['fold'],
                'best_val_acc': r['best_val_acc'],
                'metrics': {k: float(v) for k, v in r['final_metrics'].items()}
            }
            for r in all_fold_results
        ]
    }
    
    summary_file = os.path.join(output_dir, f'cv_summary_{target_feature.lower().replace(" ", "_")}.json')
    with open(summary_file, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    print(f"\nCross-validation summary saved to: {summary_file}")
    
    # Evaluate on test set using the best fold model
    if os.path.exists(test_csv):
        print(f"\n{'='*70}")
        print(f"Test Set Evaluation")
        print(f"{'='*70}")

        # Start test emissions tracking
        test_tracker = EmissionsTracker(
            project_name="EfficientNet_B0_Test_Evaluation",
            output_dir=log_dir,
            output_file=f'efficientnet_b0_test_emissions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            log_level='warning'
        )
        print("\nStarting CO2 emission tracking for test evaluation...")
        test_tracker.start()

        # Find the fold with the best validation accuracy
        best_fold_idx = np.argmax([r['best_val_acc'] for r in all_fold_results])
        best_fold = all_fold_results[best_fold_idx]

        print(f"\nUsing best fold model (Fold {best_fold['fold']}) for test evaluation")
        print(f"Best fold validation accuracy: {best_fold['best_val_acc']:.2f}%")

        # Load best fold model
        model = create_efficientnet_b0_model(num_classes=2)
        checkpoint = torch.load(best_fold['model_filename'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        # Evaluate on test set
        test_metrics = evaluate_on_test(model, test_csv, data_root, target_feature, batch_size, device)

        # Stop test emissions tracking
        test_emissions = test_tracker.stop()
        print("\n" + "="*70)
        print(f"Test Evaluation CO2 Emissions: {test_emissions:.6f} kg")
        print("="*70)

        # Add test results to summary
        cv_summary['test_metrics'] = {k: float(v) for k, v in test_metrics.items()}
        cv_summary['best_fold_for_test'] = int(best_fold['fold'])
        cv_summary['training_co2_kg'] = float(training_emissions)
        cv_summary['test_co2_kg'] = float(test_emissions)

        with open(summary_file, 'w') as f:
            json.dump(cv_summary, f, indent=2)
        
    else:
        print(f"\nTest CSV not found at {test_csv}, skipping test evaluation")
    
    # Create summary plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Cross-Validation Results for {target_feature} (EfficientNet-B0)', fontsize=16)
    
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    for idx, metric in enumerate(metrics_names):
        row = idx // 3
        col = idx % 3
        
        fold_nums = [r['fold'] for r in all_fold_results]
        metric_values = [r['final_metrics'][metric] for r in all_fold_results]
        
        axes[row, col].bar(fold_nums, metric_values, alpha=0.7)
        axes[row, col].axhline(y=np.mean(metric_values), color='r', linestyle='--', 
                              label=f'Mean: {np.mean(metric_values):.4f}')
        axes[row, col].set_xlabel('Fold')
        axes[row, col].set_ylabel(metric.capitalize())
        axes[row, col].set_title(f'{metric.capitalize()} Across Folds')
        axes[row, col].legend()
        axes[row, col].set_xticks(fold_nums)
    
    # Hide the last subplot (2,2) since we only have 5 metrics
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    summary_plot = os.path.join(output_dir, f'cv_summary_plot_{target_feature.lower().replace(" ", "_")}.png')
    plt.savefig(summary_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"Cross-Validation Complete!")
    print(f"{'='*70}")
    print(f"All outputs saved to: {output_dir}")
    print(f"Summary file: {summary_file}")
    print(f"Summary plot: {summary_plot}")

    # Close log writer
    log_writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train EfficientNet-B0 with Cross-Validation for Chest X-Ray Classification')
    parser.add_argument('--target', type=str, default='Pleural Effusion',
                       help='Target feature to classify (default: "Pleural Effusion")')
    parser.add_argument('--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                       help='List of fold numbers to train on (default: 1 2 3 4 5)')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Root directory containing the image data (default: "data")')
    parser.add_argument('--cv_splits_dir', type=str, default='cv_splits',
                       help='Directory containing CV split CSV files (default: "cv_splits")')
    parser.add_argument('--test_csv', type=str, default='test.csv',
                       help='Path to test CSV file (default: "test.csv")')
    parser.add_argument('--max_train_samples', type=int, default=200,
                       help='Maximum number of training samples per fold (default: 200, use 0 for all)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--output_dir', type=str, default='cv_outputs_efficientnet_b0',
                       help='Directory to save all outputs (default: "cv_outputs_efficientnet_b0")')
    parser.add_argument('--log_dir', type=str, default='.',
                       help='Directory to save log files and emissions data (default: current directory)')

    args = parser.parse_args()
    
    # Convert 0 to None for loading all samples
    max_samples = None if args.max_train_samples == 0 else args.max_train_samples
    
    print(f"\n{'='*70}")
    print(f"Cross-Validation Training Configuration (EfficientNet-B0)")
    print(f"{'='*70}")
    print(f"  Model: EfficientNet-B0")
    print(f"  Target feature: {args.target}")
    print(f"  Folds: {args.folds}")
    print(f"  Data root: {args.data_root}")
    print(f"  CV splits directory: {args.cv_splits_dir}")
    print(f"  Test CSV: {args.test_csv}")
    print(f"  Max training samples per fold: {max_samples if max_samples else 'All available'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*70}\n")
    
    main(target_feature=args.target,
         folds=args.folds,
         data_root=args.data_root,
         cv_splits_dir=args.cv_splits_dir,
         test_csv=args.test_csv,
         max_train_samples=max_samples,
         num_epochs=args.epochs,
         batch_size=args.batch_size,
         learning_rate=args.learning_rate,
         output_dir=args.output_dir,
         log_dir=args.log_dir)
