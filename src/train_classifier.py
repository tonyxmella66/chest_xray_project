#!/usr/bin/env python3
"""
Binary Classifier for Pleural Effusion Detection
Using DenseNet121 on CheXpert Dataset

Usage:
    # Quick test with 200 training samples
    python train_classifier.py --max_train_samples 200 --epochs 10
    
    # Full training with all data
    python train_classifier.py --max_train_samples 0 --epochs 50
    
    # Custom configuration
    python train_classifier.py --max_train_samples 5000 --epochs 30 --batch_size 64 --learning_rate 0.0001
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


class ChestXRayDataset(Dataset):
    """
    Dataset class for CheXpert chest X-rays with binary classification support.
    """
    def __init__(self, csv_file, root_dir, target_feature='Pleural Effusion', 
                 transform=None, balance_classes=True, max_samples=None):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            root_dir (str): Directory with all the images (or parent directory).
            target_feature (str): Name of the target feature column.
            transform (callable, optional): Optional transform to be applied.
            balance_classes (bool): Whether to balance positive and negative classes.
            max_samples (int, optional): Maximum number of samples to load.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_feature = target_feature
        
        # Check if target feature exists
        if target_feature not in self.data_frame.columns:
            available_features = [col for col in self.data_frame.columns 
                                if col not in ['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA']]
            raise ValueError(f"Target feature '{target_feature}' not found. Available: {available_features}")
        
        # Filter out rows with missing target values and keep only binary labels
        self.data_frame = self.data_frame.dropna(subset=[target_feature])
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
        
        # Limit the number of samples if specified
        if max_samples is not None and len(self.data_frame) > max_samples:
            if balance_classes:
                # Maintain balance when limiting
                pos_cases = self.data_frame[self.data_frame['label'] == 1]
                neg_cases = self.data_frame[self.data_frame['label'] == 0]
                samples_per_class = max_samples // 2
                
                if samples_per_class > 0:
                    pos_limited = pos_cases.sample(n=min(samples_per_class, len(pos_cases)), random_state=42)
                    neg_limited = neg_cases.sample(n=min(samples_per_class, len(neg_cases)), random_state=42)
                    self.data_frame = pd.concat([pos_limited, neg_limited]).reset_index(drop=True)
            else:
                self.data_frame = self.data_frame.sample(n=max_samples, random_state=42)
        
        print(f"Dataset created: {len(self.data_frame)} samples for '{target_feature}'")
        print(f"   Positive cases (PE=1): {len(self.data_frame[self.data_frame['label'] == 1]):,}")
        print(f"   Negative cases (PE=0): {len(self.data_frame[self.data_frame['label'] == 0]):,}")
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image path - handle both absolute and relative paths
        img_path = self.data_frame.iloc[idx, 0]
        
        # Try different path combinations
        possible_paths = [
            os.path.join(self.root_dir, img_path),
            img_path,  # Try absolute path
            os.path.join(self.root_dir, os.path.basename(img_path))
        ]
        
        image = None
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    image = Image.open(path).convert('RGB')
                    break
                except Exception as e:
                    continue
        
        if image is None:
            # Return black image as fallback
            print(f"Warning: Could not load image at index {idx}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
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


def create_densenet_model(num_classes=2, pretrained=True):
    """Create a DenseNet121 model for binary classification"""
    model = models.densenet121(pretrained=pretrained)
    
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
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in progress_bar:
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
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
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
        progress_bar = tqdm(dataloader, desc="Validating", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and probabilities
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            probabilities = torch.softmax(outputs, dim=1)
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_labels, all_probabilities


def calculate_metrics(predictions, labels, probabilities):
    """Calculate comprehensive classification metrics"""
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    auc = roc_auc_score(labels, probabilities)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'confusion_matrix': cm
    }


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path='training_curves.png'):
    """Plot and save training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(train_losses, label='Train Loss', marker='o')
    axes[0].plot(val_losses, label='Val Loss', marker='s')
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(train_accs, label='Train Acc', marker='o')
    axes[1].plot(val_accs, label='Val Acc', marker='s')
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    
    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No PE', 'PE'])
    ax.set_yticklabels(['No PE', 'PE'])
    
    # Text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{cm[i, j]}',
                          ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black",
                          fontsize=20, fontweight='bold')
    
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")


def main(args):
    """Main training function"""
    print("\n" + "="*80)
    print("PLEURAL EFFUSION BINARY CLASSIFIER TRAINING")
    print("="*80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Configuration
    print(f"\nConfiguration:")
    print(f"   Target: {args.target}")
    print(f"   Max training samples: {args.max_train_samples if args.max_train_samples > 0 else 'All available'}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Image root directory: {args.image_root}")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    print(f"\nLoading datasets...")
    max_samples = None if args.max_train_samples == 0 else args.max_train_samples
    
    train_dataset = ChestXRayDataset(
        args.train_csv, 
        args.image_root, 
        args.target, 
        transform=train_transform,
        balance_classes=True, 
        max_samples=max_samples
    )
    
    val_dataset = ChestXRayDataset(
        args.val_csv, 
        args.image_root, 
        args.target, 
        transform=val_transform,
        balance_classes=False,  # Keep natural distribution for validation
        max_samples=None
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    print(f"\nCreating DenseNet121 model...")
    model = create_densenet_model(num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training tracking
    best_val_acc = 0.0
    best_val_f1 = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Early stopping
    patience = 5
    patience_counter = 0
    
    print(f"\nStarting training...")
    print("="*80)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        
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
        print(f"\nResults:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"   Val Precision: {metrics['precision']:.4f}")
        print(f"   Val Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"   Val Specificity: {metrics['specificity']:.4f}")
        print(f"   Val F1-Score: {metrics['f1']:.4f}")
        print(f"   Val AUC-ROC: {metrics['auc']:.4f}")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = metrics['f1']
            patience_counter = 0
            
            model_filename = f'best_model_{args.target.lower().replace(" ", "_")}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'metrics': metrics,
                'target_feature': args.target,
                'train_config': vars(args)
            }, model_filename)
            print(f"\nNew best model saved! (Acc: {best_val_acc:.2f}%, F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"\nNo improvement for {patience_counter} epoch(s)")
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            break
    
    print("\n" + "="*80)
    print(f"Training completed!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Best validation F1-score: {best_val_f1:.4f}")
    print("="*80)
    
    # Plot training curves
    print(f"\nGenerating visualizations...")
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)
    
    # Load best model and evaluate
    print(f"\nFinal evaluation with best model...")
    model_filename = f'best_model_{args.target.lower().replace(" ", "_")}.pth'
    checkpoint = torch.load(model_filename, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    val_loss, val_acc, val_predictions, val_labels, val_probabilities = validate_epoch(
        model, val_loader, criterion, device
    )
    final_metrics = calculate_metrics(val_predictions, val_labels, val_probabilities)
    
    # Plot confusion matrix
    plot_confusion_matrix(final_metrics['confusion_matrix'])
    
    print(f"\n" + "="*80)
    print(f"FINAL EVALUATION RESULTS")
    print("="*80)
    print(f"Target: {args.target}")
    print(f"Accuracy:    {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)")
    print(f"Precision:   {final_metrics['precision']:.4f}")
    print(f"Recall:      {final_metrics['recall']:.4f} (Sensitivity)")
    print(f"Specificity: {final_metrics['specificity']:.4f}")
    print(f"F1-Score:    {final_metrics['f1']:.4f}")
    print(f"AUC-ROC:     {final_metrics['auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(final_metrics['confusion_matrix'])
    print(f"  TN: {final_metrics['confusion_matrix'][0,0]}  FP: {final_metrics['confusion_matrix'][0,1]}")
    print(f"  FN: {final_metrics['confusion_matrix'][1,0]}  TP: {final_metrics['confusion_matrix'][1,1]}")
    print(f"\nModel saved as: {model_filename}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train DenseNet121 for Pleural Effusion Binary Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 200 samples
  python train_classifier.py --max_train_samples 200 --epochs 10
  
  # Full training
  python train_classifier.py --max_train_samples 0 --epochs 50
  
  # Custom configuration
  python train_classifier.py --max_train_samples 5000 --epochs 30 --batch_size 64
        """
    )
    
    # Data arguments
    parser.add_argument('--train_csv', type=str, default='train_filtered.csv',
                       help='Path to training CSV file (default: train_filtered.csv)')
    parser.add_argument('--val_csv', type=str, default='valid_filtered.csv',
                       help='Path to validation CSV file (default: valid_filtered.csv)')
    parser.add_argument('--image_root', type=str, default='.',
                       help='Root directory for images (default: current directory)')
    
    # Training arguments
    parser.add_argument('--target', type=str, default='Pleural Effusion',
                       help='Target feature to classify (default: "Pleural Effusion")')
    parser.add_argument('--max_train_samples', type=int, default=200,
                       help='Maximum training samples (0 for all, default: 200)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    main(args)
