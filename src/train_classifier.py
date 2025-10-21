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
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

def create_densenet_model(num_classes=2):
    """Create a DenseNet model for binary classification"""
    model = models.densenet121(pretrained=True)
    
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

def main(target_feature="Pleural Effusion", max_train_samples=200, num_epochs=20, batch_size=32, learning_rate=0.001):
    """
    Main training function
    
    Args:
        target_feature (str): Name of the target feature to classify
        max_train_samples (int): Maximum number of training samples to load. 
                                If None, loads all available samples.
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Data paths
    train_csv = "data/train_filtered.csv"
    val_csv = "data/valid_filtered.csv"
    train_root = "data/train"
    val_root = "data/valid"
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    print(f"Loading training data for target '{target_feature}' (max {max_train_samples} samples)...")
    train_dataset = ChestXRayDataset(train_csv, train_root, target_feature, transform=train_transform, 
                                   balance_classes=True, max_samples=max_train_samples)
    print(f"Loading validation data for target '{target_feature}'...")
    val_dataset = ChestXRayDataset(val_csv, val_root, target_feature, transform=val_transform, 
                                 balance_classes=False, max_samples=None)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'metrics': metrics,
                'target_feature': target_feature
            }, f'best_densenet_{target_feature.lower().replace(" ", "_")}.pth')
            print("New best model saved!")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
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
    plt.plot([optimizer.param_groups[0]['lr']] * num_epochs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Load best model and evaluate
    print("\nLoading best model for final evaluation...")
    model_filename = f'best_densenet_{target_feature.lower().replace(" ", "_")}.pth'
    checkpoint = torch.load(model_filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    val_loss, val_acc, val_predictions, val_labels, val_probabilities = validate_epoch(
        model, val_loader, criterion, device
    )
    final_metrics = calculate_metrics(val_predictions, val_labels, val_probabilities)
    
    print(f"\nFinal Evaluation Results for {target_feature}:")
    print(f"Validation Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Validation Precision: {final_metrics['precision']:.4f}")
    print(f"Validation Recall: {final_metrics['recall']:.4f}")
    print(f"Validation F1-Score: {final_metrics['f1']:.4f}")
    print(f"Validation AUC: {final_metrics['auc']:.4f}")
    print(f"Model saved as: {model_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DenseNet for Chest X-Ray Medical Condition Classification')
    parser.add_argument('--target', type=str, default='Pleural Effusion',
                       help='Target feature to classify (default: "Pleural Effusion")')
    parser.add_argument('--max_train_samples', type=int, default=200,
                       help='Maximum number of training samples to load (default: 200, use 0 for all samples)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    # Convert 0 to None for loading all samples
    max_samples = None if args.max_train_samples == 0 else args.max_train_samples
    
    print(f"Training configuration:")
    print(f"  Target feature: {args.target}")
    print(f"  Max training samples: {max_samples if max_samples else 'All available'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print("-" * 50)
    
    main(target_feature=args.target,
         max_train_samples=max_samples, 
         num_epochs=args.epochs, 
         batch_size=args.batch_size, 
         learning_rate=args.learning_rate)

