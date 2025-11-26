"""
DenseNet model definitions for chest X-ray classification.
"""
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, FREEZE_LAYERS


def create_densenet121(num_classes=NUM_CLASSES, freeze_layers=FREEZE_LAYERS):
    """
    Create a DenseNet-121 model for binary classification

    Args:
        num_classes (int): Number of output classes (default: 2 for binary classification)
        freeze_layers (bool): Whether to freeze early layers for faster fine-tuning

    Returns:
        torch.nn.Module: DenseNet-121 model
    """
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
