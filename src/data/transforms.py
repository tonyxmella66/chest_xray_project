"""
Data transformation and augmentation utilities.
"""
from torchvision import transforms
from config import (
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    RANDOM_HORIZONTAL_FLIP_P,
    RANDOM_ROTATION_DEGREES,
    COLOR_JITTER_BRIGHTNESS,
    COLOR_JITTER_CONTRAST
)


def get_train_transform():
    """
    Get training data transformation pipeline with augmentation

    Returns:
        torchvision.transforms.Compose: Training transforms
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=RANDOM_HORIZONTAL_FLIP_P),
        transforms.RandomRotation(RANDOM_ROTATION_DEGREES),
        transforms.ColorJitter(
            brightness=COLOR_JITTER_BRIGHTNESS,
            contrast=COLOR_JITTER_CONTRAST
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])


def get_val_transform():
    """
    Get validation/test data transformation pipeline without augmentation

    Returns:
        torchvision.transforms.Compose: Validation/test transforms
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])


def get_transforms():
    """
    Get both training and validation transforms

    Returns:
        tuple: (train_transform, val_transform)
    """
    return get_train_transform(), get_val_transform()
