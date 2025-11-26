"""
Dataset classes for chest X-ray classification.
"""
import os
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)


class ChestXRayDataset(Dataset):
    """Dataset class for chest X-ray images"""

    def __init__(self, dataframe, root_dir, target_feature, transform=None,
                 balance_classes=True, max_samples=None):
        """
        Args:
            dataframe (DataFrame): Pandas dataframe with annotations
            root_dir (str): Directory with all the images
            target_feature (str): Name of the target feature column to use for classification
            transform (callable, optional): Optional transform to be applied on a sample
            balance_classes (bool): Whether to balance positive and negative classes
            max_samples (int, optional): Maximum number of samples to load. If None, loads all samples
        """
        self.data_frame = dataframe.copy()
        self.root_dir = root_dir
        self.transform = transform
        self.target_feature = target_feature

        # Check if target feature exists in the dataset
        if target_feature not in self.data_frame.columns:
            available_features = [
                col for col in self.data_frame.columns
                if col not in ['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA']
            ]
            raise ValueError(
                f"Target feature '{target_feature}' not found in dataset. "
                f"Available features: {available_features}"
            )

        # Filter out rows with missing target feature values
        self.data_frame = self.data_frame.dropna(subset=[target_feature])

        # Create binary labels: 1 for target = 1.0, 0 for target = 0.0
        self.data_frame = self.data_frame[self.data_frame[target_feature].isin([0.0, 1.0])]
        self.data_frame['label'] = self.data_frame[target_feature].astype(int)

        if balance_classes:
            self._balance_classes()

        # Limit the number of samples if max_samples is specified
        if max_samples is not None and len(self.data_frame) > max_samples:
            self._limit_samples(max_samples, balance_classes)

        self._print_summary()

    def _balance_classes(self):
        """Balance the dataset by sampling equal numbers of positive and negative cases"""
        pos_cases = self.data_frame[self.data_frame['label'] == 1]
        neg_cases = self.data_frame[self.data_frame['label'] == 0]

        min_samples = min(len(pos_cases), len(neg_cases))
        if min_samples > 0:
            pos_sampled = pos_cases.sample(n=min_samples, random_state=42)
            neg_sampled = neg_cases.sample(n=min_samples, random_state=42)
            self.data_frame = pd.concat([pos_sampled, neg_sampled]).reset_index(drop=True)

    def _limit_samples(self, max_samples, balance_classes):
        """Limit the number of samples while optionally maintaining class balance"""
        if balance_classes:
            pos_cases = self.data_frame[self.data_frame['label'] == 1]
            neg_cases = self.data_frame[self.data_frame['label'] == 0]

            # Calculate how many samples per class we can take
            samples_per_class = max_samples // 2
            if samples_per_class > 0:
                pos_limited = pos_cases.sample(
                    n=min(samples_per_class, len(pos_cases)),
                    random_state=42
                )
                neg_limited = neg_cases.sample(
                    n=min(samples_per_class, len(neg_cases)),
                    random_state=42
                )
                self.data_frame = pd.concat([pos_limited, neg_limited]).reset_index(drop=True)
            else:
                # If max_samples is very small, just take the first max_samples
                self.data_frame = self.data_frame.head(max_samples)
        else:
            # If not balancing, just take the first max_samples
            self.data_frame = self.data_frame.head(max_samples)

    def _print_summary(self):
        """Print dataset summary statistics"""
        logger.info(f"Dataset created with {len(self.data_frame)} samples for target: {self.target_feature}")
        logger.info(f"Positive cases ({self.target_feature}=1): {len(self.data_frame[self.data_frame['label'] == 1])}")
        logger.info(f"Negative cases ({self.target_feature}=0): {len(self.data_frame[self.data_frame['label'] == 0])}")

    def __len__(self):
        """Return the size of the dataset"""
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset

        Args:
            idx (int): Index of the sample

        Returns:
            tuple: (image, label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])

        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_name}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # Get the binary label
        label = self.data_frame.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label
