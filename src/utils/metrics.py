"""
Metrics calculation utilities.
"""
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)


def calculate_metrics(predictions, labels, probabilities):
    """
    Calculate comprehensive classification metrics

    Args:
        predictions (array-like): Predicted class labels
        labels (array-like): True class labels
        probabilities (array-like): Predicted probabilities for positive class

    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1, and auc
    """
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    auc = roc_auc_score(labels, probabilities)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
