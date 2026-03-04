"""
Evaluation metrics helpers.
"""

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np


def compute_all_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    average: str = "macro",
) -> dict:
    """Compute F1-macro, precision, recall, and accuracy."""
    return {
        "f1_macro": round(float(f1_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }
