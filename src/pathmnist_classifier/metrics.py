"""Classification metrics used consistently during validation and testing."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .constants import CLASS_NAMES


def classification_metrics(
    targets: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    """Calculate scalar metrics used for monitoring and summary reporting."""
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
    }


def per_class_report(
    targets: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    """Return precision, recall, F1, and support for all nine tissue classes."""
    return classification_report(
        targets,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

