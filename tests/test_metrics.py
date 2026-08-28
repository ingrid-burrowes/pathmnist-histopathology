import numpy as np

from pathmnist_classifier.metrics import classification_metrics, per_class_report


def test_perfect_predictions_have_perfect_metrics() -> None:
    targets = np.arange(9)
    metrics = classification_metrics(targets, targets.copy())
    report = per_class_report(targets, targets.copy())

    assert metrics == {"accuracy": 1.0, "macro_f1": 1.0}
    summary_keys = {"accuracy", "macro avg", "weighted avg"}
    assert len([key for key in report if key not in summary_keys]) == 9
