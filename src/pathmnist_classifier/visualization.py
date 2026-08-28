"""Plots for transparent model evaluation."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from .data import denormalize


def plot_confusion_matrices(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    """Save raw-count and row-normalized confusion matrices side by side."""
    labels = list(range(len(class_names)))
    raw = confusion_matrix(targets, predictions, labels=labels)
    normalized = confusion_matrix(
        targets, predictions, labels=labels, normalize="true"
    )
    short_names = [name.replace(" ", "\n") for name in class_names]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
    sns.heatmap(
        raw,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        ax=axes[0],
    )
    axes[0].set(
        title="Test confusion matrix (counts)", xlabel="Predicted", ylabel="True"
    )
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=short_names,
        yticklabels=short_names,
        ax=axes[1],
    )
    axes[1].set(
        title="Test confusion matrix (row-normalized)",
        xlabel="Predicted",
        ylabel="True",
    )
    for axis in axes:
        axis.tick_params(axis="x", rotation=45)
        axis.tick_params(axis="y", rotation=0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_representative_errors(
    dataset,
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    max_images: int = 16,
) -> None:
    """Plot the most confident test mistakes as auditable failure examples."""
    mistakes = np.flatnonzero(targets != predictions)
    if mistakes.size == 0:
        return
    confidence = probabilities[np.arange(len(predictions)), predictions]
    selected = mistakes[np.argsort(confidence[mistakes])[::-1][:max_images]]

    columns = min(4, len(selected))
    rows = math.ceil(len(selected) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes_array = np.atleast_1d(axes).ravel()

    for axis, index in zip(axes_array, selected, strict=False):
        image, _ = dataset[int(index)]
        image = denormalize(image).permute(1, 2, 0).cpu().numpy()
        axis.imshow(image)
        axis.set_title(
            f"True: {class_names[int(targets[index])]}\n"
            f"Pred: {class_names[int(predictions[index])]} "
            f"({confidence[index]:.1%})",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes_array[len(selected) :]:
        axis.axis("off")

    fig.suptitle("Highest-confidence test-set errors", fontsize=14)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
    """Convert one normalized CHW tensor into a plottable RGB array."""
    return denormalize(image).permute(1, 2, 0).detach().cpu().numpy()
