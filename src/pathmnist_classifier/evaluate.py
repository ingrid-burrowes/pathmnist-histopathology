"""Held-out test evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .constants import CLASS_NAMES
from .data import create_dataset
from .engine import predict
from .metrics import classification_metrics, per_class_report
from .models import build_model
from .utils import resolve_device, save_json, seed_everything
from .visualization import plot_confusion_matrices, plot_representative_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one checkpoint on PathMNIST test."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", help="Default: <checkpoint directory>/test")
    parser.add_argument("--data-root", help="Override the checkpoint's data root")
    parser.add_argument("--batch-size", type=int, help="Override evaluation batch size")
    parser.add_argument("--num-workers", type=int, help="Override data-loader workers")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    data_root: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    download: bool = True,
) -> dict[str, float]:
    """Evaluate a selected checkpoint once on the official held-out test split."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    seed_everything(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))

    root = data_root or config["data"]["root"]
    dataset = create_dataset(
        split="test",
        root=root,
        image_size=config["data"]["image_size"],
        augment=False,
        download=download,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size or config["data"]["batch_size"],
        shuffle=False,
        num_workers=(
            num_workers
            if num_workers is not None
            else config["data"].get("num_workers", 2)
        ),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model({**config["model"], "pretrained": False})
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    output = predict(model, loader, criterion, device, description="test")
    targets = np.asarray(output["targets"])
    predictions = np.asarray(output["predictions"])
    probabilities = np.asarray(output["probabilities"])

    scalar_metrics = classification_metrics(targets, predictions)
    scalar_metrics["loss"] = float(output["loss"])
    report = per_class_report(targets, predictions)

    destination = Path(output_dir) if output_dir else checkpoint_path.parent / "test"
    destination.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "split": "test",
            "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_validation_metrics": checkpoint["validation_metrics"],
            "test_metrics": scalar_metrics,
            "per_class": report,
        },
        destination / "metrics.json",
    )

    class_rows = {
        name: values for name, values in report.items() if name in CLASS_NAMES
    }
    pd.DataFrame.from_dict(class_rows, orient="index").to_csv(
        destination / "per_class_metrics.csv", index_label="class"
    )
    prediction_frame = pd.DataFrame(
        {
            "sample_index": np.arange(len(targets)),
            "true_index": targets,
            "true_class": [CLASS_NAMES[int(value)] for value in targets],
            "predicted_index": predictions,
            "predicted_class": [CLASS_NAMES[int(value)] for value in predictions],
            "confidence": probabilities.max(axis=1),
            "correct": targets == predictions,
        }
    )
    prediction_frame.to_csv(destination / "predictions.csv", index=False)
    plot_confusion_matrices(
        targets, predictions, CLASS_NAMES, destination / "confusion_matrix.png"
    )
    plot_representative_errors(
        dataset,
        targets,
        predictions,
        probabilities,
        CLASS_NAMES,
        destination / "representative_errors.png",
    )
    print(
        f"Test accuracy: {scalar_metrics['accuracy']:.4f} | "
        f"Test macro-F1: {scalar_metrics['macro_f1']:.4f}"
    )
    return scalar_metrics


def main() -> None:
    args = parse_args()
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=not args.no_download,
    )


if __name__ == "__main__":
    main()
