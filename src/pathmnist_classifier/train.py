"""Command-line training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from .config import load_config, with_output_dir
from .constants import CLASS_NAMES
from .data import create_dataloaders
from .engine import predict, train_one_epoch
from .metrics import classification_metrics
from .models import build_model
from .utils import resolve_device, save_json, save_yaml, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PathMNIST classifier using validation-only selection."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    parser.add_argument("--output-dir", help="Override config output_dir")
    parser.add_argument(
        "--no-download", action="store_true", help="Require data to exist locally"
    )
    return parser.parse_args()


def _is_improvement(value: float, best: float, monitor: str, min_delta: float) -> bool:
    if monitor == "loss":
        return value < best - min_delta
    return value > best + min_delta


def run_training(config: dict, download: bool = True) -> Path:
    """Train, select on validation data, and return the best checkpoint path."""
    seed_everything(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output_dir / "config.yaml")

    loaders = create_dataloaders(
        config["data"],
        seed=int(config["seed"]),
        splits=("train", "val"),
        download=download,
    )
    model = build_model(config["model"]).to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["training"].get("label_smoothing", 0.0))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    monitor = config["training"]["monitor"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min" if monitor == "loss" else "max", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_value = float("inf") if monitor == "loss" else float("-inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path = output_dir / "best_checkpoint.pt"

    print(f"Device: {device}")
    print(
        "Train/validation samples: "
        f"{len(loaders['train'].dataset)}/{len(loaders['val'].dataset)}"
    )
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {parameter_count:,}")

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler
        )
        val_output = predict(model, loaders["val"], criterion, device, "validate")
        val_metrics = classification_metrics(
            val_output["targets"], val_output["predictions"]  # type: ignore[arg-type]
        )
        val_metrics["loss"] = float(val_output["loss"])

        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row["learning_rate"] = optimizer.param_groups[0]["lr"]
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        monitored_value = val_metrics[monitor]
        scheduler.step(monitored_value)
        improved = _is_improvement(
            monitored_value,
            best_value,
            monitor,
            float(config["training"].get("min_delta", 0.0)),
        )
        if improved:
            best_value = monitored_value
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "class_names": CLASS_NAMES,
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d} | train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val acc {val_metrics['accuracy']:.4f} | "
            f"val macro-F1 {val_metrics['macro_f1']:.4f}"
        )

        if epochs_without_improvement >= int(config["training"]["patience"]):
            print(f"Early stopping after epoch {epoch}.")
            break

    save_json(
        {
            "selection_split": "validation",
            "monitor": monitor,
            "best_value": best_value,
            "best_checkpoint": str(checkpoint_path),
        },
        output_dir / "training_summary.json",
    )
    return checkpoint_path


def main() -> None:
    args = parse_args()
    config = with_output_dir(load_config(args.config), args.output_dir)
    run_training(config, download=not args.no_download)


if __name__ == "__main__":
    main()
