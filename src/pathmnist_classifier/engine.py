"""Training and inference loops."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import classification_metrics


def _targets_to_vector(targets: torch.Tensor, device: torch.device) -> torch.Tensor:
    return targets.view(-1).long().to(device, non_blocking=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    """Train for one epoch and return loss, accuracy, and macro-F1."""
    model.train()
    total_loss = 0.0
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    use_amp = device.type == "cuda"

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = _targets_to_vector(targets, device)
        optimizer.zero_grad(set_to_none=True)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with amp_context:
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        all_targets.append(targets.detach().cpu().numpy())
        all_predictions.append(logits.argmax(dim=1).detach().cpu().numpy())

    targets_np = np.concatenate(all_targets)
    predictions_np = np.concatenate(all_predictions)
    metrics = classification_metrics(targets_np, predictions_np)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    description: str = "evaluate",
) -> dict[str, np.ndarray | float]:
    """Run inference once and return loss, labels, predictions, and probabilities."""
    model.eval()
    total_loss = 0.0
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for images, targets in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        targets = _targets_to_vector(targets, device)
        logits = model(images)
        loss = criterion(logits, targets)
        probabilities = logits.softmax(dim=1)

        total_loss += loss.item() * images.size(0)
        all_targets.append(targets.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    targets_np = np.concatenate(all_targets)
    probabilities_np = np.concatenate(all_probabilities)
    predictions_np = probabilities_np.argmax(axis=1)
    return {
        "loss": total_loss / len(loader.dataset),
        "targets": targets_np,
        "predictions": predictions_np,
        "probabilities": probabilities_np,
    }
