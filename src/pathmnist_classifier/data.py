"""PathMNIST datasets, transforms, and deterministic data loaders."""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from medmnist import PathMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

from .constants import IMAGENET_MEAN, IMAGENET_STD


def build_transform(augment: bool) -> transforms.Compose:
    """Build morphology-preserving train or evaluation transforms."""
    operations: list[Any] = []
    if augment:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


def create_dataset(
    split: str,
    root: str | Path,
    image_size: int,
    augment: bool = False,
    download: bool = True,
) -> PathMNIST:
    """Create one official PathMNIST split without resampling or leakage."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be 'train', 'val', or 'test'")
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    return PathMNIST(
        split=split,
        root=str(root_path),
        size=image_size,
        transform=build_transform(augment=augment),
        download=download,
        as_rgb=True,
    )


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(
    data_config: dict[str, Any],
    seed: int,
    splits: Iterable[str] = ("train", "val"),
    download: bool = True,
) -> dict[str, DataLoader]:
    """Create deterministic loaders for the requested official splits."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    loaders: dict[str, DataLoader] = {}

    for split in splits:
        dataset = create_dataset(
            split=split,
            root=data_config["root"],
            image_size=data_config["image_size"],
            augment=bool(data_config.get("augment", True) and split == "train"),
            download=download,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=data_config["batch_size"],
            shuffle=split == "train",
            num_workers=data_config.get("num_workers", 2),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=data_config.get("num_workers", 2) > 0,
            worker_init_fn=_seed_worker,
            generator=generator,
        )
    return loaders


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for plotting."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    if images.ndim == 3:
        return (images.unsqueeze(0) * std + mean).squeeze(0).clamp(0, 1)
    return (images * std + mean).clamp(0, 1)

