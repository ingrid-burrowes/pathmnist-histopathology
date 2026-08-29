"""Custom CNN and transfer-learning model definitions."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .constants import NUM_CLASSES


class HistologyCNN(nn.Module):
    """Compact CNN written for 64x64 RGB histopathology patches."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> None:
        super().__init__()
        self.block1 = self._block(3, 32)
        self.block2 = self._block(32, 64)
        self.block3 = self._block(64, 128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.block1(inputs)
        features = self.block2(features)
        features = self.block3(features)
        return self.classifier(self.pool(features))


def build_model(model_config: dict, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Build a model from the experiment configuration."""
    name = model_config["name"]
    dropout = float(model_config.get("dropout", 0.3))

    if name == "cnn":
        return HistologyCNN(num_classes=num_classes, dropout=dropout)

    if name == "resnet18":
        weights = (
            ResNet18_Weights.DEFAULT if model_config.get("pretrained", True) else None
        )
        model = resnet18(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )
        return model

    raise ValueError(f"Unknown model: {name}")


def gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the final spatial feature layer for Grad-CAM."""
    if model_name == "cnn":
        if not isinstance(model, HistologyCNN):
            raise TypeError("Expected HistologyCNN for model_name='cnn'")
        return model.block3[3]
    if model_name == "resnet18":
        # PathMNIST images are only 64x64. ResNet's final stage produces a 2x2
        # feature map, which is too coarse for useful localization. The preceding
        # 4x4 stage offers a better spatial/semantic trade-off for Grad-CAM.
        return model.layer3[-1]  # type: ignore[attr-defined]
    raise ValueError(f"Unknown model: {model_name}")
