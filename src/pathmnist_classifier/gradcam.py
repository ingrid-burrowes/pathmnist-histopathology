"""Dependency-free Grad-CAM for qualitative model inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .constants import CLASS_NAMES
from .data import create_dataset
from .models import build_model, gradcam_target_layer
from .utils import resolve_device, seed_everything
from .visualization import tensor_to_rgb


class GradCAM:
    """Capture activations and gradients from a chosen convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(
            self._save_activations
        )
        self._backward_handle = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, _module, _inputs, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(
        self, image: torch.Tensor, class_index: int | None = None
    ) -> tuple[np.ndarray, int, float]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        probabilities = logits.softmax(dim=1)
        predicted = int(logits.argmax(dim=1).item())
        selected = predicted if class_index is None else class_index
        logits[0, selected].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmap = F.interpolate(
            heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        heatmap -= heatmap.min()
        heatmap /= heatmap.max().clamp_min(1e-8)
        confidence = float(probabilities[0, predicted].item())
        return heatmap.cpu().numpy(), predicted, confidence

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Grad-CAM for a PathMNIST sample."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--index", type=int, default=0, help="Index within the chosen split"
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--class-index", type=int, choices=range(len(CLASS_NAMES)))
    parser.add_argument("--output", help="Output PNG path")
    parser.add_argument("--data-root")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    seed_everything(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))
    dataset = create_dataset(
        split=args.split,
        root=args.data_root or config["data"]["root"],
        image_size=config["data"]["image_size"],
        augment=False,
        download=not args.no_download,
    )
    if not 0 <= args.index < len(dataset):
        raise IndexError(f"index must be between 0 and {len(dataset) - 1}")

    model = build_model({**config["model"], "pretrained": False})
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    image, target = dataset[args.index]
    input_tensor = image.unsqueeze(0).to(device)
    target_layer = gradcam_target_layer(model, config["model"]["name"])
    cam = GradCAM(model, target_layer)
    heatmap, predicted, confidence = cam(input_tensor, args.class_index)
    cam.close()

    rgb = tensor_to_rgb(image)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title(f"Input\nTrue: {CLASS_NAMES[int(np.asarray(target).item())]}")
    axes[1].imshow(heatmap, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(rgb)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.42, vmin=0, vmax=1)
    axes[2].set_title(f"Overlay\nPred: {CLASS_NAMES[predicted]} ({confidence:.1%})")
    for axis in axes:
        axis.axis("off")

    output = (
        Path(args.output)
        if args.output
        else checkpoint_path.parent / f"gradcam_{args.split}_{args.index}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
