import numpy as np
import pytest
import torch

from pathmnist_classifier.gradcam import GradCAM
from pathmnist_classifier.models import build_model, gradcam_target_layer


@pytest.mark.parametrize("name", ["cnn", "resnet18"])
def test_gradcam_returns_normalized_heatmap(name: str) -> None:
    model = build_model({"name": name, "pretrained": False})
    cam = GradCAM(model, gradcam_target_layer(model, name))
    heatmap, prediction, confidence = cam(torch.randn(1, 3, 64, 64))
    cam.close()

    assert heatmap.shape == (64, 64)
    assert np.isfinite(heatmap).all()
    assert 0 <= heatmap.min() <= heatmap.max() <= 1
    assert 0 <= prediction < 9
    assert 0 <= confidence <= 1

