import pytest
import torch

from pathmnist_classifier.models import build_model, gradcam_target_layer


@pytest.mark.parametrize("name", ["cnn", "resnet18"])
def test_model_output_shape(name: str) -> None:
    model = build_model({"name": name, "pretrained": False, "dropout": 0.2})
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 9)


@pytest.mark.parametrize("name", ["cnn", "resnet18"])
def test_gradcam_target_is_a_module(name: str) -> None:
    model = build_model({"name": name, "pretrained": False})
    assert isinstance(gradcam_target_layer(model, name), torch.nn.Module)

