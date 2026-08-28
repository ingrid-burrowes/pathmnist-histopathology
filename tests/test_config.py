from pathlib import Path

import pytest

from pathmnist_classifier.config import load_config


def test_example_configs_are_valid() -> None:
    project_root = Path(__file__).parents[1]
    for name in ("cnn.yaml", "resnet18.yaml"):
        config = load_config(project_root / "configs" / name)
        assert config["data"]["image_size"] == 64
        assert config["training"]["monitor"] == "macro_f1"


def test_invalid_image_size_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
data: {image_size: 32, batch_size: 8}
model: {name: cnn}
training: {epochs: 1, learning_rate: 0.001, patience: 1, monitor: loss}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="image_size"):
        load_config(config_path)

