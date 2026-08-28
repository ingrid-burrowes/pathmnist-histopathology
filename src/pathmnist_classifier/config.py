"""Configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = {"data", "model", "training"}
VALID_MODELS = {"cnn", "resnet18"}
VALID_MONITORS = {"loss", "accuracy", "macro_f1"}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration and validate important fields."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    missing = REQUIRED_SECTIONS - config.keys()
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    if config["model"].get("name") not in VALID_MODELS:
        raise ValueError(f"model.name must be one of {sorted(VALID_MODELS)}")

    if config["training"].get("monitor") not in VALID_MONITORS:
        raise ValueError(f"training.monitor must be one of {sorted(VALID_MONITORS)}")

    image_size = config["data"].get("image_size")
    if image_size not in {28, 64, 128, 224}:
        raise ValueError("data.image_size must be one of 28, 64, 128, or 224")

    for section, key in [
        ("data", "batch_size"),
        ("training", "epochs"),
        ("training", "learning_rate"),
        ("training", "patience"),
    ]:
        if config[section].get(key, 0) <= 0:
            raise ValueError(f"{section}.{key} must be positive")

    return config


def with_output_dir(config: dict[str, Any], output_dir: str | None) -> dict[str, Any]:
    """Return a copied config with an optional command-line output override."""
    result = deepcopy(config)
    if output_dir is not None:
        result["output_dir"] = output_dir
    if not result.get("output_dir"):
        raise ValueError("output_dir is required in the config or command line")
    return result

