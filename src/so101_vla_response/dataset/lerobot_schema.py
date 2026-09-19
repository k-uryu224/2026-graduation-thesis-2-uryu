from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(
    "configs/dataset/so101_top_wrist.yaml"
)


def load_lerobot_dataset_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"Dataset config must be a mapping: {path}"
        )

    return config


def build_lerobot_features(
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    images = config["images"]

    height = int(images["height"])
    width = int(images["width"])

    features: dict[str, dict[str, Any]] = {}

    for camera in config["cameras"].values():
        key = camera["lerobot_key"]

        visual_dtype = (
            "video"
            if bool(images.get("use_videos", False))
            else "image"
        )

        features[key] = {
            "dtype": visual_dtype,
            "shape": (height, width, 3),
            "names": [
                "height",
                "width",
                "channel",
            ],
        }

    state_names = list(config["state"]["names"])
    action_names = list(config["action"]["names"])

    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(state_names),),
        "names": state_names,
    }

    features["action"] = {
        "dtype": "float32",
        "shape": (len(action_names),),
        "names": action_names,
    }

    return features
