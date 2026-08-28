from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_dataset_info(dataset_root: str | Path) -> dict[str, Any]:
    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset info not found: {info_path}")
    with info_path.open("r", encoding="utf-8") as info_file:
        info = json.load(info_file)
    if not isinstance(info, dict):
        raise ValueError(f"Invalid dataset info format: {info_path}")
    return info


def observation_features_from_dataset(dataset_root: str | Path) -> dict[str, dict]:
    features = load_dataset_info(dataset_root).get("features", {})
    if not isinstance(features, dict):
        raise ValueError("Dataset info.features must be an object.")

    observation_features = {
        key: value
        for key, value in features.items()
        if key.startswith("observation.") and isinstance(value, dict)
    }
    if not observation_features:
        raise ValueError(f"No observation.* features found under {dataset_root}")
    return observation_features


def action_names_from_dataset(dataset_root: str | Path) -> list[str]:
    features = load_dataset_info(dataset_root).get("features", {})
    action_feature = features.get("action")
    if not isinstance(action_feature, dict):
        raise ValueError("Dataset info.features.action is missing.")
    names = action_feature.get("names")
    if not isinstance(names, list) or not names:
        raise ValueError("Dataset action feature names are missing.")
    return [str(name) for name in names]
