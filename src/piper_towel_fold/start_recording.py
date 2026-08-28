import argparse
import json
from pathlib import Path

def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as config_file:
        data = json.load(config_file)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")
    return data


def cameras_to_csv(config: dict, key: str = "cameras") -> tuple[str, str]:
    cameras = config.get(key, [])
    if not cameras:
        return "", ""
    if not isinstance(cameras, list):
        raise ValueError(f"'{key}' must be a list.")

    camera_names: list[str] = []
    camera_refs: list[str] = []
    for index, camera in enumerate(cameras):
        if not isinstance(camera, dict):
            raise ValueError(f"Camera entry #{index} must be an object.")
        name = camera.get("name")
        ref = camera.get("ref")
        if not name or ref is None:
            raise ValueError(f"Camera entry #{index} must contain both 'name' and 'ref'.")
        camera_names.append(str(name))
        camera_refs.append(str(ref))

    return ",".join(camera_refs), ",".join(camera_names)


SKIP_CONFIG_KEYS = {"cameras", "training", "policy_live", "preprocessing"}


def build_namespace(config: dict) -> argparse.Namespace:
    from .record_episode import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args([])

    camera_indices, camera_names = cameras_to_csv(config, "recording_cameras")
    if not camera_indices:
        camera_indices, camera_names = cameras_to_csv(config)
    if camera_indices:
        config["camera_indices"] = camera_indices
        config["camera_names"] = camera_names

    for key, value in config.items():
        if key in SKIP_CONFIG_KEYS:
            continue
        if hasattr(args, key):
            setattr(args, key, value)

    args.preprocessing = config.get("preprocessing")
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Piper data collection from a JSON config file.")
    parser.add_argument(
        "--config",
        default="configs/record_pick_cube.json",
        help="Path to the JSON config file.",
    )
    parsed = parser.parse_args()

    config_path = Path(parsed.config)
    args = build_namespace(load_config(config_path))

    from .record_episode import run_recording

    run_recording(args)


if __name__ == "__main__":
    main()
