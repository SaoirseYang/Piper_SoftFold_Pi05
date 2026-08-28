import argparse
import json
from pathlib import Path
from typing import Any

from .start_recording import cameras_to_csv


TOP_LEVEL_KEYS = {
    "task",
    "repo_id",
    "fps",
    "duration",
    "follower_left_can",
    "follower_right_can",
    "camera_width",
    "camera_height",
    "camera_fps",
}


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        data = json.load(config_file)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")
    return data


def default_policy_path(config: dict[str, Any]) -> str | None:
    training = config.get("training", {})
    if not isinstance(training, dict):
        return None

    output_dir = training.get("output_dir")
    if output_dir:
        return str(Path(str(output_dir)) / "checkpoints" / "last" / "pretrained_model")

    job_name = training.get("job_name")
    policy_type = training.get("policy_type", "act")
    repo_id = config.get("repo_id")
    if job_name is None and repo_id:
        job_name = f"{policy_type}_{str(repo_id).split('/')[-1]}"
    if job_name:
        return str(Path("outputs") / "train" / str(job_name) / "checkpoints" / "last" / "pretrained_model")

    return None


def default_dataset_root(config: dict[str, Any]) -> str | None:
    repo_id = config.get("repo_id")
    if not repo_id:
        return None
    return str(Path(str(config.get("root", "data/lerobot"))) / str(repo_id))


def build_namespace(config: dict[str, Any]) -> argparse.Namespace:
    from .run_policy_live import build_arg_parser

    args = build_arg_parser().parse_args([])

    can_keys = {
        "follower_left_can",
        "follower_right_can",
        "leader_left_can",
        "leader_right_can",
    }
    for key in TOP_LEVEL_KEYS:
        if key not in config:
            continue
        value = config[key]
        # JSON null must disable a CAN port; otherwise argparse defaults (e.g. can2) stick.
        if key in can_keys:
            setattr(args, key, None if value in (None, "") else value)
        elif value is not None:
            setattr(args, key, value)

    dataset_root = default_dataset_root(config)
    if dataset_root is not None:
        args.dataset_root = dataset_root

    policy_path = default_policy_path(config)
    if policy_path is not None:
        args.policy_path = policy_path

    policy_live = config.get("policy_live", {})
    if policy_live is None:
        policy_live = {}
    if not isinstance(policy_live, dict):
        raise ValueError("'policy_live' must be an object when present.")

    camera_indices, camera_names = cameras_to_csv(policy_live)
    if not camera_indices:
        camera_indices, camera_names = cameras_to_csv(config)
    if camera_indices:
        args.camera_indices = camera_indices
        args.camera_names = camera_names

    for key, value in policy_live.items():
        if key in {"root", "cameras"}:
            continue
        if not hasattr(args, key):
            raise ValueError(f"Unsupported policy_live config key: {key}")
        setattr(args, key, value)

    if "repo_id" in policy_live and "dataset_root" not in policy_live:
        root = policy_live.get("root", config.get("root", "data/lerobot"))
        args.dataset_root = str(Path(str(root)) / str(policy_live["repo_id"]))

    preprocessing = config.get("preprocessing")
    if preprocessing is None:
        preprocessing = policy_live.get("preprocessing")
    args.preprocessing = preprocessing

    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live Piper policy from a JSON config file.")
    parser.add_argument(
        "--config",
        default="configs/record_pick_cube.json",
        help="Path to the JSON config file.",
    )
    parsed = parser.parse_args()

    from .run_policy_live import run_live_policy

    run_live_policy(build_namespace(load_config(Path(parsed.config))))


if __name__ == "__main__":
    main()
