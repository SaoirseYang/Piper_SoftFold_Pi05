import argparse
import json
from pathlib import Path
from typing import Any

from .start_policy_live import default_dataset_root, default_policy_path
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


def apply_config_to_namespace(args: argparse.Namespace, config_path: Path) -> argparse.Namespace:
    config = load_config(config_path)

    for key in TOP_LEVEL_KEYS:
        if key in config and config[key] is not None:
            setattr(args, key, config[key])

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

    for key, value in policy_live.items():
        if key in {"root", "device", "inference_dtype", "compile_model", "num_inference_steps"}:
            continue
        if not hasattr(args, key):
            continue
        setattr(args, key, value)

    if "repo_id" in policy_live and "dataset_root" not in policy_live:
        root = policy_live.get("root", config.get("root", "data/lerobot"))
        args.dataset_root = str(Path(str(root)) / str(policy_live["repo_id"]))
    if "dataset_root" in policy_live:
        args.dataset_root = policy_live["dataset_root"]
    if "policy_path" in policy_live:
        args.policy_path = policy_live["policy_path"]

    async_cfg = config.get("async_inference", {})
    if async_cfg is None:
        async_cfg = {}
    if not isinstance(async_cfg, dict):
        raise ValueError("'async_inference' must be an object when present.")

    camera_indices, camera_names = cameras_to_csv(async_cfg)
    if not camera_indices:
        camera_indices, camera_names = cameras_to_csv(policy_live)
    if not camera_indices:
        camera_indices, camera_names = cameras_to_csv(config)
    if camera_indices:
        args.camera_indices = camera_indices
        args.camera_names = camera_names

    async_key_map = {
        "server_address": "server_address",
        "policy_type": "policy_type",
        "policy_path": "policy_path",
        "policy_device": "policy_device",
        "actions_per_chunk": "actions_per_chunk",
        "chunk_size_threshold": "chunk_size_threshold",
        "aggregate_fn_name": "aggregate_fn_name",
        "execute": "execute",
        "duration": "duration",
        "fps": "fps",
        "control_speed": "control_speed",
        "max_joint_step_rad": "max_joint_step_rad",
        "max_gripper_step_m": "max_gripper_step_m",
        "gripper_effort": "gripper_effort",
        "smoothing_alpha": "smoothing_alpha",
        "hold_last_action_on_idle": "hold_last_action_on_idle",
        "print_every": "print_every",
        "log_jsonl": "log_jsonl",
        "debug_visualize_queue_size": "debug_visualize_queue_size",
        "network_benchmark_samples": "network_benchmark_samples",
        "log_latency": "log_latency",
        "latency_log_jsonl": "latency_log_jsonl",
        "obs_image_compression": "obs_image_compression",
        "obs_jpeg_quality": "obs_jpeg_quality",
    }
    for config_key, attr_name in async_key_map.items():
        if config_key in async_cfg and hasattr(args, attr_name):
            setattr(args, attr_name, async_cfg[config_key])

    training = config.get("training", {})
    if isinstance(training, dict) and "policy_type" in training and not async_cfg.get("policy_type"):
        args.policy_type = training["policy_type"]

    rename_map = (
        async_cfg.get("rename_map")
        or policy_live.get("rename_map")
        or (training.get("rename_map") if isinstance(training, dict) else None)
        or {}
    )
    args.rename_map = rename_map if isinstance(rename_map, dict) else {}

    preprocessing = config.get("preprocessing")
    if preprocessing is None:
        preprocessing = policy_live.get("preprocessing")
    args.preprocessing = preprocessing

    return args


def main() -> None:
    from .async_robot_client import build_arg_parser, run_async_live_policy

    parser = build_arg_parser()
    parser.add_argument(
        "--config",
        default="configs/softfold_piper_pi05.json",
        help="Path to the JSON config file.",
    )
    args = parser.parse_args()
    apply_config_to_namespace(args, Path(args.config))
    run_async_live_policy(args)


if __name__ == "__main__":
    main()
