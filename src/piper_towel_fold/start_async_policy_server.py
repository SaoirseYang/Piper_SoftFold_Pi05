import argparse
import json
from pathlib import Path
from typing import Any

from .offline_infer import _parse_optional_bool
from .start_policy_live import default_dataset_root, default_policy_path
from .async_rtc import resolve_rtc_options


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        data = json.load(config_file)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")
    return data


def build_server_config(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    try:
        from lerobot.async_inference.configs import PolicyServerConfig
    except ImportError as exc:
        raise ImportError(
            "LeRobot async inference is not installed. "
            'Install with: pip install "lerobot[async,pi]"'
        ) from exc

    server_cfg = config.get("policy_server", {})
    if server_cfg is None:
        server_cfg = {}
    if not isinstance(server_cfg, dict):
        raise ValueError("'policy_server' must be an object when present.")

    policy_live = config.get("policy_live", {})
    if not isinstance(policy_live, dict):
        policy_live = {}

    kwargs = {
        "host": server_cfg.get("host", "0.0.0.0"),
        "port": server_cfg.get("port", 8080),
        "fps": int(server_cfg.get("fps", policy_live.get("fps", config.get("fps", 5)))),
        "inference_latency": server_cfg.get("inference_latency"),
        "obs_queue_timeout": server_cfg.get("obs_queue_timeout", 10),
    }
    if kwargs["inference_latency"] is None:
        kwargs["inference_latency"] = 1.0 / float(kwargs["fps"])

    inference_options = {
        "inference_dtype": server_cfg.get(
            "inference_dtype", policy_live.get("inference_dtype")
        ),
        "compile_model": _parse_optional_bool(
            server_cfg.get("compile_model", policy_live.get("compile_model"))
        ),
        "num_inference_steps": server_cfg.get(
            "num_inference_steps", policy_live.get("num_inference_steps")
        ),
        "rtc_options": resolve_rtc_options(config),
    }
    return PolicyServerConfig(**kwargs), inference_options


def build_preload_policy(config: dict[str, Any]) -> dict[str, Any] | None:
    server_cfg = config.get("policy_server", {})
    if not isinstance(server_cfg, dict):
        return None
    if not server_cfg.get("preload_at_startup", False):
        return None

    policy_live = config.get("policy_live", {})
    async_cfg = config.get("async_inference", {})
    training = config.get("training", {})
    if not isinstance(policy_live, dict):
        policy_live = {}
    if not isinstance(async_cfg, dict):
        async_cfg = {}
    if not isinstance(training, dict):
        training = {}

    policy_path = (
        server_cfg.get("policy_path")
        or policy_live.get("policy_path")
        or default_policy_path(config)
    )
    dataset_root = (
        server_cfg.get("dataset_root")
        or policy_live.get("dataset_root")
        or default_dataset_root(config)
    )
    if policy_path is None or dataset_root is None:
        raise ValueError(
            "policy_server.preload_at_startup=true requires policy_path and dataset_root "
            "(set in policy_server, policy_live, or training/repo_id)."
        )

    return {
        "policy_type": (
            server_cfg.get("policy_type")
            or async_cfg.get("policy_type")
            or training.get("policy_type", "pi05")
        ),
        "policy_path": str(policy_path),
        "dataset_root": str(dataset_root),
        "device": server_cfg.get("policy_device", async_cfg.get("policy_device", "cuda")),
        "actions_per_chunk": int(
            server_cfg.get(
                "actions_per_chunk",
                async_cfg.get("actions_per_chunk", 50),
            )
        ),
        "rename_map": (
            server_cfg.get("rename_map")
            or async_cfg.get("rename_map")
            or training.get("rename_map")
            or {}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Piper async policy server from JSON config.")
    parser.add_argument(
        "--config",
        default="configs/softfold_piper_pi05.json",
        help="Path to the JSON config file.",
    )
    args = parser.parse_args()

    from .async_policy_server import serve_piper_policy_server
    from .local_tokenizer import install

    install()
    server_config, inference_options = build_server_config(load_config(Path(args.config)))
    preload_policy = build_preload_policy(load_config(Path(args.config)))
    serve_piper_policy_server(
        server_config,
        preload_policy=preload_policy,
        **inference_options,
    )


if __name__ == "__main__":
    main()
