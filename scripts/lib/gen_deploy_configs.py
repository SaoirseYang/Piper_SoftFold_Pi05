#!/usr/bin/env python3
"""Generate SoftFold async-inference deploy instance configs + manifest.

Used by scripts/deploy_async_inference.sh. Not a user-facing CLI entry.

Inputs (either):
  - CLI: --ckpts / --steps / --chunk / --alpha
  - YAML: --deploy-yaml  (checkpoints+grid 或 instances 列表)
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from itertools import product
from pathlib import Path
from typing import Any


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _resolve_ckpt(item: str | int, training_output: str) -> str:
    text = str(item).strip()
    if text in ("last", "LAST"):
        return f"{training_output}/checkpoints/last/pretrained_model"
    if re.fullmatch(r"\d+", text):
        return f"{training_output}/checkpoints/{text}/pretrained_model"
    return text


def _ckpt_token(item: Any) -> str:
    """Normalize a YAML/CLI checkpoint token to a string step name or path."""
    if isinstance(item, bool):
        raise SystemExit(f"Invalid checkpoint token (bool): {item!r}")
    if isinstance(item, int):
        # Bare YAML ints are fine for 100000; leading-zero steps must be quoted in YAML
        # (otherwise YAML 1.1 octal: 050000 -> 20480). We keep decimal int as digits.
        return str(item)
    return str(item).strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required for --deploy-yaml. Install with: pip install pyyaml"
        ) from exc

    class DeployLoader(yaml.SafeLoader):
        pass

    def _int_keep_leading_zero_strings(loader: Any, node: Any) -> Any:
        # YAML 1.1 treats 050000 as octal(20480). Checkpoint step folders are decimal
        # digit strings — keep the written scalar when it has a leading zero.
        value = loader.construct_scalar(node)
        if re.fullmatch(r"0\d+", value):
            print(
                f"[deploy yaml] treating {value!r} as checkpoint step string "
                f"(not octal). Prefer writing \"{value}\" explicitly.",
                flush=True,
            )
            return value
        try:
            return int(value, 10)
        except ValueError:
            return int(value, 0)

    DeployLoader.add_constructor("tag:yaml.org,2002:int", _int_keep_leading_zero_strings)

    data = yaml.load(path.read_text(encoding="utf-8"), Loader=DeployLoader)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"Deploy YAML must be a mapping: {path}")
    return data


def _parse_ckpts(raw: str, base: dict[str, Any], training_output: str) -> list[str]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    out = [_resolve_ckpt(item, training_output) for item in items]
    if out:
        return out
    policy_live = base.get("policy_live") or {}
    path = policy_live.get("policy_path")
    if path:
        return [str(path)]
    return [f"{training_output}/checkpoints/last/pretrained_model"]


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return (s or "inst")[:max_len]


def _expand_from_yaml(
    deploy: dict[str, Any],
    *,
    base: dict[str, Any],
    training_output: str,
    defaults: dict[str, Any],
    default_base_config: str,
) -> list[dict[str, Any]]:
    """Return list of instance combo dicts (policy_path + overlays)."""
    explicit = deploy.get("instances")
    if isinstance(explicit, list) and explicit:
        combos: list[dict[str, Any]] = []
        for row in explicit:
            if not isinstance(row, dict):
                raise SystemExit(f"instances[] entries must be objects, got: {row!r}")

            row_base_path = str(row.get("base_config") or default_base_config)
            row_base = base
            row_training_output = training_output
            row_defaults = defaults
            if row_base_path and Path(row_base_path).resolve() != Path(default_base_config).resolve():
                if not Path(row_base_path).is_file():
                    raise SystemExit(f"instance base_config not found: {row_base_path}")
                row_base = json.loads(Path(row_base_path).read_text(encoding="utf-8"))
                row_training = row_base.get("training") or {}
                row_training_output = str(
                    row_training.get("output_dir") or training_output
                )
                row_ps = row_base.get("policy_server") or {}
                row_pl = row_base.get("policy_live") or {}
                row_ai = row_base.get("async_inference") or {}
                row_defaults = {
                    "num_inference_steps": int(
                        row_ps.get(
                            "num_inference_steps",
                            row_pl.get("num_inference_steps", defaults["num_inference_steps"]),
                        )
                    ),
                    "chunk_size_threshold": float(
                        row_ai.get("chunk_size_threshold", defaults["chunk_size_threshold"])
                    ),
                    "smoothing_alpha": float(
                        row_ai.get("smoothing_alpha", defaults["smoothing_alpha"])
                    ),
                }

            ckpt_raw = row.get("checkpoint", row.get("ckpt", row.get("policy_path", "last")))
            combo: dict[str, Any] = {
                "name": row.get("name"),
                "base_config": row_base_path,
                "policy_path": _resolve_ckpt(_ckpt_token(ckpt_raw), row_training_output),
                "num_inference_steps": int(
                    row.get("num_inference_steps", row_defaults["num_inference_steps"])
                ),
                "chunk_size_threshold": float(
                    row.get("chunk_size_threshold", row_defaults["chunk_size_threshold"])
                ),
                "smoothing_alpha": float(
                    row.get("smoothing_alpha", row_defaults["smoothing_alpha"])
                ),
            }
            if "aggregate_fn_name" in row:
                combo["aggregate_fn_name"] = str(row["aggregate_fn_name"])
            if "rtc_enabled" in row:
                combo["rtc_enabled"] = bool(row["rtc_enabled"])
            if "rtc" in row:
                if row["rtc"] is not None and not isinstance(row["rtc"], dict):
                    raise SystemExit("instances[].rtc must be a mapping or null")
                combo["rtc"] = row["rtc"]
            combos.append(combo)
        return combos

    ckpt_items = _as_list(deploy.get("checkpoints", deploy.get("ckpts", ["last"])))
    if not ckpt_items:
        ckpt_items = ["last"]
    ckpts = [_resolve_ckpt(_ckpt_token(x), training_output) for x in ckpt_items]

    grid = deploy.get("grid") or {}
    if not isinstance(grid, dict):
        raise SystemExit("'grid' in deploy YAML must be a mapping")

    steps = [int(x) for x in _as_list(grid.get("num_inference_steps", [defaults["num_inference_steps"]]))]
    chunks = [
        float(x)
        for x in _as_list(grid.get("chunk_size_threshold", [defaults["chunk_size_threshold"]]))
    ]
    alphas = [float(x) for x in _as_list(grid.get("smoothing_alpha", [defaults["smoothing_alpha"]]))]

    return [
        {
            "name": None,
            "base_config": default_base_config,
            "policy_path": ckpt,
            "num_inference_steps": step,
            "chunk_size_threshold": chunk,
            "smoothing_alpha": alpha,
        }
        for ckpt, step, chunk, alpha in product(ckpts, steps, chunks, alphas)
    ]


def _expand_from_cli(
    args: argparse.Namespace,
    *,
    base: dict[str, Any],
    training_output: str,
    defaults: dict[str, Any],
    default_base_config: str,
) -> list[dict[str, Any]]:
    ckpts = _parse_ckpts(args.ckpts, base, training_output)
    steps = _parse_csv_ints(args.steps) if args.steps else [int(defaults["num_inference_steps"])]
    chunks = (
        _parse_csv_floats(args.chunk) if args.chunk else [float(defaults["chunk_size_threshold"])]
    )
    alphas = (
        _parse_csv_floats(args.alpha) if args.alpha else [float(defaults["smoothing_alpha"])]
    )
    return [
        {
            "name": None,
            "base_config": default_base_config,
            "policy_path": ckpt,
            "num_inference_steps": step,
            "chunk_size_threshold": chunk,
            "smoothing_alpha": alpha,
        }
        for ckpt, step, chunk, alpha in product(ckpts, steps, chunks, alphas)
    ]


def _apply_rtc_overlay(derived: dict[str, Any], combo: dict[str, Any]) -> str:
    """Apply rtc / rtc_enabled overlays. Returns tag: on|off|keep."""
    server = derived.setdefault("policy_server", {})
    if "rtc" in combo:
        rtc_val = combo["rtc"]
        if rtc_val is None:
            server.pop("rtc", None)
            return "off"
        if not isinstance(rtc_val, dict):
            raise SystemExit("rtc overlay must be a mapping")
        merged = dict(server.get("rtc") or {})
        merged.update(rtc_val)
        server["rtc"] = merged
        return "on" if merged.get("enabled", False) else "off"
    if "rtc_enabled" in combo:
        enabled = bool(combo["rtc_enabled"])
        rtc = dict(server.get("rtc") or {})
        rtc["enabled"] = enabled
        # Sensible defaults when enabling RTC on a base that had no rtc block.
        if enabled:
            rtc.setdefault("execution_horizon", 10)
            rtc.setdefault("max_guidance_weight", 10.0)
            rtc.setdefault("prefix_attention_schedule", "EXP")
            rtc.setdefault("inference_delay_steps", 4)
            rtc.setdefault("debug", False)
        server["rtc"] = rtc
        return "on" if enabled else "off"
    rtc = server.get("rtc") or {}
    if isinstance(rtc, dict) and rtc.get("enabled"):
        return "on"
    return "off" if "rtc" in server else "keep"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="")
    parser.add_argument("--deploy-yaml", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ckpts", default="last")
    parser.add_argument("--steps", default="")
    parser.add_argument("--chunk", default="")
    parser.add_argument("--alpha", default="")
    parser.add_argument("--base-port", type=int, default=0)
    parser.add_argument("--max-instances", type=int, default=4)
    args = parser.parse_args()

    deploy: dict[str, Any] = {}
    yaml_path: Path | None = None
    if args.deploy_yaml:
        yaml_path = Path(args.deploy_yaml)
        if not yaml_path.is_file():
            raise SystemExit(f"Deploy YAML not found: {yaml_path}")
        deploy = _load_yaml(yaml_path)

    base_config_arg = args.base_config or str(deploy.get("base_config") or "")
    if not base_config_arg:
        raise SystemExit("Need --base-config or deploy YAML base_config")
    base_path = Path(base_config_arg)
    if not base_path.is_file():
        raise SystemExit(f"Base config not found: {base_path}")

    base = json.loads(base_path.read_text(encoding="utf-8"))
    training = base.get("training") or {}
    policy_live = base.get("policy_live") or {}
    policy_server = base.get("policy_server") or {}
    async_cfg = base.get("async_inference") or {}
    remote = base.get("remote_gpu") or {}

    training_output = str(training.get("output_dir") or "outputs/train/unknown")
    defaults = {
        "num_inference_steps": int(
            policy_server.get("num_inference_steps", policy_live.get("num_inference_steps", 5))
        ),
        "chunk_size_threshold": float(async_cfg.get("chunk_size_threshold", 0.5)),
        "smoothing_alpha": float(async_cfg.get("smoothing_alpha", 0.3)),
    }

    if deploy:
        # YAML is the source of truth for ckpt/params when --deploy-yaml is set.
        combos = _expand_from_yaml(
            deploy,
            base=base,
            training_output=training_output,
            defaults=defaults,
            default_base_config=str(base_path),
        )
    else:
        combos = _expand_from_cli(
            args,
            base=base,
            training_output=training_output,
            defaults=defaults,
            default_base_config=str(base_path),
        )

    # Precedence: CLI/env MAX_INSTANCES (always passed) > YAML max_instances > default.
    max_instances = int(args.max_instances)
    if deploy.get("max_instances") is not None and args.max_instances == 4:
        max_instances = int(deploy["max_instances"])

    if args.base_port > 0:
        base_port = int(args.base_port)
    elif deploy.get("base_port"):
        base_port = int(deploy["base_port"])
    else:
        base_port = int(remote.get("tunnel_local_port", policy_server.get("port", 8081)))

    if len(combos) > max_instances:
        raise SystemExit(
            f"Refusing to create {len(combos)} instances (max={max_instances}). "
            "Narrow YAML grid/instances or raise max_instances / MAX_INSTANCES."
        )

    out_dir = Path(args.out_dir)
    cfg_dir = out_dir / "cfgs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    base_cache: dict[str, dict[str, Any]] = {str(base_path.resolve()): base}

    instances: list[dict[str, Any]] = []
    for idx, combo in enumerate(combos):
        ckpt = str(combo["policy_path"])
        step = int(combo["num_inference_steps"])
        chunk = float(combo["chunk_size_threshold"])
        alpha = float(combo["smoothing_alpha"])
        port = base_port + idx

        combo_base_path = Path(str(combo.get("base_config") or base_path))
        cache_key = str(combo_base_path.resolve())
        if cache_key not in base_cache:
            if not combo_base_path.is_file():
                raise SystemExit(f"Base config not found: {combo_base_path}")
            base_cache[cache_key] = json.loads(combo_base_path.read_text(encoding="utf-8"))
        derived = copy.deepcopy(base_cache[cache_key])
        derived.setdefault("policy_live", {})
        derived.setdefault("policy_server", {})
        derived.setdefault("async_inference", {})
        derived.setdefault("remote_gpu", {})

        derived["policy_live"]["policy_path"] = ckpt
        derived["policy_live"]["num_inference_steps"] = step
        derived["policy_live"]["smoothing_alpha"] = alpha

        derived["policy_server"]["port"] = port
        derived["policy_server"]["policy_path"] = ckpt
        derived["policy_server"]["num_inference_steps"] = step
        derived["policy_server"]["preload_at_startup"] = True

        derived["async_inference"]["server_address"] = f"127.0.0.1:{port}"
        derived["async_inference"]["chunk_size_threshold"] = chunk
        derived["async_inference"]["smoothing_alpha"] = alpha
        if "aggregate_fn_name" in combo:
            derived["async_inference"]["aggregate_fn_name"] = str(combo["aggregate_fn_name"])

        rtc_tag = _apply_rtc_overlay(derived, combo)
        # When enabling RTC without explicit aggregate, prefer latest_only (avoids double smoothing).
        if rtc_tag == "on" and "aggregate_fn_name" not in combo:
            derived["async_inference"].setdefault("aggregate_fn_name", "latest_only")
        if rtc_tag == "off" and "aggregate_fn_name" not in combo:
            derived["async_inference"].setdefault("aggregate_fn_name", "weighted_average")

        aggregate = str(derived["async_inference"].get("aggregate_fn_name", ""))
        rtc_enabled = bool((derived.get("policy_server") or {}).get("rtc", {}).get("enabled"))

        derived["remote_gpu"]["use_ssh_tunnel"] = True
        derived["remote_gpu"]["tunnel_local_port"] = port
        derived["remote_gpu"]["tunnel_remote_port"] = port
        derived["remote_gpu"]["tunnel_remote_host"] = "127.0.0.1"

        ckpt_tag = _slug(Path(ckpt).parts[-3] if "checkpoints" in ckpt else Path(ckpt).name)
        rtc_suffix = f"_rtc-{rtc_tag}" if rtc_tag in ("on", "off") else ""
        auto_name = f"i{idx:02d}_ckpt-{ckpt_tag}_s{step}_c{chunk:g}_a{alpha:g}{rtc_suffix}"
        name = str(combo.get("name") or auto_name)
        # Keep filename filesystem-safe.
        file_stem = _slug(name, 80) if combo.get("name") else auto_name
        rel_cfg = f"cfgs/{file_stem}.json"
        abs_cfg = cfg_dir / f"{file_stem}.json"
        abs_cfg.write_text(json.dumps(derived, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        tmux = f"sf-srv-{_slug(args.run_id, 24)}-{idx:02d}"
        instances.append(
            {
                "id": idx,
                "name": name,
                "config_rel": rel_cfg,
                "config_path": str(abs_cfg),
                "base_config": str(combo_base_path),
                "policy_path": ckpt,
                "num_inference_steps": step,
                "chunk_size_threshold": chunk,
                "smoothing_alpha": alpha,
                "rtc_enabled": rtc_enabled,
                "aggregate_fn_name": aggregate,
                "port": port,
                "server_address": f"127.0.0.1:{port}",
                "tmux_session": tmux,
                "tunnel_pid_file": f"pids/tunnel_{idx:02d}.pid",
            }
        )

    manifest = {
        "run_id": args.run_id,
        "base_config": str(base_path.resolve()),
        "deploy_yaml": str(yaml_path.resolve()) if yaml_path else "",
        "out_dir": str(out_dir.resolve()),
        "base_port": base_port,
        "cuda_visible_devices": str(deploy.get("cuda_visible_devices") or ""),
        "instances": instances,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if yaml_path is not None:
        # Keep a copy beside the run for reproducibility.
        (out_dir / "deploy.yaml").write_text(yaml_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "n": len(instances),
                "out_dir": str(out_dir),
                "cuda_visible_devices": manifest["cuda_visible_devices"],
            }
        )
    )


if __name__ == "__main__":
    main()
