"""Compose training action targets from observation.state joints + action grippers.

LeRobot training always supervises the `action` column. When teleop was recorded with
`action_source=leader`, joint targets lag / disagree with the follower. Replay showed
that executing joints from observation.state and grippers from action reproduces the
recording. This module rewrites dataset parquet actions accordingly and refreshes stats.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from .recorder import ARM_STATE_KEYS

ActionComposeMode = Literal["obs_joints_action_gripper"]

ACTION_COMPOSE_MODES: tuple[str, ...] = ("obs_joints_action_gripper",)

GRIPPER_INDICES = [index for index, key in enumerate(ARM_STATE_KEYS) if "gripper" in key]
JOINT_INDICES = [index for index in range(len(ARM_STATE_KEYS)) if index not in GRIPPER_INDICES]
ARM_INDICES = {
    "left": [index for index, key in enumerate(ARM_STATE_KEYS) if key.startswith("left_")],
    "right": [index for index, key in enumerate(ARM_STATE_KEYS) if key.startswith("right_")],
}


def compose_action_vector(
    observation_state: np.ndarray,
    action: np.ndarray,
    mode: str = "obs_joints_action_gripper",
) -> np.ndarray:
    """Return action vector with joints from observation and grippers from action."""
    if mode != "obs_joints_action_gripper":
        raise ValueError(f"Unsupported action compose mode: {mode}")

    state = np.asarray(observation_state, dtype=np.float32).reshape(-1)
    act = np.asarray(action, dtype=np.float32).reshape(-1)
    if state.shape != act.shape:
        raise ValueError(f"state/action shape mismatch: {state.shape} vs {act.shape}")
    if state.shape[0] < len(ARM_STATE_KEYS):
        raise ValueError(f"expected at least {len(ARM_STATE_KEYS)} dims, got {state.shape[0]}")

    out = act.copy()
    out[JOINT_INDICES] = state[JOINT_INDICES]
    # gripper dims already from action
    return out


def compose_action_batch(
    observation_states: np.ndarray,
    actions: np.ndarray,
    mode: str = "obs_joints_action_gripper",
) -> np.ndarray:
    states = np.asarray(observation_states, dtype=np.float32)
    acts = np.asarray(actions, dtype=np.float32)
    if states.ndim != 2 or acts.ndim != 2:
        raise ValueError("expected batched (N, D) arrays")
    if states.shape != acts.shape:
        raise ValueError(f"batch shape mismatch: {states.shape} vs {acts.shape}")

    out = acts.copy()
    out[:, JOINT_INDICES] = states[:, JOINT_INDICES]
    return out


def suffix_repo_id(source_repo_id: str, suffix: str) -> str:
    if "/" in source_repo_id:
        prefix, name = source_repo_id.rsplit("/", 1)
        return f"{prefix}/{name}{suffix}"
    return f"{source_repo_id}{suffix}"


def default_compose_repo_id(source_repo_id: str, mode: str = "obs_joints_action_gripper") -> str:
    if mode != "obs_joints_action_gripper":
        raise ValueError(f"Unsupported action compose mode: {mode}")
    return suffix_repo_id(source_repo_id, "_ojag")


def normalize_camera_names(camera_names: Sequence[str] | None) -> list[str] | None:
    if camera_names is None:
        return None
    names = [str(name).strip() for name in camera_names if str(name).strip()]
    return names or None


def normalize_arm_sides(arm_sides: Sequence[str] | str | None) -> list[str] | None:
    if arm_sides is None:
        return None
    if isinstance(arm_sides, str):
        values = [part.strip() for part in arm_sides.split(",")]
    else:
        values = [str(part).strip() for part in arm_sides]
    sides = [side for side in values if side]
    if not sides:
        return None
    invalid = [side for side in sides if side not in ARM_INDICES]
    if invalid:
        raise ValueError(f"Unsupported arm_sides {invalid}; choose from {sorted(ARM_INDICES)}")
    # Keep canonical left/right order so feature order is stable.
    return [side for side in ("left", "right") if side in sides]


def derivation_suffix(
    mode: str | None,
    camera_names: Sequence[str] | None,
    arm_sides: Sequence[str] | str | None,
) -> str:
    parts: list[str] = []
    if mode:
        if mode == "obs_joints_action_gripper":
            parts.append("ojag")
        else:
            parts.append(mode.replace("_", "-"))
    cams = normalize_camera_names(camera_names)
    if cams is not None:
        parts.append("cams-" + "-".join(name.removeprefix("cam_") for name in cams))
    sides = normalize_arm_sides(arm_sides)
    if sides is not None and sides != ["left", "right"]:
        parts.append("arms-" + "-".join(sides))
    return "_" + "_".join(parts) if parts else ""


def default_derived_repo_id(
    source_repo_id: str,
    mode: str | None = "obs_joints_action_gripper",
    *,
    camera_names: Sequence[str] | None = None,
    arm_sides: Sequence[str] | str | None = None,
) -> str:
    suffix = derivation_suffix(mode, camera_names, arm_sides)
    return suffix_repo_id(source_repo_id, suffix) if suffix else source_repo_id


def _link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_dataset_tree(
    source_root: Path,
    target_root: Path,
    *,
    hardlink_videos: bool = True,
) -> None:
    """Copy a LeRobot dataset directory; hard-link videos when possible to save disk."""
    if target_root.exists():
        raise FileExistsError(f"Target dataset already exists: {target_root}")

    for src_path in source_root.rglob("*"):
        if src_path.is_dir():
            continue
        rel = src_path.relative_to(source_root)
        dst_path = target_root / rel
        is_video = "videos" in rel.parts and src_path.suffix.lower() in {".mp4", ".avi", ".mkv"}
        if hardlink_videos and is_video:
            _link_or_copy_file(src_path, dst_path)
        else:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)


def rewrite_parquet_actions(data_dir: Path, mode: str = "obs_joints_action_gripper") -> int:
    """Rewrite action columns in-place under data/. Returns number of frames updated."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_files = sorted(data_dir.glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")

    total = 0
    for path in parquet_files:
        table = pq.read_table(path)
        if "action" not in table.column_names or "observation.state" not in table.column_names:
            raise KeyError(f"{path} missing action or observation.state")

        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        composed = compose_action_batch(states, actions, mode=mode)

        # Replace action column while preserving other columns / schema-ish layout
        arrays = []
        names = []
        for name in table.column_names:
            if name == "action":
                action_list = [row.astype(np.float32).tolist() for row in composed]
                arrays.append(pa.array(action_list, type=table.schema.field("action").type))
            else:
                arrays.append(table[name])
            names.append(name)
        new_table = pa.Table.from_arrays(arrays, names=names)
        pq.write_table(new_table, path)
        total += composed.shape[0]
        print(f"  rewrote {path.name}: {composed.shape[0]} frames")
    return total


def select_vector_columns(dataset_root: Path, arm_sides: Sequence[str] | str | None) -> list[int] | None:
    sides = normalize_arm_sides(arm_sides)
    if sides is None or sides == ["left", "right"]:
        return None
    indices: list[int] = []
    for side in sides:
        indices.extend(ARM_INDICES[side])
    return indices


def rewrite_parquet_vector_dims(data_dir: Path, indices: list[int] | None) -> int:
    if indices is None:
        return 0
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_files = sorted(data_dir.glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")

    total = 0
    for path in parquet_files:
        table = pq.read_table(path)
        arrays = []
        names = []
        for name in table.column_names:
            if name in {"action", "observation.state"}:
                arr = np.asarray(table[name].to_pylist(), dtype=np.float32)
                selected = arr[:, indices]
                selected_list = [row.astype(np.float32).tolist() for row in selected]
                value_type = pa.list_(pa.float32(), len(indices))
                arrays.append(pa.array(selected_list, type=value_type))
            else:
                arrays.append(table[name])
            names.append(name)
        pq.write_table(pa.Table.from_arrays(arrays, names=names), path)
        total += table.num_rows
        print(f"  selected vector dims in {path.name}: {len(indices)} dims, {table.num_rows} frames")
    return total


def filter_dataset_cameras(dataset_root: Path, camera_names: Sequence[str] | None) -> list[str] | None:
    selected = normalize_camera_names(camera_names)
    if selected is None:
        return None

    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    existing = [key.removeprefix("observation.images.") for key in features if key.startswith("observation.images.")]
    missing = [name for name in selected if name not in existing]
    if missing:
        raise ValueError(f"Selected cameras {missing} not found in dataset. Existing cameras: {existing}")

    keep_feature_keys = {f"observation.images.{name}" for name in selected}
    remove_feature_keys = [
        key for key in list(features) if key.startswith("observation.images.") and key not in keep_feature_keys
    ]
    for key in remove_feature_keys:
        features.pop(key, None)

    videos_root = dataset_root / "videos"
    if videos_root.exists():
        for path in list(videos_root.iterdir()):
            if path.is_dir() and path.name.startswith("observation.images.") and path.name not in keep_feature_keys:
                shutil.rmtree(path)

    images_root = dataset_root / "images"
    if images_root.exists():
        for path in list(images_root.iterdir()):
            if path.is_dir() and path.name.startswith("observation.images.") and path.name not in keep_feature_keys:
                shutil.rmtree(path)

    info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")
    print(f"Selected cameras: {selected}")
    return selected


def filter_dataset_arms(dataset_root: Path, arm_sides: Sequence[str] | str | None) -> list[str] | None:
    sides = normalize_arm_sides(arm_sides)
    indices = select_vector_columns(dataset_root, sides)
    if indices is None:
        return None

    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    selected_names = [ARM_STATE_KEYS[index] for index in indices]
    for key in ("action", "observation.state"):
        feature = info.get("features", {}).get(key)
        if not isinstance(feature, dict):
            raise KeyError(f"meta/info.json missing feature {key}")
        feature["shape"] = [len(indices)]
        feature["names"] = selected_names

    info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")
    n = rewrite_parquet_vector_dims(dataset_root / "data", indices)
    print(f"Selected arm sides: {sides}; updated {n} frames")
    return sides


def recompute_numeric_stats(dataset_root: Path) -> None:
    """Refresh meta/stats.json numeric features after action rewrite."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # repo_id is only used as an identifier when root is absolute/local
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    repo_id = f"local/{dataset_root.name}"
    dataset = LeRobotDataset(repo_id=repo_id, root=str(dataset_root), video_backend="pyav")
    try:
        from lerobot.datasets.dataset_tools import recompute_stats

        recompute_stats(dataset, skip_image_video=True)
    except Exception as exc:
        # Fallback: lightweight action-only stats from parquet if API differs
        print(f"警告：lerobot recompute_stats 失败（{exc}），改用本地 action 统计。")
        _recompute_action_stats_fallback(dataset_root, info)


def _recompute_action_stats_fallback(dataset_root: Path, info: dict[str, Any]) -> None:
    import pyarrow.parquet as pq

    stats_path = dataset_root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

    def collect(feature_name: str) -> np.ndarray:
        values = []
        for path in sorted((dataset_root / "data").glob("chunk-*/*.parquet")):
            table = pq.read_table(path, columns=[feature_name])
            values.extend(table[feature_name].to_pylist())
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 2:
            raise RuntimeError(f"unexpected {feature_name} array shape: {arr.shape}")
        return arr

    def quantiles(values: np.ndarray) -> dict[str, list[float]]:
        qs = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
        return {
            "q01": qs[0].tolist(),
            "q10": qs[1].tolist(),
            "q50": qs[2].tolist(),
            "q90": qs[3].tolist(),
            "q99": qs[4].tolist(),
        }

    def numeric_stats(arr: np.ndarray) -> dict[str, list[float]]:
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(arr.shape[0])],
            **quantiles(arr),
        }

    for feature_name in ("observation.state", "action"):
        stats[feature_name] = numeric_stats(collect(feature_name))
    stats_path.write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")
    print(f"Wrote fallback action stats: {stats_path}")


def annotate_info(
    dataset_root: Path,
    *,
    mode: str,
    source_repo_id: str,
    camera_names: Sequence[str] | None = None,
    arm_sides: Sequence[str] | str | None = None,
) -> None:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["piper_action_compose"] = {
        "mode": mode,
        "source_repo_id": source_repo_id,
        "joint_indices": JOINT_INDICES,
        "gripper_indices": GRIPPER_INDICES,
        "joint_keys": [ARM_STATE_KEYS[i] for i in JOINT_INDICES],
        "gripper_keys": [ARM_STATE_KEYS[i] for i in GRIPPER_INDICES],
        "description": "action joints copied from observation.state; grippers kept from original action",
    }
    selected_cameras = normalize_camera_names(camera_names)
    selected_sides = normalize_arm_sides(arm_sides)
    if selected_cameras is not None or selected_sides is not None:
        info["piper_dataset_filter"] = {
            "camera_names": selected_cameras,
            "arm_sides": selected_sides,
        }
    info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")


def build_composed_dataset(
    source_root: Path,
    target_root: Path,
    *,
    source_repo_id: str,
    mode: str = "obs_joints_action_gripper",
    overwrite: bool = False,
    hardlink_videos: bool = True,
    camera_names: Sequence[str] | None = None,
    arm_sides: Sequence[str] | str | None = None,
) -> Path:
    if mode not in ACTION_COMPOSE_MODES:
        raise ValueError(f"Unsupported mode {mode!r}; choose from {ACTION_COMPOSE_MODES}")
    if not (source_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Source dataset missing meta/info.json: {source_root}")

    if target_root.exists():
        if not overwrite:
            print(f"Target already exists, reuse: {target_root}")
            return target_root
        shutil.rmtree(target_root)

    print(f"Copying dataset → {target_root}")
    copy_dataset_tree(source_root, target_root, hardlink_videos=hardlink_videos)
    print(f"Composing actions mode={mode} (joints←obs, gripper←action)")
    n = rewrite_parquet_actions(target_root / "data", mode=mode)
    print(f"Updated {n} frames")
    filter_dataset_cameras(target_root, camera_names)
    filter_dataset_arms(target_root, arm_sides)
    print("Recomputing stats…")
    recompute_numeric_stats(target_root)
    annotate_info(
        target_root,
        mode=mode,
        source_repo_id=source_repo_id,
        camera_names=camera_names,
        arm_sides=arm_sides,
    )
    print(f"Done: {target_root}")
    return target_root
