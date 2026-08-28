#!/usr/bin/env python3
"""EEF → Piper joint batch converter for Soft-Fold / X-VLA virtual datasets.

Reads a Soft-Fold style LeRobot dataset (eef6d in ``observation.state[30:50]``),
solves Piper dual-arm IK, optionally masks robot regions in RGB frames, smooths
joint trajectories, and writes a Piper joint LeRobot dataset.

Episode access uses ``meta/episodes`` ``dataset_from_index`` / ``dataset_to_index``
(no HF ``filter``, which is prohibitively slow on 1.5k episodes).

Example
-------
  python data/eef_to_joint_converter.py \\
    --src-root ../piper/data/lerobot/xvla-soft-fold \\
    --out-root data/lerobot/piper-softfold-virtual-v1 \\
    --repo-id piper-softfold-virtual-v1 \\
    --max-episodes 50 --mask-mode band --smooth-window 5

  # IK feasibility only (parquet state, no videos):
  python data/eef_to_joint_converter.py \\
    --src-root ../piper/data/lerobot/xvla-soft-fold \\
    --out-root /tmp/_unused --dry-run --max-episodes 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(SRC_ROOT), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from softfold.kinematics import (  # noqa: E402
    DEFAULT_JOINT_LIMITS,
    StationCalibration,
    dual_eef6d_to_joint_action,
    eef_in_world,
)
from softfold.masking import mask_robot_region  # noqa: E402
from softfold.smoothing import moving_average  # noqa: E402
from softfold.workspace_map import (  # noqa: E402
    PerArmWorkspaceMap,
    clamp_eef20_workspace,
    default_per_arm_map,
    load_workspace_map,
)

# Soft-fold observation.state layout
EEF6D_SLICE = slice(30, 50)
QPOS_SLICE = slice(52, 66)

JOINT_ACTION_NAMES = [
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_joint_6.pos",
    "left_gripper.pos",
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_joint_3.pos",
    "right_joint_4.pos",
    "right_joint_5.pos",
    "right_joint_6.pos",
    "right_gripper.pos",
]

CAMERA_KEYS_DEFAULT = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-root", type=Path, required=True, help="Soft-Fold / source LeRobot root")
    p.add_argument("--out-root", type=Path, required=True, help="Output Piper joint LeRobot root")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--station", type=Path, default=None, help="Station calibration JSON")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--episode-stride", type=int, default=1, help="Keep 1 of every N episodes")
    p.add_argument("--min-ik-success-ratio", type=float, default=0.7)
    p.add_argument("--smooth-window", type=int, default=5)
    p.add_argument(
        "--mask-mode",
        choices=("none", "band", "black"),
        default="band",
        help="band=top/side crude mask; black=fill whole frame black (debug); none=keep source RGB",
    )
    p.add_argument("--mask-top-fraction", type=float, default=0.15)
    p.add_argument("--mask-side-fraction", type=float, default=0.08)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Only report IK stats, do not write")
    p.add_argument("--frame-stride", type=int, default=1, help="Subsample frames inside each episode")
    p.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=None,
        help="Cap frames per episode after stride (useful for dry-run)",
    )
    p.add_argument(
        "--workspace-map",
        type=Path,
        default=REPO_ROOT / "configs" / "workspace_map.json",
        help="Per-arm Soft-Fold→Piper workspace map JSON",
    )
    p.add_argument(
        "--no-workspace-map",
        action="store_true",
        help="Disable workspace remap (expect very low IK success)",
    )
    p.add_argument("--z-min-m", type=float, default=0.03, help="Clamp remapped TCP z (world)")
    p.add_argument("--max-delta-xy-m", type=float, default=0.22, help="Max XY deviation from Piper arm center")
    p.add_argument(
        "--orient-mode",
        choices=("seed", "source"),
        default="seed",
        help="seed=keep Piper FK orientation (recommended); source=use Soft-Fold rot6d",
    )
    p.add_argument("--ik-pos-tol-m", type=float, default=0.01)
    p.add_argument("--ik-rot-tol-rad", type=float, default=0.2)
    p.add_argument("--ik-max-iters", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task", default="fold the cloth")
    return p.parse_args()


def _action_dict_to_vec14(action: dict[str, float]) -> np.ndarray:
    return np.array([float(action[n]) for n in JOINT_ACTION_NAMES], dtype=np.float32)


def _extract_eef20(state: np.ndarray) -> np.ndarray:
    s = np.asarray(state, dtype=np.float64).reshape(-1)
    if s.shape[0] < 50:
        raise ValueError(f"state dim {s.shape[0]} < 50; need eef6d in soft-fold layout")
    return s[EEF6D_SLICE].copy()


def _seed_qpos(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use source qpos as IK seed when available (Franka≠Piper but better than zeros)."""
    s = np.asarray(state, dtype=np.float64).reshape(-1)
    if s.shape[0] >= 66:
        q = s[QPOS_SLICE]
        return q[:6].copy(), q[7:13].copy()
    return np.zeros(6), np.zeros(6)


def _maybe_mask(img: np.ndarray, mode: str, top: float, side: float) -> np.ndarray:
    if mode == "none":
        return img
    if mode == "black":
        return np.zeros_like(img)
    return mask_robot_region(img, top_fraction=top, side_fraction=side)


def load_episode_index(src_root: Path) -> list[dict[str, int]]:
    """Return per-episode metadata sorted by episode_index."""
    files = sorted((src_root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episodes parquet under {src_root}/meta/episodes")
    cols = [
        "episode_index",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "data/chunk_index",
        "data/file_index",
    ]
    rows: list[dict[str, int]] = []
    for f in files:
        table = pq.read_table(f, columns=cols)
        pdf = table.to_pandas()
        for _, r in pdf.iterrows():
            rows.append(
                {
                    "episode_index": int(r["episode_index"]),
                    "length": int(r["length"]),
                    "from_idx": int(r["dataset_from_index"]),
                    "to_idx": int(r["dataset_to_index"]),
                    "chunk_index": int(r["data/chunk_index"]),
                    "file_index": int(r["data/file_index"]),
                }
            )
    rows.sort(key=lambda x: x["episode_index"])
    return rows


def load_info(src_root: Path) -> dict[str, Any]:
    return json.loads((src_root / "meta" / "info.json").read_text())


def _episode_data_path(src_root: Path, ep: dict[str, int], info: dict[str, Any]) -> Path:
    template = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    rel = template.format(chunk_index=ep["chunk_index"], file_index=ep["file_index"])
    path = src_root / rel
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_states_for_episode_parquet(src_root: Path, ep: dict[str, int], info: dict[str, Any]) -> np.ndarray:
    """Load observation.state for one episode directly from its data parquet (no videos)."""
    path = _episode_data_path(src_root, ep, info)
    table = pq.read_table(
        path,
        columns=["observation.state"],
        filters=[("episode_index", "=", ep["episode_index"])],
    )
    if table.num_rows == 0:
        # Fallback: slice by global index range inside the same file
        table = pq.read_table(
            path,
            columns=["index", "observation.state"],
            filters=[
                ("index", ">=", ep["from_idx"]),
                ("index", "<", ep["to_idx"]),
            ],
        )
        states_list = table.column("observation.state").to_pylist()
    else:
        states_list = table.column("observation.state").to_pylist()
    if not states_list:
        raise RuntimeError(
            f"failed to load states for episode {ep['episode_index']} from {path}"
        )
    return np.stack([np.asarray(x, dtype=np.float64) for x in states_list])


def _replace_orient_with_seed(
    eef20: np.ndarray,
    q_left: np.ndarray,
    q_right: np.ndarray,
    station: StationCalibration,
    grip_left: float,
    grip_right: float,
) -> np.ndarray:
    """Keep remapped XYZ/grippers; replace rot6d with current Piper seed orientation."""
    out = np.asarray(eef20, dtype=np.float64).copy()
    left = eef_in_world(q_left, grip_left, station.base_T("left")).as_eef6d()
    right = eef_in_world(q_right, grip_right, station.base_T("right")).as_eef6d()
    out[3:9] = left[3:9]
    out[13:19] = right[3:9]
    out[9] = float(grip_left)
    out[19] = float(grip_right)
    return out


def _remap_eef20(
    eef20: np.ndarray,
    *,
    workspace_map: PerArmWorkspaceMap | None,
    z_min_m: float,
    max_delta_xy_m: float,
) -> np.ndarray:
    out = np.asarray(eef20, dtype=np.float64).copy()
    if workspace_map is not None:
        out = workspace_map.apply_eef20(out)
        out = clamp_eef20_workspace(
            out,
            center_left=workspace_map.piper_center_left,
            center_right=workspace_map.piper_center_right,
            z_min_m=z_min_m,
            max_delta_xy_m=max_delta_xy_m,
        )
    return out


def convert_episode_frames(
    states: np.ndarray,
    station: StationCalibration,
    *,
    min_success_ratio: float,
    workspace_map: PerArmWorkspaceMap | None = None,
    orient_mode: str = "seed",
    z_min_m: float = 0.03,
    max_delta_xy_m: float = 0.22,
    frame_stride: int = 1,
    max_frames: int | None = None,
    ik_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Convert (T, state_dim) → (T', 14) Piper joints or None if episode filtered."""
    idxs = list(range(0, states.shape[0], max(1, frame_stride)))
    if max_frames is not None:
        idxs = idxs[: max_frames]
    t = len(idxs)
    # Prefer a reachable Piper home seed over Franka qpos
    q_left = np.array([0.0, 1.8, -1.5, 0.0, 0.9, 0.0], dtype=np.float64)
    q_right = q_left.copy()

    joints = np.zeros((t, 14), dtype=np.float64)
    ok = 0
    pos_errs: list[float] = []
    kwargs = dict(ik_kwargs or {})
    kwargs.setdefault("max_iters", 40)
    kwargs.setdefault("pos_tol_m", 0.01)
    kwargs.setdefault("rot_tol_rad", 0.2)

    for j, i in enumerate(idxs):
        raw = _extract_eef20(states[i])
        eef20 = _remap_eef20(
            raw,
            workspace_map=workspace_map,
            z_min_m=z_min_m,
            max_delta_xy_m=max_delta_xy_m,
        )
        if orient_mode == "seed":
            eef20 = _replace_orient_with_seed(
                eef20, q_left, q_right, station, float(raw[9]), float(raw[19])
            )
        action, meta = dual_eef6d_to_joint_action(
            eef20, q_left, q_right, station, ik_kwargs=kwargs
        )
        joints[j] = _action_dict_to_vec14(action)
        q_left = joints[j, :6]
        q_right = joints[j, 7:13]
        if meta["success"]:
            ok += 1
        pos_errs.append(float(meta["max_pos_err_m"]))

    ratio = ok / max(t, 1)
    stats = {
        "frames": t,
        "src_frames": int(states.shape[0]),
        "frame_stride": int(frame_stride),
        "ik_ok": ok,
        "ik_success_ratio": ratio,
        "mean_pos_err_m": float(np.mean(pos_errs)) if pos_errs else None,
        "max_pos_err_m": float(np.max(pos_errs)) if pos_errs else None,
        "orient_mode": orient_mode,
    }
    if ratio < min_success_ratio:
        return None, stats
    return joints.astype(np.float32), stats



def main() -> int:
    args = parse_args()
    station = StationCalibration.load(args.station)
    src_root = args.src_root.resolve()
    info = load_info(src_root)
    episodes = load_episode_index(src_root)
    n_ep = len(episodes)
    selected = episodes[:: max(1, args.episode_stride)]
    if args.max_episodes is not None:
        selected = selected[: args.max_episodes]

    workspace_map: PerArmWorkspaceMap | None = None
    if not args.no_workspace_map:
        if args.workspace_map.is_file():
            workspace_map = load_workspace_map(args.workspace_map)
        else:
            workspace_map = default_per_arm_map()
            args.workspace_map.parent.mkdir(parents=True, exist_ok=True)
            args.workspace_map.write_text(
                __import__("json").dumps(workspace_map.to_dict(), indent=2)
            )
        print(f"[converter] workspace_map={workspace_map.name}", flush=True)
        print(
            f"  piper_center_L={np.round(workspace_map.piper_center_left, 3).tolist()} "
            f"R={np.round(workspace_map.piper_center_right, 3).tolist()}",
            flush=True,
        )
    else:
        print("[converter] workspace_map=DISABLED", flush=True)

    print(f"[converter] src={src_root} episodes={n_ep} selected={len(selected)}", flush=True)
    print(
        f"[converter] orient_mode={args.orient_mode} z_min={args.z_min_m} "
        f"max_dxy={args.max_delta_xy_m} stride={args.frame_stride}",
        flush=True,
    )

    convert_kwargs = dict(
        workspace_map=workspace_map,
        orient_mode=args.orient_mode,
        z_min_m=args.z_min_m,
        max_delta_xy_m=args.max_delta_xy_m,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames_per_episode,
        ik_kwargs={
            "max_iters": args.ik_max_iters,
            "pos_tol_m": args.ik_pos_tol_m,
            "rot_tol_rad": args.ik_rot_tol_rad,
        },
    )

    all_stats: list[dict[str, Any]] = []
    kept = 0

    if args.dry_run:
        if args.frame_stride == 1 and args.max_frames_per_episode is None:
            convert_kwargs["frame_stride"] = 10
            convert_kwargs["max_frames"] = 120
            print("[dry-run] auto subsample stride=10 max_frames=120", flush=True)
        for ep in selected[: min(20, len(selected))]:
            states = load_states_for_episode_parquet(src_root, ep, info)
            joints, st = convert_episode_frames(
                states,
                station,
                min_success_ratio=args.min_ik_success_ratio,
                **convert_kwargs,
            )
            st["episode_index"] = ep["episode_index"]
            st["kept"] = joints is not None
            all_stats.append(st)
            kept += int(st["kept"])
            print(
                f"  ep={ep['episode_index']:4d} frames={st['frames']:5d}/{st['src_frames']} "
                f"ik_ok={st['ik_success_ratio']:.3f} mean_err={st['mean_pos_err_m']:.4f} "
                f"kept={st['kept']}",
                flush=True,
            )
        print(f"[dry-run] kept {kept}/{len(all_stats)} sampled episodes", flush=True)
        report_path = Path("/tmp/ik_dry_run_report.json")
        report_path.write_text(
            json.dumps(
                {
                    "src_root": str(src_root),
                    "workspace_map": None if workspace_map is None else workspace_map.to_dict(),
                    "orient_mode": args.orient_mode,
                    "per_episode": all_stats,
                },
                indent=2,
            )
        )
        print(f"[dry-run] report={report_path}", flush=True)
        return 0

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit("lerobot is required for writing datasets. Activate piper env.") from exc

    if args.out_root.exists():
        if not args.overwrite:
            raise SystemExit(f"out-root exists: {args.out_root} (pass --overwrite)")
        import shutil

        shutil.rmtree(args.out_root)

    src = LeRobotDataset(repo_id="local/src", root=str(src_root))
    meta = src.meta
    repo_id = args.repo_id or args.out_root.name

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": JOINT_ACTION_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": {"motors": JOINT_ACTION_NAMES},
        },
    }
    for cam in CAMERA_KEYS_DEFAULT:
        if cam in meta.features:
            features[cam] = meta.features[cam]

    dst = LeRobotDataset.create(
        repo_id=repo_id,
        fps=max(1, int(info.get("fps", meta.fps)) // max(1, args.frame_stride)),
        root=str(args.out_root),
        robot_type="piper",
        features=features,
        use_videos=True,
    )

    for ep in selected:
        states = load_states_for_episode_parquet(src_root, ep, info)
        joints, st = convert_episode_frames(
            states,
            station,
            min_success_ratio=args.min_ik_success_ratio,
            **convert_kwargs,
        )
        st["episode_index"] = ep["episode_index"]
        all_stats.append(st)
        if joints is None:
            print(
                f"  skip ep={ep['episode_index']} ik_success_ratio={st['ik_success_ratio']:.3f}",
                flush=True,
            )
            continue

        if args.smooth_window > 1:
            arms = np.concatenate([joints[:, :6], joints[:, 7:13]], axis=1)
            arms_s = moving_average(arms, args.smooth_window).astype(np.float32)
            joints[:, :6] = arms_s[:, :6]
            joints[:, 7:13] = arms_s[:, 6:]

        src_idxs = list(range(ep["from_idx"], ep["to_idx"], max(1, args.frame_stride)))
        if args.max_frames_per_episode is not None:
            src_idxs = src_idxs[: args.max_frames_per_episode]

        for i, gidx in enumerate(src_idxs):
            if i >= joints.shape[0]:
                break
            row = src[gidx]
            frame: dict[str, Any] = {
                "observation.state": joints[i],
                "action": joints[i],
                "task": args.task,
            }
            for cam in CAMERA_KEYS_DEFAULT:
                if cam not in features:
                    continue
                img = row.get(cam) if isinstance(row, dict) else None
                if img is None and hasattr(row, "get"):
                    img = row.get(cam)
                if img is None and isinstance(row, dict):
                    obs = row.get("observation", {})
                    if isinstance(obs, dict):
                        images = obs.get("images", {})
                        short = cam.split(".")[-1]
                        img = images.get(short) if isinstance(images, dict) else None
                if img is None:
                    continue
                arr = np.asarray(img)
                if hasattr(img, "permute") and arr.ndim == 3 and arr.shape[0] in (1, 3):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.dtype != np.uint8:
                    if arr.max() <= 1.0:
                        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        arr = arr.clip(0, 255).astype(np.uint8)
                if arr.ndim == 3:
                    arr = _maybe_mask(
                        arr,
                        args.mask_mode,
                        args.mask_top_fraction,
                        args.mask_side_fraction,
                    )
                frame[cam] = arr
            dst.add_frame(frame)
        dst.save_episode()
        kept += 1
        print(
            f"  keep ep={ep['episode_index']} frames={st['frames']} "
            f"ik_ok={st['ik_success_ratio']:.3f} written_total={kept}",
            flush=True,
        )

    report = {
        "src_root": str(src_root),
        "out_root": str(args.out_root),
        "selected_episodes": len(selected),
        "kept_episodes": kept,
        "mask_mode": args.mask_mode,
        "smooth_window": args.smooth_window,
        "frame_stride": args.frame_stride,
        "workspace_map": None if workspace_map is None else workspace_map.to_dict(),
        "orient_mode": args.orient_mode,
        "min_ik_success_ratio": args.min_ik_success_ratio,
        "joint_limits": [list(x) for x in DEFAULT_JOINT_LIMITS],
        "per_episode": all_stats,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    report_path = args.out_root / "ik_conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[converter] done kept={kept}/{len(selected)} report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
