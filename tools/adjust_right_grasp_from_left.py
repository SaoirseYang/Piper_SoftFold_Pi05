#!/usr/bin/env python3
"""Align right-arm near-table depth to left-arm contact depth (constant offset).

Folding needs pick-up AND put-down. This tool compares per-episode left/right
contact depth (default: p5 of near-table clearances) and lowers the right arm's
near-table segment by that constant delta, preserving trajectory shape.

Example:
  PYTHONPATH=src python tools/adjust_right_grasp_from_left.py \\
    --source-root data/lerobot/local/softfold_piper_v2 \\
    --target-root data/lerobot/local/softfold_piper_v2_rgrasp \\
    --dry-run

  PYTHONPATH=src python tools/adjust_right_grasp_from_left.py \\
    --source-root data/lerobot/local/softfold_piper_v2 \\
    --target-root data/lerobot/local/softfold_piper_v2_rgrasp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from piper_towel_fold.action_compose import (  # noqa: E402
    copy_dataset_tree,
    recompute_numeric_stats,
)
from softfold.grasp_height_adjust import (  # noqa: E402
    NearTableConfig,
    adjust_right_grasp_states,
    clamp_right_after_smooth,
    episode_contact_delta_mm,
    episode_left_grasp_reference,
    is_near_table_frame,
    tcp_clearance_mm,
)
from softfold.kinematics import StationCalibration  # noqa: E402
from softfold.smoothing import moving_average  # noqa: E402
from piper_towel_fold.episode_outcomes import (  # noqa: E402
    align_outcomes_to_episodes,
    load_outcome_records,
    read_episode_count,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--target-root", type=Path, required=True)
    p.add_argument("--station", type=Path, default=None)
    p.add_argument("--ref-stat", choices=("min", "p5", "p10", "p50"), default="p5")
    p.add_argument(
        "--grip-max-m",
        type=float,
        default=None,
        help="If set, require gripper < this (m). Default: ignore grip (pick+place+near-table).",
    )
    p.add_argument("--max-clearance-mm", type=float, default=30.0, help="Near-table band upper bound.")
    p.add_argument("--min-delta-mm", type=float, default=0.5, help="Skip episode if scaled gap is below this.")
    p.add_argument("--max-lower-mm", type=float, default=5.0, help="Cap per-episode constant lowering.")
    p.add_argument(
        "--delta-scale",
        type=float,
        default=0.5,
        help="Apply this fraction of (right_contact - left_contact). Default 0.5 = milder.",
    )
    p.add_argument("--smooth-window", type=int, default=5, help="Moving-average on adjusted right joints.")
    p.add_argument(
        "--outcomes-filter",
        choices=("all", "success"),
        default="success",
        help="Only adjust episodes labeled success in episode_outcomes.jsonl (default).",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--copy-videos", action="store_true", help="Copy videos instead of hard-linking.")
    return p.parse_args()


def load_all_states(data_dir: Path) -> tuple[list[Path], list[np.ndarray], list[np.ndarray]]:
    import pyarrow.parquet as pq

    paths = sorted(data_dir.glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files under {data_dir}")
    states_list: list[np.ndarray] = []
    episodes_list: list[np.ndarray] = []
    for path in paths:
        table = pq.read_table(path, columns=["observation.state", "episode_index"])
        states_list.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
        episodes_list.append(np.asarray(table["episode_index"].to_pylist(), dtype=np.int64))
    return paths, states_list, episodes_list


def summarize_near_table(
    states: np.ndarray,
    episode_ids: np.ndarray,
    station: StationCalibration,
    cfg: NearTableConfig,
    side: str,
) -> dict[str, float]:
    vals: list[float] = []
    for state, _ep in zip(states, episode_ids):
        j0, gidx = (0, 6) if side == "left" else (7, 13)
        q = state[j0 : j0 + 6]
        g = float(state[gidx])
        if is_near_table_frame(q, g, side, station, cfg=cfg):
            vals.append(tcp_clearance_mm(q, g, side, station))
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": float(arr.size),
        "min": float(arr.min()),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.median(arr)),
        "mean": float(arr.mean()),
    }


def right_near_table_mask(
    states: np.ndarray,
    station: StationCalibration,
    cfg: NearTableConfig,
) -> np.ndarray:
    mask = np.zeros(states.shape[0], dtype=bool)
    for i, state in enumerate(states):
        mask[i] = is_near_table_frame(state[7:13], float(state[13]), "right", station, cfg=cfg)
    return mask


def smooth_right_near_table_joints(
    states: np.ndarray,
    episode_ids: np.ndarray,
    near_mask: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    if window <= 1:
        return states
    out = states.copy()
    for ep in np.unique(episode_ids):
        idx = np.where((episode_ids == ep) & near_mask)[0]
        if idx.size < 2:
            continue
        traj = out[idx, 7:13].astype(np.float64)
        out[idx, 7:13] = moving_average(traj, window=window).astype(np.float32)
    return out


def write_states(
    paths: list[Path],
    states_list: list[np.ndarray],
    actions_list: list[np.ndarray],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    for path, states, actions in zip(paths, states_list, actions_list, strict=True):
        table = pq.read_table(path)
        arrays = []
        names = []
        for name in table.column_names:
            if name == "observation.state":
                arrays.append(
                    pa.array(
                        [row.astype(np.float32).tolist() for row in states],
                        type=table.schema.field("observation.state").type,
                    )
                )
            elif name == "action":
                arrays.append(
                    pa.array(
                        [row.astype(np.float32).tolist() for row in actions],
                        type=table.schema.field("action").type,
                    )
                )
            else:
                arrays.append(table[name])
            names.append(name)
        pq.write_table(pa.Table.from_arrays(arrays, names=names), path)


def print_stats(
    label: str,
    left: dict[str, float],
    right: dict[str, float],
    left_refs: dict[int, float],
    deltas: dict[int, float] | None = None,
) -> None:
    print(f"\n{label}")
    for side, stats in ("left", left), ("right", right):
        if not stats:
            print(f"  {side} near-table: none")
            continue
        print(
            f"  {side} near-table n={int(stats['n'])}  "
            f"min={stats['min']:.1f}mm  p5={stats['p5']:.1f}mm  "
            f"p50={stats['p50']:.1f}mm  mean={stats['mean']:.1f}mm"
        )
    if left_refs:
        vals = np.asarray(list(left_refs.values()), dtype=np.float64)
        print(
            f"  left contact ref ({len(left_refs)} eps): "
            f"min={vals.min():.1f}mm  median={np.median(vals):.1f}mm  max={vals.max():.1f}mm"
        )
    if deltas:
        d = np.asarray(list(deltas.values()), dtype=np.float64)
        print(
            f"  episode lower delta ({len(deltas)} eps): "
            f"min={d.min():.1f}mm  median={np.median(d):.1f}mm  max={d.max():.1f}mm"
        )


def resolve_episode_allowlist(dataset_root: Path, outcomes_filter: str) -> set[int] | None:
    if outcomes_filter == "all":
        return None
    if outcomes_filter != "success":
        raise ValueError(f"unsupported outcomes_filter={outcomes_filter!r}")
    records = load_outcome_records(dataset_root)
    if not records:
        raise FileNotFoundError(
            f"outcomes_filter=success but no episode_outcomes.jsonl under {dataset_root}"
        )
    n = read_episode_count(dataset_root)
    outcomes = align_outcomes_to_episodes(records, n)
    allow = {ep for ep, label in outcomes.items() if label == "success"}
    print(f"Outcomes filter=success: {len(allow)}/{n} episodes")
    return allow


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    station = StationCalibration.load(args.station)
    if station.table_height_m is None:
        raise ValueError("station calibration missing table_height_m")

    cfg = NearTableConfig(clearance_max_mm=args.max_clearance_mm, grip_max_m=args.grip_max_m)
    paths, states_list, episodes_list = load_all_states(source_root / "data")
    all_states = np.concatenate(states_list, axis=0)
    all_eps = np.concatenate(episodes_list, axis=0)
    allowlist = resolve_episode_allowlist(source_root, args.outcomes_filter)

    left_refs = episode_left_grasp_reference(all_states, all_eps, station, cfg=cfg, stat=args.ref_stat)
    if allowlist is not None:
        left_refs = {ep: v for ep, v in left_refs.items() if ep in allowlist}
    planned = episode_contact_delta_mm(
        all_states,
        all_eps,
        station,
        cfg=cfg,
        stat=args.ref_stat,
        min_delta_mm=args.min_delta_mm,
        max_lower_mm=args.max_lower_mm,
        delta_scale=args.delta_scale,
        episode_allowlist=allowlist,
    )
    before_left = summarize_near_table(all_states, all_eps, station, cfg, "left")
    before_right = summarize_near_table(all_states, all_eps, station, cfg, "right")
    print(
        f"Settings: source={source_root}  ref_stat={args.ref_stat}  "
        f"delta_scale={args.delta_scale}  max_lower={args.max_lower_mm}mm  "
        f"outcomes={args.outcomes_filter}"
    )
    print_stats("Before", before_left, before_right, left_refs, planned)

    adjusted_states, result, ep_deltas = adjust_right_grasp_states(
        all_states,
        all_eps,
        station,
        cfg=cfg,
        stat=args.ref_stat,
        min_delta_mm=args.min_delta_mm,
        max_lower_mm=args.max_lower_mm,
        delta_scale=args.delta_scale,
        episode_allowlist=allowlist,
    )
    near_mask = right_near_table_mask(all_states, station, cfg)
    if allowlist is not None:
        near_mask = near_mask & np.isin(all_eps, list(allowlist))
    if args.smooth_window > 1:
        adjusted_states = smooth_right_near_table_joints(
            adjusted_states,
            all_eps,
            near_mask,
            window=args.smooth_window,
        )
        adjusted_states, clamped = clamp_right_after_smooth(
            adjusted_states,
            all_eps,
            station,
            ep_deltas,
            all_states,
            cfg=cfg,
        )
        if clamped:
            print(f"Post-smooth clamp re-IK frames: {clamped}")

    after_left = summarize_near_table(adjusted_states, all_eps, station, cfg, "left")
    after_right = summarize_near_table(adjusted_states, all_eps, station, cfg, "right")
    print_stats("After", after_left, after_right, left_refs, ep_deltas)
    print(
        f"\nAdjust summary: right_near_table={result.right_near_table_frames}  "
        f"adjusted={result.adjusted_frames}  eps={result.episodes_adjusted}/{result.episodes_adjusted + result.episodes_skipped}  "
        f"ik_fail={result.ik_failures}  "
        f"mean_episode_delta={result.mean_episode_delta_mm:.2f}mm"
    )

    if args.dry_run:
        print("\nDry run only; no files written.")
        return

    if target_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Target exists: {target_root} (pass --overwrite)")
        import shutil

        shutil.rmtree(target_root)

    print(f"\nCopying dataset → {target_root}")
    copy_dataset_tree(source_root, target_root, hardlink_videos=not args.copy_videos)

    offset = 0
    new_states_list: list[np.ndarray] = []
    new_actions_list: list[np.ndarray] = []
    for states in states_list:
        n = states.shape[0]
        chunk_states = adjusted_states[offset : offset + n]
        actions = states.copy()
        actions[:, 7:13] = chunk_states[:, 7:13]
        new_states_list.append(chunk_states)
        new_actions_list.append(actions)
        offset += n

    write_states(
        [target_root / "data" / p.relative_to(source_root / "data") for p in paths],
        new_states_list,
        new_actions_list,
    )

    info_path = target_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["piper_right_grasp_adjust"] = {
        "mode": "constant_offset_align_contact_depth",
        "source_root": str(source_root),
        "ref_arm": "left",
        "target_arm": "right",
        "ref_stat": args.ref_stat,
        "grip_max_m": args.grip_max_m,
        "max_clearance_mm": args.max_clearance_mm,
        "min_delta_mm": args.min_delta_mm,
        "max_lower_mm": args.max_lower_mm,
        "delta_scale": args.delta_scale,
        "smooth_window": args.smooth_window,
        "outcomes_filter": args.outcomes_filter,
        "episode_deltas_mm": {str(k): float(v) for k, v in sorted(ep_deltas.items())},
        "adjusted_frames": result.adjusted_frames,
        "notes": (
            "Per-episode constant Z offset on right near-table frames so contact depth "
            "(pick+place) matches left; trajectory shape preserved. Negative FK is artifact."
        ),
    }
    info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")

    print("Recomputing stats…")
    recompute_numeric_stats(target_root)
    print(f"Done: {target_root}")


if __name__ == "__main__":
    main()
