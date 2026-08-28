#!/usr/bin/env python3
"""Probe EEF / TCP height ranges for LeRobot datasets (joint-space or eef6d)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from softfold.grasp_height_adjust import NearTableConfig, is_near_table_frame, tcp_clearance_mm  # noqa: E402
from softfold.kinematics import StationCalibration, eef_in_world  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/agilex/code/yjw/SoftFold/data/lerobot/local/softfold_piper_v2"),
    )
    p.add_argument("--station", type=Path, default=None)
    p.add_argument(
        "--grip-max-m",
        type=float,
        default=None,
        help="If set, require gripper < this (m). Default: ignore grip (near-table only).",
    )
    p.add_argument("--max-clearance-mm", type=float, default=30.0)
    return p.parse_args()


def load_joint_states(dataset_root: Path) -> np.ndarray:
    import pyarrow.parquet as pq

    values = []
    for path in sorted((dataset_root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["observation.state"])
        values.extend(table["observation.state"].to_pylist())
    return np.asarray(values, dtype=np.float64)


def summarize_clearance(clearance_mm: np.ndarray) -> str:
    return (
        f"min={clearance_mm.min():6.1f}  p5={np.percentile(clearance_mm, 5):6.1f}  "
        f"p50={np.median(clearance_mm):6.1f}  mean={clearance_mm.mean():6.1f}  max={clearance_mm.max():6.1f}"
    )


def main() -> None:
    args = parse_args()
    station = StationCalibration.load(args.station)
    cfg = NearTableConfig(grip_max_m=args.grip_max_m, clearance_max_mm=args.max_clearance_mm)

    states = load_joint_states(args.dataset_root)
    print(f"dataset: {args.dataset_root}")
    print(f"frames: {states.shape[0]}")
    print(f"table_height_m: {station.table_height_m}")
    print(f"near-table: clearance≤{cfg.clearance_max_mm}mm  grip_max={cfg.grip_max_m}")

    for side, j0, gidx in [("left", 0, 6), ("right", 7, 13)]:
        clearance = []
        near_clearance = []
        for row in states:
            q = row[j0 : j0 + 6]
            g = float(row[gidx])
            dz = tcp_clearance_mm(q, g, side, station)
            clearance.append(dz)
            if is_near_table_frame(q, g, side, station, cfg=cfg):
                near_clearance.append(dz)
        clearance = np.asarray(clearance, dtype=np.float64)
        near_clearance = np.asarray(near_clearance, dtype=np.float64)
        print(f"\n[{side}] all frames clearance (mm): {summarize_clearance(clearance)}")
        if near_clearance.size:
            print(f"[{side}] near-table (mm):         {summarize_clearance(near_clearance)}")
        else:
            print(f"[{side}] near-table: none")

    # FK sanity poses
    print("\nSample FK poses:")
    qs = [
        np.array([0.0, 1.5, -1.2, 0.0, 0.8, 0.0]),
        np.array([0.3, 2.0, -1.8, 0.0, 1.0, 0.0]),
    ]
    for q in qs:
        left = eef_in_world(q, 0.0, station.base_T("left")).xyz
        right = eef_in_world(q, 0.0, station.base_T("right")).xyz
        print(f"  q={np.round(q, 2)}  Lxyz={np.round(left, 3)}  Rxyz={np.round(right, 3)}")


if __name__ == "__main__":
    main()
