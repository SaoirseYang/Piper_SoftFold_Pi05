"""Per-arm Soft-Fold → Piper workspace alignment for IK virtual data.

Soft-Fold (Franka) keeps both EEFs in a shared central table frame.
Piper dual bases are ~0.7 m apart, so a single planar SE2 map cannot land
both arms in reachable regions. We therefore center each arm independently:

  piper_xyz = piper_center_arm + S * R @ (sf_xyz - sf_center_arm)

Centers default to Soft-Fold episode percentiles + Piper station touch samples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from softfold.frame_map import PRESETS


@dataclass
class PerArmWorkspaceMap:
    sf_center_left: np.ndarray  # (3,)
    sf_center_right: np.ndarray
    piper_center_left: np.ndarray
    piper_center_right: np.ndarray
    R: np.ndarray  # (2,2) applied to xy deltas
    scale_xy: float = 1.0
    scale_z: float = 1.0
    name: str = "per_arm"

    def _map_one(self, xyz: np.ndarray, sf_c: np.ndarray, piper_c: np.ndarray) -> np.ndarray:
        d = np.asarray(xyz, dtype=np.float64) - sf_c
        out = piper_c.copy()
        out[:2] = piper_c[:2] + self.scale_xy * (self.R @ d[:2])
        out[2] = piper_c[2] + self.scale_z * d[2]
        return out

    def apply_eef20(self, eef: np.ndarray) -> np.ndarray:
        eef = np.asarray(eef, dtype=np.float64).copy()
        single = eef.ndim == 1
        if single:
            eef = eef[None, :]
        for row in eef:
            row[0:3] = self._map_one(row[0:3], self.sf_center_left, self.piper_center_left)
            row[10:13] = self._map_one(row[10:13], self.sf_center_right, self.piper_center_right)
        return eef[0] if single else eef

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sf_center_left": self.sf_center_left.tolist(),
            "sf_center_right": self.sf_center_right.tolist(),
            "piper_center_left": self.piper_center_left.tolist(),
            "piper_center_right": self.piper_center_right.tolist(),
            "R": self.R.tolist(),
            "scale_xy": float(self.scale_xy),
            "scale_z": float(self.scale_z),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerArmWorkspaceMap":
        return cls(
            sf_center_left=np.asarray(data["sf_center_left"], dtype=np.float64),
            sf_center_right=np.asarray(data["sf_center_right"], dtype=np.float64),
            piper_center_left=np.asarray(data["piper_center_left"], dtype=np.float64),
            piper_center_right=np.asarray(data["piper_center_right"], dtype=np.float64),
            R=np.asarray(data.get("R", PRESETS["identity"]), dtype=np.float64),
            scale_xy=float(data.get("scale_xy", 1.0)),
            scale_z=float(data.get("scale_z", 1.0)),
            name=str(data.get("name", "per_arm")),
        )


def piper_centers_from_station(station_json: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(Path(station_json).read_text())
    samples = data.get("table_height_samples") or []
    left, right = [], []
    for s in samples:
        xyz = np.array(s["tcp_world_xyz_m"], dtype=np.float64)
        if s.get("side") == "left":
            left.append(xyz)
        elif s.get("side") == "right":
            right.append(xyz)
    if not left or not right:
        # Fallback: bases + forward reach guess
        arms = data["arms"]
        pl = np.array(arms["left"]["base_xyz_m"], dtype=np.float64) + np.array([0.35, 0.0, 0.12])
        pr = np.array(arms["right"]["base_xyz_m"], dtype=np.float64) + np.array([0.35, 0.0, 0.12])
        return pl, pr
    return np.median(np.stack(left), axis=0), np.median(np.stack(right), axis=0)


def default_per_arm_map(
    *,
    station_json: Path | None = None,
    sf_center_left: tuple[float, float, float] = (0.166, -0.073, 0.340),
    sf_center_right: tuple[float, float, float] = (0.196, 0.035, 0.307),
    scale_xy: float = 0.6,
    scale_z: float = 0.5,
    preset: str = "identity",
) -> PerArmWorkspaceMap:
    if station_json is None:
        here = Path(__file__).resolve()
        station_json = next(
            (p / "configs" / "calibration" / "station.json" for p in here.parents
             if (p / "configs" / "calibration" / "station.json").is_file()),
            here.parents[2] / "configs" / "calibration" / "station.json",
        )
    pl, pr = piper_centers_from_station(station_json)
    # Keep Piper centers slightly above table for safer IK seeds
    pl = pl.copy(); pr = pr.copy()
    pl[2] = max(pl[2], 0.08)
    pr[2] = max(pr[2], 0.08)
    return PerArmWorkspaceMap(
        sf_center_left=np.asarray(sf_center_left, dtype=np.float64),
        sf_center_right=np.asarray(sf_center_right, dtype=np.float64),
        piper_center_left=pl,
        piper_center_right=pr,
        R=PRESETS[preset].astype(np.float64).copy(),
        scale_xy=scale_xy,
        scale_z=scale_z,
        name=f"per_arm[{preset}]",
    )


def load_workspace_map(path: Path | None = None) -> PerArmWorkspaceMap:
    if path is None:
        here = Path(__file__).resolve()
        path = next(
            (p / "configs" / "workspace_map.json" for p in here.parents
             if (p / "configs" / "workspace_map.json").is_file()),
            here.parents[2] / "configs" / "workspace_map.json",
        )
    path = Path(path)
    if path.is_file():
        return PerArmWorkspaceMap.from_dict(json.loads(path.read_text()))
    return default_per_arm_map()


def clamp_eef20_workspace(
    eef20: np.ndarray,
    *,
    center_left: np.ndarray,
    center_right: np.ndarray,
    z_min_m: float = 0.03,
    max_delta_xy_m: float = 0.22,
) -> np.ndarray:
    """Keep remapped EEFs inside a reachable cylinder above the table."""
    out = np.asarray(eef20, dtype=np.float64).copy()
    for sl, center in ((slice(0, 3), center_left), (slice(10, 13), center_right)):
        p = out[sl]
        p[2] = max(float(p[2]), float(z_min_m))
        d = p[:2] - center[:2]
        n = float(np.linalg.norm(d))
        if n > max_delta_xy_m and n > 1e-9:
            p[:2] = center[:2] + d / n * max_delta_xy_m
        out[sl] = p
    return out
