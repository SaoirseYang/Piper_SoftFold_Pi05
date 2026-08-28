"""Soft-Fold ↔ Piper planar frame mapping (XY rotation/reflection + Z offset).

The align toolkit maps *local Piper → Soft-Fold-like* coordinates. For IK virtual
data we need the inverse: Soft-Fold EEF → Piper world frame before calling IK.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Presets: 2x2 applied to (x,y). Same names as piper compare_eef_softfold_local.
PRESETS: dict[str, np.ndarray] = {
    "identity": np.array([[1.0, 0.0], [0.0, 1.0]]),
    "swap_xy": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "y_as_x": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "neg_y_as_x": np.array([[0.0, -1.0], [1.0, 0.0]]),
    "rot90_ccw": np.array([[0.0, -1.0], [1.0, 0.0]]),
    "rot90_cw": np.array([[0.0, 1.0], [-1.0, 0.0]]),
    "flip_x": np.array([[-1.0, 0.0], [0.0, 1.0]]),
    "flip_y": np.array([[1.0, 0.0], [0.0, -1.0]]),
}


@dataclass
class XyzMap:
    """Forward map: source_xyz -> target_xyz  (xy' = R @ xy + t, z' = z + z_offset)."""

    R: np.ndarray
    t_xy: np.ndarray
    z_offset: float
    name: str = "custom"

    def apply_xyz(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64)
        out = xyz.copy()
        if xyz.ndim == 1:
            out[:2] = self.R @ xyz[:2] + self.t_xy
            out[2] = xyz[2] + self.z_offset
            return out
        out[:, :2] = (self.R @ xyz[:, :2].T).T + self.t_xy
        out[:, 2] = xyz[:, 2] + self.z_offset
        return out

    def apply_eef20(self, eef: np.ndarray) -> np.ndarray:
        eef = np.asarray(eef, dtype=np.float64).copy()
        single = eef.ndim == 1
        if single:
            eef = eef[None, :]
        eef[:, 0:3] = self.apply_xyz(eef[:, 0:3])
        eef[:, 10:13] = self.apply_xyz(eef[:, 10:13])
        return eef[0] if single else eef

    def inverse(self) -> "XyzMap":
        """Invert planar map: xy = R^{-1}(xy' - t), z = z' - z_offset."""
        R_inv = np.linalg.inv(self.R)
        t_inv = -R_inv @ self.t_xy
        return XyzMap(R=R_inv, t_xy=t_inv, z_offset=-self.z_offset, name=f"inv({self.name})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "R": self.R.tolist(),
            "t_xy": self.t_xy.tolist(),
            "z_offset": float(self.z_offset),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "XyzMap":
        return cls(
            R=np.asarray(data["R"], dtype=np.float64),
            t_xy=np.asarray(data["t_xy"], dtype=np.float64).reshape(2),
            z_offset=float(data["z_offset"]),
            name=str(data.get("name", "custom")),
        )

    @classmethod
    def from_preset(
        cls,
        name: str,
        *,
        t_xy: tuple[float, float] | np.ndarray = (0.0, 0.0),
        z_offset: float = 0.0,
    ) -> "XyzMap":
        if name not in PRESETS:
            raise KeyError(f"unknown preset {name}; choose from {sorted(PRESETS)}")
        return cls(
            R=PRESETS[name].astype(np.float64).copy(),
            t_xy=np.asarray(t_xy, dtype=np.float64).reshape(2),
            z_offset=float(z_offset),
            name=name,
        )


def load_sf_to_piper_map(path: Path | None = None) -> XyzMap:
    """Load Soft-Fold→Piper map.

    Config stores the *forward* local→SF mapper (as produced by align tools);
    this helper returns its inverse for IK conversion.
    """
    if path is None:
        here = Path(__file__).resolve()
        path = next(
            (p / "configs" / "frame_map.json" for p in here.parents
             if (p / "configs" / "frame_map.json").is_file()),
            here.parents[2] / "configs" / "frame_map.json",
        )
    path = Path(path)
    data = json.loads(path.read_text())
    # Prefer explicit sf_to_piper block; else invert local_to_sf.
    if "sf_to_piper" in data:
        return XyzMap.from_dict(data["sf_to_piper"])
    if "local_to_sf" in data:
        return XyzMap.from_dict(data["local_to_sf"]).inverse()
    if "mapper" in data:
        return XyzMap.from_dict(data["mapper"]).inverse()
    return XyzMap.from_dict(data).inverse()
