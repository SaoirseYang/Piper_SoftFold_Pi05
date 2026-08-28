"""Piper single-arm forward kinematics (URDF joint chain → EEF).

Chain matches ``piper_description.urdf`` (joint1..joint6 → gripper_base).
Default TCP is the fingertip contact point (joint7 fixed origin along approach).

EEF layouts (X-VLA / soft-fold compatible):
  eef6d  [10] = xyz(3) + rot6d(6) + grip(1)
  rot6d       = first two columns of R, column-major flattened (r00,r10,r20,r01,r11,r21)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import json
import math
import numpy as np

# Fixed SE3 origins from piper_description.urdf (parent→child before joint rotation).
# Each revolute joint rotates about local Z after this origin.
_URDF_JOINT_ORIGINS: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((0.0, 0.0, 0.123), (0.0, 0.0, 0.0)),  # joint1
    ((0.0, 0.0, 0.0), (1.5708, -0.1359, -3.1416)),  # joint2
    ((0.28503, 0.0, 0.0), (0.0, 0.0, -1.7939)),  # joint3
    ((-0.021984, -0.25075, 0.0), (1.5708, 0.0, 0.0)),  # joint4
    ((0.0, 0.0, 0.0), (-1.5708, 0.0, 0.0)),  # joint5
    ((8.8259e-05, -0.091, 0.0), (1.5708, 0.0, 0.0)),  # joint6 → link6/gripper_base
)

# Fingertip TCP relative to gripper_base (joint7 origin at closed mid-point).
DEFAULT_TCP_OFFSET_XYZ_M = (0.0, 0.0, 0.1358)
DEFAULT_TCP_OFFSET_RPY_RAD = (1.5708, 0.0, 0.0)

def _default_station_path() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs" / "calibration" / "station.json"
        if candidate.is_file():
            return candidate
    return here.parents[1] / "configs" / "calibration" / "station.json"


DEFAULT_STATION_PATH = _default_station_path()


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic XYZ (URDF rpy) → 3x3 rotation."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def xyz_rpy_to_T(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    T = np.eye(4, dtype=np.float64)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    return T


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → quaternion (w, x, y, z)."""
    m = np.asarray(R, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """First two columns of R, column-major (matches common VLA / soft-fold packing)."""
    r = np.asarray(R, dtype=np.float64)
    return np.array([r[0, 0], r[1, 0], r[2, 0], r[0, 1], r[1, 1], r[2, 1]], dtype=np.float64)


def matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Intrinsic XYZ euler (roll, pitch, yaw)."""
    r = np.asarray(R, dtype=np.float64)
    pitch = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
    if abs(float(np.cos(pitch))) < 1e-6:
        roll = 0.0
        yaw = np.arctan2(-r[0, 1], r[1, 1])
    else:
        roll = np.arctan2(r[2, 1], r[2, 2])
        yaw = np.arctan2(r[1, 0], r[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float64)


@dataclass(frozen=True)
class ArmBasePose:
    xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def as_T(self) -> np.ndarray:
        return xyz_rpy_to_T(self.xyz_m, self.rpy_rad)


@dataclass
class StationCalibration:
    """Dual-arm station frame + optional table height (world Z of table top)."""

    left_base: ArmBasePose
    right_base: ArmBasePose
    tcp_offset_xyz_m: tuple[float, float, float] = DEFAULT_TCP_OFFSET_XYZ_M
    tcp_offset_rpy_rad: tuple[float, float, float] = DEFAULT_TCP_OFFSET_RPY_RAD
    table_height_m: float | None = None
    table_height_samples: list[dict] | None = None
    notes: str = ""

    def base_T(self, side: str) -> np.ndarray:
        if side == "left":
            return self.left_base.as_T()
        if side == "right":
            return self.right_base.as_T()
        raise ValueError(f"side must be left|right, got {side!r}")

    def tcp_T(self) -> np.ndarray:
        return xyz_rpy_to_T(self.tcp_offset_xyz_m, self.tcp_offset_rpy_rad)

    def to_dict(self) -> dict:
        return {
            "tcp_offset_xyz_m": list(self.tcp_offset_xyz_m),
            "tcp_offset_rpy_rad": list(self.tcp_offset_rpy_rad),
            "arms": {
                "left": {
                    "base_xyz_m": list(self.left_base.xyz_m),
                    "base_rpy_rad": list(self.left_base.rpy_rad),
                },
                "right": {
                    "base_xyz_m": list(self.right_base.xyz_m),
                    "base_rpy_rad": list(self.right_base.rpy_rad),
                },
            },
            "table_height_m": self.table_height_m,
            "table_height_samples": self.table_height_samples or [],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StationCalibration":
        arms = data.get("arms", {})
        left = arms.get("left", {})
        right = arms.get("right", {})
        tcp_xyz = data.get("tcp_offset_xyz_m", DEFAULT_TCP_OFFSET_XYZ_M)
        tcp_rpy = data.get("tcp_offset_rpy_rad", DEFAULT_TCP_OFFSET_RPY_RAD)
        return cls(
            left_base=ArmBasePose(
                xyz_m=tuple(left.get("base_xyz_m", (-0.35, 0.0, 0.0))),
                rpy_rad=tuple(left.get("base_rpy_rad", (0.0, 0.0, 0.0))),
            ),
            right_base=ArmBasePose(
                xyz_m=tuple(right.get("base_xyz_m", (0.35, 0.0, 0.0))),
                rpy_rad=tuple(right.get("base_rpy_rad", (0.0, 0.0, 0.0))),
            ),
            tcp_offset_xyz_m=tuple(tcp_xyz),
            tcp_offset_rpy_rad=tuple(tcp_rpy),
            table_height_m=data.get("table_height_m"),
            table_height_samples=list(data.get("table_height_samples") or []),
            notes=str(data.get("notes") or ""),
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "StationCalibration":
        path = Path(path) if path else DEFAULT_STATION_PATH
        if not path.is_file():
            return cls.default()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else DEFAULT_STATION_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def default(cls) -> "StationCalibration":
        # Defaults mirror sim dual-piper spacing; set base z so table_height is meaningful in world.
        return cls(
            left_base=ArmBasePose(xyz_m=(-0.35, 0.0, 0.0)),
            right_base=ArmBasePose(xyz_m=(0.35, 0.0, 0.0)),
            notes="Edit base_xyz_m to match your dual-arm mount. Calibrate table_height_m with tools/calibrate_table_height.py",
        )


def fk_gripper_base(q: Sequence[float]) -> np.ndarray:
    """FK to gripper_base / link6 frame in the arm base frame. ``q`` length 6 (rad)."""
    if len(q) != 6:
        raise ValueError(f"expected 6 joint angles, got {len(q)}")
    T = np.eye(4, dtype=np.float64)
    for (xyz, rpy), qi in zip(_URDF_JOINT_ORIGINS, q, strict=True):
        T = T @ xyz_rpy_to_T(xyz, rpy) @ rot_z(float(qi))
    return T


def fk_tcp(q: Sequence[float], tcp_T: np.ndarray | None = None) -> np.ndarray:
    """FK to TCP (default: fingertip) in the arm base frame."""
    tip = DEFAULT_TCP_T if tcp_T is None else tcp_T
    return fk_gripper_base(q) @ tip


DEFAULT_TCP_T = xyz_rpy_to_T(DEFAULT_TCP_OFFSET_XYZ_M, DEFAULT_TCP_OFFSET_RPY_RAD)


@dataclass(frozen=True)
class EEFPose:
    """Pose in a chosen frame (arm-base or world)."""

    xyz: np.ndarray  # (3,)
    rot: np.ndarray  # (3,3)
    grip: float

    @property
    def quat_wxyz(self) -> np.ndarray:
        return matrix_to_quat_wxyz(self.rot)

    @property
    def euler_xyz(self) -> np.ndarray:
        return matrix_to_euler_xyz(self.rot)

    @property
    def rot6d(self) -> np.ndarray:
        return matrix_to_rot6d(self.rot)

    def as_eef6d(self) -> np.ndarray:
        return np.concatenate([self.xyz, self.rot6d, np.array([self.grip], dtype=np.float64)])

    def as_xyz_quat_grip(self) -> np.ndarray:
        return np.concatenate([self.xyz, self.quat_wxyz, np.array([self.grip], dtype=np.float64)])

    def as_xyz_euler_grip(self) -> np.ndarray:
        return np.concatenate([self.xyz, self.euler_xyz, np.array([self.grip], dtype=np.float64)])


def pose_from_T(T: np.ndarray, grip: float) -> EEFPose:
    return EEFPose(xyz=np.asarray(T[:3, 3], dtype=np.float64).copy(), rot=np.asarray(T[:3, :3], dtype=np.float64).copy(), grip=float(grip))


def eef_in_base(q: Sequence[float], grip: float, *, tcp_T: np.ndarray | None = None) -> EEFPose:
    return pose_from_T(fk_tcp(q, tcp_T=tcp_T), grip)


def eef_in_world(
    q: Sequence[float],
    grip: float,
    base_T: np.ndarray,
    *,
    tcp_T: np.ndarray | None = None,
) -> EEFPose:
    return pose_from_T(base_T @ fk_tcp(q, tcp_T=tcp_T), grip)


def joints_from_obs(obs: dict, side: str) -> np.ndarray:
    return np.array([obs[f"{side}_joint_{i}.pos"] for i in range(1, 7)], dtype=np.float64)


def grip_from_obs(obs: dict, side: str) -> float:
    return float(obs[f"{side}_gripper.pos"])


def dual_eef6d_from_obs(
    obs: dict,
    station: StationCalibration,
    *,
    frame: str = "world",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (left_eef6d[10], right_eef6d[10]) in world or arm-base frame."""
    tcp_T = station.tcp_T()
    out: list[np.ndarray] = []
    for side in ("left", "right"):
        q = joints_from_obs(obs, side)
        g = grip_from_obs(obs, side)
        if frame == "world":
            pose = eef_in_world(q, g, station.base_T(side), tcp_T=tcp_T)
        elif frame == "base":
            pose = eef_in_base(q, g, tcp_T=tcp_T)
        else:
            raise ValueError("frame must be 'world' or 'base'")
        out.append(pose.as_eef6d())
    return out[0], out[1]


def pack_softfold_style_state(
    obs: dict,
    station: StationCalibration,
    *,
    left_time: float = 0.0,
    right_time: float = 0.0,
) -> np.ndarray:
    """Build a soft-fold-like vector focused on EEF (euler+quat+eef6d+times+qpos).

    Layout (66 floats, subset of soft-fold 96 without qvel/effort):
      eef_euler[14] | eef_quat[16] | eef6d[20] | eef_left_time | eef_right_time | qpos[14]

    XVLA ``max_state_dim=20`` keeps the leading ``eef_euler[14] + eef_quat[:6]``.
    """
    tcp_T = station.tcp_T()
    poses: list[EEFPose] = []
    qpos: list[float] = []
    for side in ("left", "right"):
        q = joints_from_obs(obs, side)
        g = grip_from_obs(obs, side)
        poses.append(eef_in_world(q, g, station.base_T(side), tcp_T=tcp_T))
        qpos.extend(q.tolist())
        qpos.append(g)

    eef_euler = np.concatenate([p.as_xyz_euler_grip() for p in poses])
    eef_quat = np.concatenate([p.as_xyz_quat_grip() for p in poses])
    eef6d = np.concatenate([p.as_eef6d() for p in poses])
    return np.concatenate(
        [
            eef_euler,
            eef_quat,
            eef6d,
            np.array([left_time, right_time], dtype=np.float64),
            np.asarray(qpos, dtype=np.float64),
        ]
    )


def softfold_proprio_from_obs(
    obs: dict,
    station: StationCalibration,
    *,
    dim: int = 20,
) -> np.ndarray:
    """Proprio vector matching soft-fold training truncation (default first 20)."""
    full = pack_softfold_style_state(obs, station)
    if dim <= 0:
        return full
    if full.shape[0] >= dim:
        return full[:dim].copy()
    out = np.zeros(dim, dtype=np.float64)
    out[: full.shape[0]] = full
    return out


def state96_from_eef20_qpos(
    eef20: Sequence[float],
    qpos14: Sequence[float],
    *,
    left_time: float = 0.0,
    right_time: float = 0.0,
    qpos_left_time: float | None = None,
    qpos_right_time: float | None = None,
) -> np.ndarray:
    """Pack soft-fold 96D ``observation.state`` from recorded eef6d + qpos (no FK).

    Layout matches ``lerobot/xvla-soft-fold`` meta:
      eef_euler[14] | eef_quat[16] | eef6d[20] | eef_times[2] | qpos[14]
      | qvel[14]=0 | effort[14]=0 | qpos_times[2]
    """
    eef = np.asarray(eef20, dtype=np.float64).reshape(20)
    qpos = np.asarray(qpos14, dtype=np.float64).reshape(14)
    eulers: list[float] = []
    quats: list[float] = []
    for arm in (0, 1):
        sl = slice(arm * 10, arm * 10 + 10)
        chunk = eef[sl]
        xyz = chunk[:3]
        R = rot6d_to_matrix(chunk[3:9])
        grip = float(chunk[9])
        eulers.extend([*xyz.tolist(), *matrix_to_euler_xyz(R).tolist(), grip])
        quats.extend([*xyz.tolist(), *matrix_to_quat_wxyz(R).tolist(), grip])
    lt = float(left_time)
    rt = float(right_time)
    qlt = float(qpos_left_time if qpos_left_time is not None else lt)
    qrt = float(qpos_right_time if qpos_right_time is not None else rt)
    return np.concatenate(
        [
            np.asarray(eulers, dtype=np.float64),
            np.asarray(quats, dtype=np.float64),
            eef,
            np.array([lt, rt], dtype=np.float64),
            qpos,
            np.zeros(14, dtype=np.float64),  # qvel
            np.zeros(14, dtype=np.float64),  # effort
            np.array([qlt, qrt], dtype=np.float64),
        ]
    )


def current_eef20_from_obs(obs: dict, station: StationCalibration) -> np.ndarray:
    """World-frame dual eef6d[20] from follower joints."""
    left, right = dual_eef6d_from_obs(obs, station, frame="world")
    return np.concatenate([left, right])


def clip_eef20_toward_current(
    predicted: np.ndarray,
    current: np.ndarray,
    *,
    max_pos_step_m: float = 0.02,
    max_rot_step_rad: float = 0.15,
    blend: float = 1.0,
) -> np.ndarray:
    """Limit per-arm XYZ / rotation jump from current eef; optionally blend.

    ``blend`` in (0,1]: 1=full (clipped) prediction, smaller = stick closer to current.
    """
    out = current.copy()
    pred = np.asarray(predicted, dtype=np.float64).reshape(20)
    cur = np.asarray(current, dtype=np.float64).reshape(20)
    blend = float(np.clip(blend, 0.0, 1.0))
    for arm in (0, 1):
        sl = slice(arm * 10, arm * 10 + 10)
        p = pred[sl]
        c = cur[sl]
        # position
        dpos = p[:3] - c[:3]
        n = float(np.linalg.norm(dpos))
        if n > max_pos_step_m > 0:
            dpos = dpos * (max_pos_step_m / n)
        pos = c[:3] + dpos
        # rotation via rot6d → R, interpolate with rotvec clamp
        Rp = rot6d_to_matrix(p[3:9])
        Rc = rot6d_to_matrix(c[3:9])
        R_err = Rc.T @ Rp
        # rotvec of R_err
        cos_theta = float(np.clip(0.5 * (np.trace(R_err) - 1.0), -1.0, 1.0))
        theta = math.acos(cos_theta)
        if theta > 1e-8:
            wx = R_err[2, 1] - R_err[1, 2]
            wy = R_err[0, 2] - R_err[2, 0]
            wz = R_err[1, 0] - R_err[0, 1]
            axis = np.array([wx, wy, wz], dtype=np.float64)
            axis = axis / max(np.linalg.norm(axis), 1e-8)
            theta_use = min(theta, max_rot_step_rad) if max_rot_step_rad > 0 else theta
            # Rodrigues
            K = np.array(
                [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
                dtype=np.float64,
            )
            R_delta = np.eye(3) + math.sin(theta_use) * K + (1 - math.cos(theta_use)) * (K @ K)
            R_tgt = Rc @ R_delta
        else:
            R_tgt = Rc
        grip = c[9] + float(np.clip(p[9] - c[9], -0.05, 0.05))
        blended_pos = (1.0 - blend) * c[:3] + blend * pos
        # blend rotation in rot6d space after clamp
        R_out = Rc if blend < 1e-6 else R_tgt  # already clamped toward target
        if blend < 1.0 - 1e-6 and theta > 1e-8:
            theta_b = theta_use * blend
            R_delta_b = np.eye(3) + math.sin(theta_b) * K + (1 - math.cos(theta_b)) * (K @ K)
            R_out = Rc @ R_delta_b
        out[sl] = np.concatenate([blended_pos, matrix_to_rot6d(R_out), np.array([grip])])
    return out


def eef6d_names(side: str) -> list[str]:
    prefix = f"{side}_eef"
    return [
        f"{prefix}_x",
        f"{prefix}_y",
        f"{prefix}_z",
        f"{prefix}_rot6d_0",
        f"{prefix}_rot6d_1",
        f"{prefix}_rot6d_2",
        f"{prefix}_rot6d_3",
        f"{prefix}_rot6d_4",
        f"{prefix}_rot6d_5",
        f"{prefix}_grip",
    ]


def unified_eef_feature_names() -> list[str]:
    return [*eef6d_names("left"), *eef6d_names("right")]


def batch_fk_xyz(qs: np.ndarray, tcp_T: np.ndarray | None = None) -> np.ndarray:
    """Vectorized convenience: qs (N,6) → xyz (N,3) in arm base."""
    tip = DEFAULT_TCP_T if tcp_T is None else tcp_T
    out = np.zeros((len(qs), 3), dtype=np.float64)
    for i, q in enumerate(qs):
        out[i] = (fk_gripper_base(q) @ tip)[:3, 3]
    return out


# ---------------------------------------------------------------------------
# Inverse kinematics (numerical, matches URDF FK / eef6d layout)
# ---------------------------------------------------------------------------

# Soft joint limits (rad) — conservative Piper-like ranges for IK clamping.
DEFAULT_JOINT_LIMITS = (
    (-2.618, 2.618),
    (0.0, 3.14),
    (-2.967, 0.0),
    (-1.745, 1.745),
    (-1.22, 1.22),
    (-2.0944, 2.0944),
)


def rot6d_to_matrix(rot6d: Sequence[float]) -> np.ndarray:
    """Inverse of ``matrix_to_rot6d``: first two columns → orthonormal R."""
    v = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a1 = v[:3].copy()
    a2 = v[3:].copy()
    n1 = np.linalg.norm(a1)
    if n1 < 1e-8:
        a1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        a1 /= n1
    a2 = a2 - np.dot(a1, a2) * a1
    n2 = np.linalg.norm(a2)
    if n2 < 1e-8:
        # pick any orthogonal axis
        helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(a1, helper)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        a2 = np.cross(a1, helper)
        a2 /= np.linalg.norm(a2)
    else:
        a2 /= n2
    a3 = np.cross(a1, a2)
    return np.stack([a1, a2, a3], axis=1)


def eef6d_to_T(eef10: Sequence[float]) -> np.ndarray:
    """Single-arm eef6d[10] → 4x4 TCP pose (frame of the eef vector)."""
    v = np.asarray(eef10, dtype=np.float64).reshape(10)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot6d_to_matrix(v[3:9])
    T[:3, 3] = v[:3]
    return T


def _rotation_error_vec(R_cur: np.ndarray, R_tgt: np.ndarray) -> np.ndarray:
    """Orientation error as rotation-vector in base frame (rad)."""
    R_err = R_cur.T @ R_tgt
    cos_theta = float(np.clip(0.5 * (np.trace(R_err) - 1.0), -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    # vee(R_err - R_err.T) / (2 sin θ) * θ
    wx = R_err[2, 1] - R_err[1, 2]
    wy = R_err[0, 2] - R_err[2, 0]
    wz = R_err[1, 0] - R_err[0, 1]
    axis = np.array([wx, wy, wz], dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = axis / n
    # express in base: R_cur @ (axis * theta)
    return R_cur @ (axis * theta)


def pose_error_6(T_cur: np.ndarray, T_tgt: np.ndarray) -> np.ndarray:
    err = np.zeros(6, dtype=np.float64)
    err[:3] = T_tgt[:3, 3] - T_cur[:3, 3]
    err[3:] = _rotation_error_vec(T_cur[:3, :3], T_tgt[:3, :3])
    return err


def _clamp_joints(q: np.ndarray, limits: Sequence[tuple[float, float]] = DEFAULT_JOINT_LIMITS) -> np.ndarray:
    out = q.copy()
    for i, (lo, hi) in enumerate(limits):
        out[i] = float(np.clip(out[i], lo, hi))
    return out


def numerical_jacobian_tcp(
    q: np.ndarray,
    *,
    tcp_T: np.ndarray | None = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """6x6 geometric Jacobian of TCP pose wrt joints (finite difference)."""
    tip = DEFAULT_TCP_T if tcp_T is None else tcp_T
    T0 = fk_tcp(q, tcp_T=tip)
    J = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        dq = np.zeros(6, dtype=np.float64)
        dq[i] = eps
        Tp = fk_tcp(q + dq, tcp_T=tip)
        Tm = fk_tcp(q - dq, tcp_T=tip)
        J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2.0 * eps)
        # rotation: average of ± errors / (2 eps) via relative to T0
        ep = _rotation_error_vec(T0[:3, :3], Tp[:3, :3])
        em = _rotation_error_vec(T0[:3, :3], Tm[:3, :3])
        J[3:, i] = (ep - em) / (2.0 * eps)
    return J


def ik_tcp(
    target_T: np.ndarray,
    q0: Sequence[float],
    *,
    tcp_T: np.ndarray | None = None,
    max_iters: int = 60,
    pos_tol_m: float = 1e-3,
    rot_tol_rad: float = 2e-2,
    damp: float = 1e-2,
    step_scale: float = 1.0,
    joint_limits: Sequence[tuple[float, float]] = DEFAULT_JOINT_LIMITS,
) -> tuple[np.ndarray, dict]:
    """Damped least-squares IK for TCP pose in the **arm-base** frame.

    Returns ``(q, info)`` with residual norms and success flag.
    """
    tip = DEFAULT_TCP_T if tcp_T is None else tcp_T
    q = _clamp_joints(np.asarray(q0, dtype=np.float64).reshape(6), joint_limits)
    info: dict = {"success": False, "iters": 0, "pos_err_m": None, "rot_err_rad": None}
    for it in range(max_iters):
        T = fk_tcp(q, tcp_T=tip)
        err = pose_error_6(T, target_T)
        pos_err = float(np.linalg.norm(err[:3]))
        rot_err = float(np.linalg.norm(err[3:]))
        info.update(iters=it + 1, pos_err_m=pos_err, rot_err_rad=rot_err)
        if pos_err < pos_tol_m and rot_err < rot_tol_rad:
            info["success"] = True
            break
        J = numerical_jacobian_tcp(q, tcp_T=tip)
        # dq = J^T (J J^T + λ I)^{-1} err
        jj = J @ J.T + (damp**2) * np.eye(6)
        try:
            dq = J.T @ np.linalg.solve(jj, err)
        except np.linalg.LinAlgError:
            dq = J.T @ (err / max(damp**2, 1e-8))
        q = _clamp_joints(q + step_scale * dq, joint_limits)
    info["q"] = q.tolist()
    return q, info


def dual_eef6d_to_joint_action(
    eef20: Sequence[float],
    q0_left: Sequence[float],
    q0_right: Sequence[float],
    station: StationCalibration,
    *,
    ik_kwargs: dict | None = None,
) -> tuple[dict[str, float], dict]:
    """World-frame dual eef6d[20] → Piper joint action dict (14 keys).

    Grip values are passed through as ``*_gripper.pos`` (meters / dataset units).
    """
    v = np.asarray(eef20, dtype=np.float64).reshape(20)
    tcp_T = station.tcp_T()
    kwargs = dict(ik_kwargs or {})
    action: dict[str, float] = {}
    infos: dict[str, dict] = {}
    for side, eef10, q0 in (
        ("left", v[:10], q0_left),
        ("right", v[10:], q0_right),
    ):
        T_world = eef6d_to_T(eef10)
        # world → arm base
        T_base = np.linalg.inv(station.base_T(side)) @ T_world
        q, info = ik_tcp(T_base, q0, tcp_T=tcp_T, **kwargs)
        infos[side] = info
        for i in range(6):
            action[f"{side}_joint_{i + 1}.pos"] = float(q[i])
        action[f"{side}_gripper.pos"] = float(eef10[9])
    meta = {
        "left": infos["left"],
        "right": infos["right"],
        "success": bool(infos["left"].get("success") and infos["right"].get("success")),
        "max_pos_err_m": max(float(infos["left"]["pos_err_m"] or 0), float(infos["right"]["pos_err_m"] or 0)),
        "max_rot_err_rad": max(float(infos["left"]["rot_err_rad"] or 0), float(infos["right"]["rot_err_rad"] or 0)),
    }
    return action, meta


def eef_action_dict_to_vec(action: dict[str, float]) -> np.ndarray:
    names = unified_eef_feature_names()
    return np.array([float(action[name]) for name in names], dtype=np.float64)
