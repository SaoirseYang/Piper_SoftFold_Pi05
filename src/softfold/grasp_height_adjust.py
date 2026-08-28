"""Align right-arm near-table depth to left-arm contact depth with a constant offset.

Folding includes pick-up, near-table manipulation, and put-down. We do not collapse
every near-table frame to the left arm's deepest clearance. Instead, each episode
compares left/right contact depth (e.g. p5 of near-table clearances) and applies a
uniform Z offset to the right arm's near-table segment so trajectory shape is kept.

Negative FK clearance is treated as a modeling artifact (sideways gripper), not
physical table penetration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from softfold.kinematics import StationCalibration, eef_in_world, ik_tcp

LEFT_JOINT_SLICE = slice(0, 6)
LEFT_GRIP_INDEX = 6
RIGHT_JOINT_SLICE = slice(7, 13)
RIGHT_GRIP_INDEX = 13

RefStat = Literal["min", "p5", "p10", "p50"]


@dataclass(frozen=True)
class NearTableConfig:
    """Near-table = FK clearance ≤ max. Grip filter is optional (None = ignore grip)."""

    clearance_max_mm: float = 30.0
    grip_max_m: float | None = None  # None: pick+place+near-table hold; set e.g. 0.02 to require closed


@dataclass
class AdjustResult:
    frames_total: int = 0
    right_near_table_frames: int = 0
    adjusted_frames: int = 0
    episodes_adjusted: int = 0
    episodes_skipped: int = 0
    ik_failures: int = 0
    mean_lower_mm: float = 0.0
    mean_episode_delta_mm: float = 0.0


# Backward-compatible alias used by older call sites / probe script naming.
GraspMaskConfig = NearTableConfig


def tcp_clearance_mm(
    q: np.ndarray,
    grip: float,
    side: str,
    station: StationCalibration,
) -> float:
    z = eef_in_world(q, grip, station.base_T(side)).xyz[2]
    return float((z - station.table_height_m) * 1000.0)


def is_near_table_frame(
    q: np.ndarray,
    grip: float,
    side: str,
    station: StationCalibration,
    *,
    cfg: NearTableConfig,
) -> bool:
    if tcp_clearance_mm(q, grip, side, station) > cfg.clearance_max_mm:
        return False
    if cfg.grip_max_m is not None and float(grip) >= cfg.grip_max_m:
        return False
    return True


# Alias for probe script / older naming ("grasp" ≈ near-table contact work).
def is_grasp_frame(
    q: np.ndarray,
    grip: float,
    side: str,
    station: StationCalibration,
    *,
    cfg: NearTableConfig,
) -> bool:
    return is_near_table_frame(q, grip, side, station, cfg=cfg)


def ref_clearance_mm(values: np.ndarray, stat: RefStat) -> float:
    if values.size == 0:
        raise ValueError("empty clearance array")
    if stat == "min":
        return float(values.min())
    if stat == "p5":
        return float(np.percentile(values, 5))
    if stat == "p10":
        return float(np.percentile(values, 10))
    if stat == "p50":
        return float(np.percentile(values, 50))
    raise ValueError(f"unsupported stat: {stat}")


def lower_tcp_by_mm(
    q: np.ndarray,
    grip: float,
    side: str,
    station: StationCalibration,
    lower_mm: float,
) -> tuple[np.ndarray, bool]:
    """IK: keep x/y/orientation, lower TCP clearance by ``lower_mm``."""
    cur = tcp_clearance_mm(q, grip, side, station)
    return lower_tcp_to_clearance_mm(q, grip, side, station, cur - lower_mm)


def lower_tcp_to_clearance_mm(
    q: np.ndarray,
    grip: float,
    side: str,
    station: StationCalibration,
    target_clearance_mm: float,
) -> tuple[np.ndarray, bool]:
    """IK: keep x/y/orientation, set FK clearance to target_clearance_mm."""
    pose = eef_in_world(q, grip, station.base_T(side))
    target_z = float(station.table_height_m) + target_clearance_mm / 1000.0

    T_world = np.eye(4, dtype=np.float64)
    T_world[:3, :3] = pose.rot
    T_world[:3, 3] = pose.xyz
    T_world[2, 3] = target_z

    T_base = np.linalg.inv(station.base_T(side)) @ T_world
    q_new, info = ik_tcp(T_base, q, tcp_T=station.tcp_T())
    return q_new, bool(info.get("success"))


def episode_near_table_clearances(
    states: np.ndarray,
    episode_ids: np.ndarray,
    station: StationCalibration,
    *,
    cfg: NearTableConfig,
    side: str,
) -> dict[int, np.ndarray]:
    j0, gidx = (0, 6) if side == "left" else (7, 13)
    buckets: dict[int, list[float]] = {}
    for state, ep in zip(states, episode_ids):
        q = state[j0 : j0 + 6]
        g = float(state[gidx])
        if not is_near_table_frame(q, g, side, station, cfg=cfg):
            continue
        buckets.setdefault(int(ep), []).append(tcp_clearance_mm(q, g, side, station))
    return {ep: np.asarray(vals, dtype=np.float64) for ep, vals in buckets.items()}


def episode_contact_delta_mm(
    states: np.ndarray,
    episode_ids: np.ndarray,
    station: StationCalibration,
    *,
    cfg: NearTableConfig,
    stat: RefStat,
    min_delta_mm: float,
    max_lower_mm: float,
    delta_scale: float = 1.0,
    episode_allowlist: set[int] | None = None,
) -> dict[int, float]:
    """Per episode: how many mm to lower right so its contact depth matches left.

    ``delta > 0`` means lower the right arm by that many mm (constant offset).
    ``delta_scale`` (e.g. 0.5) applies only a fraction of the measured gap.
    """
    left = episode_near_table_clearances(states, episode_ids, station, cfg=cfg, side="left")
    right = episode_near_table_clearances(states, episode_ids, station, cfg=cfg, side="right")
    scale = float(np.clip(delta_scale, 0.0, 1.0))
    deltas: dict[int, float] = {}
    for ep in sorted(set(left) | set(right)):
        if episode_allowlist is not None and ep not in episode_allowlist:
            continue
        if ep not in left or ep not in right:
            continue
        left_ref = ref_clearance_mm(left[ep], stat)
        right_ref = ref_clearance_mm(right[ep], stat)
        raw = (right_ref - left_ref) * scale  # >0 => right higher / shallower
        if raw < min_delta_mm:
            continue
        deltas[ep] = float(min(raw, max_lower_mm))
    return deltas


def episode_left_grasp_reference(
    states: np.ndarray,
    episode_ids: np.ndarray,
    station: StationCalibration,
    *,
    cfg: NearTableConfig,
    stat: RefStat,
) -> dict[int, float]:
    """Left near-table contact depth per episode (for reporting)."""
    left = episode_near_table_clearances(states, episode_ids, station, cfg=cfg, side="left")
    return {ep: ref_clearance_mm(vals, stat) for ep, vals in left.items()}


def clamp_right_after_smooth(
    states: np.ndarray,
    episode_ids: np.ndarray,
    station: StationCalibration,
    ep_delta_mm: dict[int, float],
    original_states: np.ndarray,
    *,
    cfg: NearTableConfig,
    overshoot_mm: float = 2.0,
) -> tuple[np.ndarray, int]:
    """Re-IK if smoothing drove a frame deeper than original - delta - overshoot."""
    out = states.copy()
    fixed = 0
    for i, (state, ep) in enumerate(zip(out, episode_ids)):
        delta = ep_delta_mm.get(int(ep))
        if delta is None or delta <= 0:
            continue
        q = state[RIGHT_JOINT_SLICE].astype(np.float64)
        g = float(state[RIGHT_GRIP_INDEX])
        if not is_near_table_frame(q, g, "right", station, cfg=cfg):
            continue
        q0 = original_states[i, RIGHT_JOINT_SLICE].astype(np.float64)
        g0 = float(original_states[i, RIGHT_GRIP_INDEX])
        floor = tcp_clearance_mm(q0, g0, "right", station) - delta - overshoot_mm
        cur = tcp_clearance_mm(q, g, "right", station)
        if cur >= floor:
            continue
        q_new, ok = lower_tcp_to_clearance_mm(q, g, "right", station, floor)
        if not ok:
            continue
        out[i, RIGHT_JOINT_SLICE] = q_new.astype(np.float32)
        fixed += 1
    return out, fixed


def clamp_right_grasp_to_reference(
    states: np.ndarray,
    episode_ids: np.ndarray,
    station: StationCalibration,
    ep_ref: dict[int, float],
    *,
    cfg: NearTableConfig,
    overshoot_mm: float = 2.0,
) -> tuple[np.ndarray, int]:
    """Legacy clamp: do not go deeper than absolute episode left contact ref."""
    out = states.copy()
    fixed = 0
    for i, (state, ep) in enumerate(zip(out, episode_ids)):
        q = state[RIGHT_JOINT_SLICE].astype(np.float64)
        g = float(state[RIGHT_GRIP_INDEX])
        if not is_near_table_frame(q, g, "right", station, cfg=cfg):
            continue
        ref = ep_ref.get(int(ep))
        if ref is None:
            continue
        cur = tcp_clearance_mm(q, g, "right", station)
        if cur >= ref - overshoot_mm:
            continue
        q_new, ok = lower_tcp_to_clearance_mm(q, g, "right", station, ref)
        if not ok:
            continue
        out[i, RIGHT_JOINT_SLICE] = q_new.astype(np.float32)
        fixed += 1
    return out, fixed


def adjust_right_grasp_states(
    states: np.ndarray,
    episode_ids: np.ndarray,
    station: StationCalibration,
    *,
    cfg: NearTableConfig = NearTableConfig(),
    stat: RefStat = "p5",
    min_delta_mm: float = 0.5,
    max_lower_mm: float = 5.0,
    delta_scale: float = 0.5,
    episode_allowlist: set[int] | None = None,
    global_ref_mm: float | None = None,  # unused; kept for call-site compatibility
) -> tuple[np.ndarray, AdjustResult, dict[int, float]]:
    """Lower right near-table frames by a per-episode constant offset.

    Returns ``(states, result, ep_delta_mm)`` where ``ep_delta_mm[ep] > 0`` is the
    uniform lowering applied to that episode's right near-table segment.
    """
    del global_ref_mm
    out = np.asarray(states, dtype=np.float32).copy()
    result = AdjustResult(frames_total=int(states.shape[0]))

    ep_delta = episode_contact_delta_mm(
        states,
        episode_ids,
        station,
        cfg=cfg,
        stat=stat,
        min_delta_mm=min_delta_mm,
        max_lower_mm=max_lower_mm,
        delta_scale=delta_scale,
        episode_allowlist=episode_allowlist,
    )
    result.episodes_adjusted = len(ep_delta)
    if episode_allowlist is not None:
        candidate_eps = set(episode_allowlist)
    else:
        candidate_eps = {int(ep) for ep in episode_ids}
    result.episodes_skipped = len(candidate_eps) - len(ep_delta)
    if ep_delta:
        result.mean_episode_delta_mm = float(np.mean(list(ep_delta.values())))

    lowered: list[float] = []
    for i, (state, ep) in enumerate(zip(out, episode_ids)):
        if episode_allowlist is not None and int(ep) not in episode_allowlist:
            continue
        q = state[RIGHT_JOINT_SLICE].astype(np.float64)
        g = float(state[RIGHT_GRIP_INDEX])
        if not is_near_table_frame(q, g, "right", station, cfg=cfg):
            continue
        result.right_near_table_frames += 1

        delta = ep_delta.get(int(ep))
        if delta is None or delta <= 0:
            continue

        q_new, ok = lower_tcp_by_mm(q, g, "right", station, delta)
        if not ok:
            result.ik_failures += 1
            continue

        out[i, RIGHT_JOINT_SLICE] = q_new.astype(np.float32)
        result.adjusted_frames += 1
        lowered.append(delta)

    if lowered:
        result.mean_lower_mm = float(np.mean(lowered))
    return out, result, ep_delta
