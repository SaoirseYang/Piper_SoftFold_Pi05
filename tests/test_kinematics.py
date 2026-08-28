"""Unit tests for Piper IK / smoothing / workspace map (no GPU required)."""

from __future__ import annotations

import numpy as np

from softfold.kinematics import (
    StationCalibration,
    dual_eef6d_to_joint_action,
    eef_in_world,
    fk_tcp,
    ik_tcp,
)
from softfold.smoothing import moving_average
from softfold.workspace_map import clamp_eef20_workspace, default_per_arm_map


def test_fk_ik_roundtrip():
    q0 = np.array([0.1, 1.2, -1.0, 0.2, 0.5, -0.1], dtype=np.float64)
    T = fk_tcp(q0)
    q, info = ik_tcp(T, q0 + 0.15)
    assert info["success"], info
    assert info["pos_err_m"] < 2e-3


def test_dual_ik_at_home():
    station = StationCalibration.load()
    q0 = np.array([0.0, 1.8, -1.5, 0.0, 0.9, 0.0], dtype=np.float64)
    left = eef_in_world(q0, 0.0, station.base_T("left"))
    right = eef_in_world(q0, 0.0, station.base_T("right"))
    eef20 = np.concatenate([left.as_eef6d(), right.as_eef6d()])
    action, meta = dual_eef6d_to_joint_action(eef20, q0, q0, station)
    assert meta["success"], meta
    assert len(action) == 14


def test_workspace_map_centers_separated():
    wm = default_per_arm_map()
    assert wm.piper_center_right[0] - wm.piper_center_left[0] > 0.4


def test_clamp_keeps_z_min():
    wm = default_per_arm_map()
    eef = np.zeros(20, dtype=np.float64)
    eef[2] = -0.5
    eef[12] = -0.5
    out = clamp_eef20_workspace(
        eef,
        center_left=wm.piper_center_left,
        center_right=wm.piper_center_right,
        z_min_m=0.03,
        max_delta_xy_m=0.22,
    )
    assert out[2] >= 0.03 and out[12] >= 0.03


def test_moving_average_shape():
    traj = np.random.randn(100, 12)
    out = moving_average(traj, window=5)
    assert out.shape == traj.shape
