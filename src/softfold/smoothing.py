"""Joint trajectory smoothing utilities."""

from __future__ import annotations

import numpy as np


def moving_average(traj: np.ndarray, window: int = 5) -> np.ndarray:
    """Low-pass joint trajectories along time axis. ``traj`` shape (T, D)."""
    if window <= 1:
        return np.asarray(traj, dtype=np.float64)
    x = np.asarray(traj, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (T,D), got {x.shape}")
    w = int(window)
    kernel = np.ones(w, dtype=np.float64) / w
    out = np.zeros_like(x)
    for d in range(x.shape[1]):
        pad = w // 2
        padded = np.pad(x[:, d], (pad, pad), mode="edge")
        out[:, d] = np.convolve(padded, kernel, mode="valid")[: x.shape[0]]
    return out


def ema_smooth(prev: np.ndarray, cur: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Exponential moving average for online inference smoothing."""
    a = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - a) * np.asarray(prev, dtype=np.float64) + a * np.asarray(cur, dtype=np.float64)


def normalize_joints(q: np.ndarray, limits: list[tuple[float, float]]) -> np.ndarray:
    """Map joint angles linearly to [-1, 1] given per-dim (lo, hi)."""
    q = np.asarray(q, dtype=np.float64)
    out = np.zeros_like(q)
    for i, (lo, hi) in enumerate(limits):
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        out[..., i] = np.clip((q[..., i] - mid) / max(half, 1e-8), -1.0, 1.0)
    return out


def denormalize_joints(qn: np.ndarray, limits: list[tuple[float, float]]) -> np.ndarray:
    qn = np.asarray(qn, dtype=np.float64)
    out = np.zeros_like(qn)
    for i, (lo, hi) in enumerate(limits):
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        out[..., i] = qn[..., i] * half + mid
    return out
