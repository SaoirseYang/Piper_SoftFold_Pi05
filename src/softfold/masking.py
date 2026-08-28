"""Image masking / augmentation helpers for visual-action consistency."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def mask_robot_region(
    image: np.ndarray,
    *,
    fill: Sequence[int] = (0, 0, 0),
    mask: np.ndarray | None = None,
    top_fraction: float = 0.0,
    side_fraction: float = 0.0,
) -> np.ndarray:
    """Mask robot-looking regions so the model does not bind Aloha looks to Piper joints.

    If ``mask`` is provided (H,W bool/uint8), those pixels are filled.
    Otherwise optional crude geometry masks (top/side bands) can be used as a
    weak baseline before a proper detector/segmentation pipeline is plugged in.
    """
    out = np.array(image, copy=True)
    if out.ndim != 3 or out.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got {out.shape}")
    h, w = out.shape[:2]
    color = np.asarray(fill, dtype=out.dtype).reshape(1, 1, 3)

    if mask is not None:
        m = np.asarray(mask)
        if m.shape[:2] != (h, w):
            raise ValueError(f"mask shape {m.shape} != image {(h, w)}")
        out[m.astype(bool)] = color
        return out

    if top_fraction > 0:
        y1 = int(h * float(top_fraction))
        out[:y1, :, :] = color
    if side_fraction > 0:
        x1 = int(w * float(side_fraction))
        out[:, :x1, :] = color
        out[:, w - x1 :, :] = color
    return out


def random_train_augment(
    image: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    brightness: float = 0.15,
    contrast: float = 0.15,
    saturate_noise: float = 8.0,
    blur_prob: float = 0.1,
) -> np.ndarray:
    """Lightweight RGB augmentations for domain gap (train-time only)."""
    rng = rng or np.random.default_rng()
    x = image.astype(np.float32)
    # brightness / contrast
    b = 1.0 + float(rng.uniform(-brightness, brightness))
    c = 1.0 + float(rng.uniform(-contrast, contrast))
    mean = x.mean(axis=(0, 1), keepdims=True)
    x = (x - mean) * c + mean
    x = x * b
    if saturate_noise > 0:
        x = x + rng.normal(0.0, saturate_noise, size=x.shape)
    x = np.clip(x, 0, 255)
    if rng.random() < blur_prob:
        # cheap 3x3 box blur
        pad = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="edge")
        x = (
            pad[:-2, :-2]
            + pad[:-2, 1:-1]
            + pad[:-2, 2:]
            + pad[1:-1, :-2]
            + pad[1:-1, 1:-1]
            + pad[1:-1, 2:]
            + pad[2:, :-2]
            + pad[2:, 1:-1]
            + pad[2:, 2:]
        ) / 9.0
    return x.astype(image.dtype)
