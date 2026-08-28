"""JPEG encode/decode for async observation image transport."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

JPEG_MARKER = "__piper_jpeg_v1__"


def is_rgb_image_array(value: Any) -> bool:
    return (
        isinstance(value, np.ndarray)
        and value.ndim == 3
        and value.shape[-1] in (3, 4)
        and value.size > 0
    )


def is_jpeg_payload(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get(JPEG_MARKER)) and "data" in value


def encode_image_jpeg(image: np.ndarray, quality: int = 85) -> dict[str, Any]:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(image[..., :3])
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 1, 100))],
    )
    if not ok:
        raise RuntimeError("Failed to JPEG-encode observation image.")
    return {JPEG_MARKER: True, "data": buffer.tobytes()}


def decode_image_jpeg(payload: dict[str, Any]) -> np.ndarray:
    data = payload.get("data")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("JPEG payload missing bytes in 'data'.")
    encoded = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to JPEG-decode observation image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compress_observation_images(
    observation: dict[str, Any],
    *,
    quality: int = 85,
) -> dict[str, Any]:
    """Replace HWC uint8 image arrays with JPEG payloads. Non-image keys unchanged."""
    compressed: dict[str, Any] = {}
    for key, value in observation.items():
        if is_rgb_image_array(value):
            compressed[key] = encode_image_jpeg(value, quality=quality)
        else:
            compressed[key] = value
    return compressed


def decompress_observation_images(observation: dict[str, Any]) -> dict[str, Any]:
    """Decode JPEG payloads back to RGB uint8 arrays. Pass through raw arrays."""
    decompressed: dict[str, Any] = {}
    for key, value in observation.items():
        if is_jpeg_payload(value):
            decompressed[key] = decode_image_jpeg(value)
        else:
            decompressed[key] = value
    return decompressed


def observation_has_jpeg_payloads(observation: dict[str, Any]) -> bool:
    return any(is_jpeg_payload(value) for value in observation.values())
