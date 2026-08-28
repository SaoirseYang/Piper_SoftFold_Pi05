from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .recorder import ARM_STATE_KEYS

try:
    import cv2
except ImportError:
    cv2 = None

GRIPPER_INDICES = (6, 13)


@dataclass
class OrangeBoostConfig:
    enabled: bool = False
    hue_min: int = 5
    hue_max: int = 25
    sat_scale: float = 1.2
    val_scale: float = 1.05


@dataclass
class ImagePreprocessingConfig:
    enabled: bool = False
    white_balance: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    orange_boost: OrangeBoostConfig = field(default_factory=OrangeBoostConfig)


@dataclass
class SmoothingConfig:
    enabled: bool = False
    method: str = "ema"
    ema_alpha: float = 0.25
    gripper_ema_alpha: float | None = 0.35
    savgol_window: int = 7
    savgol_polyorder: int = 2


@dataclass
class DualRecordConfig:
    enabled: bool = False
    augmented_repo_id: str | None = None


@dataclass
class PreprocessingConfig:
    enabled: bool = False
    images: ImagePreprocessingConfig = field(default_factory=ImagePreprocessingConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    dual_record: DualRecordConfig = field(default_factory=DualRecordConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PreprocessingConfig:
        if not data:
            return cls()

        images_data = data.get("images", {})
        if images_data is None:
            images_data = {}
        orange_data = images_data.get("orange_boost", {})
        if orange_data is None:
            orange_data = {}
        hue_range = orange_data.get("hue_range", [5, 25])
        if not isinstance(hue_range, list) or len(hue_range) != 2:
            hue_range = [5, 25]

        smoothing_data = data.get("smoothing", {})
        if smoothing_data is None:
            smoothing_data = {}
        dual_record_data = data.get("dual_record", False)
        if isinstance(dual_record_data, bool):
            dual_record = DualRecordConfig(enabled=dual_record_data)
        elif isinstance(dual_record_data, dict):
            dual_record = DualRecordConfig(
                enabled=bool(dual_record_data.get("enabled", False)),
                augmented_repo_id=(
                    str(dual_record_data["augmented_repo_id"])
                    if dual_record_data.get("augmented_repo_id")
                    else None
                ),
            )
        else:
            dual_record = DualRecordConfig()

        return cls(
            enabled=bool(data.get("enabled", False)),
            images=ImagePreprocessingConfig(
                enabled=bool(images_data.get("enabled", False)),
                white_balance=bool(images_data.get("white_balance", True)),
                clahe_clip_limit=float(images_data.get("clahe_clip_limit", 2.0)),
                clahe_tile_grid_size=int(images_data.get("clahe_tile_grid_size", 8)),
                orange_boost=OrangeBoostConfig(
                    enabled=bool(orange_data.get("enabled", False)),
                    hue_min=int(hue_range[0]),
                    hue_max=int(hue_range[1]),
                    sat_scale=float(orange_data.get("sat_scale", 1.2)),
                    val_scale=float(orange_data.get("val_scale", 1.05)),
                ),
            ),
            smoothing=SmoothingConfig(
                enabled=bool(smoothing_data.get("enabled", False)),
                method=str(smoothing_data.get("method", "ema")),
                ema_alpha=float(smoothing_data.get("ema_alpha", 0.25)),
                gripper_ema_alpha=(
                    float(smoothing_data["gripper_ema_alpha"])
                    if smoothing_data.get("gripper_ema_alpha") is not None
                    else 0.35
                ),
                savgol_window=int(smoothing_data.get("savgol_window", 7)),
                savgol_polyorder=int(smoothing_data.get("savgol_polyorder", 2)),
            ),
            dual_record=dual_record,
        )

    @property
    def active(self) -> bool:
        return self.enabled and (self.images.enabled or self.smoothing.enabled)

    @property
    def dual_record_active(self) -> bool:
        return self.active and self.dual_record.enabled


def load_preprocessing_config(data: dict[str, Any] | None) -> PreprocessingConfig:
    return PreprocessingConfig.from_dict(data)


def resolve_augmented_repo_id(repo_id: str, augmented_repo_id: str | None = None) -> str:
    if augmented_repo_id:
        return augmented_repo_id
    if "/" in repo_id:
        prefix, name = repo_id.rsplit("/", 1)
        return f"{prefix}/{name}_pp"
    return f"{repo_id}_pp"


def numpy_image_from_sample(image: Any) -> np.ndarray:
    if isinstance(image, np.ndarray):
        array = image
    else:
        try:
            import torch
        except ImportError as exc:
            raise TypeError(f"Unsupported image type: {type(image)!r}") from exc

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Unsupported image type: {type(image)!r}")
        array = image.detach().cpu().numpy()

    if array.ndim != 3:
        raise ValueError(f"Expected HWC or CHW image, got shape {array.shape}.")

    if array.shape[0] == 3 and array.shape[2] != 3:
        array = np.transpose(array, (1, 2, 0))

    if np.issubdtype(array.dtype, np.floating):
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    else:
        array = array.astype(np.uint8, copy=False)

    return np.ascontiguousarray(array)


def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    means = image.reshape(-1, 3).mean(axis=0)
    gray_mean = float(means.mean())
    scale = gray_mean / np.maximum(means, 1.0)
    balanced = image.astype(np.float32) * scale
    return np.clip(balanced, 0.0, 255.0).astype(np.uint8)


def apply_clahe_rgb(image: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    if cv2 is None:
        raise ImportError("opencv-python is required for CLAHE image preprocessing.")

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, green_red, blue_yellow = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    enhanced = clahe.apply(lightness)
    merged = cv2.merge([enhanced, green_red, blue_yellow])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def boost_orange_hsv(
    image: np.ndarray,
    hue_min: int,
    hue_max: int,
    sat_scale: float,
    val_scale: float,
) -> np.ndarray:
    if cv2 is None:
        raise ImportError("opencv-python is required for HSV orange boost.")

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    mask = (hue >= hue_min) & (hue <= hue_max)
    saturation[mask] = np.clip(saturation[mask] * sat_scale, 0.0, 255.0)
    value[mask] = np.clip(value[mask] * val_scale, 0.0, 255.0)
    hsv[..., 1] = saturation
    hsv[..., 2] = value
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def enhance_image(image: np.ndarray, config: ImagePreprocessingConfig) -> np.ndarray:
    if not config.enabled:
        return image

    output = numpy_image_from_sample(image)
    if config.white_balance:
        output = gray_world_white_balance(output)
    if config.clahe_clip_limit > 0.0:
        output = apply_clahe_rgb(
            output,
            clip_limit=config.clahe_clip_limit,
            tile_grid_size=config.clahe_tile_grid_size,
        )
    if config.orange_boost.enabled:
        boost = config.orange_boost
        output = boost_orange_hsv(
            output,
            hue_min=boost.hue_min,
            hue_max=boost.hue_max,
            sat_scale=boost.sat_scale,
            val_scale=boost.val_scale,
        )
    return output


def ema_vector_step(
    current: np.ndarray,
    previous: np.ndarray | None,
    alpha: float,
    gripper_alpha: float | None = None,
) -> np.ndarray:
    if previous is None:
        return current.astype(np.float32, copy=True)

    smoothed = previous.astype(np.float32) * (1.0 - alpha) + current.astype(np.float32) * alpha
    if gripper_alpha is not None:
        for index in GRIPPER_INDICES:
            smoothed[index] = (
                previous[index] * (1.0 - gripper_alpha) + current[index] * gripper_alpha
            )
    return smoothed.astype(np.float32)


def _normalize_savgol_window(window: int, length: int, polyorder: int) -> int:
    window = max(polyorder + 1, window)
    if window % 2 == 0:
        window += 1
    if length < window:
        window = length if length % 2 == 1 else max(1, length - 1)
    return max(1, window)


def savgol_smooth_series(
    values: np.ndarray,
    window: int,
    polyorder: int,
) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("Expected a 2D array shaped (frames, dims).")

    frame_count, dim_count = values.shape
    if frame_count <= 1:
        return values.astype(np.float32, copy=True)

    window = _normalize_savgol_window(window, frame_count, polyorder)
    if window <= polyorder:
        return values.astype(np.float32, copy=True)

    try:
        from scipy.signal import savgol_filter
    except ImportError:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        smoothed = np.stack(
            [np.convolve(values[:, index], kernel, mode="same") for index in range(dim_count)],
            axis=1,
        )
        return smoothed.astype(np.float32)

    return savgol_filter(values, window_length=window, polyorder=polyorder, axis=0).astype(np.float32)


def smooth_series(
    values: np.ndarray,
    config: SmoothingConfig,
    *,
    force_method: str | None = None,
) -> np.ndarray:
    if not config.enabled or values.size == 0:
        return values.astype(np.float32, copy=True)

    method = force_method or config.method
    if method == "savgol":
        return savgol_smooth_series(values, config.savgol_window, config.savgol_polyorder)

    smoothed = values.astype(np.float32, copy=True)
    previous = smoothed[0].copy()
    for index in range(1, len(smoothed)):
        previous = ema_vector_step(
            smoothed[index],
            previous,
            config.ema_alpha,
            config.gripper_ema_alpha,
        )
        smoothed[index] = previous
    return smoothed


def vector_to_state_dict(values: np.ndarray) -> dict[str, float]:
    return {key: float(values[index]) for index, key in enumerate(ARM_STATE_KEYS)}


def state_dict_to_vector(values: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(values.get(key, 0.0)) for key in ARM_STATE_KEYS], dtype=np.float32)


def apply_state_dict(vector: np.ndarray, values: dict[str, Any]) -> dict[str, Any]:
    updated = dict(values)
    for index, key in enumerate(ARM_STATE_KEYS):
        updated[key] = float(vector[index])
    return updated


class FramePreprocessor:
    def __init__(self, config: PreprocessingConfig, *, mode: str = "online") -> None:
        if mode not in {"online", "batch"}:
            raise ValueError("mode must be 'online' or 'batch'.")
        self.config = config
        self.mode = mode
        self._previous_state: np.ndarray | None = None
        self._previous_action: np.ndarray | None = None
        self._episode_frames: list[dict[str, Any]] = []

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, mode: str = "online") -> FramePreprocessor:
        return cls(load_preprocessing_config(data), mode=mode)

    @property
    def enabled(self) -> bool:
        return self.config.active

    def reset(self) -> None:
        self._previous_state = None
        self._previous_action = None
        self._episode_frames = []

    def process_images(
        self,
        observation: dict[str, Any],
        camera_names: list[str],
    ) -> dict[str, Any]:
        if not self.config.enabled or not self.config.images.enabled:
            return observation

        updated = dict(observation)
        for camera_name in camera_names:
            image = observation.get(camera_name)
            if not isinstance(image, np.ndarray):
                continue
            updated[camera_name] = enhance_image(image, self.config.images)
        return updated

    def process_observation(
        self,
        observation: dict[str, Any],
        camera_names: list[str],
    ) -> dict[str, Any]:
        if not self.enabled:
            return observation

        updated = self.process_images(observation, camera_names)
        if not self.config.smoothing.enabled:
            return updated

        state_vector = state_dict_to_vector(updated)
        if self.mode == "online":
            state_vector = ema_vector_step(
                state_vector,
                self._previous_state,
                self.config.smoothing.ema_alpha,
                self.config.smoothing.gripper_ema_alpha,
            )
            self._previous_state = state_vector.copy()
        return apply_state_dict(state_vector, updated)

    def process_frame(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
        camera_names: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.enabled:
            return observation, action

        if self.mode == "batch":
            self._episode_frames.append(
                {
                    "observation": dict(observation),
                    "action": dict(action),
                    "camera_names": list(camera_names),
                }
            )
            return observation, action

        processed_observation = self.process_observation(observation, camera_names)
        processed_action = dict(action)
        if self.config.smoothing.enabled:
            action_vector = state_dict_to_vector(action)
            action_vector = ema_vector_step(
                action_vector,
                self._previous_action,
                self.config.smoothing.ema_alpha,
                self.config.smoothing.gripper_ema_alpha,
            )
            self._previous_action = action_vector.copy()
            processed_action = vector_to_state_dict(action_vector)
        return processed_observation, processed_action

    def flush_episode(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        if not self._episode_frames:
            return []

        camera_names = self._episode_frames[0]["camera_names"]
        observations = [frame["observation"] for frame in self._episode_frames]
        actions = [frame["action"] for frame in self._episode_frames]

        state_series = np.stack([state_dict_to_vector(observation) for observation in observations])
        action_series = np.stack([state_dict_to_vector(action) for action in actions])

        smoothing_method = "savgol" if self.config.smoothing.method != "ema" else "ema"
        if self.mode == "batch" and self.config.smoothing.enabled:
            smoothing_method = "savgol"

        if self.config.smoothing.enabled:
            state_series = smooth_series(
                state_series,
                self.config.smoothing,
                force_method=smoothing_method,
            )
            action_series = smooth_series(
                action_series,
                self.config.smoothing,
                force_method=smoothing_method,
            )

        processed_frames: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for index, (observation, action) in enumerate(zip(observations, actions, strict=True)):
            processed_observation = apply_state_dict(state_series[index], observation)
            processed_action = vector_to_state_dict(action_series[index])
            if self.config.images.enabled:
                processed_observation = self.process_images(processed_observation, camera_names)
            processed_frames.append((processed_observation, processed_action))

        self._episode_frames = []
        return processed_frames
