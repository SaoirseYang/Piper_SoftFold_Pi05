from __future__ import annotations

import json
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None


ARM_STATE_KEYS = [
    *(f"left_joint_{index}.pos" for index in range(1, 7)),
    "left_gripper.pos",
    *(f"right_joint_{index}.pos" for index in range(1, 7)),
    "right_gripper.pos",
]

IMAGE_DIMENSION_NAMES = ["height", "width", "channels"]


@dataclass(frozen=True)
class EpisodeMetadata:
    task: str
    fps: float
    started_at: str
    action_source: str
    cameras: list[str]
    image_format: str


class PiperEpisodeRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        task: str,
        fps: float,
        action_source: str = "leader",
        camera_names: list[str] | None = None,
        image_format: str = "jpg",
        image_quality: int = 95,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.task = task
        self.fps = fps
        self.action_source = action_source
        self.camera_names = camera_names or []
        self.image_format = image_format.lower()
        self.image_quality = image_quality
        self.episode_dir: Path | None = None
        self.frames_path: Path | None = None
        self._frames_file: Any | None = None
        self._frame_index = 0

        if self.image_format not in {"jpg", "jpeg", "png"}:
            raise ValueError("image_format must be one of: jpg, jpeg, png.")

    def __enter__(self) -> "PiperEpisodeRecorder":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def start(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        episode_name = time.strftime("episode_%Y%m%d_%H%M%S")
        self.episode_dir = self.output_dir / episode_name
        self.episode_dir.mkdir()

        metadata = EpisodeMetadata(
            task=self.task,
            fps=self.fps,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            action_source=self.action_source,
            cameras=self.camera_names,
            image_format=self.image_format,
        )
        metadata_path = self.episode_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata.__dict__, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        self.frames_path = self.episode_dir / "frames.jsonl"
        self._frames_file = self.frames_path.open("w", encoding="utf-8")

        for camera_name in self.camera_names:
            (self.episode_dir / "images" / camera_name).mkdir(parents=True, exist_ok=True)

        return self.episode_dir

    def record_frame(self, observation: dict[str, Any], action: dict[str, Any]) -> None:
        if self._frames_file is None:
            raise RuntimeError("Recorder has not been started.")

        timestamp = time.time()
        frame_observation = self._save_camera_images(observation)
        frame = {
            "frame_index": self._frame_index,
            "timestamp": timestamp,
            "observation": self._json_ready(frame_observation),
            "action": self._json_ready(action),
        }
        self._frames_file.write(json.dumps(frame, ensure_ascii=False) + "\n")
        self._frame_index += 1

    def close(self) -> None:
        if self._frames_file is not None:
            self._frames_file.close()
            self._frames_file = None

    def _json_ready(self, values: dict[str, Any]) -> dict[str, Any]:
        json_values: dict[str, Any] = {}
        for key, value in values.items():
            if isinstance(value, np.ndarray):
                json_values[key] = {
                    "type": "ndarray",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "omitted": True,
                }
            elif isinstance(value, np.generic):
                json_values[key] = value.item()
            else:
                json_values[key] = value
        return json_values

    def _save_camera_images(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.episode_dir is None:
            raise RuntimeError("Recorder has not been started.")

        frame_observation = dict(observation)
        for camera_name in self.camera_names:
            image = observation.get(camera_name)
            if not isinstance(image, np.ndarray):
                continue

            image_path = self._image_path(camera_name)
            self._write_image(image_path, image)
            frame_observation[camera_name] = {
                "type": "image",
                "path": str(image_path.relative_to(self.episode_dir)),
                "shape": list(image.shape),
                "dtype": str(image.dtype),
            }
        return frame_observation

    def _image_path(self, camera_name: str) -> Path:
        if self.episode_dir is None:
            raise RuntimeError("Recorder has not been started.")

        extension = "jpg" if self.image_format == "jpeg" else self.image_format
        filename = f"frame_{self._frame_index:06d}.{extension}"
        return self.episode_dir / "images" / camera_name / filename

    def _write_image(self, image_path: Path, image: np.ndarray) -> None:
        if cv2 is not None:
            image_to_write = image
            if image.ndim == 3 and image.shape[2] == 3:
                image_to_write = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            params: list[int] = []
            if image_path.suffix.lower() in {".jpg", ".jpeg"}:
                params = [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]

            ok = cv2.imwrite(str(image_path), image_to_write, params)
            if not ok:
                raise RuntimeError(f"Failed to write image: {image_path}")
            return

        if Image is None:
            raise ImportError("opencv-python or Pillow is required to save camera images.")

        pil_image = Image.fromarray(image)
        save_kwargs: dict[str, Any] = {}
        if image_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = self.image_quality
        pil_image.save(image_path, **save_kwargs)


class LeRobotEpisodeRecorder:
    def __init__(
        self,
        root: str | Path,
        repo_id: str,
        task: str,
        fps: int,
        camera_names: list[str],
        camera_shape: tuple[int, int, int],
        robot_type: str = "piper",
        use_videos: bool = True,
    ) -> None:
        self.root = Path(root)
        self.repo_id = repo_id
        self.task = task
        self.fps = fps
        self.camera_names = camera_names
        self.camera_shape = camera_shape
        self.robot_type = robot_type
        self.use_videos = use_videos
        self.dataset: Any | None = None
        self.dataset_root = self.root / repo_id
        self.episode_dir = self.dataset_root
        self._frame_count = 0

    def __enter__(self) -> "LeRobotEpisodeRecorder":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def start(self) -> Path:
        LeRobotDataset = self._import_lerobot_dataset()
        features = self._make_features()
        self.dataset_root.parent.mkdir(parents=True, exist_ok=True)

        if (self.dataset_root / "meta" / "info.json").exists():
            self._validate_existing_features(features)
            resume = getattr(LeRobotDataset, "resume", None)
            if callable(resume):
                self.dataset = resume(repo_id=self.repo_id, root=self.dataset_root)
            else:
                self.dataset = LeRobotDataset(repo_id=self.repo_id, root=self.dataset_root)
            return self.episode_dir

        create_kwargs = {
            "repo_id": self.repo_id,
            "fps": self.fps,
            "features": features,
            "robot_type": self.robot_type,
            "root": self.dataset_root,
            "use_videos": self.use_videos,
        }
        self.dataset = self._create_dataset(LeRobotDataset, create_kwargs)
        return self.episode_dir

    def record_frame(self, observation: dict[str, Any], action: dict[str, Any]) -> None:
        if self.dataset is None:
            raise RuntimeError("Recorder has not been started.")

        frame: dict[str, Any] = {
            "observation.state": self._state_vector(observation),
            "action": self._state_vector(action),
            "task": self.task,
        }

        for camera_name in self.camera_names:
            image = observation.get(camera_name)
            if image is None:
                image = np.zeros(self.camera_shape, dtype=np.uint8)
            frame[f"observation.images.{camera_name}"] = image

        self.dataset.add_frame(frame)
        self._frame_count += 1

    def close(self) -> None:
        if self.dataset is None:
            return

        if self._frame_count > 0:
            self.dataset.save_episode()
        finalize = getattr(self.dataset, "finalize", None)
        if callable(finalize):
            finalize()
        self.dataset = None

    def _make_features(self) -> dict[str, dict[str, Any]]:
        features: dict[str, dict[str, Any]] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(ARM_STATE_KEYS),),
                "names": ARM_STATE_KEYS,
            },
            "action": {
                "dtype": "float32",
                "shape": (len(ARM_STATE_KEYS),),
                "names": ARM_STATE_KEYS,
            },
        }

        for camera_name in self.camera_names:
            features[f"observation.images.{camera_name}"] = {
                "dtype": "video" if self.use_videos else "image",
                "shape": self.camera_shape,
                "names": IMAGE_DIMENSION_NAMES,
            }
        return features

    def _validate_existing_features(self, expected_features: dict[str, dict[str, Any]]) -> None:
        info_path = self.dataset_root / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        existing_features = info.get("features", {})
        if not isinstance(existing_features, dict):
            return

        expected_keys = set(expected_features.keys())
        existing_keys = {
            key
            for key in existing_features.keys()
            if key in {"action", "observation.state"} or key.startswith("observation.images.")
        }
        if expected_keys != existing_keys:
            raise ValueError(
                "Existing dataset schema does not match the current recording schema. "
                f"Expected features {sorted(expected_keys)}, found {sorted(existing_keys)}. "
                "Use a new repo_id/root for the new camera layout, or remove the old dataset first."
            )

    def _state_vector(self, values: dict[str, Any]) -> np.ndarray:
        return np.asarray([float(values.get(key, 0.0)) for key in ARM_STATE_KEYS], dtype=np.float32)

    def _import_lerobot_dataset(self) -> type:
        for module_name in ("lerobot.datasets", "lerobot.datasets.lerobot_dataset"):
            try:
                module = import_module(module_name)
            except ImportError:
                continue
            LeRobotDataset = getattr(module, "LeRobotDataset", None)
            if LeRobotDataset is not None:
                return LeRobotDataset

        try:
            module = import_module("lerobot.common.datasets.lerobot_dataset")
        except ImportError as exc:
            raise ImportError("lerobot is required to record LeRobot datasets.") from exc

        return module.LeRobotDataset

    def _create_dataset(self, dataset_cls: type, kwargs: dict[str, Any]) -> Any:
        try:
            return dataset_cls.create(**kwargs)
        except TypeError:
            kwargs = dict(kwargs)
            kwargs.pop("use_videos", None)
            return dataset_cls.create(**kwargs)


class DualLeRobotEpisodeRecorder:
    """Write one episode to a raw dataset and a preprocessed dataset in parallel."""

    def __init__(
        self,
        raw_recorder: LeRobotEpisodeRecorder,
        augmented_recorder: LeRobotEpisodeRecorder,
    ) -> None:
        self.raw_recorder = raw_recorder
        self.augmented_recorder = augmented_recorder

    @property
    def episode_dirs(self) -> list[Path]:
        return [Path(self.raw_recorder.episode_dir), Path(self.augmented_recorder.episode_dir)]

    def __enter__(self) -> "DualLeRobotEpisodeRecorder":
        self.raw_recorder.__enter__()
        self.augmented_recorder.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.augmented_recorder.__exit__(exc_type, exc, tb)
        self.raw_recorder.__exit__(exc_type, exc, tb)

    def record_frame(
        self,
        raw_observation: dict[str, Any],
        raw_action: dict[str, Any],
        augmented_observation: dict[str, Any],
        augmented_action: dict[str, Any],
    ) -> None:
        self.raw_recorder.record_frame(observation=raw_observation, action=raw_action)
        self.augmented_recorder.record_frame(
            observation=augmented_observation,
            action=augmented_action,
        )
