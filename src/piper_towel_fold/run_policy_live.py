import argparse
import json
import math
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TextIO

import numpy as np
import torch
from lerobot.cameras.opencv import OpenCVCameraConfig

from .config import PiperRobotConfig
from .offline_infer import (
    action_tensor_to_dict,
    inference_options_from_namespace,
    load_dataset,
    load_policy,
)
from .piper import PiperRobot
from .preprocessing import FramePreprocessor, load_preprocessing_config
from .recorder import ARM_STATE_KEYS

DEFAULT_LOG_DIR = Path("outputs/logs/policy_live")
RIGHT_JOINT_KEYS = tuple(f"right_joint_{index}.pos" for index in range(1, 7))


class _OmitErrorLineFilter:
    """Drop any stderr line that contains ``[ERROR]`` (piper_sdk CAN spam etc.)."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._buffer = ""

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if "[ERROR]" not in line:
                self._stream.write(line + "\n")
        return len(data)

    def flush(self) -> None:
        if self._buffer and "[ERROR]" not in self._buffer:
            self._stream.write(self._buffer)
        self._buffer = ""
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()

    def isatty(self) -> bool:
        return self._stream.isatty()


def install_error_line_filter() -> None:
    if not isinstance(sys.stderr, _OmitErrorLineFilter):
        sys.stderr = _OmitErrorLineFilter(sys.stderr)  # type: ignore[assignment]


def resolve_live_log_paths(log_jsonl: str | None) -> tuple[Path | None, Path | None]:
    """Return (jsonl_path, summary_path). Empty/auto → timestamped files under outputs/logs."""
    value = (log_jsonl or "").strip()
    if value.lower() in {"none", "off", "-", "false"}:
        return None, None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if value == "" or value.lower() == "auto":
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        jsonl_path = DEFAULT_LOG_DIR / f"live_{stamp}.jsonl"
    else:
        jsonl_path = Path(value)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    summary_path = jsonl_path.with_name(jsonl_path.stem + "_summary.json")
    return jsonl_path, summary_path


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _stats_block(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "max": None, "p95": None}
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "max": float(max(values)),
        "p95": _percentile(values, 0.95),
    }


def right_arm_keys_delta(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(RIGHT_JOINT_KEYS) | {"right_gripper.pos"}
    present = keys & left.keys() & right.keys()
    if not present:
        return 0.0
    return max(abs(left[key] - right[key]) for key in present)


class LiveRunRecorder:
    """Accumulate per-step metrics for jsonl dump + end-of-run diagnosis."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.loop_s: list[float] = []
        self.pred_step_delta: list[float] = []
        self.send_step_delta: list[float] = []
        self.pred_send_gap: list[float] = []
        self.track_err: list[float] = []
        self.held_empty: list[int] = []
        self.queue_size: list[int] = []
        self.chunk_boundary_pred_delta: list[float] = []
        self.overrun_steps = 0
        self._prev_pred: dict[str, float] | None = None
        self._prev_send: dict[str, float] | None = None
        self._prev_chunks: int | None = None

    def add(
        self,
        *,
        record: dict[str, Any],
        predicted: dict[str, float],
        sent: dict[str, float],
        current: dict[str, float],
        loop_s: float,
        period_s: float,
        chunks_appended: int | None,
        held_empty_steps: int | None,
        queue_size: int | None,
    ) -> dict[str, Any]:
        pred_delta = (
            right_arm_keys_delta(predicted, self._prev_pred) if self._prev_pred is not None else 0.0
        )
        send_delta = (
            right_arm_keys_delta(sent, self._prev_send) if self._prev_send is not None else 0.0
        )
        gap = right_arm_keys_delta(predicted, sent)
        track = right_arm_keys_delta(sent, current)
        chunk_boundary = False
        if chunks_appended is not None and self._prev_chunks is not None:
            chunk_boundary = chunks_appended > self._prev_chunks

        enriched = {
            **record,
            "pred_step_delta": pred_delta,
            "send_step_delta": send_delta,
            "pred_send_gap": gap,
            "right_track_err": track,
            "chunk_boundary": chunk_boundary,
            "loop_overrun": loop_s > period_s * 1.25,
        }
        self.records.append(enriched)
        self.loop_s.append(loop_s)
        if self._prev_pred is not None:
            self.pred_step_delta.append(pred_delta)
            self.send_step_delta.append(send_delta)
            if chunk_boundary:
                self.chunk_boundary_pred_delta.append(pred_delta)
        self.pred_send_gap.append(gap)
        self.track_err.append(track)
        if held_empty_steps is not None:
            self.held_empty.append(int(held_empty_steps))
        if queue_size is not None:
            self.queue_size.append(int(queue_size))
        if loop_s > period_s * 1.25:
            self.overrun_steps += 1

        self._prev_pred = dict(predicted)
        self._prev_send = dict(sent)
        if chunks_appended is not None:
            self._prev_chunks = chunks_appended
        return enriched

    def build_summary(self, meta: dict[str, Any]) -> dict[str, Any]:
        held_empty_max = max(self.held_empty) if self.held_empty else 0
        held_empty_nonzero = sum(1 for value in self.held_empty if value > 0)
        pred_stats = _stats_block(self.pred_step_delta)
        send_stats = _stats_block(self.send_step_delta)
        boundary_stats = _stats_block(self.chunk_boundary_pred_delta)
        gap_stats = _stats_block(self.pred_send_gap)
        track_stats = _stats_block(self.track_err)
        loop_stats = _stats_block(self.loop_s)

        pred_max = pred_stats["max"] or 0.0
        send_max = send_stats["max"] or 0.0
        boundary_max = boundary_stats["max"] or 0.0
        track_max = track_stats["max"] or 0.0

        if held_empty_max > 0 or self.overrun_steps > max(3, int(0.02 * max(len(self.loop_s), 1))):
            verdict = "inference_stall"
            verdict_zh = "推理/预取跟不上：出现空队列 hold 或控制周期超时，顿挫更像 chunk 断档"
        elif boundary_max >= 0.05 and boundary_max >= 1.5 * max(pred_stats["mean"] or 0.0, 1e-6):
            verdict = "chunk_seam"
            verdict_zh = "chunk 接缝跳变偏大：更像拼接/插值不足，而不是整段轨迹高频抖"
        elif pred_max >= 0.04 and send_max >= 0.8 * pred_max:
            verdict = "inference_jitter"
            verdict_zh = "pred 帧间跳变大，且 send 几乎没压住：更像推理轨迹本身不顺/平滑偏弱"
        elif pred_max >= 0.04 and send_max < 0.5 * pred_max:
            verdict = "smoothing_ok_track_or_other"
            verdict_zh = "pred 有跳变但 send 已被抹平：平滑在工作；目视顿挫更可能来自跟踪/执行层"
        elif track_max >= 0.05 and pred_max < 0.03:
            verdict = "tracking"
            verdict_zh = "指令较顺但跟踪误差大：更像 control_speed / 步进限制 / 机械侧"
        else:
            verdict = "mostly_smooth"
            verdict_zh = "统计上看路径较顺；若仍顿挫，对照 jsonl 里 chunk_boundary=true 的几帧"

        return {
            **meta,
            "steps": len(self.records),
            "loop_s": loop_stats,
            "overrun_steps": self.overrun_steps,
            "held_empty_max": held_empty_max,
            "held_empty_nonzero_steps": held_empty_nonzero,
            "queue_size": _stats_block([float(v) for v in self.queue_size]),
            "pred_step_delta_rad": pred_stats,
            "send_step_delta_rad": send_stats,
            "chunk_boundary_pred_delta_rad": boundary_stats,
            "pred_send_gap_rad": gap_stats,
            "right_track_err_rad": track_stats,
            "verdict": verdict,
            "verdict_zh": verdict_zh,
        }

    def print_summary(self, summary: dict[str, Any]) -> None:
        def fmt(block: dict[str, Any]) -> str:
            if not block.get("count"):
                return "n/a"
            return (
                f"mean={block['mean']:.4f}  p95={block['p95']:.4f}  max={block['max']:.4f}"
            )

        print()
        print("=" * 60)
        print("Live policy summary")
        print("=" * 60)
        print(f"  steps:        {summary.get('steps')}")
        print(f"  duration_s:   {summary.get('duration_s')}")
        print(f"  control_hz:   {summary.get('control_hz')}")
        print(f"  log_jsonl:    {summary.get('log_jsonl')}")
        print(f"  log_summary:  {summary.get('log_summary')}")
        print("  timing:")
        print(f"    loop_s:           {fmt(summary['loop_s'])}")
        print(f"    overrun_steps:    {summary['overrun_steps']}")
        print(f"    held_empty_max:   {summary['held_empty_max']}")
        print(f"    held_empty>0:     {summary['held_empty_nonzero_steps']}")
        print("  right-arm deltas (rad):")
        print(f"    |Δpred| step:     {fmt(summary['pred_step_delta_rad'])}")
        print(f"    |Δsend| step:     {fmt(summary['send_step_delta_rad'])}")
        print(f"    |Δpred|@chunk:    {fmt(summary['chunk_boundary_pred_delta_rad'])}")
        print(f"    |pred-send|:      {fmt(summary['pred_send_gap_rad'])}")
        print(f"    |send-current|:   {fmt(summary['right_track_err_rad'])}")
        print(f"  verdict:      {summary['verdict']}")
        print(f"  判断:         {summary['verdict_zh']}")
        print("=" * 60)
        print()


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_camera_ref(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def make_camera_configs(args: argparse.Namespace) -> dict[str, OpenCVCameraConfig]:
    camera_refs = parse_csv(args.camera_indices)
    camera_names = parse_csv(args.camera_names)
    if len(camera_refs) != len(camera_names):
        raise ValueError("--camera-indices and --camera-names must have the same number of items.")

    return {
        camera_name: OpenCVCameraConfig(
            index_or_path=parse_camera_ref(camera_ref),
            fps=args.camera_fps,
            width=args.camera_width,
            height=args.camera_height,
        )
        for camera_name, camera_ref in zip(camera_names, camera_refs, strict=True)
    }


def install_stop_handler() -> dict[str, bool]:
    stop_requested = {"value": False}

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        stop_requested["value"] = True
        print("\nStop requested. Finishing current loop.")

    signal.signal(signal.SIGINT, request_stop)
    return stop_requested


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {image.shape}.")
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    return tensor.float() / 255.0


def observation_to_policy_input(
    observation: dict[str, Any],
    camera_names: list[str],
    task: str,
) -> dict[str, Any]:
    policy_input: dict[str, Any] = {
        "observation.state": torch.tensor(
            [float(observation.get(key, 0.0)) for key in ARM_STATE_KEYS],
            dtype=torch.float32,
        ),
        "task": task,
    }

    for camera_name in camera_names:
        image = observation[camera_name]
        policy_input[f"observation.images.{camera_name}"] = image_to_tensor(image)

    return policy_input


def smooth_action(
    action: dict[str, float],
    previous_action: dict[str, float] | None,
    alpha: float,
) -> dict[str, float]:
    if previous_action is None:
        return dict(action)

    return {
        key: previous_action.get(key, value) * (1.0 - alpha) + value * alpha
        for key, value in action.items()
    }


def lerp_action(left: dict[str, float], right: dict[str, float], weight: float) -> dict[str, float]:
    weight = float(min(1.0, max(0.0, weight)))
    keys = left.keys() | right.keys()
    return {
        key: left.get(key, right[key]) * (1.0 - weight) + right.get(key, left[key]) * weight
        for key in keys
    }


def densify_actions(waypoints: list[dict[str, float]], substeps: int) -> list[dict[str, float]]:
    """Upsample consecutive policy waypoints so wall-clock duration stays T / policy_fps.

    Each of the T policy actions is expanded to ``substeps`` control ticks. Between
    waypoint i and i+1 the ticks linearly interpolate; the final waypoint is held.
    """
    if not waypoints:
        return []
    if substeps <= 1 or len(waypoints) == 1:
        return [dict(item) for item in waypoints]

    densified: list[dict[str, float]] = []
    last_index = len(waypoints) - 1
    for index in range(len(waypoints)):
        if index == last_index:
            for _ in range(substeps):
                densified.append(dict(waypoints[index]))
            continue
        start = waypoints[index]
        end = waypoints[index + 1]
        for step in range(substeps):
            densified.append(lerp_action(start, end, step / substeps))
    return densified


def blend_chunk_boundary(
    previous: dict[str, float] | None,
    chunk: list[dict[str, float]],
    blend_steps: int,
) -> list[dict[str, float]]:
    """C0-join a new chunk onto the last queued/executed action via linear crossfade."""
    if not chunk:
        return []
    if previous is None or blend_steps <= 0:
        return [dict(item) for item in chunk]

    blended = [dict(item) for item in chunk]
    steps = min(blend_steps, len(blended))
    for index in range(steps):
        # weight 1/steps .. 1.0 so the last blended sample equals the original chunk sample
        weight = (index + 1) / steps
        blended[index] = lerp_action(previous, blended[index], weight)
    return blended


def clone_policy_input(policy_input: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in policy_input.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.detach().cpu().clone()
        elif isinstance(value, np.ndarray):
            cloned[key] = np.array(value, copy=True)
        else:
            cloned[key] = value
    return cloned


class ActionChunkStreamer:
    """Own the ACT action queue: async prefetch + seam blend + optional densify.

    Avoids the stall in ``policy.select_action`` when the internal queue empties and
    inference blocks the control loop.
    """

    def __init__(
        self,
        *,
        predict_chunk: Callable[[dict[str, Any]], list[dict[str, float]]],
        n_action_steps: int,
        interp_substeps: int,
        prefetch_remaining: int,
        chunk_blend_steps: int,
    ) -> None:
        if n_action_steps <= 0:
            raise ValueError("n_action_steps must be > 0")
        if interp_substeps <= 0:
            raise ValueError("interp_substeps must be > 0")
        if prefetch_remaining < 0:
            raise ValueError("prefetch_remaining must be >= 0")
        if chunk_blend_steps < 0:
            raise ValueError("chunk_blend_steps must be >= 0")

        self._predict_chunk = predict_chunk
        self._n_action_steps = n_action_steps
        self._interp_substeps = interp_substeps
        self._prefetch_remaining = prefetch_remaining
        self._chunk_blend_steps = chunk_blend_steps

        self._queue: deque[dict[str, float]] = deque()
        self._lock = threading.Lock()
        self._infer_thread: threading.Thread | None = None
        self._pending_chunk: list[dict[str, float]] | None = None
        self._infer_error: BaseException | None = None
        self._last_action: dict[str, float] | None = None
        self._chunks_appended = 0
        self._held_empty_steps = 0

    @property
    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def chunks_appended(self) -> int:
        return self._chunks_appended

    @property
    def held_empty_steps(self) -> int:
        return self._held_empty_steps

    def _is_inferring(self) -> bool:
        return self._infer_thread is not None and self._infer_thread.is_alive()

    def _prepare_chunk(self, waypoints: list[dict[str, float]]) -> list[dict[str, float]]:
        return densify_actions(waypoints, self._interp_substeps)

    def _append_prepared(self, prepared: list[dict[str, float]]) -> None:
        if not prepared:
            return
        with self._lock:
            seam_ref = self._queue[-1] if self._queue else self._last_action
            blend_steps = self._chunk_blend_steps * self._interp_substeps
            blended = blend_chunk_boundary(seam_ref, prepared, blend_steps)
            self._queue.extend(blended)
            self._chunks_appended += 1

    def _run_inference(self, policy_input: dict[str, Any]) -> None:
        try:
            waypoints = self._predict_chunk(policy_input)
            if len(waypoints) > self._n_action_steps:
                waypoints = waypoints[: self._n_action_steps]
            prepared = self._prepare_chunk(waypoints)
            with self._lock:
                self._pending_chunk = prepared
        except BaseException as exc:  # noqa: BLE001 - surface to control loop
            with self._lock:
                self._infer_error = exc

    def _start_inference(self, policy_input: dict[str, Any]) -> None:
        if self._is_inferring():
            return
        with self._lock:
            if self._pending_chunk is not None:
                return
        snapshot = clone_policy_input(policy_input)
        self._infer_thread = threading.Thread(
            target=self._run_inference,
            args=(snapshot,),
            name="act-chunk-prefetch",
            daemon=True,
        )
        self._infer_thread.start()

    def _drain_pending(self) -> None:
        with self._lock:
            error = self._infer_error
            self._infer_error = None
            pending = self._pending_chunk
            self._pending_chunk = None
        if error is not None:
            raise RuntimeError("Async action-chunk inference failed") from error
        if pending is not None:
            self._append_prepared(pending)

    def set_seam_reference(self, action: dict[str, float]) -> None:
        """Seed the boundary reference (e.g. live robot pose) before the first chunk."""
        with self._lock:
            self._last_action = dict(action)

    def prime(self, policy_input: dict[str, Any]) -> None:
        """Synchronously fill the first chunk before the control loop starts."""
        self._drain_pending()
        waypoints = self._predict_chunk(policy_input)
        if len(waypoints) > self._n_action_steps:
            waypoints = waypoints[: self._n_action_steps]
        prepared = self._prepare_chunk(waypoints)
        self._append_prepared(prepared)

    def maybe_prefetch(self, policy_input: dict[str, Any]) -> None:
        self._drain_pending()
        if self.queue_size <= self._prefetch_remaining:
            self._start_inference(policy_input)

    def pop_action(self) -> dict[str, float]:
        self._drain_pending()
        with self._lock:
            if self._queue:
                action = self._queue.popleft()
                self._last_action = dict(action)
                self._held_empty_steps = 0
                return action

        # Queue empty: never block the control loop. Hold last action (or wait briefly
        # only if inference is in flight and we have never produced an action).
        if self._last_action is not None:
            self._held_empty_steps += 1
            return dict(self._last_action)

        if self._is_inferring():
            assert self._infer_thread is not None
            self._infer_thread.join(timeout=2.0)
            self._drain_pending()
            with self._lock:
                if self._queue:
                    action = self._queue.popleft()
                    self._last_action = dict(action)
                    return action

        raise RuntimeError("Action queue is empty and no prefetch result is available.")


def current_action_from_observation(observation: dict[str, Any]) -> dict[str, float]:
    return {key: float(observation.get(key, 0.0)) for key in ARM_STATE_KEYS}


def max_abs_delta(left: dict[str, float], right: dict[str, float]) -> float:
    return max(abs(left[key] - right[key]) for key in left.keys() & right.keys())


def right_arm_delta(left: dict[str, float], right: dict[str, float]) -> float:
    keys = {
        *(f"right_joint_{index}.pos" for index in range(1, 7)),
        "right_gripper.pos",
    }
    return max(abs(left[key] - right[key]) for key in keys & left.keys() & right.keys())


def json_ready(action: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in action.items()}


def get_action_names(policy_config: Any) -> list[str]:
    output_features = getattr(policy_config, "output_features", None)
    if isinstance(output_features, dict):
        action_feature = output_features.get("action")
        if isinstance(action_feature, dict):
            names = action_feature.get("names")
        else:
            names = getattr(action_feature, "names", None)
        if names is not None:
            return list(names)

    return list(ARM_STATE_KEYS)


def load_dataset_stats(args: argparse.Namespace) -> Any | None:
    if not args.dataset_root:
        return None

    try:
        dataset = load_dataset(
            repo_id=args.repo_id,
            dataset_root=args.dataset_root,
            video_backend=args.video_backend,
        )
    except Exception as exc:
        print(f"Warning: could not load dataset stats from {args.dataset_root}: {exc}")
        return None

    return getattr(dataset.meta, "stats", None)


def print_action_summary(prefix: str, action: dict[str, float]) -> None:
    right = ", ".join(
        f"rj{index}={action[f'right_joint_{index}.pos']:.3f}" for index in range(1, 7)
    )
    left = ", ".join(
        f"lj{index}={action[f'left_joint_{index}.pos']:.3f}" for index in range(1, 7)
    )
    print(
        f"{prefix} {left}, lg={action['left_gripper.pos']:.4f} | "
        f"{right}, rg={action['right_gripper.pos']:.4f}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a trained LeRobot policy on the real Piper robot.")
    parser.add_argument(
        "--policy-path",
        default="outputs/train/act_piper_pick_cube/checkpoints/last/pretrained_model",
        help="Path to the local pretrained_model checkpoint directory.",
    )
    parser.add_argument("--repo-id", default="local/piper_pick_cube")
    parser.add_argument("--dataset-root", default="data/lerobot/local/piper_pick_cube")
    parser.add_argument("--video-backend", default="pyav", choices=("pyav", "torchcodec"))
    parser.add_argument("--task", default="pick_cube", help="Task string passed to the policy.")
    parser.add_argument("--execute", action="store_true", help="Actually send actions to the robot.")
    parser.add_argument("--duration", type=float, default=10.0, help="Run duration in seconds.")
    parser.add_argument("--fps", type=float, default=10.0, help="Policy control frequency.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--inference-dtype",
        default=None,
        choices=("bfloat16", "float32"),
        help="Override checkpoint dtype for inference (e.g. bfloat16 to reduce VRAM).",
    )
    parser.add_argument(
        "--compile-model",
        default=None,
        help="Override torch.compile during inference (false reduces peak VRAM).",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override flow-matching denoising steps (lower values use less VRAM).",
    )
    parser.add_argument("--follower-left-can", default="can2")
    parser.add_argument("--follower-right-can", default="can0")
    parser.add_argument("--camera-indices", default="2,4,0")
    parser.add_argument("--camera-names", default="cam_top,cam_left,cam_right")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--control-speed", type=int, default=10)
    parser.add_argument(
        "--max-joint-step-rad",
        type=float,
        default=0.025,
        help="每控制周期关节最大步进（弧度）；<=0 表示不限速",
    )
    parser.add_argument(
        "--max-gripper-step-m",
        type=float,
        default=0.001,
        help="每控制周期夹爪最大步进（米）；<=0 或不限速请设 0",
    )
    parser.add_argument("--gripper-effort", type=int, default=1000)
    parser.add_argument("--smoothing-alpha", type=float, default=0.25)
    parser.add_argument(
        "--control-hz",
        type=float,
        default=None,
        help=(
            "Actual robot command rate. If greater than --fps, consecutive ACT actions are "
            "linearly interpolated so wall-clock speed stays matched to training fps."
        ),
    )
    parser.add_argument(
        "--use-action-chunk-stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Async-prefetch ACT chunks with seam blending (avoids chunk-boundary stalls).",
    )
    parser.add_argument(
        "--prefetch-remaining",
        type=int,
        default=15,
        help="Start next-chunk inference when densified queue length <= this.",
    )
    parser.add_argument(
        "--chunk-blend-steps",
        type=int,
        default=8,
        help="Policy-step count to crossfade at each chunk boundary (before densify).",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Override policy.config.n_action_steps for live chunk length.",
    )
    parser.add_argument(
        "--pace-by-reach",
        action="store_true",
        help="Hold the current policy target until the right arm gets close, instead of advancing every tick.",
    )
    parser.add_argument(
        "--advance-threshold-rad",
        type=float,
        default=0.08,
        help="Right-arm max error threshold for advancing to the next policy action.",
    )
    parser.add_argument(
        "--max-hold-steps",
        type=int,
        default=30,
        help="Maximum control ticks to hold one policy action when --pace-by-reach is enabled.",
    )
    parser.add_argument("--print-every", type=int, default=1, help="Print every N policy steps.")
    parser.add_argument(
        "--log-jsonl",
        default="auto",
        help=(
            "Per-step JSONL path. Use 'auto' (default) for outputs/logs/policy_live/live_*.jsonl, "
            "'none' to disable, or an explicit path."
        ),
    )
    parser.add_argument(
        "--omit-error-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide stderr lines containing [ERROR] (piper_sdk CAN noise).",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def run_live_policy(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0.")
    if not 0.0 < args.smoothing_alpha <= 1.0:
        raise ValueError("--smoothing-alpha must be in (0, 1].")

    if getattr(args, "omit_error_logs", True):
        install_error_line_filter()

    control_hz = float(args.control_hz) if args.control_hz is not None else float(args.fps)
    if control_hz <= 0:
        raise ValueError("--control-hz must be greater than 0.")
    if control_hz + 1e-6 < args.fps:
        raise ValueError("--control-hz must be >= --fps (interpolation densifies, never slows policy time).")

    interp_substeps = max(1, int(round(control_hz / args.fps)))
    effective_control_hz = args.fps * interp_substeps

    camera_configs = make_camera_configs(args)
    config = PiperRobotConfig(
        follower_left_port=args.follower_left_can,
        follower_right_port=args.follower_right_can,
        cameras=camera_configs,
        enable_control=args.execute,
        control_speed=args.control_speed,
        max_joint_step_rad=args.max_joint_step_rad,
        max_gripper_step_m=args.max_gripper_step_m,
        gripper_effort=args.gripper_effort,
    )

    policy_config, policy, make_pre_post_processors = load_policy(
        args.policy_path,
        args.device,
        **inference_options_from_namespace(args),
    )
    dataset_stats = load_dataset_stats(args)
    preprocessing_config = load_preprocessing_config(getattr(args, "preprocessing", None))
    frame_preprocessor = (
        FramePreprocessor(preprocessing_config, mode="online")
        if preprocessing_config.active
        else None
    )
    if frame_preprocessor is not None and frame_preprocessor.enabled:
        print("Frame preprocessing enabled for live inference.")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=args.policy_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    robot = PiperRobot(config)
    stop_requested = install_stop_handler()
    period = 1.0 / effective_control_hz
    started_at = time.monotonic()
    previous_action: dict[str, float] | None = None
    held_predicted_action: dict[str, float] | None = None
    hold_steps = 0
    camera_names = list(camera_configs.keys())
    action_feature_names = get_action_names(policy_config)
    n_action_steps = int(args.n_action_steps or getattr(policy_config, "n_action_steps", 1))
    use_stream = bool(args.use_action_chunk_stream)
    if use_stream and getattr(policy_config, "temporal_ensemble_coeff", None) is not None:
        print(
            "Warning: temporal_ensemble_coeff is set; action-chunk stream bypasses "
            "select_action ensembling. Prefer leaving temporal_ensemble_coeff null."
        )

    log_jsonl_path, log_summary_path = resolve_live_log_paths(getattr(args, "log_jsonl", "auto"))
    log_file = None
    if log_jsonl_path is not None:
        log_file = log_jsonl_path.open("w", encoding="utf-8")
    recorder = LiveRunRecorder()

    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()

    def predict_chunk_actions(policy_input: dict[str, Any]) -> list[dict[str, float]]:
        processed = preprocessor(policy_input)
        with torch.inference_mode():
            action_chunk = policy.predict_action_chunk(processed)
        # (batch, chunk, dim) -> per-step postprocess to match select_action path
        if action_chunk.ndim != 3:
            raise RuntimeError(f"Expected action chunk (B, T, D), got shape {tuple(action_chunk.shape)}")
        steps = min(n_action_steps, int(action_chunk.shape[1]))
        actions: list[dict[str, float]] = []
        for index in range(steps):
            step_tensor = postprocessor(action_chunk[:, index])
            actions.append(action_tensor_to_dict(step_tensor, action_feature_names))
        return actions

    streamer: ActionChunkStreamer | None = None
    if use_stream:
        streamer = ActionChunkStreamer(
            predict_chunk=predict_chunk_actions,
            n_action_steps=n_action_steps,
            interp_substeps=interp_substeps,
            prefetch_remaining=max(1, int(args.prefetch_remaining) * interp_substeps),
            chunk_blend_steps=int(args.chunk_blend_steps),
        )

    print("Running live policy.")
    print(f"  mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print(f"  policy: {args.policy_path}")
    print(f"  policy_fps: {args.fps}")
    print(f"  control_hz: {effective_control_hz:.3f} (interp_substeps={interp_substeps})")
    print(f"  cameras: {', '.join(camera_names)}")
    if use_stream:
        print(
            f"  chunk_stream: on | n_action_steps={n_action_steps} "
            f"| prefetch_remaining={args.prefetch_remaining} "
            f"| chunk_blend_steps={args.chunk_blend_steps}"
        )
    else:
        print("  chunk_stream: off (legacy select_action)")
    if log_jsonl_path is not None:
        print(f"  log_jsonl: {log_jsonl_path}")
        print(f"  log_summary: {log_summary_path}")
    else:
        print("  log_jsonl: disabled")
    if not args.execute:
        print("  no actions will be sent; add --execute only after dry-run output looks sane")
    print("Press Ctrl+C to stop.")
    print()

    try:
        robot.connect()
        if frame_preprocessor is not None:
            frame_preprocessor.reset()
        step = 0
        primed = False
        while not stop_requested["value"]:
            loop_started_at = time.monotonic()
            if time.monotonic() - started_at >= args.duration:
                break

            observation = robot.get_observation()
            if frame_preprocessor is not None and frame_preprocessor.enabled:
                observation = frame_preprocessor.process_observation(observation, camera_names)
            current_action = current_action_from_observation(observation)
            if previous_action is None:
                previous_action = current_action

            policy_input = observation_to_policy_input(observation, camera_names, args.task)
            should_advance = True
            if args.pace_by_reach and held_predicted_action is not None:
                target_error = right_arm_delta(held_predicted_action, current_action)
                should_advance = (
                    target_error <= args.advance_threshold_rad
                    or hold_steps >= args.max_hold_steps
                )

            if should_advance:
                if streamer is not None:
                    if not primed:
                        # Blend the first chunk from the live robot pose to avoid a startup jump.
                        streamer.set_seam_reference(current_action)
                        streamer.prime(policy_input)
                        primed = True
                    else:
                        streamer.maybe_prefetch(policy_input)
                    predicted_action = streamer.pop_action()
                else:
                    processed = preprocessor(policy_input)
                    with torch.inference_mode():
                        action_tensor = policy.select_action(processed)
                        action_tensor = postprocessor(action_tensor)
                    predicted_action = action_tensor_to_dict(action_tensor, action_feature_names)
                held_predicted_action = predicted_action
                hold_steps = 0
            elif held_predicted_action is not None:
                predicted_action = held_predicted_action
                hold_steps += 1
            else:
                raise RuntimeError("No held action is available.")
            smoothed_action = smooth_action(predicted_action, previous_action, args.smoothing_alpha)

            if args.execute:
                sent_action = robot.send_action(smoothed_action)
                previous_action = dict(sent_action)
            else:
                sent_action = smoothed_action
                previous_action = smoothed_action

            loop_s = time.monotonic() - loop_started_at
            chunks_appended = streamer.chunks_appended if streamer is not None else None
            held_empty_steps = streamer.held_empty_steps if streamer is not None else None
            queue_size = streamer.queue_size if streamer is not None else None

            if step % args.print_every == 0:
                print_action_summary(f"step {step:04d} pred:", predicted_action)
                print_action_summary(f"step {step:04d} send:", sent_action)
                queue_info = ""
                if streamer is not None:
                    queue_info = (
                        f" q={queue_size}"
                        f" chunks={chunks_appended}"
                        f" held_empty={held_empty_steps}"
                    )
                print(
                    f"step {step:04d} delta:"
                    f" pred-current={max_abs_delta(predicted_action, current_action):.4f}"
                    f" send-current={max_abs_delta(sent_action, current_action):.4f}"
                    f" right-target={right_arm_delta(predicted_action, current_action):.4f}"
                    f" hold={hold_steps}{queue_info}"
                )

            log_record = {
                "timestamp": time.time(),
                "step": step,
                "loop_s": loop_s,
                "current_action": json_ready(current_action),
                "predicted_action": json_ready(predicted_action),
                "sent_action": json_ready(sent_action),
                "pred_current_max_abs_delta": max_abs_delta(predicted_action, current_action),
                "sent_current_max_abs_delta": max_abs_delta(sent_action, current_action),
                "right_target_delta": right_arm_delta(predicted_action, current_action),
                "hold_steps": hold_steps,
                "queue_size": queue_size,
                "chunks_appended": chunks_appended,
                "held_empty_steps": held_empty_steps,
            }
            enriched = recorder.add(
                record=log_record,
                predicted=predicted_action,
                sent=sent_action,
                current=current_action,
                loop_s=loop_s,
                period_s=period,
                chunks_appended=chunks_appended,
                held_empty_steps=held_empty_steps,
                queue_size=queue_size,
            )
            if log_file is not None:
                log_file.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                if step % 10 == 0:
                    log_file.flush()

            step += 1
            elapsed = time.monotonic() - loop_started_at
            time.sleep(max(0.0, period - elapsed))
    finally:
        duration_s = time.monotonic() - started_at
        summary = recorder.build_summary(
            {
                "duration_s": round(duration_s, 3),
                "policy_path": args.policy_path,
                "execute": bool(args.execute),
                "fps": args.fps,
                "control_hz": effective_control_hz,
                "interp_substeps": interp_substeps,
                "smoothing_alpha": args.smoothing_alpha,
                "control_speed": args.control_speed,
                "max_joint_step_rad": args.max_joint_step_rad,
                "use_action_chunk_stream": use_stream,
                "n_action_steps": n_action_steps,
                "prefetch_remaining": args.prefetch_remaining if use_stream else None,
                "chunk_blend_steps": args.chunk_blend_steps if use_stream else None,
                "log_jsonl": str(log_jsonl_path) if log_jsonl_path is not None else None,
                "log_summary": str(log_summary_path) if log_summary_path is not None else None,
                "chunks_appended": streamer.chunks_appended if streamer is not None else None,
            }
        )
        if log_summary_path is not None and recorder.records:
            log_summary_path.parent.mkdir(parents=True, exist_ok=True)
            log_summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if recorder.records:
            recorder.print_summary(summary)
        if log_file is not None:
            log_file.flush()
            log_file.close()
        if robot.is_connected:
            robot.disconnect()


def main() -> None:
    run_live_policy(parse_args())


if __name__ == "__main__":
    main()
