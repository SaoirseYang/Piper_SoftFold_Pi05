"""Piper robot client for LeRobot async PolicyServer (remote pi05 inference)."""

from __future__ import annotations

import argparse
import json
import logging
import pickle  # nosec
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import grpc
import torch

from .async_features import action_names_from_dataset, observation_features_from_dataset
from .async_image_codec import compress_observation_images
from .async_latency import LatencyTracker
from .config import PiperRobotConfig
from .piper import PiperRobot
from .preprocessing import FramePreprocessor, load_preprocessing_config
from .recorder import ARM_STATE_KEYS
from .run_policy_live import (
    json_ready,
    make_camera_configs,
    max_abs_delta,
    print_action_summary,
    right_arm_delta,
    smooth_action,
)


def _import_async_helpers():
    try:
        from lerobot.async_inference.configs import get_aggregate_function
        from lerobot.async_inference.helpers import (
            FPSTracker,
            RemotePolicyConfig,
            TimedAction,
            TimedObservation,
            get_logger,
            visualize_action_queue_size,
        )
        from lerobot.transport import services_pb2, services_pb2_grpc
        from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
    except ImportError as exc:
        raise ImportError(
            "LeRobot async inference is not installed. "
            'Install with: pip install "lerobot[async,pi]"'
        ) from exc

    return (
        FPSTracker,
        RemotePolicyConfig,
        TimedAction,
        TimedObservation,
        get_aggregate_function,
        get_logger,
        services_pb2,
        services_pb2_grpc,
        grpc_channel_options,
        send_bytes_in_chunks,
        visualize_action_queue_size,
    )


def install_stop_handler(on_stop: Callable[[], None] | None = None) -> dict[str, bool]:
    stop_requested = {"value": False}

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        stop_requested["value"] = True
        print("\nStop requested. Finishing current loop.", flush=True)
        if on_stop is not None:
            on_stop()

    signal.signal(signal.SIGINT, request_stop)
    return stop_requested


def action_tensor_to_dict(action_tensor: torch.Tensor, action_names: list[str]) -> dict[str, float]:
    flat = action_tensor.detach().to("cpu").reshape(-1).tolist()
    values = flat[: len(action_names)]
    return {name: float(value) for name, value in zip(action_names, values, strict=True)}


class PiperAsyncRobotClient:
    prefix = "piper_robot_client"

    def __init__(self, args: argparse.Namespace) -> None:
        (
            FPSTracker,
            RemotePolicyConfig,
            TimedAction,
            TimedObservation,
            get_aggregate_function,
            get_logger,
            services_pb2,
            services_pb2_grpc,
            grpc_channel_options,
            send_bytes_in_chunks,
            _,
        ) = _import_async_helpers()

        self._TimedAction = TimedAction
        self._TimedObservation = TimedObservation
        self._services_pb2 = services_pb2
        self._send_bytes_in_chunks = send_bytes_in_chunks
        self.logger = get_logger(self.prefix)
        self.args = args

        camera_configs = make_camera_configs(args)
        robot_config = PiperRobotConfig(
            follower_left_port=args.follower_left_can,
            follower_right_port=args.follower_right_can,
            cameras=camera_configs,
            enable_control=args.execute,
            control_speed=args.control_speed,
            max_joint_step_rad=args.max_joint_step_rad,
            max_gripper_step_m=args.max_gripper_step_m,
            gripper_effort=args.gripper_effort,
        )
        self.robot = PiperRobot(robot_config)
        self.camera_names = list(camera_configs.keys())
        self.action_names = action_names_from_dataset(args.dataset_root)
        self.lerobot_features = observation_features_from_dataset(args.dataset_root)

        self.policy_config = RemotePolicyConfig(
            args.policy_type,
            args.policy_path,
            self.lerobot_features,
            args.actions_per_chunk,
            args.policy_device,
            rename_map=getattr(args, "rename_map", None) or {},
        )

        environment_dt = 1.0 / args.fps
        self.channel = grpc.insecure_channel(
            args.server_address,
            grpc_channel_options(initial_backoff=f"{environment_dt:.4f}s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        self.shutdown_event = threading.Event()
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1
        self._chunk_size_threshold = args.chunk_size_threshold
        self.action_queue: Queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.action_queue_size: list[int] = []
        self.start_barrier = threading.Barrier(2)
        self.fps_tracker = FPSTracker(target_fps=args.fps)
        self.must_go = threading.Event()
        self.must_go.set()
        self.aggregate_fn = get_aggregate_function(args.aggregate_fn_name)

        self.latency = LatencyTracker(
            enabled=getattr(args, "log_latency", True),
            log_jsonl=getattr(args, "latency_log_jsonl", "") or "",
        )
        self._pending_obs_lock = threading.Lock()
        self._pending_obs_sent_at: dict[int, float] = {}
        self._obs_outbound_queue: Queue[tuple[Any, float]] = Queue(maxsize=2)
        self._obs_sender_thread: threading.Thread | None = None

        compression = str(getattr(args, "obs_image_compression", "jpeg") or "none").lower()
        self._obs_image_compression = compression if compression in {"jpeg", "none"} else "none"
        self._obs_jpeg_quality = int(getattr(args, "obs_jpeg_quality", 85))

        preprocessing_config = load_preprocessing_config(getattr(args, "preprocessing", None))
        self.frame_preprocessor = (
            FramePreprocessor(preprocessing_config, mode="online")
            if preprocessing_config.active
            else None
        )

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def benchmark_network_rtt(self, samples: int = 5, timeout_s: float = 10.0) -> None:
        if samples <= 0:
            return
        timings_ms: list[float] = []
        print(f"Benchmarking gRPC RTT to {self.args.server_address} ({samples} x Ready)...", flush=True)
        for index in range(samples):
            started = time.perf_counter()
            self.stub.Ready(self._services_pb2.Empty(), timeout=timeout_s)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            timings_ms.append(elapsed_ms)
            self.latency.record_network_rtt(elapsed_ms)
            print(f"  Ready #{index + 1}: {elapsed_ms:.1f} ms", flush=True)
        if timings_ms:
            avg_ms = sum(timings_ms) / len(timings_ms)
            self.logger.info(
                f"gRPC RTT benchmark | min={min(timings_ms):.1f}ms "
                f"avg={avg_ms:.1f}ms max={max(timings_ms):.1f}ms"
            )

    def start(self) -> bool:
        try:
            start_time = time.perf_counter()
            self.stub.Ready(self._services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.info(f"Connected to policy server in {end_time - start_time:.4f}s")

            policy_setup = self._services_pb2.PolicySetup(data=pickle.dumps(self.policy_config))
            self.logger.info(
                f"Sending policy instructions | type={self.policy_config.policy_type} "
                f"| path={self.policy_config.pretrained_name_or_path} "
                f"| device={self.policy_config.device}"
            )
            load_started = time.perf_counter()
            print(
                "Waiting for remote pi05 to load on GPU "
                "(first start typically 1-3 minutes, please wait)...",
                flush=True,
            )
            self.stub.SendPolicyInstructions(policy_setup)
            load_elapsed = time.perf_counter() - load_started
            self.logger.info(f"Remote policy loaded in {load_elapsed:.1f}s")
            self.shutdown_event.clear()
            return True
        except grpc.RpcError as exc:
            self.logger.error(f"Failed to connect to policy server: {exc}")
            return False

    def stop(self) -> None:
        self.shutdown_event.set()
        try:
            self.channel.close()
        except Exception:
            pass
        if self._obs_sender_thread is not None:
            self._obs_sender_thread.join(timeout=3.0)
        if self.robot.is_connected:
            self.robot.disconnect()
        self.latency.print_summary()
        self.latency.close()

    def _prepare_observation_for_wire(self, obs: Any) -> Any:
        if self._obs_image_compression != "jpeg":
            return obs
        compressed_observation = compress_observation_images(
            obs.get_observation(),
            quality=self._obs_jpeg_quality,
        )
        return self._TimedObservation(
            timestamp=obs.get_timestamp(),
            timestep=obs.get_timestep(),
            observation=compressed_observation,
            must_go=bool(getattr(obs, "must_go", False)),
        )

    def send_observation(self, obs: Any) -> tuple[bool, float, float, float]:
        if not self.running:
            raise RuntimeError("Client not running.")

        serialize_started = time.perf_counter()
        wire_obs = self._prepare_observation_for_wire(obs)
        observation_bytes = pickle.dumps(wire_obs)
        serialize_ms = (time.perf_counter() - serialize_started) * 1000.0
        payload_mb = len(observation_bytes) / (1024.0 * 1024.0)

        try:
            grpc_started = time.perf_counter()
            observation_iterator = self._send_bytes_in_chunks(
                observation_bytes,
                self._services_pb2.Observation,
                log_prefix="[PIPER CLIENT] Observation",
                silent=True,
            )
            self.stub.SendObservations(observation_iterator)
            grpc_ms = (time.perf_counter() - grpc_started) * 1000.0
            return True, payload_mb, serialize_ms, grpc_ms
        except grpc.RpcError as exc:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {exc}")
            return False, payload_mb, serialize_ms, 0.0

    def _observation_sender_loop(self) -> None:
        while self.running:
            try:
                timed_observation, capture_ms = self._obs_outbound_queue.get(timeout=0.5)
            except Empty:
                continue

            obs_timestep = timed_observation.get_timestep()
            success, payload_mb, serialize_ms, grpc_ms = self.send_observation(timed_observation)
            self.latency.record_obs_send(
                obs_timestep,
                payload_mb=payload_mb,
                serialize_ms=serialize_ms,
                grpc_ms=grpc_ms,
                success=success,
            )
            if success:
                with self._pending_obs_lock:
                    self._pending_obs_sent_at[obs_timestep] = time.monotonic()
                self.logger.info(
                    f"Sent observation #{obs_timestep} | must_go={timed_observation.must_go} | "
                    f"payload={payload_mb:.2f}MB | compression={self._obs_image_compression} | "
                    f"capture={capture_ms:.0f}ms | "
                    f"upload={serialize_ms + grpc_ms:.0f}ms "
                    f"(serialize={serialize_ms:.0f}ms grpc={grpc_ms:.0f}ms)"
                )
            else:
                self.logger.warning(
                    f"Failed to send observation #{obs_timestep} | payload={payload_mb:.2f}MB"
                )

            if timed_observation.must_go:
                self.must_go.clear()
            self._obs_outbound_queue.task_done()

    def start_observation_sender(self) -> None:
        if self._obs_sender_thread is not None and self._obs_sender_thread.is_alive():
            return
        self._obs_sender_thread = threading.Thread(
            target=self._observation_sender_loop,
            daemon=True,
            name="observation-sender",
        )
        self._obs_sender_thread.start()

    def _aggregate_action_queues(
        self,
        incoming_actions: list[Any],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        future_action_queue: Queue = Queue()
        with self.action_queue_lock:
            internal_actions = list(self.action_queue.queue)
            current_action_queue = {
                action.get_timestep(): action.get_action() for action in internal_actions
            }
            handled_timesteps: set[int] = set()

            for new_action in incoming_actions:
                with self.latest_action_lock:
                    latest_action = self.latest_action

                timestep = new_action.get_timestep()
                if timestep <= latest_action:
                    continue

                handled_timesteps.add(timestep)
                if timestep not in current_action_queue:
                    future_action_queue.put(new_action)
                    continue

                future_action_queue.put(
                    self._TimedAction(
                        timestamp=new_action.get_timestamp(),
                        timestep=timestep,
                        action=aggregate_fn(
                            current_action_queue[timestep],
                            new_action.get_action(),
                        ),
                    )
                )

            for old_action in internal_actions:
                old_timestep = old_action.get_timestep()
                with self.latest_action_lock:
                    latest_action = self.latest_action
                if old_timestep > latest_action and old_timestep not in handled_timesteps:
                    future_action_queue.put(old_action)

            self.action_queue = future_action_queue

    def _enqueue_timed_actions(self, timed_actions: list[Any]) -> None:
        if not timed_actions:
            self.logger.warning("Received empty action chunk from server.")
            return

        self._aggregate_action_queues(timed_actions, self.aggregate_fn)
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
        first_step = timed_actions[0].get_timestep()
        last_step = timed_actions[-1].get_timestep()
        self.logger.info(
            f"Received action chunk steps [{first_step}, {last_step}] | queue_size={queue_size}"
        )

        receive_time = time.time()
        with self._pending_obs_lock:
            sent_at = self._pending_obs_sent_at.pop(first_step, None)
        if sent_at is not None:
            obs_to_action_ms = (time.monotonic() - sent_at) * 1000.0
            self.latency.record_obs_to_action(first_step, obs_to_action_ms)
            self.logger.info(
                f"Latency obs #{first_step} -> action chunk: {obs_to_action_ms:.0f}ms"
            )

        server_to_client_ms = (receive_time - timed_actions[0].get_timestamp()) * 1000.0
        if server_to_client_ms >= 0:
            self.latency.record_server_to_client(first_step, server_to_client_ms)
            self.logger.info(
                f"Latency server stamp -> client receive: {server_to_client_ms:.0f}ms"
            )

    def receive_actions(self) -> None:
        self.start_barrier.wait()
        self.logger.info("Action receiver thread started")

        while self.running:
            try:
                self.latency.begin_getactions_wait()
                actions_chunk = self.stub.GetActions(
                    self._services_pb2.Empty(),
                    timeout=20.0,
                )
                if not self.running:
                    break
                if len(actions_chunk.data) == 0:
                    self.latency.record_getactions_empty()
                    self.logger.warning(
                        "GetActions returned empty "
                        "(server waiting for obs, obs filtered, or obs_queue_timeout elapsed)"
                    )
                    continue

                self.latency.finish_getactions_wait()

                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
                self._enqueue_timed_actions(timed_actions)
                self.must_go.set()
            except grpc.RpcError as exc:
                if not self.running:
                    break
                if exc.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    continue
                self.logger.error(f"Error receiving actions: {exc}")
            except Exception as exc:
                if not self.running:
                    break
                self.logger.exception(f"Failed to decode action chunk: {exc}")

    def actions_available(self) -> bool:
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _ready_to_send_observation(self) -> bool:
        with self.action_queue_lock:
            if self.action_chunk_size <= 0:
                return True
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def _current_action_from_observation(self, observation: dict[str, Any]) -> dict[str, float]:
        return {key: float(observation.get(key, 0.0)) for key in ARM_STATE_KEYS}

    def pop_action_dict(self) -> dict[str, float]:
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            timed_action = self.action_queue.get_nowait()

        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        return action_tensor_to_dict(timed_action.get_action(), self.action_names)

    def _build_timed_observation(
        self,
        task: str,
        observation: dict[str, Any],
        *,
        force: bool,
    ) -> Any:
        with self.latest_action_lock:
            latest_action = self.latest_action

        timed_observation = self._TimedObservation(
            timestamp=time.time(),
            observation={**observation, "task": task},
            timestep=max(latest_action, 0),
        )

        with self.action_queue_lock:
            if force:
                timed_observation.must_go = True
            else:
                timed_observation.must_go = self.must_go.is_set() and self.action_queue.empty()

        return timed_observation

    def enqueue_observation(
        self,
        task: str,
        *,
        force: bool = False,
        observation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        capture_started = time.perf_counter()
        if observation is None:
            observation = self.robot.get_observation()
        if self.frame_preprocessor is not None and self.frame_preprocessor.enabled:
            observation = self.frame_preprocessor.process_observation(observation, self.camera_names)
        capture_ms = (time.perf_counter() - capture_started) * 1000.0

        timed_observation = self._build_timed_observation(task, observation, force=force)
        self.latency.record_obs_capture(timed_observation.get_timestep(), capture_ms)

        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()

        payload = (timed_observation, capture_ms)
        if self._obs_outbound_queue.full():
            try:
                dropped, _ = self._obs_outbound_queue.get_nowait()
                self.logger.warning(
                    f"Dropped pending observation #{dropped.get_timestep()} "
                    f"because upload queue is full"
                )
            except Empty:
                pass
        self._obs_outbound_queue.put(payload)
        self.logger.info(
            f"Queued observation #{timed_observation.get_timestep()} | "
            f"must_go={timed_observation.must_go} | queue_size={queue_size} | "
            f"capture={capture_ms:.0f}ms"
        )
        return observation

    def capture_and_send_observation(
        self,
        task: str,
        *,
        force: bool = False,
        observation: dict[str, Any] | None = None,
        blocking: bool = False,
    ) -> dict[str, Any]:
        if blocking or self._obs_sender_thread is None:
            capture_started = time.perf_counter()
            if observation is None:
                observation = self.robot.get_observation()
            if self.frame_preprocessor is not None and self.frame_preprocessor.enabled:
                observation = self.frame_preprocessor.process_observation(
                    observation, self.camera_names
                )
            capture_ms = (time.perf_counter() - capture_started) * 1000.0

            timed_observation = self._build_timed_observation(task, observation, force=force)
            self.latency.record_obs_capture(timed_observation.get_timestep(), capture_ms)

            with self.action_queue_lock:
                queue_size = self.action_queue.qsize()

            success, payload_mb, serialize_ms, grpc_ms = self.send_observation(timed_observation)
            self.latency.record_obs_send(
                timed_observation.get_timestep(),
                payload_mb=payload_mb,
                serialize_ms=serialize_ms,
                grpc_ms=grpc_ms,
                success=success,
            )
            if success:
                with self._pending_obs_lock:
                    self._pending_obs_sent_at[timed_observation.get_timestep()] = time.monotonic()
                self.logger.info(
                    f"Sent observation #{timed_observation.get_timestep()} | "
                    f"must_go={timed_observation.must_go} | queue_size={queue_size} | "
                    f"payload={payload_mb:.2f}MB | compression={self._obs_image_compression} | "
                    f"capture={capture_ms:.0f}ms | "
                    f"upload={serialize_ms + grpc_ms:.0f}ms"
                )
            if timed_observation.must_go:
                self.must_go.clear()
            return observation

        return self.enqueue_observation(task, force=force, observation=observation)

    def run_control_loop(self, stop_requested: dict[str, bool]) -> None:
        self.start_barrier.wait()
        self.logger.info("Control loop started")

        period = 1.0 / self.args.fps
        started_at = time.monotonic()
        previous_action: dict[str, float] | None = None
        step = 0
        idle_steps = 0
        log_file = None
        if self.args.log_jsonl:
            log_path = Path(self.args.log_jsonl)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", encoding="utf-8")

        try:
            while self.running and not stop_requested["value"]:
                loop_started_at = time.monotonic()
                if time.monotonic() - started_at >= self.args.duration:
                    break

                current_observation = self.robot.get_observation()
                current_action = self._current_action_from_observation(current_observation)
                if previous_action is None:
                    previous_action = current_action

                if self.actions_available():
                    idle_steps = 0
                    predicted_action = self.pop_action_dict()
                    smoothed_action = smooth_action(
                        predicted_action,
                        previous_action,
                        self.args.smoothing_alpha,
                    )

                    if self.args.execute:
                        sent_action = self.robot.send_action(smoothed_action)
                        previous_action = dict(sent_action)
                    else:
                        sent_action = smoothed_action
                        previous_action = smoothed_action

                    if step % self.args.print_every == 0:
                        print_action_summary(f"step {step:04d} pred:", predicted_action)
                        print_action_summary(f"step {step:04d} send:", sent_action)
                        print(
                            f"step {step:04d} delta:"
                            f" pred-current={max_abs_delta(predicted_action, current_action):.4f}"
                            f" send-current={max_abs_delta(sent_action, current_action):.4f}"
                            f" right-target={right_arm_delta(predicted_action, current_action):.4f}",
                            flush=True,
                        )

                    if log_file is not None:
                        log_record = {
                            "timestamp": time.time(),
                            "step": step,
                            "loop_s": time.monotonic() - loop_started_at,
                            "current_action": json_ready(current_action),
                            "predicted_action": json_ready(predicted_action),
                            "sent_action": json_ready(sent_action),
                        }
                        log_file.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                        log_file.flush()

                    step += 1
                else:
                    # Optional hold (ACT-like): avoid starving the arm when remote
                    # inference is late. Disable via --hold-last-action-on-idle=false
                    # / async_inference.hold_last_action_on_idle for baseline A/B.
                    idle_steps += 1
                    hold_enabled = bool(getattr(self.args, "hold_last_action_on_idle", True))
                    holding = False
                    if hold_enabled and previous_action is not None and self.args.execute:
                        sent_action = self.robot.send_action(previous_action)
                        previous_action = dict(sent_action)
                        holding = True
                    if idle_steps == 1 or idle_steps % max(1, int(self.args.fps * 2)) == 0:
                        with self.action_queue_lock:
                            queue_size = self.action_queue.qsize()
                        self.logger.info(
                            f"Waiting for remote actions... idle_steps={idle_steps} "
                            f"queue_size={queue_size} holding_last_action={holding}"
                        )

                if self._ready_to_send_observation():
                    self.enqueue_observation(
                        self.args.task,
                        force=True,
                        observation=current_observation,
                    )

                elapsed = time.monotonic() - loop_started_at
                sleep_remaining = max(0.0, period - elapsed)
                while sleep_remaining > 0 and self.running and not stop_requested["value"]:
                    chunk = min(0.05, sleep_remaining)
                    time.sleep(chunk)
                    sleep_remaining = max(0.0, period - (time.monotonic() - loop_started_at))
        finally:
            if log_file is not None:
                log_file.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Piper as an async robot client.")
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument(
        "--policy-path",
        default="outputs/train/pi05_cube_pi05_v1/checkpoints/last/pretrained_model",
    )
    parser.add_argument("--policy-type", default="pi05")
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--dataset-root", default="data/lerobot/local/cube_pi05_v1")
    parser.add_argument("--task", default="pick_cube")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--actions-per-chunk", type=int, default=50)
    parser.add_argument("--chunk-size-threshold", type=float, default=0.5)
    parser.add_argument(
        "--aggregate-fn-name",
        default="weighted_average",
        choices=("weighted_average", "latest_only", "average", "conservative"),
    )
    parser.add_argument("--follower-left-can", default="can2")
    parser.add_argument("--follower-right-can", default="can0")
    parser.add_argument("--camera-indices", default="8,6,10,12")
    parser.add_argument("--camera-names", default="cam_right,cam_side,cam_top,cam_left")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--control-speed", type=int, default=10)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.025)
    parser.add_argument("--max-gripper-step-m", type=float, default=0.003,
                        help="每控制周期夹爪最大步进（米）；<=0 表示不限速")
    parser.add_argument("--gripper-effort", type=int, default=1000)
    parser.add_argument("--smoothing-alpha", type=float, default=0.25)
    parser.add_argument(
        "--hold-last-action-on-idle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When the remote action queue is empty, keep sending the last action "
        "(ACT-like). Use --no-hold-last-action-on-idle for baseline behavior.",
    )
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--log-jsonl", default="")
    parser.add_argument(
        "--debug-visualize-queue-size",
        action="store_true",
        help="Plot action queue size after the run finishes.",
    )
    parser.add_argument(
        "--network-benchmark-samples",
        type=int,
        default=5,
        help="Ready() RTT samples before connecting (0 to skip).",
    )
    parser.add_argument(
        "--log-latency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print latency summary and per-chunk timing logs.",
    )
    parser.add_argument(
        "--latency-log-jsonl",
        default="",
        help="Optional JSONL path for detailed latency events.",
    )
    parser.add_argument(
        "--obs-image-compression",
        default="jpeg",
        choices=("jpeg", "none"),
        help="Compress camera frames before gRPC upload (jpeg recommended for remote).",
    )
    parser.add_argument(
        "--obs-jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality 1-100 when --obs-image-compression=jpeg.",
    )
    return parser


def run_async_live_policy(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0.")
    if not 0.0 < args.smoothing_alpha <= 1.0:
        raise ValueError("--smoothing-alpha must be in (0, 1].")
    if args.actions_per_chunk <= 0:
        raise ValueError("--actions-per-chunk must be greater than 0.")
    if not 0.0 <= args.chunk_size_threshold <= 1.0:
        raise ValueError("--chunk-size-threshold must be in [0, 1].")
    if not 1 <= int(args.obs_jpeg_quality) <= 100:
        raise ValueError("--obs-jpeg-quality must be in [1, 100].")

    client = PiperAsyncRobotClient(args)

    def interrupt_grpc() -> None:
        client.shutdown_event.set()
        try:
            client.channel.close()
        except Exception:
            pass

    stop_requested = install_stop_handler(interrupt_grpc)

    print("Running Piper async robot client.")
    print(f"  mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print(f"  server: {args.server_address}")
    print(f"  policy: {args.policy_path}")
    print(f"  policy device (remote): {args.policy_device}")
    print(f"  fps: {args.fps}")
    print(f"  cameras: {', '.join(client.camera_names)}")
    print(
        f"  obs compression: {args.obs_image_compression}"
        + (f" q={args.obs_jpeg_quality}" if args.obs_image_compression == "jpeg" else "")
    )
    if not args.execute:
        print("  no actions will be sent; add --execute only after dry-run output looks sane")
    print("Press Ctrl+C to stop.")
    print()

    benchmark_samples = getattr(args, "network_benchmark_samples", 0)
    if benchmark_samples > 0:
        client.benchmark_network_rtt(samples=benchmark_samples)

    if not client.start():
        raise RuntimeError(f"Could not connect to policy server at {args.server_address}")

    print("Remote policy loaded. Connecting robot...", flush=True)
    client.robot.connect()
    if client.frame_preprocessor is not None:
        client.frame_preprocessor.reset()
    client.start_observation_sender()
    client.must_go.set()
    client.capture_and_send_observation(args.task, force=True, blocking=True)
    print("Bootstrap observation sent. Starting action receiver...", flush=True)

    receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
    receiver_thread.start()

    try:
        client.run_control_loop(stop_requested)
    finally:
        client.stop()
        receiver_thread.join(timeout=3.0)
        if args.debug_visualize_queue_size and client.action_queue_size:
            _, _, _, _, _, _, _, _, _, visualize_action_queue_size = _import_async_helpers()
            visualize_action_queue_size(client.action_queue_size)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run_async_live_policy(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
