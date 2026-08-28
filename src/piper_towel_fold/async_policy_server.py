"""Piper wrapper around LeRobot async PolicyServer with pi05 memory optimizations."""

from __future__ import annotations

import logging
import pickle  # nosec
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from typing import Any

import grpc

from .async_features import observation_features_from_dataset
from .async_image_codec import (
    decompress_observation_images,
    observation_has_jpeg_payloads,
)
from .async_rtc import (
    apply_rtc_to_policy,
    leftover_from_previous_chunk,
    resolve_inference_delay_steps,
)
from .offline_infer import load_policy


def _rename_map_from_preprocessor(preprocessor: Any) -> dict[str, str]:
    steps = getattr(preprocessor, "steps", None) or []
    for step in steps:
        rename = getattr(step, "rename_map", None)
        if isinstance(rename, dict) and rename:
            return dict(rename)
    return {}


def raw_observation_to_observation_with_rename(
    raw_observation: dict[str, Any],
    lerobot_features: dict[str, Any],
    policy_image_features: dict[str, Any],
    rename_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Like LeRobot raw_observation_to_observation, but map dataset camera keys to policy keys.

    X-VLA checkpoints keep robot names (cam_high / cam_left_wrist / ...) in the dataset
    and rename them to image / image2 / image3 in the preprocessor. Upstream async
    inference looks up policy.image_features[cam_high] and KeyErrors.
    """
    from lerobot.async_inference.helpers import (
        extract_state_from_raw_observation,
        is_image_key,
        make_lerobot_observation,
        prepare_image,
        resize_robot_observation_image,
    )
    from lerobot.utils.constants import OBS_STATE
    import torch

    rename_map = rename_map or {}
    lerobot_obs = make_lerobot_observation(raw_observation, lerobot_features)
    image_keys = list(filter(is_image_key, lerobot_obs))
    observation: dict[str, Any] = {OBS_STATE: extract_state_from_raw_observation(lerobot_obs)}
    if "task" in raw_observation:
        observation["task"] = raw_observation["task"]

    for dataset_key in image_keys:
        policy_key = rename_map.get(dataset_key, dataset_key)
        if policy_key not in policy_image_features:
            available = sorted(policy_image_features)
            raise KeyError(
                f"{dataset_key!r} maps to policy key {policy_key!r}, which is not in "
                f"policy.image_features {available}. Need training.rename_map "
                f"(cam_* → image/image2/image3) for X-VLA async inference."
            )
        observation[policy_key] = resize_robot_observation_image(
            torch.tensor(lerobot_obs[dataset_key]),
            policy_image_features[policy_key].shape,
        )

    for key, value in list(observation.items()):
        if isinstance(value, torch.Tensor) and "image" in key:
            observation[key] = prepare_image(value).unsqueeze(0)

    return observation


def _import_policy_server():
    try:
        from lerobot.async_inference.configs import PolicyServerConfig
        from lerobot.async_inference.policy_server import PolicyServer
        from lerobot.transport import services_pb2, services_pb2_grpc
        from lerobot.async_inference.helpers import get_logger
    except ImportError as exc:
        raise ImportError(
            "LeRobot async inference is not installed. "
            'Install with: pip install "lerobot[async,pi]"'
        ) from exc

    return PolicyServer, PolicyServerConfig, get_logger, services_pb2, services_pb2_grpc


class PiperPolicyServer:
    """PolicyServer with load_policy() overrides for low-VRAM pi05 inference."""

    def __init__(
        self,
        config: Any,
        *,
        inference_dtype: str | None = None,
        compile_model: bool | None = None,
        num_inference_steps: int | None = None,
        rtc_options: dict[str, Any] | None = None,
    ) -> None:
        (
            base_server_cls,
            _,
            get_logger,
            services_pb2,
            _,
        ) = _import_policy_server()

        self._base_server_cls = base_server_cls
        self._services_pb2 = services_pb2
        self.logger = get_logger("policy_server")

        self.config = config
        self.inference_dtype = inference_dtype
        self.compile_model = compile_model
        self.num_inference_steps = num_inference_steps
        self.rtc_options = rtc_options if rtc_options and rtc_options.get("enabled") else None
        self._rtc_active = False
        self._rtc_inference_delay = resolve_inference_delay_steps(
            self.rtc_options,
            fps=float(getattr(config, "fps", 20) or 20),
            inference_latency=getattr(config, "inference_latency", None),
        )
        self._rtc_last_raw_chunk: Any | None = None
        self._rtc_last_chunk_timestep: int | None = None
        self._rtc_current_obs_timestep: int = 0

        self._server = base_server_cls(config)
        self._policy_preloaded = False
        self._preloaded_policy_path: str | None = None
        self._image_rename_map: dict[str, str] = {}
        self._patch_raw_observation_converter()
        self._patch_send_policy_instructions()
        self._patch_send_observations()
        self._patch_action_chunk_inference()
        if self.rtc_options:
            self.logger.info(
                "RTC options loaded | delay_steps=%s horizon=%s weight=%s schedule=%s",
                self._rtc_inference_delay,
                self.rtc_options.get("execution_horizon"),
                self.rtc_options.get("max_guidance_weight"),
                self.rtc_options.get("prefix_attention_schedule"),
            )

    def _apply_policy_specs(
        self,
        policy_specs: Any,
        *,
        log_client: str | None = None,
    ) -> float:
        from lerobot.async_inference.constants import SUPPORTED_POLICIES

        # SoftFold extras: upstream lerobot async list may lag (e.g. xvla / act_piper).
        allowed_policies = set(SUPPORTED_POLICIES) | {"xvla", "act_piper"}
        if policy_specs.policy_type not in allowed_policies:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {sorted(allowed_policies)}"
            )

        if log_client is not None:
            self.logger.info(
                f"Receiving policy instructions from {log_client} | "
                f"Policy type: {policy_specs.policy_type} | "
                f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
                f"Actions per chunk: {policy_specs.actions_per_chunk} | "
                f"Device: {policy_specs.device}"
            )

        self._server.device = policy_specs.device
        self._server.policy_type = policy_specs.policy_type
        self._server.lerobot_features = policy_specs.lerobot_features
        self._server.actions_per_chunk = policy_specs.actions_per_chunk

        if (
            self._policy_preloaded
            and self._server.policy is not None
            and self._preloaded_policy_path == policy_specs.pretrained_name_or_path
        ):
            self.logger.info(
                "Policy already loaded at server startup; skipping model reload."
            )
            return 0.0

        start = time.perf_counter()
        policy_config, policy, make_pre_post_processors = load_policy(
            policy_specs.pretrained_name_or_path,
            policy_specs.device,
            inference_dtype=self.inference_dtype,
            compile_model=self.compile_model,
            num_inference_steps=self.num_inference_steps,
        )
        self._rtc_active = apply_rtc_to_policy(policy, self.rtc_options, logger=self.logger)
        self._rtc_last_raw_chunk = None
        self._rtc_last_chunk_timestep = None
        self._server.policy = policy

        device_override = {"device": policy_specs.device}
        # Do not pass an empty rename_map: it would wipe the map saved in the
        # checkpoint preprocessor (needed for xvla cam_* → image/image2/image3).
        preprocessor_overrides: dict[str, Any] = {
            "device_processor": device_override,
        }
        if policy_specs.rename_map:
            preprocessor_overrides["rename_observations_processor"] = {
                "rename_map": policy_specs.rename_map
            }
        self._server.preprocessor, self._server.postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=policy_config.pretrained_path,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides={"device_processor": device_override},
        )
        self._image_rename_map = dict(policy_specs.rename_map or {})
        if not self._image_rename_map:
            self._image_rename_map = _rename_map_from_preprocessor(self._server.preprocessor)
        if self._image_rename_map:
            self.logger.info(f"Image rename_map for async prepare: {self._image_rename_map}")
        elapsed = time.perf_counter() - start
        self._policy_preloaded = True
        self._preloaded_policy_path = policy_specs.pretrained_name_or_path
        self.logger.info(f"Time taken to put policy on {policy_specs.device}: {elapsed:.4f}s")
        return elapsed

    def preload_policy(
        self,
        *,
        policy_type: str,
        policy_path: str,
        dataset_root: str,
        device: str,
        actions_per_chunk: int,
        rename_map: dict[str, str] | None = None,
    ) -> None:
        from lerobot.async_inference.helpers import RemotePolicyConfig

        self.logger.info(
            f"Preloading policy at startup | type={policy_type} | path={policy_path} | device={device}"
        )
        policy_specs = RemotePolicyConfig(
            policy_type,
            policy_path,
            observation_features_from_dataset(dataset_root),
            actions_per_chunk,
            device,
            rename_map=rename_map or {},
        )
        elapsed = self._apply_policy_specs(policy_specs)
        if elapsed > 0:
            self.logger.info(f"Startup preload finished in {elapsed:.1f}s")
        else:
            self.logger.info("Startup preload finished (policy was already loaded)")

    def _patch_send_policy_instructions(self) -> None:
        original = self._server.SendPolicyInstructions

        def send_policy_instructions(request, context):  # noqa: N802
            if not self._server.running:
                self.logger.warning("Server is not running. Ignoring policy instructions.")
                return self._services_pb2.Empty()

            client_id = context.peer()
            policy_specs = pickle.loads(request.data)  # nosec

            try:
                from lerobot.async_inference.helpers import RemotePolicyConfig
            except ImportError as exc:
                raise ImportError(
                    'LeRobot async inference is not installed. '
                    'Install with: pip install "lerobot[async,pi]"'
                ) from exc

            if not isinstance(policy_specs, RemotePolicyConfig):
                raise TypeError(
                    f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}"
                )

            # New client session: drop leftover from any previous run.
            self._rtc_last_raw_chunk = None
            self._rtc_last_chunk_timestep = None
            self._apply_policy_specs(policy_specs, log_client=client_id)
            return self._services_pb2.Empty()

        self._server.SendPolicyInstructions = send_policy_instructions
        _ = original

    def _patch_raw_observation_converter(self) -> None:
        from lerobot.async_inference import policy_server as lerobot_policy_server

        def converted(raw_observation, lerobot_features, policy_image_features):
            return raw_observation_to_observation_with_rename(
                raw_observation,
                lerobot_features,
                policy_image_features,
                self._image_rename_map,
            )

        lerobot_policy_server.raw_observation_to_observation = converted

    def _patch_send_observations(self) -> None:
        """Decode Piper JPEG image payloads before LeRobot enqueue/inference."""
        original = self._server.SendObservations

        def send_observations(request_iterator, context):  # noqa: N802
            from lerobot.async_inference.helpers import TimedObservation
            from lerobot.transport.utils import receive_bytes_in_chunks

            if not self._server.running:
                self.logger.warning("Server is not running. Ignoring observations.")
                return self._services_pb2.Empty()

            client_id = context.peer()
            self.logger.debug(f"Receiving observations from {client_id}")

            receive_time = time.time()
            start_deserialize = time.perf_counter()
            received_bytes = receive_bytes_in_chunks(
                request_iterator,
                None,
                self._server.shutdown_event,
                self.logger,
            )
            timed_observation = pickle.loads(received_bytes)  # nosec
            deserialize_time = time.perf_counter() - start_deserialize

            if not isinstance(timed_observation, TimedObservation):
                raise TypeError(
                    f"Expected TimedObservation, got {type(timed_observation)}"
                )

            raw_observation = timed_observation.get_observation()
            if observation_has_jpeg_payloads(raw_observation):
                decode_started = time.perf_counter()
                timed_observation.observation = decompress_observation_images(raw_observation)
                decode_ms = (time.perf_counter() - decode_started) * 1000.0
                self.logger.debug(
                    f"Decoded JPEG observation #{timed_observation.get_timestep()} "
                    f"in {decode_ms:.1f}ms | wire={len(received_bytes) / (1024 * 1024):.2f}MB"
                )

            obs_timestep = timed_observation.get_timestep()
            obs_timestamp = timed_observation.get_timestamp()
            fps_metrics = self._server.fps_tracker.calculate_fps_metrics(obs_timestamp)

            self.logger.debug(
                f"Received observation #{obs_timestep} | "
                f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                f"Target: {fps_metrics['target_fps']:.2f} | "
                f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
            )
            self.logger.debug(
                f"Server timestamp: {receive_time:.6f} | "
                f"Client timestamp: {obs_timestamp:.6f} | "
                f"Deserialization time: {deserialize_time:.6f}s"
            )

            if not self._server._enqueue_observation(timed_observation):
                self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

            return self._services_pb2.Empty()

        self._server.SendObservations = send_observations
        _ = original

    def _rtc_predict_kwargs(self) -> dict[str, Any]:
        leftover = leftover_from_previous_chunk(
            self._rtc_last_raw_chunk,
            chunk_start_timestep=int(self._rtc_last_chunk_timestep or 0),
            observation_timestep=int(self._rtc_current_obs_timestep),
        )
        if leftover is not None:
            device = next(self._server.policy.parameters()).device
            leftover = leftover.to(device=device)
            if leftover.ndim == 2:
                leftover = leftover.unsqueeze(0)

        return {
            "inference_delay": int(self._rtc_inference_delay),
            "prev_chunk_left_over": leftover,
        }

    def _patch_action_chunk_inference(self) -> None:
        """Optionally pass RTC kwargs into ``predict_action_chunk``.

        When RTC is disabled / absent, behavior matches upstream LeRobot.
        """
        server = self._server
        original_get = server._get_action_chunk
        original_predict = server._predict_action_chunk

        def get_action_chunk(observation):  # noqa: ANN001
            if not self._rtc_active:
                return original_get(observation)

            kwargs = self._rtc_predict_kwargs()
            leftover = kwargs.get("prev_chunk_left_over")
            leftover_steps = 0 if leftover is None else int(leftover.shape[-2])
            self.logger.info(
                "RTC predict_action_chunk | obs_timestep=%s delay=%s leftover_steps=%s",
                self._rtc_current_obs_timestep,
                kwargs.get("inference_delay"),
                leftover_steps,
            )

            chunk = server.policy.predict_action_chunk(observation, **kwargs)
            if chunk.ndim != 3:
                chunk = chunk.unsqueeze(0)
            chunk = chunk[:, : server.actions_per_chunk, :]

            # Store raw (pre-postprocess) chunk for the next RTC prefix.
            self._rtc_last_raw_chunk = chunk.detach().cpu()
            self._rtc_last_chunk_timestep = int(self._rtc_current_obs_timestep)
            return chunk

        def predict_action_chunk(observation_t):  # noqa: ANN001
            self._rtc_current_obs_timestep = int(observation_t.get_timestep())
            return original_predict(observation_t)

        server._get_action_chunk = get_action_chunk
        server._predict_action_chunk = predict_action_chunk
        _ = original_get

    @property
    def servicer(self) -> Any:
        return self._server

    def stop(self) -> None:
        self._server.stop()


def serve_piper_policy_server(
    config: Any,
    *,
    inference_dtype: str | None = None,
    compile_model: bool | None = None,
    num_inference_steps: int | None = None,
    rtc_options: dict[str, Any] | None = None,
    preload_policy: dict[str, Any] | None = None,
) -> None:
    _, _, _, _, services_pb2_grpc = _import_policy_server()

    logging.info(pformat(asdict(config)))
    policy_server = PiperPolicyServer(
        config,
        inference_dtype=inference_dtype,
        compile_model=compile_model,
        num_inference_steps=num_inference_steps,
        rtc_options=rtc_options,
    )

    if preload_policy is not None:
        policy_server.preload_policy(**preload_policy)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server.servicer, server)
    server.add_insecure_port(f"{config.host}:{config.port}")

    policy_server.logger.info(f"PolicyServer started on {config.host}:{config.port}")
    server.start()
    server.wait_for_termination()
    policy_server.logger.info("Server terminated")
