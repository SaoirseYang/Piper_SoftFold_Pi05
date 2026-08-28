"""Optional Real-Time Chunking (RTC) helpers for SoftFold async PolicyServer.

RTC is server-side guided flow-matching (LeRobot ``lerobot.policies.rtc``).
Configs without a ``rtc`` section keep the previous async-inference behavior.
"""

from __future__ import annotations

from typing import Any


def _as_dict(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be an object when present.")
    return value


def resolve_rtc_options(config: dict[str, Any]) -> dict[str, Any] | None:
    """Parse RTC options from JSON config.

    Lookup order: ``policy_server.rtc`` then ``async_inference.rtc``.
    Returns ``None`` when the section is absent or ``enabled`` is false so
    legacy configs keep identical inference behavior.
    """
    server_cfg = _as_dict(config.get("policy_server"), name="policy_server")
    async_cfg = _as_dict(config.get("async_inference"), name="async_inference")
    rtc = server_cfg.get("rtc")
    if rtc is None:
        rtc = async_cfg.get("rtc")
    if rtc is None:
        return None

    rtc = _as_dict(rtc, name="rtc")
    if not bool(rtc.get("enabled", False)):
        return None

    schedule = str(rtc.get("prefix_attention_schedule", "EXP")).strip().upper()
    delay = rtc.get("inference_delay_steps")
    if delay is not None:
        delay = int(delay)
        if delay < 0:
            raise ValueError(f"rtc.inference_delay_steps must be >= 0, got {delay}")

    execution_horizon = int(rtc.get("execution_horizon", 10))
    if execution_horizon <= 0:
        raise ValueError(f"rtc.execution_horizon must be > 0, got {execution_horizon}")

    max_guidance_weight = float(rtc.get("max_guidance_weight", 10.0))
    if max_guidance_weight <= 0:
        raise ValueError(f"rtc.max_guidance_weight must be > 0, got {max_guidance_weight}")

    return {
        "enabled": True,
        "execution_horizon": execution_horizon,
        "max_guidance_weight": max_guidance_weight,
        "prefix_attention_schedule": schedule,
        "inference_delay_steps": delay,
        "debug": bool(rtc.get("debug", False)),
        "debug_maxlen": int(rtc.get("debug_maxlen", 100)),
    }


def resolve_inference_delay_steps(
    rtc_options: dict[str, Any] | None,
    *,
    fps: float,
    inference_latency: float | None,
) -> int:
    """Prefer explicit ``inference_delay_steps``; else round(latency * fps)."""
    if not rtc_options or not rtc_options.get("enabled"):
        return 0
    explicit = rtc_options.get("inference_delay_steps")
    if explicit is not None:
        return max(0, int(explicit))
    if inference_latency is None or fps <= 0:
        return 0
    return max(0, int(round(float(inference_latency) * float(fps))))


def build_lerobot_rtc_config(rtc_options: dict[str, Any]) -> Any:
    """Build ``lerobot.policies.rtc.RTCConfig`` from SoftFold options."""
    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.rtc import RTCConfig

    schedule_name = str(rtc_options.get("prefix_attention_schedule", "EXP")).upper()
    try:
        schedule = RTCAttentionSchedule[schedule_name]
    except KeyError as exc:
        valid = ", ".join(item.name for item in RTCAttentionSchedule)
        raise ValueError(
            f"Unknown rtc.prefix_attention_schedule={schedule_name!r}. Valid: {valid}"
        ) from exc

    return RTCConfig(
        enabled=True,
        execution_horizon=int(rtc_options["execution_horizon"]),
        max_guidance_weight=float(rtc_options["max_guidance_weight"]),
        prefix_attention_schedule=schedule,
        debug=bool(rtc_options.get("debug", False)),
        debug_maxlen=int(rtc_options.get("debug_maxlen", 100)),
    )


def apply_rtc_to_policy(policy: Any, rtc_options: dict[str, Any] | None, logger: Any = None) -> bool:
    """Attach RTC config + processor to a loaded policy.

    Returns True if RTC is active on the policy. No-op / False when options are
    absent, disabled, or the policy type has no ``rtc_config`` support.
    """
    if not rtc_options or not rtc_options.get("enabled"):
        return False

    config = getattr(policy, "config", None)
    if config is None or not hasattr(config, "rtc_config"):
        message = (
            f"RTC enabled in config but policy type "
            f"{getattr(config, 'type', type(policy).__name__)!r} has no rtc_config; "
            "continuing without RTC."
        )
        if logger is not None:
            logger.warning(message)
        else:
            print(message, flush=True)
        return False

    rtc_cfg = build_lerobot_rtc_config(rtc_options)
    config.rtc_config = rtc_cfg
    if hasattr(policy, "init_rtc_processor"):
        policy.init_rtc_processor()
    elif logger is not None:
        logger.warning("Policy has rtc_config but no init_rtc_processor(); RTC may be inactive.")

    active = bool(getattr(policy, "_rtc_enabled", lambda: False)())
    if logger is not None:
        logger.info(
            "RTC %s | horizon=%s weight=%s schedule=%s",
            "enabled" if active else "configured but inactive",
            rtc_cfg.execution_horizon,
            rtc_cfg.max_guidance_weight,
            rtc_cfg.prefix_attention_schedule,
        )
    return active


def leftover_from_previous_chunk(
    previous_chunk: Any,
    *,
    chunk_start_timestep: int,
    observation_timestep: int,
) -> Any | None:
    """Slice unexecuted prefix of the previous raw action chunk.

    SoftFold client sets observation timestep to the latest *executed* action
    timestep, so actions with index ``<= observation_timestep - chunk_start``
    are already consumed.
    """
    import torch

    if previous_chunk is None:
        return None
    if not isinstance(previous_chunk, torch.Tensor):
        return None
    if previous_chunk.ndim == 3:
        previous_chunk = previous_chunk.squeeze(0)
    if previous_chunk.ndim != 2:
        return None

    consumed = int(observation_timestep) - int(chunk_start_timestep) + 1
    if consumed <= 0:
        return previous_chunk.clone()
    if consumed >= previous_chunk.shape[0]:
        return None
    return previous_chunk[consumed:].clone()
