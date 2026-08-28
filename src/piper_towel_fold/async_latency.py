"""Latency tracking for async remote inference."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(index, len(ordered) - 1))]


def _summarize_ms(values: list[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"n={len(values)} "
        f"min={min(values):.0f}ms "
        f"avg={statistics.fmean(values):.0f}ms "
        f"p50={_percentile(values, 50):.0f}ms "
        f"p95={_percentile(values, 95):.0f}ms "
        f"max={max(values):.0f}ms"
    )


@dataclass
class LatencyTracker:
    enabled: bool = True
    log_jsonl: str = ""

    network_rtt_ms: list[float] = field(default_factory=list)
    obs_capture_ms: list[float] = field(default_factory=list)
    obs_payload_mb: list[float] = field(default_factory=list)
    obs_send_ms: list[float] = field(default_factory=list)
    obs_to_action_ms: list[float] = field(default_factory=list)
    server_to_client_ms: list[float] = field(default_factory=list)
    getactions_empty_count: int = 0
    getactions_empty_ms: list[float] = field(default_factory=list)

    _jsonl_file: Any | None = field(default=None, init=False, repr=False)
    _getactions_wait_started: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.log_jsonl:
            path = Path(self.log_jsonl)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_file = path.open("w", encoding="utf-8")

    def close(self) -> None:
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None

    def _write_event(self, event: dict[str, Any]) -> None:
        if self._jsonl_file is None:
            return
        self._jsonl_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._jsonl_file.flush()

    def record_network_rtt(self, rtt_ms: float) -> None:
        if not self.enabled:
            return
        self.network_rtt_ms.append(rtt_ms)
        self._write_event({"type": "network_rtt_ms", "value": rtt_ms})

    def record_obs_capture(self, timestep: int, capture_ms: float) -> None:
        if not self.enabled:
            return
        self.obs_capture_ms.append(capture_ms)
        self._write_event({"type": "obs_capture_ms", "timestep": timestep, "value": capture_ms})

    def record_obs_send(
        self,
        timestep: int,
        *,
        payload_mb: float,
        serialize_ms: float,
        grpc_ms: float,
        success: bool,
    ) -> None:
        if not self.enabled:
            return
        total_ms = serialize_ms + grpc_ms
        self.obs_payload_mb.append(payload_mb)
        self.obs_send_ms.append(total_ms)
        self._write_event(
            {
                "type": "obs_send_ms",
                "timestep": timestep,
                "payload_mb": payload_mb,
                "serialize_ms": serialize_ms,
                "grpc_ms": grpc_ms,
                "total_ms": total_ms,
                "success": success,
            }
        )

    def record_obs_to_action(self, obs_timestep: int, latency_ms: float) -> None:
        if not self.enabled:
            return
        self.obs_to_action_ms.append(latency_ms)
        self._write_event(
            {"type": "obs_to_action_ms", "obs_timestep": obs_timestep, "value": latency_ms}
        )

    def record_server_to_client(self, action_timestep: int, latency_ms: float) -> None:
        if not self.enabled:
            return
        self.server_to_client_ms.append(latency_ms)
        self._write_event(
            {
                "type": "server_to_client_ms",
                "action_timestep": action_timestep,
                "value": latency_ms,
            }
        )

    def begin_getactions_wait(self) -> None:
        if not self.enabled:
            return
        self._getactions_wait_started = time.monotonic()

    def record_getactions_empty(self) -> None:
        if not self.enabled:
            return
        self.getactions_empty_count += 1
        if self._getactions_wait_started is not None:
            waited_ms = (time.monotonic() - self._getactions_wait_started) * 1000.0
            self.getactions_empty_ms.append(waited_ms)
            self._write_event({"type": "getactions_empty_ms", "value": waited_ms})
        self._getactions_wait_started = None

    def finish_getactions_wait(self) -> None:
        self._getactions_wait_started = None

    def print_summary(self) -> None:
        if not self.enabled:
            return
        print("\n=== Async inference latency summary ===", flush=True)
        print(f"  network RTT (Ready):     {_summarize_ms(self.network_rtt_ms)}", flush=True)
        print(f"  obs capture:             {_summarize_ms(self.obs_capture_ms)}", flush=True)
        print(f"  obs payload (MB):        {_summarize_mb(self.obs_payload_mb)}", flush=True)
        print(f"  obs upload (serialize+gRPC): {_summarize_ms(self.obs_send_ms)}", flush=True)
        print(f"  obs -> action chunk:     {_summarize_ms(self.obs_to_action_ms)}", flush=True)
        print(f"  server -> client stamp: {_summarize_ms(self.server_to_client_ms)}", flush=True)
        if self.getactions_empty_count:
            print(
                f"  GetActions empty waits: count={self.getactions_empty_count} "
                f"| wait time {_summarize_ms(self.getactions_empty_ms)}",
                flush=True,
            )
        if self.log_jsonl:
            print(f"  latency log: {self.log_jsonl}", flush=True)
        print("=======================================\n", flush=True)


def _summarize_mb(values: list[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"n={len(values)} "
        f"avg={statistics.fmean(values):.2f}MB "
        f"max={max(values):.2f}MB"
    )
