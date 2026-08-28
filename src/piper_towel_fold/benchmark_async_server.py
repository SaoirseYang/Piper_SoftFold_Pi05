"""Benchmark gRPC latency to the remote async policy server (no robot required)."""

from __future__ import annotations

import argparse
import statistics
import time


def _import_grpc():
    try:
        from lerobot.transport import services_pb2, services_pb2_grpc
        from lerobot.transport.utils import grpc_channel_options
    except ImportError as exc:
        raise ImportError(
            'LeRobot async inference is not installed. Install with: pip install "lerobot[async]"'
        ) from exc
    return services_pb2, services_pb2_grpc, grpc_channel_options


def benchmark_ready_rtt(server_address: str, samples: int, timeout_s: float) -> list[float]:
    import grpc

    services_pb2, services_pb2_grpc, grpc_channel_options = _import_grpc()
    channel = grpc.insecure_channel(server_address, grpc_channel_options())
    stub = services_pb2_grpc.AsyncInferenceStub(channel)

    timings_ms: list[float] = []
    try:
        for index in range(samples):
            started = time.perf_counter()
            stub.Ready(services_pb2.Empty(), timeout=timeout_s)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            timings_ms.append(elapsed_ms)
            print(f"  Ready #{index + 1}: {elapsed_ms:.1f} ms")
    finally:
        channel.close()
    return timings_ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark network RTT to async policy server via Ready()."
    )
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be > 0")

    print(f"Benchmarking Ready() RTT to {args.server_address} ({args.samples} samples)...")
    timings_ms = benchmark_ready_rtt(args.server_address, args.samples, args.timeout)
    if not timings_ms:
        raise RuntimeError("No successful Ready() samples.")

    print("\nSummary:")
    print(f"  min : {min(timings_ms):.1f} ms")
    print(f"  avg : {statistics.fmean(timings_ms):.1f} ms")
    print(f"  max : {max(timings_ms):.1f} ms")
    print(
        "\nNote: Ready() only measures lightweight gRPC round-trip. "
        "Full obs->action latency is dominated by image upload + GPU inference (~1s+)."
    )


if __name__ == "__main__":
    main()
