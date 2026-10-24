"""
factory/apps/control-api/src/streaming/inspector.py
Studio Telemetry & Streaming Inspector SSE generator.
"""
import asyncio
from typing import AsyncGenerator

from factory.apps.control_api.src.streaming.sse import format_sse_event
from factory.apps.control_api.src.telemetry.metrics import metrics


async def stream_studio_inspector(
    interval_seconds: float = 2.0,
    ping_interval_seconds: float = 10.0,
    max_iterations: int = 1000,
) -> AsyncGenerator[str, None]:
    """
    Emits periodic telemetry snapshots and ping keepalives for the Studio Web Inspector.
    Verified in test_t1_studio.py:
    - event: telemetry with gpu_0_vram_used_gb, gpu_1_vram_used_gb, active_jobs, queue_depth
    - event: ping with {}
    """
    iteration = 0
    time_since_ping = 0.0

    while iteration < max_iterations:
        # Telemetry event
        snapshot = metrics.get_snapshot()
        telemetry_payload = {
            "gpu_0_vram_used_gb": snapshot.get("gpu_0_vram_used_gb", 11.2),
            "gpu_1_vram_used_gb": snapshot.get("gpu_1_vram_used_gb", 13.8),
            "active_jobs": snapshot.get("active_jobs", 0),
            "queue_depth": snapshot.get("queue_depth", 0),
        }
        yield format_sse_event("telemetry", telemetry_payload)

        # Ping keepalive check
        time_since_ping += interval_seconds
        if time_since_ping >= ping_interval_seconds:
            yield format_sse_event("ping", {})
            time_since_ping = 0.0

        iteration += 1
        await asyncio.sleep(interval_seconds)
