"""
factory/apps/control-api/src/streaming/sse.py
Server-Sent Events (SSE) formatting, keepalives, and generator streams.
"""
import asyncio
import json
from typing import Any, AsyncGenerator, Dict, Optional

from factory.apps.control_api.src.streaming.manager import StreamManager


def format_sse_event(
    event: str,
    data: Any,
    event_id: Optional[str] = None,
    retry: Optional[int] = None,
) -> str:
    """Formats payload into standard W3C text/event-stream chunk."""
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    if event:
        lines.append(f"event: {event}")
    if retry is not None:
        lines.append(f"retry: {retry}")

    if isinstance(data, (dict, list)):
        payload_str = json.dumps(data)
    else:
        payload_str = str(data)

    for line in payload_str.split("\n"):
        lines.append(f"data: {line}")

    return "\n".join(lines) + "\n\n"


async def stream_job_events_generator(
    job_id: str,
    stream_manager: StreamManager,
    last_event_id: Optional[str] = None,
    initial_event: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    """
    SSE generator yielding real-time job events for GET /v1/jobs/{job_id}/events.
    Handles replay of historical events via Last-Event-ID.
    """
    channel = f"factory:events:job:{job_id}"

    if initial_event:
        yield format_sse_event(
            event=initial_event.get("event", "progress"),
            data=initial_event.get("data", {}),
            event_id=initial_event.get("id", "0"),
        )
        if initial_event.get("event") in ("completed", "failed", "cancelled", "error"):
            return

    keepalive_interval = 15.0

    async def _subscriber():
        async for item in stream_manager.subscribe(channel, last_event_id=last_event_id):
            yield item

    sub_gen = _subscriber()

    try:
        while True:
            try:
                item = await asyncio.wait_for(anext(sub_gen), timeout=keepalive_interval)
                yield format_sse_event(
                    event=item.get("event", "progress"),
                    data=item.get("data", {}),
                    event_id=item.get("id"),
                )
                if item.get("event") in ("completed", "failed", "cancelled", "error"):
                    break
            except asyncio.TimeoutError:
                # Keepalive comment
                yield ": keepalive\n\n"
            except StopAsyncIteration:
                break
    finally:
        await sub_gen.aclose()


async def stream_tokens_generator(
    job_id: str,
    stream_manager: StreamManager,
) -> AsyncGenerator[str, None]:
    """
    SSE generator yielding OpenAI-compatible delta chunks for GET /v1/jobs/{job_id}/stream/tokens.
    Ends with 'data: [DONE]\\n\\n'.
    """
    channel = f"factory:tokens:job:{job_id}"
    keepalive_interval = 15.0

    async def _subscriber():
        async for item in stream_manager.subscribe(channel):
            yield item

    sub_gen = _subscriber()

    try:
        while True:
            try:
                item = await asyncio.wait_for(anext(sub_gen), timeout=keepalive_interval)
                data = item.get("data", {})
                if item.get("event") == "done" or data == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break

                yield f"data: {json.dumps(data)}\n\n"
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
            except StopAsyncIteration:
                yield "data: [DONE]\n\n"
                break
    finally:
        await sub_gen.aclose()
