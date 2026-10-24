"""
factory/apps/control-api/src/streaming/audio.py
Binary audio chunk streaming for chunked HTTP and WebSocket speech synthesis/transcription.
"""
import asyncio
from typing import AsyncGenerator

from factory.apps.control_api.src.streaming.manager import StreamManager


class AudioChunkStreamer:
    """Streams binary audio chunks (PCM 16-bit 24kHz or Opus) with backpressure handling."""

    @staticmethod
    async def stream_chunks(
        job_id: str,
        stream_manager: StreamManager,
        timeout_seconds: float = 30.0,
    ) -> AsyncGenerator[bytes, None]:
        channel = f"factory:audio:job:{job_id}"

        async def _subscriber():
            async for item in stream_manager.subscribe(channel):
                yield item

        sub_gen = _subscriber()

        try:
            while True:
                try:
                    item = await asyncio.wait_for(anext(sub_gen), timeout=timeout_seconds)
                    event = item.get("event")
                    if event == "audio_end":
                        break
                    
                    data = item.get("data", {})
                    if isinstance(data, dict):
                        chunk = data.get("chunk_bytes")
                        if isinstance(chunk, bytes):
                            yield chunk
                        elif isinstance(chunk, str):
                            import base64
                            yield base64.b64decode(chunk)
                    elif isinstance(data, bytes):
                        yield data
                except (asyncio.TimeoutError, StopAsyncIteration):
                    break
        finally:
            await sub_gen.aclose()
