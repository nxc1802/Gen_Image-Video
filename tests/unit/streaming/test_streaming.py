"""
tests/unit/streaming/test_streaming.py
Comprehensive unit tests for Multi-Modal Streaming, EventHistoryBuffer, StreamManager,
SSE formatting, Studio Inspector stream, and AudioChunkStreamer.
"""
import asyncio
import json
import pytest

from factory.apps.control_api.src.streaming.audio import AudioChunkStreamer
from factory.apps.control_api.src.streaming.history import EventHistoryBuffer
from factory.apps.control_api.src.streaming.inspector import stream_studio_inspector
from factory.apps.control_api.src.streaming.manager import StreamManager
from factory.apps.control_api.src.streaming.sse import (
    format_sse_event,
    stream_job_events_generator,
    stream_tokens_generator,
)


class TestEventHistoryBuffer:
    def test_record_and_get_since(self):
        buf = EventHistoryBuffer(max_events_per_channel=10)
        id1 = buf.record("job-1", "progress", {"percent": 10}, 1000.0)
        id2 = buf.record("job-1", "progress", {"percent": 50}, 1001.0)
        id3 = buf.record("job-1", "completed", {"status": "ok"}, 1002.0)

        assert id1 == 1
        assert id2 == 2
        assert id3 == 3

        # Replay events after id=1
        replayed = buf.get_since("job-1", "1")
        assert len(replayed) == 2
        assert replayed[0].event_id == 2
        assert replayed[0].data["percent"] == 50
        assert replayed[1].event_id == 3

        # Replay after id=3
        empty_replay = buf.get_since("job-1", "3")
        assert len(empty_replay) == 0


class TestSSEFormatting:
    def test_format_sse_event(self):
        out = format_sse_event(
            event="progress",
            data={"percent": 75.0, "step": 15},
            event_id="42",
        )
        assert "id: 42\n" in out
        assert "event: progress\n" in out
        assert 'data: {"percent": 75.0, "step": 15}\n\n' in out

    def test_format_multiline_data(self):
        out = format_sse_event(
            event="message",
            data="line1\nline2",
            event_id="1",
        )
        assert "data: line1\n" in out
        assert "data: line2\n\n" in out


@pytest.mark.asyncio
class TestStreamManager:
    async def test_publish_and_subscribe(self):
        sm = StreamManager(history_size=50)
        channel = "test-channel"

        # Subscribe in background
        received = []

        async def _reader():
            async for item in sm.subscribe(channel):
                received.append(item)
                if item["event"] == "completed":
                    break

        reader_task = asyncio.create_task(_reader())
        await asyncio.sleep(0.01)

        await sm.publish(channel, "progress", {"percent": 10})
        await sm.publish(channel, "progress", {"percent": 50})
        await sm.publish(channel, "completed", {"status": "SUCCEEDED"})

        await asyncio.wait_for(reader_task, timeout=2.0)
        assert len(received) == 3
        assert received[0]["data"]["percent"] == 10
        assert received[1]["data"]["percent"] == 50
        assert received[2]["event"] == "completed"

    async def test_last_event_id_reconnection_replay(self):
        sm = StreamManager(history_size=50)
        channel = "test-reconnect"

        await sm.publish(channel, "progress", {"step": 1})
        await sm.publish(channel, "progress", {"step": 2})
        await sm.publish(channel, "progress", {"step": 3})

        replayed = []

        async def _reconnect_reader():
            # Reconnect with last_event_id="1"
            async for item in sm.subscribe(channel, last_event_id="1"):
                replayed.append(item)
                if len(replayed) == 2:
                    break

        reader_task = asyncio.create_task(_reconnect_reader())
        await asyncio.wait_for(reader_task, timeout=2.0)

        assert len(replayed) == 2
        assert replayed[0]["data"]["step"] == 2
        assert replayed[1]["data"]["step"] == 3

    async def test_reconnection_replay_no_duplicates_on_concurrent_publish(self):
        sm = StreamManager(history_size=50)
        channel = "test-dedup"

        await sm.publish(channel, "progress", {"step": 1})
        await sm.publish(channel, "progress", {"step": 2})
        await sm.publish(channel, "progress", {"step": 3})

        yielded = []
        gen = sm.subscribe(channel, last_event_id="1")

        # First item from historical replay
        item2 = await gen.asend(None)
        yielded.append(item2)

        # Simulate concurrent publish / pre-loading of duplicate event 3 into queue
        sub_queues = list(sm._subscribers.get(channel, []))
        assert len(sub_queues) == 1
        q = sub_queues[0]
        # Put duplicate of event 3 in the live queue
        await q.put({"id": "3", "event": "progress", "data": {"step": 3}, "timestamp": 123.0})
        # And put a terminal event 4
        await q.put({"id": "4", "event": "completed", "data": {"step": 4}, "timestamp": 124.0})

        # Next should be historical event 3
        item3 = await gen.asend(None)
        yielded.append(item3)

        # Next should be event 4 (event 3 popped from q is skipped because eid <= highest_event_id)
        item4 = await gen.asend(None)
        yielded.append(item4)

        assert [y["id"] for y in yielded] == ["2", "3", "4"]
        assert len([y for y in yielded if y["id"] == "3"]) == 1



@pytest.mark.asyncio
class TestStudioInspector:
    async def test_inspector_emits_telemetry_and_ping(self):
        generator = stream_studio_inspector(interval_seconds=0.01, ping_interval_seconds=0.02, max_iterations=4)
        chunks = []
        async for chunk in generator:
            chunks.append(chunk)

        telemetry_chunks = [c for c in chunks if "event: telemetry" in c]
        ping_chunks = [c for c in chunks if "event: ping" in c]

        assert len(telemetry_chunks) >= 1
        assert "gpu_0_vram_used_gb" in telemetry_chunks[0]
        assert "gpu_1_vram_used_gb" in telemetry_chunks[0]
        assert "queue_depth" in telemetry_chunks[0]
        assert len(ping_chunks) >= 1


@pytest.mark.asyncio
class TestAudioChunkStreamer:
    async def test_stream_chunks(self):
        sm = StreamManager()
        channel = "factory:audio:job:job-audio-test"

        received_chunks = []

        async def _client():
            async for chunk in AudioChunkStreamer.stream_chunks("job-audio-test", sm, timeout_seconds=1.0):
                received_chunks.append(chunk)

        client_task = asyncio.create_task(_client())
        await asyncio.sleep(0.01)

        await sm.publish(channel, "audio_chunk", {"chunk_bytes": b"pcm_sample_1"})
        await sm.publish(channel, "audio_chunk", {"chunk_bytes": b"pcm_sample_2"})
        await sm.publish(channel, "audio_end", {})

        await asyncio.wait_for(client_task, timeout=2.0)
        assert len(received_chunks) == 2
        assert received_chunks[0] == b"pcm_sample_1"
        assert received_chunks[1] == b"pcm_sample_2"


@pytest.mark.asyncio
class TestWebSocketHub:
    async def test_websocket_cancellation_with_type_cancel(self):
        from unittest.mock import AsyncMock, MagicMock
        from factory.apps.control_api.src.streaming.ws import WebSocketHub
        from fastapi import WebSocketDisconnect

        hub = WebSocketHub()
        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.receive_text = AsyncMock(side_effect=[
            json.dumps({"type": "cancel"}),
            WebSocketDisconnect(),
        ])

        cancelled_jobs = []
        async def _on_cancel(jid):
            cancelled_jobs.append(jid)

        mock_sm = MagicMock()
        async def _dummy_sub(channel):
            if False:
                yield {}
        mock_sm.subscribe = _dummy_sub

        await hub.handle_stream(
            websocket=mock_ws,
            job_id="job-cancel-test-1",
            stream_manager=mock_sm,
            on_cancel=_on_cancel,
        )

        assert cancelled_jobs == ["job-cancel-test-1"]

    async def test_websocket_cancellation_with_action_cancel(self):
        from unittest.mock import AsyncMock, MagicMock
        from factory.apps.control_api.src.streaming.ws import WebSocketHub
        from fastapi import WebSocketDisconnect

        hub = WebSocketHub()
        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.receive_text = AsyncMock(side_effect=[
            json.dumps({"action": "cancel"}),
            WebSocketDisconnect(),
        ])

        cancelled_jobs = []
        async def _on_cancel(jid):
            cancelled_jobs.append(jid)

        mock_sm = MagicMock()
        async def _dummy_sub(channel):
            if False:
                yield {}
        mock_sm.subscribe = _dummy_sub

        await hub.handle_stream(
            websocket=mock_ws,
            job_id="job-cancel-test-2",
            stream_manager=mock_sm,
            on_cancel=_on_cancel,
        )

        assert cancelled_jobs == ["job-cancel-test-2"]

    async def test_websocket_cancellation_with_event_cancel(self):
        from unittest.mock import AsyncMock, MagicMock
        from factory.apps.control_api.src.streaming.ws import WebSocketHub
        from fastapi import WebSocketDisconnect

        hub = WebSocketHub()
        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.receive_text = AsyncMock(side_effect=[
            json.dumps({"event": "cancel"}),
            WebSocketDisconnect(),
        ])

        cancelled_jobs = []
        async def _on_cancel(jid):
            cancelled_jobs.append(jid)

        mock_sm = MagicMock()
        async def _dummy_sub(channel):
            if False:
                yield {}
        mock_sm.subscribe = _dummy_sub

        await hub.handle_stream(
            websocket=mock_ws,
            job_id="job-cancel-test-3",
            stream_manager=mock_sm,
            on_cancel=_on_cancel,
        )

        assert cancelled_jobs == ["job-cancel-test-3"]

