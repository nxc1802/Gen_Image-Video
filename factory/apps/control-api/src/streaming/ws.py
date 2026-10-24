"""
factory/apps/control-api/src/streaming/ws.py
WebSocket connection manager and bidirectional JSON event transport.
"""
import asyncio
import json
from collections import defaultdict
from typing import Any, Dict, Set
from fastapi import WebSocket, WebSocketDisconnect

from factory.apps.control_api.src.streaming.manager import StreamManager


class WebSocketHub:
    """Manages active WebSocket connections grouped by job_id."""

    def __init__(self):
        self._connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, job_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[job_id].add(websocket)

    async def disconnect(self, job_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections[job_id].discard(websocket)
            if not self._connections[job_id]:
                self._connections.pop(job_id, None)

    async def broadcast_json(self, job_id: str, payload: Dict[str, Any]) -> None:
        if job_id not in self._connections:
            return
        msg = json.dumps(payload)
        for ws in list(self._connections[job_id]):
            try:
                await ws.send_text(msg)
            except Exception:
                await self.disconnect(job_id, ws)

    async def handle_stream(
        self,
        websocket: WebSocket,
        job_id: str,
        stream_manager: StreamManager,
        on_cancel=None,
    ) -> None:
        await self.connect(job_id, websocket)
        channel = f"factory:events:job:{job_id}"

        async def _forward_events():
            async for item in stream_manager.subscribe(channel):
                await websocket.send_text(json.dumps(item))
                if item.get("event") in ("completed", "failed", "cancelled", "error"):
                    break

        forward_task = asyncio.create_task(_forward_events())

        try:
            while True:
                text = await websocket.receive_text()
                try:
                    data = json.loads(text)
                    if (
                        data.get("action") == "cancel"
                        or data.get("event") == "cancel"
                        or data.get("type") == "cancel"
                    ):
                        if on_cancel:
                            await on_cancel(job_id)
                except Exception:
                    pass
        except WebSocketDisconnect:
            pass
        finally:
            forward_task.cancel()
            await self.disconnect(job_id, websocket)


ws_hub = WebSocketHub()
