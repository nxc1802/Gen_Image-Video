"""
services/video/client.py
Async HTTP & WebSocket client for ComfyUI Video Daemon.
"""
import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional
import httpx

try:
    import websockets
except ImportError:
    websockets = None

logger = logging.getLogger("services.video.client")


class ComfyUIVideoClient:
    """Async client for interacting with the headless ComfyUI Video instance."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8189, timeout: float = 600.0):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.timeout = timeout

    async def get_system_stats(self) -> Dict[str, Any]:
        """Fetches ComfyUI system stats."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=10.0) as client:
            resp = await client.get("/system_stats")
            resp.raise_for_status()
            return resp.json()

    async def is_ready(self) -> bool:
        """Checks if ComfyUI is up and responsive."""
        try:
            stats = await self.get_system_stats()
            return bool(stats)
        except Exception:
            return False

    async def queue_prompt(self, prompt: Dict[str, Any], client_id: str) -> str:
        """Submits prompt graph to ComfyUI /prompt endpoint. Returns prompt_id."""
        payload = {"prompt": prompt, "client_id": client_id}
        async with httpx.AsyncClient(base_url=self.base_url, timeout=30.0) as client:
            resp = await client.post("/prompt", json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"ComfyUI rejected video prompt ({resp.status_code}): {resp.text}")
            data = resp.json()
            prompt_id = data.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"Missing prompt_id in ComfyUI response: {data}")
            return prompt_id

    async def interrupt(self) -> bool:
        """Interrupts currently running execution."""
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0) as client:
                resp = await client.post("/interrupt")
                return resp.status_code == 200
        except Exception as e:
            logger.warning(f"Error interrupting ComfyUI video: {e}")
            return False

    async def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Retrieves execution history for a prompt."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=15.0) as client:
            resp = await client.get(f"/history/{prompt_id}")
            resp.raise_for_status()
            return resp.json()

    async def listen_execution(
        self,
        prompt_id: str,
        client_id: str,
        on_progress: Optional[Callable[[int, int, float, str], Any]] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """Listens on WebSocket for video execution events."""
        uri = f"{self.ws_url}?clientId={client_id}"
        outputs: Dict[str, Any] = {}

        if websockets is not None:
            try:
                async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as ws:
                    while True:
                        if cancel_event and cancel_event.is_set():
                            await self.interrupt()
                            raise asyncio.CancelledError("Execution cancelled by worker")

                        try:
                            raw_msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        except asyncio.TimeoutError:
                            continue

                        if isinstance(raw_msg, bytes):
                            continue

                        msg = json.loads(raw_msg)
                        mtype = msg.get("type")
                        data = msg.get("data", {})

                        if mtype == "progress":
                            step = data.get("value", 0)
                            total = data.get("max", 1)
                            pct = round((step / max(1, total)) * 100.0, 1)
                            if on_progress:
                                res = on_progress(step, total, pct, f"Generating video frames ({step}/{total})")
                                if asyncio.iscoroutine(res):
                                    await res

                        elif mtype == "executed":
                            if data.get("prompt_id") == prompt_id:
                                node_output = data.get("output", {})
                                node_id = data.get("node")
                                outputs[node_id] = node_output

                        elif mtype == "execution_error":
                            if data.get("prompt_id") == prompt_id:
                                err_msg = data.get("exception_message") or data.get("message") or "Unknown ComfyUI error"
                                raise RuntimeError(f"ComfyUI video execution failed: {err_msg}")

                        elif mtype == "status":
                            exec_info = data.get("status", {}).get("exec_info", {})
                            if exec_info.get("queue_remaining") == 0 and outputs:
                                break

            except (asyncio.CancelledError, RuntimeError):
                raise
            except Exception as e:
                logger.warning(f"Video WebSocket closed or failed: {e}, polling history")

        for _ in range(60):
                if cancel_event and cancel_event.is_set():
                    await self.interrupt()
                    raise asyncio.CancelledError("Execution cancelled by worker")
                try:
                    hist = await self.get_history(prompt_id)
                    if prompt_id in hist and "outputs" in hist[prompt_id]:
                        return hist[prompt_id]["outputs"]
                except Exception:
                    pass
                await asyncio.sleep(1.0)

        return outputs
