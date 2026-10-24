"""
Strict asynchronous client for the native ACE-Step 1.5 REST service.

If ACE-Step is unavailable the request fails closed; Factory never substitutes
harmonic/sine-wave audio while claiming ACE-Step generation.
"""
import asyncio
import os
from typing import Any, Callable, Dict, Optional

import httpx

from factory_contracts.errors import ModelUnavailableError


class ACEStepClient:
    def __init__(self, api_base_url: Optional[str] = None, timeout: float = 600.0):
        self.api_base_url = (
            api_base_url or os.environ.get("ACESTEP_API_URL") or "http://127.0.0.1:8008"
        ).rstrip("/")
        self.timeout = timeout

    async def generate_music(
        self,
        task: str,
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[int, float, str], None]] = None,
    ) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(base_url=self.api_base_url, timeout=5.0) as client:
                health = await client.get("/health")
                if health.status_code >= 400:
                    raise ModelUnavailableError(
                        message="ACE-Step backend is unhealthy.",
                        details={"status_code": health.status_code},
                    )
            async with httpx.AsyncClient(base_url=self.api_base_url, timeout=self.timeout) as client:
                return await self._execute_remote(client, task, params, progress_callback)
        except ModelUnavailableError:
            raise
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(
                message="ACE-Step backend is unavailable. Synthetic music fallback is disabled.",
                details={"error_type": type(exc).__name__},
            ) from exc

    async def _execute_remote(
        self,
        client: httpx.AsyncClient,
        task: str,
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[int, float, str], None]],
    ) -> Dict[str, Any]:
        response = await client.post(f"/v1/{task}", json=params)
        if response.status_code >= 400:
            raise ModelUnavailableError(
                message="ACE-Step rejected the generation request.",
                details={"status_code": response.status_code, "task": task},
            )
        payload = response.json()

        if payload.get("status") == "completed" and payload.get("result"):
            return payload["result"]

        task_id = payload.get("task_id")
        if not task_id:
            raise ModelUnavailableError(message="ACE-Step did not return a task_id.")

        while True:
            await asyncio.sleep(1.0)
            status_response = await client.get(f"/v1/tasks/{task_id}")
            if status_response.status_code >= 400:
                raise ModelUnavailableError(
                    message="ACE-Step task polling failed.",
                    details={"status_code": status_response.status_code, "task_id": task_id},
                )
            data = status_response.json()
            progress = data.get("progress") or {}
            if progress_callback and progress:
                pct = float(progress.get("percent", 0.0))
                result = progress_callback(int(progress.get("step", pct)), pct, str(progress.get("stage", "running")))
                if asyncio.iscoroutine(result):
                    await result
            if data.get("status") == "completed":
                result = data.get("result")
                if not isinstance(result, dict):
                    raise ModelUnavailableError(message="ACE-Step returned an invalid completion payload.")
                return result
            if data.get("status") == "failed":
                raise ModelUnavailableError(
                    message="ACE-Step generation failed.",
                    details={"task_id": task_id},
                )
