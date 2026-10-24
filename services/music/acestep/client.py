"""
Strict client for ACE-Step 1.5 native asynchronous REST API.

Contract:
  POST /release_task -> data.task_id
  POST /query_result {"task_id_list": [...]} -> status 0/1/2
  GET  /v1/audio?path=... -> generated audio bytes
"""
import asyncio
import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urljoin, urlparse, parse_qs

import httpx

from factory_contracts.errors import InvalidRequestError, ModelUnavailableError


class ACEStepClient:
    def __init__(self, api_base_url: Optional[str] = None, timeout: float = 600.0):
        self.api_base_url = (
            api_base_url or os.environ.get("ACESTEP_API_URL") or "http://127.0.0.1:8008"
        ).rstrip("/")
        self.api_key = os.environ.get("ACESTEP_API_KEY") or os.environ.get("FACTORY_API_KEY", "")
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _release_payload(task: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if task not in {"text_to_music", "lyrics_to_music"}:
            raise InvalidRequestError(
                f"ACE-Step Factory adapter does not yet support source-audio task '{task}'."
            )

        prompt = str(params.get("prompt") or "").strip()
        if not prompt:
            raise InvalidRequestError("Music generation requires a non-empty prompt.")

        lyrics = str(params.get("lyrics") or "").strip()
        if task == "text_to_music" and not lyrics:
            lyrics = "[inst]"

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "lyrics": lyrics,
            "task_type": "text2music",
            "thinking": bool(params.get("thinking", False)),
            "audio_format": str(params.get("audio_format", "mp3")),
            "audio_duration": float(params.get("duration_seconds", params.get("audio_duration", 60))),
            "bpm": int(params.get("tempo", params.get("bpm", 120))),
            "use_random_seed": params.get("seed") is None,
        }
        if params.get("seed") is not None:
            payload["seed"] = int(params["seed"])
            payload["use_random_seed"] = False
        return payload

    async def generate_music(
        self,
        task: str,
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[int, float, str], Any]] = None,
    ) -> Dict[str, Any]:
        payload = self._release_payload(task, params)
        try:
            async with httpx.AsyncClient(
                base_url=self.api_base_url,
                timeout=self.timeout,
                headers=self._headers(),
            ) as client:
                health = await client.get("/health")
                if health.status_code >= 400:
                    raise ModelUnavailableError(
                        message="ACE-Step backend is unhealthy.",
                        details={"status_code": health.status_code},
                    )

                release = await client.post("/release_task", json=payload)
                if release.status_code >= 400:
                    raise ModelUnavailableError(
                        message="ACE-Step rejected the generation request.",
                        details={"status_code": release.status_code},
                    )
                wrapped = release.json()
                data = wrapped.get("data") if isinstance(wrapped, dict) else None
                task_id = data.get("task_id") if isinstance(data, dict) else None
                if not task_id:
                    raise ModelUnavailableError(message="ACE-Step did not return data.task_id.")

                while True:
                    await asyncio.sleep(1.0)
                    result_response = await client.post(
                        "/query_result",
                        json={"task_id_list": [task_id]},
                    )
                    if result_response.status_code >= 400:
                        raise ModelUnavailableError(
                            message="ACE-Step result polling failed.",
                            details={"status_code": result_response.status_code},
                        )
                    result_wrapper = result_response.json()
                    rows = result_wrapper.get("data") if isinstance(result_wrapper, dict) else None
                    if not isinstance(rows, list) or not rows:
                        continue

                    row = rows[0]
                    status = int(row.get("status", 0))
                    raw_result = row.get("result", "[]")
                    try:
                        result_items = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
                    except json.JSONDecodeError as exc:
                        raise ModelUnavailableError(message="ACE-Step returned malformed result JSON.") from exc

                    if status == 0:
                        actual_progress = 0.0
                        stage = str(row.get("progress_text") or "generating")
                        if isinstance(result_items, list) and result_items and isinstance(result_items[0], dict):
                            actual_progress = float(result_items[0].get("progress", 0.0) or 0.0)
                            stage = str(result_items[0].get("stage") or stage)
                        if progress_callback:
                            callback_result = progress_callback(
                                int(round(actual_progress)),
                                max(0.0, min(100.0, actual_progress)),
                                stage,
                            )
                            if asyncio.iscoroutine(callback_result):
                                await callback_result
                        continue

                    if status == 2:
                        raise ModelUnavailableError(
                            message="ACE-Step generation failed.",
                            details={"task_id": task_id},
                        )

                    if status != 1 or not isinstance(result_items, list) or not result_items:
                        raise ModelUnavailableError(message="ACE-Step returned an invalid completed result.")

                    first = result_items[0]
                    file_ref = first.get("file") if isinstance(first, dict) else None
                    if not file_ref:
                        raise ModelUnavailableError(message="ACE-Step completed without an audio file reference.")

                    audio_url = urljoin(self.api_base_url + "/", str(file_ref).lstrip("/"))
                    audio_response = await client.get(audio_url)
                    if audio_response.status_code >= 400 or not audio_response.content:
                        raise ModelUnavailableError(
                            message="ACE-Step generated audio could not be downloaded.",
                            details={"status_code": audio_response.status_code},
                        )

                    parsed = urlparse(str(file_ref))
                    source_path = (parse_qs(parsed.query).get("path") or ["output.mp3"])[0]
                    filename = Path(source_path).name or "output.mp3"
                    mime = audio_response.headers.get("content-type") or mimetypes.guess_type(filename)[0] or "audio/mpeg"
                    audio_bytes = audio_response.content
                    return {
                        "task_id": task_id,
                        "filename": filename,
                        "mime_type": mime.split(";", 1)[0],
                        "audio_bytes": audio_bytes,
                        "size_bytes": len(audio_bytes),
                        "sha256": hashlib.sha256(audio_bytes).hexdigest(),
                        "metadata": first.get("metas", {}) if isinstance(first, dict) else {},
                    }
        except (InvalidRequestError, ModelUnavailableError):
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                message="ACE-Step backend is unavailable or returned an invalid response.",
                details={"error_type": type(exc).__name__},
            ) from exc
