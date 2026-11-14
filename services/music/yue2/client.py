"""
services/music/yue2/client.py
HTTP and backend client for YuE2 music generation and symbolic score pipeline.
Translates network and backend failures into Factory contract exceptions:
ModelUnavailableError(code="BACKEND_UNAVAILABLE").
"""
import asyncio
import base64
import hashlib
import json
import logging
import os
import uuid
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urljoin

import httpx
from factory_contracts.errors import (
    ErrorCode,
    InvalidRequestError,
    ModelUnavailableError,
)
from services.music.yue2.score import edit_score as local_edit_score, validate_abc_score

logger = logging.getLogger("services.music.yue2.client")


class YuE2Client:
    """Client for YuE2 HTTP inference and score manipulation backend."""

    # Adapter binding to the legacy native protocol, not a request-alias table.
    NATIVE_TASKS = {
        "music.text_to_music": "text_to_music",
        "music.lyrics_to_music": "lyrics_to_music",
        "music.symbolic_score": "symbolic_score",
        "music.custom_score": "custom_score",
        "music.score_edit": "score_edit",
        "music.song_variation": "song_variation",
    }

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        timeout: float = 600.0,
    ):
        raw_url = (
            api_base_url
            if api_base_url is not None
            else (os.environ.get("YUE2_API_URL") or "http://127.0.0.1:8009")
        )
        self.api_base_url = str(raw_url).strip().rstrip("/")
        self.timeout = timeout
        self.api_key = os.environ.get("YUE2_API_KEY") or os.environ.get("FACTORY_API_KEY", "")

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def health(self) -> Dict[str, Any]:
        """Checks if YuE2 backend is reachable and healthy."""
        url = f"{self.api_base_url}/health"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=self._headers())
                if resp.status_code == 200:
                    return resp.json()
                raise ModelUnavailableError(
                    f"YuE2 backend returned unhealthy status {resp.status_code}: {resp.text}",
                    code="BACKEND_UNAVAILABLE",
                )
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                f"YuE2 backend is unavailable at {self.api_base_url}: {exc}",
                code="BACKEND_UNAVAILABLE",
            ) from exc

    async def generate_song(
        self,
        task: str,
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[int, float, str], Any]] = None,
    ) -> Dict[str, Any]:
        """
        Dispatches song generation task to YuE2 backend and collects audio & score artifacts.
        """
        if task not in self.NATIVE_TASKS:
            raise InvalidRequestError(f"Unsupported canonical YuE2 capability: {task}")
        url = f"{self.api_base_url}/v1/generate"
        payload = {
            "task": self.NATIVE_TASKS[task],
            "variant": params.get("variant", "standard"),
            "prompt": params.get("prompt", ""),
            "lyrics": params.get("lyrics", ""),
            "genres": params.get("genres", ""),
            "audio_duration": params.get("audio_duration", 30.0),
            "seed": params.get("seed"),
            "score_prompt": params.get("score_prompt"),
            "sampling_config": params.get("sampling_config", {}),
        }

        if progress_callback:
            try:
                res = progress_callback(1, 0.1, "submitting_task")
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, headers=self._headers(), json=payload)
                if resp.status_code != 200:
                    raise ModelUnavailableError(
                        f"YuE2 backend generate returned status {resp.status_code}: {resp.text}",
                        code="BACKEND_UNAVAILABLE",
                    )
                data = resp.json()

        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                f"YuE2 backend request failed: {exc}",
                code="BACKEND_UNAVAILABLE",
            ) from exc

        # Parse generated audio bytes
        raw_audio: bytes
        if "audio_base64" in data:
            raw_audio = base64.b64decode(data["audio_base64"])
        elif "audio_bytes" in data:
            val = data["audio_bytes"]
            raw_audio = val if isinstance(val, bytes) else base64.b64decode(val)
        else:
            # Check if backend provided audio download path
            audio_url = data.get("audio_url")
            if audio_url:
                try:
                    target_url = urljoin(f"{self.api_base_url}/", audio_url)
                    async with httpx.AsyncClient(timeout=self.timeout) as fetch_client:
                        get_resp = await fetch_client.get(target_url, headers=self._headers())
                        get_resp.raise_for_status()
                        raw_audio = get_resp.content
                except Exception as exc:
                    raise ModelUnavailableError(
                        f"Failed to fetch generated audio from YuE2 backend: {exc}",
                        code="BACKEND_UNAVAILABLE",
                    ) from exc
            else:
                raise ModelUnavailableError(
                    "YuE2 backend response missing audio data or url",
                    code="BACKEND_UNAVAILABLE",
                )

        if progress_callback:
            try:
                res = progress_callback(10, 1.0, "completed")
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass

        job_uid = uuid.uuid4().hex[:12]
        filename = f"yue2_{task}_{job_uid}.wav"
        sha256_hash = hashlib.sha256(raw_audio).hexdigest()
        size_bytes = len(raw_audio)

        audio_art = {
            "type": "audio",
            "audio_bytes": raw_audio,
            "filename": filename,
            "mime_type": "audio/wav",
            "size_bytes": size_bytes,
            "sha256": sha256_hash,
        }

        artifacts = [audio_art]

        # Check for intermediate or generated ABC score artifact
        abc_score = data.get("abc_score") or params.get("score_prompt")
        if abc_score and isinstance(abc_score, str) and abc_score.strip():
            # Validate score syntax
            try:
                validate_abc_score(abc_score)
            except Exception as e:
                logger.warning(f"YuE2 returned score failed validation: {e}")

            score_bytes = abc_score.encode("utf-8")
            score_filename = f"yue2_{task}_{job_uid}.abc"
            score_art = {
                "type": "score",
                "audio_bytes": score_bytes,  # Compatible with worker artifact upload loop
                "filename": score_filename,
                "mime_type": "text/vnd.abc",
                "size_bytes": len(score_bytes),
                "sha256": hashlib.sha256(score_bytes).hexdigest(),
            }
            artifacts.append(score_art)

        return {
            "audio_bytes": raw_audio,
            "filename": filename,
            "mime_type": "audio/wav",
            "size_bytes": size_bytes,
            "sha256": sha256_hash,
            "abc_score": abc_score,
            "artifacts": artifacts,
            "metadata": {
                "model": "yue2",
                "task": task,
                "variant": params.get("variant", "standard"),
                "duration": params.get("audio_duration"),
                "seed": params.get("seed"),
                "backend_version": data.get("version", "2.0.0"),
            },
        }

    async def plan_score(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plans symbolic ABC music score without requiring audio synthesis (YUE-02, YUE-03, YUE-06).
        """
        url = f"{self.api_base_url}/v1/plan_score"
        payload = {
            "prompt": params.get("prompt", ""),
            "lyrics": params.get("lyrics", ""),
            "genres": params.get("genres", ""),
            "sampling_config": params.get("sampling_config", {}),
            "seed": params.get("seed"),
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, headers=self._headers(), json=payload)
                if resp.status_code != 200:
                    raise ModelUnavailableError(
                        f"YuE2 plan_score returned status {resp.status_code}: {resp.text}",
                        code="BACKEND_UNAVAILABLE",
                    )
                data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                f"YuE2 plan_score request failed: {exc}",
                code="BACKEND_UNAVAILABLE",
            ) from exc

        abc_score = data.get("abc_score", "")
        # Validate planned ABC score
        score_info = validate_abc_score(abc_score)

        score_bytes = abc_score.encode("utf-8")
        job_uid = uuid.uuid4().hex[:12]
        filename = f"yue2_plan_{job_uid}.abc"
        sha256_hash = hashlib.sha256(score_bytes).hexdigest()

        score_art = {
            "type": "score",
            "audio_bytes": score_bytes,
            "filename": filename,
            "mime_type": "text/vnd.abc",
            "size_bytes": len(score_bytes),
            "sha256": sha256_hash,
        }

        return {
            "abc_score": abc_score,
            "score_info": score_info,
            "artifacts": [score_art],
            "metadata": {
                "model": "yue2",
                "task": "music.symbolic_score",
                "seed": params.get("seed"),
            },
        }

    async def edit_score(self, base_score: str, edits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Edits symbolic ABC score (notes, chords, tempo, key) and returns modified score (YUE-15, YUE-16).
        Tries remote backend endpoint /v1/edit_score first; falls back to verified local score editor.
        """
        validate_abc_score(base_score)
        url = f"{self.api_base_url}/v1/edit_score"
        payload = {
            "base_score": base_score,
            "modifications": edits,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=self._headers(), json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    new_score = data.get("abc_score") or data.get("score")
                    score_info = validate_abc_score(new_score)
                    score_bytes = new_score.encode("utf-8")
                    job_uid = uuid.uuid4().hex[:12]
                    return {
                        "abc_score": new_score,
                        "score_info": score_info,
                        "artifacts": [{
                            "type": "score",
                            "audio_bytes": score_bytes,
                            "filename": f"yue2_edited_{job_uid}.abc",
                            "mime_type": "text/vnd.abc",
                            "size_bytes": len(score_bytes),
                            "sha256": hashlib.sha256(score_bytes).hexdigest(),
                        }],
                    }
        except (httpx.HTTPError, ValueError):
            # Remote endpoint unavailable; perform local validated editing
            pass

        edited_score = local_edit_score(base_score, edits)
        score_info = validate_abc_score(edited_score)
        score_bytes = edited_score.encode("utf-8")
        job_uid = uuid.uuid4().hex[:12]
        return {
            "abc_score": edited_score,
            "score_info": score_info,
            "artifacts": [{
                "type": "score",
                "audio_bytes": score_bytes,
                "filename": f"yue2_edited_{job_uid}.abc",
                "mime_type": "text/vnd.abc",
                "size_bytes": len(score_bytes),
                "sha256": hashlib.sha256(score_bytes).hexdigest(),
            }],
        }
