"""
Strict client for ACE-Step 1.5 native asynchronous REST API.

Contract:
  POST /release_task -> data.task_id
  POST /query_result {"task_id_list": [...]} -> status 0/1/2
  GET  /v1/audio?path=... -> generated audio bytes
  POST /understand -> audio caption and metadata
  POST /score -> audio quality scoring
  POST /cancel_task -> task abort
  POST /train_lora -> LoRA training workflow contract
"""
import asyncio
import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urljoin, urlparse

import httpx

from factory_contracts.errors import (
    ErrorCode,
    InvalidRequestError,
    ModelUnavailableError,
    UnsupportedCapabilityError,
)


class ACEStepClient:
    def __init__(self, api_base_url: Optional[str] = None, timeout: float = 600.0):
        if api_base_url is not None:
            self.api_base_url = (api_base_url or "").strip().rstrip("/")
        else:
            self.api_base_url = (
                os.environ.get("ACESTEP_API_URL") or "http://127.0.0.1:8008"
            ).strip().rstrip("/")
        self.api_key = os.environ.get("ACESTEP_API_KEY") or os.environ.get("FACTORY_API_KEY", "")
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # Canonical capability -> native API operation, not request aliases.
    TASK_TYPE_MAPPING = {'music.text_to_music': 'text2music', 'music.lyrics_to_music': 'lyrics2music', 'music.cover': 'cover', 'music.repaint': 'repaint', 'music.reference_audio': 'reference', 'music.extract': 'extract', 'music.lego': 'lego', 'music.complete': 'complete', 'music.extend': 'extend', 'music.audio_to_audio': 'audio2audio'}

    @classmethod
    def _release_payload(cls, task: str, params: Dict[str, Any]) -> Dict[str, Any]:
        norm_task = task
        task_type = cls.TASK_TYPE_MAPPING.get(norm_task)
        if not task_type:
            raise InvalidRequestError(
                f"ACE-Step Factory adapter does not support task '{task}'. "
                f"Supported tasks: {sorted(list(cls.TASK_TYPE_MAPPING.keys()))}"
            )

        prompt = str(params.get("prompt") or "").strip()
        if not prompt and task_type not in {"extract", "complete"}:
            raise InvalidRequestError("Music generation requires a non-empty prompt.")

        lyrics = str(params.get("lyrics") or "").strip()
        if task_type == "text2music" and not lyrics:
            lyrics = "[inst]"

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "lyrics": lyrics,
            "task_type": task_type,
            "variant": str(params.get("variant", "base")),
            "thinking": bool(params.get("thinking", False)),
            "audio_format": str(params.get("audio_format", params.get("output_format", "mp3"))),
            "audio_duration": float(params.get("duration_seconds", params.get("audio_duration", 60))),
            "bpm": int(params.get("tempo", params.get("bpm", 120))),
            "use_random_seed": params.get("seed") is None,
        }
        if params.get("seed") is not None:
            payload["seed"] = int(params["seed"])
            payload["use_random_seed"] = False

        if params.get("key"):
            payload["key"] = str(params["key"])
        if params.get("time_signature"):
            payload["time_signature"] = str(params["time_signature"])
        if params.get("prompt_expansion"):
            payload["prompt_expansion"] = bool(params["prompt_expansion"])

        # Native source audio & reference mappings
        src_path = (
            params.get("src_audio_path")
            or params.get("source_audio_path")
            or params.get("source_audio_asset_id")
            or params.get("audio_url")
        )
        if src_path:
            payload["src_audio_path"] = str(src_path)

        ref_path = (
            params.get("reference_audio_path")
            or params.get("reference_audio_asset_id")
            or params.get("reference_audio_url")
        )
        if ref_path:
            payload["reference_audio_path"] = str(ref_path)

        if "audio_cover_strength" in params or "cover_strength" in params:
            payload["audio_cover_strength"] = float(params.get("audio_cover_strength", params.get("cover_strength", 0.8)))

        if "repainting_start" in params:
            payload["repainting_start"] = float(params["repainting_start"])
        if "repainting_end" in params:
            payload["repainting_end"] = float(params["repainting_end"])

        # Specialized task params
        if "stem_name" in params or "target_stem" in params:
            payload["stem_name"] = str(params.get("stem_name") or params.get("target_stem"))
        if "add_instrument" in params:
            payload["add_instrument"] = str(params["add_instrument"])
        if "extend_duration" in params:
            payload["extend_duration"] = float(params["extend_duration"])

        # LoRA params
        if "lora_path" in params:
            payload["lora_path"] = str(params["lora_path"])
            payload["lora_weight"] = float(params.get("lora_weight", 1.0))

        if "batch_size" in params and int(params["batch_size"]) > 1:
            payload["batch_size"] = int(params["batch_size"])

        return payload

    async def generate_music(
        self,
        task: str,
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[int, float, str], Any]] = None,
    ) -> Dict[str, Any]:
        # Fail-closed when backend is unconfigured
        if not self.api_base_url or not self.api_base_url.startswith(("http://", "https://")):
            raise ModelUnavailableError(
                message="ACE-Step backend is unconfigured or has an invalid base URL.",
                details={"api_base_url": self.api_base_url},
            )

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

                    # Download all generated items (supporting batch generation ACE-16)
                    artifacts: List[Dict[str, Any]] = []
                    for idx, item in enumerate(result_items):
                        if not isinstance(item, dict):
                            continue
                        file_ref = item.get("file")
                        if not file_ref:
                            continue
                        audio_url = urljoin(self.api_base_url + "/", str(file_ref).lstrip("/"))
                        audio_response = await client.get(audio_url)
                        if audio_response.status_code >= 400 or not audio_response.content:
                            raise ModelUnavailableError(
                                message=f"ACE-Step generated audio could not be downloaded for item {idx}.",
                                details={"status_code": audio_response.status_code},
                            )

                        parsed = urlparse(str(file_ref))
                        source_path = (parse_qs(parsed.query).get("path") or [f"output_{idx}.mp3"])[0]
                        filename = Path(source_path).name or f"output_{idx}.mp3"
                        mime = audio_response.headers.get("content-type") or mimetypes.guess_type(filename)[0] or "audio/mpeg"
                        content_bytes = audio_response.content

                        item_metas = item.get("metas", {}) if isinstance(item, dict) else {}
                        artifacts.append({
                            "index": idx,
                            "filename": filename,
                            "mime_type": mime.split(";", 1)[0],
                            "audio_bytes": content_bytes,
                            "size_bytes": len(content_bytes),
                            "sha256": hashlib.sha256(content_bytes).hexdigest(),
                            "metadata": item_metas,
                        })

                    if not artifacts:
                        raise ModelUnavailableError(message="ACE-Step completed without an audio file reference.")

                    primary = artifacts[0]
                    first_item = result_items[0]
                    metas = first_item.get("metas", {}) if isinstance(first_item, dict) else {}

                    return {
                        "task_id": task_id,
                        "filename": primary["filename"],
                        "mime_type": primary["mime_type"],
                        "audio_bytes": primary["audio_bytes"],
                        "size_bytes": primary["size_bytes"],
                        "sha256": primary["sha256"],
                        "metadata": metas,
                        "artifacts": artifacts,
                    }
        except (InvalidRequestError, ModelUnavailableError, UnsupportedCapabilityError):
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                message="ACE-Step backend is unavailable or returned an invalid response.",
                details={"error_type": type(exc).__name__},
            ) from exc

    async def understand_audio(self, audio_path: str, prompt: str = "Analyze genre, BPM, key, mood") -> Dict[str, Any]:
        """ACE-19: Audio understanding (metadata extraction and captioning)."""
        if not self.api_base_url:
            raise ModelUnavailableError(message="ACE-Step backend is unconfigured.")
        try:
            async with httpx.AsyncClient(base_url=self.api_base_url, timeout=self.timeout, headers=self._headers()) as client:
                resp = await client.post("/understand", json={"audio_path": str(audio_path), "prompt": prompt})
                if resp.status_code >= 400:
                    raise ModelUnavailableError(message="ACE-Step audio understanding failed.", details={"status_code": resp.status_code})
                return resp.json()
        except (InvalidRequestError, ModelUnavailableError, UnsupportedCapabilityError):
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                message=f"ACE-Step backend request failed: {exc}",
                code="BACKEND_UNAVAILABLE",
                details={"error_type": type(exc).__name__},
            ) from exc

    async def score_audio(self, audio_path: str) -> Dict[str, Any]:
        """ACE-21: Quality scoring of generated audio with provenance."""
        if not self.api_base_url:
            raise ModelUnavailableError(message="ACE-Step backend is unconfigured.")
        try:
            async with httpx.AsyncClient(base_url=self.api_base_url, timeout=self.timeout, headers=self._headers()) as client:
                resp = await client.post("/score", json={"audio_path": str(audio_path)})
                if resp.status_code >= 400:
                    raise ModelUnavailableError(message="ACE-Step audio scoring failed.", details={"status_code": resp.status_code})
                return resp.json()
        except (InvalidRequestError, ModelUnavailableError, UnsupportedCapabilityError):
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                message=f"ACE-Step backend request failed: {exc}",
                code="BACKEND_UNAVAILABLE",
                details={"error_type": type(exc).__name__},
            ) from exc

    async def cancel_task(self, task_id: str) -> bool:
        """ACE-24: Clean cancellation of in-flight generation task."""
        if not self.api_base_url:
            return False
        try:
            async with httpx.AsyncClient(base_url=self.api_base_url, timeout=10.0, headers=self._headers()) as client:
                resp = await client.post("/cancel_task", json={"task_id": task_id})
                return resp.status_code in (200, 204)
        except Exception:
            return False

    async def train_lora(
        self,
        dataset_path: str,
        output_dir: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """ACE-23: LoRA training workflow contract validation and submission."""
        if not dataset_path:
            raise InvalidRequestError("LoRA training requires a non-empty dataset_path.")
        if not output_dir:
            raise InvalidRequestError("LoRA training requires a non-empty output_dir.")
        hp = hyperparameters or {}
        rank = int(hp.get("rank", 16))
        if rank < 4 or rank > 64:
            raise InvalidRequestError(f"LoRA rank must be between 4 and 64, got {rank}")
        lr = float(hp.get("learning_rate", 1e-4))
        if lr <= 0.0 or lr > 1e-2:
            raise InvalidRequestError(f"Learning rate must be positive and <= 0.01, got {lr}")

        if not self.api_base_url:
            raise ModelUnavailableError(message="ACE-Step backend is unconfigured.")

        try:
            async with httpx.AsyncClient(base_url=self.api_base_url, timeout=self.timeout, headers=self._headers()) as client:
                resp = await client.post(
                    "/train_lora",
                    json={
                        "dataset_path": dataset_path,
                        "output_dir": output_dir,
                        "rank": rank,
                        "learning_rate": lr,
                        "epochs": int(hp.get("epochs", 10)),
                    },
                )
                if resp.status_code >= 400:
                    raise ModelUnavailableError(message="LoRA training rejected by backend.", details={"status_code": resp.status_code})
                return resp.json()
        except (InvalidRequestError, ModelUnavailableError, UnsupportedCapabilityError):
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                message=f"ACE-Step backend request failed: {exc}",
                code="BACKEND_UNAVAILABLE",
                details={"error_type": type(exc).__name__},
            ) from exc
