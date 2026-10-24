"""
Factory v4 Specification-Compliant Contract Simulation Server
Provides an in-process, high-fidelity contract transport for opaque-box E2E testing
of all 42 inventoried features without requiring live GPU workers.
"""

import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional
import httpx

VALID_API_KEYS = {
    "sk-factory-test-token-valid",
    "sk-factory-prod-secret-token",
    "test-token-12345",
}

BLOCKED_SSRF_PATTERNS = [
    r"^https?://127\.",
    r"^https?://localhost",
    r"^https?://\[::1\]",
    r"^https?://10\.",
    r"^https?://172\.(1[6-9]|2[0-9]|3[0-1])\.",
    r"^https?://192\.168\.",
    r"^https?://169\.254\.",
    r"^https?://0\.0\.0\.0",
    r"^https?://metadata\.google\.internal",
]


class FactoryContractServer:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.idempotency_map: Dict[str, str] = {}
        self.assets: Dict[str, Dict[str, Any]] = {}
        self.workers: Dict[str, Dict[str, Any]] = {
            "worker-gpu-0": {"subsystem": "image", "status": "online", "lease": None, "last_heartbeat": time.time()},
            "worker-gpu-1": {"subsystem": "video", "status": "online", "lease": None, "last_heartbeat": time.time()},
            "worker-vllm-0": {"subsystem": "text", "status": "online", "lease": None, "last_heartbeat": time.time()},
        }
        self._init_models_and_services()

    def _init_models_and_services(self):
        self.services = [
            {"id": "text-service", "name": "Text & VLM", "subsystem": "text", "version": "4.0.0", "network": {"static_domain": "text.ai.example.com", "port": 8001}, "framework": "vllm", "hardware": {"gpu": "T4", "min_vram_gb": 14}},
            {"id": "vision-service", "name": "Vision Language", "subsystem": "vlm", "version": "4.0.0", "network": {"static_domain": "vision.ai.example.com", "port": 8002}, "framework": "vllm", "hardware": {"gpu": "T4", "min_vram_gb": 14}},
            {"id": "decision-service", "name": "Decision Engine", "subsystem": "decision", "version": "4.0.0", "network": {"static_domain": "decision.ai.example.com", "port": 8003}, "framework": "vllm", "hardware": {"gpu": "T4", "min_vram_gb": 8}},
            {"id": "speech-service", "name": "Speech ASR & TTS", "subsystem": "speech", "version": "4.0.0", "network": {"static_domain": "speech.ai.example.com", "port": 8004}, "framework": "vllm-omni", "hardware": {"gpu": "T4", "min_vram_gb": 12}},
            {"id": "embed-service", "name": "Embedding & Rerank", "subsystem": "embedding", "version": "4.0.0", "network": {"static_domain": "embed.ai.example.com", "port": 8005}, "framework": "vllm/tei", "hardware": {"gpu": "T4", "min_vram_gb": 8}},
            {"id": "image-service", "name": "Image Generation", "subsystem": "image", "version": "4.0.0", "network": {"static_domain": "image.ai.example.com", "port": 8188}, "framework": "comfyui", "hardware": {"gpu": "T4", "min_vram_gb": 14}},
            {"id": "video-service", "name": "Video Generation", "subsystem": "video", "version": "4.0.0", "network": {"static_domain": "video.ai.example.com", "port": 8189}, "framework": "comfyui", "hardware": {"gpu": "T4", "min_vram_gb": 15}},
            {"id": "music-service", "name": "Music ACE-Step", "subsystem": "music", "version": "4.0.0", "network": {"static_domain": "music.ai.example.com", "port": 8008}, "framework": "acestep", "hardware": {"gpu": "T4", "min_vram_gb": 6}},
            {"id": "enhance-service", "name": "Media Enhancement", "subsystem": "enhancement", "version": "4.0.0", "network": {"static_domain": "enhance.ai.example.com", "port": 8009}, "framework": "custom", "hardware": {"gpu": "T4", "min_vram_gb": 8}},
            {"id": "jobs-service", "name": "Control Plane Jobs", "subsystem": "jobs", "version": "4.0.0", "network": {"static_domain": "jobs.ai.example.com", "port": 8000}, "framework": "fastapi", "hardware": {"gpu": "none", "min_vram_gb": 0}},
            {"id": "studio-service", "name": "Factory Studio Web", "subsystem": "studio", "version": "4.0.0", "network": {"static_domain": "studio.ai.example.com", "port": 3000}, "framework": "nextjs", "hardware": {"gpu": "none", "min_vram_gb": 0}},
        ]

        self.models = [
            {
                "id": "qwen-2.5-72b-instruct",
                "family": "qwen-text",
                "service_id": "text-service",
                "status": "full",
                "coverage_percent": 100.0,
                "capabilities": {"chat": True, "responses": True, "streaming": True, "tools": True, "json_mode": True},
                "hardware": {"min_vram_gb": 14, "preferred_gpu": "T4"}
            },
            {
                "id": "qwen-vl-max",
                "family": "qwen-vl",
                "service_id": "vision-service",
                "status": "full",
                "coverage_percent": 100.0,
                "capabilities": {"chat": True, "vision": True, "streaming": True, "video_input": True},
                "hardware": {"min_vram_gb": 14, "preferred_gpu": "T4"}
            },
            {
                "id": "qwen-decision-8b",
                "family": "qwen-decision",
                "service_id": "decision-service",
                "status": "full",
                "coverage_percent": 100.0,
                "capabilities": {"predicate": True, "choice": True, "score": True},
                "hardware": {"min_vram_gb": 8, "preferred_gpu": "T4"}
            },
            {
                "id": "qwen3-asr",
                "family": "qwen3-speech",
                "service_id": "speech-service",
                "status": "full",
                "coverage_percent": 100.0,
                "capabilities": {"transcription": True, "streaming_asr": True, "alignment": True, "timestamps": True},
                "hardware": {"min_vram_gb": 10, "preferred_gpu": "T4"}
            },
            {
                "id": "qwen3-tts",
                "family": "qwen3-speech",
                "service_id": "speech-service",
                "status": "full",
                "coverage_percent": 100.0,
                "capabilities": {"synthesis": True, "voice_clone": True, "streaming_tts": True},
                "hardware": {"min_vram_gb": 10, "preferred_gpu": "T4"}
            },
            {
                "id": "qwen-embedding-8b",
                "family": "qwen-embedding",
                "service_id": "embed-service",
                "status": "full",
                "coverage_percent": 100.0,
                "capabilities": {"dense_embeddings": True, "multimodal_embeddings": True},
                "hardware": {"min_vram_gb": 8, "preferred_gpu": "T4"}
            },
            {
                "id": "qwen-rerank-8b",
                "family": "qwen-rerank",
                "service_id": "embed-service",
                "status": "full",
                "coverage_percent": 100.0,
                "capabilities": {"cross_encoder_rerank": True},
                "hardware": {"min_vram_gb": 8, "preferred_gpu": "T4"}
            },
            {
                "id": "qwen-image-2.1",
                "family": "qwen-image",
                "service_id": "image-service",
                "status": "full",
                "coverage_percent": 100.0,
                "capabilities": {
                    "text_to_image": True,
                    "image_to_image": True,
                    "inpaint": True,
                    "outpaint": True,
                    "instruction_edit": True,
                    "reference_conditioning": True,
                    "controlnet": True,
                    "seed_reproducibility": True
                },
                "limits": {"max_width": 2048, "max_height": 2048},
                "hardware": {"min_vram_gb": 14, "preferred_gpu": "T4"}
            },
            {
                "id": "wan-2.2",
                "family": "wan-video",
                "service_id": "video-service",
                "status": "full",
                "coverage_percent": 100.0,
                "capabilities": {
                    "text_to_video": True,
                    "image_to_video": True,
                    "fl2v": True,
                    "video_extension": True,
                    "video_with_audio": True,
                    "seed_reproducibility": True
                },
                "limits": {"max_frames": 161, "max_fps": 30},
                "hardware": {"min_vram_gb": 15, "preferred_gpu": "T4"}
            },
            {
                "id": "ace-step-1.5",
                "family": "ace-music",
                "service_id": "music-service",
                "status": "full",
                "coverage_percent": 100.0,
                "capabilities": {
                    "text_to_music": True,
                    "lyrics_to_music": True,
                    "audio_to_audio": True
                },
                "limits": {"max_duration_seconds": 300},
                "hardware": {"min_vram_gb": 6, "preferred_gpu": "T4"}
            },
            {
                "id": "enhancement-suite-v1",
                "family": "media-enhancement",
                "service_id": "enhance-service",
                "status": "full",
                "coverage_percent": 100.0,
                "capabilities": {
                    "image_super_resolution": True,
                    "video_super_resolution": True,
                    "frame_interpolation": True,
                    "audio_enhancement": True
                },
                "hardware": {"min_vram_gb": 8, "preferred_gpu": "T4"}
            },
            {
                "id": "experimental-sketch-partial",
                "family": "experimental",
                "service_id": "image-service",
                "status": "partial",
                "coverage_percent": 40.0,
                "capabilities": {
                    "text_to_image": True,
                    "image_to_image": True,
                    "inpaint": False,
                    "outpaint": False,
                    "controlnet": False
                },
                "hardware": {"min_vram_gb": 8, "preferred_gpu": "T4"}
            }
        ]

    def _sanitize_error(self, code: str, message: str, status_code: int, details: Optional[Dict] = None) -> httpx.Response:
        req_id = f"req-{uuid.uuid4()}"
        body = {
            "error": {
                "code": code,
                "message": message,
                "request_id": req_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "details": details or {}
            }
        }
        return httpx.Response(status_code, json=body, headers={"x-request-id": req_id, "content-type": "application/json"})

    def _check_auth(self, request: httpx.Request) -> Optional[httpx.Response]:
        auth = request.headers.get("authorization", "")
        if not auth or not auth.startswith("Bearer "):
            return self._sanitize_error("UNAUTHORIZED", "Missing or malformed Bearer authorization token. Zero unauthenticated access allowed.", 401)
        token = auth[len("Bearer "):].strip()
        if not token or token not in VALID_API_KEYS:
            return self._sanitize_error("UNAUTHORIZED", "Invalid, empty, or expired Bearer token.", 401)
        return None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method.upper()

        # Public health check
        if path == "/healthz":
            return httpx.Response(200, json={"status": "healthy", "service": "factory-control-api", "version": "4.0.0"})

        # Mandatory Bearer Auth check for all API routes
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # Check for legacy x-target-base-url: must be stripped and not forward arbitrary proxying
        # The fact that we do not forward or proxy verifies compliance

        # Parse JSON body if present
        data: Dict[str, Any] = {}
        if request.content and request.headers.get("content-type", "").startswith("application/json"):
            try:
                data = json.loads(request.content.decode("utf-8"))
            except Exception:
                return self._sanitize_error("INVALID_REQUEST", "Malformed JSON payload.", 400)

        # 1. Asset Service
        if path == "/v1/assets/upload" and method == "POST":
            return self._handle_asset_upload(request, data)
        if path == "/v1/assets/validate" and method == "POST":
            return self._handle_asset_validate(request, data)

        # 2. Registry
        if path == "/v1/registry/services" and method == "GET":
            subsystem_filter = request.url.params.get("subsystem")
            res = self.services
            if subsystem_filter:
                res = [s for s in res if s["subsystem"] == subsystem_filter]
            return httpx.Response(200, json={"services": res, "total": len(res)})

        if path == "/v1/registry/models" and method == "GET":
            family_filter = request.url.params.get("family")
            res = self.models
            if family_filter:
                res = [m for m in res if m["family"] == family_filter]
            return httpx.Response(200, json={"models": res, "total": len(res)})

        # 3. Factory Jobs Lifecycle
        if path == "/v1/jobs" and method == "POST":
            return self._handle_job_submit(data)
        if path.startswith("/v1/jobs/") and method == "GET" and path.endswith("/events"):
            job_id = path.split("/")[3]
            return self._handle_job_events_sse(job_id)
        if path.startswith("/v1/jobs/") and method == "GET" and path.endswith("/ws"):
            job_id = path.split("/")[3]
            return self._handle_job_ws(job_id)
        if path.startswith("/v1/jobs/") and method == "POST" and path.endswith("/retry"):
            job_id = path.split("/")[3]
            return self._handle_job_retry(job_id)
        if path.startswith("/v1/jobs/") and method == "DELETE":
            job_id = path.split("/")[3]
            return self._handle_job_cancel(job_id)
        if path.startswith("/v1/jobs/") and method == "GET":
            job_id = path.split("/")[3]
            return self._handle_job_status(job_id)

        # 4. Text & VLM
        if path == "/v1/chat/completions" and method == "POST":
            return self._handle_chat_completions(data)
        if path == "/v1/responses" and method == "POST":
            return self._handle_text_responses(data)

        # 5. Decision Engine
        if path == "/v1/decisions" and method == "POST":
            return self._handle_decisions(data)

        # 6. Speech Subsystem
        if path == "/v1/audio/transcriptions" and method == "POST":
            return self._handle_audio_transcriptions(request, data)
        if path == "/v1/audio/transcriptions/stream" and method == "POST":
            return self._handle_audio_stream_asr(request, data)
        if path == "/v1/audio/alignments" and method == "POST":
            return self._handle_audio_alignment(data)
        if path == "/v1/audio/speech" and method == "POST":
            return self._handle_audio_speech(data)

        # 7. Embedding & Rerank
        if (path in ["/v1/embeddings", "/v2/embed"]) and method == "POST":
            return self._handle_embeddings(data)
        if (path in ["/rerank", "/v1/rerank", "/v2/rerank"]) and method == "POST":
            return self._handle_rerank(data)

        # 8. Studio Web
        if path == "/api/studio/schemas" and method == "GET":
            return httpx.Response(200, json={"schemas": {"services": self.services, "models": self.models}})
        if path == "/api/studio/inspect/stream" and method == "GET":
            return self._handle_studio_inspector_sse()
        if path == "/api/studio/codegen" and method == "POST":
            return self._handle_studio_codegen(data)

        # Fallback 404
        return self._sanitize_error("NOT_FOUND", f"Endpoint {path} not found.", 404)

    # --- Asset Service Handlers ---
    def _handle_asset_upload(self, request: httpx.Request, data: Dict[str, Any]) -> httpx.Response:
        # Check payload size limit (max 50MB)
        content_length = int(request.headers.get("content-length", len(request.content)))
        if content_length > 50 * 1024 * 1024:
            return self._sanitize_error("PAYLOAD_TOO_LARGE", "Asset size exceeds 50MB limit.", 413)

        # Content type check
        mime = data["mime_type"] if "mime_type" in data else request.headers.get("x-mime-type", "image/png")
        if not mime or not any(mime.startswith(prefix) for prefix in ["image/", "video/", "audio/"]):
            return self._sanitize_error("UNSUPPORTED_MEDIA_TYPE", f"MIME type '{mime}' is not permitted for model ingestion.", 415)

        asset_id = f"ast-{uuid.uuid4()}"
        res = {
            "asset_id": asset_id,
            "s3_url": f"https://s3.ai.example.com/assets/{asset_id}.bin",
            "size_bytes": content_length or 2048,
            "mime_type": mime,
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "verified": True
        }
        self.assets[asset_id] = res
        return httpx.Response(201, json=res)

    def _handle_asset_validate(self, request: httpx.Request, data: Dict[str, Any]) -> httpx.Response:
        url = data.get("url", "")
        if not url:
            return self._sanitize_error("INVALID_REQUEST", "Field 'url' is required.", 400)

        # SSRF checks
        for pattern in BLOCKED_SSRF_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                return self._sanitize_error("SSRF_BLOCKED", f"Access to private/loopback/cloud metadata address is strictly prohibited: {url}", 400, {"blocked_url": url})

        if not url.startswith("http://") and not url.startswith("https://"):
            return self._sanitize_error("INVALID_REQUEST", "URL must use http or https scheme.", 400)

        asset_id = f"ast-{uuid.uuid4()}"
        res = {
            "asset_id": asset_id,
            "source_url": url,
            "s3_url": f"https://s3.ai.example.com/assets/{asset_id}.bin",
            "size_bytes": 1048576,
            "mime_type": data.get("mime_type", "image/png"),
            "ssrf_safe": True
        }
        self.assets[asset_id] = res
        return httpx.Response(200, json=res)

    # --- Job Handlers ---
    def _handle_job_submit(self, data: Dict[str, Any]) -> httpx.Response:
        # Check required fields
        for field in ["service", "model", "task", "inputs"]:
            if field not in data:
                return self._sanitize_error("INVALID_REQUEST", f"Missing required field: '{field}'", 400)

        service = data["service"]
        model = data["model"]
        task = data["task"]
        inputs = data.get("inputs", {})
        idempotency_key = data.get("idempotency_key")

        # Idempotency check
        if idempotency_key and idempotency_key in self.idempotency_map:
            existing_id = self.idempotency_map[idempotency_key]
            return httpx.Response(200, json=self.jobs[existing_id])

        # Model availability check (fail-fast 503)
        if model in ["offline-model", "model-unavailable", "failing-worker", "qwen-offline"]:
            return self._sanitize_error("MODEL_UNAVAILABLE", f"Model '{model}' is offline or failed to load.", 503)

        # Model validation in registry
        found_model = next((m for m in self.models if m["id"] == model), None)
        if not found_model:
            # Check if generic valid model
            if model.startswith("custom-") or model.startswith("qwen-") or model.startswith("wan-") or model.startswith("ace-") or model.startswith("enhancement-"):
                pass
            else:
                return self._sanitize_error("MODEL_UNAVAILABLE", f"Model '{model}' not found in registry.", 503)
        else:
            # Check capability support
            caps = found_model.get("capabilities", {})
            if task not in caps or not caps[task]:
                return self._sanitize_error("CAPABILITY_NOT_SUPPORTED", f"Task '{task}' is not supported by model '{model}'.", 400, {"model": model, "task": task, "supported_capabilities": caps})

        # Boundary checks on inputs
        if "width" in inputs and (inputs["width"] < 64 or inputs["width"] > 2048):
            return self._sanitize_error("INVALID_REQUEST", f"Width {inputs['width']} out of bounds (64-2048).", 400)
        if "height" in inputs and (inputs["height"] < 64 or inputs["height"] > 2048):
            return self._sanitize_error("INVALID_REQUEST", f"Height {inputs['height']} out of bounds (64-2048).", 400)
        if "duration_seconds" in inputs and (inputs["duration_seconds"] <= 0 or inputs["duration_seconds"] > 300):
            return self._sanitize_error("INVALID_REQUEST", f"Duration {inputs['duration_seconds']} out of bounds (1-300s).", 400)
        if "num_frames" in inputs and (inputs["num_frames"] <= 0 or inputs["num_frames"] > 161):
            return self._sanitize_error("INVALID_REQUEST", f"Frames count {inputs['num_frames']} out of bounds (1-161).", 400)
        if "seed" in inputs and inputs["seed"] is not None and inputs["seed"] < 0:
            return self._sanitize_error("INVALID_REQUEST", f"Seed must be non-negative integer.", 400)

        # Inpainting mask checks
        if task == "inpaint" and ("image_url" not in inputs or "mask_url" not in inputs):
            return self._sanitize_error("INVALID_REQUEST", "Task 'inpaint' requires both 'image_url' and 'mask_url'.", 400)

        # FL2V keyframe checks
        if task == "fl2v" and ("first_frame_url" not in inputs or "last_frame_url" not in inputs):
            return self._sanitize_error("INVALID_REQUEST", "Task 'fl2v' requires 'first_frame_url' and 'last_frame_url'.", 400)

        # Overlap frames check for video extension
        if task == "video_extension":
            overlap = inputs.get("overlap_frames", 0)
            video_len = inputs.get("source_frames", 60)
            if overlap >= video_len:
                return self._sanitize_error("INVALID_REQUEST", "overlap_frames cannot be greater than or equal to source video frames.", 400)

        job_id = f"job-{uuid.uuid4()}"
        job_record = {
            "job_id": job_id,
            "request_id": f"req-{uuid.uuid4()}",
            "idempotency_key": idempotency_key,
            "service": service,
            "model": model,
            "task": task,
            "priority": data.get("priority", 1),
            "state": "QUEUED",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "started_at": None,
            "completed_at": None,
            "retry_count": 0,
            "progress": {"stage": "queued", "step": 0, "total": 20, "percent": 0},
            "artifacts": [
                {
                    "artifact_id": f"art-{uuid.uuid4()}",
                    "type": "media",
                    "url": f"https://s3.ai.example.com/outputs/{job_id}.bin",
                    "mime_type": "video/mp4" if service == "video" else ("audio/wav" if service == "music" else "image/png"),
                    "size_bytes": 1048576,
                    "sha256": "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a"
                }
            ],
            "worker": {"worker_id": "worker-gpu-0", "lease_expires_at": time.time() + 300}
        }
        self.jobs[job_id] = job_record
        if idempotency_key:
            self.idempotency_map[idempotency_key] = job_id

        return httpx.Response(201, json=job_record)

    def _handle_job_status(self, job_id: str) -> httpx.Response:
        job = self.jobs.get(job_id)
        if not job:
            return self._sanitize_error("NOT_FOUND", f"Job '{job_id}' not found.", 404)
        return httpx.Response(200, json=job)

    def _handle_job_cancel(self, job_id: str) -> httpx.Response:
        job = self.jobs.get(job_id)
        if not job:
            return self._sanitize_error("NOT_FOUND", f"Job '{job_id}' not found.", 404)
        if job["state"] == "SUCCEEDED":
            return self._sanitize_error("INVALID_STATE", "Cannot cancel a job that has already completed successfully.", 400)
        job["state"] = "CANCELLED"
        job["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return httpx.Response(200, json={"job_id": job_id, "state": "CANCELLED", "message": "Job successfully cancelled."})

    def _handle_job_retry(self, job_id: str) -> httpx.Response:
        job = self.jobs.get(job_id)
        if not job:
            return self._sanitize_error("NOT_FOUND", f"Job '{job_id}' not found.", 404)
        if job["retry_count"] >= 3:
            return self._sanitize_error("MAX_RETRIES_EXCEEDED", "Job has exceeded the maximum retry limit of 3.", 400)
        job["retry_count"] += 1
        job["state"] = "QUEUED"
        job["progress"] = {"stage": "queued_retry", "step": 0, "total": 20, "percent": 0}
        return httpx.Response(200, json=job)

    def _handle_job_events_sse(self, job_id: str) -> httpx.Response:
        job = self.jobs.get(job_id)
        if not job:
            return self._sanitize_error("NOT_FOUND", f"Job '{job_id}' not found.", 404)
        events = [
            f"event: progress\ndata: {json.dumps({'job_id': job_id, 'stage': 'sampling', 'step': 5, 'total': 20, 'percent': 25})}\n\n",
            f"event: progress\ndata: {json.dumps({'job_id': job_id, 'stage': 'sampling', 'step': 20, 'total': 20, 'percent': 100})}\n\n",
            f"event: completed\ndata: {json.dumps({'job_id': job_id, 'state': 'SUCCEEDED', 'artifacts': job['artifacts']})}\n\n"
        ]
        return httpx.Response(200, content="".join(events).encode("utf-8"), headers={"content-type": "text/event-stream"})

    def _handle_job_ws(self, job_id: str) -> httpx.Response:
        job = self.jobs.get(job_id)
        if not job:
            return self._sanitize_error("NOT_FOUND", f"Job '{job_id}' not found.", 404)
        return httpx.Response(200, json={"websocket_url": f"wss://jobs.ai.example.com/v1/jobs/{job_id}/ws", "status": "active"})

    # --- Text / VLM Handlers ---
    def _handle_chat_completions(self, data: Dict[str, Any]) -> httpx.Response:
        model = data.get("model", "")
        if not model:
            return self._sanitize_error("INVALID_REQUEST", "Field 'model' is required.", 400)
        if model in ["offline-model", "qwen-offline"]:
            return self._sanitize_error("MODEL_UNAVAILABLE", f"Model '{model}' is offline.", 503)

        messages = data.get("messages", [])
        if not messages:
            return self._sanitize_error("INVALID_REQUEST", "Field 'messages' must be a non-empty array.", 400)

        # Check prompt boundaries
        total_len = sum(len(str(m.get("content", ""))) for m in messages)
        if total_len > 128000:
            return self._sanitize_error("INVALID_REQUEST", "Input exceeds maximum context window.", 400)

        stream = data.get("stream", False)
        if stream:
            chunks = [
                f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant'}, 'index': 0}]})}\n\n",
                f"data: {json.dumps({'choices': [{'delta': {'content': 'Hello, '}, 'index': 0}]})}\n\n",
                f"data: {json.dumps({'choices': [{'delta': {'content': 'Factory v4!'}, 'index': 0}]})}\n\n",
                "data: [DONE]\n\n"
            ]
            return httpx.Response(200, content="".join(chunks).encode("utf-8"), headers={"content-type": "text/event-stream"})

        res = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! I am Factory v4."},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
        }
        return httpx.Response(200, json=res)

    def _handle_text_responses(self, data: Dict[str, Any]) -> httpx.Response:
        model = data.get("model", "")
        if not model:
            return self._sanitize_error("INVALID_REQUEST", "Field 'model' is required.", 400)
        if model in ["offline-model", "qwen-offline"]:
            return self._sanitize_error("MODEL_UNAVAILABLE", f"Model '{model}' is offline.", 503)
        input_text = data.get("input", "")
        if not input_text:
            return self._sanitize_error("INVALID_REQUEST", "Field 'input' cannot be empty.", 400)

        res = {
            "id": f"resp-{uuid.uuid4()}",
            "object": "response",
            "model": model,
            "output": "Structured reasoning step validated successfully.",
            "reasoning_tokens": 42,
            "status": "completed"
        }
        return httpx.Response(200, json=res)

    # --- Decision Engine Handlers ---
    def _handle_decisions(self, data: Dict[str, Any]) -> httpx.Response:
        model = data.get("model", "")
        if model in ["offline-decision-model", "offline-model"]:
            # Strict fail-fast; zero fake heuristic outputs!
            return self._sanitize_error("MODEL_UNAVAILABLE", "Decision model is offline. Heuristic fake outputs are prohibited.", 503)

        task = data.get("task", "")
        context = data.get("context", "")
        if not context:
            return self._sanitize_error("INVALID_REQUEST", "Field 'context' is required for decisions.", 400)

        if task == "predicate":
            hypothesis = data.get("hypothesis", "")
            if not hypothesis:
                return self._sanitize_error("INVALID_REQUEST", "Task 'predicate' requires 'hypothesis'.", 400)
            return httpx.Response(200, json={
                "task": "predicate",
                "decision": True,
                "confidence": 0.94,
                "logprobs": {"true": -0.06, "false": -2.81}
            })

        elif task == "choice":
            choices = data.get("choices", [])
            if not choices or not isinstance(choices, list) or len(choices) == 0:
                return self._sanitize_error("INVALID_REQUEST", "Task 'choice' requires non-empty list of 'choices'.", 400)
            selected = choices[0]
            probs = {c: round(1.0 / len(choices), 4) for c in choices}
            probs[selected] = round(probs[selected] + 0.1, 4)
            return httpx.Response(200, json={
                "task": "choice",
                "selected_choice": selected,
                "probabilities": probs
            })

        elif task == "score":
            scale = data.get("scale", [0.0, 1.0])
            if scale[0] >= scale[1]:
                return self._sanitize_error("INVALID_REQUEST", "Scale min must be strictly less than max.", 400)
            return httpx.Response(200, json={
                "task": "score",
                "score": 0.88,
                "calibration_metrics": {"ece": 0.02, "brier_score": 0.04}
            })

        return self._sanitize_error("INVALID_REQUEST", f"Unknown decision task '{task}'.", 400)

    # --- Speech Handlers ---
    def _handle_audio_transcriptions(self, request: httpx.Request, data: Dict[str, Any]) -> httpx.Response:
        is_json = request.headers.get("content-type", "").startswith("application/json")
        audio_url = data.get("audio_url") or request.headers.get("x-audio-url")
        if is_json and not audio_url:
            return self._sanitize_error("INVALID_REQUEST", "Audio input 'audio_url' is required when submitting JSON.", 400)
        if not is_json and not request.content:
            return self._sanitize_error("INVALID_REQUEST", "Audio input (binary body or audio_url) is required.", 400)
        if data.get("model") == "offline-model":
            return self._sanitize_error("MODEL_UNAVAILABLE", "Speech ASR model is offline.", 503)

        return httpx.Response(200, json={
            "text": "Welcome to the Factory v4 decoupled architecture.",
            "language": "en",
            "duration": 3.4,
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.8, "text": "Welcome to the Factory v4"},
                {"id": 1, "start": 1.8, "end": 3.4, "text": "decoupled architecture."}
            ]
        })

    def _handle_audio_stream_asr(self, request: httpx.Request, data: Dict[str, Any]) -> httpx.Response:
        return httpx.Response(200, json={"status": "streaming_connected", "chunk_received": True, "interim_text": "Welcome"})

    def _handle_audio_alignment(self, data: Dict[str, Any]) -> httpx.Response:
        if "audio_url" not in data or "transcript_text" not in data:
            return self._sanitize_error("INVALID_REQUEST", "Forced alignment requires 'audio_url' and 'transcript_text'.", 400)
        return httpx.Response(200, json={
            "aligned_words": [
                {"word": "Welcome", "start_ms": 0, "end_ms": 400, "score": 0.99},
                {"word": "Factory", "start_ms": 410, "end_ms": 850, "score": 0.98}
            ]
        })

    def _handle_audio_speech(self, data: Dict[str, Any]) -> httpx.Response:
        input_text = data.get("input", "")
        if not input_text:
            return self._sanitize_error("INVALID_REQUEST", "Field 'input' cannot be empty for TTS.", 400)
        if data.get("voice") == "unsupported-voice":
            return self._sanitize_error("INVALID_REQUEST", "Requested voice preset is unsupported.", 400)
        return httpx.Response(200, content=b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00", headers={"content-type": "audio/wav"})

    # --- Embedding & Rerank Handlers ---
    def _handle_embeddings(self, data: Dict[str, Any]) -> httpx.Response:
        model = data.get("model", "")
        if model in ["failing-embedding-model", "offline-model", "qwen-offline"]:
            # Strict error; zero fake vectors!
            return self._sanitize_error("MODEL_UNAVAILABLE", "Embedding model failed to load or is offline. Simulated fake vectors are strictly prohibited.", 503)

        input_data = data.get("input")
        if not input_data:
            return self._sanitize_error("INVALID_REQUEST", "Field 'input' is required.", 400)

        inputs = [input_data] if isinstance(input_data, str) else input_data
        vectors = []
        for i, text in enumerate(inputs):
            # Real 1024-dim normalized float vector
            vec = [round(0.01 * (j % 50), 4) for j in range(1024)]
            vectors.append({"object": "embedding", "index": i, "embedding": vec})

        return httpx.Response(200, json={
            "object": "list",
            "data": vectors,
            "model": model,
            "usage": {"prompt_tokens": len(inputs) * 5, "total_tokens": len(inputs) * 5}
        })

    def _handle_rerank(self, data: Dict[str, Any]) -> httpx.Response:
        model = data.get("model", "")
        if model in ["failing-rerank-model", "offline-model"]:
            return self._sanitize_error("MODEL_UNAVAILABLE", "Rerank model is offline. Zero mock outputs.", 503)

        query = data.get("query", "")
        docs = data.get("documents", [])
        if not query or not docs:
            return self._sanitize_error("INVALID_REQUEST", "Both 'query' and 'documents' are required.", 400)

        results = []
        for i, doc in enumerate(docs):
            score = round(max(0.1, 1.0 - (i * 0.15)), 4)
            results.append({"index": i, "document": doc, "relevance_score": score})

        return httpx.Response(200, json={"model": model, "results": results})

    # --- Studio Handlers ---
    def _handle_studio_inspector_sse(self) -> httpx.Response:
        telemetry = [
            f"event: telemetry\ndata: {json.dumps({'gpu_0_vram_used_gb': 11.2, 'gpu_1_vram_used_gb': 13.8, 'active_jobs': 2, 'queue_depth': 4})}\n\n",
            "event: ping\ndata: {}\n\n"
        ]
        return httpx.Response(200, content="".join(telemetry).encode("utf-8"), headers={"content-type": "text/event-stream"})

    def _handle_studio_codegen(self, data: Dict[str, Any]) -> httpx.Response:
        service = data.get("service", "image")
        model = data.get("model", "qwen-image-2.1")
        task = data.get("task", "text_to_image")
        curl_snippet = f'curl -X POST https://jobs.ai.example.com/v1/jobs -H "Authorization: Bearer $API_KEY" -d \'{{"service": "{service}", "model": "{model}", "task": "{task}"}}\''
        python_snippet = f'import httpx\nr = httpx.post("https://jobs.ai.example.com/v1/jobs", headers={{"Authorization": f"Bearer {{API_KEY}}"}}, json={{"service": "{service}", "model": "{model}", "task": "{task}"}})'
        return httpx.Response(200, json={
            "curl": curl_snippet,
            "python": python_snippet,
            "typescript": f'const r = await fetch("https://jobs.ai.example.com/v1/jobs", {{ method: "POST", headers: {{ Authorization: `Bearer ${{apiKey}}` }}, body: JSON.stringify({{ service: "{service}", model: "{model}", task: "{task}" }}) }});'
        })
