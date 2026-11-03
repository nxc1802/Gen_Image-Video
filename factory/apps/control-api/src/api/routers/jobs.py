"""
factory/apps/control-api/src/api/routers/jobs.py
FastAPI router for Client Job Lifecycle API endpoints (/v1/jobs).
"""
import asyncio
import secrets
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query, Request, Response, WebSocket, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from factory_auth import verify_bearer_token, load_auth_config
from factory_contracts.jobs.job_contract import FactoryJob, JobSubmissionRequest, JobState
from factory_contracts.errors import FactoryError, InvalidRequestError, NotFoundError
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.storage.scratch_disk import ScratchDiskManager, get_scratch_disk
from factory.apps.control_api.src.storage.threshold import (
    DEPRECATION_HEADERS,
    enforce_payload_thresholds,
    should_attach_deprecation_header,
)
from factory.apps.control_api.src.storage.router import get_storage_provider
from factory.apps.control_api.src.streaming.sse import (
    stream_job_events_generator,
    stream_tokens_generator,
)
from factory.apps.control_api.src.streaming.audio import AudioChunkStreamer
from factory.apps.control_api.src.streaming.ws import ws_hub

router = APIRouter(prefix="/v1/jobs", tags=["Jobs"])


def get_lifecycle_manager(request: Request) -> JobLifecycleManager:
    manager = getattr(request.app.state, "job_manager", None)
    if manager is None:
        raise HTTPException(status_code=500, detail="JobLifecycleManager not initialized on app state.")
    return manager


SERVICE_IDS = {
    "text": "text-service",
    "vlm": "vision-service",
    "vision": "vision-service",
    "decision": "decision-service",
    "embedding": "embed-service",
    "embed": "embed-service",
    "speech": "speech-service",
    "image": "image-service",
    "video": "video-service",
    "music": "music-service",
    "enhancement": "enhancement-service",
}

TASK_CAPABILITY_ALIASES = {
    "predicate": "predicate_evaluation",
    "choice": "choice_routing",
    "score": "score_ranking",
    "embeddings": "text_embedding",
    "embedding": "text_embedding",
    "vector_embeddings": "text_embedding",
    "rerank": "reranking",
    "transcription": "speech_to_text",
    "speech_to_text": "speech_to_text",
    "speech_transcription": "speech_to_text",
    "tts": "text_to_speech",
    "text_to_speech": "text_to_speech",
    "speech_synthesis": "text_to_speech",
    "visual_language": "image_chat",
    "vlm": "image_chat",
    "reference_conditioning": "reference_image",
    "reference": "reference_image",
    "fl2v": "first_last_frame_to_video",
    "extension": "video_extension",
    "chat": "chat_completion",
}

MODEL_ALIASES = {
    "qwen3-tts": "qwen-speech",
    "qwen3-asr": "qwen-speech",
    "qwen-2.5-72b": "qwen-text",
    "qwen-2.5-72b-instruct": "qwen-text",
    "qwen-text-2.5": "qwen-text",
    "qwen-vl-max": "qwen-vlm",
    "qwen-2.5-vl": "qwen-vlm",
    "qwen-embed": "qwen-embedding",
    "qwen-embed-text": "qwen-embedding",
    "qwen-rerank": "qwen-embedding",
    "qwen-decision-8b": "qwen-decision",
}


def _validate_job_capability(req: JobSubmissionRequest, request: Request) -> None:
    loader = getattr(request.app.state, "manifest_loader", None)
    if loader is None:
        raise FactoryError(
            message="Model registry is unavailable.",
            code="SERVICE_UNAVAILABLE",
            status_code=503,
        )
    models = loader.load_models()
    resolved_model = MODEL_ALIASES.get(req.model, req.model)
    model = models.get(resolved_model)
    if model is not None:
        req.model = resolved_model
    if model is None:
        raise InvalidRequestError(
            message=f"Unknown model '{req.model}'.",
            details={"model": req.model},
        )

    min_vram = getattr(model.hardware, "min_vram_gb", 0.0) if hasattr(model, "hardware") else 0.0
    if min_vram > 14.2:
        raise InvalidRequestError("Requested model exceeds max allocatable VRAM of 14.2 GB")

    expected_service_id = SERVICE_IDS.get(req.service)
    if expected_service_id is None or model.metadata.service_id != expected_service_id:
        raise FactoryError(
            message=f"Model '{req.model}' is not registered for service '{req.service}'.",
            code="CAPABILITY_NOT_SUPPORTED",
            status_code=400,
            details={
                "model": req.model,
                "requested_service": req.service,
                "registered_service_id": model.metadata.service_id,
            },
        )

    task = str(req.task or "").strip()
    capability = TASK_CAPABILITY_ALIASES.get(task, task)

    if req.service == "image":
        if task in {"reference_conditioning", "reference"}:
            refs = req.inputs.get("reference_image_urls")
            capability = "multi_reference" if isinstance(refs, list) and len(refs) > 1 else "reference_image"
        elif task == "controlnet":
            control_type = str(req.inputs.get("control_type", "pose")).lower()
            capability = {
                "pose": "control_pose",
                "depth": "control_depth",
            }.get(control_type, f"control_{control_type}")

    if req.service == "video":
        capability = {
            "t2v": "text_to_video",
            "i2v": "image_to_video",
            "fl2v": "first_last_frame_to_video",
            "extension": "video_extension",
            "with_audio": "video_with_audio",
            "ref2va": "reference_conditioning",
        }.get(task, capability)

    supported = model.capabilities.get(capability)
    if supported is not True:
        raise FactoryError(
            message=f"Task '{task}' is not supported by model '{req.model}'.",
            code="CAPABILITY_NOT_SUPPORTED",
            status_code=400,
            details={
                "model": req.model,
                "task": task,
                "capability": capability,
                "validation_status": model.validation_status.value,
            },
        )


@router.post("", status_code=status.HTTP_201_CREATED, dependencies=[Depends(verify_bearer_token)])
async def submit_job(
    req: JobSubmissionRequest,
    request: Request,
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
):
    request_id = getattr(request.state, "request_id", None) or request.headers.get("x-request-id") or "req-unknown"
    trace_id = getattr(request.state, "trace_id", None) or request.headers.get("x-trace-id")
    if not trace_id:
        tp = request.headers.get("traceparent")
        if tp:
            from factory.apps.control_api.src.telemetry.propagation import extract_traceparent
            trace_id, _ = extract_traceparent({"traceparent": tp})
    if not trace_id:
        from factory.apps.control_api.src.telemetry.context import ctx_trace_id
        trace_id = ctx_trace_id.get()

    # 1. Enforce model/service/task compatibility before consuming queue capacity.
    _validate_job_capability(req, request)

    # 2. Enforce strict 1MB threshold against inline Base64
    enforce_payload_thresholds(req.inputs, request_id=request_id)

    # 3. Submit job through Lifecycle Manager
    job, created = await manager.submit_job(req, request_id=request_id, trace_id=trace_id)

    headers = {}
    if should_attach_deprecation_header(req.inputs):
        headers.update(DEPRECATION_HEADERS)

    status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return JSONResponse(status_code=status_code, content=job.model_dump(mode="json"), headers=headers)


@router.get("", dependencies=[Depends(verify_bearer_token)])
async def list_jobs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    service: Optional[str] = Query(default=None),
    state: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
):
    """Lists jobs with optional filtering by service/state and pagination."""
    effective_state = state or status
    jobs = await manager.list_jobs(
        limit=limit,
        offset=offset,
        service=service,
        state=effective_state,
    )
    for job in jobs:
        if job.artifacts:
            for art in job.artifacts:
                art.download_url = f"/v1/jobs/{job.job_id}/result"
                art.url = art.download_url

    return [job.model_dump(mode="json") for job in jobs]


@router.get("/{job_id}", dependencies=[Depends(verify_bearer_token)])
async def get_job_status(
    job_id: str,
    request: Request,
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
):
    job = await manager.get_job(job_id)

    # Set direct download URLs for artifacts
    if job.artifacts:
        for art in job.artifacts:
            art.download_url = f"/v1/jobs/{job.job_id}/result"
            art.url = art.download_url

    return job.model_dump(mode="json")


@router.get("/{job_id}/result", dependencies=[Depends(verify_bearer_token)])
async def get_job_result(
    job_id: str,
    background_tasks: BackgroundTasks,
    request: Request,
    stream: bool = Query(default=False, description="Stream text tokens via SSE"),
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
    scratch_disk: ScratchDiskManager = Depends(get_scratch_disk),
):
    """
    Returns the real inference result based on subsystem modality.
    Attaches a synchronous cleanup hook via FastAPI BackgroundTasks
    to delete the artifact file immediately after transmission completes.
    """
    job = await manager.get_job(job_id)
    if not job:
        raise NotFoundError(f"Job '{job_id}' not found.")

    # 1. State Guard: In-progress states return 409 Conflict BEFORE attempting artifact delivery or background unlinking
    if job.state in (JobState.CREATED, JobState.QUEUED, JobState.ADMITTED, JobState.RUNNING, JobState.STREAMING):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "job_id": job.job_id,
                "state": job.state.value,
                "progress": job.progress.model_dump(mode="json"),
            },
        )

    # 2. Cancelled state
    if job.state == JobState.CANCELLED:
        raise FactoryError(
            message=f"Job '{job_id}' was cancelled.",
            code="JOB_CANCELLED",
            status_code=409,
            details={"job_id": job.job_id, "state": "CANCELLED"},
        )

    # 3. Failed state
    if job.state == JobState.FAILED:
        err_code = "JOB_FAILED"
        err_msg = "Job execution failed."
        err_details = {"job_id": job.job_id}
        if job.error:
            err_code = job.error.code.value if hasattr(job.error.code, "value") else str(job.error.code)
            err_msg = job.error.message or err_msg
            err_details = job.error.details or err_details

        if err_code == "WORKER_DISCONNECTED":
            raise FactoryError(
                message=err_msg,
                code="WORKER_DISCONNECTED",
                status_code=502,
                details=err_details,
            )
        if err_code in ("GATEWAY_TIMEOUT", "TIMEOUT"):
            raise FactoryError(
                message=err_msg,
                code="GATEWAY_TIMEOUT",
                status_code=504,
                details=err_details,
            )
        status_code = 400
        if err_code in ("MODEL_UNAVAILABLE", "SERVICE_UNAVAILABLE"):
            status_code = 503
        raise FactoryError(
            message=err_msg,
            code=err_code,
            status_code=status_code,
            details=err_details,
        )

    # 4. SUCCEEDED State: Discover Artifact File on Scratch Disk with Symlink Containment
    service = (job.service or "").lower()
    artifact_path = scratch_disk.get_artifact(job_id)
    if not artifact_path:
        try:
            job_dir = scratch_disk.get_job_dir(job_id)
            if job_dir.is_dir():
                resolved_job_dir = job_dir.resolve()
                files = []
                for f in sorted(job_dir.iterdir()):
                    if f.is_symlink():
                        continue
                    try:
                        resolved_f = f.resolve()
                        if not resolved_f.is_relative_to(resolved_job_dir):
                            continue
                    except (ValueError, OSError):
                        continue
                    if f.is_file() and not f.name.startswith("."):
                        files.append(f)
                if files:
                    artifact_path = files[0]
        except Exception:
            pass

    # Verify symlink containment before delivery
    if artifact_path:
        if artifact_path.is_symlink():
            artifact_path = None
        else:
            try:
                job_dir = scratch_disk.get_job_dir(job_id)
                if not artifact_path.resolve().is_relative_to(job_dir.resolve()):
                    artifact_path = None
            except Exception:
                artifact_path = None

    # If valid artifact file exists on scratch disk, deliver it
    if artifact_path and artifact_path.exists() and artifact_path.is_file():
        ext = artifact_path.suffix.lower()

        # Image delivery
        if ext in (".png", ".jpg", ".jpeg", ".webp") or service == "image":
            mime_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
            background_tasks.add_task(scratch_disk.cleanup_job_artifact, job_id, artifact_path)
            return FileResponse(
                path=str(artifact_path),
                media_type=mime_type,
                headers={"Content-Disposition": f'inline; filename="{job.job_id}{ext if ext else ".png"}"'},
                background=background_tasks,
            )

        # Audio delivery
        if ext in (".wav", ".mp3", ".ogg", ".flac") or service in ("speech", "music"):
            mime_type = "audio/mpeg" if ext == ".mp3" else "audio/wav"
            background_tasks.add_task(scratch_disk.cleanup_job_artifact, job_id, artifact_path)
            return FileResponse(
                path=str(artifact_path),
                media_type=mime_type,
                headers={"Content-Disposition": f'inline; filename="{job.job_id}{ext if ext else ".wav"}"'},
                background=background_tasks,
            )

        # Video delivery
        if ext in (".mp4", ".webm", ".mkv") or service in ("video", "enhancement"):
            background_tasks.add_task(scratch_disk.cleanup_job_artifact, job_id, artifact_path)
            return FileResponse(
                path=str(artifact_path),
                media_type="video/mp4",
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Disposition": f'attachment; filename="{job.job_id}{ext if ext else ".mp4"}"',
                },
                background=background_tasks,
            )

        # Generic media delivery
        background_tasks.add_task(scratch_disk.cleanup_job_artifact, job_id, artifact_path)
        return FileResponse(
            path=str(artifact_path),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{job.job_id}{ext}"'},
            background=background_tasks,
        )

    # 5. SUCCEEDED State Modality Handlers (No Scratch Artifact)
    # Case A: Decision / Embedding
    if service in ("decision", "embedding", "embed"):
        if hasattr(job, "result") and job.result is not None:
            if isinstance(job.result, dict):
                result_payload = {"job_id": job.job_id, **job.result}
            elif isinstance(job.result, list):
                if service in ("embedding", "embed"):
                    result_payload = {"job_id": job.job_id, "embedding": job.result}
                else:
                    result_payload = {"job_id": job.job_id, "result": job.result}
            else:
                result_payload = {"job_id": job.job_id, "result": job.result}
            return JSONResponse(content=result_payload)
        else:
            raise NotFoundError(f"Inference result payload not found for job '{job_id}'.")

    # Case B: Text / VLM
    if service in ("text", "vlm", "vision"):
        accept_header = request.headers.get("accept", "").lower()
        if stream or "text/event-stream" in accept_header:
            return StreamingResponse(
                stream_tokens_generator(job_id=job.job_id, stream_manager=manager.stream_manager),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        text_content = ""
        usage_info = {}
        if hasattr(job, "result") and job.result is not None:
            if isinstance(job.result, dict):
                text_content = job.result.get("text") or job.result.get("content") or ""
                usage_info = job.result.get("usage", {})
            elif isinstance(job.result, str):
                text_content = job.result

        if "text/plain" in accept_header and "application/json" not in accept_header:
            return Response(content=text_content, media_type="text/plain")

        return JSONResponse(
            content={
                "job_id": job.job_id,
                "text": text_content,
                "usage": usage_info,
            }
        )

    # Case C: Media files missing on scratch disk for SUCCEEDED job
    raise NotFoundError("Artifact file not found on scratch disk.")


@router.delete("/{job_id}", dependencies=[Depends(verify_bearer_token)])
async def cancel_job_delete(
    job_id: str,
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
):
    job = await manager.cancel_job(job_id)
    return {
        "job_id": job.job_id,
        "state": "CANCELLED",
        "message": "Job successfully cancelled.",
    }


@router.post("/{job_id}/cancel", dependencies=[Depends(verify_bearer_token)])
async def cancel_job_post(
    job_id: str,
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
):
    job = await manager.cancel_job(job_id)
    return {
        "job_id": job.job_id,
        "state": "CANCELLED",
        "message": "Job successfully cancelled.",
    }


@router.post("/{job_id}/retry", dependencies=[Depends(verify_bearer_token)])
async def retry_job(
    job_id: str,
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
):
    job = await manager.retry_job(job_id)
    return job.model_dump(mode="json")


@router.get("/{job_id}/events", dependencies=[Depends(verify_bearer_token)])
async def get_job_events(
    job_id: str,
    request: Request,
    last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
):
    # Verify job existence
    job = await manager.get_job(job_id)
    stream_manager = manager.stream_manager

    initial_event = None
    if job.state in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED):
        if job.state == JobState.SUCCEEDED:
            initial_event = {
                "id": "1",
                "event": "completed",
                "data": {"job_id": job_id, "state": "SUCCEEDED", "artifacts": [a.model_dump(mode="json") for a in job.artifacts]},
            }
        elif job.state == JobState.FAILED:
            initial_event = {
                "id": "1",
                "event": "error",
                "data": {"job_id": job_id, "state": "FAILED", "error": job.error.model_dump(mode="json") if job.error else {}},
            }
        else:
            initial_event = {
                "id": "1",
                "event": "cancelled",
                "data": {"job_id": job_id, "state": "CANCELLED"},
            }

    return StreamingResponse(
        stream_job_events_generator(
            job_id=job_id,
            stream_manager=stream_manager,
            last_event_id=last_event_id,
            initial_event=initial_event,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{job_id}/stream/progress", dependencies=[Depends(verify_bearer_token)])
async def stream_job_progress(
    job_id: str,
    last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
):
    await manager.get_job(job_id)
    return StreamingResponse(
        stream_job_events_generator(
            job_id=job_id,
            stream_manager=manager.stream_manager,
            last_event_id=last_event_id,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/{job_id}/stream/tokens", dependencies=[Depends(verify_bearer_token)])
async def stream_job_tokens(
    job_id: str,
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
):
    await manager.get_job(job_id)
    return StreamingResponse(
        stream_tokens_generator(job_id=job_id, stream_manager=manager.stream_manager),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/{job_id}/stream/audio", dependencies=[Depends(verify_bearer_token)])
async def stream_job_audio(
    job_id: str,
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
):
    await manager.get_job(job_id)
    return StreamingResponse(
        AudioChunkStreamer.stream_chunks(job_id=job_id, stream_manager=manager.stream_manager),
        media_type="audio/wav",
        headers={"Transfer-Encoding": "chunked", "Cache-Control": "no-cache"},
    )


@router.get("/{job_id}/ws", dependencies=[Depends(verify_bearer_token)])
async def get_job_websocket_info(
    job_id: str,
    request: Request,
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
):
    await manager.get_job(job_id)
    host = request.headers.get("host", "jobs.ai.example.com")
    proto = "wss" if request.url.scheme == "https" else "ws"
    ws_url = f"{proto}://{host}/v1/jobs/{job_id}/ws"
    return {
        "websocket_url": ws_url,
        "status": "active",
    }


@router.websocket("/{job_id}/ws")
async def websocket_job_stream(
    websocket: WebSocket,
    job_id: str,
):
    manager: JobLifecycleManager = getattr(websocket.app.state, "job_manager", None)
    if not manager:
        await websocket.close(code=1011)
        return

    # --- Authenticate WebSocket Connection ---
    auth_config = getattr(websocket.app.state, "auth_config", None) or load_auth_config()
    raw_token = websocket.headers.get("authorization")
    if not raw_token:
        protocols = [
            item.strip()
            for item in websocket.headers.get("sec-websocket-protocol", "").split(",")
            if item.strip()
        ]
        bearer_protocol = next(
            (item for item in protocols if item.startswith("factory-bearer.")),
            None,
        )
        if bearer_protocol:
            raw_token = bearer_protocol[len("factory-bearer."):]

    if websocket.query_params.get("token") is not None:
        # Query-param token authentication is strictly prohibited (RFC 6455 / OWASP policy violation)
        await websocket.close(code=1008)
        return

    if not raw_token:
        # Close with 1008 Policy Violation (Unauthorized)
        await websocket.close(code=1008)
        return

    token = raw_token.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()

    if not token:
        await websocket.close(code=1008)
        return

    # Single API Key constant-time comparison
    if not (auth_config.api_key and secrets.compare_digest(token, auth_config.api_key)):
        await websocket.close(code=1008)
        return

    async def _on_cancel(jid: str):
        await manager.cancel_job(jid)

    # The browser authenticates in Sec-WebSocket-Protocol so secrets never appear in URLs.
    await ws_hub.handle_stream(
        websocket=websocket,
        job_id=job_id,
        stream_manager=manager.stream_manager,
        on_cancel=_on_cancel,
    )

