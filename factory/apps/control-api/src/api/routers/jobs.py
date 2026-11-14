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
from factory_contracts.tasks import normalize_model
from factory_contracts.capabilities import canonical_model_capabilities, model_capability_truth
from model_registry.deployment import DeploymentRegistry
from factory_contracts.errors import ModelNotLoadedError
from factory_contracts.errors import (
    ArtifactNotFoundError,
    FactoryError,
    InvalidRequestError,
    ModelUnavailableError,
    NotFoundError,
    UnsupportedCapabilityError,
)

from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.storage.scratch_disk import ScratchDiskManager, get_scratch_disk
from factory.apps.control_api.src.storage.keys import sanitize_id
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

# Read-only import compatibility. The catalog owns the data; request resolution
# below uses the catalog selected by the current app's loader.
from factory_contracts.tasks import CANONICAL_MODEL_MAP as MODEL_ALIASES

def _validate_job_capability(req: JobSubmissionRequest, request: Request) -> None:
    if any(key in req.inputs for key in {"gpu", "gpu_id", "gpu_ids", "gpu_device", "devices", "device", "backend"}):
        raise InvalidRequestError("Backend and GPU assignment belong to deployment configuration, not inference requests.")
    loader = getattr(request.app.state, "manifest_loader", None)
    if loader is None:
        raise FactoryError(
            message="Model registry is unavailable.",
            code="SERVICE_UNAVAILABLE",
            status_code=503,
        )
    models = loader.load_models()
    # Identity resolution uses this app's catalog, including custom paths.
    identities = {identity: key for key, item in models.items() for identity in [key, *item.aliases]}
    resolved_model = identities.get(str(req.model).strip().lower(), str(req.model).strip().lower())
    model = models.get(resolved_model)
    if model is not None:
        req.model = resolved_model
    if model is None:
        raise InvalidRequestError(
            message=f"Unknown model '{req.model}'.",
            details={"model": req.model},
        )
    auth_config = getattr(request.app.state, "auth_config", None)
    if model.apiVersion != "factory.ai/v2" and getattr(auth_config, "environment", None) == "production":
        raise ModelNotLoadedError(message="Legacy catalog readiness cannot authorize production inference.")


    # FIX-07: Hardware feasibility and variant validation
    min_vram = getattr(model.hardware, "min_vram_gb", 0.0) if hasattr(model, "hardware") else 0.0
    variant_name = str(req.inputs.get("variant", "")).lower().strip() if req.inputs else ""
    if model.apiVersion == "factory.ai/v2" and variant_name not in ("", "standard", "default"):
        raise UnsupportedCapabilityError("Checkpoint variants require an explicit model ID.")
    if model.apiVersion == "factory.ai/v2":
        registry = getattr(request.app.state, "deployment_registry", None)
        if not isinstance(registry, DeploymentRegistry):
            raise ModelNotLoadedError(message=f"Model '{req.model}' has no active deployment in this session.")
        registry.require_active(req.model, req.task)
    if variant_name and hasattr(model, "weights") and hasattr(model.weights, "variants"):
        variants = model.weights.variants
        if variants and variant_name not in variants and variant_name not in ("standard", "base", "default"):
            raise InvalidRequestError(
                message=f"Variant '{variant_name}' is not supported by model '{req.model}'.",
                details={"model": req.model, "variant": variant_name, "supported_variants": list(variants.keys())},
            )
        if variant_name in variants and variants[variant_name].min_vram_gb:
            min_vram = max(min_vram, variants[variant_name].min_vram_gb)

    if model.apiVersion != "factory.ai/v2" and min_vram > 14.2:
        raise InvalidRequestError("Requested model exceeds max allocatable VRAM of 14.2 GB")

    # FIX-07: Backend service readiness check
    services = loader.load_services()
    service_manifest = services.get(model.metadata.service_id)
    if service_manifest is None:
        raise FactoryError(
            message=f"Backend service '{model.metadata.service_id}' for model '{req.model}' is not registered or unavailable.",
            code="SERVICE_UNAVAILABLE",
            status_code=503,
            details={"service_id": model.metadata.service_id, "model": req.model},
        )

    # C2: Fail closed if model backend is marked offline or not installed
    if model.apiVersion != "factory.ai/v2" and (not getattr(model, "backend_ready", True) or not getattr(model, "installed", True)):
        raise FactoryError(
            message=f"Backend for model '{req.model}' is offline or not installed.",
            code="BACKEND_UNAVAILABLE",
            status_code=503,
            details={
                "model": req.model,
                "service_id": model.metadata.service_id,
                "backend_ready": getattr(model, "backend_ready", False),
                "installed": getattr(model, "installed", False),
            },
        )

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

    task = req.task
    canonical_task = task
    capability = task
    support = canonical_model_capabilities(model)

    supported = support.get(capability)
    if supported is not True:
        raise FactoryError(
            message=f"Task '{task}' (canonical: '{canonical_task}') is not supported by model '{req.model}'.",
            code="CAPABILITY_NOT_SUPPORTED",
            status_code=400,
            details={
                "model": req.model,
                "task": task,
                "canonical_task": canonical_task,
                "capability": capability,
                "validation_status": model.validation_status.value,
                "error_code": "UNSUPPORTED_CAPABILITY",
            },
        )

    # Capability Truth: Fail closed if capability is declared but not implemented in truth model
    if capability in support:
        cap_truth = model_capability_truth(model, capability)
        if not cap_truth.implemented:
            raise UnsupportedCapabilityError(
                message=f"Capability '{capability}' is declared but not implemented for model '{req.model}'.",
                details={
                    "model": req.model,
                    "task": task,
                    "canonical_task": canonical_task,
                    "capability": capability,
                    "error_code": "UNSUPPORTED_CAPABILITY",
                },
            )

    # Phase 1 Variant-Safe Contracts: Task-Variant Matrix Validation (§5.2 & C2)
    effective_variant = variant_name or "standard"
    variant_task = capability
    if model.apiVersion != "factory.ai/v2":
        from factory_contracts.capabilities import CAPABILITIES
        definition = CAPABILITIES.get(capability)
        if definition:
            variant_task = definition.legacy_key or capability.split(".", 1)[1]
    if hasattr(model, "task_variant_matrix") and model.task_variant_matrix:
        for t in (variant_task,):
            if t and t in model.task_variant_matrix:
                if not model.is_variant_supported_for_task(t, effective_variant):
                    raise UnsupportedCapabilityError(
                        message=f"Variant '{effective_variant}' is not supported for task '{t}' by model '{req.model}'.",
                        details={
                            "model": req.model,
                            "variant": effective_variant,
                            "task": t,
                            "error_code": "UNSUPPORTED_CAPABILITY",
                        },
                    )
    elif variant_name:
        for t in (variant_task,):
            if t and not model.is_variant_supported_for_task(t, variant_name):
                raise UnsupportedCapabilityError(
                    message=f"Variant '{variant_name}' is not supported for task '{t}' by model '{req.model}'.",
                    details={
                        "model": req.model,
                        "variant": variant_name,
                        "task": t,
                        "error_code": "UNSUPPORTED_CAPABILITY",
                    },
                )

    # Validate steps for turbo variant (§5.2)
    if variant_name == "turbo":
        steps_val = req.inputs.get("steps") if req.inputs else None
        if steps_val is not None:
            if not isinstance(steps_val, int) or steps_val < 1 or steps_val > 25:
                raise InvalidRequestError(
                    message=f"Invalid steps '{steps_val}' for turbo variant (must be an integer between 1 and 25).",
                    details={"model": req.model, "variant": variant_name, "steps": steps_val, "error_code": "INVALID_INPUT"},
                )

    # Enforce model dimension limits if declared

    if hasattr(model, "limits") and model.limits and req.inputs:
        max_w = getattr(model.limits, "max_width", None)
        max_h = getattr(model.limits, "max_height", None)
        if max_w and "width" in req.inputs:
            try:
                w_val = int(req.inputs["width"])
                if w_val > max_w:
                    raise InvalidRequestError(
                        message=f"Requested width {w_val} exceeds maximum allowed width {max_w} for model '{req.model}'.",
                        details={"width": w_val, "max_width": max_w, "model": req.model, "error_code": "INVALID_INPUT"},
                    )
            except (ValueError, TypeError):
                pass
        if max_h and "height" in req.inputs:
            try:
                h_val = int(req.inputs["height"])
                if h_val > max_h:
                    raise InvalidRequestError(
                        message=f"Requested height {h_val} exceeds maximum allowed height {max_h} for model '{req.model}'.",
                        details={"height": h_val, "max_height": max_h, "model": req.model, "error_code": "INVALID_INPUT"},
                    )
            except (ValueError, TypeError):
                pass




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
        if err_code in ("GATEWAY_TIMEOUT", "TIMEOUT", "INFERENCE_TIMEOUT"):
            raise FactoryError(
                message=err_msg,
                code="GATEWAY_TIMEOUT",
                status_code=504,
                details=err_details,
            )
        if err_code in ("ARTIFACT_NOT_FOUND", "ARTIFACT_MISSING"):
            msg = err_msg if ("Artifact file not found" in err_msg or "not found" in err_msg.lower()) else f"Artifact file not found: {err_msg}"
            raise FactoryError(
                message=msg,
                code="ARTIFACT_NOT_FOUND",
                status_code=404,
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

    # Fallback to job.artifacts storage_path if not located by direct job_id directory
    if not artifact_path and job.artifacts:
        root_resolved = scratch_disk.root_dir.resolve()
        for art in job.artifacts:
            if art.storage_path:
                try:
                    clean_sp = art.storage_path.lstrip("/")
                    candidates = [
                        (scratch_disk.root_dir / clean_sp).resolve(),
                        (scratch_disk.root_dir / "factory-artifacts" / clean_sp).resolve(),
                    ]
                    for cand in candidates:
                        if cand.is_file() and not cand.is_symlink() and cand.is_relative_to(root_resolved):
                            artifact_path = cand
                            break
                    if artifact_path:
                        break
                except Exception:
                    pass

    # Verify symlink and scratch disk root containment before delivery
    if artifact_path:
        if artifact_path.is_symlink():
            artifact_path = None
        else:
            try:
                resolved_art = artifact_path.resolve()
                root_resolved = scratch_disk.root_dir.resolve()
                clean_job = sanitize_id(job_id, default="unknown")
                if not resolved_art.is_relative_to(root_resolved) or clean_job not in str(resolved_art):
                    artifact_path = None
            except Exception:
                artifact_path = None

    # If valid artifact file exists on scratch disk, deliver it
    if artifact_path and artifact_path.exists() and artifact_path.is_file():
        ext = artifact_path.suffix.lower()

        # Image delivery
        if ext in (".png", ".jpg", ".jpeg", ".webp") or service == "image":
            mime_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
            if scratch_disk.config.cleanup_on_download:
                background_tasks.add_task(scratch_disk.cleanup_job_artifact, job_id, artifact_path)
            return FileResponse(
                path=str(artifact_path),
                media_type=mime_type,
                headers={"Content-Disposition": f'inline; filename="{job.job_id}{ext if ext else ".png"}"'},
                background=background_tasks if scratch_disk.config.cleanup_on_download else None,
            )

        # Audio delivery
        if ext in (".wav", ".mp3", ".ogg", ".flac") or service in ("speech", "music"):
            mime_type = "audio/mpeg" if ext == ".mp3" else "audio/wav"
            if scratch_disk.config.cleanup_on_download:
                background_tasks.add_task(scratch_disk.cleanup_job_artifact, job_id, artifact_path)
            return FileResponse(
                path=str(artifact_path),
                media_type=mime_type,
                headers={"Content-Disposition": f'inline; filename="{job.job_id}{ext if ext else ".wav"}"'},
                background=background_tasks if scratch_disk.config.cleanup_on_download else None,
            )

        # Video delivery
        if ext in (".mp4", ".webm", ".mkv") or service in ("video", "enhancement"):
            if scratch_disk.config.cleanup_on_download:
                background_tasks.add_task(scratch_disk.cleanup_job_artifact, job_id, artifact_path)
            return FileResponse(
                path=str(artifact_path),
                media_type="video/mp4",
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Disposition": f'inline; filename="{job.job_id}{ext if ext else ".mp4"}"',
                },
                background=background_tasks if scratch_disk.config.cleanup_on_download else None,
            )

        # Generic media delivery
        if scratch_disk.config.cleanup_on_download:
            background_tasks.add_task(scratch_disk.cleanup_job_artifact, job_id, artifact_path)
        return FileResponse(
            path=str(artifact_path),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{job.job_id}{ext}"'},
            background=background_tasks if scratch_disk.config.cleanup_on_download else None,
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
        reasoning_content = None
        usage_info = {}
        if hasattr(job, "result") and job.result is not None:
            if isinstance(job.result, dict):
                text_content = job.result.get("text") or job.result.get("content") or ""
                reasoning_content = job.result.get("reasoning_content")
                usage_info = job.result.get("usage", {})
                if not reasoning_content and ("<think>" in text_content or "</think>" in text_content):
                    from services.text.src.engine import decouple_think_tokens
                    r_c, clean_c = decouple_think_tokens(text_content)
                    if r_c:
                        reasoning_content = r_c
                        text_content = clean_c
            elif isinstance(job.result, str):
                text_content = job.result
                if "<think>" in text_content or "</think>" in text_content:
                    from services.text.src.engine import decouple_think_tokens
                    r_c, clean_c = decouple_think_tokens(text_content)
                    if r_c:
                        reasoning_content = r_c
                        text_content = clean_c

        if "text/plain" in accept_header and "application/json" not in accept_header:
            return Response(content=text_content, media_type="text/plain")

        resp_content = {
            "job_id": job.job_id,
            "text": text_content,
            "usage": usage_info,
        }
        if reasoning_content is not None:
            resp_content["reasoning_content"] = reasoning_content

        return JSONResponse(content=resp_content)

    # Case C: Media files missing on scratch disk for SUCCEEDED job
    raise ArtifactNotFoundError("Artifact file not found on scratch disk.")


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

