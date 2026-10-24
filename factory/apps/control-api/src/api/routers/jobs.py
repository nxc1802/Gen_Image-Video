"""
factory/apps/control-api/src/api/routers/jobs.py
FastAPI router for Client Job Lifecycle API endpoints (/v1/jobs).
"""
import asyncio
import secrets
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, WebSocket, status
from fastapi.responses import JSONResponse, StreamingResponse

from factory_auth import verify_bearer_token, load_auth_config, decode_jwt_token
from factory_contracts.jobs.job_contract import FactoryJob, JobSubmissionRequest, JobState
from factory_contracts.errors import InvalidRequestError, NotFoundError
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
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


@router.post("", status_code=status.HTTP_201_CREATED, dependencies=[Depends(verify_bearer_token)])
async def submit_job(
    req: JobSubmissionRequest,
    request: Request,
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
):
    request_id = getattr(request.state, "request_id", None) or request.headers.get("x-request-id") or "req-unknown"
    
    # 1. Enforce strict 1MB threshold against inline Base64
    enforce_payload_thresholds(req.inputs, request_id=request_id)

    # 2. Submit job through Lifecycle Manager
    job, created = await manager.submit_job(req, request_id=request_id)

    headers = {}
    if should_attach_deprecation_header(req.inputs):
        headers.update(DEPRECATION_HEADERS)

    status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return JSONResponse(status_code=status_code, content=job.model_dump(mode="json"), headers=headers)


@router.get("/{job_id}", dependencies=[Depends(verify_bearer_token)])
async def get_job_status(
    job_id: str,
    request: Request,
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
):
    job = await manager.get_job(job_id)
    storage = get_storage_provider(request)

    # Refresh presigned download URLs for artifacts on the fly
    if job.artifacts and storage:
        for art in job.artifacts:
            if art.storage_path:
                try:
                    fresh_url = storage.generate_presigned_download_url(
                        bucket="",
                        key=art.storage_path,
                        expires_in_seconds=86400,
                    )
                    art.download_url = fresh_url
                    art.url = fresh_url
                except Exception:
                    pass

    return job.model_dump(mode="json")


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
    raw_token = websocket.query_params.get("token") or websocket.headers.get("authorization")

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

    is_authenticated = False
    # 1. API Key constant-time comparison
    if auth_config.api_key and secrets.compare_digest(token, auth_config.api_key):
        is_authenticated = True
    # 2. JWT verification
    elif auth_config.jwt_secret and token.count(".") == 2:
        try:
            decode_jwt_token(token, auth_config.jwt_secret)
            is_authenticated = True
        except Exception:
            is_authenticated = False

    if not is_authenticated:
        await websocket.close(code=1008)
        return

    async def _on_cancel(jid: str):
        await manager.cancel_job(jid)

    await ws_hub.handle_stream(
        websocket=websocket,
        job_id=job_id,
        stream_manager=manager.stream_manager,
        on_cancel=_on_cancel,
    )

