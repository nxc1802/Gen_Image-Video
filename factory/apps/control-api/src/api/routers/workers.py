"""
factory/apps/control-api/src/api/routers/workers.py
FastAPI router for Subsystem Worker Control API (/v1/jobs/lease, heartbeat, progress, complete, fail).
Binds JWT claims (sub / worker_id) to worker operations with strict ownership enforcement.
"""
from fastapi import APIRouter, Depends, HTTPException, Request

from factory_auth import verify_bearer_token
from factory_contracts.errors import ForbiddenError
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    JobFailureRequest,
    ProgressUpdateRequest,
    WorkerHeartbeatRequest,
    WorkerHeartbeatResponse,
    WorkerLeaseRequest,
    WorkerLeaseResponse,
)
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.manager import JobLifecycleManager

router = APIRouter(prefix="/v1", tags=["Workers"])


async def verify_worker_identity(user: dict = Depends(verify_bearer_token)) -> dict:
    role = str(user.get("role", "")).lower()
    if role not in {"worker", "admin"}:
        raise ForbiddenError("Worker control endpoints require role 'worker' or 'admin'.")
    if role == "worker":
        worker_id = user.get("sub") or user.get("worker_id")
        if not worker_id:
            raise ForbiddenError("Worker authentication token must contain 'sub' or 'worker_id' claim.")
    return user


def get_lease_coordinator(request: Request) -> LeaseCoordinator:
    coord = getattr(request.app.state, "lease_coordinator", None)
    if coord is None:
        raise HTTPException(status_code=500, detail="LeaseCoordinator not initialized on app state.")
    return coord


def get_lifecycle_manager(request: Request) -> JobLifecycleManager:
    manager = getattr(request.app.state, "job_manager", None)
    if manager is None:
        raise HTTPException(status_code=500, detail="JobLifecycleManager not initialized on app state.")
    return manager


@router.post("/jobs/lease", response_model=WorkerLeaseResponse)
async def lease_job(
    req: WorkerLeaseRequest,
    coordinator: LeaseCoordinator = Depends(get_lease_coordinator),
    user: dict = Depends(verify_worker_identity),
):
    """Worker daemon requests the next matching job from the queue."""
    role = str(user.get("role", "")).lower()
    auth_worker = user.get("sub") or user.get("worker_id")
    if role != "admin" and auth_worker:
        if req.worker_id and req.worker_id != auth_worker:
            raise ForbiddenError(f"Caller '{auth_worker}' is not authorized to lease jobs as worker '{req.worker_id}'.")
        if not req.worker_id:
            req.worker_id = auth_worker

    resp = await coordinator.lease_job(req)
    return resp


@router.post("/workers/{worker_id}/heartbeat", response_model=WorkerHeartbeatResponse)
async def worker_heartbeat(
    worker_id: str,
    req: WorkerHeartbeatRequest,
    coordinator: LeaseCoordinator = Depends(get_lease_coordinator),
    user: dict = Depends(verify_worker_identity),
):
    """Periodic worker heartbeat extending lease and checking for cancellation."""
    role = str(user.get("role", "")).lower()
    auth_worker = user.get("sub") or user.get("worker_id")
    if role != "admin" and auth_worker:
        if auth_worker != worker_id:
            raise ForbiddenError(f"Caller '{auth_worker}' is not authorized to act as worker '{worker_id}'.")
        if req.worker_id and req.worker_id != auth_worker:
            raise ForbiddenError(f"Caller '{auth_worker}' cannot spoof worker_id '{req.worker_id}'.")

    req.worker_id = worker_id
    resp = await coordinator.process_heartbeat(req)
    return resp


@router.post("/jobs/{job_id}/progress")
async def update_job_progress(
    job_id: str,
    req: ProgressUpdateRequest,
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
    user: dict = Depends(verify_worker_identity),
):
    """Worker reports incremental progress."""
    role = str(user.get("role", "")).lower()
    auth_worker = user.get("sub") or user.get("worker_id")
    if role != "admin" and auth_worker:
        if req.worker_id and req.worker_id != auth_worker:
            raise ForbiddenError(f"Worker '{auth_worker}' cannot claim worker identity '{req.worker_id}'.")
        if not req.worker_id:
            req.worker_id = auth_worker

    job = await manager.update_progress(
        job_id,
        req,
        caller_worker_id=auth_worker,
        caller_role=role,
    )
    return {"status": "acknowledged", "job_id": job.job_id}


@router.post("/jobs/{job_id}/complete")
async def complete_job(
    job_id: str,
    req: JobCompletionRequest,
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
    user: dict = Depends(verify_worker_identity),
):
    """Worker reports successful generation and artifact paths."""
    role = str(user.get("role", "")).lower()
    auth_worker = user.get("sub") or user.get("worker_id")
    if role != "admin" and auth_worker:
        if req.worker_id and req.worker_id != auth_worker:
            raise ForbiddenError(f"Worker '{auth_worker}' cannot claim worker identity '{req.worker_id}'.")
        if not req.worker_id:
            req.worker_id = auth_worker

    job = await manager.complete_job(
        job_id,
        req,
        caller_worker_id=auth_worker,
        caller_role=role,
    )
    return {"status": "completed", "job_id": job.job_id}


@router.post("/jobs/{job_id}/fail")
async def fail_job(
    job_id: str,
    req: JobFailureRequest,
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
    user: dict = Depends(verify_worker_identity),
):
    """Worker reports unrecoverable generation failure."""
    role = str(user.get("role", "")).lower()
    auth_worker = user.get("sub") or user.get("worker_id")
    if role != "admin" and auth_worker:
        if req.worker_id and req.worker_id != auth_worker:
            raise ForbiddenError(f"Worker '{auth_worker}' cannot claim worker identity '{req.worker_id}'.")
        if not req.worker_id:
            req.worker_id = auth_worker

    job = await manager.fail_job(
        job_id,
        req,
        caller_worker_id=auth_worker,
        caller_role=role,
    )
    return {"status": "failed", "job_id": job.job_id}
