"""
factory/apps/control-api/src/api/routers/workers.py
FastAPI router for Subsystem Worker Control API (/v1/jobs/lease, heartbeat, progress, complete, fail).
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError

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


@router.post("/jobs/lease", response_model=WorkerLeaseResponse, dependencies=[Depends(verify_worker_identity)])
async def lease_job(
    req: WorkerLeaseRequest,
    coordinator: LeaseCoordinator = Depends(get_lease_coordinator),
):
    """Worker daemon requests the next matching job from the queue."""
    resp = await coordinator.lease_job(req)
    return resp


@router.post("/workers/{worker_id}/heartbeat", response_model=WorkerHeartbeatResponse, dependencies=[Depends(verify_worker_identity)])
async def worker_heartbeat(
    worker_id: str,
    req: WorkerHeartbeatRequest,
    coordinator: LeaseCoordinator = Depends(get_lease_coordinator),
):
    """Periodic worker heartbeat extending lease and checking for cancellation."""
    if req.worker_id != worker_id:
        req.worker_id = worker_id
    resp = await coordinator.process_heartbeat(req)
    return resp


@router.post("/jobs/{job_id}/progress", dependencies=[Depends(verify_worker_identity)])
async def update_job_progress(
    job_id: str,
    req: ProgressUpdateRequest,
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
):
    """Worker reports incremental progress."""
    job = await manager.update_progress(job_id, req)
    return {"status": "acknowledged", "job_id": job.job_id}


@router.post("/jobs/{job_id}/complete", dependencies=[Depends(verify_worker_identity)])
async def complete_job(
    job_id: str,
    req: JobCompletionRequest,
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
):
    """Worker reports successful generation and artifact paths."""
    job = await manager.complete_job(job_id, req)
    return {"status": "completed", "job_id": job.job_id}


@router.post("/jobs/{job_id}/fail", dependencies=[Depends(verify_worker_identity)])
async def fail_job(
    job_id: str,
    req: JobFailureRequest,
    manager: JobLifecycleManager = Depends(get_lifecycle_manager),
):
    """Worker reports unrecoverable generation failure."""
    job = await manager.fail_job(job_id, req)
    return {"status": "failed", "job_id": job.job_id}
