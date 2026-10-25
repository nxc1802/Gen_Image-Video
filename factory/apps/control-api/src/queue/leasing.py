"""
factory/apps/control-api/src/queue/leasing.py
Worker leasing and heartbeat coordination with capability matching.
"""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from factory_contracts.jobs.job_contract import JobState
from factory_contracts.jobs.worker_contract import (
    WorkerHeartbeatRequest,
    WorkerHeartbeatResponse,
    WorkerLeaseRequest,
    WorkerLeaseResponse,
)
from factory.apps.control_api.src.queue.broker.base import BaseQueueBroker
from factory.apps.control_api.src.queue.repository.base import BaseJobRepository
from factory.apps.control_api.src.telemetry.metrics import metrics


class LeaseCoordinator:
    """Coordinates capability-based worker leasing and liveness heartbeats."""

    def __init__(
        self,
        repository: BaseJobRepository,
        broker: Optional[BaseQueueBroker] = None,
        candidate_limit: int = 50,
    ):
        self.repository = repository
        self.broker = broker
        self.candidate_limit = candidate_limit

    async def lease_job(self, req: WorkerLeaseRequest) -> WorkerLeaseResponse:
        """
        Matches next available QUEUED job matching worker subsystem, models, and tasks.
        Acquires atomic lease with TTL and evicts job from active broker queue.
        """
        candidates = await self.repository.find_eligible_jobs_for_worker(
            subsystem=req.subsystem,
            supported_models=req.supported_models,
            supported_tasks=req.supported_tasks,
            limit=self.candidate_limit,
        )

        if not candidates:
            return WorkerLeaseResponse(job=None, lease_token=None, lease_expires_at=None)

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=req.lease_duration_seconds)
        lease_token = f"lease-{uuid.uuid4()}"

        for job in candidates:
            acquired = await self.repository.acquire_lease(
                job_id=job.job_id,
                worker_id=req.worker_id,
                lease_token=lease_token,
                expires_at=expires_at,
                gpu_device=req.gpu_device,
            )
            if acquired:
                # Evict leased job from broker active queue
                if self.broker:
                    await self.broker.remove(job.job_id, job.service)

                updated_job = await self.repository.get_job(job.job_id)
                # Register/update worker state
                await self.repository.upsert_worker_heartbeat(
                    worker_id=req.worker_id,
                    subsystem=req.subsystem,
                    supported_models=req.supported_models,
                    supported_tasks=req.supported_tasks,
                    gpu_device=req.gpu_device,
                    active_job_id=job.job_id,
                )
                return WorkerLeaseResponse(
                    job=updated_job,
                    lease_token=lease_token,
                    lease_expires_at=expires_at,
                )

        return WorkerLeaseResponse(job=None, lease_token=None, lease_expires_at=None)

    async def process_heartbeat(self, req: WorkerHeartbeatRequest) -> WorkerHeartbeatResponse:
        """
        Processes worker heartbeat, checking if current job was cancelled,
        extending the lease if valid, and updating worker telemetry.
        """
        cancel_requested = False
        extended_until: Optional[datetime] = None

        if req.active_job_id:
            job = await self.repository.get_job(req.active_job_id)
            if job:
                if job.state == JobState.CANCELLED:
                    cancel_requested = True
                elif job.state in (JobState.RUNNING, JobState.STREAMING):
                    if (
                        job.worker is not None
                        and job.worker.worker_id == req.worker_id
                        and req.lease_token
                        and job.worker.lease_token == req.lease_token
                    ):
                        now = datetime.now(timezone.utc)
                        extended_until = now + timedelta(seconds=300)
                        await self.repository.extend_lease(
                            worker_id=req.worker_id,
                            job_id=req.active_job_id,
                            new_expires_at=extended_until,
                        )

        # Update real telemetry VRAM metrics
        if req.vram_used_mb is not None:
            vram_gb = round(req.vram_used_mb / 1024.0, 1)
            worker = await self.repository.get_worker(req.worker_id)
            gpu_device = worker.get("gpu_device") if worker else None
            if gpu_device == 1 or "gpu-1" in req.worker_id or "gpu_1" in req.worker_id or "gpu1" in req.worker_id:
                metrics.set_gpu_vram(gpu_1_gb=vram_gb)
            else:
                metrics.set_gpu_vram(gpu_0_gb=vram_gb)

        await self.repository.upsert_worker_heartbeat(
            worker_id=req.worker_id,
            active_job_id=req.active_job_id,
            gpu_utilization=req.gpu_utilization_percent,
            vram_used_mb=req.vram_used_mb,
        )

        return WorkerHeartbeatResponse(
            status="acknowledged",
            cancel_requested=cancel_requested,
            lease_extended_until=extended_until,
        )
