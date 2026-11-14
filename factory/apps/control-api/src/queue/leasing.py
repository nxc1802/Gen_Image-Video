"""
factory/apps/control-api/src/queue/leasing.py
Capability-based worker dispatch and liveness heartbeats for Factory v4 Lite.
Eliminates cryptographic lease tokens, distributed TTL lease renewals, and lease ownership.
"""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from factory_contracts.errors import UnauthorizedError
from factory_contracts.capabilities import canonical_model_capabilities, engine_context
from factory_contracts.jobs.job_contract import JobState
from factory_contracts.jobs.worker_contract import (
    WorkerHeartbeatRequest,
    WorkerHeartbeatResponse,
    WorkerLeaseRequest,
    WorkerLeaseResponse,
)
from factory_contracts.tasks import (
    get_model_aliases,
    normalize_model,
)
from factory.apps.control_api.src.queue.broker.base import BaseQueueBroker
from factory.apps.control_api.src.queue.repository.base import BaseJobRepository
from factory.apps.control_api.src.telemetry.metrics import metrics
from model_registry.deployment import DeploymentRegistry
from factory.apps.control_api.src.queue.policy import binding_for_job

class LeaseCoordinator:
    """Coordinates capability-based worker job dispatch and liveness heartbeats."""

    def __init__(
        self,
        repository: BaseJobRepository,
        broker: Optional[BaseQueueBroker] = None,
        candidate_limit: int = 50,
        manifest_loader: Optional[object] = None,
        deployment_registry: Optional[DeploymentRegistry] = None,
    ):
        self.repository = repository
        self.broker = broker
        self.candidate_limit = candidate_limit
        self.manifest_loader = manifest_loader
        self.deployment_registry = deployment_registry if deployment_registry is not None else DeploymentRegistry.from_environment(getattr(manifest_loader, "manifests_dir", None))

    def _verify_manifest_capabilities(
        self,
        subsystem: str,
        requested_models: Optional[List[str]],
        requested_tasks: Optional[List[str]],
    ) -> Tuple[List[str], List[str]]:
        """Validates and filters requested capabilities against authoritative subsystem manifests."""
        norm_sub = str(subsystem).strip().lower() if subsystem else ""
        from model_registry.loader import ManifestLoader
        loader = self.manifest_loader or ManifestLoader()
        models = loader.load_models()
        services = loader.load_services()
        configured = {binding.model: binding for binding in self.deployment_registry.bindings()}
        allowed_models = []
        for model_id, model in models.items():
            if model_id not in configured:
                continue
            service = services.get(model.metadata.service_id)
            service_subsystem = service.metadata.subsystem.value if service else None
            if service_subsystem == norm_sub:
                allowed_models.append(model_id)
            elif norm_sub == "rerank" and canonical_model_capabilities(model).get("vector.reranking", False):
                allowed_models.append(model_id)


        if requested_models:
            identities = {identity: mid for mid, model in models.items() for identity in [mid, *model.aliases]}
            valid_models = [identities[m.strip().lower()] for m in requested_models
                            if identities.get(m.strip().lower()) in allowed_models]
        else:
            valid_models = allowed_models

        allowed_tasks = sorted({cid for mid in valid_models
                                for cid, supported in canonical_model_capabilities(models[mid]).items()
                                if supported and cid.startswith(engine_context(norm_sub) + ".")})

        if requested_tasks:
            valid_tasks = [
                t for t in requested_tasks
                if t in allowed_tasks
            ]
        else:
            valid_tasks = allowed_tasks

        return valid_models, valid_tasks

    async def lease_job(self, req: WorkerLeaseRequest) -> WorkerLeaseResponse:
        """
        Matches next available QUEUED job matching worker subsystem, models, and tasks.
        Uses verified manifest capabilities rather than self-declared parameters.
        Directly assigns job to worker without cryptographic lease tokens or distributed TTL.
        """
        # 1. Fetch registered worker record from repository using worker_id
        registered_worker = await self.repository.get_worker(req.worker_id)

        # 2. If not yet registered, validate worker against declarative manifest capabilities
        if not registered_worker:
            verified_models, verified_tasks = self._verify_manifest_capabilities(
                subsystem=req.subsystem,
                requested_models=req.supported_models,
                requested_tasks=req.supported_tasks,
            )
            if not verified_models or not verified_tasks:
                return WorkerLeaseResponse(job=None, lease_token=None, lease_expires_at=None)

            await self.repository.upsert_worker_heartbeat(
                worker_id=req.worker_id,
                subsystem=req.subsystem,
                supported_models=verified_models,
                supported_tasks=verified_tasks,
                gpu_device=req.gpu_device,
            )
            registered_worker = await self.repository.get_worker(req.worker_id)

        if not registered_worker:
            return WorkerLeaseResponse(job=None, lease_token=None, lease_expires_at=None)

        # 3. Use authoritative registered capabilities
        reg_models = registered_worker.get("supported_models", [])
        reg_tasks = registered_worker.get("supported_tasks", [])

        if req.supported_models:
            effective_models = [
                m for m in req.supported_models
                if normalize_model(m) in [normalize_model(rm) for rm in reg_models]
            ]
        else:
            effective_models = reg_models

        if req.supported_tasks:
            effective_tasks = [
                t for t in req.supported_tasks
                if t in reg_tasks
            ]
        else:
            effective_tasks = reg_tasks

        if not effective_models or not effective_tasks:
            return WorkerLeaseResponse(job=None, lease_token=None, lease_expires_at=None)

        expanded_models = set(effective_models)
        for m in effective_models:
            norm_m = normalize_model(m)
            expanded_models.add(norm_m)
            expanded_models.update(get_model_aliases(norm_m))

        canonical_tasks = set(effective_tasks)

        # The worker declares a device; deployment configuration owns placement.
        candidates = await self.repository.find_eligible_jobs_for_worker(
            subsystem=registered_worker.get("subsystem", req.subsystem),
            supported_models=list(expanded_models),
            supported_tasks=list(canonical_tasks),
            limit=self.candidate_limit,
        )

        if not candidates:
            return WorkerLeaseResponse(job=None, lease_token=None, lease_expires_at=None)

        now = datetime.now(timezone.utc)
        duration = getattr(req, "lease_duration_seconds", 300) or 300
        expires_at = now + timedelta(seconds=duration)

        for job in candidates:
            binding = binding_for_job(self.deployment_registry, job)
            if req.gpu_device is not None and req.gpu_device != binding.devices[0]:
                continue
            target_devices = list(binding.devices)
            gpu_device_for_lease = binding.devices[0]
            if self.broker and hasattr(self.broker, "is_gpu_busy"):
                if any(self.broker.is_gpu_busy(device) for device in target_devices):
                    continue
            if self.broker and hasattr(self.broker, "acquire_gpus"):
                if not self.broker.acquire_gpus(target_devices, job.job_id):
                    continue
            elif len(target_devices) > 1:
                continue  # Never half-reserve a multi-GPU deployment.
            elif self.broker and hasattr(self.broker, "acquire_gpu"):
                if not self.broker.acquire_gpu(gpu_device_for_lease, job.job_id):
                    continue

            lease_token = getattr(req, "lease_token", None) or f"lease-{uuid.uuid4()}"
            try:
                acquired = await self.repository.acquire_lease(
                    job_id=job.job_id,
                    worker_id=req.worker_id,
                    lease_token=lease_token,
                    expires_at=expires_at,
                    gpu_device=gpu_device_for_lease,
                )
                if acquired:
                    if self.broker:
                        await self.broker.remove(job.job_id, job.service)

                    updated_job = await self.repository.get_job(job.job_id)
                    await self.repository.upsert_worker_heartbeat(
                        worker_id=req.worker_id,
                        subsystem=registered_worker.get("subsystem", req.subsystem),
                        supported_models=effective_models,
                        supported_tasks=effective_tasks,
                        gpu_device=gpu_device_for_lease,
                        active_job_id=job.job_id,
                    )
                    return WorkerLeaseResponse(
                        job=updated_job,
                        lease_token=lease_token,
                        lease_expires_at=expires_at,
                    )
                else:
                    if self.broker:
                        if hasattr(self.broker, "release_gpus"):
                            self.broker.release_gpus(target_devices, job.job_id)
                        elif hasattr(self.broker, "release_gpu"):
                            self.broker.release_gpu(gpu_device_for_lease, job.job_id)
            except Exception:
                if self.broker:
                    if hasattr(self.broker, "release_gpus"):
                        self.broker.release_gpus(target_devices, job.job_id)
                    elif hasattr(self.broker, "release_gpu"):
                        self.broker.release_gpu(gpu_device_for_lease, job.job_id)
                raise

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
                    if job.worker is None or job.worker.worker_id != req.worker_id:
                        raise UnauthorizedError(
                            message=f"Worker '{req.worker_id}' is not the active lessee of job '{req.active_job_id}'.",
                            status_code=403,
                            details={"job_id": req.active_job_id, "worker_id": req.worker_id},
                        )

                    if job.worker.lease_token and req.lease_token != job.worker.lease_token:
                        raise UnauthorizedError(
                            message=f"Invalid lease token provided for active job '{req.active_job_id}'.",
                            status_code=403,
                            details={"job_id": req.active_job_id, "worker_id": req.worker_id},
                        )

                    now = datetime.now(timezone.utc)
                    expires_at = getattr(job.worker, "lease_expires_at", None)
                    if expires_at and expires_at.tzinfo is None:
                        expires_at = expires_at.replace(tzinfo=timezone.utc)
                    if expires_at and now > expires_at:
                        raise UnauthorizedError(
                            message=f"Lease for job '{req.active_job_id}' has expired.",
                            status_code=403,
                            details={"job_id": req.active_job_id, "worker_id": req.worker_id},
                        )

                    extended_until = now + timedelta(seconds=300)
                    await self.repository.extend_lease(
                        worker_id=req.worker_id,
                        job_id=req.active_job_id,
                        new_expires_at=extended_until,
                    )

        # Update telemetry VRAM metrics
        used_mb = req.system_vram_used_mb if req.system_vram_used_mb is not None else req.vram_used_mb
        worker = await self.repository.get_worker(req.worker_id)
        gpu_device = worker.get("gpu_device") if worker else None
        is_gpu_1 = gpu_device == 1 or "gpu-1" in req.worker_id or "gpu_1" in req.worker_id or "gpu1" in req.worker_id

        if used_mb is not None:
            vram_gb = round(used_mb / 1024.0, 1)
            if is_gpu_1:
                metrics.set_gpu_vram(gpu_1_gb=vram_gb)
            else:
                metrics.set_gpu_vram(gpu_0_gb=vram_gb)

        if req.vram_total_mb is not None:
            total_gb = round(req.vram_total_mb / 1024.0, 1)
            if is_gpu_1:
                metrics.set_gpu_totals(gpu_1_total_gb=total_gb)
            else:
                metrics.set_gpu_totals(gpu_0_total_gb=total_gb)

        if req.process_vram_used_mb is not None:
            proc_gb = round(req.process_vram_used_mb / 1024.0, 2)
            if is_gpu_1:
                metrics.set_gpu_process_vram(gpu_1_proc_gb=proc_gb)
            else:
                metrics.set_gpu_process_vram(gpu_0_proc_gb=proc_gb)

        if req.gpu_name:
            if is_gpu_1:
                metrics.set_gpu_names(gpu_1_name=req.gpu_name)
            else:
                metrics.set_gpu_names(gpu_0_name=req.gpu_name)

        if req.telemetry_status:
            metrics.set_telemetry_status(req.telemetry_status)

        await self.repository.upsert_worker_heartbeat(
            worker_id=req.worker_id,
            active_job_id=req.active_job_id,
            gpu_utilization=req.gpu_utilization_percent,
            vram_used_mb=req.vram_used_mb,
            telemetry_status=req.telemetry_status,
            temperature_c=req.temperature_c,
        )

        return WorkerHeartbeatResponse(
            status="acknowledged",
            cancel_requested=cancel_requested,
            lease_extended_until=extended_until,
        )
