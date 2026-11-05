"""
factory/apps/control-api/src/queue/leasing.py
Capability-based worker dispatch and liveness heartbeats for Factory v4 Lite.
Eliminates cryptographic lease tokens, distributed TTL lease renewals, and lease ownership.
"""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from factory_contracts.errors import UnauthorizedError
from factory_contracts.jobs.job_contract import JobState
from factory_contracts.jobs.worker_contract import (
    WorkerHeartbeatRequest,
    WorkerHeartbeatResponse,
    WorkerLeaseRequest,
    WorkerLeaseResponse,
)
from factory_contracts.tasks import (
    get_model_aliases,
    get_task_aliases,
    normalize_model,
    normalize_task,
)
from factory.apps.control_api.src.queue.broker.base import BaseQueueBroker
from factory.apps.control_api.src.queue.repository.base import BaseJobRepository
from factory.apps.control_api.src.telemetry.metrics import metrics

SUBSYSTEM_GPU_MAP: Dict[str, int] = {
    "text": 0, "vlm": 0, "vision": 0, "decision": 0, "image": 0, "speech": 0,
    "video": 1, "music": 1, "enhancement": 1, "embedding": 1, "embed": 1, "rerank": 1,
}

# Authoritative Declarative Manifest Capabilities per Subsystem
SUBSYSTEM_MANIFEST_CAPABILITIES: Dict[str, Dict[str, List[str]]] = {
    "image": {
        "models": ["qwen-image-2.1", "qwen-image"],
        "tasks": [
            "text_to_image", "image.t2i", "t2i",
            "image_to_image", "image.i2i", "i2i",
            "inpaint", "image.inpaint",
            "outpaint", "image.outpaint",
            "instruction_edit", "image.instruction_edit", "edit",
            "reference_conditioning", "image.reference", "image.reference_conditioning",
            "controlnet", "image.controlnet",
        ],
    },
    "video": {
        "models": ["wan-2.2", "wan", "minimax-h3", "minimax"],
        "tasks": [
            "text_to_video", "video.t2v", "t2v",
            "image_to_video", "video.i2v", "i2v",
            "first_last_frame_to_video", "video.fl2v", "fl2v",
            "video_extension", "video.extend", "video.extension",
            "video_with_audio", "video.audio", "video.with_audio",
            "reference_conditioning", "video.reference", "video.ref2va",
        ],
    },
    "music": {
        "models": ["ace-step-1.5", "ace-step"],
        "tasks": [
            "text_to_music", "music.t2m", "t2m",
            "lyrics_to_music", "music.lyrics", "music.l2m", "l2m",
            "audio_to_audio", "music.a2a", "a2a",
        ],
    },
    "speech": {
        "models": ["qwen-speech", "qwen3-tts", "qwen3-asr"],
        "tasks": [
            "speech_synthesis", "speech.tts", "tts", "text_to_speech", "text2speech",
            "speech_transcription", "speech.asr", "asr", "speech_to_text", "speech2text",
        ],
    },
    "enhancement": {
        "models": ["enhancement-suite-v1", "enhancement-suite"],
        "tasks": [
            "image_super_resolution", "enhance.sr_image",
            "video_super_resolution", "enhance.sr_video",
            "frame_interpolation", "enhance.interpolation",
            "audio_enhancement", "enhance.audio",
        ],
    },
    "text": {
        "models": ["qwen-text", "qwen-2.5-72b", "qwen-2.5-72b-instruct"],
        "tasks": ["chat_completion", "chat", "text.chat"],
    },
    "vision": {
        "models": ["qwen-vlm", "qwen-2.5-vl-72b"],
        "tasks": ["visual_language", "vlm", "vision.vlm"],
    },
    "vlm": {
        "models": ["qwen-vlm", "qwen-2.5-vl-72b"],
        "tasks": ["visual_language", "vlm", "vision.vlm"],
    },
    "decision": {
        "models": ["qwen-decision"],
        "tasks": ["decision", "decision.classify"],
    },
    "embedding": {
        "models": ["qwen-embedding"],
        "tasks": ["embedding", "embedding.text"],
    },
    "rerank": {
        "models": ["qwen-embedding"],
        "tasks": ["rerank", "rerank.text"],
    },
}


class LeaseCoordinator:
    """Coordinates capability-based worker job dispatch and liveness heartbeats."""

    def __init__(
        self,
        repository: BaseJobRepository,
        broker: Optional[BaseQueueBroker] = None,
        candidate_limit: int = 50,
    ):
        self.repository = repository
        self.broker = broker
        self.candidate_limit = candidate_limit

    def _verify_manifest_capabilities(
        self,
        subsystem: str,
        requested_models: Optional[List[str]],
        requested_tasks: Optional[List[str]],
    ) -> Tuple[List[str], List[str]]:
        """Validates and filters requested capabilities against authoritative subsystem manifests."""
        norm_sub = str(subsystem).strip().lower() if subsystem else ""
        manifest_info = SUBSYSTEM_MANIFEST_CAPABILITIES.get(norm_sub)
        if not manifest_info:
            return [], []

        allowed_models = manifest_info["models"]
        allowed_tasks = manifest_info["tasks"]

        if requested_models:
            valid_models = [
                m for m in requested_models
                if normalize_model(m) in [normalize_model(am) for am in allowed_models]
            ]
        else:
            valid_models = allowed_models

        if requested_tasks:
            valid_tasks = [
                t for t in requested_tasks
                if normalize_task(t) in [normalize_task(at) for at in allowed_tasks]
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
                if normalize_task(t) in [normalize_task(rt) for rt in reg_tasks]
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

        expanded_tasks = set(effective_tasks)
        for t in effective_tasks:
            norm_t = normalize_task(t)
            expanded_tasks.add(norm_t)
            expanded_tasks.update(get_task_aliases(norm_t))

        target_gpu = req.gpu_device
        if target_gpu is None and registered_worker:
            target_gpu = registered_worker.get("gpu_device")
        if target_gpu is None:
            target_gpu = SUBSYSTEM_GPU_MAP.get(req.subsystem, 0)

        # Enforce strict GPU concurrency bound (max 1 concurrent job per discrete GPU)
        if self.broker and hasattr(self.broker, "is_gpu_busy") and target_gpu is not None:
            if self.broker.is_gpu_busy(target_gpu):
                return WorkerLeaseResponse(job=None, lease_token=None, lease_expires_at=None)

        candidates = await self.repository.find_eligible_jobs_for_worker(
            subsystem=registered_worker.get("subsystem", req.subsystem),
            supported_models=list(expanded_models),
            supported_tasks=list(expanded_tasks),
            limit=self.candidate_limit,
        )

        if not candidates:
            return WorkerLeaseResponse(job=None, lease_token=None, lease_expires_at=None)

        now = datetime.now(timezone.utc)
        duration = getattr(req, "lease_duration_seconds", 300) or 300
        expires_at = now + timedelta(seconds=duration)

        for job in candidates:
            job_gpu = target_gpu if target_gpu is not None else SUBSYSTEM_GPU_MAP.get(job.service, 0)
            if self.broker and hasattr(self.broker, "is_gpu_busy") and self.broker.is_gpu_busy(job_gpu):
                continue

            if self.broker and hasattr(self.broker, "acquire_gpu"):
                if not self.broker.acquire_gpu(job_gpu, job.job_id):
                    continue

            lease_token = getattr(req, "lease_token", None) or f"lease-{uuid.uuid4()}"
            try:
                acquired = await self.repository.acquire_lease(
                    job_id=job.job_id,
                    worker_id=req.worker_id,
                    lease_token=lease_token,
                    expires_at=expires_at,
                    gpu_device=job_gpu,
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
                        gpu_device=job_gpu,
                        active_job_id=job.job_id,
                    )
                    return WorkerLeaseResponse(
                        job=updated_job,
                        lease_token=lease_token,
                        lease_expires_at=expires_at,
                    )
                else:
                    if self.broker and hasattr(self.broker, "release_gpu"):
                        self.broker.release_gpu(job_gpu, job.job_id)
            except Exception:
                if self.broker and hasattr(self.broker, "release_gpu"):
                    self.broker.release_gpu(job_gpu, job.job_id)
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
