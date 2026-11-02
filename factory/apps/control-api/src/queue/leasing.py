"""
factory/apps/control-api/src/queue/leasing.py
Worker leasing and heartbeat coordination with verified manifest capability matching.
"""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

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
        Validates worker existence in worker_registrations and uses verified capabilities
        rather than self-declared parameters.
        Acquires atomic lease with TTL and evicts job from active broker queue.
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

            # Register verified worker in worker_registrations
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

        # 3. Use authoritative registered capabilities from repository, NOT self-declared request parameters
        reg_models = registered_worker.get("supported_models", [])
        reg_tasks = registered_worker.get("supported_tasks", [])

        # If worker specifies subsets, intersect with verified registered capabilities
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

        candidates = await self.repository.find_eligible_jobs_for_worker(
            subsystem=registered_worker.get("subsystem", req.subsystem),
            supported_models=list(expanded_models),
            supported_tasks=list(expanded_tasks),
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
                    subsystem=registered_worker.get("subsystem", req.subsystem),
                    supported_models=effective_models,
                    supported_tasks=effective_tasks,
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
            telemetry_status=req.telemetry_status,
            temperature_c=req.temperature_c,
        )

        return WorkerHeartbeatResponse(
            status="acknowledged",
            cancel_requested=cancel_requested,
            lease_extended_until=extended_until,
        )
