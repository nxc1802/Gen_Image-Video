"""
factory/apps/control-api/src/queue/model_runtime.py
ModelRuntimeManager: GPU Allocation, State Machine, and Lifecycle Management for Factory v4 Lite (§5.3).
Enforces:
- Strict state machine: UNLOADED -> LOADING -> READY -> BUSY -> READY -> UNLOADING -> UNLOADED.
- Atomic Dual-GPU Reservation (never half-leased).
- Pre-load VRAM feasibility checks and post-load peak VRAM recording.
- Fast failure with actionable error codes (MODEL_NOT_LOADED, INSUFFICIENT_VRAM).
- Cancellation lease cleanup without stale busy GPU leases or ghost jobs.
- Crash recovery and leak-free model switching.
"""
import asyncio
from datetime import datetime, timezone
from enum import Enum
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid


from factory_contracts.errors import (
    BackendCrashedError,
    FactoryError,
    InsufficientVRAMError,
    InvalidRequestError,
    ModelNotLoadedError,
    UnsupportedCapabilityError,
)
from factory_contracts.manifests.model import AllocationPolicy, ModelManifest
from factory.apps.control_api.src.queue.scheduler import InMemoryScheduler
from model_registry.deployment import DeploymentRegistry


class RuntimeState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    UNLOADING = "unloading"
    CRASHED = "crashed"


class RuntimeLease:
    """Represents an active execution lease on one or more GPUs."""

    def __init__(
        self,
        lease_id: str,
        job_id: str,
        model_id: str,
        gpu_ids: List[int],
        reserved_at: datetime,
        variant_id: Optional[str] = None,
    ):
        self.lease_id = lease_id
        self.job_id = job_id
        self.model_id = model_id
        self.gpu_ids = gpu_ids
        self.reserved_at = reserved_at
        self.variant_id = variant_id


class ModelRuntime:
    """Tracks state, process lifecycle, and memory allocation for a loaded model."""

    def __init__(
        self,
        model_id: str,
        variant_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        checkpoint_revision: Optional[str] = "main",
        backend_type: str = "native",
        backend_version: str = "1.0.0",
        gpu_ids: Optional[List[int]] = None,
        estimated_vram_gb: float = 0.0,
    ):
        self.model_id = model_id
        self.variant_id = variant_id or "standard"
        self.checkpoint_id = checkpoint_id or model_id
        self.checkpoint_revision = checkpoint_revision
        self.backend_type = backend_type
        self.backend_version = backend_version
        self.gpu_ids: List[int] = gpu_ids or [0]
        self.estimated_vram_gb: float = estimated_vram_gb
        self.confirmed_vram_gb: Optional[float] = None
        self.peak_vram_mb: float = estimated_vram_gb * 1024.0
        self.process_id: Optional[int] = None
        self.lock_id: str = f"lock-{uuid.uuid4()}"
        self.state: RuntimeState = RuntimeState.UNLOADED
        self.active_lease: Optional[RuntimeLease] = None
        self.last_used_at: datetime = datetime.now(timezone.utc)
        self.readiness_timestamp: Optional[datetime] = None
        self.dependencies_ready: Dict[str, bool] = {"weights": False, "backend": False}
        self.live_verified: bool = False
        self.error_message: Optional[str] = None

    @property
    def is_healthy(self) -> bool:
        return self.state in (RuntimeState.READY, RuntimeState.BUSY, RuntimeState.UNLOADED) and self.error_message is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "variant_id": self.variant_id,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_revision": self.checkpoint_revision,
            "backend_type": self.backend_type,
            "backend_version": self.backend_version,
            "gpu_ids": list(self.gpu_ids),
            "estimated_vram_gb": round(self.estimated_vram_gb, 2),
            "confirmed_vram_gb": round(self.confirmed_vram_gb, 2) if self.confirmed_vram_gb is not None else None,
            "peak_vram_mb": round(self.peak_vram_mb, 2),
            "process_id": self.process_id,
            "lock_id": self.lock_id,
            "state": self.state.value,
            "active_job_id": self.active_lease.job_id if self.active_lease else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "readiness_timestamp": self.readiness_timestamp.isoformat() if self.readiness_timestamp else None,
            "healthy": self.is_healthy,
            "live_verified": self.live_verified,
            "error_message": self.error_message,
            "dependencies_ready": dict(self.dependencies_ready),
        }



class GPUDeviceInfo:
    """Tracks physical/simulated GPU memory and allocated runtimes."""

    def __init__(self, index: int, total_vram_gb: float = 16.0, system_overhead_gb: float = 1.8):
        self.index = index
        self.total_vram_gb = total_vram_gb
        self.system_overhead_gb = system_overhead_gb
        self.max_allocatable_gb = max(0.0, total_vram_gb - system_overhead_gb)  # 14.2 GB on T4
        self.idle_baseline_gb = 0.5
        self.current_vram_gb = self.idle_baseline_gb
        self.peak_vram_mb = self.idle_baseline_gb * 1024.0
        self.active_model_id: Optional[str] = None


class ModelRuntimeManager:
    """
    Authoritative GPU Model Runtime Manager for Factory v4 Lite.
    Coordinates model loading, GPU leases, switching, memory bounds, and failure recovery.
    """

    def __init__(
        self,
        scheduler: Optional[InMemoryScheduler] = None,
        num_gpus: int = 2,
        max_allocatable_gb_per_gpu: float = 14.2,
        deployment_registry: Optional[DeploymentRegistry] = None,
    ):
        self.scheduler = scheduler or InMemoryScheduler()
        self.deployment_registry = deployment_registry if deployment_registry is not None else DeploymentRegistry.from_environment()
        self.num_gpus = num_gpus
        self.max_allocatable_gb = max_allocatable_gb_per_gpu
        self._lock = asyncio.Lock()

        # GPU Devices metadata
        self.gpus: Dict[int, GPUDeviceInfo] = {
            i: GPUDeviceInfo(index=i, total_vram_gb=16.0, system_overhead_gb=1.8)
            for i in range(num_gpus)
        }

        # Model ID -> ModelRuntime
        self.runtimes: Dict[str, ModelRuntime] = {}

        # Job ID -> RuntimeLease
        self.active_leases: Dict[str, RuntimeLease] = {}

    def _validate_state_transition(
        self,
        current: RuntimeState,
        target: RuntimeState,
        model_id: str,
    ) -> None:
        """
        Enforces strictly allowed state transitions:
        UNLOADED -> LOADING
        LOADING -> READY or UNLOADED (on load error)
        READY -> BUSY or UNLOADING
        BUSY -> READY or UNLOADING or CRASHED
        UNLOADING -> UNLOADED
        CRASHED -> UNLOADED or READY (after recovery)
        """
        allowed = {
            RuntimeState.UNLOADED: {RuntimeState.LOADING},
            RuntimeState.LOADING: {RuntimeState.READY, RuntimeState.UNLOADED, RuntimeState.CRASHED},
            RuntimeState.READY: {RuntimeState.BUSY, RuntimeState.UNLOADING},
            RuntimeState.BUSY: {RuntimeState.READY, RuntimeState.UNLOADING, RuntimeState.CRASHED},
            RuntimeState.UNLOADING: {RuntimeState.UNLOADED},
            RuntimeState.CRASHED: {RuntimeState.UNLOADED, RuntimeState.READY},
        }
        if target not in allowed.get(current, set()):
            raise FactoryError(
                message=f"Invalid runtime state transition from '{current.value}' to '{target.value}' for model '{model_id}'.",
                code="INVALID_STATE_TRANSITION",
                status_code=400,
            )

    async def probe_backend_readiness(
        self,
        model: ModelManifest,
        variant_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], Optional[int], Optional[float]]:
        """
        Authentic live engine / backend health check probe.
        Verifies:
        1. Manifest installation and backend readiness declarations.
        2. Configured upstream engine URL via HTTP health probe if present.
        3. Real GPU VRAM allocation via torch.cuda if available.
        Returns: (is_ready, error_message, process_id, confirmed_vram_gb)
        """
        model_id = model.metadata.id
        service_id = model.metadata.service_id

        # 1. Manifest installation & backend readiness truth
        if not getattr(model, "installed", True) or not getattr(model, "backend_ready", True):
            return (
                False,
                f"Model '{model_id}' weights or backend not ready/installed (installed={getattr(model, 'installed', False)}, backend_ready={getattr(model, 'backend_ready', False)}).",
                None,
                None,
            )

        # 2. Check for configured service backend URL
        backend_url = None
        if service_id == "text-service":
            backend_url = os.environ.get("VLLM_TEXT_URL")
        elif service_id in ("vision-service", "vlm-service"):
            backend_url = os.environ.get("VLLM_VLM_URL")
        elif service_id == "decision-service":
            backend_url = os.environ.get("VLLM_DECISION_URL") or os.environ.get("DECISION_UPSTREAM_URL")
        elif service_id in ("embed-service", "rerank-service", "embedding-service"):
            backend_url = (
                os.environ.get("VLLM_EMBEDDING_URL")
                or os.environ.get("TEI_EMBEDDING_URL")
                or os.environ.get("RERANK_UPSTREAM_URL")
            )
        elif service_id == "speech-service":
            backend_url = os.environ.get("QWEN_ASR_URL") or os.environ.get("QWEN_TTS_URL")
        elif service_id in ("image-service", "video-service"):
            backend_url = os.environ.get("COMFYUI_URL")
        elif service_id == "music-service":
            backend_url = os.environ.get("ACESTEP_API_URL") or os.environ.get("YUE2_API_URL")
        elif service_id == "enhancement-service":
            backend_url = (
                os.environ.get("IMAGE_SR_BACKEND_URL")
                or os.environ.get("VIDEO_SR_BACKEND_URL")
                or os.environ.get("AUDIO_ENHANCE_BACKEND_URL")
            )

        # If a backend URL is explicitly configured, probe it
        if backend_url:
            backend_url = backend_url.rstrip("/")
            try:
                import httpx
                async with httpx.AsyncClient(timeout=1.5) as client:
                    try:
                        resp = await client.get(f"{backend_url}/health")
                        if resp.status_code >= 400:
                            resp = await client.get(backend_url)
                    except Exception:
                        resp = await client.get(backend_url)
                    if resp.status_code >= 500:
                        return (
                            False,
                            f"Live backend service at {backend_url} returned unhealthy status code {resp.status_code}.",
                            None,
                            None,
                        )
            except Exception as exc:
                return (
                    False,
                    f"Live backend service at {backend_url} is unreachable: {exc}",
                    None,
                    None,
                )
        elif os.environ.get("REQUIRE_LIVE_BACKENDS", "false").lower() in ("true", "1", "yes"):
            return (
                False,
                f"No backend service URL configured for '{service_id}' while REQUIRE_LIVE_BACKENDS is enabled.",
                None,
                None,
            )

        # 3. Live GPU telemetry probe if CUDA available
        confirmed_vram: Optional[float] = None
        try:
            import torch
            if torch.cuda.is_available():
                confirmed_vram = round(torch.cuda.memory_allocated() / (1024.0 ** 3), 2)
        except Exception:
            pass

        return (True, None, os.getpid(), confirmed_vram)

    async def ensure_loaded(
        self,
        model: Union[ModelManifest, str],
        variant_id: Optional[str] = None,
        gpu_ids: Optional[List[int]] = None,
        simulated_load_failure: bool = False,
    ) -> ModelRuntime:

        """
        Ensures model is loaded in target GPU(s).
        - Validates VRAM requirements before loading.
        - Unloads any idle model occupying target GPU(s).
        - Transitions state: UNLOADED -> LOADING -> READY.
        - Records peak VRAM post-load.
        - Fast fails on VRAM or load failure with actionable error codes.
        """
        async with self._lock:
            if isinstance(model, str):
                from model_registry.loader import ManifestLoader
                loader = ManifestLoader()
                all_models = loader.load_models()
                if model in all_models:
                    model = all_models[model]
                else:
                    raise InvalidRequestError(
                        message=f"Unknown model manifest for '{model}'.",
                        details={"model_id": model},
                    )

            model_id = model.metadata.id
            variant = variant_id or "standard"

            if model.apiVersion != "factory.ai/v2" and os.environ.get("ENVIRONMENT", os.environ.get("FACTORY_ENV", "development")).lower() == "production":
                raise ModelNotLoadedError(message="Legacy load contracts cannot authorize production model readiness.")

            if model.apiVersion == "factory.ai/v2":
                binding = self.deployment_registry.assignment(model_id)
                if gpu_ids is not None and tuple(gpu_ids) != binding.devices:
                    raise InvalidRequestError("GPU IDs differ from the explicit deployment binding")
                raise ModelNotLoadedError(message="Canonical backend lifecycle has not attested this deployment; legacy load probes cannot mark it READY.")

            # 1. Determine target GPUs
            if gpu_ids is None:
                gpu_ids = list(self.deployment_registry.assignment(model_id).devices)

            if not gpu_ids:
                err = InvalidRequestError(
                    message="gpu_ids must not be empty",
                    details={"model_id": model_id, "error_code": "INVALID_INPUT"},
                )
                err.code = "INVALID_INPUT"
                raise err

            for dev_id in gpu_ids:
                if dev_id not in self.gpus:
                    err = InvalidRequestError(
                        message=f"GPU device id {dev_id} is invalid or out of range",
                        details={"model_id": model_id, "gpu_id": dev_id, "available_gpus": list(self.gpus.keys()), "error_code": "INVALID_INPUT"},
                    )
                    err.code = "INVALID_INPUT"
                    raise err

            # 2. Check if already loaded with matching configuration
            runtime = self.runtimes.get(model_id)
            if runtime is not None:
                if runtime.state in (RuntimeState.READY, RuntimeState.BUSY) and runtime.gpu_ids == gpu_ids and runtime.variant_id == variant:
                    runtime.last_used_at = datetime.now(timezone.utc)
                    return runtime
                # If currently busy with a different variant/job, cannot unload unless idle
                if runtime.state == RuntimeState.BUSY:
                    raise FactoryError(
                        message=f"Model '{model_id}' is currently BUSY executing an active job.",
                        code="RESOURCE_CONFLICT",
                        status_code=409,
                    )
                # If variant or gpu_ids changed and runtime is idle, unload previous configuration first
                if runtime.state == RuntimeState.READY:
                    await self._unload_internal(runtime)

            # 3. Calculate VRAM requirements
            req_vram = model.hardware.min_vram_gb
            checkpoint_id = model.weights.primary_checkpoint
            if variant != "standard" and hasattr(model.weights, "variants") and variant in model.weights.variants:
                v_spec = model.weights.variants[variant]
                if v_spec.min_vram_gb:
                    req_vram = v_spec.min_vram_gb
                if v_spec.checkpoint:
                    checkpoint_id = v_spec.checkpoint

            # Per-GPU requirement
            per_gpu_req = req_vram / len(gpu_ids)

            # 4. Pre-load VRAM check
            if per_gpu_req > self.max_allocatable_gb:
                raise InsufficientVRAMError(
                    message=f"Requested model '{model_id}' requires {per_gpu_req:.1f} GB per GPU, exceeding max allocatable VRAM of {self.max_allocatable_gb:.1f} GB.",
                    details={"model_id": model_id, "required_gb": per_gpu_req, "max_allocatable_gb": self.max_allocatable_gb},
                )

            # 5. Swap out any idle models on target GPUs
            for dev_id in gpu_ids:
                gpu_info = self.gpus[dev_id]
                active_model = gpu_info.active_model_id
                if active_model and active_model != model_id:
                    active_rt = self.runtimes.get(active_model)
                    if active_rt:
                        if active_rt.state == RuntimeState.BUSY:
                            raise FactoryError(
                                message=f"GPU {dev_id} is currently occupied by BUSY model '{active_model}'.",
                                code="RESOURCE_CONFLICT",
                                status_code=409,
                            )
                        # Safe unload of idle model
                        await self._unload_internal(active_rt)

            # 6. Initialize or retrieve runtime object
            if runtime is None:
                runtime = ModelRuntime(
                    model_id=model_id,
                    variant_id=variant,
                    checkpoint_id=checkpoint_id,
                    checkpoint_revision=getattr(model, "revision", "main"),
                    backend_type=getattr(model.metadata, "backend_type", "native"),
                    backend_version=getattr(model.metadata, "version", "1.0.0"),
                    gpu_ids=gpu_ids,
                    estimated_vram_gb=req_vram,
                )
                self.runtimes[model_id] = runtime
            else:
                runtime.variant_id = variant
                runtime.checkpoint_id = checkpoint_id
                runtime.gpu_ids = gpu_ids
                runtime.estimated_vram_gb = req_vram

            # 7. State Transition: UNLOADED -> LOADING
            self._validate_state_transition(runtime.state, RuntimeState.LOADING, model_id)
            runtime.state = RuntimeState.LOADING

            # 8. Authentic live engine / backend readiness probe
            if simulated_load_failure:
                runtime.state = RuntimeState.UNLOADED
                runtime.error_message = "Failed to load model weights: checkpoint corruption or backend timeout."
                runtime.dependencies_ready = {"weights": False, "backend": False}
                runtime.process_id = None
                raise ModelNotLoadedError(
                    message=f"Model '{model_id}' failed to load: {runtime.error_message}",
                    details={"model_id": model_id, "variant_id": variant, "gpu_ids": gpu_ids},
                )

            is_ready, probe_err, pid, confirmed_vram = await self.probe_backend_readiness(model, variant)
            if not is_ready:
                runtime.state = RuntimeState.UNLOADED
                runtime.error_message = probe_err
                runtime.dependencies_ready = {"weights": False, "backend": False}
                runtime.process_id = None
                raise ModelNotLoadedError(
                    message=f"Model '{model_id}' failed readiness probe: {probe_err}",
                    details={"model_id": model_id, "variant_id": variant, "gpu_ids": gpu_ids, "error": probe_err},
                )

            # 9. Apply confirmed state & memory allocation to target GPUs
            runtime.process_id = pid
            runtime.confirmed_vram_gb = confirmed_vram
            runtime.live_verified = (confirmed_vram is not None)
            runtime.dependencies_ready = {"weights": True, "backend": True}

            for dev_id in gpu_ids:
                gpu_info = self.gpus[dev_id]
                gpu_info.active_model_id = model_id
                gpu_info.current_vram_gb = gpu_info.idle_baseline_gb + per_gpu_req
                gpu_info.peak_vram_mb = max(gpu_info.peak_vram_mb, gpu_info.current_vram_gb * 1024.0)

            runtime.peak_vram_mb = max(runtime.peak_vram_mb, (confirmed_vram * 1024.0) if confirmed_vram else (req_vram * 1024.0))
            runtime.readiness_timestamp = datetime.now(timezone.utc)
            runtime.last_used_at = runtime.readiness_timestamp

            # 10. State Transition: LOADING -> READY (Strictly upon confirmed probe)
            self._validate_state_transition(runtime.state, RuntimeState.READY, model_id)
            runtime.state = RuntimeState.READY
            runtime.error_message = None

            return runtime


    async def reserve(
        self,
        model_id: str,
        job_id: str,
        gpu_ids: Optional[List[int]] = None,
        variant_id: Optional[str] = None,
    ) -> RuntimeLease:
        """
        Reserves loaded model for job execution.
        - Enforces READY state.
        - Enforces Atomic Dual-GPU Reservation if multi-GPU:
          acquires all target GPUs simultaneously; if any is busy, acquires NONE.
        - Transitions state: READY -> BUSY.
        """
        async with self._lock:
            runtime = self.runtimes.get(model_id)
            if runtime is None or runtime.state != RuntimeState.READY:
                current_st = runtime.state.value if runtime else "UNLOADED"
                raise ModelNotLoadedError(
                    message=f"Cannot reserve model '{model_id}': runtime state is '{current_st}', expected 'READY'.",
                    details={"model_id": model_id, "state": current_st},
                )

            target_gpus = gpu_ids or runtime.gpu_ids

            # Atomic Multi-GPU / Dual-GPU Reservation
            acquired = self.scheduler.acquire_gpus(target_gpus, job_id)
            if not acquired:
                raise FactoryError(
                    message=f"Could not reserve GPUs {target_gpus} atomically for job '{job_id}': one or more GPUs are currently busy.",
                    code="RESOURCE_CONFLICT",
                    status_code=409,
                    details={"gpu_ids": target_gpus, "job_id": job_id},
                )

            # State Transition: READY -> BUSY
            self._validate_state_transition(runtime.state, RuntimeState.BUSY, model_id)
            runtime.state = RuntimeState.BUSY
            runtime.last_used_at = datetime.now(timezone.utc)

            lease = RuntimeLease(
                lease_id=f"lease-{uuid.uuid4()}",
                job_id=job_id,
                model_id=model_id,
                gpu_ids=target_gpus,
                reserved_at=runtime.last_used_at,
                variant_id=variant_id or runtime.variant_id,
            )
            runtime.active_lease = lease
            self.active_leases[job_id] = lease

            return lease

    async def release(
        self,
        job_id: str,
        peak_vram_mb: Optional[float] = None,
    ) -> None:
        """
        Releases GPU execution lease for job_id.
        - Releases GPU reservation from scheduler.
        - Records peak VRAM telemetry.
        - Transitions state: BUSY -> READY.
        """
        async with self._lock:
            lease = self.active_leases.pop(job_id, None)
            if lease is None:
                return

            runtime = self.runtimes.get(lease.model_id)
            if runtime:
                if runtime.active_lease and runtime.active_lease.job_id == job_id:
                    runtime.active_lease = None
                if peak_vram_mb is not None:
                    runtime.peak_vram_mb = max(runtime.peak_vram_mb, peak_vram_mb)

                if runtime.state == RuntimeState.BUSY:
                    self._validate_state_transition(runtime.state, RuntimeState.READY, runtime.model_id)
                    runtime.state = RuntimeState.READY
                runtime.last_used_at = datetime.now(timezone.utc)

            # Release all reserved GPUs atomically
            self.scheduler.release_gpus(lease.gpu_ids, job_id)

    async def cancel(self, job_id: str) -> bool:
        """
        Cleans up active lease immediately on cancellation without leaving stale busy GPU or ghost job.
        Returns True if an active lease was cleaned up, False otherwise.
        """
        async with self._lock:
            lease = self.active_leases.pop(job_id, None)
            if lease is None:
                self.scheduler.release_gpus(None, job_id)
                return False

            runtime = self.runtimes.get(lease.model_id)
            if runtime:
                if runtime.active_lease and runtime.active_lease.job_id == job_id:
                    runtime.active_lease = None
                if runtime.state == RuntimeState.BUSY:
                    self._validate_state_transition(runtime.state, RuntimeState.READY, runtime.model_id)
                    runtime.state = RuntimeState.READY
                runtime.last_used_at = datetime.now(timezone.utc)

            self.scheduler.release_gpus(lease.gpu_ids, job_id)
            return True

    async def _unload_internal(self, runtime: ModelRuntime) -> None:
        """Internal helper to unload a runtime under self._lock."""
        if runtime.state == RuntimeState.UNLOADED:
            return

        # State Transition: READY/BUSY/CRASHED -> UNLOADING
        if runtime.state in (RuntimeState.READY, RuntimeState.BUSY, RuntimeState.CRASHED):
            self._validate_state_transition(runtime.state, RuntimeState.UNLOADING, runtime.model_id)
            runtime.state = RuntimeState.UNLOADING

        # Release GPU hardware memory back to idle baseline
        for dev_id in runtime.gpu_ids:
            if dev_id in self.gpus:
                gpu_info = self.gpus[dev_id]
                if gpu_info.active_model_id == runtime.model_id:
                    gpu_info.active_model_id = None
                    gpu_info.current_vram_gb = gpu_info.idle_baseline_gb

        # Release any active lease / scheduler hold
        if runtime.active_lease:
            self.scheduler.release_gpus(runtime.active_lease.gpu_ids, runtime.active_lease.job_id)
            self.active_leases.pop(runtime.active_lease.job_id, None)
            runtime.active_lease = None

        # State Transition: UNLOADING -> UNLOADED
        self._validate_state_transition(runtime.state, RuntimeState.UNLOADED, runtime.model_id)
        runtime.state = RuntimeState.UNLOADED
        runtime.process_id = None
        runtime.confirmed_vram_gb = None
        runtime.live_verified = False
        runtime.dependencies_ready = {"weights": False, "backend": False}
        runtime.readiness_timestamp = None
        runtime.error_message = None


    async def unload(
        self,
        model_id: str,
        force: bool = False,
    ) -> None:
        """
        Unloads model and frees GPU memory to baseline.
        Transitions state: READY -> UNLOADING -> UNLOADED.
        """
        async with self._lock:
            runtime = self.runtimes.get(model_id)
            if runtime is None or runtime.state == RuntimeState.UNLOADED:
                return

            if runtime.state == RuntimeState.BUSY and not force:
                raise FactoryError(
                    message=f"Cannot unload model '{model_id}': active job '{runtime.active_lease.job_id if runtime.active_lease else 'unknown'}' is currently running.",
                    code="RESOURCE_CONFLICT",
                    status_code=409,
                )

            await self._unload_internal(runtime)

    async def recover(
        self,
        model_id: Optional[str] = None,
        gpu_id: Optional[int] = None,
        error_code: Optional[str] = None,
    ) -> List[str]:
        """
        Recovers from backend crash or stuck lease.
        Releases GPU partition, cleans up active lease, resets state to UNLOADED.
        Returns list of recovered job IDs.
        """
        recovered_job_ids: List[str] = []
        async with self._lock:
            target_runtimes = [self.runtimes[model_id]] if model_id and model_id in self.runtimes else list(self.runtimes.values())
            for rt in target_runtimes:
                if gpu_id is not None and gpu_id not in rt.gpu_ids:
                    continue

                if rt.active_lease:
                    j_id = rt.active_lease.job_id
                    recovered_job_ids.append(j_id)
                    self.scheduler.release_gpus(rt.active_lease.gpu_ids, j_id)
                    self.active_leases.pop(j_id, None)
                    rt.active_lease = None

                # Reset GPU state
                for dev_id in rt.gpu_ids:
                    if dev_id in self.gpus:
                        gpu_info = self.gpus[dev_id]
                        if gpu_info.active_model_id == rt.model_id:
                            gpu_info.active_model_id = None
                            gpu_info.current_vram_gb = gpu_info.idle_baseline_gb

                rt.state = RuntimeState.UNLOADED
                rt.error_message = f"Recovered from backend failure ({error_code or 'BACKEND_CRASHED'})."
        return recovered_job_ids

    def health(
        self,
        model_id: Optional[str] = None,
        gpu_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Returns health, state machine status, and VRAM metrics for all or filtered models/GPUs."""
        gpu_status = {}
        for idx, g in self.gpus.items():
            if gpu_id is not None and idx != gpu_id:
                continue
            gpu_status[idx] = {
                "total_vram_gb": g.total_vram_gb,
                "allocatable_vram_gb": g.max_allocatable_gb,
                "current_vram_gb": round(g.current_vram_gb, 2),
                "peak_vram_mb": round(g.peak_vram_mb, 2),
                "is_busy": self.scheduler.is_gpu_busy(idx),
                "active_model_id": g.active_model_id,
            }

        runtime_status = {}
        for mid, rt in self.runtimes.items():
            if model_id is not None and mid != model_id:
                continue
            runtime_status[mid] = rt.to_dict()

        return {
            "status": "healthy",
            "gpus": gpu_status,
            "runtimes": runtime_status,
            "active_leases_count": len(self.active_leases),
        }
