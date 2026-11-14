"""
tests/unit/gpu/test_gpu_lifecycle_p1.py
Comprehensive Phase 1 GPU Lifecycle, Model Switching, Truth Model, and Variant Routing Suite (§5.1, §5.2, §5.3, §5.4).

Covers test cases:
- GPU-01: Load model A, infer, unload (verify memory cleanup, process state, idle VRAM return).
- GPU-02: Switch A -> B (verify model B checkpoint and outputs belong to B).
- GPU-03: Two jobs on distinct GPUs (independent completion without allocation conflict).
- GPU-04: Two heavy jobs on same GPU (second waits or returns conflict on busy GPU).
- GPU-05: Dual-GPU model (atomic reservation and release across both GPUs, no half-leasing).
- GPU-06: Backend crash (lease recovered, job terminates with actionable error BACKEND_CRASHED).
- GPU-07: Cancellation during load and inference (no stale busy GPU or ghost job).
- GPU-08: Heartbeat, timeout, retry (no false timeout or infinite retry loop).
- GPU-09: Multi-model pipeline sequence (ACE-Step -> YuE2 -> Video -> ACE-Step on GPU 1, no OOM).
- GPU-10: 10-20 repeated swaps (no cumulative memory leak or stuck queue).
- TRUTH-01 to TRUTH-03: Capability Truth Model (Declared vs Implemented vs Live Verified, schema fields, readiness invariants).
- VARIANT-01 to VARIANT-03: Variant-Safe API Contracts (turbo vs standard checkpoint selection, early rejection).
"""
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import pytest

# v1 lifecycle compatibility fixtures; never native model evidence.
pytestmark = [pytest.mark.contract_mock, pytest.mark.usefixtures("legacy_catalog")]

from factory_contracts.errors import (
    ErrorCode,
    FactoryError,
    InsufficientVRAMError,
    InvalidRequestError,
    ModelNotLoadedError,
)
from factory_contracts.jobs.job_contract import (
    FactoryJob,
    JobProgress,
    JobState,
    JobSubmissionRequest,
    WorkerLease,
)
from factory_contracts.jobs.worker_contract import WorkerHeartbeatRequest, WorkerLeaseRequest
from factory_contracts.manifests.model import (
    AllocationPolicy,
    CapabilityTruthLevel,
    ModelHardwareSpec,
    ModelLimitsSpec,
    ModelManifest,
    ModelMetadata,
    ModelStatus,
    ModelValidationStatus,
    ModelWeightFormat,
    ModelWeightsSpec,
    PrecisionType,
    QuantizationType,
)
from factory.apps.control_api.src.api.routers.jobs import _validate_job_capability
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.model_runtime import (
    GPUDeviceInfo,
    ModelRuntime,
    ModelRuntimeManager,
    RuntimeLease,
    RuntimeState,
)
from factory.apps.control_api.src.queue.repository.memory import InMemoryJobStore
from factory.apps.control_api.src.queue.scheduler import InMemoryScheduler
from factory.apps.control_api.src.queue.watchdog import check_active_jobs_watchdog
from factory.apps.control_api.src.streaming.manager import StreamManager
from model_registry.loader import ManifestLoader
from workflows.image.parameter_mappings import ImageParameterMapper


@pytest.fixture
def manifest_loader():
    return ManifestLoader()


@pytest.fixture
def scheduler():
    return InMemoryScheduler()


@pytest.fixture
def runtime_manager(scheduler):
    return ModelRuntimeManager(scheduler=scheduler, num_gpus=2, max_allocatable_gb_per_gpu=14.2)


# ==============================================================================
# GPU-01: Load Model A, Infer, Unload
# ==============================================================================
@pytest.mark.asyncio
async def test_gpu_01_load_infer_unload(runtime_manager, manifest_loader):
    """
    GPU-01: Load model A, infer, unload.
    PASS: VRAM returns near idle baseline (0.5 GB); process state exits / transitions UNLOADED.
    """
    gpu0 = runtime_manager.gpus[0]
    assert gpu0.current_vram_gb == 0.5
    assert gpu0.active_model_id is None

    # 1. Load Model A (qwen-image-2.1)
    model_a = manifest_loader.load_models()["qwen-image-2.1"]
    rt = await runtime_manager.ensure_loaded(model_a, variant_id="standard", gpu_ids=[0])

    assert rt.state == RuntimeState.READY
    assert rt.variant_id == "standard"
    assert rt.checkpoint_id == model_a.weights.primary_checkpoint
    assert gpu0.active_model_id == "qwen-image-2.1"
    assert gpu0.current_vram_gb > 0.5  # 0.5 + 14.0 = 14.5 GB (allocated)
    assert rt.readiness_timestamp is not None
    assert rt.is_healthy is True

    # 2. Reserve for Job
    lease = await runtime_manager.reserve("qwen-image-2.1", "job-gpu-01", gpu_ids=[0])
    assert lease.job_id == "job-gpu-01"
    assert rt.state == RuntimeState.BUSY
    assert rt.active_lease is not None
    assert runtime_manager.scheduler.is_gpu_busy(0)

    # 3. Simulate Inference and Record Peak VRAM
    await runtime_manager.release("job-gpu-01", peak_vram_mb=14500.0)
    assert rt.state == RuntimeState.READY
    assert rt.peak_vram_mb == 14500.0
    assert not runtime_manager.scheduler.is_gpu_busy(0)

    # 4. Unload Model A
    await runtime_manager.unload("qwen-image-2.1")
    assert rt.state == RuntimeState.UNLOADED
    assert gpu0.current_vram_gb == 0.5  # Returns precisely to idle baseline
    assert gpu0.active_model_id is None


# ==============================================================================
# GPU-02: Switch A -> B
# ==============================================================================
@pytest.mark.asyncio
async def test_gpu_02_model_switching_a_to_b(runtime_manager, manifest_loader):
    """
    GPU-02: Switch A -> B on the same GPU.
    PASS: Model B checkpoint and outputs belong strictly to B; Model A safely unloaded.
    """
    models = manifest_loader.load_models()
    model_a = models["qwen-image-2.1"]
    model_b = models["qwen-decision"]

    # 1. Load Model A on GPU 0
    rt_a = await runtime_manager.ensure_loaded(model_a, gpu_ids=[0])
    assert rt_a.state == RuntimeState.READY
    assert runtime_manager.gpus[0].active_model_id == "qwen-image-2.1"

    # 2. Switch to Model B on GPU 0
    rt_b = await runtime_manager.ensure_loaded(model_b, gpu_ids=[0])

    # 3. Model A must be unloaded, Model B must be READY
    assert rt_a.state == RuntimeState.UNLOADED
    assert rt_b.state == RuntimeState.READY
    assert runtime_manager.gpus[0].active_model_id == "qwen-decision"
    assert rt_b.checkpoint_id == model_b.weights.primary_checkpoint

    # 4. Reserve and execute on B
    lease_b = await runtime_manager.reserve("qwen-decision", "job-gpu-02", gpu_ids=[0])
    assert lease_b.model_id == "qwen-decision"
    assert rt_b.state == RuntimeState.BUSY
    await runtime_manager.release("job-gpu-02")
    assert rt_b.state == RuntimeState.READY


# ==============================================================================
# GPU-03: Two Jobs on Distinct GPUs
# ==============================================================================
@pytest.mark.asyncio
async def test_gpu_03_two_jobs_on_distinct_gpus(runtime_manager, manifest_loader):
    """
    GPU-03: Two jobs executing concurrently on distinct GPUs (GPU 0 vs GPU 1).
    PASS: Independent completion without allocation conflict or interference.
    """
    models = manifest_loader.load_models()
    model_img = models["qwen-image-2.1"]
    model_vid = models["wan-2.2"]

    # Load Model A on GPU 0 and Model B on GPU 1
    rt_0 = await runtime_manager.ensure_loaded(model_img, gpu_ids=[0])
    rt_1 = await runtime_manager.ensure_loaded(model_vid, gpu_ids=[1])

    assert rt_0.state == RuntimeState.READY
    assert rt_1.state == RuntimeState.READY

    # Reserve Job 1 on GPU 0
    lease_1 = await runtime_manager.reserve("qwen-image-2.1", "job-gpu03-1", gpu_ids=[0])
    assert rt_0.state == RuntimeState.BUSY
    assert runtime_manager.scheduler.is_gpu_busy(0)
    assert not runtime_manager.scheduler.is_gpu_busy(1)

    # Reserve Job 2 on GPU 1 concurrently
    lease_2 = await runtime_manager.reserve("wan-2.2", "job-gpu03-2", gpu_ids=[1])
    assert rt_1.state == RuntimeState.BUSY
    assert runtime_manager.scheduler.is_gpu_busy(0)
    assert runtime_manager.scheduler.is_gpu_busy(1)
    assert len(runtime_manager.active_leases) == 2

    # Independent release of Job 1
    await runtime_manager.release("job-gpu03-1")
    assert rt_0.state == RuntimeState.READY
    assert not runtime_manager.scheduler.is_gpu_busy(0)
    assert rt_1.state == RuntimeState.BUSY  # Job 2 still running safely
    assert runtime_manager.scheduler.is_gpu_busy(1)

    # Release Job 2
    await runtime_manager.release("job-gpu03-2")
    assert rt_1.state == RuntimeState.READY
    assert not runtime_manager.scheduler.is_gpu_busy(1)
    assert len(runtime_manager.active_leases) == 0


# ==============================================================================
# GPU-04: Two Heavy Jobs on Same GPU
# ==============================================================================
@pytest.mark.asyncio
async def test_gpu_04_two_heavy_jobs_on_same_gpu(runtime_manager, manifest_loader):
    """
    GPU-04: Two heavy jobs competing for the same GPU.
    PASS: Second request waits or fails with RESOURCE_CONFLICT; cannot double-acquire busy GPU.
    """
    models = manifest_loader.load_models()
    model_a = models["qwen-image-2.1"]
    model_b = models["qwen-decision"]

    await runtime_manager.ensure_loaded(model_a, gpu_ids=[0])
    await runtime_manager.reserve("qwen-image-2.1", "job-gpu04-1", gpu_ids=[0])

    # 1. Second job attempting to reserve same busy model raises conflict
    with pytest.raises(FactoryError) as exc_reserve:
        await runtime_manager.reserve("qwen-image-2.1", "job-gpu04-2", gpu_ids=[0])
    assert exc_reserve.value.code in ("RESOURCE_CONFLICT", "MODEL_NOT_LOADED")

    # 2. Attempting to switch models on busy GPU 0 raises conflict
    with pytest.raises(FactoryError) as exc_switch:
        await runtime_manager.ensure_loaded(model_b, gpu_ids=[0])
    assert exc_switch.value.code == "RESOURCE_CONFLICT"

    # 3. Once Job 1 completes, second job can acquire GPU 0
    await runtime_manager.release("job-gpu04-1")
    rt_b = await runtime_manager.ensure_loaded(model_b, gpu_ids=[0])
    lease_2 = await runtime_manager.reserve("qwen-decision", "job-gpu04-2", gpu_ids=[0])
    assert lease_2.job_id == "job-gpu04-2"
    assert rt_b.state == RuntimeState.BUSY
    await runtime_manager.release("job-gpu04-2")


# ==============================================================================
# GPU-05: Dual-GPU Model (Atomic Reservation)
# ==============================================================================
@pytest.mark.asyncio
@pytest.mark.contract_mock
async def test_gpu_05_atomic_dual_gpu_reservation(runtime_manager):
    """
    GPU-05: Dual-GPU model requires atomic reservation across both GPUs.
    PASS: Both GPUs reserved and released atomically; never half-leased when one GPU is busy.
    """
    # Create synthetic dual-GPU manifest (e.g. Wan 2.2 14B or YuE2 full)
    dual_model = ModelManifest(
        metadata=ModelMetadata(
            id="wan-2.2-14b",
            family="wan-video",
            display_name="Wan 2.2 14B Dual-GPU",
            service_id="video-service",
            version="2.2.0",
        ),
        weights=ModelWeightsSpec(
            primary_checkpoint="Wan-AI/Wan2.2-T2V-14B",
            format=ModelWeightFormat.DIFFUSERS,
            quantization=QuantizationType.FOUR_BIT,
            default_precision=PrecisionType.BF16,
        ),
        hardware=ModelHardwareSpec(
            min_vram_gb=24.0,  # 12.0 GB per GPU across 2 GPUs
            preferred_gpu="t4",
            allocation_policy=AllocationPolicy.DUAL_GPU_SPLIT,
        ),
        capabilities={"text_to_video": True},
        status=ModelStatus.FULL,
        coverage_percent=100.0,
        missing_capabilities=[],
        # Explicit fixture readiness for reservation logic; a validation label
        # must not fabricate installation, checkpoint integrity or health.
        validation_status=ModelValidationStatus.PENDING_LIVE,
        installed=True,
        checkpoint_valid=True,
        backend_ready=True,
    )

    # 1. Occupy GPU 1 with a single-GPU job
    runtime_manager.scheduler.acquire_gpu(1, "job-occupy-gpu1")
    assert runtime_manager.scheduler.is_gpu_busy(1)
    assert not runtime_manager.scheduler.is_gpu_busy(0)

    # Load Dual Model
    rt = await runtime_manager.ensure_loaded(dual_model, gpu_ids=[0, 1])
    assert rt.gpu_ids == [0, 1]

    # 2. Attempt dual-GPU reserve: MUST FAIL atomically because GPU 1 is busy
    with pytest.raises(FactoryError) as exc:
        await runtime_manager.reserve("wan-2.2-14b", "job-dual-fail", gpu_ids=[0, 1])
    assert exc.value.code == "RESOURCE_CONFLICT"

    # CRITICAL INVARIANT: GPU 0 MUST NOT be half-leased!
    assert not runtime_manager.scheduler.is_gpu_busy(0)
    assert "job-dual-fail" not in runtime_manager.active_leases

    # 3. Free GPU 1
    runtime_manager.scheduler.release_gpu(1, "job-occupy-gpu1")
    assert not runtime_manager.scheduler.is_gpu_busy(1)

    # 4. Atomic reservation now succeeds for both GPUs
    lease = await runtime_manager.reserve("wan-2.2-14b", "job-dual-pass", gpu_ids=[0, 1])
    assert lease.gpu_ids == [0, 1]
    assert runtime_manager.scheduler.is_gpu_busy(0)
    assert runtime_manager.scheduler.is_gpu_busy(1)

    # 5. Atomic release frees both GPUs simultaneously
    await runtime_manager.release("job-dual-pass")
    assert not runtime_manager.scheduler.is_gpu_busy(0)
    assert not runtime_manager.scheduler.is_gpu_busy(1)
    assert rt.state == RuntimeState.READY


@pytest.mark.asyncio
async def test_gpu_05b_real_qwen_text_dual_gpu_atomic_reservation(runtime_manager, manifest_loader):
    """
    GPU-05b: Real Qwen-Text manifest requires atomic reservation across both GPUs (dual_gpu_split).
    PASS: Both GPUs 0 and 1 reserved and released atomically; fails when GPU 1 is busy.
    """
    models = manifest_loader.load_models()
    qwen_text = models["qwen-text"]
    assert qwen_text.hardware.allocation_policy == AllocationPolicy.DUAL_GPU_SPLIT

    # 1. Occupy GPU 1 with a single-GPU job
    runtime_manager.scheduler.acquire_gpu(1, "job-occupy-gpu1-text")
    assert runtime_manager.scheduler.is_gpu_busy(1)

    # 2. Loading qwen-text targets both GPUs [0, 1]
    rt = await runtime_manager.ensure_loaded(qwen_text)
    assert rt.gpu_ids == [0, 1]

    with pytest.raises(FactoryError) as exc:
        await runtime_manager.reserve("qwen-text", "job-text-fail", gpu_ids=[0, 1])
    assert exc.value.code == "RESOURCE_CONFLICT"
    assert not runtime_manager.scheduler.is_gpu_busy(0)

    # 3. Free GPU 1
    runtime_manager.scheduler.release_gpu(1, "job-occupy-gpu1-text")

    # 4. Atomic reservation now succeeds across both GPUs
    lease = await runtime_manager.reserve("qwen-text", "job-text-pass", gpu_ids=[0, 1])
    assert lease.gpu_ids == [0, 1]
    assert runtime_manager.scheduler.is_gpu_busy(0)
    assert runtime_manager.scheduler.is_gpu_busy(1)

    # 5. Atomic release frees both GPUs
    await runtime_manager.release("job-text-pass")
    assert not runtime_manager.scheduler.is_gpu_busy(0)
    assert not runtime_manager.scheduler.is_gpu_busy(1)


# ==============================================================================
# GPU-06: Backend Crash Recovery
# ==============================================================================
@pytest.mark.asyncio
async def test_gpu_06_backend_crash_recovery(runtime_manager, manifest_loader):
    """
    GPU-06: Backend crash during active execution.
    PASS: Lease recovered, GPU freed, job terminated with actionable error BACKEND_CRASHED.
    """
    model = manifest_loader.load_models()["qwen-image-2.1"]
    await runtime_manager.ensure_loaded(model, gpu_ids=[0])
    await runtime_manager.reserve("qwen-image-2.1", "job-crash-test", gpu_ids=[0])

    assert runtime_manager.scheduler.is_gpu_busy(0)
    assert "job-crash-test" in runtime_manager.active_leases

    # Simulate backend crash trigger
    recovered = await runtime_manager.recover(model_id="qwen-image-2.1", error_code="BACKEND_CRASHED")
    assert "job-crash-test" in recovered

    # Verify complete cleanup
    assert not runtime_manager.scheduler.is_gpu_busy(0)
    assert "job-crash-test" not in runtime_manager.active_leases
    assert runtime_manager.gpus[0].current_vram_gb == 0.5
    assert runtime_manager.runtimes["qwen-image-2.1"].state == RuntimeState.UNLOADED
    assert "BACKEND_CRASHED" in runtime_manager.runtimes["qwen-image-2.1"].error_message


# ==============================================================================
# GPU-07: Cancellation During Load and Inference
# ==============================================================================
@pytest.mark.asyncio
async def test_gpu_07_cancellation_during_load_and_inference(runtime_manager, manifest_loader):
    """
    GPU-07: Cancellation during execution.
    PASS: Active lease immediately cleaned up, no stale busy GPU or ghost job.
    """
    model = manifest_loader.load_models()["qwen-image-2.1"]
    await runtime_manager.ensure_loaded(model, gpu_ids=[0])
    await runtime_manager.reserve("qwen-image-2.1", "job-cancel-test", gpu_ids=[0])

    assert runtime_manager.scheduler.is_gpu_busy(0)
    assert "job-cancel-test" in runtime_manager.active_leases

    # Cancel execution
    cancelled = await runtime_manager.cancel("job-cancel-test")
    assert cancelled is True

    # Check that GPU is immediately free and state is clean
    assert not runtime_manager.scheduler.is_gpu_busy(0)
    assert "job-cancel-test" not in runtime_manager.active_leases
    assert runtime_manager.runtimes["qwen-image-2.1"].state == RuntimeState.READY
    assert runtime_manager.runtimes["qwen-image-2.1"].active_lease is None


# ==============================================================================
# GPU-08: Heartbeat, Timeout, and Retry
# ==============================================================================
@pytest.mark.asyncio
async def test_gpu_08_heartbeat_timeout_and_retry(scheduler):
    """
    GPU-08: Heartbeat-aware deadlines and bounded retries.
    PASS: No false timeout while heartbeats arrive; stale worker evicted; retry count bounded.
    """
    repo = InMemoryJobStore()
    stream_mgr = StreamManager()
    manager = JobLifecycleManager(repository=repo, broker=scheduler, stream_manager=stream_mgr)

    req = JobSubmissionRequest(
        service="image",
        model="qwen-image-2.1",
        task="text_to_image",
        inputs={"prompt": "test heartbeat", "variant": "turbo"},
    )
    job, _ = await manager.submit_job(req, request_id="req-gpu08")

    # Worker leases job and marks RUNNING
    coordinator = LeaseCoordinator(repository=repo, broker=scheduler)
    lease_req = WorkerLeaseRequest(
        worker_id="worker-gpu08",
        subsystem="image",
        gpu_device=0,
        supported_models=["qwen-image-2.1"],
        supported_tasks=["text_to_image"],
    )
    lease_resp = await coordinator.lease_job(lease_req)
    assert lease_resp.job is not None
    job_id = lease_resp.job.job_id
    lease_token = lease_resp.lease_token
    assert scheduler.is_gpu_busy(0)

    # 1. Normal heartbeats prevent premature timeout
    for _ in range(3):
        hb_resp = await coordinator.process_heartbeat(
            WorkerHeartbeatRequest(
                worker_id="worker-gpu08",
                active_job_id=job_id,
                lease_token=lease_token,
            )
        )
        assert hb_resp.cancel_requested is False

    # Run watchdog: job must NOT time out because heartbeats are fresh
    timed_out = await check_active_jobs_watchdog(
        repository=repo,
        scheduler=scheduler,
        job_manager=manager,
        heartbeat_timeout_seconds=5.0,
    )
    assert timed_out == 0
    assert scheduler.is_gpu_busy(0)

    # 2. Simulate worker abandonment (last_heartbeat aged past deadline)
    db_job = await repo.get_job(job_id)
    repo._workers["worker-gpu08"]["last_heartbeat"] = (
        datetime.now(timezone.utc) - timedelta(seconds=10.0)
    ).isoformat()
    if db_job.worker and db_job.worker.leased_at:
        db_job.worker.leased_at = datetime.now(timezone.utc) - timedelta(seconds=10.0)
    if db_job.started_at:
        db_job.started_at = datetime.now(timezone.utc) - timedelta(seconds=10.0)
    db_job.created_at = datetime.now(timezone.utc) - timedelta(seconds=10.0)
    await repo.update_job(db_job)

    # Run watchdog: detects stale worker and cleans up GPU lease
    timed_out = await check_active_jobs_watchdog(
        repository=repo,
        scheduler=scheduler,
        job_manager=manager,
        heartbeat_timeout_seconds=5.0,
    )
    assert timed_out == 1
    assert not scheduler.is_gpu_busy(0)  # GPU released!

    updated = await repo.get_job(job_id)
    assert updated.state == JobState.FAILED
    assert updated.error.code.value in ("WORKER_DISCONNECTED", "TIMEOUT")


# ==============================================================================
# GPU-09: Multi-Model Pipeline Sequence (ACE-Step -> YuE2 -> Video -> ACE-Step)
# ==============================================================================
@pytest.mark.asyncio
async def test_gpu_09_multi_model_pipeline_sequence(runtime_manager, manifest_loader):
    """
    GPU-09: Multi-model pipeline sequence: ACE-Step -> YuE2 -> Video (Wan 2.2) -> ACE-Step on GPU 1.
    PASS: Correct backend/checkpoint per step, VRAM strictly bounded under 14.2 GB, no OOM.
    """
    models = manifest_loader.load_models()
    gpu1 = runtime_manager.gpus[1]

    # Step 1: Load ACE-Step (music)
    rt_ace1 = await runtime_manager.ensure_loaded(models["ace-step-1.5"], gpu_ids=[1])
    assert rt_ace1.state == RuntimeState.READY
    assert rt_ace1.model_id == "ace-step-1.5"
    assert rt_ace1.estimated_vram_gb <= runtime_manager.max_allocatable_gb
    assert gpu1.current_vram_gb <= gpu1.total_vram_gb
    lease1 = await runtime_manager.reserve("ace-step-1.5", "pipe-step1", gpu_ids=[1])
    await runtime_manager.release("pipe-step1", peak_vram_mb=7000.0)

    # Step 2: Switch to YuE2 (music)
    rt_yue = await runtime_manager.ensure_loaded(models["yue2"], gpu_ids=[1])
    assert rt_ace1.state == RuntimeState.UNLOADED
    assert rt_yue.state == RuntimeState.READY
    assert rt_yue.model_id == "yue2"
    assert rt_yue.estimated_vram_gb <= runtime_manager.max_allocatable_gb
    assert gpu1.current_vram_gb <= gpu1.total_vram_gb
    lease2 = await runtime_manager.reserve("yue2", "pipe-step2", gpu_ids=[1])
    await runtime_manager.release("pipe-step2", peak_vram_mb=12500.0)

    # Step 3: Switch to Video (Wan 2.2)
    rt_wan = await runtime_manager.ensure_loaded(models["wan-2.2"], gpu_ids=[1])
    assert rt_yue.state == RuntimeState.UNLOADED
    assert rt_wan.state == RuntimeState.READY
    assert rt_wan.model_id == "wan-2.2"
    assert rt_wan.estimated_vram_gb <= runtime_manager.max_allocatable_gb
    assert gpu1.current_vram_gb <= gpu1.total_vram_gb
    lease3 = await runtime_manager.reserve("wan-2.2", "pipe-step3", gpu_ids=[1])
    await runtime_manager.release("pipe-step3", peak_vram_mb=14200.0)

    # Step 4: Switch back to ACE-Step
    rt_ace2 = await runtime_manager.ensure_loaded(models["ace-step-1.5"], gpu_ids=[1])
    assert rt_wan.state == RuntimeState.UNLOADED
    assert rt_ace2.state == RuntimeState.READY
    assert rt_ace2.model_id == "ace-step-1.5"
    assert rt_ace2.estimated_vram_gb <= runtime_manager.max_allocatable_gb
    assert gpu1.current_vram_gb <= gpu1.total_vram_gb
    lease4 = await runtime_manager.reserve("ace-step-1.5", "pipe-step4", gpu_ids=[1])
    await runtime_manager.release("pipe-step4", peak_vram_mb=7000.0)

    # Final cleanup
    await runtime_manager.unload("ace-step-1.5")
    assert gpu1.current_vram_gb == 0.5


# ==============================================================================
# GPU-10: 10-20 Repeated Swaps (Memory Leak Verifier)
# ==============================================================================
@pytest.mark.asyncio
async def test_gpu_10_repeated_swaps_memory_leak_check(runtime_manager, manifest_loader):
    """
    GPU-10: 15 repeated model swaps on GPU 0.
    PASS: Zero cumulative memory leak; VRAM returns strictly to baseline; queue free.
    """
    models = manifest_loader.load_models()
    model_a = models["qwen-image-2.1"]
    model_b = models["qwen-decision"]
    gpu0 = runtime_manager.gpus[0]

    initial_baseline = gpu0.current_vram_gb  # 0.5 GB
    assert initial_baseline == 0.5

    for cycle in range(15):
        # Swap in Model A
        rt_a = await runtime_manager.ensure_loaded(model_a, gpu_ids=[0])
        lease_a = await runtime_manager.reserve("qwen-image-2.1", f"swap-a-{cycle}", gpu_ids=[0])
        await runtime_manager.release(f"swap-a-{cycle}", peak_vram_mb=14000.0)

        # Swap in Model B
        rt_b = await runtime_manager.ensure_loaded(model_b, gpu_ids=[0])
        lease_b = await runtime_manager.reserve("qwen-decision", f"swap-b-{cycle}", gpu_ids=[0])
        await runtime_manager.release(f"swap-b-{cycle}", peak_vram_mb=12000.0)

    # Unload final model
    await runtime_manager.unload("qwen-decision")

    # VERIFY: Memory returned precisely to baseline, delta == 0
    final_vram = gpu0.current_vram_gb
    assert final_vram == initial_baseline
    assert gpu0.active_model_id is None
    assert len(runtime_manager.active_leases) == 0
    assert not runtime_manager.scheduler.is_gpu_busy(0)


# ==============================================================================
# Capability Truth Model (§5.1)
# ==============================================================================
def test_truth_model_three_distinct_levels(manifest_loader):
    """
    TRUTH-01: Capability Truth Model distinguishes Declared, Implemented, and Live Verified.
    """
    models = manifest_loader.load_models()
    qwen_img = models["qwen-image-2.1"]

    # Declared & Implemented, pending live
    t2i_truth = qwen_img.get_capability_truth("text_to_image")
    assert isinstance(t2i_truth, CapabilityTruthLevel)
    assert t2i_truth.declared is True
    assert t2i_truth.implemented is True
    assert t2i_truth.live_verified is False
    assert t2i_truth.evidence_run_id is None

    # ACE-Step unbacked capability (audio_to_audio)
    ace = models["ace-step-1.5"]
    a2a_truth = ace.get_capability_truth("audio_to_audio")
    assert a2a_truth.declared is True
    assert a2a_truth.implemented is False
    assert a2a_truth.live_verified is False


def test_truth_model_manifest_schema_fields(manifest_loader):
    """
    TRUTH-02: Manifests contain all mandatory lifecycle and validation schema fields.
    """
    models = manifest_loader.load_models()
    for mid, m in models.items():
        assert hasattr(m, "revision") and m.revision is not None
        assert hasattr(m, "dependencies") and isinstance(m.dependencies, list)
        assert hasattr(m, "task_variant_matrix") and isinstance(m.task_variant_matrix, dict)
        assert hasattr(m, "validated_hardware_profiles") and isinstance(m.validated_hardware_profiles, list)
        assert hasattr(m, "installed") and isinstance(m.installed, bool)
        assert hasattr(m, "checkpoint_valid") and isinstance(m.checkpoint_valid, bool)
        assert hasattr(m, "backend_ready") and isinstance(m.backend_ready, bool)
        assert hasattr(m, "validation_status") and isinstance(m.validation_status, ModelValidationStatus)


def test_truth_model_readiness_not_inferred_from_coverage(manifest_loader):
    """
    TRUTH-03: 100% coverage does NOT imply live certification without valid checkpoint/backend.
    """
    models = manifest_loader.load_models()
    qwen_img = models["qwen-image-2.1"]
    assert qwen_img.coverage_percent == 100.0
    # Pending live must NOT be mistaken for LIVE_VERIFIED
    assert qwen_img.validation_status == ModelValidationStatus.PENDING_LIVE


# ==============================================================================
# Variant-Safe API Contracts (§5.2)
# ==============================================================================
def test_variant_turbo_vs_standard_checkpoint_selection():
    """
    VARIANT-01: variant=turbo selects actual Turbo weights, standard selects Standard weights.
    """
    base_wf = {
        "478": {"class_type": "UNETLoaderGGUF", "inputs": {"unet_name": "placeholder.gguf"}},
        "476": {"class_type": "KSampler", "inputs": {}},
    }

    # Standard variant
    wf_std = ImageParameterMapper.apply(
        dict(base_wf),
        task='image.text_to_image',
        inputs={"prompt": "ocean", "variant": "standard"},
    )
    assert wf_std["478"]["inputs"]["unet_name"] == "qwen-image-2.1-Q4_K_M.gguf"
    assert wf_std["476"]["inputs"]["steps"] == 20

    # Turbo variant
    wf_turbo = ImageParameterMapper.apply(
        dict(base_wf),
        task='image.text_to_image',
        inputs={"prompt": "ocean", "variant": "turbo"},
    )
    assert wf_turbo["478"]["inputs"]["unet_name"] == "qwen-image-2.1-turbo-Q4_K_M.gguf"
    assert wf_turbo["476"]["inputs"]["steps"] == 8


def test_variant_early_rejection_incompatible_task(manifest_loader):
    """
    VARIANT-02: Submitting an incompatible variant for a task rejects early.
    E.g. Inpainting with turbo variant when inpaint only supports standard.
    """
    mock_req = MagicMock()
    mock_req.app.state.manifest_loader = manifest_loader

    req = JobSubmissionRequest(
        service="image",
        model="qwen-image-2.1",
        task="inpaint",
        inputs={"prompt": "repair", "variant": "turbo"},
    )
    with pytest.raises(InvalidRequestError) as exc:
        _validate_job_capability(req, mock_req)
    assert "Variant 'turbo' is not supported for task 'inpaint'" in str(exc.value)


def test_variant_early_rejection_incompatible_steps(manifest_loader):
    """
    VARIANT-03: Submitting invalid step counts for turbo variant rejects early with INVALID_INPUT.
    """
    mock_req = MagicMock()
    mock_req.app.state.manifest_loader = manifest_loader

    # Steps too high for turbo distillation (> 25)
    req_high = JobSubmissionRequest(
        service="image",
        model="qwen-image-2.1",
        task="text_to_image",
        inputs={"prompt": "sunset", "variant": "turbo", "steps": 50},
    )
    with pytest.raises(InvalidRequestError) as exc_high:
        _validate_job_capability(req_high, mock_req)
    assert "Invalid steps '50' for turbo variant" in str(exc_high.value)

    # Negative steps
    req_neg = JobSubmissionRequest(
        service="image",
        model="qwen-image-2.1",
        task="text_to_image",
        inputs={"prompt": "sunset", "variant": "turbo", "steps": -5},
    )
    with pytest.raises(InvalidRequestError) as exc_neg:
        _validate_job_capability(req_neg, mock_req)
    assert "Invalid steps '-5' for turbo variant" in str(exc_neg.value)
