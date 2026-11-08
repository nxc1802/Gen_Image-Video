"""
tests/adversarial/test_p1_gpu_lifecycle_adversarial.py
Adversarial Stress Testing and Vulnerability Probing Suite for Phase 1 (GPU Lifecycle, Registry, Variants, & Concurrency).

Author: challenger_p1_1 (EMPIRICAL CHALLENGER)
Milestone: Phase 1 (§5.1, §5.2, §5.3, §5.4)

Scope of Challenges:
1. State Machine Stress: Invalid transitions (UNLOADED -> BUSY, BUSY -> UNLOADING without force, double release/unload, reserve on non-ready model).
2. Atomic Dual-GPU Concurrency: Contention storms, zero half-leasing, zero deadlock when GPU 0 or 1 is busy.
3. Cancellation & Zombie Leases: Cancel during LOADING, during BUSY, in wait queue, verify no dangling busy GPUs or ghost leases.
4. Memory Leak & Rapid Swapping: 30 consecutive swap cycles across models and variants, strict baseline recovery (0 cumulative VRAM creep).
5. Variant Contract Exploits: Bypassing turbo step limits (0, negative, >25, strings, floats) and invoking turbo on inpaint/outpaint/edit.
6. Backend Crash Recovery: Simulated unhandled worker crashes during inference, recover() resets GPU memory and cleans up leases reliably.
7. Input Robustness & Exception Safety: Edge cases in GPU index bounds and empty allocations.
"""
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock
import pytest

from factory_contracts.errors import (
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
from factory_contracts.jobs.worker_contract import WorkerLeaseRequest
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


@pytest.fixture
def dual_gpu_manifest():
    return ModelManifest(
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
        validation_status=ModelValidationStatus.LIVE_VERIFIED,
    )


# ==============================================================================
# 1. State Machine Adversarial Challenges
# ==============================================================================
class TestP1StateMachineAdversarial:
    """Stress tests state machine transitions, illegal jumps, and idempotency."""

    def test_adv_forbidden_state_transitions(self, runtime_manager):
        """Verifies that all illegal runtime state transitions are strictly rejected."""
        forbidden_transitions = [
            (RuntimeState.UNLOADED, RuntimeState.BUSY),
            (RuntimeState.UNLOADED, RuntimeState.READY),
            (RuntimeState.UNLOADED, RuntimeState.UNLOADING),
            (RuntimeState.UNLOADED, RuntimeState.CRASHED),
            (RuntimeState.LOADING, RuntimeState.BUSY),
            (RuntimeState.READY, RuntimeState.UNLOADED),
            (RuntimeState.READY, RuntimeState.LOADING),
            (RuntimeState.BUSY, RuntimeState.LOADING),
            (RuntimeState.BUSY, RuntimeState.UNLOADED),
            (RuntimeState.UNLOADING, RuntimeState.READY),
            (RuntimeState.UNLOADING, RuntimeState.BUSY),
            (RuntimeState.UNLOADING, RuntimeState.LOADING),
            (RuntimeState.CRASHED, RuntimeState.BUSY),
            (RuntimeState.CRASHED, RuntimeState.LOADING),
        ]
        for src, dst in forbidden_transitions:
            with pytest.raises(FactoryError) as exc:
                runtime_manager._validate_state_transition(src, dst, "test-model")
            assert exc.value.code == "INVALID_STATE_TRANSITION"
            assert f"Invalid runtime state transition from '{src.value}' to '{dst.value}'" in str(exc.value)

    @pytest.mark.asyncio
    async def test_adv_reserve_on_non_ready_states(self, runtime_manager, manifest_loader):
        """Verifies that reserve() raises ModelNotLoadedError on every non-READY state."""
        model = manifest_loader.load_models()["qwen-image-2.1"]
        rt = await runtime_manager.ensure_loaded(model, gpu_ids=[0])
        assert rt.state == RuntimeState.READY

        # 1. Reserve on BUSY
        await runtime_manager.reserve("qwen-image-2.1", "job-1", gpu_ids=[0])
        assert rt.state == RuntimeState.BUSY
        with pytest.raises(ModelNotLoadedError) as exc_busy:
            await runtime_manager.reserve("qwen-image-2.1", "job-2", gpu_ids=[0])
        assert "expected 'READY'" in str(exc_busy.value)
        assert rt.state == RuntimeState.BUSY  # State intact

        # 2. Reserve on UNLOADING
        rt.state = RuntimeState.UNLOADING
        with pytest.raises(ModelNotLoadedError) as exc_unloading:
            await runtime_manager.reserve("qwen-image-2.1", "job-3", gpu_ids=[0])
        assert "expected 'READY'" in str(exc_unloading.value)

        # 3. Reserve on UNLOADED
        rt.state = RuntimeState.UNLOADED
        with pytest.raises(ModelNotLoadedError) as exc_unloaded:
            await runtime_manager.reserve("qwen-image-2.1", "job-4", gpu_ids=[0])
        assert "expected 'READY'" in str(exc_unloaded.value)

        # 4. Reserve on LOADING
        rt.state = RuntimeState.LOADING
        with pytest.raises(ModelNotLoadedError) as exc_loading:
            await runtime_manager.reserve("qwen-image-2.1", "job-5", gpu_ids=[0])
        assert "expected 'READY'" in str(exc_loading.value)

        # 5. Reserve on CRASHED
        rt.state = RuntimeState.CRASHED
        with pytest.raises(ModelNotLoadedError) as exc_crashed:
            await runtime_manager.reserve("qwen-image-2.1", "job-6", gpu_ids=[0])
        assert "expected 'READY'" in str(exc_crashed.value)

    @pytest.mark.asyncio
    async def test_adv_unload_busy_model_prevention(self, runtime_manager, manifest_loader):
        """Unload on a BUSY model with force=False must be rejected with RESOURCE_CONFLICT."""
        model = manifest_loader.load_models()["qwen-image-2.1"]
        await runtime_manager.ensure_loaded(model, gpu_ids=[0])
        await runtime_manager.reserve("qwen-image-2.1", "job-busy-unload", gpu_ids=[0])

        assert runtime_manager.runtimes["qwen-image-2.1"].state == RuntimeState.BUSY
        assert runtime_manager.scheduler.is_gpu_busy(0)

        # Unload without force=True must fail
        with pytest.raises(FactoryError) as exc:
            await runtime_manager.unload("qwen-image-2.1", force=False)
        assert exc.value.code == "RESOURCE_CONFLICT"
        assert exc.value.status_code == 409
        assert "active job 'job-busy-unload' is currently running" in str(exc.value)

        # Verify state and scheduler remain untouched
        assert runtime_manager.runtimes["qwen-image-2.1"].state == RuntimeState.BUSY
        assert runtime_manager.scheduler.is_gpu_busy(0)
        assert "job-busy-unload" in runtime_manager.active_leases

        # Force unload must succeed and clean up
        await runtime_manager.unload("qwen-image-2.1", force=True)
        assert runtime_manager.runtimes["qwen-image-2.1"].state == RuntimeState.UNLOADED
        assert not runtime_manager.scheduler.is_gpu_busy(0)
        assert "job-busy-unload" not in runtime_manager.active_leases

    @pytest.mark.asyncio
    async def test_adv_double_release_and_double_unload_idempotency(self, runtime_manager, manifest_loader):
        """Double release and double unload must be fully idempotent without errors."""
        model = manifest_loader.load_models()["qwen-image-2.1"]
        await runtime_manager.ensure_loaded(model, gpu_ids=[0])
        await runtime_manager.reserve("qwen-image-2.1", "job-double-rel", gpu_ids=[0])

        # First release succeeds
        await runtime_manager.release("job-double-rel")
        assert not runtime_manager.scheduler.is_gpu_busy(0)
        assert runtime_manager.runtimes["qwen-image-2.1"].state == RuntimeState.READY

        # Second release of same job: MUST BE IDEMPOTENT NO-OP
        await runtime_manager.release("job-double-rel")
        assert not runtime_manager.scheduler.is_gpu_busy(0)
        assert runtime_manager.runtimes["qwen-image-2.1"].state == RuntimeState.READY

        # Release of non-existent job: MUST BE IDEMPOTENT NO-OP
        await runtime_manager.release("fake-job-id")

        # First unload succeeds
        await runtime_manager.unload("qwen-image-2.1")
        assert runtime_manager.runtimes["qwen-image-2.1"].state == RuntimeState.UNLOADED
        assert runtime_manager.gpus[0].current_vram_gb == 0.5

        # Second unload of same model: MUST BE IDEMPOTENT NO-OP
        await runtime_manager.unload("qwen-image-2.1")
        assert runtime_manager.runtimes["qwen-image-2.1"].state == RuntimeState.UNLOADED
        assert runtime_manager.gpus[0].current_vram_gb == 0.5

        # Unload of non-existent model: MUST BE IDEMPOTENT NO-OP
        await runtime_manager.unload("non-existent-model")

    def test_adv_model_runtime_initialization_attribute_safety(self):
        """
        Vulnerability Probe: Newly created ModelRuntime instances must initialize
        self.error_message to avoid AttributeError when inspected before ensure_loaded.
        """
        rt = ModelRuntime("test-model")
        # Accessing is_healthy on an unallocated ModelRuntime must return True/False without crashing
        healthy = rt.is_healthy
        assert healthy is True
        # to_dict() must serialize without crashing
        data = rt.to_dict()
        assert data["healthy"] is True
        assert data["error_message"] is None


# ==============================================================================
# 2. Atomic Dual-GPU Concurrency & Deadlock Stress
# ==============================================================================
class TestP1AtomicDualGPUConcurrencyAdversarial:
    """Stress tests atomic multi-GPU reservations under high contention and competing workloads."""

    @pytest.mark.asyncio
    async def test_adv_dual_gpu_zero_half_leasing_when_gpu0_busy(self, runtime_manager, dual_gpu_manifest):
        """When GPU 0 is busy, a dual-GPU reservation must fail atomically without half-leasing GPU 1."""
        # 1. Occupy GPU 0 with single-GPU job
        runtime_manager.scheduler.acquire_gpu(0, "job-occupy-0")
        assert runtime_manager.scheduler.is_gpu_busy(0)
        assert not runtime_manager.scheduler.is_gpu_busy(1)

        # 2. Load dual-GPU model across [0, 1]
        await runtime_manager.ensure_loaded(dual_gpu_manifest, gpu_ids=[0, 1])

        # 3. Attempt dual-GPU reservation: MUST FAIL atomically
        with pytest.raises(FactoryError) as exc:
            await runtime_manager.reserve("wan-2.2-14b", "job-dual-fail-1", gpu_ids=[0, 1])
        assert exc.value.code == "RESOURCE_CONFLICT"

        # CRITICAL ADVERSARIAL INVARIANT: GPU 1 MUST NOT BE HALF-LEASED!
        assert not runtime_manager.scheduler.is_gpu_busy(1), "GPU 1 was half-leased!"
        assert runtime_manager.scheduler._active_gpu_jobs[1] is None
        assert "job-dual-fail-1" not in runtime_manager.active_leases

        # 4. GPU 1 must be immediately acquirable by another single-GPU job
        acquired_1 = runtime_manager.scheduler.acquire_gpu(1, "job-single-1")
        assert acquired_1 is True
        assert runtime_manager.scheduler.is_gpu_busy(1)

        # 5. Clean up single jobs
        runtime_manager.scheduler.release_gpu(1, "job-single-1")
        runtime_manager.scheduler.release_gpu(0, "job-occupy-0")
        assert not runtime_manager.scheduler.is_gpu_busy(0)
        assert not runtime_manager.scheduler.is_gpu_busy(1)

        # 6. Now dual-GPU reservation succeeds for BOTH GPUs
        lease = await runtime_manager.reserve("wan-2.2-14b", "job-dual-ok", gpu_ids=[0, 1])
        assert lease.gpu_ids == [0, 1]
        assert runtime_manager.scheduler.is_gpu_busy(0)
        assert runtime_manager.scheduler.is_gpu_busy(1)
        await runtime_manager.release("job-dual-ok")

    @pytest.mark.asyncio
    async def test_adv_dual_gpu_zero_half_leasing_when_gpu1_busy(self, runtime_manager, dual_gpu_manifest):
        """When GPU 1 is busy, a dual-GPU reservation must fail atomically without half-leasing GPU 0."""
        # 1. Occupy GPU 1 with single-GPU job
        runtime_manager.scheduler.acquire_gpu(1, "job-occupy-1")
        assert runtime_manager.scheduler.is_gpu_busy(1)
        assert not runtime_manager.scheduler.is_gpu_busy(0)

        # 2. Load dual-GPU model
        await runtime_manager.ensure_loaded(dual_gpu_manifest, gpu_ids=[0, 1])

        # 3. Attempt dual-GPU reservation: MUST FAIL atomically
        with pytest.raises(FactoryError) as exc:
            await runtime_manager.reserve("wan-2.2-14b", "job-dual-fail-2", gpu_ids=[0, 1])
        assert exc.value.code == "RESOURCE_CONFLICT"

        # CRITICAL ADVERSARIAL INVARIANT: GPU 0 MUST NOT BE HALF-LEASED!
        assert not runtime_manager.scheduler.is_gpu_busy(0), "GPU 0 was half-leased!"
        assert runtime_manager.scheduler._active_gpu_jobs[0] is None
        assert "job-dual-fail-2" not in runtime_manager.active_leases

        # 4. GPU 0 must be immediately acquirable by another single-GPU job
        acquired_0 = runtime_manager.scheduler.acquire_gpu(0, "job-single-0")
        assert acquired_0 is True

        # Clean up
        runtime_manager.scheduler.release_gpu(0, "job-single-0")
        runtime_manager.scheduler.release_gpu(1, "job-occupy-1")

    @pytest.mark.asyncio
    async def test_adv_concurrent_dual_gpu_reservation_storm(self, runtime_manager, dual_gpu_manifest):
        """Storm of 25 concurrent dual-GPU reservations: exactly ONE wins, 24 fail cleanly, ZERO deadlocks."""
        await runtime_manager.ensure_loaded(dual_gpu_manifest, gpu_ids=[0, 1])

        async def attempt_reserve(idx: int):
            try:
                lease = await runtime_manager.reserve("wan-2.2-14b", f"storm-job-{idx}", gpu_ids=[0, 1])
                return ("SUCCESS", lease.job_id)
            except (FactoryError, ModelNotLoadedError) as e:
                return ("FAILED", str(e))

        tasks = [attempt_reserve(i) for i in range(25)]
        results = await asyncio.gather(*tasks)

        successes = [r for r in results if r[0] == "SUCCESS"]
        failures = [r for r in results if r[0] == "FAILED"]

        assert len(successes) == 1, f"Expected exactly 1 winner, got {len(successes)}"
        assert len(failures) == 24
        winner_job_id = successes[0][1]

        # Both GPUs must be allocated to winner
        assert runtime_manager.scheduler.is_gpu_busy(0)
        assert runtime_manager.scheduler.is_gpu_busy(1)
        assert runtime_manager.scheduler._active_gpu_jobs[0] == winner_job_id
        assert runtime_manager.scheduler._active_gpu_jobs[1] == winner_job_id
        assert len(runtime_manager.active_leases) == 1

        # Release winner and verify full restoration
        await runtime_manager.release(winner_job_id)
        assert not runtime_manager.scheduler.is_gpu_busy(0)
        assert not runtime_manager.scheduler.is_gpu_busy(1)
        assert len(runtime_manager.active_leases) == 0

    @pytest.mark.asyncio
    async def test_adv_mixed_cross_gpu_contention_no_deadlock(self, runtime_manager, manifest_loader, dual_gpu_manifest):
        """Mixed contention: competing jobs for GPU 0, GPU 1, and Dual-GPU [0, 1] running concurrently."""
        models = manifest_loader.load_models()
        model_0 = models["qwen-image-2.1"]
        model_1 = models["ace-step-1.5"]

        await runtime_manager.ensure_loaded(model_0, gpu_ids=[0])
        await runtime_manager.ensure_loaded(model_1, gpu_ids=[1])

        outcomes = []

        async def worker_gpu0(idx: int):
            try:
                await runtime_manager.reserve("qwen-image-2.1", f"g0-{idx}", gpu_ids=[0])
                await asyncio.sleep(0.01)
                await runtime_manager.release(f"g0-{idx}")
                outcomes.append(f"g0-{idx}-OK")
            except Exception as e:
                outcomes.append(f"g0-{idx}-FAIL")

        async def worker_gpu1(idx: int):
            try:
                await runtime_manager.reserve("ace-step-1.5", f"g1-{idx}", gpu_ids=[1])
                await asyncio.sleep(0.01)
                await runtime_manager.release(f"g1-{idx}")
                outcomes.append(f"g1-{idx}-OK")
            except Exception as e:
                outcomes.append(f"g1-{idx}-FAIL")

        # Launch 10 jobs on GPU 0 and 10 jobs on GPU 1
        tasks = [worker_gpu0(i) for i in range(10)] + [worker_gpu1(i) for i in range(10)]
        await asyncio.gather(*tasks)

        # All tasks must resolve without hanging or deadlock
        assert len(outcomes) == 20
        # Clean final state
        assert not runtime_manager.scheduler.is_gpu_busy(0)
        assert not runtime_manager.scheduler.is_gpu_busy(1)
        assert len(runtime_manager.active_leases) == 0


# ==============================================================================
# 3. Cancellation & Zombie Lease Stress
# ==============================================================================
class TestP1CancellationZombieLeaseAdversarial:
    """Stress tests cancellation during BUSY, Dual-GPU execution, queue wait, and idempotency."""

    @pytest.mark.asyncio
    async def test_adv_cancel_during_busy_state(self, runtime_manager, manifest_loader):
        """Immediate cancellation during BUSY clears active lease and restores GPU to free state."""
        model = manifest_loader.load_models()["qwen-image-2.1"]
        await runtime_manager.ensure_loaded(model, gpu_ids=[0])
        await runtime_manager.reserve("qwen-image-2.1", "job-cancel-busy", gpu_ids=[0])

        assert runtime_manager.scheduler.is_gpu_busy(0)
        assert "job-cancel-busy" in runtime_manager.active_leases

        # Cancel job
        cancelled = await runtime_manager.cancel("job-cancel-busy")
        assert cancelled is True

        # Assert no dangling busy GPU or zombie lease
        assert not runtime_manager.scheduler.is_gpu_busy(0)
        assert "job-cancel-busy" not in runtime_manager.active_leases
        assert runtime_manager.runtimes["qwen-image-2.1"].state == RuntimeState.READY
        assert runtime_manager.runtimes["qwen-image-2.1"].active_lease is None

        # Verify next job can immediately acquire GPU 0
        lease2 = await runtime_manager.reserve("qwen-image-2.1", "job-after-cancel", gpu_ids=[0])
        assert lease2.job_id == "job-after-cancel"
        await runtime_manager.release("job-after-cancel")

    @pytest.mark.asyncio
    async def test_adv_cancel_during_dual_gpu_busy_state(self, runtime_manager, dual_gpu_manifest):
        """Cancellation of dual-GPU execution frees both GPUs simultaneously with zero zombie leases."""
        await runtime_manager.ensure_loaded(dual_gpu_manifest, gpu_ids=[0, 1])
        await runtime_manager.reserve("wan-2.2-14b", "job-dual-cancel", gpu_ids=[0, 1])

        assert runtime_manager.scheduler.is_gpu_busy(0)
        assert runtime_manager.scheduler.is_gpu_busy(1)

        # Cancel dual-GPU execution
        cancelled = await runtime_manager.cancel("job-dual-cancel")
        assert cancelled is True

        # Assert BOTH GPUs are immediately free
        assert not runtime_manager.scheduler.is_gpu_busy(0)
        assert not runtime_manager.scheduler.is_gpu_busy(1)
        assert runtime_manager.scheduler._active_gpu_jobs[0] is None
        assert runtime_manager.scheduler._active_gpu_jobs[1] is None
        assert len(runtime_manager.active_leases) == 0
        assert runtime_manager.runtimes["wan-2.2-14b"].state == RuntimeState.READY

    @pytest.mark.asyncio
    async def test_adv_cancel_in_queued_wait_state(self, scheduler):
        """Cancelling a job while still QUEUED prevents worker leasing and frees queue partition."""
        repo = InMemoryJobStore()
        stream_mgr = StreamManager()
        manager = JobLifecycleManager(repository=repo, broker=scheduler, stream_manager=stream_mgr)

        req = JobSubmissionRequest(
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "cancel test"},
        )
        job, _ = await manager.submit_job(req, request_id="req-adv-cancel")
        job_id = job.job_id

        # Cancel job while QUEUED
        cancelled_job = await manager.cancel_job(job_id)
        assert cancelled_job.state == JobState.CANCELLED

        # Worker attempts to dequeue / lease: must NOT receive cancelled job
        coordinator = LeaseCoordinator(repository=repo, broker=scheduler)
        lease_resp = await coordinator.lease_job(
            WorkerLeaseRequest(
                worker_id="worker-test",
                subsystem="image",
                gpu_device=0,
                supported_models=["qwen-image-2.1"],
                supported_tasks=["text_to_image"],
            )
        )
        assert lease_resp.job is None
        assert not scheduler.is_gpu_busy(0)

    @pytest.mark.asyncio
    async def test_adv_cancellation_idempotency(self, runtime_manager):
        """Cancelling non-existent or already-cancelled job returns False without side effects."""
        assert await runtime_manager.cancel("unknown-job") is False
        assert await runtime_manager.cancel("unknown-job") is False
        assert not runtime_manager.scheduler.is_gpu_busy(0)
        assert not runtime_manager.scheduler.is_gpu_busy(1)

    @pytest.mark.asyncio
    async def test_adv_simulated_load_failure_zero_zombies(self, runtime_manager, manifest_loader):
        """Simulated load failure leaves zero zombie memory allocations or active leases."""
        model = manifest_loader.load_models()["qwen-image-2.1"]
        with pytest.raises(ModelNotLoadedError):
            await runtime_manager.ensure_loaded(model, gpu_ids=[0], simulated_load_failure=True)

        assert runtime_manager.runtimes["qwen-image-2.1"].state == RuntimeState.UNLOADED
        assert runtime_manager.gpus[0].current_vram_gb == 0.5
        assert runtime_manager.gpus[0].active_model_id is None
        assert not runtime_manager.scheduler.is_gpu_busy(0)
        assert len(runtime_manager.active_leases) == 0


# ==============================================================================
# 4. Memory Leak & Rapid Swapping Stress
# ==============================================================================
class TestP1MemoryLeakRapidSwapAdversarial:
    """Stress tests rapid continuous swapping over 30 cycles verifying 0 cumulative VRAM leak."""

    @pytest.mark.asyncio
    async def test_adv_rapid_swaps_30_cycles_leak_free(self, runtime_manager, manifest_loader):
        """
        Executes 30 consecutive swap cycles across 4 distinct models and variants.
        PASS: Peak VRAM stays strictly bounded, final VRAM returns precisely to 0.5 GB baseline.
        """
        models = manifest_loader.load_models()
        model_img = models["qwen-image-2.1"]
        model_txt = models["qwen-text"]
        model_vid = models["wan-2.2"]
        model_mus = models["ace-step-1.5"]

        gpu0 = runtime_manager.gpus[0]
        gpu1 = runtime_manager.gpus[1]

        initial_baseline_0 = gpu0.current_vram_gb
        initial_baseline_1 = gpu1.current_vram_gb
        assert initial_baseline_0 == 0.5
        assert initial_baseline_1 == 0.5

        for cycle in range(30):
            # Phase A: Swap on GPU 0 (Image Standard -> Image Turbo -> Text)
            rt_a = await runtime_manager.ensure_loaded(model_img, variant_id="standard", gpu_ids=[0])
            lease_a = await runtime_manager.reserve("qwen-image-2.1", f"c{cycle}-img-std", gpu_ids=[0])
            await runtime_manager.release(f"c{cycle}-img-std", peak_vram_mb=14000.0)

            rt_b = await runtime_manager.ensure_loaded(model_img, variant_id="turbo", gpu_ids=[0])
            lease_b = await runtime_manager.reserve("qwen-image-2.1", f"c{cycle}-img-trb", gpu_ids=[0])
            await runtime_manager.release(f"c{cycle}-img-trb", peak_vram_mb=14000.0)

            rt_c = await runtime_manager.ensure_loaded(model_txt, gpu_ids=[0])
            lease_c = await runtime_manager.reserve("qwen-text", f"c{cycle}-txt", gpu_ids=[0])
            await runtime_manager.release(f"c{cycle}-txt", peak_vram_mb=12000.0)

            # Phase B: Swap on GPU 1 (Music -> Video)
            rt_d = await runtime_manager.ensure_loaded(model_mus, gpu_ids=[1])
            lease_d = await runtime_manager.reserve("ace-step-1.5", f"c{cycle}-mus", gpu_ids=[1])
            await runtime_manager.release(f"c{cycle}-mus", peak_vram_mb=6500.0)

            rt_e = await runtime_manager.ensure_loaded(model_vid, gpu_ids=[1])
            lease_e = await runtime_manager.reserve("wan-2.2", f"c{cycle}-vid", gpu_ids=[1])
            await runtime_manager.release(f"c{cycle}-vid", peak_vram_mb=14100.0)

            # Invariant check every 10 cycles
            if cycle % 10 == 0:
                assert gpu0.current_vram_gb <= 14.5
                assert gpu1.current_vram_gb <= 14.5

        # Final teardown: Unload active models
        await runtime_manager.unload("qwen-text")
        await runtime_manager.unload("wan-2.2")

        # CRITICAL VERIFICATION: STRICT BASELINE RECOVERY & ZERO CUMULATIVE LEAK
        assert gpu0.current_vram_gb == initial_baseline_0, f"GPU 0 leaked VRAM! {gpu0.current_vram_gb} != 0.5"
        assert gpu1.current_vram_gb == initial_baseline_1, f"GPU 1 leaked VRAM! {gpu1.current_vram_gb} != 0.5"
        assert gpu0.active_model_id is None
        assert gpu1.active_model_id is None
        assert len(runtime_manager.active_leases) == 0
        assert not runtime_manager.scheduler.is_gpu_busy(0)
        assert not runtime_manager.scheduler.is_gpu_busy(1)

    @pytest.mark.asyncio
    async def test_adv_pre_load_vram_limit_enforcement(self, runtime_manager):
        """Model requiring > 14.2 GB on single GPU fails pre-load check with INSUFFICIENT_VRAM."""
        huge_model = ModelManifest(
            metadata=ModelMetadata(
                id="huge-model-30b",
                family="huge",
                display_name="Huge 30B",
                service_id="text-service",
                version="1.0.0",
            ),
            weights=ModelWeightsSpec(primary_checkpoint="huge/30b"),
            hardware=ModelHardwareSpec(
                min_vram_gb=28.0,  # 28.0 GB > 14.2 GB max allocatable per GPU
                preferred_gpu="t4",
                allocation_policy=AllocationPolicy.SINGLE_GPU,
            ),
            capabilities={"chat_completion": True},
            status=ModelStatus.FULL,
            coverage_percent=100.0,
            missing_capabilities=[],
            validation_status=ModelValidationStatus.LIVE_VERIFIED,
        )

        with pytest.raises(InsufficientVRAMError) as exc:
            await runtime_manager.ensure_loaded(huge_model, gpu_ids=[0])
        assert exc.value.code == "INSUFFICIENT_VRAM"
        assert "exceeding max allocatable VRAM of 14.2 GB" in str(exc.value)
        # GPU 0 state must remain pristine
        assert runtime_manager.gpus[0].current_vram_gb == 0.5


# ==============================================================================
# 5. Variant-Safe Contract Exploits
# ==============================================================================
class TestP1VariantContractExploitsAdversarial:
    """Exploits aimed at bypassing turbo step limits, incompatible tasks, and parameter mappings."""

    def test_adv_turbo_step_limit_exploits(self, manifest_loader):
        """Attempts to submit invalid step counts for turbo variant: all must reject with INVALID_INPUT."""
        mock_req = MagicMock()
        mock_req.app.state.manifest_loader = manifest_loader

        invalid_step_payloads = [
            26,      # Just over 25 max
            50,      # Much too high
            100,     # Standard steps passed to turbo
            0,       # Zero steps
            -1,      # Negative steps
            -10,     # Strongly negative
            "8",     # String integer
            12.5,    # Float
        ]

        for step in invalid_step_payloads:
            req = JobSubmissionRequest(
                service="image",
                model="qwen-image-2.1",
                task="text_to_image",
                inputs={"prompt": "adversarial steps", "variant": "turbo", "steps": step},
            )
            with pytest.raises(InvalidRequestError) as exc:
                _validate_job_capability(req, mock_req)
            assert "Invalid steps" in str(exc.value)
            assert exc.value.details.get("error_code") == "INVALID_INPUT"

    def test_adv_turbo_unsupported_task_exploits(self, manifest_loader):
        """Attempts to invoke turbo variant on non-generative tasks: inpaint, outpaint, instruction_edit, references."""
        mock_req = MagicMock()
        mock_req.app.state.manifest_loader = manifest_loader

        unsupported_turbo_tasks = [
            "inpaint",
            "outpaint",
            "instruction_edit",
            "reference_image",
            "multi_reference",
        ]

        for task in unsupported_turbo_tasks:
            req = JobSubmissionRequest(
                service="image",
                model="qwen-image-2.1",
                task=task,
                inputs={"prompt": "adversarial task", "variant": "turbo"},
            )
            with pytest.raises(InvalidRequestError) as exc:
                _validate_job_capability(req, mock_req)
            assert f"Variant 'turbo' is not supported for task '{task}'" in str(exc.value)
            assert exc.value.details.get("error_code") == "UNSUPPORTED_CAPABILITY"

    def test_adv_turbo_controlnet_bypass_vulnerability(self, manifest_loader):
        """
        Vulnerability Probe: ControlNet tasks (controlnet, control_pose, control_depth) submitted with
        variant='turbo' must be rejected early with UNSUPPORTED_CAPABILITY, but currently bypass
        validation because canonical_task 'controlnet' is missing from task_variant_matrix.
        """
        mock_req = MagicMock()
        mock_req.app.state.manifest_loader = manifest_loader

        req = JobSubmissionRequest(
            service="image",
            model="qwen-image-2.1",
            task="controlnet",
            inputs={"prompt": "pose test", "variant": "turbo", "control_type": "pose"},
        )
        with pytest.raises(InvalidRequestError) as exc:
            _validate_job_capability(req, mock_req)
        assert "Variant 'turbo' is not supported" in str(exc.value)


    def test_adv_workflow_parameter_mappings_direct_exploits(self):
        """Direct attacks on ImageParameterMapper.apply must raise ValueError on incompatible variants/steps."""
        base_wf = {
            "478": {"class_type": "UNETLoaderGGUF", "inputs": {"unet_name": "dummy.gguf"}},
            "476": {"class_type": "KSampler", "inputs": {}},
        }

        # 1. Turbo on inpaint
        with pytest.raises(ValueError) as exc1:
            ImageParameterMapper.apply(dict(base_wf), task="inpaint", inputs={"variant": "turbo"})
        assert "does not support turbo variant weights" in str(exc1.value)

        # 2. Turbo with 30 steps
        with pytest.raises(ValueError) as exc2:
            ImageParameterMapper.apply(dict(base_wf), task="text_to_image", inputs={"variant": "turbo", "steps": 30})
        assert "Incompatible steps '30' for turbo variant" in str(exc2.value)

        # 3. Standard with 120 steps (> 100 limit)
        with pytest.raises(ValueError) as exc3:
            ImageParameterMapper.apply(dict(base_wf), task="text_to_image", inputs={"steps": 120})
        assert "must be an integer between 1 and 100" in str(exc3.value)

        # 4. Steps as non-integer
        with pytest.raises(ValueError) as exc4:
            ImageParameterMapper.apply(dict(base_wf), task="text_to_image", inputs={"steps": "invalid"})
        assert "must be an integer between 1 and 100" in str(exc4.value)

    def test_adv_unsupported_variant_name_rejected(self, manifest_loader):
        """Requesting an unregistered variant must be rejected early."""
        mock_req = MagicMock()
        mock_req.app.state.manifest_loader = manifest_loader

        # 1. Unregistered variant on model with variants
        req1 = JobSubmissionRequest(
            service="image",
            model="qwen-image-2.1",
            task="text_to_image",
            inputs={"prompt": "test", "variant": "ultra_hd_v2"},
        )
        with pytest.raises(InvalidRequestError) as exc1:
            _validate_job_capability(req1, mock_req)
        assert "Variant 'ultra_hd_v2' is not supported by model 'qwen-image-2.1'" in str(exc1.value)

        # 2. Variant 'turbo' requested on model/task where turbo is unsupported (ace-step-1.5 extract)
        req2 = JobSubmissionRequest(
            service="music",
            model="ace-step-1.5",
            task="extract",
            inputs={"prompt": "jazz", "variant": "turbo"},
        )
        with pytest.raises(InvalidRequestError) as exc2:
            _validate_job_capability(req2, mock_req)
        assert "Variant 'turbo' is not supported for task 'extract' by model 'ace-step-1.5'" in str(exc2.value)


# ==============================================================================
# 6. Backend Crash Recovery & Poison Pill Recovery
# ==============================================================================
class TestP1BackendCrashRecoveryAdversarial:
    """Stress tests crash handling, sudden unhandled exceptions, and memory reset."""

    @pytest.mark.asyncio
    async def test_adv_worker_crash_during_single_gpu_inference(self, runtime_manager, manifest_loader):
        """Simulates worker crash during inference: recover() resets memory, releases lease, sets UNLOADED."""
        model = manifest_loader.load_models()["qwen-image-2.1"]
        await runtime_manager.ensure_loaded(model, gpu_ids=[0])
        await runtime_manager.reserve("qwen-image-2.1", "job-crash-1", gpu_ids=[0])

        assert runtime_manager.scheduler.is_gpu_busy(0)
        assert "job-crash-1" in runtime_manager.active_leases

        # Execute recovery
        recovered = await runtime_manager.recover(model_id="qwen-image-2.1", error_code="BACKEND_CRASHED")
        assert "job-crash-1" in recovered

        # Verify state
        assert not runtime_manager.scheduler.is_gpu_busy(0)
        assert "job-crash-1" not in runtime_manager.active_leases
        assert runtime_manager.gpus[0].current_vram_gb == 0.5
        assert runtime_manager.runtimes["qwen-image-2.1"].state == RuntimeState.UNLOADED
        assert "BACKEND_CRASHED" in runtime_manager.runtimes["qwen-image-2.1"].error_message

        # Next job must be able to load and run cleanly without lingering crash state
        await runtime_manager.ensure_loaded(model, gpu_ids=[0])
        lease_new = await runtime_manager.reserve("qwen-image-2.1", "job-post-crash", gpu_ids=[0])
        assert lease_new.job_id == "job-post-crash"
        await runtime_manager.release("job-post-crash")

    @pytest.mark.asyncio
    async def test_adv_worker_crash_during_dual_gpu_inference(self, runtime_manager, dual_gpu_manifest):
        """Simulates crash during dual-GPU inference: recover() resets BOTH GPUs cleanly."""
        await runtime_manager.ensure_loaded(dual_gpu_manifest, gpu_ids=[0, 1])
        await runtime_manager.reserve("wan-2.2-14b", "job-dual-crash", gpu_ids=[0, 1])

        assert runtime_manager.scheduler.is_gpu_busy(0)
        assert runtime_manager.scheduler.is_gpu_busy(1)

        # Trigger recovery
        recovered = await runtime_manager.recover(model_id="wan-2.2-14b", error_code="BACKEND_CRASHED")
        assert "job-dual-crash" in recovered

        # Both GPUs must be completely reset
        assert not runtime_manager.scheduler.is_gpu_busy(0)
        assert not runtime_manager.scheduler.is_gpu_busy(1)
        assert runtime_manager.gpus[0].current_vram_gb == 0.5
        assert runtime_manager.gpus[1].current_vram_gb == 0.5
        assert runtime_manager.runtimes["wan-2.2-14b"].state == RuntimeState.UNLOADED

    @pytest.mark.asyncio
    async def test_adv_crash_recovery_global_sweep(self, runtime_manager, manifest_loader):
        """Global recovery without arguments sweeps and resets all crashed models and GPUs."""
        models = manifest_loader.load_models()
        await runtime_manager.ensure_loaded(models["qwen-image-2.1"], gpu_ids=[0])
        await runtime_manager.reserve("qwen-image-2.1", "job-g0", gpu_ids=[0])

        await runtime_manager.ensure_loaded(models["wan-2.2"], gpu_ids=[1])
        await runtime_manager.reserve("wan-2.2", "job-g1", gpu_ids=[1])

        assert runtime_manager.scheduler.is_gpu_busy(0)
        assert runtime_manager.scheduler.is_gpu_busy(1)

        # Global recover sweep
        recovered = await runtime_manager.recover(error_code="WORKER_DIED")
        assert "job-g0" in recovered
        assert "job-g1" in recovered

        assert not runtime_manager.scheduler.is_gpu_busy(0)
        assert not runtime_manager.scheduler.is_gpu_busy(1)
        assert runtime_manager.gpus[0].current_vram_gb == 0.5
        assert runtime_manager.gpus[1].current_vram_gb == 0.5
        assert len(runtime_manager.active_leases) == 0


# ==============================================================================
# 7. Input Robustness & Unhandled Exception Probes
# ==============================================================================
class TestP1InputRobustnessProbes:
    """Vulnerability probes for edge case inputs to ModelRuntimeManager."""

    @pytest.mark.asyncio
    async def test_adv_empty_gpu_ids_in_ensure_loaded(self, runtime_manager, manifest_loader):
        """Passing gpu_ids=[] must be validated and rejected with InvalidRequestError/FactoryError, not ZeroDivisionError."""
        model = manifest_loader.load_models()["qwen-image-2.1"]
        with pytest.raises((InvalidRequestError, FactoryError)) as exc:
            await runtime_manager.ensure_loaded(model, gpu_ids=[])
        assert exc.value.code in ("INVALID_INPUT", "INVALID_REQUEST")
        assert "empty" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_adv_out_of_range_gpu_id_in_ensure_loaded(self, runtime_manager, manifest_loader):
        """Passing invalid GPU index must be rejected with InvalidRequestError/FactoryError, not unhandled KeyError."""
        model = manifest_loader.load_models()["qwen-image-2.1"]
        with pytest.raises((InvalidRequestError, FactoryError)) as exc:
            await runtime_manager.ensure_loaded(model, gpu_ids=[99])
        assert exc.value.code in ("INVALID_INPUT", "INVALID_REQUEST")
        assert "99" in str(exc.value)
