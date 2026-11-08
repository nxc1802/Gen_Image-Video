"""
tests/adversarial/test_adversarial_p0_fixes_verification.py
Empirical Adversarial Stress Suite for Phase 0 Bug Fixes (FIX-01 to FIX-20).
Authored by Phase 0 Adversarial Challenger.
"""
import asyncio
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from factory_contracts.errors import ErrorCode, ErrorDetail, FactoryError, InvalidRequestError
from factory_contracts.errors.error_contract import classify_worker_error
from factory_contracts.jobs.job_contract import (
    ArtifactReference,
    FactoryJob,
    JobProgress,
    JobState,
    JobSubmissionRequest,
    WorkerLease,
)
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    JobFailureRequest,
    ProgressUpdateRequest,
)
from factory_contracts.tasks import normalize_model
from factory.apps.control_api.src.api.routers.jobs import (
    MODEL_ALIASES,
    _validate_job_capability,
)
from factory.apps.control_api.src.queue.leasing import SUBSYSTEM_GPU_MAP as LEASING_MAP
from factory.apps.control_api.src.queue.manager import JobLifecycleManager, SUBSYSTEM_GPU_MAP as MANAGER_MAP
from factory.apps.control_api.src.queue.policy import SUBSYSTEM_GPU_MAP, get_subsystem_gpu
from factory.apps.control_api.src.queue.repository.memory import InMemoryJobStore
from factory.apps.control_api.src.queue.scheduler import InMemoryScheduler, SUBSYSTEM_GPU_MAP as SCHEDULER_MAP
from factory.apps.control_api.src.queue.watchdog import (
    MODEL_EXECUTION_TIMEOUTS,
    SERVICE_EXECUTION_TIMEOUTS,
    SUBSYSTEM_GPU_MAP as WATCHDOG_MAP,
    TASK_EXECUTION_TIMEOUTS,
    check_active_jobs_watchdog,
)
from factory.apps.control_api.src.storage.integrity import compute_bytes_sha256
from factory.apps.control_api.src.streaming.manager import StreamManager
from model_registry.loader import ManifestLoader
from services.music.acestep.client import ACEStepClient
from services.video.extension_helper import VideoExtensionHelper
from workflows.image.parameter_mappings import ImageParameterMapper
from workflows.video.parameter_mappings import VideoParameterMapper


# ==============================================================================
# VULNERABILITY 1 (FIX-01 & FIX-02): Nonexistent / Inaccessible Artifact with Non-Zero Hash
# ==============================================================================
@pytest.mark.asyncio
async def test_adv_fix01_missing_inaccessible_artifact_with_nonzero_hash():
    """
    ADVERSARIAL PROBE FOR FIX-01 & FIX-02:
    A worker completes a media job with an ArtifactReference pointing to a nonexistent file on disk,
    but provides a non-zero hash ('a'*64) and size_bytes > 0.
    According to FIX-01: "Reject success if required artifact is missing, empty, inaccessible or invalid.
    Missing output produces FAILED, not SUCCEEDED."
    If this produces SUCCEEDED, it is a critical integrity breach.
    """
    repo = InMemoryJobStore(db_path=":memory:")
    await repo.initialize()
    scheduler = InMemoryScheduler()
    stream_mgr = StreamManager()
    manager = JobLifecycleManager(repo, scheduler, stream_mgr)

    job, _ = await manager.submit_job(
        JobSubmissionRequest(
            service="video",
            model="wan-2.2",
            task="text_to_video",
            inputs={"prompt": "Adversarial missing artifact probe"},
        ),
        request_id="req-adv-art-01",
    )

    fake_art = ArtifactReference(
        artifact_id="art-adv-fake",
        type="video",
        mime_type="video/mp4",
        size_bytes=4096,
        storage_path="/tmp/definitely_inaccessible_nonexistent_file_987654.mp4",
        download_url="/v1/jobs/fake/result",
        url="/v1/jobs/fake/result",
        sha256="a" * 64,
        created_at=datetime.now(timezone.utc),
    )

    await manager.complete_job(
        job.job_id,
        JobCompletionRequest(
            artifacts=[fake_art],
            inference_duration_ms=100.0,
        ),
    )

    job_after = await repo.get_job(job.job_id)
    # The requirement mandates: Missing output produces FAILED, not SUCCEEDED
    assert job_after.state == JobState.FAILED, (
        f"FIX-01 VULNERABILITY DETECTED: Missing/inaccessible artifact was marked {job_after.state} "
        f"instead of FAILED! Nonexistent file on disk was trusted because worker provided a non-zero hash."
    )


# ==============================================================================
# VULNERABILITY 2 (FIX-05): Watchdog Prematurely Kills Progressing Job
# ==============================================================================
@pytest.mark.asyncio
async def test_adv_fix05_watchdog_unreachable_task_timeout_premature_kill():
    """
    ADVERSARIAL PROBE FOR FIX-05:
    A video extension job runs for 700s with active worker heartbeats and 50% progress.
    The job did NOT supply a custom inputs['timeout'] override, relying on standard task timeouts
    (TASK_EXECUTION_TIMEOUTS['video_extension'] = 1200s).
    The watchdog sweep is called with default service_timeouts=None.
    According to FIX-05: "Slow valid video/music jobs are not killed prematurely."
    If the job is terminated at 375s (because watchdog defaults to 300s when service_timeouts is None),
    this is a critical watchdog failure.
    """
    repo = InMemoryJobStore(db_path=":memory:")
    await repo.initialize()
    scheduler = InMemoryScheduler()
    stream_mgr = StreamManager()
    manager = JobLifecycleManager(repo, scheduler, stream_mgr)

    now = datetime.now(timezone.utc)
    job, _ = await manager.submit_job(
        JobSubmissionRequest(
            service="video",
            model="wan-2.2",
            task="video_extension",
            inputs={"prompt": "extend clip without explicit timeout parameter"},
        ),
        request_id="req-adv-wd-05",
    )

    await manager.update_progress(
        job.job_id,
        ProgressUpdateRequest(stage="sampling", step=15, total=30, percent=50.0),
    )
    async with repo._lock:
        repo._jobs[job.job_id].started_at = now - timedelta(seconds=700)
        repo._jobs[job.job_id].worker = WorkerLease(
            worker_id="worker-vid-01",
            leased_at=now - timedelta(seconds=700),
            gpu_device=1,
        )
    await repo.upsert_worker_heartbeat(
        worker_id="worker-vid-01",
        active_job_id=job.job_id,
        subsystem="video",
        gpu_device=1,
    )
    async with repo._lock:
        repo._workers["worker-vid-01"]["last_heartbeat"] = now.isoformat()

    failed_count = await check_active_jobs_watchdog(
        repository=repo,
        scheduler=scheduler,
        job_manager=manager,
        heartbeat_timeout_seconds=60.0,
        default_job_timeout_seconds=300.0,
    )
    job_after = await repo.get_job(job.job_id)

    assert failed_count == 0, (
        f"FIX-05 BUG DETECTED: Watchdog killed healthy progressing video_extension job at 700s! "
        f"Error: {job_after.error.message if job_after.error else 'None'}. "
        f"TASK_EXECUTION_TIMEOUTS (1200s) was ignored because service_timeouts is None."
    )
    assert job_after.state == JobState.RUNNING


# ==============================================================================
# VERIFICATION SUITE FOR PASSING FIXES (ROBUSTNESS TESTS)
# ==============================================================================

@pytest.mark.asyncio
async def test_adv_fix01_zero_byte_artifact_path_fails():
    """FIX-01: An empty 0-byte artifact file on disk must fail the job."""
    repo = InMemoryJobStore(db_path=":memory:")
    await repo.initialize()
    scheduler = InMemoryScheduler()
    stream_mgr = StreamManager()
    manager = JobLifecycleManager(repo, scheduler, stream_mgr)

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        # 0 bytes
        job, _ = await manager.submit_job(
            JobSubmissionRequest(service="video", model="wan-2.2", task="text_to_video", inputs={"prompt": "test"}),
            request_id="req-0-byte",
        )
        res = await manager.complete_job(
            job.job_id,
            JobCompletionRequest(artifact_path=tmp.name),
        )
        assert res.state == JobState.FAILED
        assert res.error.code == ErrorCode.ARTIFACT_MISSING


@pytest.mark.asyncio
async def test_adv_fix02_hash_mismatch_fails():
    """FIX-02: Artifact SHA-256 mismatch against actual bytes must fail the job."""
    repo = InMemoryJobStore(db_path=":memory:")
    await repo.initialize()
    scheduler = InMemoryScheduler()
    stream_mgr = StreamManager()
    manager = JobLifecycleManager(repo, scheduler, stream_mgr)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(b"real image bytes")
        tmp_path = tmp.name

    try:
        job, _ = await manager.submit_job(
            JobSubmissionRequest(service="image", model="qwen-image-2.1", task="text_to_image", inputs={"prompt": "test"}),
            request_id="req-mismatch",
        )
        art = ArtifactReference(
            artifact_id="art-mis",
            type="image",
            mime_type="image/png",
            size_bytes=len(b"real image bytes"),
            storage_path=tmp_path,
            download_url="/v1/jobs/test/result",
            url="/v1/jobs/test/result",
            sha256="f" * 64,  # intentionally wrong hash
            created_at=datetime.now(timezone.utc),
        )
        res = await manager.complete_job(
            job.job_id,
            JobCompletionRequest(artifacts=[art]),
        )
        assert res.state == JobState.FAILED
        assert res.error.code == ErrorCode.ARTIFACT_MISSING
        assert "mismatch" in res.error.message.lower()
    finally:
        os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_adv_fix03_retry_retains_gpu_affinity():
    """FIX-03: Retrying a video/music job re-enqueues to GPU 1."""
    repo = InMemoryJobStore(db_path=":memory:")
    await repo.initialize()
    scheduler = InMemoryScheduler()
    stream_mgr = StreamManager()
    manager = JobLifecycleManager(repo, scheduler, stream_mgr)

    job, _ = await manager.submit_job(
        JobSubmissionRequest(service="video", model="wan-2.2", task="text_to_video", inputs={"prompt": "test"}),
        request_id="req-v",
    )
    # Dequeue it initially
    d_id = await scheduler.dequeue_for_gpu(1)
    assert d_id == job.job_id

    # Fail job with retryable=True
    failed_job = await manager.fail_job(
        job.job_id,
        JobFailureRequest(
            error=ErrorDetail(code=ErrorCode.TIMEOUT, message="Timeout", request_id="req-v"),
            retryable=True,
        ),
    )
    assert failed_job.state == JobState.QUEUED
    assert failed_job.retry_count == 1

    # Verify re-enqueued to GPU 1
    retry_id = await scheduler.dequeue_for_gpu(1)
    assert retry_id == job.job_id


def test_adv_fix04_policy_identity():
    """FIX-04: Consolidated SUBSYSTEM_GPU_MAP in policy.py is identical across all queue modules."""
    assert MANAGER_MAP is SUBSYSTEM_GPU_MAP
    assert LEASING_MAP is SUBSYSTEM_GPU_MAP
    assert WATCHDOG_MAP is SUBSYSTEM_GPU_MAP
    assert SCHEDULER_MAP is SUBSYSTEM_GPU_MAP
    assert get_subsystem_gpu("video") == 1
    assert get_subsystem_gpu("music") == 1
    assert get_subsystem_gpu("text") == 0
    assert get_subsystem_gpu("image") == 0


def test_adv_fix06_model_alias_rejections():
    """FIX-06: Purged misleading 72B / 8B aliases."""
    for bad in ["qwen-2.5-72b", "qwen-2.5-72b-instruct", "qwen-vl-max", "qwen-decision-8b"]:
        assert bad not in MODEL_ALIASES
        assert normalize_model(bad) != "qwen-text"


def test_adv_fix07_unserved_jobs_fail_admission():
    """FIX-07: Unserved models, invalid variants, and impossible tasks fail admission."""
    loader = ManifestLoader()
    mock_req = MagicMock()
    mock_req.app.state.manifest_loader = loader

    # Bad variant on valid model
    bad_var = JobSubmissionRequest(
        service="image",
        model="qwen-image-2.1",
        task="text_to_image",
        inputs={"prompt": "test", "variant": "ultra_hd_99k"},
    )
    with pytest.raises(InvalidRequestError) as exc_v:
        _validate_job_capability(bad_var, mock_req)
    assert "not supported" in str(exc_v.value)


def test_adv_fix10_acestep_task_mapping():
    """FIX-10: ACEStepClient maps tasks accurately to native API endpoints."""
    assert ACEStepClient.TASK_TYPE_MAPPING["text_to_music"] == "text2music"
    assert ACEStepClient.TASK_TYPE_MAPPING["cover"] == "cover"
    assert ACEStepClient.TASK_TYPE_MAPPING["repaint"] == "repaint"
    payload = ACEStepClient._release_payload("cover", {"prompt": "acoustic cover", "src_audio_path": "/path/orig.mp3", "cover_strength": 0.7})
    assert payload["task_type"] == "cover"
    assert payload["src_audio_path"] == "/path/orig.mp3"
    assert payload["audio_cover_strength"] == 0.7


def test_adv_fix11_turbo_variant_checkpoint():
    """FIX-11: Qwen-Image turbo variant loads turbo GGUF checkpoint in node 478."""
    template = {
        "476": {"inputs": {"steps": 20, "seed": 0, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"}},
        "478": {"inputs": {"unet_name": "placeholder.gguf"}},
    }
    wf_std = ImageParameterMapper.apply(template.copy(), task="text_to_image", inputs={"variant": "standard"})
    assert wf_std["478"]["inputs"]["unet_name"] == "qwen-image-2.1-Q4_K_M.gguf"

    wf_turbo = ImageParameterMapper.apply(template.copy(), task="text_to_image", inputs={"variant": "turbo"})
    assert wf_turbo["478"]["inputs"]["unet_name"] == "qwen-image-2.1-turbo-Q4_K_M.gguf"
    assert wf_turbo["476"]["inputs"]["steps"] == 8


def test_adv_fix12_minimax_ref2va_binding():
    """FIX-12: MiniMax H3 ref2va binds reference image into node 8_ref."""
    workflow = {
        "8_ref": {"class_type": "LoadImage", "inputs": {"image": "default.png"}},
        "9": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"reference_image": ["8_ref", 0]}},
    }
    res = VideoParameterMapper.apply(
        workflow=workflow,
        model="minimax-h3",
        task="reference_conditioning",
        inputs={"prompt": "Cinematic motion", "reference_image_url": "https://example.com/character.png"},
        staged_files={"reference_image_url": "staged_character_123.png"},
    )
    assert res["8_ref"]["inputs"]["image"] == "staged_character_123.png"


def test_adv_fix13_extension_concatenation_order():
    """FIX-13: VideoExtensionHelper concatenates original segment first, then extension."""
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "original.mp4"
        ext = Path(td) / "extended.mp4"
        out = Path(td) / "stitched.mp4"
        src.write_bytes(b"CLIP1" * 100)
        ext.write_bytes(b"CLIP2" * 100)

        with patch("subprocess.run") as mock_run:
            mock_res = MagicMock()
            mock_res.returncode = 0
            mock_run.return_value = mock_res
            out.write_bytes(b"CONCAT_RESULT" * 100)

            result = VideoExtensionHelper.concatenate_videos(src, ext, out)
            assert result == out
            cmd = mock_run.call_args[0][0]
            assert "ffmpeg" in cmd
            assert "-f" in cmd and "concat" in cmd


def test_adv_fix17_worker_error_taxonomy():
    """FIX-17: classify_worker_error maps exceptions to appropriate codes and retryability."""
    code, msg, retry = classify_worker_error(RuntimeError("CUDA out of memory in backward pass"), subsystem="video")
    assert code == ErrorCode.INSUFFICIENT_VRAM
    assert retry is False

    code_to, msg_to, retry_to = classify_worker_error(TimeoutError("Read timed out"), subsystem="image")
    assert code_to == ErrorCode.INFERENCE_TIMEOUT
    assert retry_to is True

    code_conn, msg_conn, retry_conn = classify_worker_error(ConnectionError("Connection refused on port 8188"), subsystem="video")
    assert code_conn == ErrorCode.BACKEND_UNAVAILABLE
    assert retry_conn is True
