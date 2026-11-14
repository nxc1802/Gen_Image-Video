"""Phase 2 software contracts; no model backend, inference or GPU evidence."""
from pathlib import Path
from unittest.mock import AsyncMock
import pytest
from pydantic import ValidationError

from benchmarks.capability_gate import audit_source
from factory_contracts.capabilities import normalize_capability
from factory_contracts.jobs.job_contract import FactoryJob, JobState, JobSubmissionRequest
from factory_contracts.jobs.worker_contract import WorkerLeaseRequest
from factory.apps.control_api.src.queue.repository.memory import InMemoryJobStore
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from model_registry.loader import ManifestLoader

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["t2i", "image.t2i", "text2image"])
async def test_aliases_are_canonical_before_persistence_and_lease_query(alias, routing_registry):
    request = JobSubmissionRequest(service="image", model="qwen-image-2.1", task=alias, inputs={"prompt": "a cat"})
    repo = InMemoryJobStore()
    await repo.initialize()
    try:
        # Queue contract only: deliberately bypass admission/backend execution.
        job = FactoryJob(job_id="job-00000000-0000-0000-0000-000000000001",
            request_id="req-00000000-0000-0000-0000-000000000001", service=request.service,
            model=request.model, task=request.task, inputs=request.inputs, state=JobState.QUEUED)
        await repo.create_job(job)
        query = AsyncMock(wraps=repo.find_eligible_jobs_for_worker)
        repo.find_eligible_jobs_for_worker = query
        coordinator = LeaseCoordinator(repo, manifest_loader=ManifestLoader(ROOT / "manifests"), deployment_registry=routing_registry([("qwen-image-2.1", [0])]))
        lease = await coordinator.lease_job(WorkerLeaseRequest(worker_id="vocabulary-contract", subsystem="image",
            supported_models=[request.model], supported_tasks=[alias], gpu_device=0))
        assert lease.job.task == request.task == "image.text_to_image"
        assert query.call_args.kwargs["supported_tasks"] == ["image.text_to_image"]
        worker = await repo.get_worker("vocabulary-contract")
        assert worker["supported_tasks"] == ["image.text_to_image"]
    finally:
        await repo.close()


def test_reference_names_are_scoped_and_multiple_references_are_explicit():
    assert normalize_capability("reference", engine="video", inputs={}) == "video.reference_conditioning"
    assert normalize_capability("reference", engine="music", inputs={}) == "music.reference_audio"
    request = JobSubmissionRequest(service="image", model="qwen-image-2.1", task="reference",
        inputs={"reference_image_urls": ["https://example.com/one.png", "https://example.com/two.png"]})
    assert request.task == "image.multi_reference"


def test_control_type_cannot_change_an_explicit_canonical_capability():
    with pytest.raises(ValidationError, match="conflicts"):
        JobSubmissionRequest(service="image", model="qwen-image-2.1", task="image.control_depth", inputs={"control_type": "pose"})


def test_internal_canonical_job_does_not_resolve_aliases_again(monkeypatch):
    from factory_contracts.jobs import job_contract
    data = {"job_id": "job-00000000-0000-0000-0000-000000000003",
        "request_id": "req-00000000-0000-0000-0000-000000000003", "service": "image",
        "model": "qwen-image-2.1", "task": "image.text_to_image", "inputs": {}}
    def forbidden(*args, **kwargs):
        raise AssertionError("Internal canonical job attempted alias resolution")
    monkeypatch.setattr(job_contract, "normalize_task", forbidden)
    assert FactoryJob.model_validate(data).task == "image.text_to_image"


@pytest.mark.asyncio
async def test_pose_only_worker_does_not_lease_depth_job(legacy_routing):
    repo = InMemoryJobStore()
    await repo.initialize()
    try:
        job = FactoryJob(job_id="job-00000000-0000-0000-0000-000000000002",
            request_id="req-00000000-0000-0000-0000-000000000002", service="image",
            model="qwen-image-2.1", task="image.control_depth", inputs={}, state=JobState.QUEUED)
        await repo.create_job(job)
        # This software-only test uses an explicit legacy contract fixture for
        # declared pose/depth support; it does not claim native implementation.
        coordinator = LeaseCoordinator(repo, manifest_loader=ManifestLoader(ROOT / "tests/fixtures/legacy_manifests"))
        result = await coordinator.lease_job(WorkerLeaseRequest(worker_id="pose-contract", subsystem="image",
            supported_models=[job.model], supported_tasks=["control_pose"], gpu_device=0))
        assert result.job is None
        assert (await repo.get_job(job.job_id)).state == JobState.QUEUED
    finally:
        await repo.close()


def test_gate_rejects_duplicate_alias_owners_and_dispatch_normalization():
    assert audit_source('TASK_ALIASES = {"t2i": "text_to_image"}', "services/example/worker.py")
    assert audit_source('from factory_contracts.tasks import normalize_task as resolve\nresolve("t2i")', "services/example/worker.py")
    assert not audit_source('NATIVE_TASKS = {"music.cover": "cover"}', "services/example/adapter.py")
