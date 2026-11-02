"""
tests/contract/test_contract_chain.py
Contract tests verifying the end-to-end contract chain:
Capability Manifest -> Studio form submission -> POST /v1/jobs -> worker eligibility -> workflow registry.
"""
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock

from model_registry.loader import ManifestLoader
from factory_contracts.tasks import (
    CANONICAL_MODEL_MAP,
    CANONICAL_TASK_MAP,
    CANONICAL_TO_CAPABILITY_MAP,
    MODEL_TO_FAMILY_MAP,
    get_model_aliases,
    get_task_aliases,
    model_to_family,
    normalize_model,
    normalize_task,
    task_to_capability_id,
)
from factory_contracts.jobs import JobSubmissionRequest, JobState
from factory_contracts.jobs.worker_contract import WorkerLeaseRequest
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.broker.memory_broker import MemoryQueueBroker
from factory.apps.control_api.src.queue.repository.sqlite import SqliteJobRepository
from factory.apps.control_api.src.streaming.manager import StreamManager
from workflows.image.registry import ImageWorkflowRegistry
from workflows.video.registry import VideoWorkflowRegistry

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MANIFESTS_DIR = ROOT_DIR / "manifests"


@pytest.fixture
def manifest_loader():
    return ManifestLoader(MANIFESTS_DIR)


@pytest.mark.tier1
class TestCanonicalIdentifierParity:
    """Contract tests verifying bidirectional resolution between capability IDs, models, and tasks."""

    def test_all_capability_manifests_mapped_in_canonical_task_map(self, manifest_loader):
        """Every capability declared in manifests/capabilities/ must exist in CANONICAL_TASK_MAP."""
        caps = manifest_loader.load_capabilities()
        assert len(caps) > 0, "No capability manifests found"

        for cap_id, cap in caps.items():
            assert cap_id in CANONICAL_TASK_MAP, f"Capability '{cap_id}' missing from CANONICAL_TASK_MAP"
            canonical_task = normalize_task(cap_id)
            assert canonical_task != "", f"Canonical task for '{cap_id}' must not be empty"
            # Bidirectional resolution
            primary_cap = task_to_capability_id(canonical_task)
            assert primary_cap is not None, f"Canonical task '{canonical_task}' has no primary capability ID mapping"

    def test_all_model_manifests_mapped_in_canonical_model_map(self, manifest_loader):
        """Every model manifest in manifests/models/ must resolve canonically."""
        models = manifest_loader.load_models()
        assert len(models) > 0, "No model manifests found"

        for model_id, model_manifest in models.items():
            normalized = normalize_model(model_id)
            assert normalized == model_id, f"Model ID '{model_id}' should be its own canonical name"
            family = model_to_family(model_id)
            assert family is not None, f"Model '{model_id}' must map to a model family"

            # Family name must also resolve to a canonical model
            family_norm = normalize_model(family)
            assert family_norm == model_id or family_norm in models

    def test_model_and_task_aliases_bidirectional(self):
        """Verify alias lookup helpers return valid non-empty lists."""
        image_aliases = get_task_aliases("text_to_image")
        assert "image.t2i" in image_aliases
        assert "t2i" in image_aliases
        assert "text_to_image" in image_aliases

        wan_aliases = get_model_aliases("wan-2.2")
        assert "wan" in wan_aliases
        assert "wan-2.2" in wan_aliases


@pytest.mark.tier1
class TestEndToEndContractChain:
    """Verifies complete chain: CapabilityManifest -> Studio form -> POST /v1/jobs -> Worker -> Workflow."""

    @pytest.mark.asyncio
    async def test_image_chain_from_manifest_to_workflow(self, manifest_loader):
        """Validates image.t2i end-to-end contract chain."""
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        await broker.initialize()
        stream_mgr = StreamManager()
        manager = JobLifecycleManager(repo, broker, stream_mgr)
        coordinator = LeaseCoordinator(repo, broker)
        image_registry = ImageWorkflowRegistry()

        try:
            # 1. Load Capability Manifest
            caps = manifest_loader.load_capabilities()
            t2i_cap = caps["image.t2i"]
            assert t2i_cap.subsystem == "image"

            # 2. Simulate Studio Form Submission (submitting capability ID)
            studio_submission = JobSubmissionRequest(
                service=t2i_cap.subsystem,
                model="qwen-image",  # Studio may supply family or model alias
                task=t2i_cap.id,      # Studio supplies capability ID "image.t2i"
                inputs={"prompt": "Studio form submission prompt", "width": 1024, "height": 1024},
            )

            # 3. POST /v1/jobs via JobLifecycleManager
            job, created = await manager.submit_job(studio_submission, request_id="req-contract-image-01")
            assert created is True
            # Database stores canonical identifiers!
            assert job.model == "qwen-image-2.1"
            assert job.task == "text_to_image"
            assert job.state == JobState.QUEUED

            # 4. Worker Eligibility & Lease Coordination
            worker_lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-image-contract",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                )
            )
            assert worker_lease.job is not None
            assert worker_lease.job.job_id == job.job_id
            assert worker_lease.lease_token is not None

            # 5. Workflow Registry Execution
            workflow = image_registry.get_workflow(job.task)
            assert isinstance(workflow, dict)
            assert len(workflow) > 0
            # Workflow graph has ComfyUI nodes
            assert any(node.get("class_type") == "KSampler" for node in workflow.values())
        finally:
            await repo.close()

    @pytest.mark.asyncio
    async def test_video_chain_from_manifest_to_workflow(self, manifest_loader):
        """Validates video.t2v end-to-end contract chain with Wan2.2 and MiniMax H3."""
        repo = SqliteJobRepository(db_path=":memory:")
        await repo.initialize()
        broker = MemoryQueueBroker()
        await broker.initialize()
        stream_mgr = StreamManager()
        manager = JobLifecycleManager(repo, broker, stream_mgr)
        coordinator = LeaseCoordinator(repo, broker)
        video_registry = VideoWorkflowRegistry()

        try:
            caps = manifest_loader.load_capabilities()
            t2v_cap = caps["video.t2v"]
            assert t2v_cap.subsystem == "video"

            # 1. Wan-2.2 submission
            sub_wan = JobSubmissionRequest(
                service="video",
                model="wan",
                task=t2v_cap.id,
                inputs={"prompt": "Majestic mountain flight", "num_frames": 49},
            )
            job_wan, _ = await manager.submit_job(sub_wan, request_id="req-contract-video-wan")
            assert job_wan.model == "wan-2.2"
            assert job_wan.task == "text_to_video"

            # 2. Worker Leases Wan job
            lease_wan = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-video-wan",
                    subsystem="video",
                    supported_models=["wan-2.2"],
                    supported_tasks=["text_to_video"],
                )
            )
            assert lease_wan.job is not None
            assert lease_wan.job.job_id == job_wan.job_id

            # 3. Workflow resolution
            wf_wan = video_registry.get_workflow_for_task(job_wan.model, job_wan.task)
            assert isinstance(wf_wan, dict)
            assert len(wf_wan) > 0

            # 4. MiniMax H3 submission
            sub_minimax = JobSubmissionRequest(
                service="video",
                model="minimax",
                task="video.with_audio",
                inputs={"prompt": "Orchestral concert with live audio"},
            )
            job_mm, _ = await manager.submit_job(sub_minimax, request_id="req-contract-video-mm")
            assert job_mm.model == "minimax-h3"
            assert job_mm.task == "video_with_audio"

            lease_mm = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-video-minimax",
                    subsystem="video",
                    supported_models=["minimax-h3"],
                    supported_tasks=["video_with_audio"],
                )
            )
            assert lease_mm.job is not None
            assert lease_mm.job.job_id == job_mm.job_id

            wf_mm = video_registry.get_workflow_for_task(job_mm.model, job_mm.task)
            assert isinstance(wf_mm, dict)
            assert len(wf_mm) > 0
        finally:
            await repo.close()

    def test_cross_model_workflow_fallback_forbidden(self):
        """Cross-model workflow fallback (e.g. attempting video_with_audio on Wan2.2) is strictly forbidden."""
        video_registry = VideoWorkflowRegistry()
        with pytest.raises(KeyError, match="not implemented for model 'wan-2.2'"):
            video_registry.get_workflow_for_task(model="wan-2.2", task="video_with_audio")
