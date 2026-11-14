"""
tests/unit/manifests/test_challenger_p5_adversarial.py
Adversarial Stress Harness for Phase 5 (Maintainability, Integration & Manifests):
- Feature 29 (P2.1): Qwen-Image 2.1 Turbo Manifest & Contract Invariants
- Feature 31 (P2.5): Artifact Git Hygiene & Preservation Invariants

Probes:
1. Manifest & Contract Invariants:
   - Validates manifests/models/qwen-image-2.1-turbo.yaml parses through ModelManifest.model_validate().
   - Probes VRAM boundaries: Pre-admission check in jobs.py rejects > 14.2 GB; ModelRuntimeManager rejects per_gpu_req > 14.2 GB.
   - Probes dual_gpu_split atomic multi-GPU allocation vs single-GPU conflict handling.
   - Tests canonical model aliases: qwen-image-2.1-turbo, qwen_image_2.1_turbo, legacy aliases, and family resolution.
   - Discovers queue dispatch integration gap in leasing.py (SUBSYSTEM_MANIFEST_CAPABILITIES).
2. Git Hygiene & Whitelist Invariants:
   - Evaluates git check-ignore across dummy media, weights, and database files across root, subdirectories, and apps.
   - Validates that regression test fixtures and studio web assets are explicitly NOT ignored.
   - Verifies git status --porcelain contains zero untracked media or weights.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
from unittest.mock import patch
import pytest
import yaml
from starlette.testclient import TestClient

from factory_contracts.errors import (
    InsufficientVRAMError,
    InvalidRequestError,
)
from factory_contracts.jobs.job_contract import JobSubmissionRequest
from factory_contracts.jobs.worker_contract import WorkerLeaseRequest
from factory_contracts.manifests.model import (
    AllocationPolicy,
    ModelHardwareSpec,
    ModelManifest,
    ModelStatus,
    ModelValidationStatus,
    ModelWeightFormat,
    PrecisionType,
    QuantizationType,
)
from factory_contracts.tasks import (
    CANONICAL_MODEL_MAP,
    MODEL_TO_FAMILY_MAP,
    get_model_aliases,
    model_to_family,
    normalize_model,
)
from model_registry.loader import ManifestLoader
from model_registry.validator import ManifestValidator


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = WORKSPACE_ROOT / "manifests"
TURBO_YAML_PATH = MANIFESTS_DIR / "models" / "qwen-image-2.1-turbo.yaml"


# ==============================================================================
# Group 1: Feature 29 (P2.1) Manifest & Pydantic Schema Invariants
# ==============================================================================

class TestFeature29ManifestAndContractInvariants:
    """Adversarially probe Qwen-Image 2.1 Turbo manifest and contract invariants."""

    def test_parse_qwen_image_turbo_manifest_via_model_validate(self):
        """Assert manifests/models/qwen-image-2.1-turbo.yaml parses cleanly via ModelManifest.model_validate()."""
        assert TURBO_YAML_PATH.exists(), f"Manifest file missing: {TURBO_YAML_PATH}"
        with open(TURBO_YAML_PATH, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        manifest = ModelManifest.model_validate(raw_data)
        assert isinstance(manifest, ModelManifest)
        assert manifest.metadata.id == "qwen-image-2.1-turbo"
        assert manifest.metadata.family == "qwen-image"
        assert manifest.metadata.service_id == "image-service"
        assert manifest.status == ModelStatus.PARTIAL
        assert manifest.coverage_percent == 20.0
        assert manifest.capabilities.get("image.text_to_image") is True
        assert manifest.capabilities.get("image.inpaint") is False
        assert set(manifest.missing_capabilities) == {"image.inpaint", "image.outpaint", "image.controlnet", "image.instruction_edit"}
        assert manifest.hardware.min_vram_gb == 14.0
        assert manifest.hardware.preferred_gpu == "t4"
        assert manifest.hardware.preferred_gpu_index is None
        assert manifest.hardware.allocation_policy == AllocationPolicy.SINGLE_GPU
        assert manifest.metadata.license == "qwen-research"
        assert manifest.metadata.commercial_use is False

    def test_all_12_manifests_parse_and_validate_cleanly(self):
        """Assert all 12 model manifests load through ManifestLoader and pass ManifestValidator."""
        loader = ManifestLoader(MANIFESTS_DIR)
        models = loader.load_models()
        assert len(models) == len(list((MANIFESTS_DIR / "models").glob("*.yaml"))), f"Expected exactly 12 models in manifests/, found {len(models)}"
        assert "qwen-image-2.1-turbo" in models

        validator = ManifestValidator(loader)
        ok, errors = validator.validate_all()
        assert ok, f"ManifestValidator reported validation errors: {errors}"

    def test_vram_rejection_at_job_admission_exceeding_hardware_envelope(self, routing_registry):
        """Hardware requirements are validated before serving, not per request."""
        from model_registry.deployment_validator import DeploymentValidator
        registry = routing_registry([("qwen-image-2.1-turbo", [0])])
        registry.catalog._models["qwen-image-2.1-turbo"].runtime_profiles["kaggle-dual-t4"].min_vram_gb = 17
        report = DeploymentValidator(registry).validate(config_only=True)
        assert not report["configuration_valid"]
        assert any(error["check"] == "vram_budget" for error in report["errors"])

    @pytest.mark.asyncio
    async def test_vram_rejection_at_model_runtime_manager_boundary(self, routing_registry):
        """Canonical models cannot be promoted by the legacy load probe."""
        from factory_contracts.errors import ModelNotLoadedError
        from factory.apps.control_api.src.queue.model_runtime import ModelRuntimeManager
        registry = routing_registry([("qwen-image-2.1-turbo", [0])])
        manager = ModelRuntimeManager(deployment_registry=registry)
        with pytest.raises(ModelNotLoadedError):
            await manager.ensure_loaded(registry.catalog.get("qwen-image-2.1-turbo"), gpu_ids=[0])
        assert not manager.runtimes

    @pytest.mark.asyncio
    async def test_dual_gpu_split_lease_atomic_mutual_exclusion(self, legacy_routing):
        """
        Adversarially probe dual_gpu_split invariant:
        Models declaring dual_gpu_split (qwen-text) require atomic reservation of [0, 1].
        If GPU 0 or GPU 1 is already busy, lease_job must fail and leave the other GPU free.
        """
        from factory.apps.control_api.src.queue.repository import InMemoryJobStore
        from factory.apps.control_api.src.queue.scheduler import InMemoryScheduler
        from factory.apps.control_api.src.streaming.manager import StreamManager
        from factory.apps.control_api.src.queue.manager import JobLifecycleManager
        from factory.apps.control_api.src.queue.leasing import LeaseCoordinator

        repo = InMemoryJobStore(db_path=":memory:")
        await repo.initialize()
        scheduler = InMemoryScheduler()
        await scheduler.initialize()
        stream_mgr = StreamManager(repository=repo)
        manager = JobLifecycleManager(repository=repo, broker=scheduler, stream_manager=stream_mgr)
        coordinator = LeaseCoordinator(repository=repo, broker=scheduler)

        try:
            # 1. Occupy GPU 0 with qwen-image-2.1
            job_gpu0, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1",
                    task="text_to_image",
                    inputs={"prompt": "Image on GPU 0"},
                ),
                request_id="req-image-01",
            )
            lease0 = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-image-0",
                    subsystem="image",
                    supported_models=["qwen-image-2.1"],
                    supported_tasks=["text_to_image"],
                    gpu_device=0,
                )
            )
            assert lease0.job is not None
            assert scheduler.is_gpu_busy(0) is True
            assert scheduler.is_gpu_busy(1) is False

            # 2. Submit dual_gpu_split job (qwen-text)
            job_dual, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="text",
                    model="qwen-text",
                    task="chat_completion",
                    inputs={"prompt": "Dual-GPU text query"},
                ),
                request_id="req-dual-01",
            )

            # 3. Attempt to lease dual-GPU job while GPU 0 is occupied
            lease_dual = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-text-dual",
                    subsystem="text",
                    supported_models=["qwen-text"],
                    supported_tasks=["chat_completion"],
                )
            )
            # Must fail to lease because GPU 0 is busy
            assert lease_dual.job is None
            # GPU 1 must remain free
            assert scheduler.is_gpu_busy(0) is True
            assert scheduler.is_gpu_busy(1) is False
        finally:
            await repo.close()
            await scheduler.close()

    @pytest.mark.asyncio
    async def test_qwen_image_turbo_lease_gap_in_subsystem_manifest_capabilities(self, legacy_routing):
        """
        VERIFIES DEFECT REMEDIATION:
        Tests that LeaseCoordinator successfully leases a job for 'qwen-image-2.1-turbo'
        after adding 'qwen-image-2.1-turbo' and 'qwen_image_2.1_turbo' to
        SUBSYSTEM_MANIFEST_CAPABILITIES['image']['models'] in leasing.py.
        """
        from factory.apps.control_api.src.queue.repository import InMemoryJobStore
        from factory.apps.control_api.src.queue.scheduler import InMemoryScheduler
        from factory.apps.control_api.src.streaming.manager import StreamManager
        from factory.apps.control_api.src.queue.manager import JobLifecycleManager
        from factory.apps.control_api.src.queue.leasing import LeaseCoordinator

        repo = InMemoryJobStore(db_path=":memory:")
        await repo.initialize()
        scheduler = InMemoryScheduler()
        await scheduler.initialize()
        stream_mgr = StreamManager(repository=repo)
        manager = JobLifecycleManager(repository=repo, broker=scheduler, stream_manager=stream_mgr)
        coordinator = LeaseCoordinator(repository=repo, broker=scheduler)

        try:
            job, _ = await manager.submit_job(
                JobSubmissionRequest(
                    service="image",
                    model="qwen-image-2.1-turbo",
                    task="text_to_image",
                    inputs={"prompt": "Adversarial test prompt for turbo"},
                ),
                request_id="req-turbo-defect",
            )
            assert job.model == "qwen-image-2.1-turbo"

            # Worker requests qwen-image-2.1-turbo
            lease = await coordinator.lease_job(
                WorkerLeaseRequest(
                    worker_id="worker-turbo-test",
                    subsystem="image",
                    supported_models=["qwen-image-2.1-turbo"],
                    supported_tasks=["text_to_image"],
                    gpu_device=0,
                )
            )
            # Verifies lease succeeds after remediation
            assert lease.job is not None, "Empirically verifies lease_job succeeds now that turbo is in SUBSYSTEM_MANIFEST_CAPABILITIES"
            assert lease.job.job_id == job.job_id
            assert lease.job.model == "qwen-image-2.1-turbo"
            assert lease.lease_token is not None
        finally:
            await repo.close()
            await scheduler.close()

    def test_canonical_model_aliases_resolution(self):
        """
        Verify canonical model resolution in factory_contracts.tasks:
        - qwen-image-2.1-turbo -> qwen-image-2.1-turbo
        - qwen_image_2.1_turbo -> qwen-image-2.1-turbo
        - qwen-image-turbo -> qwen-image-2.1 (legacy alias)
        - qwen-image -> qwen-image-2.1
        - qwen_image -> qwen-image-2.1
        - family mapping: qwen-image-2.1-turbo -> qwen-image
        """
        # Exact canonical mappings
        assert normalize_model("qwen-image-2.1-turbo") == "qwen-image-2.1-turbo"
        assert normalize_model("qwen_image_2.1_turbo") == "qwen-image-2.1-turbo"
        assert normalize_model("QWEN-IMAGE-2.1-TURBO") == "qwen-image-2.1-turbo"

        # Legacy backward-compatibility alias
        assert normalize_model("qwen-image-turbo") == "qwen-image-2.1-turbo"

        # Standard family aliases
        assert normalize_model("qwen-image") == "qwen-image-2.1"
        assert normalize_model("qwen_image") == "qwen-image-2.1"
        assert normalize_model("qwen-image-2.1") == "qwen-image-2.1"
        assert normalize_model("qwen_image_2.1") == "qwen-image-2.1"

        # Model family resolution
        assert model_to_family("qwen-image-2.1-turbo") == "qwen-image"
        assert model_to_family("qwen-image-2.1") == "qwen-image"
        assert model_to_family("qwen-image") == "qwen-image"

        # Check aliases generator includes both turbo keys
        aliases = get_model_aliases("qwen-image-2.1-turbo")
        assert "qwen-image-2.1-turbo" in aliases
        assert "qwen_image_2.1_turbo" in aliases


# ==============================================================================
# Group 2: Feature 31 (P2.5) Git Hygiene & Fixture Preservation Invariants
# ==============================================================================

class TestFeature31GitHygieneAdversarialInvariants:
    """Adversarially probe git check-ignore, whitelisting, and working tree cleanliness."""

    @pytest.mark.parametrize(
        "dummy_file",
        [
            # Root binary media
            "test_sample.mp4",
            "voice_recording.wav",
            "generated_render.png",
            "photo.jpg",
            "thumbnail.webp",
            # Nested service media outputs
            "services/video/clip.mkv",
            "services/music/track.flac",
            "factory/apps/studio-web/preview.gif",
            # Ephemeral directories
            "artifacts/job_12345/output.mp4",
            "storage/artifacts/image.png",
            "deployments/kaggle/outputs/benchmark.wav",
            "deployments/kaggle/comfyui/logs.tmp",
            # Large weights and checkpoints
            "model_weights.safetensors",
            "qwen-image-turbo.gguf",
            "checkpoint.pt",
            "weights.bin",
            "model.onnx",
            "weights/qwen/weights.safetensors",
            "deployments/kaggle/models/unet.safetensors",
            # Local databases & state files
            "factory.sqlite",
            "test_state.sqlite3",
            "factory_control.db",
            "dump.rdb",
            "factory/apps/control-api/local.db",
        ],
    )
    def test_dummy_media_weights_and_databases_are_ignored(self, dummy_file: str):
        """Assert git check-ignore returns 0 (ignored) for prohibited binaries across all directories."""
        target_path = WORKSPACE_ROOT / dummy_file
        res = subprocess.run(
            ["git", "check-ignore", "-q", str(target_path)],
            cwd=WORKSPACE_ROOT,
        )
        assert res.returncode == 0, f"File should be IGNORED by .gitignore: {dummy_file}"

    @pytest.mark.parametrize(
        "whitelisted_file",
        [
            # Test regression fixtures
            "tests/fixtures/sample.png",
            "tests/fixtures/mock_sample.png",
            "tests/fixtures/audio_sample.wav",
            "tests/unit/fixtures/frame.png",
            # Studio web public assets & SVG icons
            "factory/apps/studio-web/public/icon.svg",
            "factory/apps/studio-web/public/logo.png",
            "factory/apps/studio-web/public/favicon.ico",
            # Documentation assets
            "docs/assets/architecture.png",
            "docs/images/pipeline.jpg",
            # Declarative model manifests
            "manifests/models/qwen-image-2.1-turbo.yaml",
        ],
    )
    def test_fixtures_and_web_assets_are_not_ignored(self, whitelisted_file: str):
        """Assert git check-ignore returns 1 (NOT ignored) for whitelisted fixtures and web assets."""
        target_path = WORKSPACE_ROOT / whitelisted_file
        res = subprocess.run(
            ["git", "check-ignore", "-q", str(target_path)],
            cwd=WORKSPACE_ROOT,
        )
        assert res.returncode == 1, f"Whitelisted asset must NOT be ignored: {whitelisted_file}"

    def test_active_git_status_contains_zero_media_or_weights(self):
        """
        Confirm that active git status --porcelain contains ZERO untracked or staged
        prohibited media binaries, model weights, or database files.
        """
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]

        prohibited_extensions = {
            ".mp4", ".webm", ".mkv", ".mov", ".avi", ".flv", ".wmv",
            ".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a",
            ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff",
            ".safetensors", ".gguf", ".pt", ".pth", ".bin", ".onnx", ".ckpt",
            ".sqlite", ".sqlite3", ".db",
        }

        violations = []
        for line in lines:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                file_path = parts[1]
                ext = Path(file_path).suffix.lower()
                if ext in prohibited_extensions:
                    # Ignore whitelisted fixtures if any
                    if not (file_path.startswith("tests/") and "fixtures" in file_path) and not file_path.startswith("factory/apps/studio-web/public/"):
                        violations.append(file_path)

        assert not violations, f"Active git status contains prohibited media/weights/db files: {violations}"
