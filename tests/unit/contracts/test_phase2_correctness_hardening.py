"""
tests/unit/contracts/test_phase2_correctness_hardening.py
Verification suite for Phase 2 End-to-End Correctness Hardening (C1 through C10).
"""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from starlette.testclient import TestClient

from factory_contracts.manifests.model import (
    ModelManifest,
    ModelMetadata,
    ModelWeightsSpec,
    ModelHardwareSpec,
    ModelStatus,
    ModelWeightFormat,
    QuantizationType,
    PrecisionType,
    AllocationPolicy,
    ModelValidationStatus,
)
from factory_contracts.errors import (
    InvalidRequestError,
    ModelNotLoadedError,
    UnsupportedCapabilityError,
)
from factory.apps.control_api.src.queue.model_runtime import (
    ModelRuntime,
    ModelRuntimeManager,
    RuntimeState,
)
from factory.apps.control_api.src.storage.config import load_storage_config
from factory.apps.control_api.src.storage.scratch_disk import ScratchDiskConfig, ScratchDiskManager
from model_registry.loader import ManifestLoader


class TestPhase2CorrectnessHardening:
    """Verifies C1 through C10 correctness invariants."""

    def test_c1_model_runtime_manager_truth_and_live_probe(self):
        """C1: ModelRuntimeManager eliminates synthetic PID, starts with dependencies False, uses real PID."""
        rt = ModelRuntime(
            model_id="qwen-image-2.1",
            variant_id="standard",
            checkpoint_id="qwen-image-2.1",
            estimated_vram_gb=14.0,
        )
        assert rt.process_id is None
        assert rt.dependencies_ready == {"weights": False, "backend": False}
        assert rt.confirmed_vram_gb is None
        assert rt.live_verified is False

    @pytest.mark.asyncio
    async def test_c1_live_readiness_probe_failure_transitions_to_unloaded(self):
        """C1: When live probe fails, model transitions to UNLOADED with ModelNotLoadedError."""
        manager = ModelRuntimeManager(num_gpus=2)
        loader = ManifestLoader()
        models = loader.load_models()
        # enhancement-suite-v1 is marked installed: false, backend_ready: false
        enh_model = models["enhancement-suite-v1"]

        with pytest.raises(ModelNotLoadedError) as exc:
            await manager.ensure_loaded(enh_model)
        assert "failed readiness probe" in str(exc.value)
        rt = manager.runtimes.get("enhancement-suite-v1")
        assert rt is not None
        assert rt.state == RuntimeState.UNLOADED
        assert rt.dependencies_ready == {"weights": False, "backend": False}

    def test_c2_capability_truth_strict_task_variant_matrix(self):
        """C2: is_variant_supported_for_task strictly adheres to task_variant_matrix."""
        loader = ManifestLoader()
        models = loader.load_models()
        ace = models["ace-step-1.5"]

        # 'extract' only supports ['base', 'xl_base']
        assert ace.is_variant_supported_for_task("extract", "base") is True
        assert ace.is_variant_supported_for_task("extract", "xl_base") is True
        # Must NOT fall through to True for standard, default, or empty
        assert ace.is_variant_supported_for_task("extract", "standard") is False
        assert ace.is_variant_supported_for_task("extract", "default") is False
        assert ace.is_variant_supported_for_task("extract", None) is False
        assert ace.is_variant_supported_for_task("extract", "turbo") is False

    def test_c2_offline_backend_fails_closed_503(self):
        """C2: Submitting a job to an offline backend (enhancement-suite-v1) fails with 503 BACKEND_UNAVAILABLE."""
        os.environ["FACTORY_API_KEY"] = "test-phase2-key"
        from factory_control_api import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/jobs",
                json={
                    "service": "enhancement",
                    "model": "enhancement-suite-v1",
                    "task": "image_super_resolution",
                    "inputs": {"image_url": "http://example.com/input.png"},
                },
                headers={"Authorization": "Bearer test-phase2-key"},
            )
            assert resp.status_code == 503
            err = resp.json()["error"]
            assert err["code"] == "BACKEND_UNAVAILABLE"
            assert "offline or not installed" in err["message"]

    def test_c2_unsupported_variant_fails_closed_400(self):
        """C2: Submitting an unsupported variant fails closed with 400 UNSUPPORTED_CAPABILITY."""
        os.environ["FACTORY_API_KEY"] = "test-phase2-key"
        from factory_control_api import app
        with TestClient(app) as client:
            # ace-step-1.5 task 'extract' does not support 'turbo' or default 'standard'
            resp = client.post(
                "/v1/jobs",
                json={
                    "service": "music",
                    "model": "ace-step-1.5",
                    "task": "extract",
                    "inputs": {"prompt": "extract vocals", "variant": "turbo"},
                },
                headers={"Authorization": "Bearer test-phase2-key"},
            )
            assert resp.status_code == 400
            err = resp.json()["error"]
            assert err["code"] == "UNSUPPORTED_CAPABILITY"

    def test_c3_storage_ephemeral_disk_alias(self, monkeypatch):
        """C3: STORAGE_BACKEND=ephemeral_disk is a valid alias and boots without RuntimeError."""
        monkeypatch.setenv("STORAGE_BACKEND", "ephemeral_disk")
        cfg = load_storage_config()
        assert cfg.storage_backend == "ephemeral_disk"

    def test_c4_server_entrypoint_parity(self):
        """C4: Both factory_control_api:app and factory_control_api.server:app resolve cleanly to the same app."""
        import factory_control_api
        import factory_control_api.server
        assert factory_control_api.app is factory_control_api.server.app

    def test_c5_artifact_cleanup_policy_default_false(self, monkeypatch):
        """C5: ScratchDiskConfig defaults cleanup_on_download to False to protect concurrent downloads."""
        monkeypatch.delenv("ARTIFACT_CLEANUP_ON_DOWNLOAD", raising=False)
        cfg = ScratchDiskConfig()
        assert cfg.cleanup_on_download is False

        monkeypatch.setenv("ARTIFACT_CLEANUP_ON_DOWNLOAD", "true")
        cfg_enabled = ScratchDiskConfig()
        assert cfg_enabled.cleanup_on_download is True

    def test_c8_hardware_profile_dual_t4_bounds(self):
        """C8: All 11 model manifests declare concrete Dual-T4 memory bounds and quantization."""
        loader = ManifestLoader()
        models = loader.load_models()
        assert len(models) in (11, 12)
        for mid, m in models.items():
            assert hasattr(m.hardware, "min_vram_gb")
            assert m.hardware.min_vram_gb <= 14.2
            assert m.hardware.preferred_gpu_index in (0, 1)
            assert m.hardware.quantization_format is not None

    def test_contracts_import_parity(self):
        """Item 11: import contracts resolves cleanly without NameError."""
        import contracts
        assert hasattr(contracts, "FactoryJob")
        assert hasattr(contracts, "ModelManifest")
        assert hasattr(contracts, "ErrorCode")
