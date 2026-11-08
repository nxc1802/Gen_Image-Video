"""
tests/unit/contracts/test_phase2_adversarial_c1_c2.py
Adversarial stress-test suite for Phase 2 Live Engine Probe and Capability Admission.

Probes:
- C1: Live Engine Probe Offline Fallback (closed port, HTTP 500/503, REQUIRE_LIVE_BACKENDS, zero fake PID, zero dummy outputs)
- C2: Capability Truth & Variant Routing Boundaries (invalid variants, strict task matrices, unimplemented truth capabilities, VRAM > 14.2 GB bounds)
"""
import os
import socket
from unittest.mock import patch, MagicMock
import pytest
from starlette.testclient import TestClient

from factory_contracts.errors import (
    ErrorCode,
    FactoryError,
    InsufficientVRAMError,
    InvalidRequestError,
    ModelNotLoadedError,
    UnsupportedCapabilityError,
)
from factory_contracts.jobs.job_contract import FactoryJob, JobState, JobSubmissionRequest, JobProgress
from factory_contracts.manifests.model import (
    AllocationPolicy,
    CapabilityTruthLevel,
    ModelHardwareSpec,
    ModelManifest,
    ModelMetadata,
    ModelStatus,
    ModelValidationStatus,
    ModelWeightFormat,
    ModelWeightsSpec,
    ModelVariantSpec,
    PrecisionType,
    QuantizationType,
)
from factory.apps.control_api.src.queue.model_runtime import (
    ModelRuntime,
    ModelRuntimeManager,
    RuntimeState,
)
from model_registry.loader import ManifestLoader


def get_free_closed_port() -> int:
    """Finds a free port on localhost and immediately closes it so it is guaranteed offline."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestPhase2AdversarialC1EngineProbe:
    """Adversarial stress tests for C1: Live Engine Probe & Offline Fallback."""

    @pytest.mark.asyncio
    async def test_c1_offline_closed_port_never_claims_ready(self, monkeypatch):
        """
        Verify that when an engine backend port is closed / offline,
        ModelRuntimeManager NEVER claims READY, transitions to UNLOADED,
        raises ModelNotLoadedError, and sets process_id to None (zero fake PID).
        """
        closed_port = get_free_closed_port()
        monkeypatch.setenv("COMFYUI_URL", f"http://127.0.0.1:{closed_port}")

        manager = ModelRuntimeManager(num_gpus=2)
        loader = ManifestLoader()
        models = loader.load_models()
        qwen_img = models["qwen-image-2.1"]

        with pytest.raises(ModelNotLoadedError) as exc_info:
            await manager.ensure_loaded(qwen_img)

        assert "failed readiness probe" in str(exc_info.value)
        rt = manager.runtimes.get("qwen-image-2.1")
        assert rt is not None
        assert rt.state == RuntimeState.UNLOADED
        assert rt.process_id is None, "process_id must be None on offline backend (zero fake PID generation)"
        assert rt.dependencies_ready == {"weights": False, "backend": False}
        assert rt.live_verified is False
        assert rt.confirmed_vram_gb is None
        assert rt.error_message is not None

    @pytest.mark.asyncio
    async def test_c1_offline_http_500_503_never_claims_ready(self, monkeypatch):
        """
        Verify that when upstream backend URL returns HTTP 500 or 503,
        ModelRuntimeManager NEVER claims READY, transitions to UNLOADED,
        and raises ModelNotLoadedError.
        """
        monkeypatch.setenv("VLLM_TEXT_URL", "http://127.0.0.1:9099")

        class MockResp:
            status_code = 503

        class MockClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def get(self, url):
                return MockResp()

        manager = ModelRuntimeManager(num_gpus=2)
        loader = ManifestLoader()
        models = loader.load_models()
        qwen_text = models["qwen-text"]

        with patch("httpx.AsyncClient", return_value=MockClient()):
            with pytest.raises(ModelNotLoadedError) as exc_info:
                await manager.ensure_loaded(qwen_text)

        assert "failed readiness probe" in str(exc_info.value)
        assert "503" in str(exc_info.value)
        rt = manager.runtimes.get("qwen-text")
        assert rt is not None
        assert rt.state == RuntimeState.UNLOADED
        assert rt.process_id is None
        assert rt.dependencies_ready == {"weights": False, "backend": False}

    @pytest.mark.asyncio
    async def test_c1_require_live_backends_without_url_raises_not_loaded(self, monkeypatch):
        """
        Verify that when REQUIRE_LIVE_BACKENDS=true and no backend URL is set,
        readiness probe fails and ModelRuntimeManager transitions to UNLOADED.
        """
        monkeypatch.setenv("REQUIRE_LIVE_BACKENDS", "true")
        monkeypatch.delenv("VLLM_TEXT_URL", raising=False)

        manager = ModelRuntimeManager(num_gpus=2)
        loader = ManifestLoader()
        models = loader.load_models()
        qwen_text = models["qwen-text"]

        with pytest.raises(ModelNotLoadedError) as exc_info:
            await manager.ensure_loaded(qwen_text)

        assert "No backend service URL configured" in str(exc_info.value)
        rt = manager.runtimes.get("qwen-text")
        assert rt is not None
        assert rt.state == RuntimeState.UNLOADED
        assert rt.process_id is None

    def test_c1_job_submission_offline_engine_fails_closed_503(self):
        """
        Verify jobs submitted for an unready/offline engine receive HTTP 503 BACKEND_UNAVAILABLE
        and zero synthetic fallback / dummy outputs are produced.
        """
        os.environ["FACTORY_API_KEY"] = "test-phase2-challenger-key"
        from factory_control_api import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/jobs",
                json={
                    "service": "enhancement",
                    "model": "enhancement-suite-v1",
                    "task": "image_super_resolution",
                    "inputs": {"image_url": "http://example.com/test.png"},
                },
                headers={"Authorization": "Bearer test-phase2-challenger-key"},
            )
            assert resp.status_code == 503
            err = resp.json()["error"]
            assert err["code"] == "BACKEND_UNAVAILABLE"
            assert "offline or not installed" in err["message"]
            assert err["details"]["backend_ready"] is False
            assert err["details"]["installed"] is False

    def test_c1_job_result_failed_state_no_dummy_fallback(self):
        """
        Verify that querying the result of a failed job strictly returns an HTTP error
        and does not synthesize a dummy artifact.
        """
        os.environ["FACTORY_API_KEY"] = "test-phase2-challenger-key"
        from factory_control_api import app
        from factory_contracts.jobs.job_contract import FactoryJob, JobState, JobProgress
        from factory_contracts.errors import ErrorDetail, ErrorCode

        with TestClient(app) as client:
            # We obtain the lifecycle manager from app state
            manager = app.state.job_manager
            import uuid
            uid = str(uuid.uuid4())
            job_id = f"job-{uid}"
            req_id = f"req-{uid}"
            failed_job = FactoryJob(
                job_id=job_id,
                request_id=req_id,
                service="image",
                model="qwen-image-2.1",
                task="text_to_image",
                inputs={"prompt": "test prompt"},
                state=JobState.FAILED,
                error=ErrorDetail(
                    request_id=req_id,
                    code=ErrorCode.MODEL_UNAVAILABLE,
                    message="Model failed live engine readiness probe: connection refused.",
                ),
                progress=JobProgress(stage="failed", step=0, total=10, percent=0.0),
            )
            # Store into repository
            import asyncio
            asyncio.run(manager.repository.create_job(failed_job))

            resp = client.get(
                f"/v1/jobs/{job_id}/result",
                headers={"Authorization": "Bearer test-phase2-challenger-key"},
            )
            assert resp.status_code == 503
            err = resp.json()["error"]
            assert err["code"] == "MODEL_UNAVAILABLE"
            assert "connection refused" in err["message"]


class TestPhase2AdversarialC2CapabilityAdmission:
    """Adversarial stress tests for C2: Capability Truth & Variant Routing Boundaries."""

    def test_c2_wan22_invalid_variant_rejected_400(self):
        """
        Adversarially request 'experimental' variant on Wan 2.2.
        Must be strictly rejected with HTTP 400 UNSUPPORTED_CAPABILITY and NOT fall through to standard.
        """
        os.environ["FACTORY_API_KEY"] = "test-phase2-challenger-key"
        from factory_control_api import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/jobs",
                json={
                    "service": "video",
                    "model": "wan-2.2",
                    "task": "text_to_video",
                    "inputs": {"prompt": "cinematic drone footage", "variant": "experimental"},
                },
                headers={"Authorization": "Bearer test-phase2-challenger-key"},
            )
            assert resp.status_code == 400
            err = resp.json()["error"]
            assert err["code"] == "UNSUPPORTED_CAPABILITY"
            assert "experimental" in err["message"]
            assert err["details"]["error_code"] == "UNSUPPORTED_CAPABILITY"

    def test_c2_wan22_turbo_variant_rejected_400(self):
        """
        Wan 2.2 does not support turbo variant. Must be strictly rejected with HTTP 400 UNSUPPORTED_CAPABILITY.
        """
        os.environ["FACTORY_API_KEY"] = "test-phase2-challenger-key"
        from factory_control_api import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/jobs",
                json={
                    "service": "video",
                    "model": "wan-2.2",
                    "task": "text_to_video",
                    "inputs": {"prompt": "ocean waves", "variant": "turbo"},
                },
                headers={"Authorization": "Bearer test-phase2-challenger-key"},
            )
            assert resp.status_code == 400
            err = resp.json()["error"]
            assert err["code"] == "UNSUPPORTED_CAPABILITY"

    def test_c2_minimax_h3_turbo_variant_rejected_400(self):
        """
        MiniMax H3 only supports standard. Turbo must be strictly rejected with HTTP 400 UNSUPPORTED_CAPABILITY.
        """
        os.environ["FACTORY_API_KEY"] = "test-phase2-challenger-key"
        from factory_control_api import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/jobs",
                json={
                    "service": "video",
                    "model": "minimax-h3",
                    "task": "text_to_video",
                    "inputs": {"prompt": "hyperrealistic waterfall", "variant": "turbo"},
                },
                headers={"Authorization": "Bearer test-phase2-challenger-key"},
            )
            assert resp.status_code == 400
            err = resp.json()["error"]
            assert err["code"] == "UNSUPPORTED_CAPABILITY"

    def test_c2_qwen_image_turbo_on_inpaint_rejected_400(self):
        """
        Qwen-Image 2.1 supports turbo for text_to_image, but inpaint matrix is standard-only.
        Asking for turbo on inpaint must be strictly rejected with HTTP 400 UNSUPPORTED_CAPABILITY.
        """
        os.environ["FACTORY_API_KEY"] = "test-phase2-challenger-key"
        from factory_control_api import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/jobs",
                json={
                    "service": "image",
                    "model": "qwen-image-2.1",
                    "task": "inpaint",
                    "inputs": {
                        "prompt": "repair background",
                        "image_url": "http://example.com/source.png",
                        "mask_url": "http://example.com/mask.png",
                        "variant": "turbo",
                    },
                },
                headers={"Authorization": "Bearer test-phase2-challenger-key"},
            )
            assert resp.status_code == 400
            err = resp.json()["error"]
            assert err["code"] == "UNSUPPORTED_CAPABILITY"

    def test_c2_ace_step_extract_task_matrix_boundaries(self):
        """
        ACE-Step 1.5 task 'extract' task_variant_matrix: ['base', 'xl_base'].
        Must strictly reject:
        - turbo (HTTP 400)
        - sft (HTTP 400)
        - standard (HTTP 400)
        - default (HTTP 400)
        - omitted/empty variant (HTTP 400)
        And must admit:
        - base (HTTP 201)
        - xl_base (HTTP 201)
        """
        os.environ["FACTORY_API_KEY"] = "test-phase2-challenger-key"
        from factory_control_api import app
        with TestClient(app) as client:
            # 1. turbo -> 400
            r1 = client.post(
                "/v1/jobs",
                json={
                    "service": "music",
                    "model": "ace-step-1.5",
                    "task": "extract",
                    "inputs": {"audio_url": "http://example.com/in.wav", "variant": "turbo"},
                },
                headers={"Authorization": "Bearer test-phase2-challenger-key"},
            )
            assert r1.status_code == 400
            assert r1.json()["error"]["code"] == "UNSUPPORTED_CAPABILITY"

            # 2. sft -> 400
            r2 = client.post(
                "/v1/jobs",
                json={
                    "service": "music",
                    "model": "ace-step-1.5",
                    "task": "extract",
                    "inputs": {"audio_url": "http://example.com/in.wav", "variant": "sft"},
                },
                headers={"Authorization": "Bearer test-phase2-challenger-key"},
            )
            assert r2.status_code == 400
            assert r2.json()["error"]["code"] == "UNSUPPORTED_CAPABILITY"

            # 3. standard -> 400
            r3 = client.post(
                "/v1/jobs",
                json={
                    "service": "music",
                    "model": "ace-step-1.5",
                    "task": "extract",
                    "inputs": {"audio_url": "http://example.com/in.wav", "variant": "standard"},
                },
                headers={"Authorization": "Bearer test-phase2-challenger-key"},
            )
            assert r3.status_code == 400
            assert r3.json()["error"]["code"] == "UNSUPPORTED_CAPABILITY"

            # 4. omitted variant (defaults to standard) -> 400
            r4 = client.post(
                "/v1/jobs",
                json={
                    "service": "music",
                    "model": "ace-step-1.5",
                    "task": "extract",
                    "inputs": {"audio_url": "http://example.com/in.wav"},
                },
                headers={"Authorization": "Bearer test-phase2-challenger-key"},
            )
            assert r4.status_code == 400
            assert r4.json()["error"]["code"] == "UNSUPPORTED_CAPABILITY"

            # 5. base -> 201 (admitted)
            r5 = client.post(
                "/v1/jobs",
                json={
                    "service": "music",
                    "model": "ace-step-1.5",
                    "task": "extract",
                    "inputs": {"audio_url": "http://example.com/in.wav", "variant": "base"},
                },
                headers={"Authorization": "Bearer test-phase2-challenger-key"},
            )
            assert r5.status_code == 201

            # 6. xl_base -> 201 (admitted)
            r6 = client.post(
                "/v1/jobs",
                json={
                    "service": "music",
                    "model": "ace-step-1.5",
                    "task": "extract",
                    "inputs": {"audio_url": "http://example.com/in.wav", "variant": "xl_base"},
                },
                headers={"Authorization": "Bearer test-phase2-challenger-key"},
            )
            assert r6.status_code == 201

    def test_c2_unimplemented_capability_truth_fails_closed_400(self):
        """
        Verify that a capability declared in manifest capabilities but implemented: false in capabilities_truth
        (e.g. video_extension in minimax-h3) is strictly rejected with HTTP 400.
        """
        os.environ["FACTORY_API_KEY"] = "test-phase2-challenger-key"
        from factory_control_api import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/jobs",
                json={
                    "service": "video",
                    "model": "minimax-h3",
                    "task": "video_extension",
                    "inputs": {"video_url": "http://example.com/source.mp4"},
                },
                headers={"Authorization": "Bearer test-phase2-challenger-key"},
            )
            assert resp.status_code == 400
            err = resp.json()["error"]
            assert err["code"] in ("CAPABILITY_NOT_SUPPORTED", "UNSUPPORTED_CAPABILITY")

    def test_c2_declared_vram_exceeding_14_2_gb_rejected_400(self):
        """
        Adversarially probe pre-admission hardware feasibility check:
        Any model or variant requiring > 14.2 GB VRAM must be rejected with HTTP 400 InvalidRequestError.
        """
        os.environ["FACTORY_API_KEY"] = "test-phase2-challenger-key"
        from factory_control_api import app
        from model_registry.loader import ManifestLoader

        # Patch loader to return a model declaring 16.0 GB VRAM
        loader = ManifestLoader()
        models = loader.load_models()
        mock_model = models["qwen-text"].model_copy(deep=True)
        mock_model.hardware.min_vram_gb = 16.0

        with patch.object(ManifestLoader, "load_models", return_value={**models, "qwen-text-heavy": mock_model}):
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/jobs",
                    json={
                        "service": "text",
                        "model": "qwen-text-heavy",
                        "task": "chat",
                        "inputs": {"messages": [{"role": "user", "content": "hello"}]},
                    },
                    headers={"Authorization": "Bearer test-phase2-challenger-key"},
                )
                assert resp.status_code == 400
                err = resp.json()["error"]
                assert "exceeds max allocatable VRAM of 14.2 GB" in err["message"]

    def test_c2_variant_vram_exceeding_14_2_gb_rejected_400(self):
        """
        Adversarially probe variant VRAM check:
        If a variant requires > 14.2 GB VRAM, pre-admission validation rejects it with HTTP 400.
        """
        os.environ["FACTORY_API_KEY"] = "test-phase2-challenger-key"
        from factory_control_api import app
        from model_registry.loader import ManifestLoader

        loader = ManifestLoader()
        models = loader.load_models()
        mock_ace = models["ace-step-1.5"].model_copy(deep=True)
        # Add an oversized variant requiring 15.0 GB VRAM
        mock_ace.weights.variants["oversized"] = ModelVariantSpec(
            id="oversized",
            display_name="Oversized Music Variant",
            checkpoint="ACE-Studio/ACE-Step-1.5-Oversized",
            min_vram_gb=15.0,
        )
        if mock_ace.task_variant_matrix:
            mock_ace.task_variant_matrix["text_to_music"].append("oversized")

        with patch.object(ManifestLoader, "load_models", return_value={**models, "ace-step-1.5": mock_ace}):
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/jobs",
                    json={
                        "service": "music",
                        "model": "ace-step-1.5",
                        "task": "text_to_music",
                        "inputs": {"prompt": "epic cinematic symphony", "variant": "oversized"},
                    },
                    headers={"Authorization": "Bearer test-phase2-challenger-key"},
                )
                assert resp.status_code == 400
                err = resp.json()["error"]
                assert "exceeds max allocatable VRAM of 14.2 GB" in err["message"]

    @pytest.mark.asyncio
    async def test_c2_model_runtime_manager_vram_boundary(self):
        """
        Verify that ModelRuntimeManager.ensure_loaded rejects per_gpu_req > 14.2 GB
        with InsufficientVRAMError.
        """
        manager = ModelRuntimeManager(num_gpus=2, max_allocatable_gb_per_gpu=14.2)
        loader = ManifestLoader()
        models = loader.load_models()
        mock_model = models["qwen-image-2.1"].model_copy(deep=True)
        mock_model.hardware.min_vram_gb = 15.0
        mock_model.hardware.allocation_policy = AllocationPolicy.SINGLE_GPU

        with pytest.raises(InsufficientVRAMError) as exc_info:
            await manager.ensure_loaded(mock_model, gpu_ids=[0])

        assert "exceeding max allocatable VRAM of 14.2 GB" in str(exc_info.value)
