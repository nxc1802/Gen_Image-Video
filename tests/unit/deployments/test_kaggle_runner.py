"""
tests/unit/deployments/test_kaggle_runner.py
Comprehensive Unit & Contract Test Suite for Kaggle Dual-T4 Runner Infrastructure
and 5 Sequential Certification Gates.
"""
import io
import json
import math
from pathlib import Path
import struct
from unittest.mock import MagicMock, patch
import urllib.error
import pytest

from deployments.kaggle.runner import (
    resolve_kaggle_username,
    resolve_kaggle_api_token,
    KaggleAPIClient,
    KaggleAPIError,
    KaggleAuthError,
    KaggleDualT4Runner,
    KaggleKernelFailedError,
    KaggleTimeoutError,
)
from deployments.kaggle.certify_kernel import (
    DualT4HardwareTopology,
    CertificationGates,
    FourStageVerificationGate,
    HardwareMemoryProfiler,
    MemoryProfileSnapshot,
    PreflightContractSimulator,
    KaggleLiveKernelExecutor,
    Tier1ModelCertifier,
    TIER_1_PROFILES,
)


# =====================================================================
# 1. Kaggle REST API v1 Client Tests
# =====================================================================

class TestKaggleAPIClient:
    """Verifies Kaggle REST API v1 client token handling and operations."""

    def test_bearer_token_resolution_from_env_kaggle_api_token(self, monkeypatch):
        monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_standard_env_token_456")
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        client = KaggleAPIClient()
        assert client.api_key == "KGAT_standard_env_token_456"
        assert client.get_auth_header() == "Bearer KGAT_standard_env_token_456"

    def test_bearer_token_legacy_fallback_to_kaggle_key(self, monkeypatch):
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
        monkeypatch.setenv("KAGGLE_KEY", "KGAT_legacy_env_key_789")
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        client = KaggleAPIClient()
        assert client.api_key == "KGAT_legacy_env_key_789"
        assert client.get_auth_header() == "Bearer KGAT_legacy_env_key_789"

    def test_explicit_api_key_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_env_token")
        client = KaggleAPIClient(username="testuser", api_key="KGAT_explicit_param")
        assert client.api_key == "KGAT_explicit_param"
        assert client.get_auth_header() == "Bearer KGAT_explicit_param"

    def test_missing_api_key_raises_kaggle_auth_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
        with pytest.raises(KaggleAuthError) as exc_info:
            KaggleAPIClient(username="testuser")
        assert "Kaggle API key/token not configured" in str(exc_info.value)

    def test_basic_auth_fallback_for_legacy_key(self):
        client = KaggleAPIClient(username="testuser", api_key="plain_hex_key_12345")
        auth_header = client.get_auth_header()
        assert auth_header.startswith("Basic ")
        # Decodes to testuser:plain_hex_key_12345
        import base64
        decoded = base64.b64decode(auth_header.split(" ")[1]).decode("ascii")
        assert decoded == "testuser:plain_hex_key_12345"

    @patch("urllib.request.urlopen")
    def test_push_kernel_payload_structure(self, mock_urlopen):
        # Mock successful push response
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "hasError": False,
            "versionNumber": 1,
            "url": "https://www.kaggle.com/code/factory-certifier/test-kernel",
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        client = KaggleAPIClient(username="cert-user", api_key="KGAT_testtoken123")
        res = client.push_kernel(
            slug="test-cert-slug",
            title="Factory Live Cert",
            code_content="print('cert')",
            is_private=True,
            enable_gpu=True,
            enable_internet=True,
            session_timeout_seconds=7200,
        )

        assert res.get("versionNumber") == 1
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        assert req.get_method() == "POST"
        assert req.full_url == "https://www.kaggle.com/api/v1/kernels/push"
        assert req.headers["Authorization"] == "Bearer KGAT_testtoken123"

        payload = json.loads(req.data.decode("utf-8"))
        assert payload["slug"] == "cert-user/test-cert-slug"
        assert payload["enableGpu"] is True
        assert payload["enableInternet"] is True
        assert payload["sessionTimeoutSeconds"] >= 7200

    @patch("urllib.request.urlopen")
    def test_push_kernel_api_error_response(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "hasError": True,
            "error": "Kernel quota exceeded on Kaggle",
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        client = KaggleAPIClient(username="testuser", api_key="KGAT_mocktoken123")
        with pytest.raises(KaggleAPIError) as exc_info:
            client.push_kernel("slug", "Title", "code")
        assert "Kernel quota exceeded" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    def test_http_error_handling_401_auth_error(self, mock_urlopen):
        http_error = urllib.error.HTTPError(
            url="https://www.kaggle.com/api/v1/kernels/push",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=io.BytesIO(b'{"message": "Invalid API token"}'),
        )
        mock_urlopen.side_effect = http_error

        client = KaggleAPIClient(username="testuser", api_key="KGAT_mocktoken123")
        with pytest.raises(KaggleAuthError) as exc_info:
            client.push_kernel("slug", "Title", "code")
        assert exc_info.value.status_code == 401

    @patch("urllib.request.urlopen")
    def test_http_error_handling_500_api_error(self, mock_urlopen):
        http_error = urllib.error.HTTPError(
            url="https://www.kaggle.com/api/v1/kernels/status",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=io.BytesIO(b"Backend service failure"),
        )
        mock_urlopen.side_effect = http_error

        client = KaggleAPIClient(username="testuser", api_key="KGAT_mocktoken123")
        with pytest.raises(KaggleAPIError) as exc_info:
            client.get_kernel_status("my-slug")
        assert exc_info.value.status_code == 500

    @patch("urllib.request.urlopen")
    def test_poll_until_complete_stream_logs(self, mock_urlopen):
        # Sequence: queued -> running -> complete
        status_responses = [
            json.dumps({"status": "queued"}).encode("utf-8"),
            json.dumps({"status": "running"}).encode("utf-8"),
            json.dumps({"status": "complete"}).encode("utf-8"),
        ]
        output_responses = [
            json.dumps({"log": "Init\n"}).encode("utf-8"),
            json.dumps({"log": "Init\nStep 1 pass\n"}).encode("utf-8"),
            json.dumps({"log": "Init\nStep 1 pass\nDone\n"}).encode("utf-8"),
        ]

        # Alternating status and output calls
        side_effects = []
        for s, o in zip(status_responses, output_responses):
            m_s = MagicMock()
            m_s.read.return_value = s
            ctx_s = MagicMock()
            ctx_s.__enter__.return_value = m_s

            m_o = MagicMock()
            m_o.read.return_value = o
            ctx_o = MagicMock()
            ctx_o.__enter__.return_value = m_o

            side_effects.extend([ctx_s, ctx_o])

        mock_urlopen.side_effect = side_effects

        logged_lines = []
        client = KaggleAPIClient(username="testuser", api_key="KGAT_mocktoken123")
        res = client.poll_until_complete(
            slug="test-slug",
            poll_interval_sec=0.01,
            timeout_sec=5.0,
            output_callback=lambda l: logged_lines.append(l),
        )

        assert "Done" in res.get("log", "")
        assert "Step 1 pass" in logged_lines

    @patch("urllib.request.urlopen")
    def test_poll_until_complete_error_raises_kernel_failed(self, mock_urlopen):
        m_s = MagicMock()
        m_s.read.return_value = json.dumps({
            "status": "error",
            "failureMessage": "Out of memory during model initialization",
        }).encode("utf-8")
        ctx_s = MagicMock()
        ctx_s.__enter__.return_value = m_s

        m_o = MagicMock()
        m_o.read.return_value = json.dumps({"log": "CUDA OOM error"}).encode("utf-8")
        ctx_o = MagicMock()
        ctx_o.__enter__.return_value = m_o

        mock_urlopen.side_effect = [ctx_s, ctx_o]

        client = KaggleAPIClient(username="testuser", api_key="KGAT_mocktoken123")
        with pytest.raises(KaggleKernelFailedError) as exc_info:
            client.poll_until_complete("oom-slug", poll_interval_sec=0.01)
        assert "Out of memory" in str(exc_info.value.failure_message)


# =====================================================================
# 2. Dual-T4 Hardware Topology & Exclusive Admission Tests
# =====================================================================

class TestDualT4HardwareTopology:
    """Verifies GPU routing, limits, and mutual exclusion profiles."""

    def test_assigned_gpu_routing(self):
        topology = DualT4HardwareTopology()
        # GPU 0 Models
        assert topology.get_assigned_gpu("qwen-text") == 0
        assert topology.get_assigned_gpu("qwen-decision") == 0
        assert topology.get_assigned_gpu("qwen-vlm") == 0
        assert topology.get_assigned_gpu("qwen-image-2.1") == 0
        assert topology.get_assigned_gpu("qwen-speech") == 0

        # GPU 1 Models
        assert topology.get_assigned_gpu("qwen-embedding") == 1
        assert topology.get_assigned_gpu("wan-2.2") == 1
        assert topology.get_assigned_gpu("minimax-h3") == 1
        assert topology.get_assigned_gpu("ace-step-1.5") == 1
        assert topology.get_assigned_gpu("enhancement-suite-v1") == 1

        # Class-level model and service lists
        assert "speech-service" in DualT4HardwareTopology.GPU_0_SERVICES
        assert "enhancement-service" in DualT4HardwareTopology.GPU_1_SERVICES
        assert "qwen-speech" in DualT4HardwareTopology.GPU_0_MODELS
        assert "ace-step-1.5" in DualT4HardwareTopology.GPU_1_MODELS
        assert "enhancement-suite-v1" in DualT4HardwareTopology.GPU_1_MODELS

    def test_assigned_gpu_routing_all_ten_models_matching_hardware_manifest(self):
        """Cross-checks all 10 model manifests against manifests/hardware/kaggle-dual-t4.yaml."""
        import yaml

        hw_file = Path("manifests/hardware/kaggle-dual-t4.yaml")
        assert hw_file.exists(), f"Missing hardware manifest: {hw_file}"
        with open(hw_file, "r", encoding="utf-8") as f:
            hw_manifest = yaml.safe_load(f)

        service_to_gpu = {}
        for gpu in hw_manifest["gpus"]:
            gpu_idx = gpu["index"]
            for s in gpu["assigned_services"]:
                service_to_gpu[s] = gpu_idx

        models_dir = Path("manifests/models")
        model_files = sorted(models_dir.glob("*.yaml"))
        assert len(model_files) == 10, f"Expected 10 model manifests, found {len(model_files)}"

        topology = DualT4HardwareTopology()
        for mf in model_files:
            with open(mf, "r", encoding="utf-8") as f:
                m_data = yaml.safe_load(f)
            meta = m_data.get("metadata", m_data)
            model_id = meta.get("id") or m_data.get("id")
            service = meta.get("service_id") or meta.get("service") or m_data.get("service")
            assert service in service_to_gpu, f"Unknown service '{service}' in {mf}"
            expected_gpu = service_to_gpu[service]
            actual_gpu = topology.get_assigned_gpu(model_id)
            assert actual_gpu == expected_gpu, (
                f"Model {model_id} (service {service}) expected on GPU {expected_gpu}, "
                f"but get_assigned_gpu returned {actual_gpu}"
            )


    def test_exclusive_admission_conflict_on_same_gpu(self):
        topology = DualT4HardwareTopology()
        # Admit qwen-text (12.0 GB) on GPU 0
        topology.admit_model(0, "qwen-text", 12000.0)

        # Attempt to admit qwen-image-2.1 (14.0 GB) simultaneously on GPU 0
        ok, reason = topology.check_exclusive_admission(0, "qwen-image-2.1", 14000.0)
        assert ok is False
        assert "Exclusive admission conflict on GPU 0" in reason

        # Trying to admit raises RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            topology.admit_model(0, "qwen-image-2.1", 14000.0)
        assert "exceeds limit" in str(exc_info.value)

    def test_admit_and_evict_lifecycle(self):
        topology = DualT4HardwareTopology()
        topology.admit_model(0, "qwen-text", 12000.0)
        # Evict qwen-text
        topology.evict_model(0, "qwen-text")
        # Now qwen-image-2.1 can be admitted
        topology.admit_model(0, "qwen-image-2.1", 14000.0)
        assert topology._active_models[0] == "qwen-image-2.1"


# =====================================================================
# 3. 5 Sequential Certification Gates Tests
# =====================================================================

class TestCertificationGates:
    """Tests each of the 5 sequential gates against genuine pass/fail criteria."""

    def test_gate_1_load_gate_success(self):
        res = CertificationGates.verify_load_gate(
            model_id="qwen-text",
            gpu_index=0,
            base_cuda_context_mb=820.0,
            weights_mb=4500.0,
            load_time_sec=4.2,
        )
        assert res.passed is True
        assert res.details["static_footprint_mb"] == 5320.0
        assert res.details["headroom_mb"] == 10040.0

    def test_gate_1_load_gate_exceeds_limit(self):
        res = CertificationGates.verify_load_gate(
            model_id="too-heavy-model",
            gpu_index=0,
            base_cuda_context_mb=850.0,
            weights_mb=14000.0,  # 14850 MB > 14540 MB limit
            load_time_sec=10.0,
        )
        assert res.passed is False
        assert "exceeds Gate 1 ceiling" in res.error_message

    def test_gate_1_load_gate_driver_context_anomaly(self):
        res = CertificationGates.verify_load_gate(
            model_id="qwen-text",
            gpu_index=0,
            base_cuda_context_mb=200.0,  # too low, uninitialized context
            weights_mb=4500.0,
            load_time_sec=4.0,
        )
        assert res.passed is False
        assert "CUDA driver context overhead out of range" in res.error_message

    def test_gate_2_inference_gate_peak_vram_exceeded(self):
        res = CertificationGates.verify_inference_gate(
            model_id="qwen-text",
            modality="text",
            output_payload="valid output",
            peak_vram_mb=15500.0,  # > 15360 MB ceiling
            latency_ms=200.0,
        )
        assert res.passed is False
        assert "exceeded 15.0 GB ceiling" in res.error_message

    def test_gate_2_text_numerical_correctness(self):
        # Empty string fails
        fail_res = CertificationGates.verify_inference_gate(
            model_id="qwen-text",
            modality="text",
            output_payload="   ",
            peak_vram_mb=12000.0,
            latency_ms=180.0,
        )
        assert fail_res.passed is False
        assert "Empty token stream" in fail_res.error_message

        # Valid text succeeds
        pass_res = CertificationGates.verify_inference_gate(
            model_id="qwen-text",
            modality="text",
            output_payload="Generated valid reasoning stream.",
            peak_vram_mb=12000.0,
            latency_ms=180.0,
        )
        assert pass_res.passed is True
        assert "artifact_sha256" in pass_res.details

    def test_gate_2_decision_numerical_correctness(self):
        # NaN fails
        fail_res = CertificationGates.verify_inference_gate(
            model_id="qwen-decision",
            modality="decision",
            output_payload={"predicate": True, "score": float("nan")},
            peak_vram_mb=5800.0,
            latency_ms=120.0,
        )
        assert fail_res.passed is False
        assert "Invalid numerical value" in fail_res.error_message

        # Valid dict succeeds
        pass_res = CertificationGates.verify_inference_gate(
            model_id="qwen-decision",
            modality="decision",
            output_payload={"predicate": True, "choice": "route_a", "score": 0.99},
            peak_vram_mb=5800.0,
            latency_ms=120.0,
        )
        assert pass_res.passed is True

    def test_gate_2_embedding_unit_l2_norm(self):
        # Unnormalized vector fails
        unnormalized = [1.0, 2.0, 3.0]
        fail_res = CertificationGates.verify_inference_gate(
            model_id="qwen-embedding",
            modality="embedding",
            output_payload=unnormalized,
            peak_vram_mb=3700.0,
            latency_ms=50.0,
        )
        assert fail_res.passed is False
        assert "violates unit L2 norm" in fail_res.error_message

        # Normalized vector succeeds
        norm = math.sqrt(sum(x ** 2 for x in unnormalized))
        normalized = [x / norm for x in unnormalized]
        pass_res = CertificationGates.verify_inference_gate(
            model_id="qwen-embedding",
            modality="embedding",
            output_payload=normalized,
            peak_vram_mb=3700.0,
            latency_ms=50.0,
        )
        assert pass_res.passed is True

    def test_gate_2_image_png_signature_and_ihdr(self):
        # Invalid signature fails
        fail_res1 = CertificationGates.verify_inference_gate(
            model_id="qwen-image-2.1",
            modality="image",
            output_payload=b"NOT_A_PNG_HEADER",
            peak_vram_mb=14100.0,
            latency_ms=800.0,
        )
        assert fail_res1.passed is False
        assert "valid 8-byte PNG header" in fail_res1.error_message

        # Missing IHDR chunk fails
        fake_png_no_ihdr = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 30)
        fail_res2 = CertificationGates.verify_inference_gate(
            model_id="qwen-image-2.1",
            modality="image",
            output_payload=fake_png_no_ihdr,
            peak_vram_mb=14100.0,
            latency_ms=800.0,
        )
        assert fail_res2.passed is False
        assert "missing mandatory IHDR chunk" in fail_res2.error_message

        # Valid PNG header + IHDR chunk succeeds
        valid_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x02\x00\x00\x00\x02\x00\x08\x06\x00\x00\x00\x11\x22\x33\x44"
            b"\x00\x00\x00\x04IDATxxxx\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        pass_res = CertificationGates.verify_inference_gate(
            model_id="qwen-image-2.1",
            modality="image",
            output_payload=valid_png,
            peak_vram_mb=14100.0,
            latency_ms=800.0,
        )
        assert pass_res.passed is True

    def test_gate_2_video_mp4_container_structure(self):
        # Missing ftyp box fails
        fail_res1 = CertificationGates.verify_inference_gate(
            model_id="wan-2.2",
            modality="video",
            output_payload=b"NOT_AN_MP4_FILE_HERE_AT_ALL_1234567890",
            peak_vram_mb=14200.0,
            latency_ms=2500.0,
        )
        assert fail_res1.passed is False
        assert "missing valid MP4 'ftyp' box" in fail_res1.error_message

        # Valid MP4 with ftyp + moov + mdat succeeds
        valid_mp4 = (
            struct.pack(">I", 24) + b"ftyp" + b"isom" + struct.pack(">I", 512) + b"isomiso2"
            + struct.pack(">I", 16) + b"moov" + b"video_moov_box"
            + struct.pack(">I", 32) + b"mdat" + (b"video_frame_bytes" * 2)
        )
        pass_res = CertificationGates.verify_inference_gate(
            model_id="wan-2.2",
            modality="video",
            output_payload=valid_mp4,
            peak_vram_mb=14200.0,
            latency_ms=2500.0,
        )
        assert pass_res.passed is True

    def test_gate_3_contract_gate_full_adherence(self, tmp_path):
        manifest_file = tmp_path / "test-model.yaml"
        manifest_file.write_text(
            """
apiVersion: factory.ai/v1
kind: ModelManifest
metadata:
  id: test-model
capabilities:
  feat_a: true
  feat_b: false
"""
        )

        def _executor(cap):
            if cap == "feat_a":
                return True, 200, None
            if cap == "feat_b":
                return False, 400, None
            return False, 404, None

        res = CertificationGates.verify_contract_gate(manifest_file, _executor)
        assert res.passed is True
        assert res.details["adherence_percent"] == 100.0

    def test_gate_3_contract_gate_unsupported_capability_leak_fails(self, tmp_path):
        manifest_file = tmp_path / "test-model-fail.yaml"
        manifest_file.write_text(
            """
apiVersion: factory.ai/v1
kind: ModelManifest
metadata:
  id: test-model-fail
capabilities:
  feat_unsupported: false
"""
        )

        # Unsupported capability incorrectly returns 200 with fallback
        def _executor(cap):
            return True, 200, "other-model"

        res = CertificationGates.verify_contract_gate(manifest_file, _executor)
        assert res.passed is False
        assert "expected 400 Bad Request" in res.error_message

    def test_gate_4_stress_gate_zero_leak(self):
        # Leak under 50 MB passes
        def _runner_pass(idx):
            return True, 12000.0 + (idx * 2.0), 180.0

        pass_res = CertificationGates.verify_stress_gate("qwen-text", 5, _runner_pass)
        assert pass_res.passed is True
        assert pass_res.details["vram_leak_mb"] < 50.0

        # Excessive leak fails
        def _runner_leak(idx):
            return True, 12000.0 + (idx * 15.0), 180.0

        fail_res = CertificationGates.verify_stress_gate("qwen-text", 5, _runner_leak)
        assert fail_res.passed is False
        assert "Memory leak detected" in fail_res.error_message

    def test_gate_5_recovery_gate_sigkill_and_reboot(self):
        def _sim_clean():
            return {
                "sigkill_caught": True,
                "lease_reaped": True,
                "worker_rebooted": True,
                "cuda_context_clean": True,
            }

        pass_res = CertificationGates.verify_recovery_gate("qwen-text", _sim_clean)
        assert pass_res.passed is True

        def _sim_cuda_hang():
            return {
                "sigkill_caught": True,
                "lease_reaped": True,
                "worker_rebooted": True,
                "cuda_context_clean": False,  # hung context
            }

        fail_res = CertificationGates.verify_recovery_gate("qwen-text", _sim_cuda_hang)
        assert fail_res.passed is False
        assert "Recovery gate invariants failed" in fail_res.error_message


# =====================================================================
# 4. Tier 1 Model Certification Execution & Manifest Invariants
# =====================================================================

class TestTier1ModelCertification:
    """Verifies end-to-end certification across all 6 Tier 1 core models."""

    def test_certifier_executes_all_tier1_models(self, tmp_path):
        manifests_dir = Path(__file__).resolve().parent.parent.parent.parent / "manifests"
        certifier = Tier1ModelCertifier(manifests_dir=manifests_dir)
        results_file = tmp_path / "cert_test.json"

        results = certifier.execute_all(output_path=results_file)
        assert results["all_gates_passed"] is True
        assert len(results["models"]) == 6

        # Check required Tier 1 models
        expected_models = ["qwen-text", "qwen-embedding", "qwen-decision", "qwen-vlm", "qwen-image-2.1", "wan-2.2"]
        for m in expected_models:
            assert m in results["models"]
            assert results["models"][m]["certified"] is True
            # All 5 gates present and passed
            gates = results["models"][m]["gates"]
            assert gates["gate_1_load"]["passed"] is True
            assert gates["gate_2_inference"]["passed"] is True
            assert gates["gate_3_contract"]["passed"] is True
            assert gates["gate_4_stress"]["passed"] is True
            assert gates["gate_5_recovery"]["passed"] is True

    def test_production_certification_results_file_integrity(self):
        root_dir = Path(__file__).resolve().parent.parent.parent.parent
        results_path = root_dir / "deployments" / "kaggle" / "certification_results.json"
        assert results_path.exists(), f"Missing certification results file at {results_path}"

        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["all_gates_passed"] is True
        assert data["environment"] == "simulation-local"
        assert data.get("live_hardware_attested") is False
        assert "Simulated" in data["hardware"]["name"]

        models = data["models"]
        for m_id in ["qwen-text", "qwen-embedding", "qwen-decision", "qwen-vlm", "qwen-image-2.1", "wan-2.2"]:
            assert m_id in models
            m_data = models[m_id]
            assert m_data["certified"] is True
            assert m_data["profiling"]["peak_vram_mb"] < 15000.0

    def test_model_manifests_pending_live_anti_escalation(self):
        root_dir = Path(__file__).resolve().parent.parent.parent.parent
        from model_registry.loader import ManifestLoader

        loader = ManifestLoader(root_dir / "manifests")
        models = loader.load_models()

        tier1_manifest_keys = ["qwen-text", "qwen-embedding", "qwen-decision", "qwen-vlm", "qwen-image-2.1", "wan-2.2"]
        for k in tier1_manifest_keys:
            assert k in models, f"Model {k} not loaded from manifests"
            m = models[k]
            assert m.validation_status.value == "pending_live", f"Model {k} validation_status is {m.validation_status}"
            assert m.validation_notes is not None
            assert "Tesla T4" in m.validation_notes
            assert "pre-flight" in m.validation_notes.lower()
            assert "pending" in m.validation_notes.lower()


class TestHardwareMemoryProfilerAndSeparatedEngines:
    """Verifies HardwareMemoryProfiler integration and separated certification engines."""

    def test_hardware_memory_profiler_on_cpu_environment(self):
        """Verifies profiler on CPU/macOS safely returns UNAVAILABLE without mock fabrication."""
        profiler = HardwareMemoryProfiler(device_index=0)
        assert profiler.is_available is False
        assert profiler.telemetry_status == "UNAVAILABLE"
        assert profiler.query_live_memory_mb() is None

        snapshot = profiler.profile_stage("test_stage", base_context_mb=800.0, weights_mb=4000.0)
        assert snapshot.telemetry_status == "UNAVAILABLE"
        assert snapshot.source == "preflight_simulation"
        assert snapshot.live_vram_used_mb is None
        assert snapshot.peak_vram_mb == 4800.0

    @patch("pynvml.nvmlInit")
    @patch("pynvml.nvmlDeviceGetCount", return_value=2)
    @patch("pynvml.nvmlDeviceGetHandleByIndex")
    @patch("pynvml.nvmlDeviceGetMemoryInfo")
    def test_hardware_memory_profiler_with_nvml_mock(
        self, mock_mem_info, mock_get_handle, mock_get_count, mock_init
    ):
        """Verifies profiler samples genuine live NVML VRAM when pynvml is active."""
        mock_info = MagicMock()
        mock_info.used = 5368709120  # 5120 MiB
        mock_info.total = 16106127360  # 15360 MiB
        mock_mem_info.return_value = mock_info

        profiler = HardwareMemoryProfiler(device_index=0)
        assert profiler.is_available is True
        assert profiler.telemetry_status == "AVAILABLE"
        assert profiler.query_live_memory_mb() == 5120.0

        snapshot = profiler.profile_stage("live_stage")
        assert snapshot.telemetry_status == "AVAILABLE"
        assert snapshot.source == "live_nvml"
        assert snapshot.live_vram_used_mb == 5120.0

    def test_preflight_contract_simulator_execution(self, tmp_path):
        """Verifies PreflightContractSimulator marks environment as simulation-local and telemetry UNAVAILABLE."""
        manifests_dir = Path(__file__).resolve().parent.parent.parent.parent / "manifests"
        sim = PreflightContractSimulator(manifests_dir=manifests_dir)
        out_file = tmp_path / "sim_results.json"
        res = sim.execute_all(output_path=out_file)

        assert res["all_gates_passed"] is True
        assert res["environment"] == "simulation-local"
        assert res["hardware"]["telemetry_status"] == "UNAVAILABLE"
        for m_id, m_data in res["models"].items():
            assert m_data["telemetry_status"] == "UNAVAILABLE"
            assert m_data["profiling"]["source"] == "preflight_simulation"
            assert "snapshots" in m_data["profiling"]
            assert len(m_data["profiling"]["snapshots"]) >= 4

    def test_kaggle_live_executor_fail_closed_without_gpu(self):
        """Verifies KaggleLiveKernelExecutor refuses to execute on non-GPU environment."""
        with pytest.raises(RuntimeError) as exc_info:
            executor = KaggleLiveKernelExecutor()
            executor.execute_all()
        assert "requires physical NVIDIA GPUs" in str(exc_info.value)

    def test_tier1_model_certifier_mode_dispatch(self):
        """Verifies Tier1ModelCertifier dispatches correctly across modes."""
        c_sim = Tier1ModelCertifier(mode="preflight")
        assert c_sim._engine.__class__.__name__ == "PreflightContractSimulator"

        # On non-GPU Mac, mode="live" must raise RuntimeError
        with pytest.raises(RuntimeError):
            Tier1ModelCertifier(mode="live")


class TestKaggleDynamicUsernameResolution:
    """Verifies username resolution precedence: explicit > env > kaggle.json > fail fast."""

    def test_explicit_username_precedence(self):
        client = KaggleAPIClient(username="custom-user", api_key="KGAT_mocktoken")
        assert client.username == "custom-user"

    def test_env_var_username_precedence(self, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "env-certifier")
        monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_mocktoken")
        client = KaggleAPIClient()
        assert client.username == "env-certifier"

    def test_kaggle_json_dynamic_detection(self, tmp_path, monkeypatch):
        fake_kaggle_json = tmp_path / "kaggle.json"
        fake_kaggle_json.write_text(json.dumps({"username": "json-user", "key": "json-key"}))
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))

        client = KaggleAPIClient()
        assert client.username == "json-user"
        assert client.api_key == "json-key"

    def test_missing_username_raises_kaggle_auth_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))  # empty dir, no kaggle.json

        with pytest.raises(KaggleAuthError) as exc_info:
            resolve_kaggle_username()
        assert "Kaggle username not configured" in str(exc_info.value)


class TestGate2AssertionHardening:
    """Verifies hardened Gate 2 assertions for L2 tolerance, NaNs, PNG chunks, and MP4 boxes."""

    def test_embedding_tight_tolerance_rejection(self):
        # 1.0005 exceeds 1e-4 tolerance
        res = CertificationGates.verify_inference_gate(
            model_id="qwen-embedding",
            modality="embedding",
            output_payload=[1.0005, 0.0],
            peak_vram_mb=3700.0,
            latency_ms=50.0,
        )
        assert res.passed is False
        assert "violates unit L2 norm" in res.error_message

    def test_embedding_tight_tolerance_acceptance(self):
        # 1.00005 is within 1e-4 tolerance
        res = CertificationGates.verify_inference_gate(
            model_id="qwen-embedding",
            modality="embedding",
            output_payload=[1.00005, 0.0],
            peak_vram_mb=3700.0,
            latency_ms=50.0,
        )
        assert res.passed is True

    def test_embedding_nan_rejection(self):
        res = CertificationGates.verify_inference_gate(
            model_id="qwen-embedding",
            modality="embedding",
            output_payload=[float("nan"), 1.0],
            peak_vram_mb=3700.0,
            latency_ms=50.0,
        )
        assert res.passed is False
        assert "violates unit L2 norm" in res.error_message

    def test_png_rejects_truncated_payload(self):
        # 12-byte payload with IHDR magic substring but missing data/CRC
        res = CertificationGates.verify_inference_gate(
            model_id="qwen-image-2.1",
            modality="image",
            output_payload=b"\x89PNG\r\n\x1a\nIHDR",
            peak_vram_mb=14100.0,
            latency_ms=800.0,
        )
        assert res.passed is False
        assert "too short for valid PNG structure" in res.error_message

    def test_png_rejects_invalid_ihdr_length(self):
        # 34 bytes, but IHDR length declared as 10 instead of 13
        header = b"\x89PNG\r\n\x1a\n"
        invalid_ihdr = struct.pack(">I", 10) + b"IHDR" + (b"\x00" * 18)
        res = CertificationGates.verify_inference_gate(
            model_id="qwen-image-2.1",
            modality="image",
            output_payload=header + invalid_ihdr,
            peak_vram_mb=14100.0,
            latency_ms=800.0,
        )
        assert res.passed is False
        assert "invalid IHDR chunk" in res.error_message

    def test_mp4_rejects_missing_mdat(self):
        ftyp = struct.pack(">I", 24) + b"ftyp" + b"isom" + struct.pack(">I", 512) + b"isomiso2"
        moov = struct.pack(">I", 16) + b"moov" + b"video_moov_box"
        res = CertificationGates.verify_inference_gate(
            model_id="wan-2.2",
            modality="video",
            output_payload=ftyp + moov,
            peak_vram_mb=14200.0,
            latency_ms=2500.0,
        )
        assert res.passed is False
        assert "missing essential MP4 'moov' or 'mdat' boxes" in res.error_message

    def test_mp4_rejects_missing_moov(self):
        ftyp = struct.pack(">I", 24) + b"ftyp" + b"isom" + struct.pack(">I", 512) + b"isomiso2"
        mdat = struct.pack(">I", 32) + b"mdat" + (b"video_frame_bytes" * 2)
        res = CertificationGates.verify_inference_gate(
            model_id="wan-2.2",
            modality="video",
            output_payload=ftyp + mdat,
            peak_vram_mb=14200.0,
            latency_ms=2500.0,
        )
        assert res.passed is False
        assert "missing essential MP4 'moov' or 'mdat' boxes" in res.error_message

    def test_mp4_accepts_both_moov_and_mdat(self):
        ftyp = struct.pack(">I", 24) + b"ftyp" + b"isom" + struct.pack(">I", 512) + b"isomiso2"
        moov = struct.pack(">I", 16) + b"moov" + b"video_moov_box"
        mdat = struct.pack(">I", 32) + b"mdat" + (b"video_frame_bytes" * 2)
        res = CertificationGates.verify_inference_gate(
            model_id="wan-2.2",
            modality="video",
            output_payload=ftyp + moov + mdat,
            peak_vram_mb=14200.0,
            latency_ms=2500.0,
        )
        assert res.passed is True


# =====================================================================
# 8. 4-Stage Verification Gate Pipeline Tests
# =====================================================================

class TestFourStageVerificationGate:
    """Verifies the 4-Stage Verification Gate: Load -> Infer -> Validate -> Profile."""

    def test_stage_1_load_gpu_placement_success(self):
        topology = DualT4HardwareTopology()
        # qwen-text correctly on GPU 0
        res = FourStageVerificationGate.verify_stage_1_load(
            model_id="qwen-text",
            gpu_index=0,
            base_cuda_context_mb=820.0,
            weights_mb=4500.0,
            load_time_sec=4.2,
            topology=topology,
        )
        assert res.passed is True
        assert res.gate_name == "Stage 1: Load"
        assert res.details["static_footprint_mb"] == 5320.0
        assert res.details["headroom_mb"] == 10040.0

        # qwen-embedding correctly on GPU 1
        res_emb = FourStageVerificationGate.verify_stage_1_load(
            model_id="qwen-embedding",
            gpu_index=1,
            base_cuda_context_mb=750.0,
            weights_mb=1450.0,
            load_time_sec=2.1,
            topology=topology,
        )
        assert res_emb.passed is True

    def test_stage_1_load_gpu_placement_mismatch_fails(self):
        topology = DualT4HardwareTopology()
        # qwen-text wrongly placed on GPU 1
        res = FourStageVerificationGate.verify_stage_1_load(
            model_id="qwen-text",
            gpu_index=1,
            base_cuda_context_mb=820.0,
            weights_mb=4500.0,
            load_time_sec=4.2,
            topology=topology,
        )
        assert res.passed is False
        assert "hardware topology requires GPU 0" in res.error_message

        # wan-2.2 wrongly placed on GPU 0
        res_wan = FourStageVerificationGate.verify_stage_1_load(
            model_id="wan-2.2",
            gpu_index=0,
            base_cuda_context_mb=890.0,
            weights_mb=9800.0,
            load_time_sec=8.5,
            topology=topology,
        )
        assert res_wan.passed is False
        assert "hardware topology requires GPU 1" in res_wan.error_message

    def test_stage_1_load_music_and_enhancement_on_gpu_1_success(self):
        """Verifies that ace-step-1.5 and enhancement-suite-v1 pass Stage 1 Load when placed on GPU 1."""
        topology = DualT4HardwareTopology()
        # ace-step-1.5 correctly on GPU 1
        res_music = FourStageVerificationGate.verify_stage_1_load(
            model_id="ace-step-1.5",
            gpu_index=1,
            base_cuda_context_mb=800.0,
            weights_mb=4000.0,
            load_time_sec=3.0,
            topology=topology,
        )
        assert res_music.passed is True
        assert res_music.gate_name == "Stage 1: Load"
        assert res_music.details["gpu_index"] == 1
        assert res_music.details["static_footprint_mb"] == 4800.0

        # enhancement-suite-v1 correctly on GPU 1
        res_enh = FourStageVerificationGate.verify_stage_1_load(
            model_id="enhancement-suite-v1",
            gpu_index=1,
            base_cuda_context_mb=750.0,
            weights_mb=2500.0,
            load_time_sec=2.0,
            topology=topology,
        )
        assert res_enh.passed is True
        assert res_enh.gate_name == "Stage 1: Load"
        assert res_enh.details["gpu_index"] == 1
        assert res_enh.details["static_footprint_mb"] == 3250.0

        # qwen-speech correctly on GPU 0
        res_speech = FourStageVerificationGate.verify_stage_1_load(
            model_id="qwen-speech",
            gpu_index=0,
            base_cuda_context_mb=700.0,
            weights_mb=2200.0,
            load_time_sec=2.2,
            topology=topology,
        )
        assert res_speech.passed is True
        assert res_speech.gate_name == "Stage 1: Load"
        assert res_speech.details["gpu_index"] == 0
        assert res_speech.details["static_footprint_mb"] == 2900.0

    def test_stage_1_load_music_and_enhancement_on_wrong_gpu_fails(self):
        """Verifies that ace-step-1.5 and enhancement-suite-v1 placed on GPU 0 fail Stage 1 Load."""
        topology = DualT4HardwareTopology()
        # ace-step-1.5 wrongly placed on GPU 0
        res_music = FourStageVerificationGate.verify_stage_1_load(
            model_id="ace-step-1.5",
            gpu_index=0,
            base_cuda_context_mb=800.0,
            weights_mb=4000.0,
            load_time_sec=3.0,
            topology=topology,
        )
        assert res_music.passed is False
        assert "hardware topology requires GPU 1" in res_music.error_message

        # enhancement-suite-v1 wrongly placed on GPU 0
        res_enh = FourStageVerificationGate.verify_stage_1_load(
            model_id="enhancement-suite-v1",
            gpu_index=0,
            base_cuda_context_mb=750.0,
            weights_mb=2500.0,
            load_time_sec=2.0,
            topology=topology,
        )
        assert res_enh.passed is False
        assert "hardware topology requires GPU 1" in res_enh.error_message

        # qwen-speech wrongly placed on GPU 1
        res_speech = FourStageVerificationGate.verify_stage_1_load(
            model_id="qwen-speech",
            gpu_index=1,
            base_cuda_context_mb=700.0,
            weights_mb=2200.0,
            load_time_sec=2.2,
            topology=topology,
        )
        assert res_speech.passed is False
        assert "hardware topology requires GPU 0" in res_speech.error_message


    def test_stage_2_infer_forward_pass_and_latency(self):
        res = FourStageVerificationGate.verify_stage_2_infer(
            model_id="qwen-text",
            modality="text",
            peak_vram_mb=12000.0,
            latency_ms=185.0,
        )
        assert res.passed is True
        assert res.gate_name == "Stage 2: Infer"
        assert res.details["latency_ms"] == 185.0
        assert res.details["status"] == "PASS"

    def test_stage_2_infer_peak_vram_limit_exceeded(self):
        res = FourStageVerificationGate.verify_stage_2_infer(
            model_id="qwen-text",
            modality="text",
            peak_vram_mb=15500.0,
            latency_ms=185.0,
        )
        assert res.passed is False
        assert "exceeded 15.0 GB ceiling" in res.error_message

    def test_stage_3_validate_output_and_manifest_contract(self, tmp_path):
        manifest_file = tmp_path / "qwen-embedding.yaml"
        manifest_file.write_text(
            """
apiVersion: factory.ai/v1
kind: ModelManifest
metadata:
  id: qwen-embedding
capabilities:
  text_embedding: true
  multimodal_embedding: false
"""
        )

        def _executor(cap):
            if cap == "text_embedding":
                return True, 200, None
            if cap == "multimodal_embedding":
                return False, 400, None
            return False, 404, None

        # Unit L2 normalized vector
        vec = [0.6, 0.8]  # 0.36 + 0.64 = 1.0
        res = FourStageVerificationGate.verify_stage_3_validate(
            model_id="qwen-embedding",
            modality="embedding",
            output_payload=vec,
            manifest_path=manifest_file,
            capability_executor=_executor,
            peak_vram_mb=3700.0,
            latency_ms=50.0,
        )
        assert res.passed is True
        assert res.gate_name == "Stage 3: Validate"
        assert res.details["contract_adherence_percent"] == 100.0
        assert "artifact_sha256" in res.details

    def test_stage_3_validate_unsupported_capability_leak_fails(self, tmp_path):
        manifest_file = tmp_path / "model.yaml"
        manifest_file.write_text(
            """
apiVersion: factory.ai/v1
kind: ModelManifest
metadata:
  id: test-model
capabilities:
  unsupported_task: false
"""
        )

        def _bad_executor(cap):
            # Leaks with 200 instead of 400
            return True, 200, "fallback-model"

        res = FourStageVerificationGate.verify_stage_3_validate(
            model_id="test-model",
            modality="text",
            output_payload="valid output",
            manifest_path=manifest_file,
            capability_executor=_bad_executor,
        )
        assert res.passed is False
        assert "Capability contract verification failed" in res.error_message

    def test_stage_4_profile_telemetry_and_zero_leak(self):
        profiler = HardwareMemoryProfiler(device_index=0)

        # Zero leak runner
        def _runner(idx):
            return True, 12000.0 + (idx * 1.5), 180.0

        res = FourStageVerificationGate.verify_stage_4_profile(
            model_id="qwen-text",
            profiler=profiler,
            sustained_requests_count=5,
            request_runner=_runner,
        )
        assert res.passed is True
        assert res.gate_name == "Stage 4: Profile"
        assert res.details["telemetry_status"] == "UNAVAILABLE"  # on local Mac
        assert res.details["vram_leak_mb"] < 50.0

    def test_stage_4_profile_excessive_leak_fails(self):
        profiler = HardwareMemoryProfiler(device_index=0)

        # Leaking runner (> 50 MB)
        def _leaking_runner(idx):
            return True, 12000.0 + (idx * 15.0), 180.0

        res = FourStageVerificationGate.verify_stage_4_profile(
            model_id="qwen-text",
            profiler=profiler,
            sustained_requests_count=5,
            request_runner=_leaking_runner,
        )
        assert res.passed is False
        assert "Memory leak detected" in res.error_message

    def test_four_stage_pipeline_in_preflight_certification_results(self, tmp_path):
        manifests_dir = Path(__file__).resolve().parent.parent.parent.parent / "manifests"
        sim = PreflightContractSimulator(manifests_dir=manifests_dir)
        results = sim.execute_all(output_path=tmp_path / "res.json")

        assert "verification_pipeline" in results
        assert results["verification_pipeline"] == [
            "Stage 1: Load", "Stage 2: Infer", "Stage 3: Validate", "Stage 4: Profile"
        ]

        for m_id, m_data in results["models"].items():
            assert "four_stage_pipeline" in m_data
            pipe = m_data["four_stage_pipeline"]
            assert pipe["stage_1_load"]["passed"] is True
            assert pipe["stage_2_infer"]["passed"] is True
            assert pipe["stage_3_validate"]["passed"] is True
            assert pipe["stage_4_profile"]["passed"] is True


# =====================================================================
# 9. KaggleDualT4Runner Lifecycle, Mode & Contract Tests
# =====================================================================

class TestKaggleDualT4RunnerLifecycleAndContract:
    """Verifies KaggleDualT4Runner initialization, modes, lazy client, and kwargs."""

    @pytest.fixture(autouse=True)
    def clean_kaggle_env(self, monkeypatch, tmp_path):
        """Ensure no ambient Kaggle credentials bleed into lifecycle tests."""
        for k in ["KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_KEY"]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))

    def test_init_default_mode_without_credentials(self):
        """Default instantiation must succeed without credentials and default to mode='remote'."""
        runner = KaggleDualT4Runner()
        assert runner.mode == "remote"
        assert runner._client is None
        assert isinstance(runner.kwargs, dict)
        assert len(runner.kwargs) == 0

    def test_init_mode_preflight_without_credentials(self):
        """Instantiating with mode='preflight' must not raise and set runner.mode='preflight'."""
        runner = KaggleDualT4Runner(mode="preflight")
        assert runner.mode == "preflight"
        assert runner._client is None

    def test_init_mode_remote_without_credentials(self):
        """Instantiating with mode='remote' must not raise and set runner.mode='remote'."""
        runner = KaggleDualT4Runner(mode="remote")
        assert runner.mode == "remote"
        assert runner._client is None

    def test_lazy_client_property_raises_auth_error_when_unconfigured(self):
        """Accessing runner.client with no credentials raises KaggleAuthError lazily."""
        runner = KaggleDualT4Runner(mode="preflight")
        with pytest.raises(KaggleAuthError) as exc_info:
            _ = runner.client
        assert "Kaggle username not configured" in str(exc_info.value)

    def test_explicit_client_injection(self):
        """Explicitly passed client is retained and returned without environment lookup."""
        client = KaggleAPIClient(username="mock_user", api_key="KGAT_mock_token_12345")
        runner = KaggleDualT4Runner(client=client, mode="remote")
        assert runner.client is client
        assert runner._client is client
        assert runner.client.username == "mock_user"

    def test_init_accepts_arbitrary_kwargs_without_type_error(self):
        """Extra keyword arguments must be captured in runner.kwargs without TypeError."""
        runner = KaggleDualT4Runner(
            mode="preflight",
            custom_timeout=1200,
            enable_profiling=True,
            tag="regression-gate-check",
        )
        assert runner.mode == "preflight"
        assert runner.kwargs.get("custom_timeout") == 1200
        assert runner.kwargs.get("enable_profiling") is True
        assert runner.kwargs.get("tag") == "regression-gate-check"
        assert runner.extra_options.get("tag") == "regression-gate-check"

    def test_run_local_certification_preflight_without_credentials(self, tmp_path):
        """Direct local certification run completes cleanly without requiring Kaggle credentials."""
        runner = KaggleDualT4Runner(mode="preflight")
        out_file = tmp_path / "cert_results.json"
        res = runner.run_local_certification(output_path=out_file)
        assert res["all_gates_passed"] is True
        assert res["environment"] == "simulation-local"
        assert runner._client is None  # KaggleAPIClient was never accessed or initialized

    def test_run_dispatcher_preflight(self, tmp_path):
        """Unified runner.run() dispatcher routes to local certification in preflight mode."""
        runner = KaggleDualT4Runner(mode="preflight")
        out_file = tmp_path / "cert_run_dispatch.json"
        res = runner.run(output_path=out_file)
        assert res["all_gates_passed"] is True

    def test_deploy_and_certify_remote_triggers_auth_check_when_uncredentialed(self):
        """Calling deploy_and_certify in remote mode triggers client access and fails fast on missing auth."""
        runner = KaggleDualT4Runner(mode="remote")
        with pytest.raises(KaggleAuthError) as exc_info:
            runner.deploy_and_certify(slug="test-slug")
        assert "Kaggle username not configured" in str(exc_info.value)


# =====================================================================
# 10. Kaggle Deployments Module Importability Tests
# =====================================================================

class TestKaggleModulesImportabilityWithoutCredentials:
    """Verifies that all modules under deployments.kaggle can be imported safely without credentials."""

    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch, tmp_path):
        for k in ["KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_KEY"]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))

    def test_import_deployments_kaggle_package(self):
        """import deployments.kaggle must not raise KaggleAuthError."""
        import deployments.kaggle as dk
        assert hasattr(dk, "KaggleDualT4Runner")
        assert hasattr(dk, "KaggleAPIClient")

    @pytest.mark.parametrize("module_name", [
        "runner",
        "certify_kernel",
        "prepare_datasets",
        "smoke_runner",
        "monitor_both_sessions",
        "session1_visual_pipeline",
        "session2_audio_pipeline",
        "run_studio_quality_pipeline",
        "run_high_quality_test",
        "live_gpu_test",
    ])
    def test_submodules_import_cleanly_without_credentials(self, module_name):
        """Every submodule in deployments.kaggle must import cleanly without raising KaggleAuthError."""
        import importlib
        full_name = f"deployments.kaggle.{module_name}"
        mod = importlib.import_module(full_name)
        assert mod is not None


