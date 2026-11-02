"""
tests/adversarial/test_adversarial_p3_challenger.py
Adversarial Challenger Test Suite for Phase P3 (Live GPU Certification):
1. Kaggle API resilience & error handling (401, 403, 404, 500, 502/503/504, URLError, timeouts, kernel failures).
2. Dual-T4 mutual exclusion violations & hardware topology invariants.
3. 5 Sequential Certification Gates failure inoculation (VRAM overflow, leak >= 50MB, numerical corruption, contract bypass, SIGKILL/reaper).
4. Empirical certification results & manifest state transitions.
"""
import io
import json
import math
from pathlib import Path
import struct
from unittest.mock import MagicMock, patch
import urllib.error
import urllib.request
import pytest
import yaml

from deployments.kaggle.runner import (
    DEFAULT_KAGGLE_BEARER_TOKEN,
    DEFAULT_KAGGLE_USER,
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
    CertificationGateResult,
    HardwareMemoryProfiler,
    MemoryProfileSnapshot,
    PreflightContractSimulator,
    Tier1ModelCertifier,
    TIER_1_PROFILES,
)


# =====================================================================
# 1. Kaggle REST API v1 Adversarial Resilience & Error Handling
# =====================================================================

class TestKaggleAPIClientAdversarialResilience:
    """Stress tests Kaggle REST API v1 client error handling and resilience."""

    @pytest.mark.parametrize("status_code,expected_error_cls", [
        (401, KaggleAuthError),
        (403, KaggleAuthError),
        (404, KaggleAPIError),
        (500, KaggleAPIError),
        (502, KaggleAPIError),
        (503, KaggleAPIError),
        (504, KaggleAPIError),
    ])
    @patch("urllib.request.urlopen")
    def test_http_error_code_mapping(self, mock_urlopen, status_code, expected_error_cls):
        """Verify all standard HTTP error statuses raise the correct typed exceptions."""
        http_error = urllib.error.HTTPError(
            url="https://www.kaggle.com/api/v1/kernels/status",
            code=status_code,
            msg=f"HTTP Error {status_code}",
            hdrs={},
            fp=io.BytesIO(f'{{"error": "Simulated {status_code}"}}'.encode("utf-8")),
        )
        mock_urlopen.side_effect = http_error

        client = KaggleAPIClient()
        with pytest.raises(expected_error_cls) as exc_info:
            client.get_kernel_status("test-slug")

        assert exc_info.value.status_code == status_code
        assert f"Simulated {status_code}" in (exc_info.value.response_body or "")

    @patch("urllib.request.urlopen")
    def test_urllib_network_urlerror_raises_kaggle_api_error(self, mock_urlopen):
        """Verify network-level errors (connection refused, DNS failure) raise KaggleAPIError."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused: [Errno 61]")

        client = KaggleAPIClient()
        with pytest.raises(KaggleAPIError) as exc_info:
            client.get_kernel_status("any-slug")

        assert "Network error connecting to Kaggle API" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    def test_unexpected_non_urllib_exception_raises_kaggle_api_error(self, mock_urlopen):
        """Verify arbitrary unexpected runtime errors are cleanly wrapped in KaggleAPIError."""
        mock_urlopen.side_effect = RuntimeError("Low-level OS socket reset")

        client = KaggleAPIClient()
        with pytest.raises(KaggleAPIError) as exc_info:
            client.get_kernel_status("any-slug")

        assert "Unexpected error calling Kaggle API" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    def test_malformed_non_json_response_raises_kaggle_api_error(self, mock_urlopen):
        """Verify non-JSON response (e.g. HTML 502 page from cloud proxy) raises KaggleAPIError."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<html><head><title>502 Bad Gateway</title></head><body>502</body></html>"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        client = KaggleAPIClient()
        with pytest.raises(KaggleAPIError) as exc_info:
            client.get_kernel_status("any-slug")

        assert "Unexpected error calling Kaggle API" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    def test_empty_response_body_returns_empty_dict(self, mock_urlopen):
        """Verify empty body 200 responses return an empty dict without crashing."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"   \n  "
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        client = KaggleAPIClient()
        res = client.get_kernel_status("any-slug")
        assert res == {}

    @patch("urllib.request.urlopen")
    def test_push_kernel_clamps_timeout_to_7200(self, mock_urlopen):
        """Verify kernel submission clamps session timeout to Kaggle minimum 7200 seconds."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"hasError": False, "versionNumber": 1}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        client = KaggleAPIClient()
        client.push_kernel(slug="test-slug", title="Test", code_content="pass", session_timeout_seconds=300)

        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["sessionTimeoutSeconds"] == 7200

    def test_base_url_trailing_slash_normalization(self):
        """Verify base URL trailing slashes are stripped to avoid double slashes."""
        client = KaggleAPIClient(base_url="https://www.kaggle.com/api/v1///")
        assert client.base_url == "https://www.kaggle.com/api/v1"

    @patch("urllib.request.urlopen")
    def test_polling_timeout_raises_kaggle_timeout_error(self, mock_urlopen):
        """Verify polling beyond timeout_sec raises KaggleTimeoutError."""
        mock_s = MagicMock()
        mock_s.read.return_value = json.dumps({"status": "running"}).encode("utf-8")
        ctx_s = MagicMock()
        ctx_s.__enter__.return_value = mock_s

        mock_o = MagicMock()
        mock_o.read.return_value = json.dumps({"log": "Still working...\n"}).encode("utf-8")
        ctx_o = MagicMock()
        ctx_o.__enter__.return_value = mock_o

        mock_urlopen.side_effect = [ctx_s, ctx_o] * 10

        client = KaggleAPIClient()
        with pytest.raises(KaggleTimeoutError) as exc_info:
            client.poll_until_complete("stuck-kernel", poll_interval_sec=0.01, timeout_sec=0.05)

        assert "timed out after" in str(exc_info.value)

    @pytest.mark.parametrize("fail_status", ["error", "failed", "cancelacknowledgement"])
    @patch("urllib.request.urlopen")
    def test_all_failure_statuses_raise_kaggle_kernel_failed_error(self, mock_urlopen, fail_status):
        """Verify kernel failure statuses raise KaggleKernelFailedError."""
        mock_s = MagicMock()
        mock_s.read.return_value = json.dumps({
            "status": fail_status,
            "failureMessage": f"Kernel terminated with status {fail_status}",
        }).encode("utf-8")
        ctx_s = MagicMock()
        ctx_s.__enter__.return_value = mock_s

        mock_o = MagicMock()
        mock_o.read.return_value = json.dumps({"log": "Crash dump"}).encode("utf-8")
        ctx_o = MagicMock()
        ctx_o.__enter__.return_value = mock_o

        mock_urlopen.side_effect = [ctx_s, ctx_o]

        client = KaggleAPIClient()
        with pytest.raises(KaggleKernelFailedError) as exc_info:
            client.poll_until_complete("failed-kernel", poll_interval_sec=0.01)

        assert fail_status in str(exc_info.value)
        assert exc_info.value.failure_message == f"Kernel terminated with status {fail_status}"


# =====================================================================
# 2. Dual-T4 Hardware Topology & Mutual Exclusion Invariants
# =====================================================================

class TestDualT4TopologyAdversarialInvariants:
    """Stress tests GPU admission mutual exclusion and boundary conditions."""

    def test_assigned_gpu_partitioning(self):
        """Verify strict hardware partitioning across services."""
        topology = DualT4HardwareTopology()
        # Card 0
        for m in ["qwen-text", "qwen-decision", "qwen-vlm", "qwen-image-2.1"]:
            assert topology.get_assigned_gpu(m) == 0

        # Card 1
        for m in ["qwen-embedding", "wan-2.2", "minimax-h3", "bge-reranker-large"]:
            assert topology.get_assigned_gpu(m) == 1

    def test_concurrent_allocation_on_same_gpu_exceeding_ceiling_fails(self):
        """Verify attempting concurrent allocation exceeding 14,540 MB on GPU 0 raises RuntimeError."""
        topology = DualT4HardwareTopology()
        # Admit qwen-text: 5,320 MB static footprint
        topology.admit_model(0, "qwen-text", 5320.0)

        # Attempt to admit qwen-image-2.1: 11,380 MB static footprint (total = 16,700 MB > 14,540 MB)
        ok, reason = topology.check_exclusive_admission(0, "qwen-image-2.1", 11380.0)
        assert ok is False
        assert "Exclusive admission conflict on GPU 0" in reason

        with pytest.raises(RuntimeError) as exc_info:
            topology.admit_model(0, "qwen-image-2.1", 11380.0)
        assert "exceeds limit" in str(exc_info.value)

    def test_concurrent_allocation_across_different_gpus_succeeds(self):
        """Verify dual-GPU concurrency: GPU 0 and GPU 1 can hold models simultaneously."""
        topology = DualT4HardwareTopology()
        # Admit qwen-image-2.1 on GPU 0
        topology.admit_model(0, "qwen-image-2.1", 11380.0)
        # Admit wan-2.2 on GPU 1
        topology.admit_model(1, "wan-2.2", 10690.0)

        assert topology._active_models[0] == "qwen-image-2.1"
        assert topology._active_models[1] == "wan-2.2"

    def test_evict_model_requires_matching_model_id_or_none(self):
        """Verify eviction of a non-active model does NOT evict the resident model."""
        topology = DualT4HardwareTopology()
        topology.admit_model(0, "qwen-text", 5320.0)

        # Attempt to evict non-active model
        topology.evict_model(0, "wrong-model-id")
        assert topology._active_models[0] == "qwen-text"
        assert topology._allocated_vram_mb[0] == 5320.0

        # Evict with exact ID
        topology.evict_model(0, "qwen-text")
        assert topology._active_models[0] is None
        assert topology._allocated_vram_mb[0] == 0.0

    def test_static_limit_boundary_conditions(self):
        """Verify exact boundary behavior for STATIC_LOAD_LIMIT_MB (14540.0 MB)."""
        topology = DualT4HardwareTopology()
        # Exactly at limit: passes
        ok, _ = topology.check_exclusive_admission(0, "boundary-model", 14540.0)
        assert ok is True

        # Pre-allocate 14540.0 MB
        topology.admit_model(0, "resident-model", 14540.0)

        # Any additional allocation exceeds limit
        ok, reason = topology.check_exclusive_admission(0, "another-model", 0.1)
        assert ok is False
        assert "exceeds limit" in reason


# =====================================================================
# 3. 5 Sequential Certification Gates Failure Inoculation
# =====================================================================

class TestCertificationGatesFailureInoculation:
    """Stress tests Gate 1 through Gate 5 against adversarial inputs and edge cases."""

    # --- Gate 1: Load Gate Inoculation ---

    def test_gate_1_driver_overhead_bounds(self):
        """Verify Gate 1 rejects CUDA driver context outside 400 - 1200 MB."""
        # Too low (< 400 MB)
        res_low = CertificationGates.verify_load_gate("qwen-text", 0, 399.9, 4500.0, 3.0)
        assert res_low.passed is False
        assert "CUDA driver context overhead out of range" in res_low.error_message

        # Too high (> 1200 MB)
        res_high = CertificationGates.verify_load_gate("qwen-text", 0, 1200.1, 4500.0, 3.0)
        assert res_high.passed is False
        assert "CUDA driver context overhead out of range" in res_high.error_message

    def test_gate_1_static_footprint_boundary(self):
        """Verify Gate 1 boundary at exactly 14,540.0 MB."""
        # 14,540.0 MB: passes
        res_pass = CertificationGates.verify_load_gate("test", 0, 800.0, 13740.0, 3.0)
        assert res_pass.passed is True

        # 14,540.1 MB: rejects
        res_fail = CertificationGates.verify_load_gate("test", 0, 800.0, 13740.1, 3.0)
        assert res_fail.passed is False
        assert "exceeds Gate 1 ceiling" in res_fail.error_message

    # --- Gate 2: Inference Gate Inoculation ---

    def test_gate_2_peak_vram_hard_ceiling_at_15360mb(self):
        """Verify Gate 2 rejects peak VRAM >= 15360.0 MB."""
        # 15359.9 MB: passes
        res_pass = CertificationGates.verify_inference_gate("qwen-text", "text", "Valid output", 15359.9, 100.0)
        assert res_pass.passed is True

        # 15360.0 MB: rejects
        res_fail = CertificationGates.verify_inference_gate("qwen-text", "text", "Valid output", 15360.0, 100.0)
        assert res_fail.passed is False
        assert "exceeded 15.0 GB ceiling" in res_fail.error_message

    @pytest.mark.parametrize("corrupt_text", ["", "   ", "\n\t  \r\n"])
    def test_gate_2_text_empty_or_whitespace_rejected(self, corrupt_text):
        """Verify empty or whitespace-only token generation is rejected."""
        res = CertificationGates.verify_inference_gate("qwen-text", "text", corrupt_text, 10000.0, 100.0)
        assert res.passed is False
        assert "Empty token stream" in res.error_message

    @pytest.mark.parametrize("corrupt_decision", [
        {"predicate": True, "score": float("nan")},
        {"predicate": True, "score": float("inf")},
        {"predicate": True, "score": float("-inf")},
    ])
    def test_gate_2_decision_numerical_corruption_rejected(self, corrupt_decision):
        """Verify NaN/inf in decision outputs are rejected."""
        res = CertificationGates.verify_inference_gate("qwen-decision", "decision", corrupt_decision, 5000.0, 50.0)
        assert res.passed is False
        assert "Invalid numerical value" in res.error_message

    @pytest.mark.parametrize("corrupt_vector", [
        [],  # empty
        [0.0, 0.0, 0.0],  # zero norm
        [1.0, 1.0, 1.0],  # norm = sqrt(3) != 1.0
        [0.99, 0.0],  # norm = 0.99 (|1.0 - 0.99| = 0.01 > 1e-3)
    ])
    def test_gate_2_embedding_unnormalized_vectors_rejected(self, corrupt_vector):
        """Verify embedding vectors violating unit L2 norm (1.0 +/- 1e-3) are rejected."""
        res = CertificationGates.verify_inference_gate("qwen-embedding", "embedding", corrupt_vector, 3000.0, 20.0)
        assert res.passed is False
        assert ("non-empty vector list" in res.error_message or "violates unit L2 norm" in res.error_message)

    @pytest.mark.parametrize("corrupt_image", [
        b"",  # empty
        b"JFIF\x00\x00",  # JPEG not PNG
        b"\x89PNG\r\n\x1a\n" + b"Corrupted chunk without IHDR",
        "not_bytes_string",
    ])
    def test_gate_2_image_corrupted_payload_rejected(self, corrupt_image):
        """Verify image payloads missing valid PNG header or IHDR chunk are rejected."""
        res = CertificationGates.verify_inference_gate("qwen-image-2.1", "image", corrupt_image, 14000.0, 500.0)
        assert res.passed is False

    @pytest.mark.parametrize("corrupt_video", [
        b"",  # empty
        b"RIFF\x00\x00\x00\x00AVI ",  # AVI not MP4
        struct.pack(">I", 24) + b"ftyp" + b"isom" + b"\x00" * 12,  # ftyp but no moov or mdat
        "not_bytes_string",
    ])
    def test_gate_2_video_corrupted_payload_rejected(self, corrupt_video):
        """Verify video payloads missing ftyp, moov, or mdat are rejected."""
        res = CertificationGates.verify_inference_gate("wan-2.2", "video", corrupt_video, 14000.0, 1500.0)
        assert res.passed is False

    # --- Gate 3: Contract Gate Inoculation ---

    def test_gate_3_missing_manifest_file_rejected(self, tmp_path):
        """Verify missing manifest file path returns gate failure."""
        res = CertificationGates.verify_contract_gate(tmp_path / "nonexistent.yaml", lambda c: (True, 200, None))
        assert res.passed is False
        assert "not found" in res.error_message

    def test_gate_3_empty_manifest_capabilities_rejected(self, tmp_path):
        """Verify manifest with 0 capabilities is rejected."""
        manifest = tmp_path / "empty-caps.yaml"
        manifest.write_text("apiVersion: factory.ai/v1\nkind: ModelManifest\ncapabilities: {}\n")
        res = CertificationGates.verify_contract_gate(manifest, lambda c: (True, 200, None))
        assert res.passed is False
        assert "No capabilities declared" in res.error_message

    def test_gate_3_declared_true_capability_failing_rejected(self, tmp_path):
        """Verify capability declared 'true' that fails execution causes gate failure."""
        manifest = tmp_path / "model.yaml"
        manifest.write_text("capabilities:\n  text_to_image: true\n")
        res = CertificationGates.verify_contract_gate(manifest, lambda c: (False, 500, None))
        assert res.passed is False
        assert "marked 'true' in manifest but execution failed" in res.error_message

    def test_gate_3_unsupported_capability_fallback_forbidden(self, tmp_path):
        """Verify capability declared 'false' that leaks to fallback domain is rejected."""
        manifest = tmp_path / "model.yaml"
        manifest.write_text("capabilities:\n  video_with_audio: false\n")
        # Returns 400 but also specifies fallback_target
        res = CertificationGates.verify_contract_gate(manifest, lambda c: (False, 400, "fallback-model"))
        assert res.passed is False
        assert "cross-model fallback forbidden" in res.error_message

    # --- Gate 4: Stress Gate Inoculation ---

    def test_gate_4_memory_leak_threshold_50mb(self):
        """Verify Gate 4 rejects memory leaks >= 50.0 MB across sustained requests."""
        # 49.9 MB leak: passes
        def _runner_pass(idx):
            return True, 12000.0 + (idx * 9.98), 100.0  # (5-1) * 9.98 = 39.92 MB

        res_pass = CertificationGates.verify_stress_gate("qwen-text", 5, _runner_pass)
        assert res_pass.passed is True
        assert res_pass.details["vram_leak_mb"] < 50.0

        # 50.0 MB leak: rejects
        def _runner_fail(idx):
            return True, 12000.0 + (idx * 12.5), 100.0  # 4 * 12.5 = 50.0 MB

        res_fail = CertificationGates.verify_stress_gate("qwen-text", 5, _runner_fail)
        assert res_fail.passed is False
        assert "Memory leak detected" in res_fail.error_message

    def test_gate_4_vram_spike_exceeding_15gb_fails(self):
        """Verify Gate 4 rejects transient VRAM spikes >= 15360 MB."""
        def _runner_spike(idx):
            vram = 15400.0 if idx == 3 else 12000.0
            return True, vram, 100.0

        res = CertificationGates.verify_stress_gate("qwen-text", 5, _runner_spike)
        assert res.passed is False
        assert "exceeded 15.0 GB" in res.error_message

    def test_gate_4_failed_intermediate_request_fails(self):
        """Verify failure on request N terminates stress gate immediately."""
        def _runner_broken(idx):
            if idx == 2:
                return False, 12000.0, 0.0
            return True, 12000.0, 100.0

        res = CertificationGates.verify_stress_gate("qwen-text", 5, _runner_broken)
        assert res.passed is False
        assert "Request 3/5 failed" in res.error_message

    # --- Gate 5: Recovery Gate Inoculation ---

    @pytest.mark.parametrize("missing_invariant", [
        "sigkill_caught",
        "lease_reaped",
        "worker_rebooted",
        "cuda_context_clean",
    ])
    def test_gate_5_missing_invariants_rejected(self, missing_invariant):
        """Verify Gate 5 requires all four recovery invariants to be True."""
        def _faulty_simulator():
            out = {
                "sigkill_caught": True,
                "lease_reaped": True,
                "worker_rebooted": True,
                "cuda_context_clean": True,
            }
            out[missing_invariant] = False
            return out

        res = CertificationGates.verify_recovery_gate("qwen-text", _faulty_simulator)
        assert res.passed is False
        assert "Recovery gate invariants failed" in res.error_message

    def test_gate_5_unhandled_exception_in_simulator_handled_gracefully(self):
        """Verify unhandled simulator exception is trapped and reported as gate failure."""
        def _crashing_simulator():
            raise TimeoutError("Deadlock waiting for GPU context reaper")

        res = CertificationGates.verify_recovery_gate("qwen-text", _crashing_simulator)
        assert res.passed is False
        assert "unhandled exception" in res.error_message


# =====================================================================
# 4. Empirical Certification Results & Manifest Invariant Checks
# =====================================================================

class TestEmpiricalCertificationResultsIntegrity:
    """Verifies physical artifact integrity and prevents premature certification escalation."""

    def test_certification_results_file_empirical_validity(self):
        """Verify certification_results.json exists and meets all physical constraints."""
        root_dir = Path(__file__).resolve().parent.parent.parent
        results_file = root_dir / "deployments" / "kaggle" / "certification_results.json"
        assert results_file.exists(), f"Missing {results_file}"

        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["all_gates_passed"] is True
        assert data["environment"] == "simulation-local"
        assert data.get("live_hardware_attested") is False
        assert data["hardware"]["total_gpus"] == 2
        assert data["hardware"]["vram_per_gpu_mb"] == 15360.0

        models = data["models"]
        assert len(models) == 6

        tier1_keys = ["qwen-text", "qwen-embedding", "qwen-decision", "qwen-vlm", "qwen-image-2.1", "wan-2.2"]
        for k in tier1_keys:
            assert k in models, f"Model {k} missing from certification results"
            m = models[k]
            assert m["certified"] is True
            # VRAM under 15,000 MB
            assert m["profiling"]["peak_vram_mb"] < 15000.0, f"{k} peak VRAM exceeded 15.0 GB"
            # Headroom > 800 MB
            assert m["profiling"]["headroom_mb"] >= 800.0, f"{k} headroom insufficient"

            # 5 Gates
            gates = m["gates"]
            for g_name in ["gate_1_load", "gate_2_inference", "gate_3_contract", "gate_4_stress", "gate_5_recovery"]:
                assert g_name in gates
                assert gates[g_name]["passed"] is True

    def test_manifest_validation_status_anti_escalation(self):
        """
        Anti-Escalation Check:
        Zero models may be marked 'live_verified' without physical GPU execution.
        Tier 1 core models must have 'pending_live' with pre-flight simulation citations.
        """
        root_dir = Path(__file__).resolve().parent.parent.parent
        manifests_dir = root_dir / "manifests" / "models"

        tier1_models = {"qwen-text", "qwen-embedding", "qwen-decision", "qwen-vlm", "qwen-image-2.1", "wan-2.2"}
        all_manifest_files = list(manifests_dir.glob("*.yaml"))
        assert len(all_manifest_files) >= 10

        for m_file in all_manifest_files:
            with open(m_file, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
            m_id = content.get("metadata", {}).get("id")
            val_status = content.get("validation_status")

            # Anti-Escalation Gate: Zero models may be marked live_verified without physical GPU execution
            assert val_status != "live_verified", (
                f"Model {m_id} was improperly escalated to 'live_verified' before live GPU execution!"
            )
            if m_id in tier1_models:
                assert val_status == "pending_live", f"Tier 1 model {m_id} expected pending_live, got {val_status}"
                notes = content.get("validation_notes", "")
                assert "Tesla T4" in notes, f"{m_id} notes missing Tesla T4 citation"
                assert "pre-flight" in notes.lower(), f"{m_id} notes missing pre-flight citation"


# =====================================================================
# 5. Gate 2 Numerical / Container Probes & Profiler Invariant Verification
# =====================================================================

class TestGate2ProbesAndProfilerInvariants:
    """
    Adversarial stress harness probing Gate 2 assertions and HardwareMemoryProfiler invariants:
    - Embedding: norm tolerances (1.0 vs 1.0+5e-5 vs 1.0+1.5e-4), NaN, Inf, zero vector.
    - Image: PNG signature, size >= 33, IHDR chunk length 13, corrupted chunks.
    - Video: MP4 moov/mdat atom checks (both required, missing moov, missing mdat).
    - HardwareMemoryProfiler snapshot invariants across all 5 stages.
    """

    # --- Embedding Probes ---

    @pytest.mark.parametrize("norm_offset,should_pass", [
        (0.0, True),        # exactly 1.0
        (5e-5, True),       # 1.0 + 5e-5 (within 1e-4 tolerance)
        (-5e-5, True),      # 1.0 - 5e-5 (within 1e-4 tolerance)
        (1.5e-4, False),    # 1.0 + 1.5e-4 (exceeds 1e-4 tolerance)
        (-1.5e-4, False),   # 1.0 - 1.5e-4 (exceeds 1e-4 tolerance)
    ])
    def test_embedding_norm_tolerances_boundary(self, norm_offset, should_pass):
        """Probe embedding norm tolerances: 1.0 (pass), 1.0+5e-5 (pass), 1.0+1.5e-4 (fail)."""
        val = 1.0 + norm_offset
        vec = [val, 0.0]
        res = CertificationGates.verify_inference_gate(
            model_id="qwen-embedding",
            modality="embedding",
            output_payload=vec,
            peak_vram_mb=3700.0,
            latency_ms=50.0,
        )
        assert res.passed is should_pass
        if not should_pass:
            assert "violates unit L2 norm" in res.error_message

    @pytest.mark.parametrize("corrupt_vector,err_substring", [
        ([float("nan"), 0.0], "violates unit L2 norm"),
        ([0.0, float("nan")], "violates unit L2 norm"),
        ([float("inf"), 0.0], "violates unit L2 norm"),
        ([-float("inf"), 0.0], "violates unit L2 norm"),
        ([0.0, 0.0, 0.0], "violates unit L2 norm"),
        ([], "non-empty vector list"),
    ])
    def test_embedding_nan_inf_zero_rejection(self, corrupt_vector, err_substring):
        """Probe embedding rejection of NaN, Inf, zero vector, and empty vector."""
        res = CertificationGates.verify_inference_gate(
            model_id="qwen-embedding",
            modality="embedding",
            output_payload=corrupt_vector,
            peak_vram_mb=3700.0,
            latency_ms=50.0,
        )
        assert res.passed is False
        assert err_substring in res.error_message

    # --- Image Probes ---

    def test_image_png_33_byte_exact_boundary(self):
        """Probe PNG minimum 33-byte boundary: 33 bytes pass, 32 bytes fail."""
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr_chunk = struct.pack(">I", 13) + b"IHDR" + (b"\x00" * 13) + struct.pack(">I", 0x12345678)
        valid_33 = sig + ihdr_chunk
        assert len(valid_33) == 33

        # Exactly 33 bytes: passes
        res_33 = CertificationGates.verify_inference_gate("qwen-image-2.1", "image", valid_33, 10000.0, 500.0)
        assert res_33.passed is True

        # Truncated 32 bytes: fails
        short_32 = valid_33[:32]
        res_32 = CertificationGates.verify_inference_gate("qwen-image-2.1", "image", short_32, 10000.0, 500.0)
        assert res_32.passed is False
        assert "minimum 33 bytes" in res_32.error_message

    @pytest.mark.parametrize("bad_sig", [
        b"\x88PNG\r\n\x1a\n",
        b"GIF89a\x00\x00\x00",
        b"RIFF\x00\x00\x00\x00WEBP",
        b"\xff\xd8\xff\xe0\x00\x10JFIF",
    ])
    def test_image_non_png_signature_rejected(self, bad_sig):
        """Probe non-PNG signature rejection."""
        ihdr_chunk = struct.pack(">I", 13) + b"IHDR" + (b"\x00" * 13) + struct.pack(">I", 0x12345678)
        payload = bad_sig + ihdr_chunk
        res = CertificationGates.verify_inference_gate("qwen-image-2.1", "image", payload, 10000.0, 500.0)
        assert res.passed is False
        assert "does not start with valid 8-byte PNG header" in res.error_message

    @pytest.mark.parametrize("bad_length", [0, 12, 14, 255])
    def test_image_invalid_ihdr_chunk_length_rejected(self, bad_length):
        """Probe invalid IHDR chunk length (must be exactly 13)."""
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr_chunk = struct.pack(">I", bad_length) + b"IHDR" + (b"\x00" * max(13, bad_length)) + struct.pack(">I", 0x12345678)
        payload = sig + ihdr_chunk
        res = CertificationGates.verify_inference_gate("qwen-image-2.1", "image", payload, 10000.0, 500.0)
        assert res.passed is False
        assert "invalid IHDR chunk" in res.error_message

    @pytest.mark.parametrize("bad_chunk_type", [b"IDAT", b"PLTE", b"sRGB", b"tEXt"])
    def test_image_non_ihdr_first_chunk_rejected(self, bad_chunk_type):
        """Probe non-IHDR first chunk rejection (first chunk must be IHDR)."""
        sig = b"\x89PNG\r\n\x1a\n"
        chunk = struct.pack(">I", 13) + bad_chunk_type + (b"\x00" * 13) + struct.pack(">I", 0x12345678)
        payload = sig + chunk
        res = CertificationGates.verify_inference_gate("qwen-image-2.1", "image", payload, 10000.0, 500.0)
        assert res.passed is False
        assert "invalid IHDR chunk" in res.error_message

    # --- Video Probes ---

    def test_video_mp4_moov_and_mdat_atom_invariants(self):
        """Probe MP4 container requirement: both moov and mdat boxes required."""
        ftyp = struct.pack(">I", 24) + b"ftyp" + b"isom" + struct.pack(">I", 512) + b"isomiso2"
        moov = struct.pack(">I", 16) + b"moov" + b"video_moov_box"
        mdat = struct.pack(">I", 32) + b"mdat" + (b"video_frame_bytes" * 2)

        # Both present: passes
        valid_mp4 = ftyp + moov + mdat
        res_valid = CertificationGates.verify_inference_gate("wan-2.2", "video", valid_mp4, 14000.0, 1500.0)
        assert res_valid.passed is True

        # Only moov (missing mdat): fails
        res_no_mdat = CertificationGates.verify_inference_gate("wan-2.2", "video", ftyp + moov, 14000.0, 1500.0)
        assert res_no_mdat.passed is False
        assert "missing essential MP4 'moov' or 'mdat' boxes" in res_no_mdat.error_message
        assert res_no_mdat.details["has_moov"] is True
        assert res_no_mdat.details["has_mdat"] is False

        # Only mdat (missing moov): fails
        res_no_moov = CertificationGates.verify_inference_gate("wan-2.2", "video", ftyp + mdat, 14000.0, 1500.0)
        assert res_no_moov.passed is False
        assert "missing essential MP4 'moov' or 'mdat' boxes" in res_no_moov.error_message
        assert res_no_moov.details["has_moov"] is False
        assert res_no_moov.details["has_mdat"] is True

    @pytest.mark.parametrize("corrupt_video", [
        b"",
        b"short",
        b"\x00" * 31,
        b"\x00" * 32 + b"moov" + b"mdat",  # missing ftyp in first 32 bytes
    ])
    def test_video_empty_and_missing_ftyp_rejected(self, corrupt_video):
        """Probe empty, truncated, or missing ftyp box rejection."""
        res = CertificationGates.verify_inference_gate("wan-2.2", "video", corrupt_video, 14000.0, 1500.0)
        assert res.passed is False
        assert "missing valid MP4 'ftyp' box" in res.error_message

    # --- HardwareMemoryProfiler Snapshot Invariants across 5 Stages ---

    def test_hardware_memory_profiler_five_stages_invariants(self):
        """Verify HardwareMemoryProfiler takes valid non-negative snapshots across all 5 stages."""
        profiler = HardwareMemoryProfiler(device_index=0)
        expected_stages = ["pre_load_idle", "post_load", "inference_peak", "post_stress", "post_recovery_clean"]

        snapshots = [
            profiler.profile_stage("pre_load_idle"),
            profiler.profile_stage("post_load", base_context_mb=820.0, weights_mb=4500.0),
            profiler.profile_stage("inference_peak", base_context_mb=820.0, weights_mb=4500.0, workspace_mb=6680.0),
            profiler.profile_stage("post_stress", base_context_mb=820.0, weights_mb=4500.0, workspace_mb=6680.0),
            profiler.profile_stage("post_recovery_clean"),
        ]

        assert len(snapshots) == 5
        for idx, snap in enumerate(snapshots):
            assert snap.stage == expected_stages[idx]
            assert snap.timestamp > 0.0
            assert snap.peak_vram_mb >= 0.0
            assert snap.headroom_mb >= 0.0
            assert snap.telemetry_status in ("AVAILABLE", "UNAVAILABLE")
            assert snap.source in ("live_nvml", "preflight_simulation")

            if snap.telemetry_status == "UNAVAILABLE":
                assert snap.live_vram_used_mb is None
                if snap.base_cuda_context_mb is not None:
                    assert snap.base_cuda_context_mb >= 0.0
                if snap.weights_mb is not None:
                    assert snap.weights_mb >= 0.0
                if snap.workspace_activations_mb is not None:
                    assert snap.workspace_activations_mb >= 0.0

    def test_hardware_memory_profiler_mocked_live_invariants(self):
        """Verify HardwareMemoryProfiler live NVML path correctly populates live metrics."""
        with patch.object(HardwareMemoryProfiler, "is_available", True), \
             patch.object(HardwareMemoryProfiler, "query_live_memory_mb", return_value=7250.25):
            profiler = HardwareMemoryProfiler(device_index=0)
            snap = profiler.profile_stage("inference_peak", base_context_mb=800.0, weights_mb=4000.0)
            assert snap.telemetry_status == "AVAILABLE"
            assert snap.source == "live_nvml"
            assert snap.live_vram_used_mb == 7250.25
            assert snap.peak_vram_mb == 7250.25
            assert snap.headroom_mb == round(15360.0 - 7250.25, 2)
            assert snap.base_cuda_context_mb is None
            assert snap.weights_mb is None

    def test_simulator_records_five_snapshots_for_all_tier1_models(self):
        """Verify PreflightContractSimulator records all 5 snapshots for all 6 Tier 1 models."""
        simulator = PreflightContractSimulator()
        res = simulator.execute_all()
        assert res["all_gates_passed"] is True
        assert len(res["models"]) == 6

        expected_stages = ["pre_load_idle", "post_load", "inference_peak", "post_stress", "post_recovery_clean"]
        for m_id, m_res in res["models"].items():
            assert m_res["certified"] is True
            snaps = m_res["profiling"]["snapshots"]
            assert len(snaps) == 5, f"{m_id} expected 5 snapshots, got {len(snaps)}"
            for idx, s in enumerate(snaps):
                assert s["stage"] == expected_stages[idx]
                assert s["peak_vram_mb"] >= 0.0
                assert s["headroom_mb"] >= 0.0
                assert s["timestamp"] > 0.0

