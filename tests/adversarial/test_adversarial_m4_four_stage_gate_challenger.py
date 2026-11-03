"""
tests/adversarial/test_adversarial_m4_four_stage_gate_challenger.py

Milestone M4 Adversarial Challenger Suite: FourStageVerificationGate Stress & Fault Injection.
Empirically stress-tests and verifies strict failure rejection and boundary enforcement across:
1. Stage 1 (Load):
   - Wrong GPU topology index (GPU 0 model on GPU 1, GPU 1 model on GPU 0, invalid indices).
   - Static VRAM footprint limit breach (> 14.2 GB / 14,540 MB).
   - Strict boundary tests at 14,540.0 MB (PASS) vs 14,540.1 MB (FAIL).
   - CUDA driver overhead bounds (< 400 MB or > 1200 MB).
   - Exclusive admission conflict rejection and eviction.
2. Stage 2 (Infer):
   - Peak VRAM hard limit breach (>= 15.0 GB / 15,360 MB).
   - Strict boundary tests at 15,359.9 MB (PASS) vs 15,360.0 MB (FAIL).
   - Latency anomaly rejection (<= 0 ms).
3. Stage 3 (Validate):
   - Embedding: Unnormalized vectors, boundary tolerance (1.0 +/- 1e-4), NaN/Inf, empty/malformed.
   - Image: Truncated (< 33 bytes), corrupted PNG header, invalid IHDR chunk length/type.
   - Video: Truncated (< 32 bytes), missing ftyp, missing moov, missing mdat boxes.
   - Text/Decision: Empty token streams, NaN/Inf decision floats, invalid types.
   - Capability Contract: Unsupported returning 200, cross-model fallback leaks, execution failures.
4. Stage 4 (Profile):
   - Memory leak threshold (>= 50 MB) rejection.
   - Strict boundary tests at 49.9 MB (PASS) vs 50.0 MB (FAIL).
   - Mid-stress VRAM spike (> 15.0 GB) and intermediate request failure rejection.
   - Minimum 5 sustained requests enforcement.
   - Genuine offline telemetry status (UNAVAILABLE) without mock fabrication.
5. End-to-End Simulation Integration:
   - Simulator rejection of corrupted model profiles.
"""
import math
from pathlib import Path
import struct
from typing import Dict, List, Optional, Tuple
import pytest
import yaml

from deployments.kaggle.certify_kernel import (
    CertificationGateResult,
    CertificationGates,
    DualT4HardwareTopology,
    FourStageVerificationGate,
    HardwareMemoryProfiler,
    ModelCertificationProfile,
    PreflightContractSimulator,
    TIER_1_PROFILES,
)


# =====================================================================
# 1. Stage 1 (Load Gate) Adversarial Stress Tests
# =====================================================================

class TestStage1LoadGateAdversarial:
    """Adversarial challenge for Stage 1 (Load Gate)."""

    def test_rejects_wrong_gpu_index_gpu0_model_on_gpu1(self):
        """GPU 0 models (text, decision, vlm, image) placed on GPU 1 must be rejected."""
        gpu0_models = ["qwen-text", "qwen-decision", "qwen-vlm", "qwen-image-2.1"]
        for m_id in gpu0_models:
            res = FourStageVerificationGate.verify_stage_1_load(
                model_id=m_id,
                gpu_index=1,  # Fault: assigned to GPU 1
                base_cuda_context_mb=800.0,
                weights_mb=4000.0,
                load_time_sec=3.0,
            )
            assert res.passed is False, f"Model {m_id} on GPU 1 should fail Stage 1"
            assert res.gate_name == "Stage 1: Load"
            assert "placed on GPU 1, but hardware topology requires GPU 0" in res.error_message
            assert res.details["gpu_index"] == 1
            assert res.details["expected_gpu"] == 0

    def test_rejects_wrong_gpu_index_gpu1_model_on_gpu0(self):
        """GPU 1 models (embedding, video, music) placed on GPU 0 must be rejected."""
        gpu1_models = ["qwen-embedding", "wan-2.2", "minimax-h3"]
        for m_id in gpu1_models:
            res = FourStageVerificationGate.verify_stage_1_load(
                model_id=m_id,
                gpu_index=0,  # Fault: assigned to GPU 0
                base_cuda_context_mb=750.0,
                weights_mb=2000.0,
                load_time_sec=2.5,
            )
            assert res.passed is False, f"Model {m_id} on GPU 0 should fail Stage 1"
            assert res.gate_name == "Stage 1: Load"
            assert "placed on GPU 0, but hardware topology requires GPU 1" in res.error_message
            assert res.details["gpu_index"] == 0
            assert res.details["expected_gpu"] == 1

    @pytest.mark.parametrize("invalid_gpu", [-1, 2, 3, 99])
    def test_rejects_out_of_range_gpu_indices(self, invalid_gpu):
        """Non-existent GPU indices must be rejected."""
        res = FourStageVerificationGate.verify_stage_1_load(
            model_id="qwen-text",
            gpu_index=invalid_gpu,
            base_cuda_context_mb=800.0,
            weights_mb=4000.0,
            load_time_sec=3.0,
        )
        assert res.passed is False
        assert f"placed on GPU {invalid_gpu}" in res.error_message

    def test_rejects_static_vram_limit_breach_above_14200mb(self):
        """Static footprint exceeding 14.2 GB (14,540 MB) must be rejected."""
        res = FourStageVerificationGate.verify_stage_1_load(
            model_id="qwen-text",
            gpu_index=0,
            base_cuda_context_mb=800.0,
            weights_mb=14000.0,  # Total: 14800 MB > 14540 MB
            load_time_sec=5.0,
        )
        assert res.passed is False
        assert res.gate_name == "Stage 1: Load"
        assert "exceeds Gate 1 ceiling" in res.error_message
        assert res.details["static_footprint_mb"] == 14800.0
        assert res.details["limit_mb"] == 14540.0

    def test_static_vram_footprint_exact_boundaries(self):
        """Strict boundary verification at 14540.0 MB (PASS) vs 14540.1 MB (FAIL)."""
        base_ctx = 800.0

        # Boundary PASS: exactly 14540.0 MB
        res_pass = FourStageVerificationGate.verify_stage_1_load(
            model_id="qwen-text",
            gpu_index=0,
            base_cuda_context_mb=base_ctx,
            weights_mb=14540.0 - base_ctx,
            load_time_sec=4.0,
        )
        assert res_pass.passed is True
        assert res_pass.details["static_footprint_mb"] == 14540.0
        assert res_pass.details["status"] == "HEALTHY"

        # Boundary FAIL: 14540.1 MB
        res_fail = FourStageVerificationGate.verify_stage_1_load(
            model_id="qwen-text",
            gpu_index=0,
            base_cuda_context_mb=base_ctx,
            weights_mb=14540.1 - base_ctx,
            load_time_sec=4.0,
        )
        assert res_fail.passed is False
        assert "exceeds Gate 1 ceiling" in res_fail.error_message

    @pytest.mark.parametrize("bad_ctx", [0.0, 200.0, 399.9, 1200.1, 1500.0, -50.0])
    def test_rejects_cuda_driver_context_out_of_bounds(self, bad_ctx):
        """CUDA driver overhead outside [400, 1200] MB must be rejected."""
        res = FourStageVerificationGate.verify_stage_1_load(
            model_id="qwen-text",
            gpu_index=0,
            base_cuda_context_mb=bad_ctx,
            weights_mb=3000.0,
            load_time_sec=3.0,
        )
        assert res.passed is False
        assert "CUDA driver context overhead out of range" in res.error_message

    def test_topology_exclusive_admission_conflict_rejection(self):
        """DualT4HardwareTopology rejects admitting two large models on same GPU without eviction."""
        topo = DualT4HardwareTopology()
        # Admit model 1 (8000 MB) on GPU 0
        topo.admit_model(0, "qwen-text", 8000.0)

        # Attempt to admit model 2 (7000 MB) on GPU 0 -> Total 15000 MB > 14540 MB
        ok, reason = topo.check_exclusive_admission(0, "qwen-image-2.1", 7000.0)
        assert ok is False
        assert "Exclusive admission conflict on GPU 0" in reason
        assert "must be evicted first" in reason

        with pytest.raises(RuntimeError, match="Exclusive admission conflict"):
            topo.admit_model(0, "qwen-image-2.1", 7000.0)

        # After eviction, admission succeeds
        topo.evict_model(0, "qwen-text")
        topo.admit_model(0, "qwen-image-2.1", 7000.0)
        assert topo._active_models[0] == "qwen-image-2.1"


# =====================================================================
# 2. Stage 2 (Infer Gate) Adversarial Stress Tests
# =====================================================================

class TestStage2InferGateAdversarial:
    """Adversarial challenge for Stage 2 (Infer Gate)."""

    def test_rejects_peak_vram_limit_breach(self):
        """Peak VRAM exceeding 15.0 GB (15,360 MB) must be rejected."""
        res = FourStageVerificationGate.verify_stage_2_infer(
            model_id="qwen-text",
            modality="text",
            peak_vram_mb=15360.0,  # Hard ceiling breach
            latency_ms=150.0,
        )
        assert res.passed is False
        assert res.gate_name == "Stage 2: Infer"
        assert "exceeded 15.0 GB ceiling on Tesla T4" in res.error_message
        assert res.details["peak_vram_mb"] == 15360.0

    def test_peak_vram_exact_boundaries(self):
        """Strict boundary verification at 15359.9 MB (PASS) vs 15360.0 MB (FAIL)."""
        # Boundary PASS: 15359.9 MB
        res_pass = FourStageVerificationGate.verify_stage_2_infer(
            model_id="qwen-text",
            modality="text",
            peak_vram_mb=15359.9,
            latency_ms=120.0,
        )
        assert res_pass.passed is True
        assert res_pass.details["status"] == "PASS"

        # Boundary FAIL: 15360.0 MB
        res_fail = FourStageVerificationGate.verify_stage_2_infer(
            model_id="qwen-text",
            modality="text",
            peak_vram_mb=15360.0,
            latency_ms=120.0,
        )
        assert res_fail.passed is False
        assert "exceeded 15.0 GB ceiling" in res_fail.error_message

    @pytest.mark.parametrize("invalid_latency", [0.0, -1.0, -100.5])
    def test_rejects_invalid_latency_measurements(self, invalid_latency):
        """Latency <= 0 ms must be rejected."""
        res = FourStageVerificationGate.verify_stage_2_infer(
            model_id="qwen-text",
            modality="text",
            peak_vram_mb=10000.0,
            latency_ms=invalid_latency,
        )
        assert res.passed is False
        assert "Invalid latency measured during forward pass" in res.error_message
        assert res.details["latency_ms"] == invalid_latency

    def test_accepts_valid_forward_pass_within_bounds(self):
        """Valid forward pass within peak VRAM and positive latency passes."""
        res = FourStageVerificationGate.verify_stage_2_infer(
            model_id="qwen-text",
            modality="text",
            peak_vram_mb=11500.0,
            latency_ms=185.25,
        )
        assert res.passed is True
        assert res.details["status"] == "PASS"
        assert res.details["latency_ms"] == 185.25


# =====================================================================
# 3. Stage 3 (Validate Gate) Adversarial Stress Tests
# =====================================================================

class TestStage3ValidateGateAdversarial:
    """Adversarial challenge for Stage 3 (Validate Gate)."""

    @pytest.fixture
    def minimal_manifest(self, tmp_path):
        manifest_file = tmp_path / "test-model.yaml"
        manifest_file.write_text(
            """
apiVersion: factory.ai/v1
kind: ModelManifest
metadata:
  id: test-model
capabilities:
  valid_cap: true
  unsupported_cap: false
"""
        )
        return manifest_file

    def _dummy_executor(self, cap: str) -> Tuple[bool, int, Optional[str]]:
        if cap == "valid_cap":
            return True, 200, None
        if cap == "unsupported_cap":
            return False, 400, None
        return False, 404, None

    # --- Embedding Subsystem Verification ---

    def test_rejects_unnormalized_embedding_vector(self, minimal_manifest):
        """Vectors violating unit L2 norm ||v||_2 = 1.0 +/- 1e-4 must be rejected."""
        unnorm_vec = [1.0, 1.0]  # norm = sqrt(2) ~ 1.414214
        res = FourStageVerificationGate.verify_stage_3_validate(
            model_id="qwen-embedding",
            modality="embedding",
            output_payload=unnorm_vec,
            manifest_path=minimal_manifest,
            capability_executor=self._dummy_executor,
        )
        assert res.passed is False
        assert res.gate_name == "Stage 3: Validate"
        assert "violates unit L2 norm" in res.error_message

    def test_embedding_l2_norm_boundary_tolerances(self, minimal_manifest):
        """Tolerance boundary tests: 1.0 +/- 1e-4."""
        # Within tolerance: 1.0 + 9e-5 -> PASS
        v_pass = [1.0 + 9e-5, 0.0]
        res_pass = FourStageVerificationGate.verify_stage_3_validate(
            model_id="qwen-embedding",
            modality="embedding",
            output_payload=v_pass,
            manifest_path=minimal_manifest,
            capability_executor=self._dummy_executor,
        )
        assert res_pass.passed is True

        # Outside tolerance: 1.0 + 1.5e-4 -> FAIL
        v_fail = [1.0 + 1.5e-4, 0.0]
        res_fail = FourStageVerificationGate.verify_stage_3_validate(
            model_id="qwen-embedding",
            modality="embedding",
            output_payload=v_fail,
            manifest_path=minimal_manifest,
            capability_executor=self._dummy_executor,
        )
        assert res_fail.passed is False
        assert "violates unit L2 norm" in res_fail.error_message

    def test_rejects_nan_and_inf_in_embedding_vector(self, minimal_manifest):
        """Embedding vector with NaN or Inf must be rejected."""
        for bad_val in [float("nan"), float("inf"), float("-inf")]:
            bad_vec = [0.6, bad_val]
            res = FourStageVerificationGate.verify_stage_3_validate(
                model_id="qwen-embedding",
                modality="embedding",
                output_payload=bad_vec,
                manifest_path=minimal_manifest,
                capability_executor=self._dummy_executor,
            )
            assert res.passed is False
            assert "violates unit L2 norm" in res.error_message

    def test_rejects_empty_or_non_list_embedding_vector(self, minimal_manifest):
        """Empty vector or non-list payload must be rejected."""
        for invalid_payload in [[], "vector_string", {"vec": [1.0, 0.0]}]:
            res = FourStageVerificationGate.verify_stage_3_validate(
                model_id="qwen-embedding",
                modality="embedding",
                output_payload=invalid_payload,
                manifest_path=minimal_manifest,
                capability_executor=self._dummy_executor,
            )
            assert res.passed is False
            assert "Embedding output must be a non-empty vector list" in res.error_message

    # --- Image Subsystem Verification ---

    def test_rejects_truncated_png_bytes(self, minimal_manifest):
        """PNG payload shorter than 33 bytes must be rejected."""
        short_png = b"\x89PNG\r\n\x1a\n" + b"short"  # 13 bytes
        res = FourStageVerificationGate.verify_stage_3_validate(
            model_id="qwen-image-2.1",
            modality="image",
            output_payload=short_png,
            manifest_path=minimal_manifest,
            capability_executor=self._dummy_executor,
        )
        assert res.passed is False
        assert "too short for valid PNG structure" in res.error_message

    def test_rejects_corrupted_png_signature(self, minimal_manifest):
        """Payload not starting with 8-byte PNG signature must be rejected."""
        bad_sig = b"GIF89a\x00\x00" + (b"\x00" * 30)
        res = FourStageVerificationGate.verify_stage_3_validate(
            model_id="qwen-image-2.1",
            modality="image",
            output_payload=bad_sig,
            manifest_path=minimal_manifest,
            capability_executor=self._dummy_executor,
        )
        assert res.passed is False
        assert "does not start with valid 8-byte PNG header" in res.error_message

    def test_rejects_corrupted_ihdr_chunk_length_or_type(self, minimal_manifest):
        """PNG with IHDR chunk length != 13 or chunk type != b'IHDR' must be rejected."""
        header = b"\x89PNG\r\n\x1a\n"

        # Case 1: Bad chunk type (e.g. IDAT instead of IHDR)
        bad_type = header + struct.pack(">I", 13) + b"IDAT" + (b"\x00" * 17)
        res_type = FourStageVerificationGate.verify_stage_3_validate(
            model_id="qwen-image-2.1",
            modality="image",
            output_payload=bad_type,
            manifest_path=minimal_manifest,
            capability_executor=self._dummy_executor,
        )
        assert res_type.passed is False
        assert "missing mandatory IHDR chunk" in res_type.error_message

        # Case 2: Bad length (e.g. 14 instead of 13)
        bad_len = header + struct.pack(">I", 14) + b"IHDR" + (b"\x00" * 18)
        res_len = FourStageVerificationGate.verify_stage_3_validate(
            model_id="qwen-image-2.1",
            modality="image",
            output_payload=bad_len,
            manifest_path=minimal_manifest,
            capability_executor=self._dummy_executor,
        )
        assert res_len.passed is False
        assert "missing mandatory IHDR chunk" in res_len.error_message

    def test_accepts_valid_png_structure(self, minimal_manifest):
        """Valid PNG header + 13-byte IHDR chunk passes Stage 3."""
        header = b"\x89PNG\r\n\x1a\n"
        ihdr_data = struct.pack(">IIBBBBB", 512, 512, 8, 6, 0, 0, 0)
        ihdr_chunk = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", 0x12345678)
        payload = header + ihdr_chunk + b"\x00" * 20
        res = FourStageVerificationGate.verify_stage_3_validate(
            model_id="qwen-image-2.1",
            modality="image",
            output_payload=payload,
            manifest_path=minimal_manifest,
            capability_executor=self._dummy_executor,
        )
        assert res.passed is True
        assert res.details["artifact_sha256"] is not None

    # --- Video Subsystem Verification ---

    def test_rejects_truncated_or_missing_ftyp_video(self, minimal_manifest):
        """Video missing valid MP4 'ftyp' box must be rejected."""
        short_video = b"\x00\x00\x00\x10bad_box_header"
        res = FourStageVerificationGate.verify_stage_3_validate(
            model_id="wan-2.2",
            modality="video",
            output_payload=short_video,
            manifest_path=minimal_manifest,
            capability_executor=self._dummy_executor,
        )
        assert res.passed is False
        assert "missing valid MP4 'ftyp' box" in res.error_message

    def test_rejects_video_missing_moov_or_mdat_boxes(self, minimal_manifest):
        """Video missing either 'moov' or 'mdat' boxes must be rejected."""
        ftyp = struct.pack(">I", 24) + b"ftyp" + b"isom" + struct.pack(">I", 512) + b"isomiso2"

        # Missing mdat
        no_mdat = ftyp + (b"moov" + b"\x00" * 20)
        res_no_mdat = FourStageVerificationGate.verify_stage_3_validate(
            model_id="wan-2.2",
            modality="video",
            output_payload=no_mdat,
            manifest_path=minimal_manifest,
            capability_executor=self._dummy_executor,
        )
        assert res_no_mdat.passed is False
        assert "missing essential MP4 'moov' or 'mdat' boxes" in res_no_mdat.error_message

        # Missing moov
        no_moov = ftyp + (b"mdat" + b"\x00" * 20)
        res_no_moov = FourStageVerificationGate.verify_stage_3_validate(
            model_id="wan-2.2",
            modality="video",
            output_payload=no_moov,
            manifest_path=minimal_manifest,
            capability_executor=self._dummy_executor,
        )
        assert res_no_moov.passed is False
        assert "missing essential MP4 'moov' or 'mdat' boxes" in res_no_moov.error_message

    def test_accepts_valid_mp4_container(self, minimal_manifest):
        """Valid MP4 container with ftyp, moov, and mdat boxes passes Stage 3."""
        ftyp = struct.pack(">I", 24) + b"ftyp" + b"isom" + struct.pack(">I", 512) + b"isomiso2"
        moov = struct.pack(">I", 16) + b"moov" + b"sample_moov_dat"
        mdat = struct.pack(">I", 32) + b"mdat" + (b"sample_video_bytes" * 2)
        valid_mp4 = ftyp + moov + mdat

        res = FourStageVerificationGate.verify_stage_3_validate(
            model_id="wan-2.2",
            modality="video",
            output_payload=valid_mp4,
            manifest_path=minimal_manifest,
            capability_executor=self._dummy_executor,
        )
        assert res.passed is True
        assert res.details["artifact_sha256"] is not None

    # --- Text & Decision Subsystem Verification ---

    def test_rejects_empty_text_token_stream(self, minimal_manifest):
        """Empty string or whitespace-only token stream must be rejected."""
        for empty_text in ["", "   ", "\n\t"]:
            res = FourStageVerificationGate.verify_stage_3_validate(
                model_id="qwen-text",
                modality="text",
                output_payload=empty_text,
                manifest_path=minimal_manifest,
                capability_executor=self._dummy_executor,
            )
            assert res.passed is False
            assert "Empty token stream received" in res.error_message

    def test_rejects_nan_and_inf_in_decision_output(self, minimal_manifest):
        """Decision output containing NaN or Inf float values must be rejected."""
        for bad_score in [float("nan"), float("inf"), float("-inf")]:
            decision_payload = {"route": "primary", "score": bad_score}
            res = FourStageVerificationGate.verify_stage_3_validate(
                model_id="qwen-decision",
                modality="decision",
                output_payload=decision_payload,
                manifest_path=minimal_manifest,
                capability_executor=self._dummy_executor,
            )
            assert res.passed is False
            assert "Invalid numerical value in Decision output" in res.error_message

    # --- Capability Contract Adherence ---

    def test_rejects_unsupported_capability_leak_or_fallback(self, minimal_manifest):
        """Unsupported capability returning 200 or returning 400 with fallback must be rejected."""
        # Case 1: Returning 200
        def _leaking_exec(cap):
            if cap == "unsupported_cap":
                return True, 200, None
            return True, 200, None

        res_leak = FourStageVerificationGate.verify_stage_3_validate(
            model_id="qwen-text",
            modality="text",
            output_payload="valid output",
            manifest_path=minimal_manifest,
            capability_executor=_leaking_exec,
        )
        assert res_leak.passed is False
        assert "expected 400 Bad Request" in res_leak.error_message

        # Case 2: Returning 400 with fallback target (cross-model fallback forbidden)
        def _fallback_exec(cap):
            if cap == "unsupported_cap":
                return False, 400, "other-model"
            return True, 200, None

        res_fb = FourStageVerificationGate.verify_stage_3_validate(
            model_id="qwen-text",
            modality="text",
            output_payload="valid output",
            manifest_path=minimal_manifest,
            capability_executor=_fallback_exec,
        )
        assert res_fb.passed is False
        assert "cross-model fallback forbidden" in res_fb.error_message


# =====================================================================
# 4. Stage 4 (Profile Gate) Adversarial Stress Tests
# =====================================================================

class TestStage4ProfileGateAdversarial:
    """Adversarial challenge for Stage 4 (Profile Gate)."""

    def test_rejects_memory_leak_equal_to_or_exceeding_50mb(self):
        """Memory leak >= 50.0 MB across sustained requests must be rejected."""
        profiler = HardwareMemoryProfiler(device_index=0)

        # Runner leaks 50.0 MB exactly (idx=0 -> 12000, idx=4 -> 12050)
        def _leak_50mb(idx):
            return True, 12000.0 + (idx * 12.5), 150.0

        res = FourStageVerificationGate.verify_stage_4_profile(
            model_id="qwen-text",
            profiler=profiler,
            sustained_requests_count=5,
            request_runner=_leak_50mb,
        )
        assert res.passed is False
        assert res.gate_name == "Stage 4: Profile"
        assert "Memory leak detected" in res.error_message
        assert res.details["vram_leak_mb"] == 50.0

    def test_memory_leak_exact_boundaries(self):
        """Strict boundary verification at 49.9 MB (PASS) vs 50.0 MB (FAIL)."""
        profiler = HardwareMemoryProfiler(device_index=0)

        # Boundary PASS: 49.9 MB leak
        def _runner_49_9(idx):
            return True, 12000.0 + (idx * (49.9 / 4.0)), 150.0

        res_pass = FourStageVerificationGate.verify_stage_4_profile(
            model_id="qwen-text",
            profiler=profiler,
            sustained_requests_count=5,
            request_runner=_runner_49_9,
        )
        assert res_pass.passed is True
        assert res_pass.details["vram_leak_mb"] == 49.9

        # Boundary FAIL: 50.0 MB leak
        def _runner_50_0(idx):
            return True, 12000.0 + (idx * (50.0 / 4.0)), 150.0

        res_fail = FourStageVerificationGate.verify_stage_4_profile(
            model_id="qwen-text",
            profiler=profiler,
            sustained_requests_count=5,
            request_runner=_runner_50_0,
        )
        assert res_fail.passed is False
        assert "Memory leak detected" in res_fail.error_message

    def test_rejects_vram_spike_exceeding_15000mb_during_stress(self):
        """Any VRAM spike >= 15,360 MB during sustained stress must fail the gate."""
        profiler = HardwareMemoryProfiler(device_index=0)

        def _spiking_runner(idx):
            if idx == 3:
                return True, 15360.0, 200.0  # Spike at request 4
            return True, 12000.0, 150.0

        res = FourStageVerificationGate.verify_stage_4_profile(
            model_id="qwen-text",
            profiler=profiler,
            sustained_requests_count=5,
            request_runner=_spiking_runner,
        )
        assert res.passed is False
        assert "exceeded 15.0 GB" in res.error_message
        assert res.details["max_peak_vram_mb"] == 15360.0

    def test_rejects_failed_request_during_stress(self):
        """If any request in sustained stress returns success=False, Stage 4 fails."""
        profiler = HardwareMemoryProfiler(device_index=0)

        def _failing_runner(idx):
            if idx == 2:
                return False, 12000.0, 0.0
            return True, 12000.0, 150.0

        res = FourStageVerificationGate.verify_stage_4_profile(
            model_id="qwen-text",
            profiler=profiler,
            sustained_requests_count=5,
            request_runner=_failing_runner,
        )
        assert res.passed is False
        assert "Request 3/5 failed during stress test" in res.error_message
        assert res.details["failed_at_index"] == 2

    def test_enforces_minimum_5_sustained_requests(self):
        """Passing count < 5 automatically executes at least 5 requests."""
        profiler = HardwareMemoryProfiler(device_index=0)
        call_count = 0

        def _tracking_runner(idx):
            nonlocal call_count
            call_count += 1
            return True, 12000.0, 100.0

        res = FourStageVerificationGate.verify_stage_4_profile(
            model_id="qwen-text",
            profiler=profiler,
            sustained_requests_count=2,  # Adversarial: requested only 2
            request_runner=_tracking_runner,
        )
        assert res.passed is True
        assert call_count >= 5
        assert res.details["stress_profile"]["requests_executed"] >= 5

    def test_telemetry_status_unavailable_offline_without_fabrication(self):
        """On local/offline environment, profiler returns UNAVAILABLE without fake numbers."""
        profiler = HardwareMemoryProfiler(device_index=0)
        assert profiler.telemetry_status == "UNAVAILABLE"
        assert profiler.query_live_memory_mb() is None
        metrics = profiler.query_device_metrics()
        assert metrics["telemetry_status"] == "UNAVAILABLE"
        assert metrics["vram_used_mb"] is None


# =====================================================================
# 5. Preflight Simulator Adversarial Profile Injection
# =====================================================================

class TestPreflightSimulatorAdversarialInjection:
    """Stress-test PreflightContractSimulator when injected with adversarial profiles."""

    def test_simulator_detects_and_rejects_vram_overload_profile(self, tmp_path):
        """Simulator must fail model certification if profile exceeds limits."""
        manifests_dir = Path(__file__).resolve().parent.parent.parent.parent / "manifests"
        sim = PreflightContractSimulator(manifests_dir=manifests_dir)

        # Overloaded profile: weights 14,000 MB + context 820 MB = 14,820 MB > 14,540 MB
        bad_profile = ModelCertificationProfile(
            model_id="qwen-text-overloaded",
            manifest_id="qwen-text",
            family="qwen-text",
            subsystem="text-service",
            target_gpu=0,
            weights_checkpoint="Qwen/Qwen2.5-7B-Instruct-AWQ",
            quantization="awq",
            base_cuda_context_mb=820.0,
            weights_mb=14000.0,
            workspace_mb=1000.0,
            declared_capabilities=["chat_completion"],
            unsupported_capabilities=[],
        )

        res = sim.certify_model(bad_profile)
        assert res["certified"] is False
        assert res["failed_gate"] == "Gate 1"
        assert "exceeds Gate 1 ceiling" in res["error"]
        # Confirm GPU was evicted cleanly upon failure
        assert sim.topology._active_models[0] is None
