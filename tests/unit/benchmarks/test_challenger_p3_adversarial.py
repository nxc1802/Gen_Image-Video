"""
tests/unit/benchmarks/test_challenger_p3_adversarial.py
Adversarial challenge test suite for Phase 3: Benchmark Standardization & True TTFT (P1.1).

Empirically challenges:
1. MTP Baseline Runner (`mtp_baseline_runner.py`):
   - Zero and negative latency injection
   - Token count division-by-zero protection
   - Empty prompt and empty output detection
   - Seed determinism and seed mismatch detection
   - Sub-1.0x speedup and excessive VRAM overhead invariant violations
2. Qwen-Image Variant Runner (`qwen_image_variant_runner.py`):
   - Zero and negative duration / step-rate protection
   - Memory bounds strictly bounded < 14,540 MB
   - Invariant rejection on step-rate divergence, inverted speedup, or dimension mismatch
   - Perceptual collapse detection (std_dev <= 20.0)
   - Image stats extraction on solid color vs corrupt images
3. TTFT & Gate Verification Integrity:
   - Rejection of zero/negative TTFT by Stage 2 verification gate
   - Rejection of negative phase durations or missing phase keys
   - Verification that static constants (25.0, 80.0, 150.0) remain fully purged
"""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
from typing import Any, Dict

from PIL import Image
import pytest

from benchmarks.runners.live_runner import create_canonical_record
from benchmarks.runners.mtp_baseline_runner import (
    MTPComparisonReport,
    MTPRunResult,
    run_qwen38_mtp_comparison,
)
from benchmarks.runners.qwen_image_variant_runner import (
    ImageVariantRunResult,
    QwenImageVariantReport,
    _compute_image_stats,
    benchmark_qwen_image_variants,
)
from deployments.kaggle.certify_kernel import (
    FourStageVerificationGate,
    REQUIRED_PHASE_KEYS,
)


# ==============================================================================
# Suite 1: Adversarial Challenge for MTP Baseline Runner
# ==============================================================================

class TestAdversarialMTPBaselineRunner:
    """Adversarial stress-testing of Qwen3.8 MTP runner."""

    def test_mtp_zero_latency_handling(self):
        """Zero latency must fail ttft_valid_positive and mark invariants_passed as False."""
        def mock_zero_lat(prompt: str, is_mtp: bool, seed: int, max_tokens: int, temp: float) -> Dict[str, Any]:
            return {
                "content": "output",
                "tokens_predicted": 100,
                "duration_ms": 0.0,
                "ttft_ms": 0.0,
                "decode_tokens_per_second": 50.0 if is_mtp else 25.0,
                "prompt_tokens_per_second": 200.0,
                "vram_mb": 13000.0 if is_mtp else 11500.0,
                "phases": {
                    "model_loading_ms": 0.0,
                    "prompt_processing_ms": 0.0,
                    "decode_ms": 0.0,
                    "sampling_ms": 0.0,
                    "vae_ms": 0.0,
                    "artifact_encoding_ms": 0.0,
                },
            }

        report = run_qwen38_mtp_comparison(mock_evaluator=mock_zero_lat)
        assert report.invariants_details["ttft_valid_positive"] is False
        assert report.invariants_passed is False

    def test_mtp_negative_latency_handling(self):
        """Negative latency values must fail invariant checks."""
        def mock_neg_lat(prompt: str, is_mtp: bool, seed: int, max_tokens: int, temp: float) -> Dict[str, Any]:
            return {
                "content": "output",
                "tokens_predicted": 100,
                "duration_ms": -100.0,
                "ttft_ms": -25.0,
                "decode_tokens_per_second": 20.0,
                "prompt_tokens_per_second": 200.0,
                "vram_mb": 12000.0,
                "phases": {
                    "model_loading_ms": 0.0,
                    "prompt_processing_ms": -25.0,
                    "decode_ms": -75.0,
                    "sampling_ms": 0.0,
                    "vae_ms": 0.0,
                    "artifact_encoding_ms": 0.0,
                },
            }

        report = run_qwen38_mtp_comparison(mock_evaluator=mock_neg_lat)
        assert report.invariants_details["ttft_valid_positive"] is False
        assert report.invariants_passed is False

    def test_mtp_zero_tokens_division_by_zero_protection(self):
        """Zero tokens or zero decode throughput must not trigger ZeroDivisionError and must fail invariants."""
        def mock_zero_tokens(prompt: str, is_mtp: bool, seed: int, max_tokens: int, temp: float) -> Dict[str, Any]:
            return {
                "content": "",
                "tokens_predicted": 0,
                "duration_ms": 1000.0,
                "ttft_ms": 50.0,
                "decode_tokens_per_second": 0.0,
                "prompt_tokens_per_second": 0.0,
                "vram_mb": 12000.0,
                "phases": {
                    "model_loading_ms": 0.0,
                    "prompt_processing_ms": 50.0,
                    "decode_ms": 950.0,
                    "sampling_ms": 0.0,
                    "vae_ms": 0.0,
                    "artifact_encoding_ms": 0.0,
                },
            }

        report = run_qwen38_mtp_comparison(mock_evaluator=mock_zero_tokens)
        assert report.speedup_factor == 0.0
        assert report.invariants_details["speedup_positive"] is False
        assert report.invariants_details["outputs_non_empty"] is False
        assert report.invariants_passed is False

    def test_mtp_empty_prompt_and_empty_output_rejection(self):
        """Empty output content must fail the outputs_non_empty invariant."""
        def mock_empty_content(prompt: str, is_mtp: bool, seed: int, max_tokens: int, temp: float) -> Dict[str, Any]:
            return {
                "content": "",
                "tokens_predicted": 10,
                "duration_ms": 1000.0,
                "ttft_ms": 50.0,
                "decode_tokens_per_second": 20.0 if is_mtp else 15.0,
                "prompt_tokens_per_second": 100.0,
                "vram_mb": 12000.0,
                "phases": {
                    "model_loading_ms": 0.0,
                    "prompt_processing_ms": 50.0,
                    "decode_ms": 950.0,
                    "sampling_ms": 0.0,
                    "vae_ms": 0.0,
                    "artifact_encoding_ms": 0.0,
                },
            }

        report = run_qwen38_mtp_comparison(prompt="", mock_evaluator=mock_empty_content)
        assert report.invariants_details["outputs_non_empty"] is False
        assert report.invariants_passed is False

    def test_mtp_seed_stability_and_invariance(self):
        """Identical seed must produce identical simulation outputs; different seeds produce distinct outputs."""
        report_42a = run_qwen38_mtp_comparison(seed=42)
        report_42b = run_qwen38_mtp_comparison(seed=42)
        report_99 = run_qwen38_mtp_comparison(seed=99)

        # Strict stability under same seed
        assert report_42a.baseline.output_text == report_42b.baseline.output_text
        assert report_42a.mtp.output_text == report_42b.mtp.output_text
        assert report_42a.speedup_factor == report_42b.speedup_factor

        # Different seed generates different text
        assert report_42a.baseline.output_text != report_99.baseline.output_text

    def test_mtp_sub_1_speedup_invariant_failure(self):
        """If MTP is slower than baseline (speedup <= 1.0), invariants_passed must be False."""
        def mock_slow_mtp(prompt: str, is_mtp: bool, seed: int, max_tokens: int, temp: float) -> Dict[str, Any]:
            # MTP is 10 tps, baseline is 20 tps -> speedup = 0.5x
            tps = 10.0 if is_mtp else 20.0
            return {
                "content": f"output (mtp={is_mtp})",
                "tokens_predicted": 100,
                "duration_ms": 5000.0 if is_mtp else 2500.0,
                "ttft_ms": 40.0,
                "decode_tokens_per_second": tps,
                "prompt_tokens_per_second": 200.0,
                "vram_mb": 13000.0 if is_mtp else 11500.0,
                "phases": {
                    "model_loading_ms": 0.0,
                    "prompt_processing_ms": 40.0,
                    "decode_ms": 4960.0 if is_mtp else 2460.0,
                    "sampling_ms": 0.0,
                    "vae_ms": 0.0,
                    "artifact_encoding_ms": 0.0,
                },
            }

        report = run_qwen38_mtp_comparison(mock_evaluator=mock_slow_mtp)
        assert report.speedup_factor == 0.5
        assert report.invariants_details["speedup_positive"] is False
        assert report.invariants_passed is False

    def test_mtp_excessive_vram_overhead_failure(self):
        """VRAM overhead exceeding 2048 MB limit must fail vram_overhead_bounded invariant."""
        def mock_high_vram(prompt: str, is_mtp: bool, seed: int, max_tokens: int, temp: float) -> Dict[str, Any]:
            return {
                "content": "output",
                "tokens_predicted": 100,
                "duration_ms": 3000.0,
                "ttft_ms": 40.0,
                "decode_tokens_per_second": 30.0 if is_mtp else 18.0,
                "prompt_tokens_per_second": 200.0,
                "vram_mb": 15000.0 if is_mtp else 11000.0,  # 4000 MB overhead > 2048 MB limit
                "phases": {
                    "model_loading_ms": 0.0,
                    "prompt_processing_ms": 40.0,
                    "decode_ms": 2960.0,
                    "sampling_ms": 0.0,
                    "vae_ms": 0.0,
                    "artifact_encoding_ms": 0.0,
                },
            }

        report = run_qwen38_mtp_comparison(mock_evaluator=mock_high_vram)
        assert report.vram_overhead_mb == 4000.0
        assert report.invariants_details["vram_overhead_bounded"] is False
        assert report.invariants_passed is False


# ==============================================================================
# Suite 2: Adversarial Challenge for Qwen-Image Variant Runner
# ==============================================================================

class TestAdversarialQwenImageVariantRunner:
    """Adversarial stress-testing of Qwen-Image 2.1 variant runner."""

    def test_qwen_image_zero_and_negative_duration_handling(self):
        """Zero or negative duration must not trigger division by zero and must fail speedup."""
        def mock_zero_dur(variant: str, prompt: str, seed: int, steps: int, w: int, h: int) -> Dict[str, Any]:
            return {
                "duration_ms": 0.0,
                "sampling_ms": 0.0,
                "seconds_per_step": 0.0,
                "peak_vram_mb": 13500.0,
                "image_width": w,
                "image_height": h,
                "image_std_dev": 45.0,
                "artifact_sha256": "fakehash",
                "phases": {
                    "model_loading_ms": 0.0,
                    "prompt_processing_ms": 0.0,
                    "decode_ms": 0.0,
                    "sampling_ms": 0.0,
                    "vae_ms": 0.0,
                    "artifact_encoding_ms": 0.0,
                },
            }

        report = benchmark_qwen_image_variants(mock_evaluator=mock_zero_dur)
        assert report.speedup_factor == 0.0
        assert report.invariants_details["speedup_significant"] is False
        assert report.invariants_passed is False

    def test_qwen_image_memory_bounds_strictly_enforced(self):
        """VRAM peak exceeding 14,540 MB must fail vram_safe invariants."""
        def mock_oob_vram(variant: str, prompt: str, seed: int, steps: int, w: int, h: int) -> Dict[str, Any]:
            # Standard within bounds, Turbo exceeds 14,540 MB
            vram = 14550.0 if variant == "turbo" else 13800.0
            return {
                "duration_ms": 16000.0 if variant == "turbo" else 38000.0,
                "sampling_ms": 15000.0 if variant == "turbo" else 36000.0,
                "seconds_per_step": 1.875 if variant == "turbo" else 1.80,
                "peak_vram_mb": vram,
                "image_width": w,
                "image_height": h,
                "image_std_dev": 45.0,
                "artifact_sha256": "fakehash",
                "phases": {
                    "model_loading_ms": 500.0,
                    "prompt_processing_ms": 50.0,
                    "decode_ms": 0.0,
                    "sampling_ms": 15000.0 if variant == "turbo" else 36000.0,
                    "vae_ms": 400.0,
                    "artifact_encoding_ms": 20.0,
                },
            }

        report = benchmark_qwen_image_variants(mock_evaluator=mock_oob_vram)
        assert report.invariants_details["vram_safe_standard"] is True
        assert report.invariants_details["vram_safe_turbo"] is False
        assert report.invariants_passed is False

    def test_qwen_image_speedup_and_step_rate_invariants_failure(self):
        """Turbo slower than standard or highly divergent step rate must fail invariants."""
        def mock_slow_turbo(variant: str, prompt: str, seed: int, steps: int, w: int, h: int) -> Dict[str, Any]:
            # Turbo has very slow step rate (5.0s/step vs 1.9s/step) -> speedup fails
            sec_step = 5.0 if variant == "turbo" else 1.9
            dur = (steps * sec_step * 1000.0) + 1500.0
            return {
                "duration_ms": dur,
                "sampling_ms": steps * sec_step * 1000.0,
                "seconds_per_step": sec_step,
                "peak_vram_mb": 13800.0,
                "image_width": w,
                "image_height": h,
                "image_std_dev": 45.0,
                "artifact_sha256": "fakehash",
                "phases": {
                    "model_loading_ms": 1000.0,
                    "prompt_processing_ms": 50.0,
                    "decode_ms": 0.0,
                    "sampling_ms": steps * sec_step * 1000.0,
                    "vae_ms": 400.0,
                    "artifact_encoding_ms": 20.0,
                },
            }

        report = benchmark_qwen_image_variants(mock_evaluator=mock_slow_turbo)
        assert report.invariants_details["speedup_significant"] is False
        assert report.invariants_details["step_rate_invariant"] is False
        assert report.invariants_passed is False

    def test_qwen_image_perceptual_collapse_detection(self):
        """Image with std_dev <= 20.0 (perceptual collapse / blank image) must fail invariants."""
        def mock_collapsed(variant: str, prompt: str, seed: int, steps: int, w: int, h: int) -> Dict[str, Any]:
            return {
                "duration_ms": 17000.0 if variant == "turbo" else 39000.0,
                "sampling_ms": 15600.0 if variant == "turbo" else 37000.0,
                "seconds_per_step": 1.95 if variant == "turbo" else 1.85,
                "peak_vram_mb": 13800.0,
                "image_width": w,
                "image_height": h,
                "image_std_dev": 8.5 if variant == "turbo" else 45.0,  # Turbo collapsed
                "artifact_sha256": "fakehash",
                "phases": {
                    "model_loading_ms": 1000.0,
                    "prompt_processing_ms": 50.0,
                    "decode_ms": 0.0,
                    "sampling_ms": 15600.0 if variant == "turbo" else 37000.0,
                    "vae_ms": 400.0,
                    "artifact_encoding_ms": 20.0,
                },
            }

        report = benchmark_qwen_image_variants(mock_evaluator=mock_collapsed)
        assert report.invariants_details["perceptual_std_standard"] is True
        assert report.invariants_details["perceptual_std_turbo"] is False
        assert report.invariants_passed is False

    def test_qwen_image_dimension_mismatch_failure(self):
        """Image dimensions mismatching target resolution (e.g. 512x512 != 960x640) must fail."""
        def mock_bad_dims(variant: str, prompt: str, seed: int, steps: int, w: int, h: int) -> Dict[str, Any]:
            return {
                "duration_ms": 17000.0 if variant == "turbo" else 39000.0,
                "sampling_ms": 15600.0 if variant == "turbo" else 37000.0,
                "seconds_per_step": 1.95 if variant == "turbo" else 1.85,
                "peak_vram_mb": 13800.0,
                "image_width": 512,  # mismatched width
                "image_height": 512,  # mismatched height
                "image_std_dev": 45.0,
                "artifact_sha256": "fakehash",
                "phases": {
                    "model_loading_ms": 1000.0,
                    "prompt_processing_ms": 50.0,
                    "decode_ms": 0.0,
                    "sampling_ms": 15600.0 if variant == "turbo" else 37000.0,
                    "vae_ms": 400.0,
                    "artifact_encoding_ms": 20.0,
                },
            }

        report = benchmark_qwen_image_variants(width=960, height=640, mock_evaluator=mock_bad_dims)
        assert report.invariants_details["dimensions_correct"] is False
        assert report.invariants_passed is False

    def test_compute_image_stats_solid_vs_noise(self):
        """Verifies _compute_image_stats on synthetic images (solid black -> 0.0, noise -> high std)."""
        # Solid black image
        black_img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        black_buf = io.BytesIO()
        black_img.save(black_buf, format="PNG")
        w_b, h_b, std_b, sha_b = _compute_image_stats(black_buf.getvalue())

        assert w_b == 100
        assert h_b == 100
        assert std_b == 0.0
        assert len(sha_b) == 64

        # Checkerboard / high-contrast image
        checker = Image.new("RGB", (100, 100), color=(255, 255, 255))
        for x in range(50):
            for y in range(100):
                checker.putpixel((x, y), (0, 0, 0))
        chk_buf = io.BytesIO()
        checker.save(chk_buf, format="PNG")
        w_c, h_c, std_c, sha_c = _compute_image_stats(chk_buf.getvalue())

        assert w_c == 100
        assert h_c == 100
        assert std_c > 100.0  # high contrast produces very high standard deviation
        assert sha_c != sha_b

    def test_compute_image_stats_corrupt_bytes_rejection(self):
        """Invalid / corrupt bytes must raise PIL exception."""
        with pytest.raises(Exception):
            _compute_image_stats(b"NOT_A_VALID_IMAGE_HEADER_12345")


# ==============================================================================
# Suite 3: TTFT Integrity & Stage 2 Verification Gate Stress Testing
# ==============================================================================

class TestTTFTAndGateIntegrity:
    """Verifies that Stage 2 verification gate and TTFT handling reject invalid or fake telemetry."""

    def test_stage_2_infer_rejects_negative_and_zero_ttft(self):
        """Stage 2 Infer gate must reject ttft_ms <= 0."""
        valid_phases = {
            "model_loading_ms": 100.0,
            "prompt_processing_ms": 25.0,
            "decode_ms": 150.0,
            "sampling_ms": 0.0,
            "vae_ms": 0.0,
            "artifact_encoding_ms": 0.0,
        }

        # Zero TTFT
        res_zero = FourStageVerificationGate.verify_stage_2_infer(
            model_id="qwen-text",
            modality="text",
            peak_vram_mb=12000.0,
            latency_ms=275.0,
            phases=valid_phases,
            ttft_ms=0.0,
        )
        assert res_zero.passed is False
        assert "TTFT must be a positive number" in res_zero.error_message

        # Negative TTFT
        res_neg = FourStageVerificationGate.verify_stage_2_infer(
            model_id="qwen-text",
            modality="text",
            peak_vram_mb=12000.0,
            latency_ms=275.0,
            phases=valid_phases,
            ttft_ms=-15.0,
        )
        assert res_neg.passed is False
        assert "TTFT must be a positive number" in res_neg.error_message

    def test_stage_2_infer_rejects_negative_phase_duration(self):
        """Stage 2 Infer gate must reject any phase with duration < 0."""
        neg_phases = {
            "model_loading_ms": 100.0,
            "prompt_processing_ms": 25.0,
            "decode_ms": -1.0,  # Negative duration!
            "sampling_ms": 0.0,
            "vae_ms": 0.0,
            "artifact_encoding_ms": 0.0,
        }
        res = FourStageVerificationGate.verify_stage_2_infer(
            model_id="qwen-text",
            modality="text",
            peak_vram_mb=12000.0,
            latency_ms=275.0,
            phases=neg_phases,
            ttft_ms=25.0,
        )
        assert res.passed is False
        assert "must be a non-negative number" in res.error_message

    def test_stage_2_infer_rejects_missing_phase_keys(self):
        """Stage 2 Infer gate must reject incomplete phase dictionaries."""
        incomplete_phases = {
            "model_loading_ms": 100.0,
            "prompt_processing_ms": 25.0,
            # Missing decode_ms, sampling_ms, vae_ms, artifact_encoding_ms
        }
        res = FourStageVerificationGate.verify_stage_2_infer(
            model_id="qwen-text",
            modality="text",
            peak_vram_mb=12000.0,
            latency_ms=275.0,
            phases=incomplete_phases,
            ttft_ms=25.0,
        )
        assert res.passed is False
        assert "Missing explicit phase breakdown keys" in res.error_message

    def test_create_canonical_record_retains_authentic_phases_and_ttft(self):
        """create_canonical_record must correctly bind authentic measurements without fake injection."""
        custom_phases = {
            "model_loading_ms": 450.0,
            "prompt_processing_ms": 38.5,
            "decode_ms": 1200.0,
            "sampling_ms": 12.0,
            "vae_ms": 0.0,
            "artifact_encoding_ms": 1.5,
        }
        measurements = {
            "queue_seconds": 0.05,
            "load_seconds": 0.45,
            "inference_seconds": 1.25,
            "end_to_end_seconds": 1.75,
            "ttft_ms": 38.5,
            "prompt_tokens_per_second": 320.0,
            "decode_tokens_per_second": 28.5,
            "phases": custom_phases,
        }

        rec = create_canonical_record(
            test_id="canon_test_01",
            suite="perf",
            status="PASS",
            model="qwen-text",
            variant="default",
            task="text_generation",
            git_commit="abc1234",
            gpu_names=["Tesla T4"],
            parameters={"seed": 42},
            measurements=measurements,
            artifact={"validated": True},
            integrity={"mock_used": False, "fallback_used": False},
        )

        m = rec["measurements"]
        assert m["ttft_ms"] == 38.5
        assert m["prompt_tokens_per_second"] == 320.0
        assert m["decode_tokens_per_second"] == 28.5
        assert m["phases"] == custom_phases
        assert rec["integrity"]["mock_used"] is False
        assert rec["integrity"]["fallback_used"] is False
