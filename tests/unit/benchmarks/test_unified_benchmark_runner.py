"""
tests/unit/benchmarks/test_unified_benchmark_runner.py
Unit test suite for Phase 3 Unified Kaggle Dual-T4 Benchmark:
1. 100% Manifest Task & Variant Coverage in cases.yaml across all 11 model manifests.
2. Robust Multi-Modal Artifact Validators (MIME, container magic headers, dimensions, durations, SHA-256).
3. Discrete Dual-T4 Memory Guard (14.2 GB max allocatable per GPU: GPU 0 vs GPU 1).
4. Strict Status Semantics (PASS, FAIL, HARDWARE_BLOCKED, BACKEND_UNAVAILABLE, NOT_IMPLEMENTED).
5. Comprehensive Report Generation (results.json, metrics.csv, summary.md).
6. Unified Kaggle Bootstrap Environment & Dependency Probing.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import struct
import tempfile
from typing import Any, Dict, List
import wave

import pytest
import yaml

from benchmarks.report import BenchmarkReportGenerator, CaseExecutionRecord
from benchmarks.runner import DualT4MemoryGuard, NVMLSampler, UnifiedBenchmarkRunner
from benchmarks.validators import (
    extract_mp4_metadata,
    extract_png_metadata,
    extract_wav_metadata,
    validate_abc_score,
    validate_artifact,
    validate_decision_schema,
    validate_embedding_vector,
)
from deployments.kaggle.bootstrap import HardwareReport, KaggleBootstrap
from model_registry.loader import ManifestLoader

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# =============================================================================
# 1. Manifest 100% Task & Variant Coverage Tests
# =============================================================================

def test_cases_cover_all_11_model_manifests():
    """Verifies cases.yaml covers 100% of declared tasks and variants for all 11 models."""
    cases_path = WORKSPACE_ROOT / "benchmarks" / "cases.yaml"
    assert cases_path.exists(), "benchmarks/cases.yaml must exist."

    with open(cases_path, "r", encoding="utf-8") as f:
        cases_doc = yaml.safe_load(f)
    cases = cases_doc.get("cases", [])
    assert len(cases) >= 80, f"Expected at least 80 cases, found {len(cases)}"

    loader = ManifestLoader(WORKSPACE_ROOT / "manifests")
    models = loader.load_models()
    assert len(models) == 11, f"Expected 11 models in registry, found {len(models)}"

    # Build index of (model, task, variant) from cases.yaml
    case_index = set()
    for c in cases:
        case_index.add((c["model"], c["task"], c.get("variant", "standard")))

    # Verify every declared task-variant pair in each manifest is covered
    missing_coverage = []
    total_declared_pairs = 0

    for model_id, manifest in models.items():
        matrix = getattr(manifest, "task_variant_matrix", {})
        if not matrix:
            continue
        for task_name, variants in matrix.items():
            for var in variants:
                total_declared_pairs += 1
                if (model_id, task_name, var) not in case_index:
                    missing_coverage.append((model_id, task_name, var))

    assert not missing_coverage, (
        f"Missing benchmark cases for declared task-variant pairs ({len(missing_coverage)} missing): "
        f"{missing_coverage}"
    )
    assert total_declared_pairs == 82, f"Expected exactly 82 declared pairs across 11 manifests, got {total_declared_pairs}"


# =============================================================================
# 2. Artifact Validation & Container Header Tests
# =============================================================================

def test_validators_png_magic_and_dimensions():
    """Validates PNG binary container magic bytes, IHDR chunk dimensions, and size checks."""
    # Synthetic valid PNG header with width=800, height=600
    ihdr_data = struct.pack(">IIBBBBB", 800, 600, 8, 6, 0, 0, 0)
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + ihdr_data + b"\x00\x00\x00\x00"

    res = validate_artifact(
        content=png_bytes,
        expected_mime="image/png",
        expected_format="png",
        min_bytes=10,
        rules={"exact_width": 800, "exact_height": 600},
    )
    assert res.valid is True
    assert res.mime_type == "image/png"
    assert res.format == "png"
    assert res.dimensions == (800, 600)
    assert res.sha256 is not None
    assert res.size_bytes == len(png_bytes)

    # Corrupt header should fail
    corrupt_bytes = b"\x00PNG\r\n\x1a\n" + png_bytes[8:]
    res_corrupt = validate_artifact(corrupt_bytes, expected_mime="image/png")
    assert res_corrupt.valid is False
    assert "MIME type mismatch" in res_corrupt.error or "magic header" in res_corrupt.error

    # Below minimum size should fail
    res_size = validate_artifact(png_bytes, min_bytes=1000)
    assert res_size.valid is False
    assert "below minimum threshold" in res_size.error


def test_validators_wav_audio_and_duration():
    """Validates WAV binary container, RIFF headers, channels, and duration bounds."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        # Write 2.0 seconds of audio data
        wf.writeframes(b"\x00\x00\x00\x00" * 88200)

    wav_bytes = buf.getvalue()
    res = validate_artifact(
        content=wav_bytes,
        expected_mime="audio/wav",
        expected_format="wav",
        min_bytes=100,
        rules={"min_duration": 1.5, "max_duration": 3.0},
    )
    assert res.valid is True
    assert res.mime_type == "audio/wav"
    assert res.format == "wav"
    assert res.duration_seconds == 2.0
    assert res.metadata["channels"] == 2
    assert res.metadata["framerate_hz"] == 44100

    # Duration violation check
    res_too_short = validate_artifact(wav_bytes, rules={"min_duration": 5.0})
    assert res_too_short.valid is False
    assert "Media duration" in res_too_short.error and "< min" in res_too_short.error


def test_validators_mp4_video_container():
    """Validates MP4 binary container ftyp box, dimensions, and duration parsing."""
    # Synthetic MP4 with ftyp box and moov/mvhd box
    ftyp_payload = b"isom\x00\x00\x02\x00isomiso2mp41"
    ftyp_box = struct.pack(">I4s", len(ftyp_payload) + 8, b"ftyp") + ftyp_payload

    # mvhd box with timescale 1000, duration 3000 (3.0s)
    mvhd_payload = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" + struct.pack(">II", 1000, 3000) + (b"\x00" * 80)
    mvhd_box = struct.pack(">I4s", len(mvhd_payload) + 8, b"mvhd") + mvhd_payload
    moov_box = struct.pack(">I4s", len(mvhd_box) + 8, b"moov") + mvhd_box

    mp4_bytes = ftyp_box + moov_box

    res = validate_artifact(
        content=mp4_bytes,
        expected_mime="video/mp4",
        expected_format="mp4",
        min_bytes=32,
    )
    assert res.valid is True
    assert res.mime_type == "video/mp4"
    assert res.format == "mp4"
    assert res.duration_seconds == 3.0
    assert res.sha256 is not None


def test_validators_decision_schema():
    """Validates structured decision / routing schemas and confidence ranges."""
    valid_decision = {"choice": "video-service", "confidence": 0.95, "reasoning": "User prompt requests video generation."}
    ok, err = validate_decision_schema(valid_decision)
    assert ok is True
    assert err is None

    # Invalid confidence > 1.0
    bad_conf = {"choice": "video-service", "confidence": 1.5}
    ok_bad, err_bad = validate_decision_schema(bad_conf)
    assert ok_bad is False
    assert "Confidence 1.5 out of valid [0.0, 1.0] range" in err_bad

    # Missing decision fields
    bad_empty = {"foo": "bar"}
    ok_empty, err_empty = validate_decision_schema(bad_empty)
    assert ok_empty is False
    assert "missing decision/choice/score fields" in err_empty


def test_validators_embedding_vector():
    """Validates embedding vectors, dimensions, and L2 norms."""
    vec = [0.1 * i for i in range(1024)]
    ok, err, meta = validate_embedding_vector(vec, expected_dim=1024)
    assert ok is True
    assert err is None
    assert meta["dim"] == 1024
    assert meta["l2_norm"] > 0.0

    # Zero vector should fail
    zero_vec = [0.0] * 1024
    ok_z, err_z, _ = validate_embedding_vector(zero_vec, expected_dim=1024)
    assert ok_z is False
    assert "zero L2 norm" in err_z

    # Dimension mismatch
    ok_dim, err_dim, _ = validate_embedding_vector(vec, expected_dim=512)
    assert ok_dim is False
    assert "dimension mismatch" in err_dim


def test_validators_yue2_abc_score():
    """Validates YuE2 ABC score format and headers."""
    valid_score = "X:1\nT:Test Melody\nM:4/4\nK:C\nCDEF GABc|"
    ok, err = validate_abc_score(valid_score)
    assert ok is True
    assert err is None

    # Missing mandatory header K:
    bad_score = "X:1\nT:Incomplete Melody\nM:4/4\nCDEF GABc|"
    ok_b, err_b = validate_abc_score(bad_score)
    assert ok_b is False
    assert "Missing required ABC score headers" in err_b


# =============================================================================
# 3. Discrete Dual-T4 Memory Guard Tests
# =============================================================================

def test_memory_guard_dual_t4_bounds():
    """Tests discrete 14.2 GB VRAM memory limits and mutual exclusion contracts."""
    guard = DualT4MemoryGuard()

    # Feasible single-card workload (12.0 GB <= 14.2 GB)
    ok_12, _ = guard.check_bounds(model_id="qwen-text", min_vram_gb=12.0, target_gpu=0)
    assert ok_12 is True

    # Infeasible single-card workload (15.5 GB > 14.2 GB)
    ok_15, reason_15 = guard.check_bounds(model_id="large-model", min_vram_gb=15.5, target_gpu=0)
    assert ok_15 is False
    assert "exceeds the discrete 14.2 GB" in reason_15
    assert "does not provide a 32 GB shared virtual memory pool" in reason_15

    # Dual-GPU split allows up to 28.4 GB across both cards
    ok_split, _ = guard.check_bounds(
        model_id="qwen-speech",
        min_vram_gb=18.0,
        target_gpu=0,
        allocation_policy="dual_gpu_split",
    )
    assert ok_split is True


# =============================================================================
# 4. Benchmark Runner Execution & Status Semantics Tests
# =============================================================================

def test_runner_negative_case_rejection():
    """Verifies runner marks PASS when an invalid task is cleanly rejected with HTTP 400."""
    runner = UnifiedBenchmarkRunner()
    try:
        case = {
            "id": "BM-TEST-NEG-01",
            "model": "qwen-text",
            "task": "text_to_video",
            "variant": "standard",
            "target_gpu": 0,
            "min_vram_gb": 12.0,
            "is_negative": True,
            "inputs": {"prompt": "Generate a video."},
            "modes": ["smoke"],
        }
        record = runner.evaluate_case(case, mode="smoke")
        assert record.status == "PASS"
        assert record.is_negative is True
        assert record.e2e_latency_ms is not None
    finally:
        runner.close()


def test_runner_backend_unavailable_status():
    """Verifies runner marks BACKEND_UNAVAILABLE for uninstalled/unbacked models."""
    runner = UnifiedBenchmarkRunner()
    try:
        case = {
            "id": "BM-TEST-ENH-01",
            "model": "enhancement-suite-v1",
            "task": "image_super_resolution",
            "variant": "standard",
            "target_gpu": 1,
            "min_vram_gb": 6.0,
            "is_negative": False,
            "inputs": {"image": "data:image/png;base64,..."},
            "modes": ["smoke"],
        }
        record = runner.evaluate_case(case, mode="smoke")
        assert record.status == "BACKEND_UNAVAILABLE"
        assert record.error_code == "BACKEND_UNAVAILABLE"
        assert "not ready/installed" in record.error_message
    finally:
        runner.close()


def test_runner_hardware_blocked_status():
    """Verifies runner marks HARDWARE_BLOCKED when memory limits are exceeded."""
    runner = UnifiedBenchmarkRunner()
    try:
        case = {
            "id": "BM-TEST-HW-01",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "variant": "standard",
            "target_gpu": 0,
            "min_vram_gb": 16.5,  # Exceeds 14.2 GB allocatable
            "is_negative": False,
            "inputs": {"prompt": "Huge prompt"},
            "modes": ["smoke"],
        }
        record = runner.evaluate_case(case, mode="smoke")
        assert record.status == "HARDWARE_BLOCKED"
        assert record.error_code == "INSUFFICIENT_VRAM"
        assert "exceeds the discrete 14.2 GB" in record.error_message
    finally:
        runner.close()


# =============================================================================
# 5. Report Generator Deliverables Tests
# =============================================================================

def test_report_generator_produces_valid_deliverables():
    """Verifies results.json, metrics.csv, and summary.md conform to Plan §6.6 schema."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir)

        records = [
            CaseExecutionRecord(
                case_id="BM-TXT-01",
                run_id="test_run",
                git_commit="82d5c23",
                model_id="qwen-text",
                task="chat_completion",
                variant="standard",
                mode="smoke",
                status="PASS",
                target_gpu=0,
                min_vram_gb=12.0,
                e2e_latency_ms=150.2,
                inference_time_ms=125.0,
                peak_vram_mb_gpu0=7500.0,
                output_bytes=256,
                artifact_mime="application/json",
                artifact_sha256="abc1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd",
                artifact_validated=True,
            ),
            CaseExecutionRecord(
                case_id="BM-ENH-01",
                run_id="test_run",
                git_commit="82d5c23",
                model_id="enhancement-suite-v1",
                task="image_super_resolution",
                variant="standard",
                mode="smoke",
                status="BACKEND_UNAVAILABLE",
                target_gpu=1,
                min_vram_gb=6.0,
                error_code="BACKEND_UNAVAILABLE",
                error_message="Backend offline",
            ),
        ]

        hw_info = {
            "cuda_available": True,
            "gpu_count": 2,
            "gpu_names": ["Tesla T4", "Tesla T4"],
            "max_allocatable_vram_mb": 14540.0,
        }

        gen = BenchmarkReportGenerator(
            run_id="test_run",
            git_commit="82d5c23",
            mode="smoke",
            hardware_info=hw_info,
            records=records,
            output_dir=out_path,
        )
        deliverables = gen.generate_all()

        # Check results.json
        res_file = deliverables["results_json"]
        assert res_file.exists()
        with open(res_file, "r", encoding="utf-8") as f:
            res_data = json.load(f)
        assert res_data["run_id"] == "test_run"
        assert len(res_data["cases"]) == 2
        assert res_data["summary"]["status_counts"]["PASS"] == 1
        assert res_data["summary"]["status_counts"]["BACKEND_UNAVAILABLE"] == 1

        # Check metrics.csv
        csv_file = deliverables["metrics_csv"]
        assert csv_file.exists()
        lines = csv_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3  # Header + 2 rows
        assert "BM-TXT-01" in lines[1]
        assert "BM-ENH-01" in lines[2]

        # Check summary.md
        sum_file = deliverables["summary_md"]
        assert sum_file.exists()
        md_text = sum_file.read_text(encoding="utf-8")
        assert "Factory v4 Lite — Unified Benchmark Summary Report" in md_text
        assert "Dual-T4 Discrete Memory Limit Compliance" in md_text
        assert "14.2 GB / 14,540 MB" in md_text


# =============================================================================
# 6. Unified Kaggle Bootstrap Tests
# =============================================================================

def test_kaggle_bootstrap_diagnostics():
    """Tests that KaggleBootstrap accurately detects environment and creates scratch storage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        scratch_dir = Path(tmp_dir) / "artifacts"
        bootstrapper = KaggleBootstrap(
            kaggle_input_dir=Path(tmp_dir) / "kaggle_input",
            scratch_artifacts_dir=scratch_dir,
        )

        hw = bootstrapper.detect_hardware()
        assert isinstance(hw, HardwareReport)
        assert hw.cpu_count >= 1

        deps = bootstrapper.validate_dependencies()
        assert "installed" in deps

        storage = bootstrapper.initialize_environment(api_key="sk-test-bootstrap-key")
        assert scratch_dir.exists()
        assert storage["storage_backend"] == "ephemeral_disk"
        assert storage["api_key_configured"] is True

        res = bootstrapper.run_bootstrap(check_only=True)
        assert isinstance(res.to_dict(), dict)
        assert res.timestamp is not None
