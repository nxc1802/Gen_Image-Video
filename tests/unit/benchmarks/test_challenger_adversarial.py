"""
tests.unit.benchmarks.test_challenger_adversarial
Empirical adversarial stress test suite for Factory v4 Lite benchmarks and bootstrap.

Stress-tests:
1. Adversarial Image Validation (truncated JPEG, corrupted WebP VP8 bitstream)
2. Adversarial Audio Validation (digital silence WAV, 0-frame WAV)
3. Adversarial Video Validation (MP4 missing mdat box, corrupt/truncated mdat container)
4. Adversarial Text & JSON Validation (whitespace text, missing required fields, zero-norm embedding)
5. Adversarial Checksum & Size Binding (mismatched SHA-256 digest, mismatched byte size)
6. Metric Separation Stress Test (9 negative passes + 82 backend unavailable -> 0% inference pass rate)
7. Bootstrap Dry-Run & Namespacing Stress Test (--check-only dry-run purity, model_id path namespacing)
"""

import io
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import wave

from PIL import Image
import pytest

from benchmarks.report import BenchmarkReportGenerator, CaseExecutionRecord
from benchmarks.validators import validate_artifact
from deployments.kaggle.bootstrap import DATASET_SPECS, KaggleBootstrap


class TestAdversarialImageValidation:
    """Stress tests for image container corruption and truncation."""

    def test_truncated_jpeg_rejection(self):
        """Valid 200x200 JPEG truncated to 70% must fail with IMAGE_TRUNCATED."""
        img = Image.new("RGB", (200, 200), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        # Truncate to 70%
        truncated_bytes = jpeg_bytes[: int(len(jpeg_bytes) * 0.7)]
        res = validate_artifact(truncated_bytes, expected_mime="image/jpeg", expected_format="jpeg")

        assert res.valid is False
        assert res.error_code == "IMAGE_TRUNCATED"
        assert res.dimensions is None

    def test_corrupted_webp_vp8_bitstream_rejection(self):
        """WebP with intact RIFF/WEBP header but corrupted VP8 bitstream must fail with IMAGE_TRUNCATED."""
        img = Image.new("RGB", (200, 200), color=(0, 255, 0))
        buf = io.BytesIO()
        img.save(buf, format="WEBP")
        webp_bytes = buf.getvalue()

        # Corrupt bytes in the VP8 frame bitstream (bytes 30..50)
        corrupted = bytearray(webp_bytes)
        corrupted[30:50] = b"\x00" * 20
        res = validate_artifact(bytes(corrupted), expected_mime="image/webp", expected_format="webp")

        assert res.valid is False
        assert res.error_code == "IMAGE_TRUNCATED"
        assert res.dimensions is None


class TestAdversarialAudioValidation:
    """Stress tests for audio PCM data integrity and silence."""

    def test_silent_wav_rejection(self):
        """1.0s WAV composed of digital zeros must fail with AUDIO_SILENT when reject_silence=True."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 32000)  # 16000 samples * 2 bytes = 1.0s
        wav_bytes = buf.getvalue()

        res = validate_artifact(
            wav_bytes,
            expected_mime="audio/wav",
            expected_format="wav",
            rules={"reject_silence": True},
        )

        assert res.valid is False
        assert res.error_code == "AUDIO_SILENT"
        assert res.metadata.get("is_silent") is True

    def test_zero_frame_wav_rejection(self):
        """0-frame WAV must fail with AUDIO_ZERO_SAMPLES."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"")
        wav_bytes = buf.getvalue()

        res = validate_artifact(
            wav_bytes,
            expected_mime="audio/wav",
            expected_format="wav",
        )

        assert res.valid is False
        assert res.error_code == "AUDIO_ZERO_SAMPLES"


class TestAdversarialVideoValidation:
    """Stress tests for video container box parsing and frame containment."""

    def test_mp4_missing_mdat_rejection(self):
        """MP4 with ftyp and moov but no mdat must fail with VIDEO_ZERO_FRAMES when require_media_payload=True."""
        ftyp = struct.pack(">I4s4sI", 16, b"ftyp", b"isom", 512)
        mvhd = struct.pack(">I4sBBBBIIII", 32, b"mvhd", 0, 0, 0, 0, 0, 0, 1000, 2000)
        moov = struct.pack(">I4s", 8 + len(mvhd), b"moov") + mvhd
        mp4_no_mdat = ftyp + moov

        res = validate_artifact(
            mp4_no_mdat,
            expected_mime="video/mp4",
            expected_format="mp4",
            rules={"require_media_payload": True},
        )

        assert res.valid is False
        assert res.error_code == "VIDEO_ZERO_FRAMES"
        assert res.metadata.get("has_mdat") is False

    def test_mp4_corrupt_mdat_size_rejection(self):
        """MP4 declaring 5,000,000 bytes for mdat with only 10 bytes present must fail with VIDEO_CONTAINER_CORRUPT."""
        ftyp = struct.pack(">I4s4sI", 16, b"ftyp", b"isom", 512)
        mvhd = struct.pack(">I4sBBBBIIII", 32, b"mvhd", 0, 0, 0, 0, 0, 0, 1000, 2000)
        moov = struct.pack(">I4s", 8 + len(mvhd), b"moov") + mvhd
        # mdat box header declaring 5000000 bytes, but followed by only 2 bytes payload
        mdat_corrupt = struct.pack(">I4s", 5000000, b"mdat") + b"\x00\x00"
        mp4_corrupt = ftyp + moov + mdat_corrupt

        res = validate_artifact(
            mp4_corrupt,
            expected_mime="video/mp4",
            expected_format="mp4",
        )

        assert res.valid is False
        assert res.error_code == "VIDEO_CONTAINER_CORRUPT"


class TestAdversarialTextAndJsonValidation:
    """Stress tests for text/plain whitespace, schema enforcement, and embedding norms."""

    def test_whitespace_text_rejection(self):
        """Whitespace string must fail with TEXT_EMPTY."""
        ws_text = "   \n\t  \r\n   "
        res = validate_artifact(ws_text, expected_mime="text/plain")

        assert res.valid is False
        assert res.error_code == "TEXT_EMPTY"

    def test_empty_json_with_required_fields_rejection(self):
        """Empty JSON object {} with required_fields must fail with REQUIRED_FIELD_MISSING."""
        res = validate_artifact({}, rules={"required_fields": ["id", "output"]})

        assert res.valid is False
        assert res.error_code == "REQUIRED_FIELD_MISSING"

    def test_zero_norm_embedding_rejection(self):
        """Embedding with all zeros must fail with EMBEDDING_ZERO_NORM."""
        emb_payload = {"data": [{"embedding": [0.0] * 1024}]}
        res = validate_artifact(emb_payload, rules={"is_embedding": True, "expected_dim": 1024})

        assert res.valid is False
        assert res.error_code == "EMBEDDING_ZERO_NORM"


class TestAdversarialChecksumAndSizeBinding:
    """Stress tests for cryptographic hash and byte length verification."""

    def test_checksum_mismatch_rejection(self):
        """Artifact with mismatched expected_sha256 must fail with CHECKSUM_MISMATCH."""
        content = b"valid payload content"
        res = validate_artifact(content, rules={"expected_sha256": "0" * 64})

        assert res.valid is False
        assert res.error_code == "CHECKSUM_MISMATCH"

    def test_size_mismatch_rejection(self):
        """Artifact with mismatched expected_size_bytes must fail with SIZE_MISMATCH."""
        content = b"valid payload content"
        res = validate_artifact(content, rules={"expected_size_bytes": 123456})

        assert res.valid is False
        assert res.error_code == "SIZE_MISMATCH"


class TestMetricSeparationStress:
    """Stress tests for strict segregation of negative API tests vs model inference metrics."""

    def test_9_negative_passes_and_82_unavailable(self, tmp_path):
        """
        9 negative cases (status=PASS, is_negative=True) + 82 positive cases (status=BACKEND_UNAVAILABLE).
        Must produce:
        - inference_metrics["pass"] == 0
        - inference_metrics["pass_rate_percent"] == 0.0%
        - contract_metrics["pass"] == 9
        - model_stats latencies empty (no pollution from negative cases)
        """
        records = []
        for i in range(9):
            records.append(
                CaseExecutionRecord(
                    case_id=f"neg_case_{i}",
                    run_id="stress_run",
                    git_commit="test_commit",
                    model_id="qwen-text",
                    task="text_generation",
                    variant="default",
                    mode="smoke",
                    status="PASS",
                    is_negative=True,
                    contract_pass=True,
                    inference_pass=False,
                    e2e_latency_ms=123.4,
                )
            )

        for i in range(82):
            records.append(
                CaseExecutionRecord(
                    case_id=f"pos_case_{i}",
                    run_id="stress_run",
                    git_commit="test_commit",
                    model_id="qwen-text",
                    task="text_generation",
                    variant="default",
                    mode="smoke",
                    status="BACKEND_UNAVAILABLE",
                    is_negative=False,
                    contract_pass=None,
                    inference_pass=False,
                    e2e_latency_ms=None,
                )
            )

        rep = BenchmarkReportGenerator(
            run_id="stress_run",
            git_commit="test_commit",
            mode="smoke",
            hardware_info={},
            records=records,
            output_dir=tmp_path / "reports",
        )

        stats = rep.compute_summary_stats()

        assert stats["inference_metrics"]["pass"] == 0
        assert stats["inference_metrics"]["pass_rate_percent"] == 0.0
        assert stats["contract_metrics"]["pass"] == 9
        assert stats["contract_metrics"]["pass_rate_percent"] == 100.0
        assert stats["model_stats"]["qwen-text"]["latencies_ms"] == []


class TestBootstrapDryRunAndNamespacing:
    """Stress tests for Kaggle bootstrap --check-only purity and dataset path namespacing."""

    def test_check_only_dry_run_purity(self):
        """--check-only must not write bootstrap_summary.json or mutate environment."""
        summary_path = Path("bootstrap_summary.json")
        if summary_path.exists():
            summary_path.unlink()

        cmd = [sys.executable, "-m", "deployments.kaggle.bootstrap", "--check-only"]
        res = subprocess.run(cmd, capture_output=True, text=True)

        assert res.returncode == 0
        assert not summary_path.exists()

        # In-process environment check
        kb = KaggleBootstrap()
        old_val = os.environ.get("FACTORY_API_KEY")
        kb.initialize_environment(dry_run=True)
        assert os.environ.get("FACTORY_API_KEY") == old_val

    def test_dataset_specs_namespacing_collision_prevention(self, tmp_path):
        """Overlapping weight filenames across DATASET_SPECS must resolve to separate models_dir / model_id / fname."""
        fake_input = tmp_path / "input"
        fake_models = tmp_path / "models"
        fake_input.mkdir()
        fake_models.mkdir()

        # Find specs with model.safetensors
        specs = [s for s in DATASET_SPECS if "model.safetensors" in s.get("files", [])]
        assert len(specs) >= 2, "Need at least 2 specs with model.safetensors to test collision"

        s1 = specs[0]
        s2 = specs[1]

        d1 = fake_input / s1["slug"]
        d1.mkdir()
        (d1 / "model.safetensors").write_bytes(b"DATASET_1_UNIQUE_PAYLOAD_ABC")

        d2 = fake_input / s2["slug"]
        d2.mkdir()
        (d2 / "model.safetensors").write_bytes(b"DATASET_2_UNIQUE_PAYLOAD_XYZ")

        kb = KaggleBootstrap(kaggle_input_dir=fake_input, models_dir=fake_models)
        kb.discover_and_mount_datasets(create_symlinks=True)

        dest1 = fake_models / s1["id"] / "model.safetensors"
        dest2 = fake_models / s2["id"] / "model.safetensors"

        assert dest1.exists()
        assert dest2.exists()
        assert dest1.resolve() != dest2.resolve()
        assert dest1.read_bytes() == b"DATASET_1_UNIQUE_PAYLOAD_ABC"
        assert dest2.read_bytes() == b"DATASET_2_UNIQUE_PAYLOAD_XYZ"
