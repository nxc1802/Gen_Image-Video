"""
tests/adversarial/test_adversarial_p3_acestep.py
Adversarial Challenger Test Suite for Phase 3 (ACE-Step 1.5 Subsystem):
1. Variant Separation & Security Exploits (§7.1, ACE-11, ACE-12, ACE-13, ACE-17).
2. Musical Parameter Boundary & Poisoning Stress (ACE-06, ACE-08, ACE-09, ACE-15).
3. Asset Staging, Multi-Artifact Delivery, Path Traversal, and Fail-Closed Resilience (ACE-07, ACE-16, ACE-20, ACE-24).
4. Manifest Structural & Referential Integrity Verification.
5. Inoculation Probes for Empirical Security & Reliability Vulnerabilities.
"""
import asyncio
import base64
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
import httpx
import pytest
import yaml

from factory_contracts.errors import (
    ErrorCode,
    InvalidRequestError,
    ModelUnavailableError,
    UnsupportedCapabilityError,
    classify_worker_error,
)
from factory_contracts.jobs.job_contract import FactoryJob
from model_registry.loader import ManifestLoader
from model_registry.validator import ManifestValidator
from services.music.acestep.adapter import (
    ALL_SUPPORTED_TASKS,
    ALL_VARIANTS,
    BASE_VARIANTS,
    KEY_REGEX,
    SPECIALIZED_BASE_ONLY_TASKS,
    TURBO_VARIANTS,
    ACEStepAdapter,
)
from services.music.acestep.client import ACEStepClient
from services.music.worker import MusicWorker


# Helper to build valid mock audio bytes
def make_wav_bytes(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    num_samples = int(duration_sec * sample_rate)
    data_size = num_samples * 2
    header = bytearray(b"RIFF")
    header.extend((data_size + 36).to_bytes(4, "little"))
    header.extend(b"WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00")
    header.extend(sample_rate.to_bytes(4, "little"))
    header.extend((sample_rate * 2).to_bytes(4, "little"))
    header.extend(b"\x02\x00\x10\x00data")
    header.extend(data_size.to_bytes(4, "little"))
    header.extend(b"\x00\x00" * num_samples)
    return bytes(header)


# ==============================================================================
# 1. Variant Separation & Security Exploits (§7.1, ACE-11, ACE-12, ACE-13, ACE-17)
# ==============================================================================

class TestVariantSeparationAdversarial:
    """Stress tests variant isolation between Base and Turbo/SFT checkpoints."""

    @pytest.mark.parametrize("task", ["extract", "lego", "complete", "music_extend"])
    @pytest.mark.parametrize("variant", [
        "turbo",
        "TURBO",
        "Turbo",
        "  turbo  ",
        "sft",
        "SFT",
        "Sft",
        "  sft  ",
        "2b_turbo",
        "2B_TURBO",
    ])
    def test_specialized_tasks_strictly_rejected_on_turbo_and_sft(self, task, variant):
        """Specialized editing/separation tasks must raise UnsupportedCapabilityError on turbo/sft."""
        with pytest.raises(UnsupportedCapabilityError) as exc_info:
            ACEStepAdapter.validate_variant_capability(task, variant)

        assert exc_info.value.code == ErrorCode.UNSUPPORTED_CAPABILITY.value
        assert "not supported on variant" in str(exc_info.value)
        assert task in str(exc_info.value)

        # Also test map_inputs rejection
        inputs = {
            "variant": variant,
            "prompt": "Arrange this track",
            "source_audio_asset_id": "asset-source-123",
        }
        with pytest.raises(UnsupportedCapabilityError) as exc_map:
            ACEStepAdapter.map_inputs(task, inputs)
        assert exc_map.value.code == ErrorCode.UNSUPPORTED_CAPABILITY.value

    @pytest.mark.parametrize("task", [
        "text_to_music",
        "lyrics_to_music",
        "reference_audio",
        "cover",
        "repaint",
        "extract",
        "lego",
        "complete",
        "music_extend",
    ])
    @pytest.mark.parametrize("variant", ["base", "2b_base", "xl_base", "standard", "default", None])
    def test_all_tasks_supported_on_base_variants(self, task, variant):
        """All 9 generative and specialized tasks must be accepted on base variants."""
        ACEStepAdapter.validate_variant_capability(task, variant)
        assert ACEStepAdapter.is_task_supported_for_variant(task, variant) is True

    @pytest.mark.parametrize("task", ["text_to_music", "lyrics_to_music", "reference_audio", "cover", "repaint"])
    @pytest.mark.parametrize("variant", ["turbo", "sft", "2b_turbo"])
    def test_generative_tasks_supported_on_turbo_variants(self, task, variant):
        """Standard generative tasks must be supported on turbo/sft variants."""
        ACEStepAdapter.validate_variant_capability(task, variant)
        assert ACEStepAdapter.is_task_supported_for_variant(task, variant) is True

    @pytest.mark.parametrize("bad_variant", [
        "ultra",
        "v3",
        "gpt4",
        "magic_turbo",
        "base_turbo",
        "123",
        "!@#$",
    ])
    def test_unknown_variant_rejected(self, bad_variant):
        """Unknown variants must be rejected with InvalidRequestError."""
        with pytest.raises(InvalidRequestError) as exc_info:
            ACEStepAdapter.validate_variant_capability("text_to_music", bad_variant)
        assert "Unknown variant" in str(exc_info.value)

    @pytest.mark.parametrize("bad_task", [
        "voice_clone",
        "speech_synthesis",
        "delete_model",
        "video_generate",
        "",
        "unknown_music_task",
    ])
    def test_unsupported_task_rejected(self, bad_task):
        """Tasks outside ACE-Step domain must be rejected with InvalidRequestError."""
        with pytest.raises(InvalidRequestError) as exc_info:
            ACEStepAdapter.validate_variant_capability(bad_task, "base")
        assert "does not support task" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_worker_early_rejection_of_turbo_specialized_tasks(self):
        """Worker must fail-fast with UNSUPPORTED_CAPABILITY before staging assets."""
        worker = MusicWorker(
            worker_id="test-worker",
            control_plane_url="http://127.0.0.1:8000",
            api_key="test-key",
            acestep_api_url="http://127.0.0.1:8008",
        )

        job_uuid = str(uuid.uuid4())
        req_uuid = str(uuid.uuid4())
        job = FactoryJob(
            job_id=f"job-{job_uuid}",
            request_id=f"req-{req_uuid}",
            service="music",
            model="ace-step-1.5",
            task="extract",
            inputs={"variant": "turbo", "source_audio_asset_id": "asset-test"},
        )

        mock_http = AsyncMock()
        mock_fail_resp = MagicMock(status_code=200)
        mock_http.post.return_value = mock_fail_resp

        # Stage asset should NEVER be called because early variant validation fails
        with patch.object(worker, "stage_asset", new_callable=AsyncMock) as mock_stage:
            res = await worker.execute_job(mock_http, job)
            assert res is False
            mock_stage.assert_not_called()

        # Verify fail request was dispatched to control plane with UNSUPPORTED_CAPABILITY
        fail_calls = [
            c for c in mock_http.post.call_args_list
            if f"/v1/jobs/job-{job_uuid}/fail" in str(c)
        ]
        assert len(fail_calls) == 1
        fail_payload = fail_calls[0].kwargs.get("json") or fail_calls[0][1].get("json")
        assert fail_payload["error"]["code"] == ErrorCode.UNSUPPORTED_CAPABILITY.value
        assert fail_payload["retryable"] is False


# ==============================================================================
# 2. Musical Parameter Boundary & Poisoning Stress (ACE-06, ACE-08, ACE-09, ACE-15)
# ==============================================================================

class TestMusicalParametersBoundaryStress:
    """Stress tests boundary limits, type errors, and malicious inputs for musical parameters."""

    # --- BPM / Tempo ---
    @pytest.mark.parametrize("invalid_bpm", [-100, -1, 0, 10, 29, 301, 999, 99999])
    def test_bpm_range_bounds_rejection(self, invalid_bpm):
        """BPM outside [30, 300] must be rejected."""
        inputs = {"prompt": "Techno track", "bpm": invalid_bpm}
        with pytest.raises(InvalidRequestError) as exc_info:
            ACEStepAdapter.map_inputs("text_to_music", inputs)
        assert "BPM must be between 30 and 300" in str(exc_info.value)

    @pytest.mark.parametrize("non_numeric_bpm", ["fast", "120bpm", "very_slow", [], {}])
    def test_bpm_non_numeric_rejection(self, non_numeric_bpm):
        """Non-numeric BPM values must raise InvalidRequestError."""
        inputs = {"prompt": "Techno track", "bpm": non_numeric_bpm}
        with pytest.raises(InvalidRequestError) as exc_info:
            ACEStepAdapter.map_inputs("text_to_music", inputs)
        assert "BPM/tempo must be an integer" in str(exc_info.value)

    @pytest.mark.parametrize("valid_bpm", [30, 60, 120, 128, 174, 300])
    def test_bpm_valid_boundaries_accepted(self, valid_bpm):
        """Valid BPM at boundaries 30 and 300 and standard tempos must succeed."""
        params = ACEStepAdapter.map_inputs("text_to_music", {"prompt": "Test", "bpm": valid_bpm})
        assert params["bpm"] == valid_bpm

    # --- Musical Key ---
    @pytest.mark.parametrize("valid_key", [
        "C", "C major", "C min", "Am", "A minor", "F#m", "F# major",
        "Bb", "Bb major", "Eb min", "Db maj", "G#m", "Gb", "D# minor",
    ])
    def test_musical_key_valid_formats_accepted(self, valid_key):
        """Valid musical keys conforming to pitch class + mode must be accepted."""
        params = ACEStepAdapter.map_inputs("text_to_music", {"prompt": "Ballad", "key": valid_key})
        assert params["key"] == valid_key

    @pytest.mark.parametrize("invalid_key", [
        "H major",
        "Z minor",
        "flat sharp",
        "C# minor extra words",
        "123",
        "!@#",
        "Do trưởng",
        "C/G",
        "major C",
        "A###",
    ])
    def test_musical_key_invalid_formats_rejected(self, invalid_key):
        """Malformed or invalid musical keys must raise InvalidRequestError."""
        inputs = {"prompt": "Ballad", "key": invalid_key}
        with pytest.raises(InvalidRequestError) as exc_info:
            ACEStepAdapter.map_inputs("text_to_music", inputs)
        assert "Invalid musical key" in str(exc_info.value)

    # --- Time Signature ---
    @pytest.mark.parametrize("valid_meter", ["4/4", "3/4", "2/4", "6/8", "12/8", "5/4", "7/8", "1/1", "16/16"])
    def test_time_signature_valid_meters_accepted(self, valid_meter):
        """Valid musical time signatures must be accepted."""
        params = ACEStepAdapter.map_inputs("text_to_music", {"prompt": "Waltz", "time_signature": valid_meter})
        assert params["time_signature"] == valid_meter

    @pytest.mark.parametrize("invalid_meter", [
        "4",
        "three-four",
        "4/0",
        "0/4",
        "-4/4",
        "4/-4",
        "4/3",
        "4/5",
        "4/7",
        "17/4",
        "0/0",
        "4/4/4",
        "4/32",
        "4/12",
        "3/8/16",
    ])
    def test_time_signature_invalid_meters_rejected(self, invalid_meter):
        """Invalid time signatures violating fractional meter rules must raise InvalidRequestError."""
        inputs = {"prompt": "Waltz", "time_signature": invalid_meter}
        with pytest.raises(InvalidRequestError) as exc_info:
            ACEStepAdapter.map_inputs("text_to_music", inputs)
        assert "Invalid time signature" in str(exc_info.value)

    # --- Duration ---
    @pytest.mark.parametrize("invalid_duration", [0.0, -0.1, -60.0, 300.1, 9999.0, float("inf")])
    def test_duration_bounds_rejection(self, invalid_duration):
        """Duration <= 0.0 or > 300.0 or Inf must raise InvalidRequestError."""
        inputs = {"prompt": "Song", "duration_seconds": invalid_duration}
        with pytest.raises(InvalidRequestError) as exc_info:
            ACEStepAdapter.map_inputs("text_to_music", inputs)
        assert "duration_seconds must be between 1.0 and 300.0" in str(exc_info.value)

    # --- Cover Strength ---
    @pytest.mark.parametrize("invalid_strength", [-0.01, -1.0, 1.01, 2.0, 999.0])
    def test_cover_strength_bounds_rejection(self, invalid_strength):
        """Cover strength outside [0.0, 1.0] must raise InvalidRequestError."""
        inputs = {
            "prompt": "Jazz cover",
            "source_audio_asset_id": "src-123",
            "cover_strength": invalid_strength,
        }
        with pytest.raises(InvalidRequestError) as exc_info:
            ACEStepAdapter.map_inputs("cover", inputs)
        assert "cover_strength must be between 0.0 and 1.0" in str(exc_info.value)

    # --- Repaint Bounds ---
    @pytest.mark.parametrize("start,end,duration,expected_err", [
        (10.0, 10.0, 60.0, "Must satisfy 0.0 <= start < end <= duration"),
        (20.0, 10.0, 60.0, "Must satisfy 0.0 <= start < end <= duration"),
        (-1.0, 10.0, 60.0, "Must satisfy 0.0 <= start < end <= duration"),
        (10.0, 65.0, 60.0, "Must satisfy 0.0 <= start < end <= duration"),
        (None, 20.0, 60.0, "Both 'repainting_start' and 'repainting_end' must be specified"),
        (10.0, None, 60.0, "Both 'repainting_start' and 'repainting_end' must be specified"),
    ])
    def test_repaint_interval_violations_rejected(self, start, end, duration, expected_err):
        """Invalid repaint intervals must be rejected."""
        inputs = {
            "prompt": "Fix guitar solo",
            "source_audio_asset_id": "src-123",
            "duration_seconds": duration,
        }
        if start is not None:
            inputs["repainting_start"] = start
        if end is not None:
            inputs["repainting_end"] = end

        with pytest.raises(InvalidRequestError) as exc_info:
            ACEStepAdapter.map_inputs("repaint", inputs)
        assert expected_err in str(exc_info.value)

    # --- Multilingual Unicode & Poisoning Stress (ACE-15) ---
    def test_multilingual_vietnamese_japanese_emoji_lyrics_preserved(self):
        """Multilingual lyrics with Vietnamese diacritics, Japanese Kanji/Kana, and emojis must be preserved."""
        vn_lyrics = "Đất nước trọn niềm vui! Lời ca rộn rã non sông, Việt Nam ơi mùa xuân đến rồi."
        jp_lyrics = "千本桜 夜ニ紛レ 君ノ声モ届カナイヨ 青藍の空 遥か彼方"
        emoji_lyrics = "🎶 Happy Birthday to You! 🎂 Sing loud! 🎸🥁🎵"

        full_lyrics = f"[verse 1]\n{vn_lyrics}\n[verse 2]\n{jp_lyrics}\n[chorus]\n{emoji_lyrics}"
        inputs = {
            "prompt": "Global anthem",
            "lyrics": full_lyrics,
        }
        params = ACEStepAdapter.map_inputs("lyrics_to_music", inputs)
        assert params["lyrics"] == full_lyrics

        # Release payload should preserve full unicode text
        payload = ACEStepClient._release_payload("lyrics_to_music", params)
        assert payload["lyrics"] == full_lyrics


# ==============================================================================
# 3. Asset Staging, Multi-Artifact Delivery, & Fail-Closed Resilience
# ==============================================================================

class TestAssetStagingAndFailClosedResilience:
    """Stress tests asset staging, cancellation, multi-artifact persistence, and error handling."""

    @pytest.mark.asyncio
    async def test_stage_asset_valid_data_uri(self):
        """Data URI base64 audio must be staged to a readable temporary file."""
        worker = MusicWorker(worker_id="test-worker")
        mock_http = AsyncMock()

        raw_wav = make_wav_bytes(duration_sec=0.5)
        b64_str = base64.b64encode(raw_wav).decode("ascii")
        data_uri = f"data:audio/wav;base64,{b64_str}"

        staged_path = await worker.stage_asset(mock_http, data_uri)
        try:
            p = Path(staged_path)
            assert p.exists()
            assert p.stat().st_size == len(raw_wav)
            assert p.read_bytes() == raw_wav
        finally:
            if Path(staged_path).exists():
                Path(staged_path).unlink()

    @pytest.mark.asyncio
    async def test_stage_asset_malformed_data_uri_rejected(self):
        """Malformed base64 data URI must raise an exception cleanly."""
        worker = MusicWorker(worker_id="test-worker")
        mock_http = AsyncMock()

        with pytest.raises(Exception):
            await worker.stage_asset(mock_http, "data:audio/wav;base64,***INVALID_B64***")

    @pytest.mark.asyncio
    async def test_client_fail_closed_when_backend_unconfigured(self):
        """Client must fail closed with ModelUnavailableError when api_base_url is empty."""
        client = ACEStepClient(api_base_url="")
        with pytest.raises(ModelUnavailableError) as exc_gen:
            await client.generate_music("text_to_music", {"prompt": "Rock", "duration_seconds": 30})
        assert "ACE-Step backend is unconfigured" in str(exc_gen.value)

        with pytest.raises(ModelUnavailableError) as exc_und:
            await client.understand_audio("/tmp/audio.wav")
        assert "ACE-Step backend is unconfigured" in str(exc_und.value)

        with pytest.raises(ModelUnavailableError) as exc_scr:
            await client.score_audio("/tmp/audio.wav")
        assert "ACE-Step backend is unconfigured" in str(exc_scr.value)

        with pytest.raises(ModelUnavailableError) as exc_trn:
            await client.train_lora(dataset_path="/tmp/ds", output_dir="/tmp/out")
        assert "ACE-Step backend is unconfigured" in str(exc_trn.value)

        # cancel_task returns False gracefully without crash
        assert await client.cancel_task("task-123") is False

    @pytest.mark.asyncio
    async def test_client_backend_health_failure_raises_model_unavailable(self):
        """Unhealthy backend (/health >= 400) raises ModelUnavailableError."""
        client = ACEStepClient(api_base_url="http://127.0.0.1:8008")

        mock_health_resp = MagicMock(status_code=503)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_health_resp
            with pytest.raises(ModelUnavailableError) as exc_info:
                await client.generate_music("text_to_music", {"prompt": "Pop"})
            assert "ACE-Step backend is unhealthy" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_client_task_status_failed_raises_model_unavailable(self):
        """Backend returning status 2 (generation failed) raises ModelUnavailableError."""
        client = ACEStepClient(api_base_url="http://127.0.0.1:8008")

        mock_health = MagicMock(status_code=200)
        mock_release = MagicMock(status_code=200)
        mock_release.json.return_value = {"code": 200, "data": {"task_id": "task-fail-001"}}

        mock_poll = MagicMock(status_code=200)
        mock_poll.json.return_value = {
            "code": 200,
            "data": [{"task_id": "task-fail-001", "status": 2, "result": "[]"}],
        }

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get, \
             patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_get.return_value = mock_health
            mock_post.side_effect = [mock_release, mock_poll]

            with pytest.raises(ModelUnavailableError) as exc_info:
                await client.generate_music("text_to_music", {"prompt": "EDM"})
            assert "ACE-Step generation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_worker_multi_artifact_batch_persistence_and_cleanup(self, tmp_path):
        """Worker must persist all generated artifacts with distinct SHA256 and clean up temp files."""
        worker = MusicWorker(
            worker_id="test-worker",
            control_plane_url="http://127.0.0.1:8000",
            api_key="test-key",
            acestep_api_url="http://127.0.0.1:8008",
        )

        job_uuid = str(uuid.uuid4())
        req_uuid = str(uuid.uuid4())
        job = FactoryJob(
            job_id=f"job-{job_uuid}",
            request_id=f"req-{req_uuid}",
            service="music",
            model="ace-step-1.5",
            task="text_to_music",
            inputs={"prompt": "Batch electronic beats", "batch_size": 2},
        )

        wav1 = make_wav_bytes(duration_sec=1.0)
        wav2 = make_wav_bytes(duration_sec=2.0)

        mock_client_result = {
            "task_id": "task-batch-99",
            "filename": "output_0.mp3",
            "mime_type": "audio/mpeg",
            "audio_bytes": wav1,
            "size_bytes": len(wav1),
            "sha256": hashlib.sha256(wav1).hexdigest(),
            "metadata": {},
            "artifacts": [
                {
                    "index": 0,
                    "filename": "output_0.mp3",
                    "mime_type": "audio/mpeg",
                    "audio_bytes": wav1,
                    "size_bytes": len(wav1),
                    "sha256": hashlib.sha256(wav1).hexdigest(),
                    "metadata": {},
                },
                {
                    "index": 1,
                    "filename": "output_1.mp3",
                    "mime_type": "audio/mpeg",
                    "audio_bytes": wav2,
                    "size_bytes": len(wav2),
                    "sha256": hashlib.sha256(wav2).hexdigest(),
                    "metadata": {},
                },
            ],
        }

        mock_http = AsyncMock()

        # Storage upload responses
        def post_side_effect(url, **kwargs):
            if "/upload-url" in url:
                req_json = kwargs.get("json", {})
                art_name = req_json.get("artifact_name", "out.mp3")
                resp = MagicMock(status_code=200)
                resp.json.return_value = {
                    "upload_url": f"http://storage.local/upload/{art_name}",
                    "storage_path": f"music/job-multi/{art_name}",
                }
                return resp
            elif "/download-url" in url:
                req_json = kwargs.get("json", {})
                sp = req_json.get("storage_path", "music/art")
                resp = MagicMock(status_code=200)
                resp.json.return_value = {
                    "download_url": f"http://storage.local/download/{sp}",
                }
                return resp
            elif "/complete" in url:
                resp = MagicMock(status_code=200)
                resp.text = "OK"
                return resp
            resp = MagicMock(status_code=200)
            return resp

        mock_http.post.side_effect = post_side_effect

        mock_upload_client = AsyncMock()
        mock_upload_client.put.return_value = MagicMock(status_code=200)

        with patch.object(worker.client, "generate_music", new_callable=AsyncMock) as mock_gen, \
             patch("httpx.AsyncClient.__aenter__", return_value=mock_upload_client):
            mock_gen.return_value = mock_client_result

            res = await worker.execute_job(mock_http, job)
            assert res is True

        # Verify complete call received 2 distinct artifacts
        complete_calls = [
            c for c in mock_http.post.call_args_list
            if f"/v1/jobs/job-{job_uuid}/complete" in str(c)
        ]
        assert len(complete_calls) == 1
        comp_payload = complete_calls[0].kwargs.get("json") or complete_calls[0][1].get("json")
        artifacts = comp_payload["artifacts"]
        assert len(artifacts) == 2
        assert artifacts[0]["sha256"] == hashlib.sha256(wav1).hexdigest()
        assert artifacts[1]["sha256"] == hashlib.sha256(wav2).hexdigest()
        assert artifacts[0]["storage_path"] != artifacts[1]["storage_path"]


# ==============================================================================
# 4. Manifest Structural & Referential Integrity Verification
# ==============================================================================

class TestManifestIntegrityVerification:
    """Verifies ace-step-1.5.yaml structural conformance, schema validity, and truth invariants."""

    def test_ace_step_manifest_validation(self):
        """Model manifest must pass full referential and structural schema validation."""
        loader = ManifestLoader(Path("manifests"))
        validator = ManifestValidator(loader)
        valid, errors = validator.validate_all()
        assert valid is True, f"Manifest validation failed with errors: {errors}"
        assert len(errors) == 0

    def test_ace_step_manifest_capabilities_truth_model(self):
        """Capabilities truth model must declare 3 levels: declared, implemented, live_verified."""
        manifest_path = Path("manifests/models/ace-step-1.5.yaml")
        assert manifest_path.exists()
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_dict = yaml.safe_load(f)

        truth = manifest_dict.get("capabilities_truth", {})
        assert len(truth) >= 11

        # Generative & specialized tasks implemented
        implemented_tasks = [
            "text_to_music",
            "lyrics_to_music",
            "reference_audio",
            "cover",
            "repaint",
            "extract",
            "lego",
            "complete",
            "music_extend",
        ]
        for t in implemented_tasks:
            assert t in truth, f"Missing {t} in capabilities_truth"
            entry = truth[t]
            assert entry["declared"] is True
            assert entry["implemented"] is True
            # Anti-escalation: live_verified must be False until physical GPU Phase 5 verification
            assert entry["live_verified"] is False, f"Anti-escalation failed: {t} is prematurely live_verified!"

        # Legacy tasks declared but unbacked
        for unbacked in ["audio_to_audio", "audio_to_music"]:
            assert unbacked in truth
            assert truth[unbacked]["declared"] is True
            assert truth[unbacked]["implemented"] is False

    def test_ace_step_manifest_task_variant_matrix(self):
        """Task variant matrix must separate Base and Turbo capabilities strictly."""
        manifest_path = Path("manifests/models/ace-step-1.5.yaml")
        assert manifest_path.exists()
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_dict = yaml.safe_load(f)

        tvm = manifest_dict.get("task_variant_matrix", {})

        # Base only tasks
        for base_task in ["extract", "lego", "complete", "music_extend"]:
            assert base_task in tvm
            allowed_variants = tvm[base_task]
            assert "turbo" not in allowed_variants, f"Violation: {base_task} allowed on turbo in manifest!"
            assert "sft" not in allowed_variants, f"Violation: {base_task} allowed on sft in manifest!"
            assert "base" in allowed_variants
            assert "xl_base" in allowed_variants

        # Universal tasks
        for univ_task in ["text_to_music", "lyrics_to_music", "reference_audio", "cover", "repaint"]:
            assert univ_task in tvm
            allowed = tvm[univ_task]
            assert "base" in allowed
            assert "turbo" in allowed
            assert "sft" in allowed


# ==============================================================================
# 5. Inoculation Probes for Empirical Security & Reliability Vulnerabilities
# ==============================================================================

class TestEmpiricalVulnerabilitiesInoculationProbes:
    """
    Empirically documents verified vulnerabilities to guarantee reproducible evidence
    for adversarial reporting and remediation.
    """

    @pytest.mark.asyncio
    async def test_file_uri_deletion_hazard_probe(self):
        """
        [CRITICAL REMEDIATION: VULN-P3-01]
        Passing file:// URI causes stage_asset() to create a secure temporary copy.
        When execute_job finishes, the finally block unlinks only the temporary staged copy,
        preserving the caller's original file on disk.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(b"precious user recording")
            user_path = f.name

        staged = None
        try:
            worker = MusicWorker(worker_id="test-worker")
            # stage_asset returns a temporary copy instead of the original user path
            staged = await worker.stage_asset(AsyncMock(), f"file://{user_path}")
            assert staged != user_path, "stage_asset must return a temporary copy, not the caller's file path!"
            assert Path(staged).exists(), "Temporary staged copy must exist!"

            # Simulating worker execute_job cleanup loop
            staged_files = [staged]
            for st_file in staged_files:
                p = Path(st_file)
                if p.exists():
                    p.unlink()

            # Prove the original user file is PRESERVED!
            assert Path(user_path).exists(), "Original user file MUST be preserved after worker cleanup!"
            assert not Path(staged).exists(), "Temporary staged copy must be cleaned up!"
        finally:
            if staged and Path(staged).exists():
                Path(staged).unlink()
            if Path(user_path).exists():
                Path(user_path).unlink()

    @pytest.mark.asyncio
    async def test_ssrf_preflight_validation_probe(self):
        """
        [HIGH REMEDIATION: VULN-P3-02]
        Remote HTTP/HTTPS URLs are validated with SSRF pre-flight validation.
        Cloud metadata (169.254.169.254) and private loopback IPs are blocked.
        """
        worker = MusicWorker(worker_id="test-worker")
        mock_client = AsyncMock()

        # Cloud metadata IP
        with pytest.raises(InvalidRequestError, match=r"(?i)blocked by ssrf policy"):
            await worker.stage_asset(mock_client, "http://169.254.169.254/latest/meta-data")

        # Local loopback IP
        with pytest.raises(InvalidRequestError, match=r"(?i)blocked by ssrf policy"):
            await worker.stage_asset(mock_client, "http://127.0.0.1:8080/internal-secret")

    def test_nan_numerical_poisoning_bypass_probe(self):
        """
        [MEDIUM REMEDIATION: VULN-P3-03]
        IEEE 754 NaN values are strictly guarded against with math.isnan().
        NaN values must be rejected with InvalidRequestError instead of slipping into params.
        """
        # 1. duration_seconds with NaN
        with pytest.raises(InvalidRequestError, match=r"duration_seconds must be between 1\.0 and 300\.0 seconds"):
            ACEStepAdapter.map_inputs("text_to_music", {"prompt": "Poison", "duration_seconds": float("nan")})

        # 2. cover_strength with NaN
        with pytest.raises(InvalidRequestError, match=r"cover_strength must be between 0\.0 and 1\.0"):
            ACEStepAdapter.map_inputs(
                "cover",
                {"prompt": "Poison", "source_audio_asset_id": "src-1", "cover_strength": float("nan")},
            )

        # 3. repainting_start / repainting_end with NaN
        with pytest.raises(InvalidRequestError, match=r"Invalid repaint interval"):
            ACEStepAdapter.map_inputs(
                "repaint",
                {
                    "prompt": "Poison",
                    "source_audio_asset_id": "src-1",
                    "repainting_start": float("nan"),
                    "repainting_end": 10.0,
                },
            )

        with pytest.raises(InvalidRequestError, match=r"Invalid repaint interval"):
            ACEStepAdapter.map_inputs(
                "repaint",
                {
                    "prompt": "Poison",
                    "source_audio_asset_id": "src-1",
                    "repainting_start": 0.0,
                    "repainting_end": float("nan"),
                },
            )

        # 4. lora_weight with NaN
        with pytest.raises(InvalidRequestError, match=r"lora_weight must be between -2\.0 and 2\.0"):
            ACEStepAdapter.map_inputs(
                "text_to_music",
                {"prompt": "Poison", "lora_path": "model.safetensors", "lora_weight": float("nan")},
            )

    def test_musical_key_embedded_newline_gap_probe(self):
        """
        [LOW REMEDIATION: VULN-P3-05]
        KEY_REGEX strictly enforces horizontal whitespace [ \t]*, rejecting
        embedded newlines and carriage returns like 'C\\nmajor'.
        """
        key_with_newline = "C\nmajor"
        match = KEY_REGEX.match(key_with_newline)
        assert match is None, "KEY_REGEX must reject multiline keys containing embedded newlines!"

        with pytest.raises(InvalidRequestError, match=r"Invalid musical key"):
            ACEStepAdapter.map_inputs("text_to_music", {"prompt": "Test", "key": key_with_newline})

    @pytest.mark.asyncio
    async def test_understand_and_score_leak_raw_httpx_exceptions_probe(self):
        """
        [MEDIUM REMEDIATION: VULN-P3-04]
        understand_audio() and score_audio() wrap httpx.HTTPError in ModelUnavailableError
        with code='BACKEND_UNAVAILABLE'.
        Whitespace api_base_url is sanitized with strip() to raise unconfigured ModelUnavailableError.
        """
        # Case 1: Connection refused raises ModelUnavailableError (not raw ConnectError)
        client_refused = ACEStepClient(api_base_url="http://127.0.0.1:9999")
        with pytest.raises(ModelUnavailableError) as exc_understand:
            await client_refused.understand_audio("/tmp/nonexistent.wav")
        assert exc_understand.value.code == "BACKEND_UNAVAILABLE"
        assert "ACE-Step backend request failed" in str(exc_understand.value)

        with pytest.raises(ModelUnavailableError) as exc_score:
            await client_refused.score_audio("/tmp/nonexistent.wav")
        assert exc_score.value.code == "BACKEND_UNAVAILABLE"
        assert "ACE-Step backend request failed" in str(exc_score.value)

        # Case 2: Whitespace base_url is sanitized and cleanly reports unconfigured backend
        client_space = ACEStepClient(api_base_url="   ")
        with pytest.raises(ModelUnavailableError, match="ACE-Step backend is unconfigured."):
            await client_space.understand_audio("/tmp/nonexistent.wav")

        with pytest.raises(ModelUnavailableError, match="ACE-Step backend is unconfigured."):
            await client_space.score_audio("/tmp/nonexistent.wav")
