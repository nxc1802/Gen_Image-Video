"""
tests/unit/services/test_yue2_p4.py
Comprehensive Unit Test Suite for YuE2 Phase 4 Implementation (YUE-01 to YUE-20):
1. Score parser, validator, chord extraction, diffing, and editing (score.py)
2. Task and parameter mapping, NaN guards, and error handling (adapter.py)
3. Client URL sanitization, task dispatch, and error translation (client.py)
4. Runtime lifecycle and Gate Y0 Feasibility Checker (runtime.py)
5. MusicWorker multi-model routing (yue2 vs ace-step-1.5) and security preservation (worker.py)
"""
import asyncio
import base64
import hashlib
import math
import os
import tempfile
import uuid
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from factory_contracts.errors import (
    ErrorCode,
    InvalidRequestError,
    ModelUnavailableError,
    UnsupportedCapabilityError,
)
from factory_contracts.jobs.job_contract import FactoryJob
from services.music.yue2.adapter import YuE2Adapter
from services.music.yue2.client import YuE2Client
from services.music.yue2.runtime import (
    FeasibilityStatus,
    RuntimeState,
    YuE2Runtime,
    check_gate_y0,
)
from services.music.yue2.score import (
    diff_scores,
    edit_score,
    extract_chords,
    extract_headers,
    generate_sample_abc,
    validate_abc_score,
)
from services.music.worker import MusicWorker, SUPPORTED_MODELS, SUPPORTED_TASKS


# ============================================================================
# 1. ABC Notation Score Parser, Validator, Diffing & Editing Tests
# ============================================================================

class TestABCScoreEngine:
    def test_valid_score_parsing_and_header_extraction(self):
        abc = generate_sample_abc(title="Pop Anthem", key="G", meter="4/4", tempo=128)
        info = validate_abc_score(abc)
        assert info["valid"] is True
        headers = info["headers"]
        assert headers["T"] == "Pop Anthem"
        assert headers["K"] == "G"
        assert headers["M"] == "4/4"
        assert headers["Q"] == "128"
        assert headers["L"] == "1/8"
        assert info["measures_count"] == 4
        assert len(info["chords"]) == 4

    def test_extract_headers_helper(self):
        text = """X:42
T:Ballad in Am
C:Traditional
M:3/4
L:1/4
Q:96
K:Am
"Am" A B c | "Dm" d e f |]"""
        headers = extract_headers(text)
        assert headers["X"] == "42"
        assert headers["T"] == "Ballad in Am"
        assert headers["C"] == "Traditional"
        assert headers["M"] == "3/4"
        assert headers["L"] == "1/4"
        assert headers["Q"] == "96"
        assert headers["K"] == "Am"

    def test_extract_chords(self):
        text = """X:1
T:Chords Test
M:4/4
K:C
"C" C4 | "G7" G4 | "Am" A4 | "Fmaj7" F4 | "D/F#" D4 |]"""
        chords = extract_chords(text)
        assert chords == ["C", "G7", "Am", "Fmaj7", "D/F#"]

    def test_reject_empty_or_whitespace_score(self):
        with pytest.raises(InvalidRequestError, match="cannot be empty"):
            validate_abc_score("")
        with pytest.raises(InvalidRequestError, match="cannot be empty"):
            validate_abc_score("   \n\t  ")
        with pytest.raises(InvalidRequestError, match="cannot be empty"):
            validate_abc_score(None)  # type: ignore

    def test_reject_missing_key_header(self):
        no_key = """X:1
T:No Key Tune
M:4/4
C D E F | G A B c |]"""
        with pytest.raises(InvalidRequestError, match="must include a valid Key header"):
            validate_abc_score(no_key)

    def test_reject_malformed_headers(self):
        # Invalid meter
        bad_meter = "X:1\nM:999/0\nK:C\nC D E F |]"
        with pytest.raises(InvalidRequestError, match="Invalid meter header"):
            validate_abc_score(bad_meter)

        # Invalid key
        bad_key = "X:1\nM:4/4\nK:H#major\nC D E F |]"
        with pytest.raises(InvalidRequestError, match="Invalid key header"):
            validate_abc_score(bad_key)

        # Invalid tempo
        bad_tempo = "X:1\nM:4/4\nQ:fast_allegro\nK:C\nC D E F |]"
        with pytest.raises(InvalidRequestError, match="Invalid tempo header"):
            validate_abc_score(bad_tempo)

        # Invalid unit note length
        bad_len = "X:1\nL:1/7\nK:C\nC D E F |]"
        with pytest.raises(InvalidRequestError, match="Invalid unit note length header"):
            validate_abc_score(bad_len)

    def test_reject_unclosed_delimiters(self):
        # Unclosed quote on line
        bad_quote = 'X:1\nK:C\n"C C2 E2 G2 |]'
        with pytest.raises(InvalidRequestError, match="unclosed quotes"):
            validate_abc_score(bad_quote)

        # Unclosed bracket
        bad_bracket = "X:1\nK:C\n[CEG C2 E2 |]"
        with pytest.raises(InvalidRequestError, match="unclosed delimiter '\\["):
            validate_abc_score(bad_bracket)

        # Unclosed brace
        bad_brace = "X:1\nK:C\n{g C2 E2 G2 |]"
        with pytest.raises(InvalidRequestError, match="unclosed delimiter '\\{"):
            validate_abc_score(bad_brace)

    def test_reject_score_without_notes_or_rests(self):
        header_only = "X:1\nT:Empty\nM:4/4\nK:C\n% Just comments here\n"
        with pytest.raises(InvalidRequestError, match="must contain musical notes or rests"):
            validate_abc_score(header_only)

    def test_score_diffing(self):
        score_a = generate_sample_abc(title="Version A", key="C", tempo=120)
        score_b = generate_sample_abc(title="Version B", key="D", tempo=135)
        diff = diff_scores(score_a, score_b)
        assert "T" in diff["header_changes"]
        assert diff["header_changes"]["T"]["from"] == "Version A"
        assert diff["header_changes"]["T"]["to"] == "Version B"
        assert diff["header_changes"]["K"]["from"] == "C"
        assert diff["header_changes"]["K"]["to"] == "D"
        assert diff["header_changes"]["Q"]["from"] == "120"
        assert diff["header_changes"]["Q"]["to"] == "135"

    def test_score_editing(self):
        base = generate_sample_abc(title="Original", key="C", tempo=100)
        edits = {
            "title": "Edited Tune",
            "tempo": 125,
            "key": "G",
            "replace_chords": {"C": "Cmaj7", "F": "Fmaj7"},
        }
        edited = edit_score(base, edits)
        val = validate_abc_score(edited)
        assert val["valid"] is True
        assert val["headers"]["T"] == "Edited Tune"
        assert val["headers"]["Q"] == "125"
        assert val["headers"]["K"] == "G"
        assert "Cmaj7" in val["chords"]
        assert "Fmaj7" in val["chords"]
        assert "C" not in val["chords"]


# ============================================================================
# 2. YuE2 Adapter Tests
# ============================================================================

class TestYuE2Adapter:
    def test_task_normalization(self):
        assert YuE2Adapter.normalize_task("text_to_music") == "text_to_music"
        assert YuE2Adapter.normalize_task("music.t2m") == "text_to_music"
        assert YuE2Adapter.normalize_task("t2m") == "text_to_music"
        assert YuE2Adapter.normalize_task("lyrics_to_music") == "lyrics_to_music"
        assert YuE2Adapter.normalize_task("l2m") == "lyrics_to_music"
        assert YuE2Adapter.normalize_task("symbolic_score") == "symbolic_score"
        assert YuE2Adapter.normalize_task("score") == "symbolic_score"
        assert YuE2Adapter.normalize_task("custom_score") == "custom_score"
        assert YuE2Adapter.normalize_task("score_edit") == "score_edit"
        assert YuE2Adapter.normalize_task("song_variation") == "song_variation"

    def test_variant_normalization(self):
        assert YuE2Adapter.normalize_variant(None) == "standard"
        assert YuE2Adapter.normalize_variant("") == "standard"
        assert YuE2Adapter.normalize_variant("default") == "standard"
        assert YuE2Adapter.normalize_variant("7b") == "7b"
        assert YuE2Adapter.normalize_variant("3b") == "3b"
        assert YuE2Adapter.normalize_variant("track_a") == "track_a"
        assert YuE2Adapter.normalize_variant("track_b") == "track_b"
        assert YuE2Adapter.normalize_variant("quantized") == "quantized"

    def test_duration_validation_guards(self):
        # Valid duration
        assert YuE2Adapter.validate_duration(45.0) == 45.0
        assert YuE2Adapter.validate_duration("60") == 60.0
        assert YuE2Adapter.validate_duration(None) == 30.0

        # NaN guard
        with pytest.raises(InvalidRequestError, match="cannot be NaN"):
            YuE2Adapter.validate_duration(float("nan"))

        # Inf guard
        with pytest.raises(InvalidRequestError, match="cannot be infinite"):
            YuE2Adapter.validate_duration(float("inf"))
        with pytest.raises(InvalidRequestError, match="cannot be infinite"):
            YuE2Adapter.validate_duration(float("-inf"))

        # Out of bounds
        with pytest.raises(InvalidRequestError, match="must be > 0"):
            YuE2Adapter.validate_duration(0.0)
        with pytest.raises(InvalidRequestError, match="must be > 0"):
            YuE2Adapter.validate_duration(-15.0)
        with pytest.raises(InvalidRequestError, match="<= 300"):
            YuE2Adapter.validate_duration(300.1)

    def test_seed_validation(self):
        assert YuE2Adapter.validate_seed(None) is None
        assert YuE2Adapter.validate_seed(12345) == 12345
        assert YuE2Adapter.validate_seed("999") == 999
        assert YuE2Adapter.validate_seed(0) == 0

        # NaN seed
        with pytest.raises(InvalidRequestError, match="cannot be NaN"):
            YuE2Adapter.validate_seed(float("nan"))

        # Negative seed
        with pytest.raises(InvalidRequestError, match="between 0 and 2\\^32 - 1"):
            YuE2Adapter.validate_seed(-1)

        # Overflow seed
        with pytest.raises(InvalidRequestError, match="between 0 and 2\\^32 - 1"):
            YuE2Adapter.validate_seed(2**33)

    def test_prompt_and_lyrics_validation(self):
        # text_to_music requires prompt
        with pytest.raises(InvalidRequestError, match="prompt is required"):
            YuE2Adapter.map_inputs("text_to_music", {"prompt": ""})

        # lyrics_to_music requires lyrics
        with pytest.raises(InvalidRequestError, match="lyrics are required"):
            YuE2Adapter.map_inputs("lyrics_to_music", {"prompt": "pop style", "lyrics": ""})

        # custom_score requires valid ABC score
        with pytest.raises(InvalidRequestError, match="custom_score"):
            YuE2Adapter.map_inputs("custom_score", {"prompt": "render score"})

        # custom_score with malformed ABC
        with pytest.raises(InvalidRequestError):
            YuE2Adapter.map_inputs("custom_score", {"custom_score": "malformed score without key"})

        # valid custom_score
        valid_abc = generate_sample_abc()
        mapped = YuE2Adapter.map_inputs("custom_score", {"custom_score": valid_abc})
        assert mapped["score_prompt"] == valid_abc

    def test_score_edit_validation(self):
        valid_abc = generate_sample_abc()
        # Missing base_score
        with pytest.raises(InvalidRequestError, match="score_edit requires base_score"):
            YuE2Adapter.map_inputs("score_edit", {"modifications": {"tempo": 140}})

        # Missing modifications
        with pytest.raises(InvalidRequestError, match="requires modifications dictionary"):
            YuE2Adapter.map_inputs("score_edit", {"base_score": valid_abc})

        # Valid score_edit
        mapped = YuE2Adapter.map_inputs("score_edit", {
            "base_score": valid_abc,
            "modifications": {"tempo": 140},
        })
        assert mapped["task"] == "score_edit"
        assert mapped["modifications"]["tempo"] == 140

    def test_sampling_config_validation(self):
        # Valid config
        cfg = YuE2Adapter.validate_sampling_config({"temperature": 0.8, "top_p": 0.9, "top_k": 40})
        assert cfg["temperature"] == 0.8
        assert cfg["top_p"] == 0.9
        assert cfg["top_k"] == 40

        # NaN temperature
        with pytest.raises(InvalidRequestError, match="temperature"):
            YuE2Adapter.validate_sampling_config({"temperature": float("nan")})

        # Out of bounds temperature
        with pytest.raises(InvalidRequestError, match="temperature"):
            YuE2Adapter.validate_sampling_config({"temperature": 2.5})

        # Out of bounds top_p
        with pytest.raises(InvalidRequestError, match="top_p"):
            YuE2Adapter.validate_sampling_config({"top_p": 1.2})

    def test_unsupported_task_and_variant(self):
        with pytest.raises(UnsupportedCapabilityError, match="does not support task 'unsupported_task'"):
            YuE2Adapter.map_inputs("unsupported_task", {"prompt": "song"})

        with pytest.raises(UnsupportedCapabilityError, match="variant 'unsupported_variant' is not recognized"):
            YuE2Adapter.map_inputs("text_to_music", {"prompt": "song", "variant": "unsupported_variant"})


# ============================================================================
# 3. YuE2 Client Tests
# ============================================================================

class TestYuE2Client:
    def test_base_url_stripping_and_initialization(self):
        client = YuE2Client(api_base_url="  http://127.0.0.1:8009///  ")
        assert client.api_base_url == "http://127.0.0.1:8009"

    def test_headers_with_and_without_api_key(self, monkeypatch):
        monkeypatch.setenv("YUE2_API_KEY", "secret-key-123")
        client = YuE2Client()
        headers = client._headers()
        assert headers["Authorization"] == "Bearer secret-key-123"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_health_success(self):
        client = YuE2Client(api_base_url="http://mock-yue2")
        mock_resp = httpx.Response(200, json={"status": "healthy", "model": "yue2", "version": "2.0.0"})

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_resp
            res = await client.health()
            assert res["status"] == "healthy"
            assert res["model"] == "yue2"

    @pytest.mark.asyncio
    async def test_health_network_failure_translates_error(self):
        client = YuE2Client(api_base_url="http://mock-yue2")
        with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection refused")):
            with pytest.raises(ModelUnavailableError) as exc_info:
                await client.health()
            assert exc_info.value.code == "BACKEND_UNAVAILABLE"
            assert "unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_song_success_with_audio_and_abc_artifacts(self):
        client = YuE2Client(api_base_url="http://mock-yue2")
        fake_wav = b"RIFF____WAVEfmt ____data____"
        fake_abc = generate_sample_abc(title="Generated Melody")
        mock_payload = {
            "status": "completed",
            "audio_base64": base64.b64encode(fake_wav).decode("ascii"),
            "abc_score": fake_abc,
            "version": "2.0.0",
        }
        mock_resp = httpx.Response(200, json=mock_payload)

        progress_calls = []
        def on_prog(step, pct, stage):
            progress_calls.append((step, pct, stage))

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            params = {
                "prompt": "happy acoustic song",
                "audio_duration": 30.0,
                "seed": 42,
                "variant": "standard",
            }
            res = await client.generate_song("text_to_music", params, progress_callback=on_prog)

            assert res["audio_bytes"] == fake_wav
            assert res["sha256"] == hashlib.sha256(fake_wav).hexdigest()
            assert res["mime_type"] == "audio/wav"
            assert len(res["artifacts"]) == 2
            assert res["artifacts"][0]["type"] == "audio"
            assert res["artifacts"][1]["type"] == "score"
            assert res["artifacts"][1]["mime_type"] == "text/vnd.abc"
            assert len(progress_calls) >= 2

    @pytest.mark.asyncio
    async def test_generate_song_network_error_translates_error(self):
        client = YuE2Client(api_base_url="http://mock-yue2")
        with patch("httpx.AsyncClient.post", side_effect=httpx.ReadTimeout("Read timed out")):
            with pytest.raises(ModelUnavailableError) as exc_info:
                await client.generate_song("text_to_music", {"prompt": "synthwave"})
            assert exc_info.value.code == "BACKEND_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_plan_score_success(self):
        client = YuE2Client(api_base_url="http://mock-yue2")
        sample_abc = generate_sample_abc(title="Planned Score")
        mock_resp = httpx.Response(200, json={"abc_score": sample_abc})

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            res = await client.plan_score({"prompt": "jazz quartet", "seed": 100})
            assert res["abc_score"] == sample_abc
            assert res["score_info"]["valid"] is True
            assert len(res["artifacts"]) == 1
            assert res["artifacts"][0]["type"] == "score"

    @pytest.mark.asyncio
    async def test_edit_score_client(self):
        client = YuE2Client(api_base_url="http://mock-yue2")
        base = generate_sample_abc(tempo=100)
        res = await client.edit_score(base, {"tempo": 130})
        assert res["score_info"]["headers"]["Q"] == "130"
        assert len(res["artifacts"]) == 1
        assert res["artifacts"][0]["type"] == "score"


# ============================================================================
# 4. YuE2 Runtime & Gate Y0 Feasibility Checker Tests
# ============================================================================

class TestYuE2RuntimeAndGateY0:
    def test_gate_y0_hardware_blocked_when_no_cuda(self):
        # Current local Mac environment has no CUDA
        res = check_gate_y0()
        assert res["status"] == FeasibilityStatus.HARDWARE_BLOCKED.value
        assert res["can_run_live"] is False
        assert "CUDA is not available" in res["reason"] or "PyTorch is not installed" in res["reason"]

    def test_gate_y0_insufficient_vram_reports_blocked(self):
        mock_props = MagicMock()
        mock_props.total_memory = 8 * (1024**3)  # 8GB VRAM (< 12GB threshold)
        mock_props.name = "NVIDIA GeForce RTX 3070"
        mock_props.major = 8
        mock_props.minor = 6

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.is_bf16_supported.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            res = check_gate_y0(0)
            assert res["status"] == FeasibilityStatus.HARDWARE_BLOCKED.value
            assert res["can_run_live"] is False
            assert "Insufficient VRAM" in res["reason"]
            assert res["details"]["total_vram_gb"] == 8.0

    def test_gate_y0_dual_t4_reports_track_b_only(self):
        # Tesla T4: 16GB VRAM, Turing CC 7.5, no native fast BF16
        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024**3)  # 16GB VRAM
        mock_props.name = "Tesla T4"
        mock_props.major = 7
        mock_props.minor = 5

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.is_bf16_supported.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            res = check_gate_y0(0)
            assert res["status"] == FeasibilityStatus.FEASIBLE_TRACK_B_ONLY.value
            assert res["can_run_live"] is True
            assert "track_b" in res["supported_tracks"]
            assert "track_a" not in res["supported_tracks"]

    def test_gate_y0_24gb_bf16_reports_fully_feasible(self):
        # A10G / A100: 24GB VRAM, Ampere CC 8.0, BF16 supported
        mock_props = MagicMock()
        mock_props.total_memory = 24 * (1024**3)
        mock_props.name = "NVIDIA A10G"
        mock_props.major = 8
        mock_props.minor = 6

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.is_bf16_supported.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            res = check_gate_y0(0)
            assert res["status"] == FeasibilityStatus.FEASIBLE.value
            assert res["can_run_live"] is True
            assert "track_a" in res["supported_tracks"]
            assert "track_b" in res["supported_tracks"]

    def test_runtime_lifecycle_transitions(self):
        runtime = YuE2Runtime(model_id="yue2", variant="standard", gpu_device=0)
        assert runtime.state == RuntimeState.UNLOADED

        with patch("services.music.yue2.runtime.check_gate_y0", return_value={
            "status": FeasibilityStatus.FEASIBLE.value,
            "reason": "OK",
        }):
            runtime.ensure_loaded()
            assert runtime.state == RuntimeState.READY
            status_data = runtime.status()
            assert status_data["state"] == "READY"

            runtime.reserve("lock-1")
            assert runtime.state == RuntimeState.BUSY

            runtime.release()
            assert runtime.state == RuntimeState.READY

            runtime.unload()
            assert runtime.state == RuntimeState.UNLOADED

    def test_runtime_ensure_loaded_blocked_raises(self):
        runtime = YuE2Runtime(model_id="yue2", variant="standard", gpu_device=0)
        with patch("services.music.yue2.runtime.check_gate_y0", return_value={
            "status": FeasibilityStatus.HARDWARE_BLOCKED.value,
            "reason": "No CUDA device found",
        }):
            with pytest.raises(ModelUnavailableError, match="No CUDA device found") as exc:
                runtime.ensure_loaded()
            assert exc.value.code == "INSUFFICIENT_VRAM"


# ============================================================================
# 5. MusicWorker Multi-Model Routing & Security Tests
# ============================================================================

class TestMusicWorkerRoutingAndSecurity:
    def test_worker_supported_models_and_tasks(self):
        worker = MusicWorker()
        assert "ace-step-1.5" in SUPPORTED_MODELS
        assert "yue2" in SUPPORTED_MODELS
        assert "symbolic_score" in SUPPORTED_TASKS
        assert "custom_score" in SUPPORTED_TASKS
        assert "score_edit" in SUPPORTED_TASKS
        assert "song_variation" in SUPPORTED_TASKS
        assert isinstance(worker.yue2_client, YuE2Client)

    @pytest.mark.asyncio
    async def test_worker_lease_next_job_includes_yue2(self):
        worker = MusicWorker(control_plane_url="http://mock-control", api_key="secret-key")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock()
        mock_http.post.return_value = httpx.Response(204)  # No job available

        await worker.lease_next_job(mock_http)
        assert mock_http.post.called
        call_json = mock_http.post.call_args[1]["json"]
        assert "ace-step-1.5" in call_json["supported_models"]
        assert "yue2" in call_json["supported_models"]
        assert "text_to_music" in call_json["supported_tasks"]
        assert "symbolic_score" in call_json["supported_tasks"]

    @pytest.mark.asyncio
    async def test_worker_routes_yue2_job_to_completion(self):
        worker = MusicWorker(control_plane_url="http://mock-control", api_key="secret-key")
        worker._active_lease_token = "lease-token-yue2"

        uid = str(uuid.uuid4())
        fake_job = FactoryJob(
            job_id=f"job-{uid}",
            request_id=f"req-{uid}",
            service="music",
            model="yue2",
            task="text_to_music",
            inputs={
                "prompt": "epic rock ballad",
                "audio_duration": 30.0,
                "seed": 42,
            },
        )

        fake_audio = b"RIFF____WAVEYUE2TEST____"
        fake_abc = generate_sample_abc(title="YuE2 Rock Ballad")
        fake_result = {
            "audio_bytes": fake_audio,
            "filename": "yue2_song.wav",
            "mime_type": "audio/wav",
            "size_bytes": len(fake_audio),
            "sha256": hashlib.sha256(fake_audio).hexdigest(),
            "artifacts": [
                {
                    "type": "audio",
                    "audio_bytes": fake_audio,
                    "filename": "yue2_song.wav",
                    "mime_type": "audio/wav",
                    "size_bytes": len(fake_audio),
                    "sha256": hashlib.sha256(fake_audio).hexdigest(),
                },
                {
                    "type": "score",
                    "audio_bytes": fake_abc.encode("utf-8"),
                    "filename": "yue2_song.abc",
                    "mime_type": "text/vnd.abc",
                    "size_bytes": len(fake_abc.encode("utf-8")),
                    "sha256": hashlib.sha256(fake_abc.encode("utf-8")).hexdigest(),
                },
            ],
        }

        # Mock yue2 client generate_song
        worker.yue2_client.generate_song = AsyncMock(return_value=fake_result)
        # Mock acestep client to confirm it is NOT called
        worker.acestep_client.generate_music = AsyncMock()

        # Mock HTTP client for upload-url, download-url, and complete calls
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        async def mock_post(url, *args, **kwargs):
            req = httpx.Request("POST", url)
            if "upload-url" in url:
                return httpx.Response(200, json={
                    "upload_url": "http://storage.internal/upload",
                    "storage_path": "music/artifacts/test",
                }, request=req)
            elif "download-url" in url:
                return httpx.Response(200, json={
                    "download_url": "http://storage.internal/download",
                }, request=req)
            elif "/complete" in url:
                return httpx.Response(200, json={"status": "completed"}, request=req)
            return httpx.Response(200, json={}, request=req)

        mock_http.post = AsyncMock(side_effect=mock_post)

        # Mock upload client PUT
        with patch("httpx.AsyncClient.put", new_callable=AsyncMock) as mock_put:
            mock_put.return_value = httpx.Response(200, request=httpx.Request("PUT", "http://storage.internal/upload"))
            success = await worker.execute_job(mock_http, fake_job)

            assert success is True
            assert worker.yue2_client.generate_song.called
            assert not worker.acestep_client.generate_music.called

            # Check that /complete received 2 artifacts: 1 audio and 1 score
            complete_calls = [c for c in mock_http.post.call_args_list if "/complete" in c[0][0]]
            assert len(complete_calls) == 1
            comp_payload = complete_calls[0][1]["json"]
            assert len(comp_payload["artifacts"]) == 2
            art_types = [a["type"] for a in comp_payload["artifacts"]]
            assert "audio" in art_types
            assert "score" in art_types

    @pytest.mark.asyncio
    async def test_worker_routes_yue2_plan_only_task(self):
        worker = MusicWorker(control_plane_url="http://mock-control", api_key="secret-key")
        uid = str(uuid.uuid4())
        fake_job = FactoryJob(
            job_id=f"job-{uid}",
            request_id=f"req-{uid}",
            service="music",
            model="yue2",
            task="symbolic_score",
            inputs={
                "prompt": "folk acoustic",
                "plan_only": True,
            },
        )
        fake_abc = generate_sample_abc(title="Planned Folk")
        worker.yue2_client.plan_score = AsyncMock(return_value={
            "abc_score": fake_abc,
            "artifacts": [{
                "type": "score",
                "audio_bytes": fake_abc.encode("utf-8"),
                "filename": "planned.abc",
                "mime_type": "text/vnd.abc",
                "size_bytes": len(fake_abc.encode("utf-8")),
                "sha256": hashlib.sha256(fake_abc.encode("utf-8")).hexdigest(),
            }],
        })

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        async def mock_post(url, *args, **kwargs):
            req = httpx.Request("POST", url)
            if "upload-url" in url:
                return httpx.Response(200, json={"upload_url": "http://s/up", "storage_path": "p"}, request=req)
            elif "download-url" in url:
                return httpx.Response(200, json={"download_url": "http://s/down"}, request=req)
            elif "/complete" in url:
                return httpx.Response(200, json={"status": "completed"}, request=req)
            return httpx.Response(200, json={}, request=req)

        mock_http.post = AsyncMock(side_effect=mock_post)
        with patch("httpx.AsyncClient.put", new_callable=AsyncMock) as mock_put:
            mock_put.return_value = httpx.Response(200, request=httpx.Request("PUT", "http://s/up"))
            success = await worker.execute_job(mock_http, fake_job)
            assert success is True
            assert worker.yue2_client.plan_score.called

    @pytest.mark.asyncio
    async def test_worker_preserves_file_url_safe_staging_and_cleanup(self):
        worker = MusicWorker(control_plane_url="http://mock-control", api_key="secret-key")

        # Create temporary file to test file:// staging
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as orig_f:
            orig_f.write(b"ORIGINAL_AUDIO_CONTENT")
            orig_path = orig_f.name

        try:
            mock_http = AsyncMock(spec=httpx.AsyncClient)
            staged_path = await worker.stage_asset(mock_http, f"file://{orig_path}")

            # Verify it created a separate staging copy with staged_local_ prefix
            assert os.path.exists(staged_path)
            assert staged_path != orig_path
            assert "staged_local_" in os.path.basename(staged_path)
            with open(staged_path, "rb") as sf:
                assert sf.read() == b"ORIGINAL_AUDIO_CONTENT"

            # Clean up staged path manually
            if os.path.exists(staged_path):
                os.unlink(staged_path)
        finally:
            if os.path.exists(orig_path):
                os.unlink(orig_path)

    @pytest.mark.asyncio
    async def test_worker_ssrf_blocked_on_unsafe_remote_url(self):
        worker = MusicWorker(control_plane_url="http://mock-control", api_key="secret-key")
        mock_http = AsyncMock(spec=httpx.AsyncClient)

        # 169.254.169.254 metadata IP should be blocked by SSRF validator
        with pytest.raises(InvalidRequestError, match="SSRF"):
            await worker.stage_asset(mock_http, "http://169.254.169.254/latest/meta-data")
