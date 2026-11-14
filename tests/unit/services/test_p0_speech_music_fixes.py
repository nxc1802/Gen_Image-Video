"""
tests/unit/services/test_p0_speech_music_fixes.py
Targeted regression tests for Phase 0 fixes:
- FIX-08: Ingest audio_url into actual bytes and handoff to ASREngine.
- FIX-09: Native ACE-Step launcher script health checks native backend.
- FIX-10: Align task type and payload with actual ACE-Step native API.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from factory_contracts.errors import InvalidRequestError, ModelUnavailableError
from services.speech.asr.engine import ASREngine
from services.music.acestep.client import ACEStepClient


def test_fix_08_asr_engine_ingests_audio_url_into_bytes():
    """FIX-08: ASREngine accepts audio_url and ingests actual bytes instead of hard erroring."""
    engine = ASREngine()
    engine.upstream_url = "http://localhost:8000"

    # Test with data: URI
    import base64
    fake_wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00data\x00\x00\x00\x00"
    data_url = f"data:audio/wav;base64,{base64.b64encode(fake_wav_bytes).decode('ascii')}"

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Hello world from ingested audio"}
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        res = engine.transcribe(audio_url=data_url)
        assert res["text"] == "Hello world from ingested audio"
        # Verify post called with real ingested bytes
        call_kwargs = mock_client.post.call_args[1]
        assert "files" in call_kwargs
        assert call_kwargs["files"]["file"][1] == fake_wav_bytes


def test_fix_09_native_acestep_launcher_health_check():
    """FIX-09: Verify run_acestep_api.sh launches native acestep and performs health check."""
    script_path = Path("services/music/launcher/run_acestep_api.sh")
    assert script_path.exists()
    content = script_path.read_text()

    # Must NOT launch deprecated services.music.server
    assert "services.music.server" not in content
    # Must invoke acestep-api / native entrypoint
    assert "acestep-api" in content or "acestep.api" in content
    # Must perform health check loop on /health
    assert "/health" in content
    assert "HEALTHY" in content


def test_fix_10_acestep_task_type_and_payload_alignment():
    """FIX-10: ACEStepClient._release_payload correctly maps native tasks and parameters."""
    # 1. Text to music
    p1 = ACEStepClient._release_payload('music.text_to_music', {"prompt": "A jazz groove", "tempo": 110})
    assert p1["task_type"] == "text2music"
    assert p1["lyrics"] == "[inst]"
    assert p1["bpm"] == 110

    # 2. Lyrics to music
    p2 = ACEStepClient._release_payload('music.lyrics_to_music', {"prompt": "Pop ballad", "lyrics": "Hello sun"})
    assert p2["task_type"] == "lyrics2music"
    assert p2["lyrics"] == "Hello sun"

    # 3. Cover task with audio path and cover strength
    p3 = ACEStepClient._release_payload(
        'music.cover',
        {
            "prompt": "Rock cover",
            "source_audio_path": "/tmp/source.wav",
            "audio_cover_strength": 0.75,
        },
    )
    assert p3["task_type"] == "cover"
    assert p3["src_audio_path"] == "/tmp/source.wav"
    assert p3["audio_cover_strength"] == 0.75

    # 4. Repaint task with interval
    p4 = ACEStepClient._release_payload(
        'music.repaint',
        {
            "prompt": "Guitar solo",
            "src_audio_path": "/tmp/source.wav",
            "repainting_start": 10.0,
            "repainting_end": 25.0,
        },
    )
    assert p4["task_type"] == "repaint"
    assert p4["repainting_start"] == 10.0
    assert p4["repainting_end"] == 25.0

    # 5. Reference conditioning
    p5 = ACEStepClient._release_payload(
        'music.reference_audio',
        {
            "prompt": "Vocal style transfer",
            "reference_audio_path": "/tmp/style_ref.wav",
        },
    )
    assert p5["task_type"] == "reference"
    assert p5["reference_audio_path"] == "/tmp/style_ref.wav"

    # 6. Extract stem
    p6 = ACEStepClient._release_payload('music.extract', {"prompt": "Vocal stem", "src_audio_path": "/tmp/mix.wav"})
    assert p6["task_type"] == "extract"
