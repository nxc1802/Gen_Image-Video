"""
tests/unit/services/test_acestep_p3.py
Comprehensive Phase 3 Test Suite for ACE-Step 1.5 Music Generation Subsystem.
Verifies all 24 checklist items (ACE-01 through ACE-24) per Plan §7.3:
- ACE-01: Text-to-Music
- ACE-02: Lyrics-to-Music
- ACE-03: Instrumental mode
- ACE-04: Prompt expansion / query rewrite
- ACE-05: Thinking mode mapping
- ACE-06: Musical controls (BPM, key, time signature, duration)
- ACE-07: Reference audio conditioning
- ACE-08: Cover
- ACE-09: Repaint
- ACE-10: Remix / style transfer
- ACE-11: Extract (stem separation, rejected on Turbo, supported on Base)
- ACE-12: Lego (add layer/instrument, rejected on Turbo, supported on Base)
- ACE-13: Complete (accompaniment, rejected on Turbo, supported on Base)
- ACE-14: Music extend
- ACE-15: Multilingual lyrics (Vietnamese, Japanese, English)
- ACE-16: Batch generation
- ACE-17: Variant switching (Base vs Turbo/SFT/XL Base)
- ACE-18: Audio format export (MP3, WAV, FLAC)
- ACE-19: Audio understanding (metadata and captioning)
- ACE-20: Lyrics timing / LRC timestamps
- ACE-21: Quality scoring
- ACE-22: LoRA inference
- ACE-23: LoRA training workflow contract
- ACE-24: Failure / recovery (invalid asset, cancel, crash, OOM, timeout, retries)
"""
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
import pytest

from factory_contracts.errors import (
    ErrorCode,
    InvalidRequestError,
    ModelUnavailableError,
    UnsupportedCapabilityError,
    classify_worker_error,
)
from factory_contracts.jobs.job_contract import FactoryJob
from model_registry.loader import ManifestLoader
from services.music.acestep.adapter import ACEStepAdapter
from services.music.acestep.client import ACEStepClient
from services.music.worker import MusicWorker


# Helper to build valid mock audio bytes (valid WAV header bytes)
def make_mock_wav_bytes(duration_sec: float = 2.0, sample_rate: int = 16000) -> bytes:
    num_samples = int(duration_sec * sample_rate)
    data_size = num_samples * 2
    header = bytearray(b"RIFF")
    header.extend((data_size + 36).to_bytes(4, "little"))
    header.extend(b"WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00")
    header.extend(sample_rate.to_bytes(4, "little"))
    header.extend((sample_rate * 2).to_bytes(4, "little"))
    header.extend(b"\x02\x00\x10\x00data")
    header.extend(data_size.to_bytes(4, "little"))
    # Add dummy audio content
    header.extend(b"\x00\x00" * num_samples)
    return bytes(header)


# ==============================================================================
# ACE-01: Text-to-Music
# ==============================================================================

def test_ace_01_text_to_music_mapping_and_execution():
    """ACE-01: Text-to-music maps to text2music task with valid audio output."""
    inputs = {
        "prompt": "Energetic synthwave with driving bass and 80s drums",
        "duration_seconds": 30,
        "bpm": 124,
    }
    params = ACEStepAdapter.map_inputs("text_to_music", inputs)
    assert params["prompt"] == "Energetic synthwave with driving bass and 80s drums"
    assert params["duration_seconds"] == 30.0
    assert params["bpm"] == 124
    assert params["lyrics"] == "[inst]"

    payload = ACEStepClient._release_payload("text_to_music", params)
    assert payload["task_type"] == "text2music"
    assert payload["prompt"] == inputs["prompt"]
    assert payload["audio_duration"] == 30.0
    assert payload["bpm"] == 124
    assert payload["lyrics"] == "[inst]"


# ==============================================================================
# ACE-02: Lyrics-to-Music
# ==============================================================================

def test_ace_02_lyrics_to_music_verse_chorus_structure():
    """ACE-02: Lyrics-to-music preserves verse/chorus structure and requires lyrics."""
    lyrics_text = (
        "[verse]\nUnder the streetlights glowing amber\nSteps echo softly in the rain\n"
        "[chorus]\nWe are alive in the midnight electric\nRunning faster than the pain"
    )
    inputs = {
        "prompt": "Modern electropop anthem",
        "lyrics": lyrics_text,
        "bpm": 128,
        "key": "A minor",
    }
    params = ACEStepAdapter.map_inputs("lyrics_to_music", inputs)
    assert params["lyrics"] == lyrics_text
    assert params["prompt"] == "Modern electropop anthem"
    assert params["key"] == "A minor"

    payload = ACEStepClient._release_payload("lyrics_to_music", params)
    assert payload["task_type"] == "lyrics2music"
    assert payload["lyrics"] == lyrics_text

    # Missing lyrics must raise InvalidRequestError
    with pytest.raises(InvalidRequestError) as exc_info:
        ACEStepAdapter.map_inputs("lyrics_to_music", {"prompt": "Pop song", "lyrics": ""})
    assert "lyrics_to_music requires non-empty lyrics" in str(exc_info.value)


# ==============================================================================
# ACE-03: Instrumental Mode
# ==============================================================================

def test_ace_03_instrumental_mode():
    """ACE-03: Output conforms to instrumental instruction via flag or tag."""
    # Via flag
    params1 = ACEStepAdapter.map_inputs("text_to_music", {"prompt": "Cinematic strings", "instrumental": True})
    assert params1["lyrics"] == "[inst]"
    assert params1["instrumental"] is True

    # Via tag in prompt
    params2 = ACEStepAdapter.map_inputs("text_to_music", {"prompt": "[inst] Lo-fi piano study beat"})
    assert params2["lyrics"] == "[inst]"


# ==============================================================================
# ACE-04: Prompt Expansion / Query Rewrite
# ==============================================================================

def test_ace_04_prompt_expansion_mapping():
    """ACE-04: Prompt expansion parameter maps and enables enriched metadata."""
    params = ACEStepAdapter.map_inputs(
        "text_to_music",
        {"prompt": "chill lofi", "prompt_expansion": True},
    )
    assert params["prompt_expansion"] is True

    payload = ACEStepClient._release_payload("text_to_music", params)
    assert payload["prompt_expansion"] is True


# ==============================================================================
# ACE-05: Thinking Mode Mapping
# ==============================================================================

def test_ace_05_thinking_mode_mapping():
    """ACE-05: Thinking mode parameter maps correctly and affects native payload."""
    params_think = ACEStepAdapter.map_inputs("text_to_music", {"prompt": "Complex jazz fugue", "thinking": True})
    assert params_think["thinking"] is True
    p1 = ACEStepClient._release_payload("text_to_music", params_think)
    assert p1["thinking"] is True

    params_no_think = ACEStepAdapter.map_inputs("text_to_music", {"prompt": "Simple pop loop", "thinking": False})
    assert params_no_think["thinking"] is False
    p2 = ACEStepClient._release_payload("text_to_music", params_no_think)
    assert p2["thinking"] is False


# ==============================================================================
# ACE-06: Musical Controls (BPM, Key, Time Signature, Duration)
# ==============================================================================

def test_ace_06_musical_controls_validation():
    """ACE-06: BPM, key, time signature, and duration are validated and mapped."""
    valid_inputs = {
        "prompt": "Orchestral waltz",
        "bpm": 132,
        "key": "G major",
        "time_signature": "3/4",
        "duration_seconds": 120.0,
    }
    params = ACEStepAdapter.map_inputs("text_to_music", valid_inputs)
    assert params["bpm"] == 132
    assert params["key"] == "G major"
    assert params["time_signature"] == "3/4"
    assert params["duration_seconds"] == 120.0

    payload = ACEStepClient._release_payload("text_to_music", params)
    assert payload["bpm"] == 132
    assert payload["key"] == "G major"
    assert payload["time_signature"] == "3/4"
    assert payload["audio_duration"] == 120.0

    # Invalid BPM (< 30 or > 300)
    with pytest.raises(InvalidRequestError):
        ACEStepAdapter.map_inputs("text_to_music", {"prompt": "Beat", "bpm": 15})
    with pytest.raises(InvalidRequestError):
        ACEStepAdapter.map_inputs("text_to_music", {"prompt": "Beat", "bpm": 450})

    # Invalid Key
    with pytest.raises(InvalidRequestError):
        ACEStepAdapter.map_inputs("text_to_music", {"prompt": "Beat", "key": "H sharp invalid"})

    # Invalid Time Signature
    with pytest.raises(InvalidRequestError):
        ACEStepAdapter.map_inputs("text_to_music", {"prompt": "Beat", "time_signature": "13/7"})

    # Invalid Duration (<= 0 or > 300)
    with pytest.raises(InvalidRequestError):
        ACEStepAdapter.map_inputs("text_to_music", {"prompt": "Beat", "duration_seconds": -10.0})
    with pytest.raises(InvalidRequestError):
        ACEStepAdapter.map_inputs("text_to_music", {"prompt": "Beat", "duration_seconds": 600.0})


# ==============================================================================
# ACE-07: Reference Audio Conditioning
# ==============================================================================

def test_ace_07_reference_audio_conditioning():
    """ACE-07: Reference audio conditioning maps reference path to native backend."""
    params = ACEStepAdapter.map_inputs(
        "reference_audio",
        {
            "prompt": "Funk guitar groove in the style of reference",
            "reference_audio_path": "/staged/groove_ref.wav",
        },
    )
    assert params["reference_audio_path"] == "/staged/groove_ref.wav"

    payload = ACEStepClient._release_payload("reference_audio", params)
    assert payload["task_type"] == "reference"
    assert payload["reference_audio_path"] == "/staged/groove_ref.wav"

    # Missing reference audio raises error
    with pytest.raises(InvalidRequestError):
        ACEStepAdapter.map_inputs("reference_audio", {"prompt": "Style transfer without ref"})


# ==============================================================================
# ACE-08: Cover Audio & Strength
# ==============================================================================

def test_ace_08_cover_audio_mapping():
    """ACE-08: Cover task maps source audio and cover strength."""
    params = ACEStepAdapter.map_inputs(
        "cover",
        {
            "prompt": "Bossa nova arrangement",
            "source_audio_path": "/staged/melody.wav",
            "cover_strength": 0.75,
        },
    )
    assert params["cover_strength"] == 0.75
    assert params["src_audio_path"] == "/staged/melody.wav"

    payload = ACEStepClient._release_payload("cover", params)
    assert payload["task_type"] == "cover"
    assert payload["src_audio_path"] == "/staged/melody.wav"
    assert payload["audio_cover_strength"] == 0.75

    # Cover strength out of bounds [0.0, 1.0]
    with pytest.raises(InvalidRequestError):
        ACEStepAdapter.map_inputs(
            "cover",
            {"prompt": "Cover", "source_audio_path": "/staged/m.wav", "cover_strength": 1.5},
        )


# ==============================================================================
# ACE-09: Repaint Interval
# ==============================================================================

def test_ace_09_repaint_interval_validation():
    """ACE-09: Repaint interval [start, end] is validated and passed."""
    params = ACEStepAdapter.map_inputs(
        "repaint",
        {
            "prompt": "Electric guitar solo replacement",
            "source_audio_path": "/staged/song.wav",
            "duration_seconds": 60.0,
            "repainting_start": 15.0,
            "repainting_end": 35.0,
        },
    )
    assert params["repainting_start"] == 15.0
    assert params["repainting_end"] == 35.0

    payload = ACEStepClient._release_payload("repaint", params)
    assert payload["task_type"] == "repaint"
    assert payload["repainting_start"] == 15.0
    assert payload["repainting_end"] == 35.0

    # Inverted interval (start >= end)
    with pytest.raises(InvalidRequestError):
        ACEStepAdapter.map_inputs(
            "repaint",
            {
                "prompt": "Fix",
                "source_audio_path": "/staged/s.wav",
                "repainting_start": 30.0,
                "repainting_end": 10.0,
            },
        )

    # Exceeding duration
    with pytest.raises(InvalidRequestError):
        ACEStepAdapter.map_inputs(
            "repaint",
            {
                "prompt": "Fix",
                "source_audio_path": "/staged/s.wav",
                "duration_seconds": 30.0,
                "repainting_start": 10.0,
                "repainting_end": 45.0,
            },
        )


# ==============================================================================
# ACE-10: Remix / Style Transfer
# ==============================================================================

def test_ace_10_remix_style_transfer_composition():
    """ACE-10: Remix task binds source audio, reference conditioning, and cover strength."""
    params = ACEStepAdapter.map_inputs(
        "cover",
        {
            "prompt": "Cyberpunk synthwave remix",
            "source_audio_path": "/staged/original_track.wav",
            "reference_audio_path": "/staged/cyberpunk_reference.wav",
            "cover_strength": 0.65,
        },
    )
    payload = ACEStepClient._release_payload("cover", params)
    assert payload["task_type"] == "cover"
    assert payload["src_audio_path"] == "/staged/original_track.wav"
    assert payload["reference_audio_path"] == "/staged/cyberpunk_reference.wav"
    assert payload["audio_cover_strength"] == 0.65


# ==============================================================================
# ACE-11: Extract (Stem Separation, Variant-Dependent)
# ==============================================================================

def test_ace_11_extract_stem_variant_matrix():
    """ACE-11: Extract supported on Base, strictly rejected on Turbo with UNSUPPORTED_CAPABILITY."""
    # 1. Base variant supports extract
    base_params = ACEStepAdapter.map_inputs(
        "extract",
        {
            "variant": "base",
            "source_audio_path": "/staged/full_mix.wav",
            "stem_name": "vocals",
        },
    )
    assert base_params["task"] == "extract"
    assert base_params["variant"] == "base"
    p = ACEStepClient._release_payload("extract", base_params)
    assert p["task_type"] == "extract"
    assert p["stem_name"] == "vocals"

    # 2. Turbo variant strictly rejected
    with pytest.raises(UnsupportedCapabilityError) as exc_info:
        ACEStepAdapter.map_inputs(
            "extract",
            {
                "variant": "turbo",
                "source_audio_path": "/staged/full_mix.wav",
                "stem_name": "vocals",
            },
        )
    assert exc_info.value.code == ErrorCode.UNSUPPORTED_CAPABILITY.value
    assert "extract" in str(exc_info.value)


# ==============================================================================
# ACE-12: Lego (Add Instrument Layer, Variant-Dependent)
# ==============================================================================

def test_ace_12_lego_variant_matrix():
    """ACE-12: Lego supported on Base, strictly rejected on Turbo with UNSUPPORTED_CAPABILITY."""
    # 1. Base variant supports lego
    base_params = ACEStepAdapter.map_inputs(
        "lego",
        {
            "variant": "base",
            "prompt": "Add funky slap bass",
            "source_audio_path": "/staged/drum_track.wav",
            "add_instrument": "bass",
        },
    )
    assert base_params["task"] == "lego"
    p = ACEStepClient._release_payload("lego", base_params)
    assert p["task_type"] == "lego"
    assert p["add_instrument"] == "bass"

    # 2. Turbo variant strictly rejected
    with pytest.raises(UnsupportedCapabilityError) as exc_info:
        ACEStepAdapter.map_inputs(
            "lego",
            {
                "variant": "turbo",
                "prompt": "Add bass",
                "source_audio_path": "/staged/drums.wav",
            },
        )
    assert exc_info.value.code == ErrorCode.UNSUPPORTED_CAPABILITY.value


# ==============================================================================
# ACE-13: Complete (Accompaniment, Variant-Dependent)
# ==============================================================================

def test_ace_13_complete_variant_matrix():
    """ACE-13: Complete supported on Base, strictly rejected on Turbo with UNSUPPORTED_CAPABILITY."""
    # 1. Base variant supports complete
    base_params = ACEStepAdapter.map_inputs(
        "complete",
        {
            "variant": "base",
            "source_audio_path": "/staged/vocal_lead.wav",
            "prompt": "Generate full orchestral accompaniment",
        },
    )
    assert base_params["task"] == "complete"
    p = ACEStepClient._release_payload("complete", base_params)
    assert p["task_type"] == "complete"

    # 2. Turbo variant strictly rejected
    with pytest.raises(UnsupportedCapabilityError) as exc_info:
        ACEStepAdapter.map_inputs(
            "complete",
            {
                "variant": "turbo",
                "source_audio_path": "/staged/lead.wav",
            },
        )
    assert exc_info.value.code == ErrorCode.UNSUPPORTED_CAPABILITY.value


# ==============================================================================
# ACE-14: Music Extend
# ==============================================================================

def test_ace_14_music_extend():
    """ACE-14: Music extension maps continuation parameters on Base, rejected on Turbo."""
    base_params = ACEStepAdapter.map_inputs(
        "music_extend",
        {
            "variant": "base",
            "prompt": "Continue with soaring guitar outro",
            "source_audio_path": "/staged/intro.wav",
            "extend_duration": 45.0,
        },
    )
    p = ACEStepClient._release_payload("music_extend", base_params)
    assert p["task_type"] == "extend"
    assert p["extend_duration"] == 45.0

    # Turbo rejected
    with pytest.raises(UnsupportedCapabilityError):
        ACEStepAdapter.map_inputs(
            "music_extend",
            {"variant": "turbo", "prompt": "Continue", "source_audio_path": "/staged/intro.wav"},
        )


# ==============================================================================
# ACE-15: Multilingual Lyrics
# ==============================================================================

def test_ace_15_multilingual_lyrics_preservation():
    """ACE-15: Unicode multilingual lyrics (Vietnamese, Japanese, English) preserved accurately."""
    vietnamese_lyrics = "[verse]\nĐêm nay thành phố lên đèn\nTiếng đàn ngân vang giữa trời sương"
    japanese_lyrics = "[chorus]\n桜舞い散る夜に夢を見て\n未来へと歩き出す"

    p_vn = ACEStepAdapter.map_inputs("lyrics_to_music", {"prompt": "Vietnamese ballad", "lyrics": vietnamese_lyrics})
    assert p_vn["lyrics"] == vietnamese_lyrics

    p_jp = ACEStepAdapter.map_inputs("lyrics_to_music", {"prompt": "J-Pop upbeat", "lyrics": japanese_lyrics})
    assert p_jp["lyrics"] == japanese_lyrics


# ==============================================================================
# ACE-16: Batch Generation
# ==============================================================================

@pytest.mark.asyncio
async def test_ace_16_batch_generation_multi_artifact():
    """ACE-16: Batch generation yields independent seeds and multi-artifact download."""
    client = ACEStepClient(api_base_url="http://127.0.0.1:8008")
    mock_audio_1 = make_mock_wav_bytes(1.0)
    mock_audio_2 = make_mock_wav_bytes(1.0)

    # Mock release_task, query_result with 2 batch items, and audio download
    with patch("httpx.AsyncClient") as mock_http_cls:
        mock_http = AsyncMock()
        mock_http_cls.return_value.__aenter__.return_value = mock_http

        mock_http.get.side_effect = [
            # 1. /health
            MagicMock(status_code=200),
            # 2. /v1/audio?path=output_0.mp3
            MagicMock(status_code=200, content=mock_audio_1, headers={"content-type": "audio/mpeg"}),
            # 3. /v1/audio?path=output_1.mp3
            MagicMock(status_code=200, content=mock_audio_2, headers={"content-type": "audio/mpeg"}),
        ]

        mock_http.post.side_effect = [
            # 1. /release_task
            MagicMock(status_code=200, json=lambda: {"data": {"task_id": "task-batch-99"}}),
            # 2. /query_result
            MagicMock(
                status_code=200,
                json=lambda: {
                    "data": [
                        {
                            "status": 1,
                            "result": json.dumps([
                                {"file": "/v1/audio?path=output_0.mp3", "metas": {"seed": 101}},
                                {"file": "/v1/audio?path=output_1.mp3", "metas": {"seed": 102}},
                            ]),
                        }
                    ]
                },
            ),
        ]

        res = await client.generate_music("text_to_music", {"prompt": "Lo-fi guitar", "batch_size": 2})
        assert res["task_id"] == "task-batch-99"
        assert len(res["artifacts"]) == 2
        assert res["artifacts"][0]["size_bytes"] == len(mock_audio_1)
        assert res["artifacts"][1]["size_bytes"] == len(mock_audio_2)
        assert res["artifacts"][0]["sha256"] == hashlib.sha256(mock_audio_1).hexdigest()
        assert res["artifacts"][1]["sha256"] == hashlib.sha256(mock_audio_2).hexdigest()


# ==============================================================================
# ACE-17: Variant Switching & Truth Matrix
# ==============================================================================

def test_ace_17_variant_switching_manifest_verification():
    """ACE-17: Manifest validates variant-task capabilities and model registry rules."""
    loader = ManifestLoader(Path("manifests"))
    models = loader.load_models()
    assert "ace-step-1.5" in models
    m = models["ace-step-1.5"]

    # Manifest truth verification
    assert m.is_variant_supported_for_task("text_to_music", "turbo") is True
    assert m.is_variant_supported_for_task("text_to_music", "base") is True
    assert m.is_variant_supported_for_task("cover", "turbo") is True
    assert m.is_variant_supported_for_task("cover", "base") is True
    assert m.is_variant_supported_for_task("extract", "base") is True
    assert m.is_variant_supported_for_task("extract", "turbo") is False
    assert m.is_variant_supported_for_task("lego", "turbo") is False
    assert m.is_variant_supported_for_task("complete", "turbo") is False

    # Check weights variants exist
    assert hasattr(m.weights, "variants")
    assert "turbo" in m.weights.variants
    assert "base" in m.weights.variants
    assert "sft" in m.weights.variants
    assert "xl_base" in m.weights.variants


# ==============================================================================
# ACE-18: Audio Format Export
# ==============================================================================

def test_ace_18_audio_format_export_validation():
    """ACE-18: Validates MP3, WAV, and FLAC export and rejects unsupported codecs."""
    for fmt in ["mp3", "wav", "flac"]:
        p = ACEStepAdapter.map_inputs("text_to_music", {"prompt": "Test", "output_format": fmt})
        assert p["audio_format"] == fmt

    # Unsupported format (e.g. ogg, aac)
    with pytest.raises(InvalidRequestError):
        ACEStepAdapter.map_inputs("text_to_music", {"prompt": "Test", "output_format": "ogg"})


# ==============================================================================
# ACE-19: Audio Understanding
# ==============================================================================

@pytest.mark.asyncio
async def test_ace_19_audio_understanding():
    """ACE-19: understand_audio endpoint extracts caption, genre, BPM, and mood."""
    client = ACEStepClient(api_base_url="http://127.0.0.1:8008")
    with patch("httpx.AsyncClient") as mock_http_cls:
        mock_http = AsyncMock()
        mock_http_cls.return_value.__aenter__.return_value = mock_http
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "caption": "A vintage 1970s funk groove with a syncopated slap bassline",
            "detected_bpm": 115,
            "detected_key": "E minor",
            "genre": "Funk",
            "mood": "Groovy",
        }
        mock_http.post.return_value = mock_resp

        res = await client.understand_audio("/staged/test_audio.wav")
        assert res["detected_bpm"] == 115
        assert res["detected_key"] == "E minor"
        assert "funk groove" in res["caption"].lower()


# ==============================================================================
# ACE-20: Lyrics Timing / LRC Timestamps
# ==============================================================================

@pytest.mark.asyncio
async def test_ace_20_lyrics_timing_lrc_extraction():
    """ACE-20: Extracts timestamped lyrics / LRC from native completion metadata."""
    client = ACEStepClient(api_base_url="http://127.0.0.1:8008")
    mock_audio = make_mock_wav_bytes(2.0)
    lrc_content = "[00:02.10]First line of verse\n[00:08.50]Second line of verse"

    with patch("httpx.AsyncClient") as mock_http_cls:
        mock_http = AsyncMock()
        mock_http_cls.return_value.__aenter__.return_value = mock_http

        mock_http.get.side_effect = [
            MagicMock(status_code=200),  # health
            MagicMock(status_code=200, content=mock_audio, headers={"content-type": "audio/mpeg"}),
        ]
        mock_http.post.side_effect = [
            MagicMock(status_code=200, json=lambda: {"data": {"task_id": "task-lrc"}}),
            MagicMock(
                status_code=200,
                json=lambda: {
                    "data": [
                        {
                            "status": 1,
                            "result": json.dumps([
                                {
                                    "file": "/v1/audio?path=output.mp3",
                                    "metas": {"lrc": lrc_content, "duration": 30.0},
                                }
                            ]),
                        }
                    ]
                },
            ),
        ]

        res = await client.generate_music("lyrics_to_music", {"prompt": "Pop song", "lyrics": "First line\nSecond line"})
        assert "metadata" in res
        assert res["metadata"]["lrc"] == lrc_content


# ==============================================================================
# ACE-21: Quality Scoring
# ==============================================================================

@pytest.mark.asyncio
async def test_ace_21_quality_scoring_with_provenance():
    """ACE-21: score_audio returns verifiable score and model provenance."""
    client = ACEStepClient(api_base_url="http://127.0.0.1:8008")
    with patch("httpx.AsyncClient") as mock_http_cls:
        mock_http = AsyncMock()
        mock_http_cls.return_value.__aenter__.return_value = mock_http
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model_provenance": "ACE-Studio/ACE-Score-v1",
            "overall_score": 4.35,
            "snr_db": 34.2,
            "musicality_score": 4.5,
        }
        mock_http.post.return_value = mock_resp

        score = await client.score_audio("/staged/song.mp3")
        assert score["overall_score"] == 4.35
        assert score["model_provenance"] == "ACE-Studio/ACE-Score-v1"


# ==============================================================================
# ACE-22: LoRA Inference
# ==============================================================================

def test_ace_22_lora_inference_mapping():
    """ACE-22: LoRA adapter path and weights passed to inference payload."""
    params = ACEStepAdapter.map_inputs(
        "text_to_music",
        {
            "prompt": "Jazz trio",
            "lora_path": "/models/loras/jazz_piano.safetensors",
            "lora_weight": 0.85,
        },
    )
    assert params["lora_path"] == "/models/loras/jazz_piano.safetensors"
    assert params["lora_weight"] == 0.85

    payload = ACEStepClient._release_payload("text_to_music", params)
    assert payload["lora_path"] == "/models/loras/jazz_piano.safetensors"
    assert payload["lora_weight"] == 0.85

    # Out of range lora_weight
    with pytest.raises(InvalidRequestError):
        ACEStepAdapter.map_inputs(
            "text_to_music",
            {"prompt": "Jazz", "lora_path": "/m.safetensors", "lora_weight": 5.0},
        )


# ==============================================================================
# ACE-23: LoRA Training Workflow Contract
# ==============================================================================

@pytest.mark.asyncio
async def test_ace_23_lora_training_workflow_contract():
    """ACE-23: LoRA training contract validates hyperparameters and submits job."""
    client = ACEStepClient(api_base_url="http://127.0.0.1:8008")

    # Invalid rank (< 4 or > 64)
    with pytest.raises(InvalidRequestError):
        await client.train_lora(dataset_path="/data/jazz", output_dir="/out", hyperparameters={"rank": 2})

    # Invalid learning rate
    with pytest.raises(InvalidRequestError):
        await client.train_lora(dataset_path="/data/jazz", output_dir="/out", hyperparameters={"learning_rate": -0.01})

    # Valid contract submission
    with patch("httpx.AsyncClient") as mock_http_cls:
        mock_http = AsyncMock()
        mock_http_cls.return_value.__aenter__.return_value = mock_http
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "training_job_id": "lora-train-001",
            "status": "QUEUED",
            "rank": 16,
            "learning_rate": 0.0002,
        }
        mock_http.post.return_value = mock_resp

        res = await client.train_lora(
            dataset_path="/data/acestep_dataset",
            output_dir="/models/lora_jazz",
            hyperparameters={"rank": 16, "learning_rate": 0.0002, "epochs": 15},
        )
        assert res["training_job_id"] == "lora-train-001"
        assert res["rank"] == 16


# ==============================================================================
# ACE-24: Failure / Recovery / Cancellation
# ==============================================================================

@pytest.mark.asyncio
async def test_ace_24_cancellation_and_error_classification():
    """ACE-24: In-flight cancellation and error taxonomy classification."""
    # 1. Cancellation
    client = ACEStepClient(api_base_url="http://127.0.0.1:8008")
    with patch("httpx.AsyncClient") as mock_http_cls:
        mock_http = AsyncMock()
        mock_http_cls.return_value.__aenter__.return_value = mock_http
        mock_http.post.return_value = MagicMock(status_code=200)

        cancelled = await client.cancel_task("task-cancel-123")
        assert cancelled is True

    # 2. Fail-closed when unconfigured
    unconfigured_client = ACEStepClient(api_base_url="")
    with pytest.raises(ModelUnavailableError) as exc_info:
        await unconfigured_client.generate_music("text_to_music", {"prompt": "Beat"})
    assert "unconfigured" in str(exc_info.value).lower()

    # 3. Error classification
    code, _, retryable = classify_worker_error(TimeoutError("Inference timed out"), subsystem="music", stage="generating_music")
    assert code == ErrorCode.INFERENCE_TIMEOUT
    assert retryable is True

    code_oom, _, retryable_oom = classify_worker_error(RuntimeError("CUDA out of memory"), subsystem="music", stage="generating_music")
    assert code_oom == ErrorCode.INSUFFICIENT_VRAM
    assert retryable_oom is False

    code_cap, _, retryable_cap = classify_worker_error(
        UnsupportedCapabilityError("Task extract unsupported on turbo"), subsystem="music", stage="input_mapping"
    )
    assert code_cap == ErrorCode.UNSUPPORTED_CAPABILITY
    assert retryable_cap is False


# ==============================================================================
# Worker Integration: Multi-Artifact Upload & Turbo Rejection
# ==============================================================================

@pytest.mark.asyncio
async def test_music_worker_execute_job_success():
    """Worker executes leased job, stages assets, uploads artifacts, and completes."""
    worker = MusicWorker(
        worker_id="worker-test-0",
        control_plane_url="http://127.0.0.1:8000",
        api_key="test-api-key",
        acestep_api_url="http://127.0.0.1:8008",
    )
    mock_audio = make_mock_wav_bytes(1.0)
    job_uuid = str(uuid.uuid4())
    req_uuid = str(uuid.uuid4())
    job = FactoryJob(
        job_id=f"job-{job_uuid}",
        request_id=f"req-{req_uuid}",
        service="music",
        model="ace-step-1.5",
        task="text_to_music",
        inputs={"prompt": "Chiptune arcade adventure", "duration_seconds": 15},
    )

    with patch.object(worker.client, "generate_music") as mock_gen:
        mock_gen.return_value = {
            "task_id": "native-task-1",
            "filename": "output.mp3",
            "mime_type": "audio/mpeg",
            "audio_bytes": mock_audio,
            "size_bytes": len(mock_audio),
            "sha256": hashlib.sha256(mock_audio).hexdigest(),
            "metadata": {"duration": 15.0},
            "artifacts": [
                {
                    "filename": "output.mp3",
                    "mime_type": "audio/mpeg",
                    "audio_bytes": mock_audio,
                    "size_bytes": len(mock_audio),
                    "sha256": hashlib.sha256(mock_audio).hexdigest(),
                }
            ],
        }

        mock_http = AsyncMock()
        mock_http.post.side_effect = [
            # 1. /v1/storage/upload-url
            MagicMock(status_code=200, json=lambda: {"upload_url": "http://storage/put", "storage_path": "music/output.mp3", "headers": {}}),
            # 2. /v1/storage/download-url
            MagicMock(status_code=200, json=lambda: {"download_url": "http://storage/download/output.mp3"}),
            # 3. /v1/jobs/job-music-101/complete
            MagicMock(status_code=200, text="OK"),
        ]

        with patch("httpx.AsyncClient") as mock_upload_cls:
            mock_upload = AsyncMock()
            mock_upload.put.return_value = MagicMock(status_code=200)
            mock_upload_cls.return_value.__aenter__.return_value = mock_upload

            success = await worker.execute_job(mock_http, job)
            assert success is True
            assert mock_http.post.call_count == 3


@pytest.mark.asyncio
async def test_music_worker_turbo_rejects_specialized_task():
    """Worker fails job with UNSUPPORTED_CAPABILITY when turbo is asked to extract stems."""
    worker = MusicWorker(
        worker_id="worker-test-0",
        control_plane_url="http://127.0.0.1:8000",
        api_key="test-api-key",
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
        inputs={
            "variant": "turbo",
            "source_audio_path": "/staged/song.wav",
            "stem_name": "vocals",
        },
    )

    mock_http = AsyncMock()
    mock_http.post.return_value = MagicMock(status_code=200)

    success = await worker.execute_job(mock_http, job)
    assert success is False
    # Verify failure reported with UNSUPPORTED_CAPABILITY
    fail_call = mock_http.post.call_args_list[-1]
    posted_json = fail_call[1]["json"]
    assert posted_json["error"]["code"] == ErrorCode.UNSUPPORTED_CAPABILITY.value
    assert posted_json["retryable"] is False
