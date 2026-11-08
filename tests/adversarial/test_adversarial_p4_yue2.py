"""
tests/adversarial/test_adversarial_p4_yue2.py
Adversarial Challenger Test Suite for Phase 4 (YuE2 Full-Song Generation Subsystem):
1. ABC Notation Security & Delimiter Imbalance (score.py):
   - Malformed headers, missing key/meter, unclosed brackets/quotes/slurs, injected control chars, diff conflicts.
2. Numerical Poisoning & Parameter Fuzzing (adapter.py):
   - IEEE 754 NaN/Inf across all float fields (duration, temperature, top_p), boundary duration, seed bounds, task requirements.
3. Client Error Sanitization & Resilience (client.py):
   - Whitespace-only base URLs, missing protocols, connection refused / timeout / HTTP error mapping to ModelUnavailableError(code="BACKEND_UNAVAILABLE").
4. MusicWorker Multi-Model Routing & Asset Safety (worker.py):
   - Multi-model routing (yue2 vs ace-step-1.5), preservation of caller file:// and local assets on success and failure (VULN-P3-01),
   - SSRF pre-flight validation blocking private/metadata IPs (VULN-P3-02), and multi-artifact persistence (audio + score).
5. Gate Y0 Feasibility & Honesty (runtime.py):
   - Honest reporting of HARDWARE_BLOCKED when CUDA is absent or VRAM < 12GB without dummy simulated PASS,
   - VRAM threshold boundaries, and manifest capabilities truth integrity.
"""
import asyncio
import base64
import hashlib
import json
import math
import os
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
)
from factory_contracts.jobs.job_contract import FactoryJob
from services.music.yue2.adapter import (
    ALL_SUPPORTED_TASKS,
    MAX_DURATION_SECONDS,
    VALID_VARIANTS,
    YuE2Adapter,
)
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


def make_dummy_wav_bytes(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Constructs valid WAV header and PCM audio bytes for mock tests."""
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


# ============================================================================
# 1. ABC Notation Security & Delimiter Imbalance Probes (score.py)
# ============================================================================

class TestABCNotationSecurityAndAdversarialProbes:
    """Adversarial stress-testing of ABC score parser, validator, chord extractor, and diff/edit engine."""

    @pytest.mark.parametrize("empty_score", [
        "",
        "   ",
        "\t\n  \r\n",
        None,
    ])
    def test_reject_empty_or_non_string_score(self, empty_score):
        """Empty or non-string inputs must raise InvalidRequestError."""
        with pytest.raises(InvalidRequestError, match=r"(?i)cannot be empty|non-string"):
            validate_abc_score(empty_score)  # type: ignore

    @pytest.mark.parametrize("missing_key_score", [
        "X:1\nT:No Key\nM:4/4\nL:1/8\nC D E F|]",
        "X:1\nT:Empty Key\nK:\nC D E F|]",
        "X:1\nT:Whitespace Key\nK:   \nC D E F|]",
        "T:Just Title\nC D E F|]",
    ])
    def test_reject_missing_or_blank_key_header(self, missing_key_score):
        """Scores without a mandatory K: (Key) header must be rejected."""
        with pytest.raises(InvalidRequestError, match=r"(?i)must include a valid Key header"):
            validate_abc_score(missing_key_score)

    @pytest.mark.parametrize("invalid_key", [
        "H",
        "Zminor",
        "C#m99",
        "K",
        "123",
        "C/G/Am",
        "Cb major clef=invalid_clef_name_999",
        "C\x00",
    ])
    def test_reject_malformed_key_header_format(self, invalid_key):
        """Keys outside standard diatonic / modal / accidental notation must be rejected."""
        score = f"X:1\nT:Bad Key\nK:{invalid_key}\nC D E F|]"
        with pytest.raises(InvalidRequestError, match=r"(?i)Invalid key header format"):
            validate_abc_score(score)

    def test_key_regex_rejects_embedded_newlines(self):
        """KEY_REGEX matches must not permit embedded newlines."""
        from services.music.yue2.score import KEY_REGEX
        assert KEY_REGEX.match("C\nmajor") is None

    @pytest.mark.parametrize("valid_key", [
        "C", "G", "D", "A", "E", "B", "F#", "C#",
        "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb",
        "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "A#m",
        "Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm", "Abm",
        "Cmajor", "Aminor", "Ddor", "Gmix", "Ephr", "Flyd", "Bloc",
        "C clef=treble", "C clef=bass", "C bass", "C treble", "C alto",
    ])
    def test_accept_valid_key_header_variants(self, valid_key):
        """Standard keys, modes, and clef annotations accepted by score.py."""
        score = f"X:1\nT:Key Test\nK:{valid_key}\nC D E F|]"
        info = validate_abc_score(score)
        assert info["valid"] is True
        assert info["headers"]["K"] == valid_key

    @pytest.mark.parametrize("invalid_meter", [
        "3/5", "17/4", "0/4", "4/0", "99/8", "random", "4/4/4", "M:4", "4-4", "4/4\x00",
    ])
    def test_reject_malformed_meter_header_format(self, invalid_meter):
        """Invalid time signatures must be rejected."""
        score = f"X:1\nM:{invalid_meter}\nK:C\nC D E F|]"
        with pytest.raises(InvalidRequestError, match=r"(?i)Invalid meter header format"):
            validate_abc_score(score)

    @pytest.mark.parametrize("valid_meter", [
        "4/4", "3/4", "2/4", "6/8", "9/8", "12/8", "2/2", "3/8", "5/8", "7/8", "C", "C|", "none",
    ])
    def test_accept_valid_meter_header_variants(self, valid_meter):
        """Standard meters (numeric, Common, Cut Common, none) must be accepted."""
        score = f"X:1\nM:{valid_meter}\nK:C\nC D E F|]"
        info = validate_abc_score(score)
        assert info["valid"] is True
        assert info["headers"]["M"] == valid_meter

    @pytest.mark.parametrize("invalid_tempo", [
        "allegro", "fast", "-120", "text_without_number", "Q:120", "120/120/120", "120\x00",
    ])
    def test_reject_malformed_tempo_header_format(self, invalid_tempo):
        """Invalid tempo specifications must be rejected."""
        score = f"X:1\nQ:{invalid_tempo}\nK:C\nC D E F|]"
        with pytest.raises(InvalidRequestError, match=r"(?i)Invalid tempo header format"):
            validate_abc_score(score)

    @pytest.mark.parametrize("valid_tempo", [
        "120", "96", "144.5", "1/4=120", "3/8=96", '"Allegro" 120', '"Moderato" 100',
    ])
    def test_accept_valid_tempo_header_variants(self, valid_tempo):
        """Standard tempos (numeric, note-assignment, quoted descriptor) must be accepted."""
        score = f"X:1\nQ:{valid_tempo}\nK:C\nC D E F|]"
        info = validate_abc_score(score)
        assert info["valid"] is True
        assert info["headers"]["Q"] == valid_tempo

    @pytest.mark.parametrize("invalid_note_length", [
        "1/3", "2/4", "1/5", "1/128", "1/0", "0/8", "1", "quarter", "1/8\x00",
    ])
    def test_reject_malformed_note_length_header_format(self, invalid_note_length):
        """Invalid unit note lengths outside standard powers of 2 (1/1 to 1/64) must be rejected."""
        score = f"X:1\nL:{invalid_note_length}\nK:C\nC D E F|]"
        with pytest.raises(InvalidRequestError, match=r"(?i)Invalid unit note length header format"):
            validate_abc_score(score)

    @pytest.mark.parametrize("unbalanced_score,expected_fragment", [
        ('X:1\nK:C\n"C C2 E2 G2 c2 |]', "unclosed quotes"),
        ('X:1\nK:C\n"C" C2 "Am E2 G2 c2 |]', "unclosed quotes"),
        ('X:1\nK:C\n[C E G c2 |]', "unclosed delimiter '['"),
        ('X:1\nK:C\nC E G ] c2 |]', "unmatched closing bracket ']'"),
        ('X:1\nK:C\n{C E G c2 |]', "unclosed delimiter '{'"),
        ('X:1\nK:C\nC E G } c2 |]', "unmatched closing brace '}'"),
        ('X:1\nK:C\n(C E G c2 |]', "unclosed delimiter '('"),
        ('X:1\nK:C\nC E G ) c2 |]', "unmatched closing paren ')'"),
        ('X:1\nK:C\n[{]} |]', "unmatched closing bracket ']'"),
        ('X:1\nK:C\n([)] |]', "unmatched closing bracket ']'"),
        ('X:1\nK:C\n{[(])} |]', "unmatched closing bracket ']'"),
    ])
    def test_reject_unbalanced_delimiters(self, unbalanced_score, expected_fragment):
        """Unbalanced or cross-nested brackets, quotes, braces, and slurs must raise InvalidRequestError."""
        with pytest.raises(InvalidRequestError, match=r"(?i)Malformed ABC score"):
            validate_abc_score(unbalanced_score)

    def test_accept_complex_balanced_and_escaped_delimiters(self):
        """Bar lines ([|, |], [:, :|]), chords ([]), tuplets ((3), and nested slurs must be accepted."""
        score = """X:1
T:Complex Delimiters Test
M:4/4
L:1/8
Q:120
K:C
[| "C" [CEG]2 (3cde (c2 d2) | "G" {ga}g4 |: "Am" (a2 b2) :| [] |]"""
        info = validate_abc_score(score)
        assert info["valid"] is True
        assert len(info["chords"]) >= 2

    @pytest.mark.parametrize("score_without_notes", [
        "X:1\nT:No Notes\nK:C\n| | | |]",
        "X:1\nT:Only Comments\nK:C\n% Just a comment\n% Another comment",
        "X:1\nT:Non Notes Text\nK:C\n1234 5678 90 !@#$%",
    ])
    def test_reject_score_without_notes_or_rests_in_body(self, score_without_notes):
        """Scores containing no notes or rests (A-G, a-g, z, x) must raise InvalidRequestError."""
        with pytest.raises(InvalidRequestError, match=r"(?i)must contain musical notes or rests"):
            validate_abc_score(score_without_notes)

    def test_extract_headers_ignores_comments_and_blank_lines(self):
        """extract_headers extracts valid headers while cleanly ignoring inline/block comments and blanks."""
        score = """
% Top level comment
X:101
T:Song of the Prairie
% Intermediate comment
C:Traditional
M:3/4
% Another comment
L:1/4
Q:90
K:G
% Body comment
"G" G B d | "D7" c A F | "G" G3 |]
"""
        headers = extract_headers(score)
        assert headers["X"] == "101"
        assert headers["T"] == "Song of the Prairie"
        assert headers["C"] == "Traditional"
        assert headers["M"] == "3/4"
        assert headers["L"] == "1/4"
        assert headers["Q"] == "90"
        assert headers["K"] == "G"
        assert "%" not in headers

    def test_extract_chords_handles_standard_jazz_and_inversions(self):
        """extract_chords correctly parses chord symbols supported by CHORD_REGEX."""
        score = """X:1
T:Jazz Chords
M:4/4
K:C
"Cmaj7" c4 | "Am7/G" a4 | "F#dim7" f4 | "C/E" c4 |]"""
        chords = extract_chords(score)
        assert chords == ["Cmaj7", "Am7/G", "F#dim7", "C/E"]

    def test_diff_scores_rejects_invalid_inputs(self):
        """diff_scores validates both inputs and raises InvalidRequestError if either is malformed."""
        valid_score = generate_sample_abc(key="C")
        invalid_score = "X:1\nT:Broken\nK:InvalidKey\nC D E |]"

        with pytest.raises(InvalidRequestError):
            diff_scores(invalid_score, valid_score)

        with pytest.raises(InvalidRequestError):
            diff_scores(valid_score, invalid_score)

    def test_diff_scores_tracks_header_and_chord_changes(self):
        """diff_scores accurately identifies changed headers, added/removed chords, and measure delta."""
        score_a = """X:1
T:Version A
M:4/4
Q:100
K:C
"C" C2 E2 G2 c2 | "G" G4 |]"""

        score_b = """X:1
T:Version B
M:3/4
Q:120
K:G
"C" C2 E2 G2 | "D7" D4 | "G" G4 |]"""

        diff = diff_scores(score_a, score_b)
        header_changes = diff["header_changes"]
        assert header_changes["T"] == {"from": "Version A", "to": "Version B"}
        assert header_changes["M"] == {"from": "4/4", "to": "3/4"}
        assert header_changes["Q"] == {"from": "100", "to": "120"}
        assert header_changes["K"] == {"from": "C", "to": "G"}

        chord_diff = diff["chord_diff"]
        assert chord_diff["added"] == ["D7"]
        assert chord_diff["removed"] == []
        assert set(chord_diff["preserved"]) == {"C", "G"}
        assert diff["measures_delta"] == 1

    def test_edit_score_validates_and_rejects_invalid_edits(self):
        """edit_score enforces validity of both initial score and resulting edited score."""
        valid_score = generate_sample_abc()

        # 1. Invalid key edit
        with pytest.raises(InvalidRequestError, match=r"(?i)Invalid key header format"):
            edit_score(valid_score, {"key": "InvalidKeyFormat"})

        # 2. Invalid meter edit
        with pytest.raises(InvalidRequestError, match=r"(?i)Invalid meter header format"):
            edit_score(valid_score, {"meter": "17/5"})

        # 3. Invalid body replacement with unclosed bracket
        with pytest.raises(InvalidRequestError, match=r"(?i)Malformed ABC score"):
            edit_score(valid_score, {"body": "[C E G |]"})

    def test_edit_score_successful_chord_replacement_and_header_updates(self):
        """edit_score applies header updates and chord substitutions while producing valid ABC output."""
        base_score = generate_sample_abc(title="Base Melody", key="C", meter="4/4", tempo=100)
        edits = {
            "title": "Remixed Anthem",
            "tempo": 132,
            "key": "Am",
            "replace_chords": {"C": "Cmaj7", "G": "G7"},
        }
        result = edit_score(base_score, edits)
        info = validate_abc_score(result)
        assert info["valid"] is True
        assert info["headers"]["T"] == "Remixed Anthem"
        assert info["headers"]["Q"] == "132"
        assert info["headers"]["K"] == "Am"
        assert "Cmaj7" in info["chords"]
        assert "G7" in info["chords"]


# ============================================================================
# 2. Numerical Poisoning & Parameter Fuzzing Probes (adapter.py)
# ============================================================================

class TestYuE2AdapterNumericalPoisoningAndValidation:
    """Stress tests parameter validation against NaN/Inf poisoning, boundary breaches, and invalid types."""

    @pytest.mark.parametrize("bad_duration", [
        float("nan"),
        float("inf"),
        float("-inf"),
        0.0,
        -0.1,
        -100.0,
        300.001,
        301.0,
        999999.0,
        "not_a_number",
        [30.0],
        {"val": 30.0},
    ])
    def test_reject_nan_inf_and_boundary_durations(self, bad_duration):
        """Duration must reject NaN, Inf, non-numbers, <= 0.0, and > 300.0."""
        with pytest.raises(InvalidRequestError, match=r"(?i)audio_duration"):
            YuE2Adapter.validate_duration(bad_duration)

        with pytest.raises(InvalidRequestError, match=r"(?i)audio_duration"):
            YuE2Adapter.map_inputs("text_to_music", {"prompt": "Valid", "duration": bad_duration})

        with pytest.raises(InvalidRequestError, match=r"(?i)audio_duration"):
            YuE2Adapter.map_inputs("text_to_music", {"prompt": "Valid", "audio_duration": bad_duration})

    @pytest.mark.parametrize("valid_duration", [0.1, 1.0, 15.5, 30.0, 120.0, 300.0])
    def test_accept_valid_duration_boundaries(self, valid_duration):
        """Durations in range (0.0, 300.0] must be cleanly parsed as floats."""
        assert YuE2Adapter.validate_duration(valid_duration) == valid_duration

    @pytest.mark.parametrize("bad_seed", [
        float("nan"),
        float("inf"),
        float("-inf"),
        -1,
        -100,
        2**32,
        2**32 + 1,
        2**64,
        "invalid_seed",
        [1234],
    ])
    def test_reject_nan_inf_and_out_of_bounds_seeds(self, bad_seed):
        """Seed must reject NaN, Inf, negative integers, and values > 2^32 - 1."""
        with pytest.raises(InvalidRequestError, match=r"(?i)seed"):
            YuE2Adapter.validate_seed(bad_seed)

        with pytest.raises(InvalidRequestError, match=r"(?i)seed"):
            YuE2Adapter.map_inputs("text_to_music", {"prompt": "Valid", "seed": bad_seed})

    @pytest.mark.parametrize("valid_seed", [0, 1, 42, 123456789, 2**32 - 1])
    def test_accept_valid_seed_boundaries(self, valid_seed):
        """Seeds in range [0, 2^32 - 1] must be accepted."""
        assert YuE2Adapter.validate_seed(valid_seed) == valid_seed

    @pytest.mark.parametrize("bad_temp", [
        float("nan"),
        float("inf"),
        float("-inf"),
        0.0,
        -0.5,
        2.001,
        5.0,
        "fast",
    ])
    def test_reject_sampling_config_temperature_poisoning(self, bad_temp):
        """sampling_config.temperature must reject NaN, Inf, <= 0.0, and > 2.0."""
        with pytest.raises(InvalidRequestError, match=r"(?i)sampling_config\.temperature"):
            YuE2Adapter.validate_sampling_config({"temperature": bad_temp})

    @pytest.mark.parametrize("bad_top_p", [
        float("nan"),
        float("inf"),
        float("-inf"),
        0.0,
        -0.1,
        1.001,
        2.0,
        "high",
    ])
    def test_reject_sampling_config_top_p_poisoning(self, bad_top_p):
        """sampling_config.top_p must reject NaN, Inf, <= 0.0, and > 1.0."""
        with pytest.raises(InvalidRequestError, match=r"(?i)sampling_config\.top_p"):
            YuE2Adapter.validate_sampling_config({"top_p": bad_top_p})

    @pytest.mark.parametrize("bad_top_k", [-1, -100, "fast", [50]])
    def test_reject_sampling_config_top_k_poisoning(self, bad_top_k):
        """sampling_config.top_k must reject negative values and non-integers."""
        with pytest.raises(InvalidRequestError, match=r"(?i)sampling_config\.top_k"):
            YuE2Adapter.validate_sampling_config({"top_k": bad_top_k})

    @pytest.mark.parametrize("invalid_config", ["default", [1, 2, 3], 42])
    def test_reject_non_dict_sampling_config(self, invalid_config):
        """sampling_config must be a dictionary."""
        with pytest.raises(InvalidRequestError, match=r"(?i)sampling_config must be a dictionary"):
            YuE2Adapter.validate_sampling_config(invalid_config)

    @pytest.mark.parametrize("unsupported_task", [
        "cover", "repaint", "extract", "lego", "complete", "music_extend", "audio_to_audio", "unknown_music_task",
    ])
    def test_reject_unsupported_tasks_with_contract_exception(self, unsupported_task):
        """Tasks outside YuE2 capabilities must raise UnsupportedCapabilityError."""
        with pytest.raises(UnsupportedCapabilityError, match=r"(?i)does not support task"):
            YuE2Adapter.validate_variant_capability(unsupported_task, "standard")

        with pytest.raises(UnsupportedCapabilityError, match=r"(?i)does not support task"):
            YuE2Adapter.map_inputs(unsupported_task, {"prompt": "Valid"})

    @pytest.mark.parametrize("unsupported_variant", [
        "turbo", "sft", "2b_turbo", "large", "invalid_variant", "gpt4",
    ])
    def test_reject_unsupported_variants(self, unsupported_variant):
        """Variants outside VALID_VARIANTS must raise UnsupportedCapabilityError."""
        with pytest.raises(UnsupportedCapabilityError, match=r"(?i)variant .* is not recognized"):
            YuE2Adapter.validate_variant_capability("text_to_music", unsupported_variant)

        with pytest.raises(UnsupportedCapabilityError, match=r"(?i)variant .* is not recognized"):
            YuE2Adapter.map_inputs("text_to_music", {"prompt": "Valid", "variant": unsupported_variant})

    def test_enforce_task_mandatory_inputs(self):
        """Enforces required fields for each specific YuE2 generative task."""
        # text_to_music requires prompt
        with pytest.raises(InvalidRequestError, match=r"(?i)prompt is required for text_to_music"):
            YuE2Adapter.map_inputs("text_to_music", {"prompt": "   "})

        # lyrics_to_music requires lyrics
        with pytest.raises(InvalidRequestError, match=r"(?i)lyrics are required for lyrics_to_music"):
            YuE2Adapter.map_inputs("lyrics_to_music", {"lyrics": ""})

        # custom_score requires custom_score (valid ABC)
        with pytest.raises(InvalidRequestError, match=r"(?i)custom_score .* is required"):
            YuE2Adapter.map_inputs("custom_score", {})

        # custom_score rejects invalid ABC score
        with pytest.raises(InvalidRequestError, match=r"(?i)must include a valid Key header"):
            YuE2Adapter.map_inputs("custom_score", {"custom_score": "X:1\nT:No Key\nC D E |]"})

        # symbolic_score requires at least one of prompt, lyrics, or score
        with pytest.raises(InvalidRequestError, match=r"(?i)requires at least one of prompt, lyrics, or existing score"):
            YuE2Adapter.map_inputs("symbolic_score", {})

        # score_edit requires base_score and modifications dict
        with pytest.raises(InvalidRequestError, match=r"(?i)requires base_score"):
            YuE2Adapter.map_inputs("score_edit", {"modifications": {"tempo": 120}})

        with pytest.raises(InvalidRequestError, match=r"(?i)requires modifications dictionary"):
            valid_abc = generate_sample_abc()
            YuE2Adapter.map_inputs("score_edit", {"base_score": valid_abc, "modifications": None})

        # song_variation requires prompt, lyrics, or score
        with pytest.raises(InvalidRequestError, match=r"(?i)requires prompt, lyrics, or base score"):
            YuE2Adapter.map_inputs("song_variation", {})

    def test_sanitize_genres_and_styles(self):
        """Genres provided as list of strings or mixed types are safely normalized."""
        mapped = YuE2Adapter.map_inputs("text_to_music", {
            "prompt": "Cyberpunk chase",
            "style": ["synthwave", "dark electro", 123, None, "  retro  "],
        })
        assert mapped["genres"] == "synthwave, dark electro, 123, retro"


# ============================================================================
# 3. Client Error Sanitization & Resilience Probes (client.py)
# ============================================================================

class TestYuE2ClientSanitizationAndResilience:
    """Stress tests client resilience against network anomalies, malformed URLs, and backend failures."""

    @pytest.mark.parametrize("raw_url,expected_stripped", [
        ("http://127.0.0.1:8009", "http://127.0.0.1:8009"),
        ("  http://127.0.0.1:8009  ", "http://127.0.0.1:8009"),
        ("http://127.0.0.1:8009/", "http://127.0.0.1:8009"),
        ("http://127.0.0.1:8009////", "http://127.0.0.1:8009"),
    ])
    def test_url_sanitization_and_whitespace_stripping(self, raw_url, expected_stripped):
        """YuE2Client cleanly sanitizes base URLs by stripping whitespace and trailing slashes."""
        client = YuE2Client(api_base_url=raw_url)
        assert client.api_base_url == expected_stripped

    @pytest.mark.asyncio
    @pytest.mark.parametrize("whitespace_url", ["   ", "\t\t", "\n  \r\n", ""])
    async def test_whitespace_only_url_wraps_in_model_unavailable(self, whitespace_url):
        """Whitespace-only base URL raises ModelUnavailableError(code='BACKEND_UNAVAILABLE') without raw unhandled exception."""
        client = YuE2Client(api_base_url=whitespace_url)

        with pytest.raises(ModelUnavailableError) as exc_health:
            await client.health()
        assert exc_health.value.code == "BACKEND_UNAVAILABLE"

        with pytest.raises(ModelUnavailableError) as exc_gen:
            await client.generate_song("text_to_music", {"prompt": "Test"})
        assert exc_gen.value.code == "BACKEND_UNAVAILABLE"

        with pytest.raises(ModelUnavailableError) as exc_plan:
            await client.plan_score({"prompt": "Test"})
        assert exc_plan.value.code == "BACKEND_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_connection_refused_wraps_cleanly_in_backend_unavailable(self):
        """Unreachable sockets wrap cleanly in ModelUnavailableError(code='BACKEND_UNAVAILABLE')."""
        client = YuE2Client(api_base_url="http://127.0.0.1:9999")  # Non-listening port

        with pytest.raises(ModelUnavailableError) as exc_health:
            await client.health()
        assert exc_health.value.code == "BACKEND_UNAVAILABLE"
        assert "unavailable" in str(exc_health.value).lower()

        with pytest.raises(ModelUnavailableError) as exc_gen:
            await client.generate_song("text_to_music", {"prompt": "Test"})
        assert exc_gen.value.code == "BACKEND_UNAVAILABLE"
        assert "failed" in str(exc_gen.value).lower()

        with pytest.raises(ModelUnavailableError) as exc_plan:
            await client.plan_score({"prompt": "Test"})
        assert exc_plan.value.code == "BACKEND_UNAVAILABLE"
        assert "failed" in str(exc_plan.value).lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 500, 502, 503, 504])
    async def test_http_error_responses_wrap_in_model_unavailable(self, status_code):
        """Non-200 responses from backend wrap in ModelUnavailableError(code='BACKEND_UNAVAILABLE')."""
        client = YuE2Client(api_base_url="http://mock-yue2-backend:8009")

        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.text = f"HTTP Error {status_code}: Backend failure"

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            with pytest.raises(ModelUnavailableError) as exc_gen:
                await client.generate_song("text_to_music", {"prompt": "Test"})
            assert exc_gen.value.code == "BACKEND_UNAVAILABLE"
            assert str(status_code) in str(exc_gen.value)

            with pytest.raises(ModelUnavailableError) as exc_plan:
                await client.plan_score({"prompt": "Test"})
            assert exc_plan.value.code == "BACKEND_UNAVAILABLE"
            assert str(status_code) in str(exc_plan.value)

    @pytest.mark.asyncio
    async def test_generate_song_missing_audio_data_fails_closed(self):
        """Backend response missing audio_base64, audio_bytes, and audio_url raises ModelUnavailableError."""
        client = YuE2Client(api_base_url="http://mock-yue2-backend:8009")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "success", "note": "missing all audio fields"}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            with pytest.raises(ModelUnavailableError, match=r"missing audio data or url"):
                await client.generate_song("text_to_music", {"prompt": "Test"})

    @pytest.mark.asyncio
    async def test_generate_song_secondary_score_artifact_emission(self):
        """When YuE2 returns both audio and intermediate ABC score, both artifacts are packaged."""
        client = YuE2Client(api_base_url="http://mock-yue2-backend:8009")
        raw_audio = make_dummy_wav_bytes(duration_sec=2.0)
        abc_score = generate_sample_abc(title="Generated Score", key="Am")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "audio_base64": base64.b64encode(raw_audio).decode("utf-8"),
            "abc_score": abc_score,
            "version": "2.0.0",
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            res = await client.generate_song("text_to_music", {"prompt": "Epic symphony"})

            assert "audio_bytes" in res
            assert res["audio_bytes"] == raw_audio
            assert res["abc_score"] == abc_score
            artifacts = res["artifacts"]
            assert len(artifacts) == 2
            assert artifacts[0]["type"] == "audio"
            assert artifacts[0]["mime_type"] == "audio/wav"
            assert artifacts[1]["type"] == "score"
            assert artifacts[1]["mime_type"] == "text/vnd.abc"
            assert artifacts[1]["audio_bytes"] == abc_score.encode("utf-8")


# ============================================================================
# 4. MusicWorker Multi-Model Routing & Asset Safety Probes (worker.py)
# ============================================================================

class TestMusicWorkerMultiModelRoutingAndAssetSafety:
    """Stress tests MusicWorker multi-model routing, SSRF validation, and asset preservation."""

    def test_supported_models_and_tasks_matrix(self):
        """Worker must declare both ace-step-1.5 and yue2 in SUPPORTED_MODELS and include YuE2 tasks."""
        assert "yue2" in SUPPORTED_MODELS
        assert "ace-step-1.5" in SUPPORTED_MODELS
        for yue_task in ["symbolic_score", "custom_score", "score_edit", "song_variation"]:
            assert yue_task in SUPPORTED_TASKS

    @pytest.mark.asyncio
    async def test_file_uri_deletion_hazard_preservation_probe(self):
        """
        [CRITICAL REMEDIATION: VULN-P3-01 PRESERVATION ON YUE2 ROUTE]
        Passing file:// URI causes stage_asset() to create a secure temporary copy.
        When execute_job finishes (both on success and failure), the finally block unlinks only
        the temporary staged copy, preserving the caller's original file on disk.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(b"precious caller recording for yue2 variation")
            user_path = f.name

        staged = None
        try:
            worker = MusicWorker(worker_id="test-worker-yue2")
            staged = await worker.stage_asset(AsyncMock(), f"file://{user_path}")
            assert staged != user_path, "stage_asset must return a temporary copy, not caller's file path!"
            assert Path(staged).exists(), "Temporary staged copy must exist!"

            # Simulate worker execute_job cleanup loop
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
    async def test_absolute_path_deletion_hazard_preservation_probe(self):
        """Passing absolute local path causes stage_asset() to copy file, protecting original from deletion."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(b"local caller sample")
            user_path = f.name

        staged = None
        try:
            worker = MusicWorker(worker_id="test-worker-yue2")
            staged = await worker.stage_asset(AsyncMock(), user_path)
            assert staged != user_path, "stage_asset must return a copy for local paths!"
            assert Path(staged).exists()

            # Simulate worker cleanup
            staged_files = [staged]
            for st_file in staged_files:
                p = Path(st_file)
                if p.exists():
                    p.unlink()

            assert Path(user_path).exists(), "Original caller file MUST be preserved!"
            assert not Path(staged).exists()
        finally:
            if staged and Path(staged).exists():
                Path(staged).unlink()
            if Path(user_path).exists():
                Path(user_path).unlink()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("malicious_url", [
        "http://169.254.169.254/latest/meta-data",
        "http://127.0.0.1:8080/internal-secrets",
        "http://10.0.0.1/private-database",
        "http://192.168.1.1/router-config",
        "http://172.16.0.1/internal-api",
    ])
    async def test_ssrf_preflight_validation_probe(self, malicious_url):
        """
        [HIGH REMEDIATION: VULN-P3-02 PRESERVATION]
        Remote HTTP/HTTPS URLs are validated with SSRF pre-flight validation.
        Cloud metadata (169.254.169.254) and private loopback/RFC 1918 IPs are blocked.
        """
        worker = MusicWorker(worker_id="test-worker-yue2")
        mock_client = AsyncMock()

        with pytest.raises(InvalidRequestError, match=r"(?i)blocked by ssrf policy"):
            await worker.stage_asset(mock_client, malicious_url)

    @pytest.mark.asyncio
    async def test_worker_multi_model_routing_yue2_vs_acestep(self):
        """MusicWorker dynamically selects YuE2Client vs ACEStepClient based on job.model."""
        worker = MusicWorker(worker_id="test-router-worker", api_key="secret-key")

        dummy_bytes = make_dummy_wav_bytes()
        valid_sha = hashlib.sha256(dummy_bytes).hexdigest()

        worker.yue2_client.generate_song = AsyncMock()
        worker.yue2_client.generate_song.return_value = {
            "audio_bytes": dummy_bytes,
            "filename": "yue2_out.wav",
            "mime_type": "audio/wav",
            "size_bytes": len(dummy_bytes),
            "sha256": valid_sha,
            "artifacts": [{
                "type": "audio",
                "audio_bytes": dummy_bytes,
                "filename": "yue2_out.wav",
                "mime_type": "audio/wav",
                "size_bytes": len(dummy_bytes),
                "sha256": valid_sha,
            }],
        }

        worker.client.generate_music = AsyncMock()
        worker.client.generate_music.return_value = {
            "audio_bytes": dummy_bytes,
            "filename": "acestep_out.wav",
            "mime_type": "audio/wav",
            "size_bytes": len(dummy_bytes),
            "sha256": valid_sha,
            "artifacts": [{
                "type": "audio",
                "audio_bytes": dummy_bytes,
                "filename": "acestep_out.wav",
                "mime_type": "audio/wav",
                "size_bytes": len(dummy_bytes),
                "sha256": valid_sha,
            }],
        }

        # Mock Control Plane responses
        mock_http = AsyncMock()
        mock_http.post.side_effect = [
            # presign upload
            MagicMock(status_code=200, json=lambda: {"upload_url": "http://mock-storage/put", "storage_path": "path1"}),
            # download url
            MagicMock(status_code=200, json=lambda: {"download_url": "http://mock-storage/get"}),
            # complete job
            MagicMock(status_code=200, text="OK"),
        ]

        with patch("httpx.AsyncClient.put", new_callable=AsyncMock) as mock_put:
            mock_put.return_value = MagicMock(status_code=200)

            # 1. Route YuE2 job
            yue_uid = str(uuid.uuid4())
            req_yue_uid = str(uuid.uuid4())
            yue_job = FactoryJob(
                job_id=f"job-{yue_uid}",
                request_id=f"req-{req_yue_uid}",
                service="music-service",
                task="text_to_music",
                model="yue2",
                inputs={"prompt": "YuE2 Epic Anthem", "duration": 30.0},
            )
            success_yue = await worker.execute_job(mock_http, yue_job)
            assert success_yue is True
            worker.yue2_client.generate_song.assert_awaited_once()
            worker.client.generate_music.assert_not_awaited()

            # Reset mocks
            worker.yue2_client.generate_song.reset_mock()
            worker.client.generate_music.reset_mock()

            # Mock Control Plane responses for ACE-Step
            mock_http.post.side_effect = [
                MagicMock(status_code=200, json=lambda: {"upload_url": "http://mock-storage/put", "storage_path": "path2"}),
                MagicMock(status_code=200, json=lambda: {"download_url": "http://mock-storage/get"}),
                MagicMock(status_code=200, text="OK"),
            ]

            # 2. Route ACE-Step job
            ace_uid = str(uuid.uuid4())
            req_ace_uid = str(uuid.uuid4())
            ace_job = FactoryJob(
                job_id=f"job-{ace_uid}",
                request_id=f"req-{req_ace_uid}",
                service="music-service",
                task="text_to_music",
                model="ace-step-1.5",
                inputs={"prompt": "ACE-Step Beat", "duration_seconds": 15.0},
            )
            success_ace = await worker.execute_job(mock_http, ace_job)
            assert success_ace is True
            worker.client.generate_music.assert_awaited_once()
            worker.yue2_client.generate_song.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_worker_persists_multi_type_artifacts_audio_and_score(self):
        """Worker properly uploads and persists multiple artifacts (audio and score) for YuE2 jobs."""
        worker = MusicWorker(worker_id="test-multi-art-worker", api_key="secret-key")
        audio_content = make_dummy_wav_bytes(duration_sec=1.0)
        score_content = b"X:1\nT:Score\nK:C\nC D E F|]"

        worker.yue2_client.generate_song = AsyncMock()
        worker.yue2_client.generate_song.return_value = {
            "artifacts": [
                {
                    "type": "audio",
                    "audio_bytes": audio_content,
                    "filename": "anthem.wav",
                    "mime_type": "audio/wav",
                    "size_bytes": len(audio_content),
                    "sha256": hashlib.sha256(audio_content).hexdigest(),
                },
                {
                    "type": "score",
                    "audio_bytes": score_content,
                    "filename": "anthem.abc",
                    "mime_type": "text/vnd.abc",
                    "size_bytes": len(score_content),
                    "sha256": hashlib.sha256(score_content).hexdigest(),
                },
            ]
        }

        # Control Plane mock responses (2 artifacts -> 2 * 2 storage calls + 1 complete call)
        uploaded_artifacts = []

        async def mock_post(url, *args, **kwargs):
            if "upload-url" in url:
                json_payload = kwargs.get("json", {})
                return MagicMock(status_code=200, json=lambda: {
                    "upload_url": f"http://mock-storage/put/{json_payload.get('artifact_name')}",
                    "storage_path": f"storage/{json_payload.get('artifact_name')}",
                })
            elif "download-url" in url:
                json_payload = kwargs.get("json", {})
                return MagicMock(status_code=200, json=lambda: {
                    "download_url": f"http://mock-storage/get/{json_payload.get('filename')}",
                })
            elif "complete" in url:
                comp_req = kwargs.get("json", {})
                uploaded_artifacts.extend(comp_req.get("artifacts", []))
                return MagicMock(status_code=200, text="OK")
            return MagicMock(status_code=200, json=lambda: {})

        mock_http = AsyncMock()
        mock_http.post.side_effect = mock_post

        with patch("httpx.AsyncClient.put", new_callable=AsyncMock) as mock_put:
            mock_put.return_value = MagicMock(status_code=200)

            job_uid = str(uuid.uuid4())
            req_uid = str(uuid.uuid4())
            job = FactoryJob(
                job_id=f"job-{job_uid}",
                request_id=f"req-{req_uid}",
                service="music-service",
                task="text_to_music",
                model="yue2",
                inputs={"prompt": "Song with ABC score", "duration": 30.0},
            )
            success = await worker.execute_job(mock_http, job)
            assert success is True
            assert len(uploaded_artifacts) == 2
            assert uploaded_artifacts[0]["type"] == "audio"
            assert uploaded_artifacts[0]["mime_type"] == "audio/wav"
            assert uploaded_artifacts[1]["type"] == "score"
            assert uploaded_artifacts[1]["mime_type"] == "text/vnd.abc"


# ============================================================================
# 5. Gate Y0 Feasibility & Honesty Probes (runtime.py & manifest)
# ============================================================================

class TestGateY0FeasibilityHonestyAndLifecycle:
    """Stress tests Gate Y0 hardware checker and model lifecycle enforcement."""

    def test_gate_y0_honestly_reports_hardware_blocked_in_non_cuda_environment(self):
        """In CPU/macOS Darwin environment, check_gate_y0() must honestly return HARDWARE_BLOCKED."""
        result = check_gate_y0()
        assert result["status"] == FeasibilityStatus.HARDWARE_BLOCKED.value
        assert result["can_run_live"] is False
        assert "cuda is not available" in result["reason"].lower() or "pytorch is not installed" in result["reason"].lower()

    def test_gate_y0_hardware_boundary_probes(self):
        """Simulates GPU profiles across VRAM and BF16 boundaries to verify exact Gate Y0 tiers."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        # 1. VRAM < 12GB (e.g. 8GB RTX 3070) -> HARDWARE_BLOCKED
        props_8gb = MagicMock(total_memory=8 * (1024**3), name="NVIDIA GeForce RTX 3070", major=8, minor=6)
        mock_torch.cuda.get_device_properties.return_value = props_8gb
        mock_torch.cuda.is_bf16_supported.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            res_8gb = check_gate_y0(0)
            assert res_8gb["status"] == FeasibilityStatus.HARDWARE_BLOCKED.value
            assert res_8gb["can_run_live"] is False
            assert "insufficient vram" in res_8gb["reason"].lower()

        # 2. 16GB Dual T4 without native BF16 -> FEASIBLE_TRACK_B_ONLY
        props_16gb = MagicMock(total_memory=15.8 * (1024**3), name="Tesla T4", major=7, minor=5)
        mock_torch.cuda.get_device_properties.return_value = props_16gb
        mock_torch.cuda.is_bf16_supported.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            res_16gb = check_gate_y0(0)
            assert res_16gb["status"] == FeasibilityStatus.FEASIBLE_TRACK_B_ONLY.value
            assert res_16gb["can_run_live"] is True
            assert res_16gb["supported_tracks"] == ["track_b"]

        # 3. >= 24GB with BF16 (e.g. A10G 24GB or A100 40GB) -> FEASIBLE
        props_24gb = MagicMock(total_memory=23.9 * (1024**3) + 500 * (1024**2), name="NVIDIA A10G", major=8, minor=6)
        mock_torch.cuda.get_device_properties.return_value = props_24gb
        mock_torch.cuda.is_bf16_supported.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            res_24gb = check_gate_y0(0)
            assert res_24gb["status"] == FeasibilityStatus.FEASIBLE.value
            assert res_24gb["can_run_live"] is True
            assert "track_a" in res_24gb["supported_tracks"]
            assert "track_b" in res_24gb["supported_tracks"]

    def test_runtime_ensure_loaded_raises_insufficient_vram_when_hardware_blocked(self):
        """YuE2Runtime.ensure_loaded() raises ModelUnavailableError(code='INSUFFICIENT_VRAM') when blocked."""
        runtime = YuE2Runtime(model_id="yue2", variant="standard", gpu_device=1)

        with pytest.raises(ModelUnavailableError) as exc_info:
            runtime.ensure_loaded()

        assert exc_info.value.code == "INSUFFICIENT_VRAM"
        assert runtime.state == RuntimeState.UNLOADED

    def test_manifest_capabilities_truth_integrity(self):
        """manifests/models/yue2.yaml correctly specifies capabilities_truth and pending_live status."""
        manifest_path = Path("/Volumes/WorkSpace/Factory/Gen_Image:Video/manifests/models/yue2.yaml")
        assert manifest_path.exists(), "yue2.yaml manifest must exist!"

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["metadata"]["id"] == "yue2"
        assert data["validation_status"] == "pending_live"

        truth = data.get("capabilities_truth", {})
        assert truth, "capabilities_truth matrix must be defined!"

        # Implemented capabilities must have declared: true, implemented: true, live_verified: false
        for task in ["text_to_music", "lyrics_to_music", "symbolic_score", "custom_score", "score_edit", "song_variation"]:
            assert task in truth, f"Task {task} must be in capabilities_truth"
            assert truth[task]["declared"] is True
            assert truth[task]["implemented"] is True
            assert truth[task]["live_verified"] is False, f"Task {task} cannot claim live_verified without Kaggle Dual T4 certification!"

        # Unimplemented capabilities must be explicitly false
        assert truth["cover_edit"]["declared"] is True
        assert truth["cover_edit"]["implemented"] is False
        assert truth["cover_edit"]["live_verified"] is False


# ============================================================================
# 6. Cross-Model Regression & Isolation Invariants
# ============================================================================

class TestCrossModelRegressionAndIsolationInvariants:
    """Verifies that YuE2 integration does not pollute or compromise ACE-Step subsystem invariants."""

    def test_acestep_adapter_unsupported_tasks_unchanged(self):
        """ACE-Step adapter must reject YuE2 symbolic score tasks with error contract exception."""
        from services.music.acestep.adapter import ACEStepAdapter

        for yue2_task in ["symbolic_score", "custom_score", "score_edit", "song_variation"]:
            with pytest.raises((InvalidRequestError, UnsupportedCapabilityError)):
                ACEStepAdapter.map_inputs(yue2_task, {"prompt": "Test"})

    def test_yue2_adapter_unsupported_tasks_unchanged(self):
        """YuE2 adapter must reject ACE-Step specialized tasks with UnsupportedCapabilityError."""
        for ace_task in ["extract", "lego", "complete", "music_extend", "repaint", "cover"]:
            with pytest.raises(UnsupportedCapabilityError):
                YuE2Adapter.map_inputs(ace_task, {"prompt": "Test"})


# ============================================================================
# 7. Inoculation Probes for Empirical Security & Reliability Vulnerabilities
# ============================================================================

class TestEmpiricalVulnerabilitiesInoculationProbes:
    """
    Empirically documents verified vulnerabilities to guarantee reproducible evidence
    for adversarial reporting and remediation.
    """

    def test_vuln_p4_01_space_in_key_header_falsely_rejected_probe(self):
        """
        [MEDIUM REMEDIATION: VULN-P4-01 - REMEDIATED]
        KEY_REGEX in services/music/yue2/score.py:16-19 allows whitespace [ \t]*
        between root note [A-G][b#]? and mode name (major, minor, dor, mix, etc.).
        Standard ABC notation (ABC 2.1 specification §3.1.4.3) e.g. 'K:C major', 'K:A minor'
        is correctly accepted.
        """
        score = "X:1\nT:Standard Key Signature\nK:C major\nC D E F|]"
        info = validate_abc_score(score)
        assert info["valid"] is True
        assert info["headers"]["K"] == "C major"

        score_minor = "X:1\nT:Standard Minor Key\nK:A minor\nA B c d|]"
        info_m = validate_abc_score(score_minor)
        assert info_m["valid"] is True
        assert info_m["headers"]["K"] == "A minor"

    def test_vuln_p4_02_zero_tempo_falsely_accepted_probe(self):
        """
        [LOW/MEDIUM REMEDIATION: VULN-P4-02 - REMEDIATED]
        TEMPO_REGEX and validate_abc_score in services/music/yue2/score.py strictly reject Q:0 (0 BPM).
        A tempo of 0 BPM is physically and musically invalid, and is rejected with InvalidRequestError.
        """
        score = "X:1\nT:Zero Tempo\nQ:0\nK:C\nC D E F|]"
        with pytest.raises(InvalidRequestError, match=r"(?i)Invalid tempo|strictly greater than 0 BPM"):
            validate_abc_score(score)

    def test_vuln_p4_03_altered_and_added_chords_silently_dropped_probe(self):
        """
        [LOW REMEDIATION: VULN-P4-03 - REMEDIATED]
        CHORD_REGEX in services/music/yue2/score.py:25-27 captures added tones
        ('add9', 'add2') and altered extension degrees ('b5', '#9', 'b9', '#11', 'b13').
        Common jazz, pop, and blues chords ('Cadd9', 'Dm7b5', 'G7#9') are extracted.
        """
        score = 'X:1\nT:Chords\nK:C\n"Cmaj7" C4 | "Cadd9" C4 | "Dm7b5" D4 | "G7#9" G4 |]'
        chords = extract_chords(score)
        assert "Cmaj7" in chords
        assert "Cadd9" in chords
        assert "Dm7b5" in chords
        assert "G7#9" in chords

    def test_vuln_p4_04_inline_comment_quote_triggers_false_unclosed_error_probe(self):
        """
        [LOW REMEDIATION: VULN-P4-04 - REMEDIATED]
        _check_balanced_delimiters in services/music/yue2/score.py strips trailing
        inline comments ('% ...') before checking quote balance.
        A single quote inside an inline comment does not trigger false unclosed quotes error.
        """
        score = 'X:1\nT:Inline Comment\nK:C\nC D E F | % note: play "fast\n'
        info = validate_abc_score(score)
        assert info["valid"] is True

