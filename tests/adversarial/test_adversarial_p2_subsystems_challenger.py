"""
tests/adversarial/test_adversarial_p2_subsystems_challenger.py
Phase 2 Adversarial Challenger Test Suite:
Empirical verification of security boundaries, edge cases, model mappings,
fail-closed behaviors, and parameter validation across all 8 subsystems (§6.1 - §6.5).
"""
import asyncio
import io
import math
import os
import struct
import tempfile
import wave
from pathlib import Path
from unittest.mock import patch
import pytest

from factory_contracts.errors import (
    InvalidMediaError,
    InvalidRequestError,
    ModelUnavailableError,
    SSRFBlockedError,
)
from services.text.src.models import ChatCompletionRequest, ChatMessage
from services.text.src.engine import (
    decouple_think_tokens,
    StreamingReasoningFilter,
    TextEngine,
)
from services.decision.src.schemas import DecisionRequest
from services.decision.src.engine import DecisionEngine
from services.embedding.src.schemas import EmbeddingRequest
from services.embedding.src.engine import EmbeddingEngine
from services.embedding.src.multimodal_embed import MultimodalEmbedder
from services.speech.server import validate_factory_asset_url
from services.speech.tts.engine import TTSEngine
from services.speech.tts.voice_cloning import VoiceCloner
from services.speech.tts.speed import adjust_audio_speed
from workflows.image.parameter_mappings import ImageParameterMapper
from workflows.video.parameter_mappings import VideoParameterMapper
from services.video.extension_helper import VideoExtensionHelper
from services.enhancement.processors.image_sr import ImageSuperResolutionProcessor
from services.enhancement.processors.video_sr import VideoSuperResolutionProcessor
from services.enhancement.processors.interpolation import FrameInterpolationProcessor
from services.enhancement.processors.audio_enhance import AudioEnhancementProcessor
from services.enhancement.server import health_check


# ==============================================================================
# 1. Text, VLM, Decision, Embedding & Reranking (§6.1)
# ==============================================================================

class TestTextVlmReasoningDecouplingAdversarial:
    """Stress-tests reasoning token decoupling, effort validation, and multimodal embeddings."""

    @pytest.mark.parametrize(
        "raw_text,expected_reasoning,expected_clean",
        [
            # Standard closed tag
            ("<think>Thinking step by step</think>The final answer is 42.", "Thinking step by step", "The final answer is 42."),
            # Leading newline and unclosed tag
            ("\n<think>Thinking without closing tag", "Thinking without closing tag", ""),
            # Indented with leading whitespace and unclosed tag
            ("   \n\t <think>Indented unclosed thought", "Indented unclosed thought", ""),
            # Prefix text before unclosed think
            ("Context before\n<think>Unclosed reasoning block", "Unclosed reasoning block", "Context before"),
            # Prefix text and suffix text around closed tag
            ("Intro note\n<think>Deep reflection</think>\nConclusion", "Deep reflection", "Intro note Conclusion"),
            # Multi-line reasoning block
            ("\n<think>\nStep 1: Check inputs\nStep 2: Solve\n</think>\nResult: Success", "Step 1: Check inputs\nStep 2: Solve", "Result: Success"),
            # Empty think tag
            ("<think></think>Direct answer", None, "Direct answer"),
            ("<think>   \n  </think>Direct answer", None, "Direct answer"),
            # Only closing tag
            ("</think>Only closing tag", None, "Only closing tag"),
            # Nested opening tag with single closing tag
            ("<think>Outer <think>Inner tag</think>Final answer", "Outer <think>Inner tag", "Final answer"),
            # Multiple closing tags: first </think> splits reasoning, remainder stays in clean
            ("<think>Outer <think>Inner</think> more</think>Final", "Outer <think>Inner", "more</think>Final"),
            # Empty or None
            ("", None, ""),
            (None, None, None),
        ],
    )
    def test_reasoning_decoupling_edge_cases(self, raw_text, expected_reasoning, expected_clean):
        reasoning, clean = decouple_think_tokens(raw_text)
        assert reasoning == expected_reasoning
        assert clean == expected_clean

    @pytest.mark.parametrize(
        "invalid_effort",
        [
            "",
            "   ",
            123,
            True,
            False,
            ["medium"],
            {"nested": "val"},
            "extreme",
            "ultra_high",
            "-1",
            "0",
        ],
    )
    def test_reasoning_effort_string_and_dict_validation(self, invalid_effort):
        # Direct reasoning_effort field
        with pytest.raises(Exception):
            ChatCompletionRequest(
                model="qwen-2.5-7b-instruct",
                messages=[ChatMessage(role="user", content="Test")],
                reasoning_effort=invalid_effort,
            )

        # Object format {"effort": invalid_effort}
        with pytest.raises(Exception):
            ChatCompletionRequest(
                model="qwen-2.5-7b-instruct",
                messages=[ChatMessage(role="user", content="Test")],
                reasoning={"effort": invalid_effort},
            )

    def test_streaming_reasoning_filter_fragmentation(self):
        filt = StreamingReasoningFilter()
        chunks = [
            {"content": "Preliminary notes: "},
            {"content": "<think>Deliberating step 1, "},
            {"content": "step 2, "},
            {"content": "concluding thought</think>Direct answer "},
            {"content": "continues here."},
        ]
        processed = [filt.process_chunk_delta(c) for c in chunks]

        assert processed[0] == {"content": "Preliminary notes: "}
        assert processed[1] == {"reasoning_content": "Deliberating step 1, "}
        assert processed[2] == {"reasoning_content": "step 2, "}
        assert processed[3] == {"reasoning_content": "concluding thought", "content": "Direct answer "}
        assert processed[4] == {"content": "continues here."}

    def test_multimodal_embedding_fails_closed_without_synthetic_fallback(self, monkeypatch):
        monkeypatch.delenv("MULTIMODAL_EMBEDDING_URL", raising=False)
        engine = EmbeddingEngine()

        # Batch multimodal embedding
        req = EmbeddingRequest(
            model="qwen3-embedding",
            input=[
                {"image_url": {"url": "https://s3.ai.example.com/item1.png"}},
                {"image_url": {"url": "https://s3.ai.example.com/item2.png"}},
            ],
        )
        with pytest.raises(ModelUnavailableError) as exc:
            engine.generate_embeddings(req)
        assert "Real multimodal embedding backend is not configured" in str(exc.value)

    def test_decision_engine_adversarial_validation(self, monkeypatch):
        monkeypatch.delenv("VLLM_DECISION_URL", raising=False)
        engine = DecisionEngine()

        # Empty context
        with pytest.raises(InvalidRequestError):
            engine.evaluate(DecisionRequest(model="qwen-2.5-3b-instruct", task="predicate", context="   "))

        # Offline model
        with pytest.raises(ModelUnavailableError):
            engine.evaluate(DecisionRequest(model="offline-model", task="predicate", context="Valid context"))


# ==============================================================================
# 2. Speech Subsystem (§6.2)
# ==============================================================================

class TestSpeechSecurityAndCloningAdversarial:
    """Stress-tests SSRF defenses, voice cloning contracts, and speed transformation."""

    @pytest.mark.parametrize(
        "malicious_url",
        [
            "http://127.0.0.1:8288/audio.wav",
            "http://localhost:8000/audio.wav",
            "http://169.254.169.254/latest/meta-data",
            "file:///etc/passwd",
            "gopher://127.0.0.1:8000/",
            "http://192.168.1.1/internal.wav",
            "http://10.0.0.5/audio.wav",
            "https://evil-untrusted-domain.com/audio.wav",
            "not_a_valid_url",
            "javascript:alert(1)",
        ],
    )
    def test_ssrf_and_untrusted_host_rejection(self, malicious_url, monkeypatch):
        monkeypatch.setenv("FACTORY_ASSET_ALLOWED_HOSTS", "s3.ai.example.com,assets.ai.example.com")
        monkeypatch.setenv("S3_PUBLIC_ENDPOINT_URL", "https://factory-storage.ai.example.com")

        with pytest.raises((InvalidRequestError, SSRFBlockedError, InvalidMediaError)):
            validate_factory_asset_url(malicious_url)

    def test_voice_cloning_strictly_requires_reference_text(self, monkeypatch):
        monkeypatch.setenv("QWEN_TTS_URL", "http://localhost:8000")
        tts = TTSEngine()

        # Missing reference_text (None)
        with pytest.raises(InvalidRequestError) as exc_none:
            tts.synthesize(
                text="Cloned speech text",
                reference_audio_url="https://s3.ai.example.com/ref.wav",
                reference_text=None,
            )
        assert "reference_text together with reference_audio_url" in str(exc_none.value)

        # Empty reference_text ("")
        with pytest.raises(InvalidRequestError) as exc_empty:
            tts.synthesize(
                text="Cloned speech text",
                reference_audio_url="https://s3.ai.example.com/ref.wav",
                reference_text="",
            )
        assert "reference_text together with reference_audio_url" in str(exc_empty.value)

    def test_voice_cloner_prohibits_fake_hashes(self):
        cloner = VoiceCloner()
        with pytest.raises(ModelUnavailableError) as exc:
            cloner.extract_speaker_profile("https://s3.ai.example.com/ref.wav")
        assert "Local speaker-profile synthesis is disabled" in str(exc.value)

    def test_speed_transformation_extreme_bounds_and_corruption(self):
        # 0.1s 16-bit PCM WAV at 8000Hz (800 samples)
        sample_rate = 8000
        num_samples = 800
        samples = [int(1000 * math.sin(2 * 3.14159 * 440 * i / sample_rate)) for i in range(num_samples)]
        raw_frames = struct.pack(f"<{num_samples}h", *samples)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(raw_frames)
        valid_wav = buf.getvalue()

        # Extreme speeds clamp within [0.25, 4.0]
        # 0.0x or negative speeds clamp to 0.25x -> 800 / 0.25 = 3200 frames
        slow_wav = adjust_audio_speed(valid_wav, 0.0)
        with wave.open(io.BytesIO(slow_wav), "rb") as wf:
            assert wf.getnframes() == 3200

        neg_wav = adjust_audio_speed(valid_wav, -2.5)
        with wave.open(io.BytesIO(neg_wav), "rb") as wf:
            assert wf.getnframes() == 3200

        # 100.0x clamps to 4.0x -> 800 / 4.0 = 200 frames
        fast_wav = adjust_audio_speed(valid_wav, 100.0)
        with wave.open(io.BytesIO(fast_wav), "rb") as wf:
            assert wf.getnframes() == 200

        # Corrupted and empty inputs pass through safely without crash
        for corrupted in [b"", b"RIFF", b"RIFF\x00\x00\x00\x00WAVEfmt ", b"\x00" * 30]:
            assert adjust_audio_speed(corrupted, 2.0) == corrupted


# ==============================================================================
# 3. Image Subsystem — Qwen-Image 2.1 (§6.3)
# ==============================================================================

class TestImageQwenParameterMappingAdversarial:
    """Stress-tests Qwen-Image 2.1 parameter mappings, Turbo bounds, and graph resilience."""

    @pytest.mark.parametrize("invalid_steps", [0, 26, 1000, -1, 4.5, "8"])
    def test_turbo_step_boundaries_strictly_rejected(self, invalid_steps):
        with pytest.raises(ValueError):
            ImageParameterMapper.apply(
                {"476": {"inputs": {}}},
                task="text_to_image",
                inputs={"variant": "turbo", "steps": invalid_steps},
            )

    @pytest.mark.parametrize("incompatible_task", ["inpaint", "outpaint", "instruction_edit", "controlnet"])
    def test_turbo_incompatible_tasks_strictly_rejected(self, incompatible_task):
        with pytest.raises(ValueError) as exc:
            ImageParameterMapper.apply(
                {},
                task=incompatible_task,
                inputs={"variant": "turbo"},
            )
        assert "does not support turbo variant weights" in str(exc.value)

    def test_multi_reference_bounding_caps_at_four(self):
        ref_urls = [f"https://s3.ai.example.com/ref_{i}.png" for i in range(12)]
        wf = ImageParameterMapper.apply(
            {"471": {"inputs": {}}, "476": {"inputs": {}}},
            task="reference_conditioning",
            inputs={"reference_image_urls": ref_urls},
        )
        created_nodes = [k for k in wf.keys() if k.startswith("ref_img_")]
        assert len(created_nodes) == 4
        assert "ref_img_0" in wf and "ref_img_3" in wf
        assert "ref_img_4" not in wf

    def test_custom_graph_missing_node_fields_resilience(self):
        # Graph missing inputs and class_type on node 474
        custom_wf = {
            "474": {},
            "unrelated_node": {"key": "value"},
        }
        res = ImageParameterMapper.apply(
            custom_wf,
            task="text_to_image",
            inputs={"variant": "standard", "prompt": "resilience test"},
        )
        assert "474" in res

    def test_outpaint_negative_padding_bounds(self):
        wf = {"474_pad": {"inputs": {}}}
        res = ImageParameterMapper.apply(
            wf,
            task="outpaint",
            inputs={"expand": {"left": -15, "top": 0, "right": 40, "bottom": -5}},
        )
        assert res["474_pad"]["inputs"]["left"] == -15
        assert res["474_pad"]["inputs"]["top"] == 0
        assert res["474_pad"]["inputs"]["right"] == 40
        assert res["474_pad"]["inputs"]["bottom"] == -5


# ==============================================================================
# 4. Video Subsystem — Wan 2.2 & MiniMax H3 (§6.4)
# ==============================================================================

class TestVideoResolutionAndExtensionAdversarial:
    """Stress-tests MiniMax H3 resolution rounding and video extension error handling."""

    @pytest.mark.parametrize(
        "w_in,h_in,expected_w,expected_h",
        [
            (770, 450, 768, 448),
            (1000, 600, 992, 608),
            (1, 1, 32, 32),
            (31, 31, 32, 32),
            (33, 33, 32, 32),
            (48, 48, 64, 64),
            (0, 0, 768, 448),
            (-10, -50, 32, 32),
            (None, None, 768, 448),
        ],
    )
    def test_minimax_h3_resolution_rounding_multiples_of_32(self, w_in, h_in, expected_w, expected_h):
        wf = {"MiniMaxH3EmptyLatent": {"class_type": "MiniMaxH3EmptyLatent", "inputs": {}}}
        out = VideoParameterMapper.apply(
            wf,
            model="minimax-h3",
            task="text_to_video",
            inputs={"width": w_in, "height": h_in},
        )
        w_out = out["MiniMaxH3EmptyLatent"]["inputs"]["width"]
        h_out = out["MiniMaxH3EmptyLatent"]["inputs"]["height"]
        assert w_out % 32 == 0
        assert h_out % 32 == 0
        assert w_out == expected_w
        assert h_out == expected_h

    def test_video_extension_helper_error_paths(self):
        with tempfile.TemporaryDirectory() as td:
            tpath = Path(td)
            missing_src = tpath / "missing.mp4"
            frame_out = tpath / "frame.png"

            # Missing source video
            with pytest.raises(RuntimeError) as exc_missing:
                VideoExtensionHelper.extract_anchor_frame(missing_src, frame_out)
            assert "Source video does not exist" in str(exc_missing.value)

            # Corrupted source video (<= 100 bytes)
            corrupt_src = tpath / "corrupt.mp4"
            corrupt_src.write_bytes(b"too small bytes")
            with pytest.raises(RuntimeError) as exc_corrupt:
                VideoExtensionHelper.extract_anchor_frame(corrupt_src, frame_out)
            assert "corrupt or too small" in str(exc_corrupt.value)

            # Missing concatenation source
            with pytest.raises(RuntimeError) as exc_concat:
                VideoExtensionHelper.concatenate_videos(missing_src, corrupt_src, tpath / "out.mp4")
            assert "Source video does not exist" in str(exc_concat.value)


# ==============================================================================
# 5. Enhancement Subsystem (§6.5)
# ==============================================================================

class TestEnhancementRoutingAndHealthAdversarial:
    """Stress-tests image SR model routing, scale factor validation, and fail-closed health reporting."""

    def test_image_sr_model_routing_and_rejection(self, monkeypatch):
        monkeypatch.setenv("IMAGE_SR_BACKEND_URL", "http://localhost:8000")
        proc = ImageSuperResolutionProcessor()

        # Unrecognized model names rejected with InvalidRequestError
        for bad_model in ["esrgan_v2", "fake_sr", "waifu2x", "unknown_sr"]:
            with pytest.raises(InvalidRequestError) as exc:
                proc.process({"image_url": "https://s3.ai.example.com/in.png", "scale_factor": 2, "model_name": bad_model})
            assert "Unsupported image SR model" in str(exc.value)

    def test_image_sr_scale_factor_validation(self, monkeypatch):
        monkeypatch.setenv("IMAGE_SR_BACKEND_URL", "http://localhost:8000")
        proc = ImageSuperResolutionProcessor()

        for invalid_scale in [1, 3, 5, 8, -2, 0]:
            with pytest.raises(InvalidRequestError) as exc:
                proc.process({"image_url": "https://s3.ai.example.com/in.png", "scale_factor": invalid_scale, "model_name": "real-esrgan"})
            assert "supports scale_factor 2 or 4" in str(exc.value)

    def test_enhancement_fail_closed_health_when_unconfigured(self, monkeypatch):
        for env_k in [
            "IMAGE_SR_BACKEND_URL",
            "VIDEO_SR_BACKEND_URL",
            "RIFE_BACKEND_URL",
            "INTERPOLATION_BACKEND_URL",
            "AUDIO_ENHANCE_BACKEND_URL",
        ]:
            monkeypatch.delenv(env_k, raising=False)

        health = asyncio.run(health_check())
        assert health["infer_ready"] is False
        assert all(val is False for val in health["backends"].values())

    def test_all_enhancement_processors_fail_closed_without_backends(self, monkeypatch):
        for env_k in [
            "IMAGE_SR_BACKEND_URL",
            "VIDEO_SR_BACKEND_URL",
            "RIFE_BACKEND_URL",
            "INTERPOLATION_BACKEND_URL",
            "AUDIO_ENHANCE_BACKEND_URL",
        ]:
            monkeypatch.delenv(env_k, raising=False)

        processors = [
            (ImageSuperResolutionProcessor(), {"image_url": "https://s3.ai.example.com/in.png", "scale_factor": 2}),
            (VideoSuperResolutionProcessor(), {"video_url": "https://s3.ai.example.com/in.mp4", "scale_factor": 2}),
            (FrameInterpolationProcessor(), {"video_url": "https://s3.ai.example.com/in.mp4", "multiplier": 2}),
            (AudioEnhancementProcessor(), {"audio_url": "https://s3.ai.example.com/in.wav"}),
        ]
        for p, payload in processors:
            with pytest.raises(ModelUnavailableError) as exc:
                p.process(payload)
            assert "Real enhancement backend is not configured" in str(exc.value)
