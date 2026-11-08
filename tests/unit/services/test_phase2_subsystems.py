"""
tests/unit/services/test_phase2_subsystems.py
Comprehensive Phase 2 Subsystem Verification Test Suite:
- §6.1 Text, VLM, Decision, Embedding & Reranking
- §6.2 Speech (ASR, TTS, Voice Cloning, Forced Alignment, Speed Adjustment)
- §6.3 Image Subsystem (Qwen-Image 2.1 IMG-01 to IMG-10)
- §6.4 Video Subsystem (Wan 2.2 & MiniMax H3)
- §6.5 Enhancement Subsystem (EN-01 to EN-06)
"""
import io
import math
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from factory_contracts.errors import InvalidRequestError, ModelUnavailableError
from services.text.src.models import ChatCompletionRequest, ChatMessage
from services.text.src.engine import decouple_think_tokens, StreamingReasoningFilter, TextEngine
from services.text.src.multimodal import extract_content_text_and_media
from services.decision.src.schemas import DecisionRequest
from services.decision.src.engine import DecisionEngine
from services.embedding.src.schemas import EmbeddingRequest, RerankRequest
from services.embedding.src.engine import EmbeddingEngine
from services.embedding.src.multimodal_embed import MultimodalEmbedder
from services.speech.asr.engine import ASREngine
from services.speech.tts.engine import TTSEngine
from services.speech.tts.voice_cloning import VoiceCloner
from services.speech.tts.speed import adjust_audio_speed
from services.speech.alignment.aligner import ForcedAligner
from services.speech.server import validate_factory_asset_url
from workflows.image.parameter_mappings import ImageParameterMapper
from workflows.video.parameter_mappings import VideoParameterMapper
from services.video.extension_helper import VideoExtensionHelper
from services.enhancement.processors.image_sr import ImageSuperResolutionProcessor
from services.enhancement.processors.video_sr import VideoSuperResolutionProcessor
from services.enhancement.processors.interpolation import FrameInterpolationProcessor
from services.enhancement.processors.audio_enhance import AudioEnhancementProcessor


# ==============================================================================
# §6.1 Text, VLM, Decision, Embedding & Reranking Subsystems
# ==============================================================================

class TestTextAndVLMSubsystems:
    def test_text_chat_request_valid(self):
        req = ChatCompletionRequest(
            model="qwen-2.5-7b-instruct",
            messages=[ChatMessage(role="user", content="Hello world")],
            reasoning_effort="medium",
        )
        assert req.reasoning_effort == "medium"
        assert req.model == "qwen-2.5-7b-instruct"

    def test_text_chat_request_invalid_effort(self):
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="qwen-2.5-7b-instruct",
                messages=[ChatMessage(role="user", content="Hello")],
                reasoning_effort="invalid_level",
            )

    def test_text_chat_request_empty_or_non_string_effort_rejected(self):
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="qwen-2.5-7b-instruct",
                messages=[ChatMessage(role="user", content="Hello")],
                reasoning_effort="",
            )
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="qwen-2.5-7b-instruct",
                messages=[ChatMessage(role="user", content="Hello")],
                reasoning={"effort": 999},
            )

    def test_decouple_think_tokens_closed_block(self):
        reasoning, clean = decouple_think_tokens("<think>Thinking step by step</think>The final answer is 42.")
        assert reasoning == "Thinking step by step"
        assert clean == "The final answer is 42."

    def test_decouple_think_tokens_unclosed_block_with_newline(self):
        reasoning, clean = decouple_think_tokens("\n<think>Thinking without closing tag")
        assert reasoning == "Thinking without closing tag"
        assert clean == ""

    def test_decouple_think_tokens_empty_tag(self):
        reasoning, clean = decouple_think_tokens("<think></think>Direct answer")
        assert reasoning is None
        assert clean == "Direct answer"

    def test_streaming_reasoning_filter(self):
        filt = StreamingReasoningFilter()
        d1 = filt.process_chunk_delta({"content": "<think>Deliberating"})
        assert d1.get("reasoning_content") == "Deliberating"
        assert "content" not in d1

        d2 = filt.process_chunk_delta({"content": " done</think>Answer begins"})
        assert d2.get("reasoning_content") == " done"
        assert d2.get("content") == "Answer begins"

    def test_vlm_multimodal_extraction_data_url(self):
        content = [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
        ]
        text, media = extract_content_text_and_media(content)
        assert text == "What is in this image?"
        assert len(media) == 1
        assert media[0]["type"] == "image"


class TestDecisionAndEmbeddingSubsystems:
    def test_decision_engine_validates_inputs(self):
        engine = DecisionEngine()
        # Missing context
        req = DecisionRequest(model="qwen-2.5-3b-instruct", task="predicate", context="")
        with pytest.raises(InvalidRequestError):
            engine.evaluate(req)

        # Offline model
        req_offline = DecisionRequest(model="offline-model", task="predicate", context="Valid context")
        with pytest.raises(ModelUnavailableError):
            engine.evaluate(req_offline)

    def test_embedding_engine_single_and_batch_text(self):
        engine = EmbeddingEngine()
        # Missing input
        with pytest.raises(InvalidRequestError):
            engine.generate_embeddings(EmbeddingRequest(model="qwen3-embedding", input=""))

    def test_multimodal_embedder_fails_closed_when_backend_unconfigured(self, monkeypatch):
        monkeypatch.delenv("MULTIMODAL_EMBEDDING_URL", raising=False)
        embedder = MultimodalEmbedder()
        with pytest.raises(ModelUnavailableError) as exc_info:
            embedder.embed_multimodal_item({"image_url": "https://example.com/img.png"}, 0, 1024)
        assert "Real multimodal embedding backend is not configured" in str(exc_info.value)


# ==============================================================================
# §6.2 Speech Subsystem
# ==============================================================================

class TestSpeechSubsystem:
    def test_asr_engine_requires_backend(self, monkeypatch):
        monkeypatch.delenv("QWEN_ASR_URL", raising=False)
        asr = ASREngine()
        with pytest.raises(ModelUnavailableError):
            asr.transcribe(audio_bytes=b"RIFF\x24\x00\x00\x00WAVE")

    def test_tts_engine_empty_input_rejected(self):
        tts = TTSEngine()
        with pytest.raises(InvalidRequestError):
            tts.synthesize(text="")

    def test_tts_engine_voice_cloning_requires_reference_text(self, monkeypatch):
        monkeypatch.setenv("QWEN_TTS_URL", "http://localhost:8000")
        tts = TTSEngine()
        with pytest.raises(InvalidRequestError) as exc:
            tts.synthesize(
                text="Speak this text",
                reference_audio_url="https://s3.ai.example.com/ref.wav",
                reference_text=None,
            )
        assert "reference_text together with reference_audio_url" in str(exc.value)

    def test_voice_cloner_fails_closed_without_fake_hashes(self):
        cloner = VoiceCloner()
        with pytest.raises(ModelUnavailableError):
            cloner.extract_speaker_profile("https://s3.ai.example.com/sample.wav")

    def test_forced_aligner_requires_backend(self, monkeypatch):
        monkeypatch.delenv("QWEN_ALIGNER_URL", raising=False)
        aligner = ForcedAligner()
        with pytest.raises(ModelUnavailableError):
            aligner.align("https://s3.ai.example.com/audio.wav", "some transcript")

    def test_speed_control_wav_transformation(self):
        # Generate 0.1s of 16-bit PCM WAV at 8000Hz (800 samples)
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
        original_wav = buf.getvalue()

        # Adjust speed to 2.0x (should reduce frame count to ~400)
        fast_wav = adjust_audio_speed(original_wav, 2.0)
        with wave.open(io.BytesIO(fast_wav), "rb") as wf:
            assert wf.getnframes() == 400
            assert wf.getframerate() == sample_rate

    def test_speech_asset_url_validation_rejects_untrusted(self, monkeypatch):
        monkeypatch.setenv("FACTORY_ASSET_ALLOWED_HOSTS", "s3.ai.example.com")
        with patch("services.speech.server.validate_url_security"):
            # Untrusted host
            with pytest.raises(InvalidRequestError):
                validate_factory_asset_url("https://malicious.evil.com/audio.wav")
            # Allowed host
            validate_factory_asset_url("https://s3.ai.example.com/assets/audio.wav")


# ==============================================================================
# §6.3 Image Subsystem — Qwen-Image 2.1 (IMG-01 to IMG-10)
# ==============================================================================

class TestImageSubsystem:
    def test_img01_t2i_standard_vs_turbo_selection(self):
        base_workflow = {
            "471": {"inputs": {}},
            "474": {"class_type": "EmptyLatentImage", "inputs": {}},
            "476": {"inputs": {}},
            "478": {"inputs": {"unet_name": "placeholder"}},
        }
        # Standard
        wf_std = ImageParameterMapper.apply(
            dict(base_workflow),
            task="text_to_image",
            inputs={"variant": "standard", "prompt": "a mountain lake", "steps": 20},
        )
        assert wf_std["478"]["inputs"]["unet_name"] == "qwen-image-2.1-Q4_K_M.gguf"
        assert wf_std["476"]["inputs"]["steps"] == 20

        # Turbo
        wf_turbo = ImageParameterMapper.apply(
            dict(base_workflow),
            task="text_to_image",
            inputs={"variant": "turbo", "prompt": "a mountain lake", "steps": 8},
        )
        assert wf_turbo["478"]["inputs"]["unet_name"] == "qwen-image-2.1-turbo-Q4_K_M.gguf"
        assert wf_turbo["476"]["inputs"]["steps"] == 8

    def test_img01_turbo_step_ceiling_enforced(self):
        base_workflow = {"476": {"inputs": {}}, "478": {"inputs": {"unet_name": "placeholder"}}}
        with pytest.raises(ValueError) as exc:
            ImageParameterMapper.apply(
                base_workflow,
                task="text_to_image",
                inputs={"variant": "turbo", "steps": 30},
            )
        assert "Incompatible steps" in str(exc.value)

    def test_img02_image_to_image_denoise_mapping(self):
        wf = {"476": {"inputs": {}}, "474_img": {"inputs": {}}}
        res = ImageParameterMapper.apply(
            wf,
            task="image_to_image",
            inputs={"image_url": "input.png", "denoise_strength": 0.55},
        )
        assert res["476"]["inputs"]["denoise"] == 0.55
        assert res["474_img"]["inputs"]["image"] == "input.png"

    def test_img04_inpaint_mask_and_blur(self):
        wf = {"476": {"inputs": {}}, "474": {"inputs": {"grow_mask_by": 0}}, "474_img": {"inputs": {}}, "474_mask": {"inputs": {}}}
        res = ImageParameterMapper.apply(
            wf,
            task="inpaint",
            inputs={"image_url": "input.png", "mask_url": "mask.png", "mask_blur": 8},
        )
        assert res["474"]["inputs"]["grow_mask_by"] == 8
        assert res["474_img"]["inputs"]["image"] == "input.png"
        assert res["474_mask"]["inputs"]["image"] == "mask.png"

    def test_img08_img09_controlnet_pose_and_depth(self):
        wf = {"ctrl_loader": {"inputs": {}}, "ctrl_img": {"inputs": {}}, "ctrl_apply": {"inputs": {}}}
        # Pose
        res_pose = ImageParameterMapper.apply(
            dict(wf),
            task="controlnet",
            inputs={"control_type": "openpose", "control_image_url": "pose.png", "strength": 0.9},
        )
        assert res_pose["ctrl_loader"]["inputs"]["control_net_name"] == "qwen_image_control_pose.safetensors"
        assert res_pose["ctrl_img"]["inputs"]["image"] == "pose.png"
        assert res_pose["ctrl_apply"]["inputs"]["strength"] == 0.9

        # Depth
        res_depth = ImageParameterMapper.apply(
            dict(wf),
            task="controlnet",
            inputs={"control_type": "depth", "control_image_url": "depth.png", "strength": 0.75},
        )
        assert res_depth["ctrl_loader"]["inputs"]["control_net_name"] == "qwen_image_control_depth.safetensors"


# ==============================================================================
# §6.4 Video Subsystem (Wan 2.2 & MiniMax H3)
# ==============================================================================

class TestVideoSubsystem:
    def test_wan_t2v_and_fl2v_parameter_mapping(self):
        wf = {
            "5": {"class_type": "CLIPTextEncode", "inputs": {}},
            "VHS_VideoCombine": {"class_type": "VHS_VideoCombine", "inputs": {}},
            "EmptyHunyuanLatentVideo": {"class_type": "EmptyHunyuanLatentVideo", "inputs": {}},
            "KSampler": {"class_type": "KSampler", "inputs": {}},
            "9_start": {"inputs": {}},
            "9_end": {"inputs": {}},
        }
        res = VideoParameterMapper.apply(
            wf,
            model="wan-2.2",
            task="first_last_frame_to_video",
            inputs={
                "prompt": "smooth camera pan",
                "fps": 30,
                "first_frame_url": "start.png",
                "last_frame_url": "end.png",
                "seed": 12345,
            },
        )
        assert res["5"]["inputs"]["text"] == "smooth camera pan"
        assert res["VHS_VideoCombine"]["inputs"]["frame_rate"] == 30
        assert res["KSampler"]["inputs"]["seed"] == 12345
        assert res["9_start"]["inputs"]["image"] == "start.png"
        assert res["9_end"]["inputs"]["image"] == "end.png"

    def test_minimax_h3_resolution_multiples_of_32(self):
        wf = {"MiniMaxH3EmptyLatent": {"class_type": "MiniMaxH3EmptyLatent", "inputs": {}}}
        res = VideoParameterMapper.apply(
            wf,
            model="minimax-h3",
            task="text_to_video",
            inputs={"width": 770, "height": 450},  # Not multiples of 32
        )
        w = res["MiniMaxH3EmptyLatent"]["inputs"]["width"]
        h = res["MiniMaxH3EmptyLatent"]["inputs"]["height"]
        assert w % 32 == 0
        assert h % 32 == 0

    def test_minimax_h3_ref2va_binding_to_8_ref(self):
        wf = {"8_ref": {"inputs": {}}}
        res = VideoParameterMapper.apply(
            wf,
            model="minimax-h3",
            task="reference_conditioning",
            inputs={"reference_image_url": "https://s3.ai.example.com/character.png"},
            staged_files={"reference_image_url": "staged_character.png"},
        )
        assert res["8_ref"]["inputs"]["image"] == "staged_character.png"


# ==============================================================================
# §6.5 Enhancement Subsystem (EN-01 to EN-06)
# ==============================================================================

class TestEnhancementSubsystem:
    def test_image_sr_processor_validates_inputs(self, monkeypatch):
        monkeypatch.setenv("IMAGE_SR_BACKEND_URL", "http://localhost:8000")
        proc = ImageSuperResolutionProcessor()

        # Missing image_url
        with pytest.raises(InvalidRequestError):
            proc.process({"scale_factor": 2})

        # Unsupported scale factor
        with pytest.raises(InvalidRequestError):
            proc.process({"image_url": "http://example.com/in.png", "scale_factor": 3})

        # Unsupported model
        with pytest.raises(InvalidRequestError):
            proc.process({"image_url": "http://example.com/in.png", "scale_factor": 2, "model_name": "unsupported-sr"})

    def test_video_sr_processor_validates_inputs(self, monkeypatch):
        monkeypatch.delenv("VIDEO_SR_BACKEND_URL", raising=False)
        proc = VideoSuperResolutionProcessor()
        with pytest.raises(ModelUnavailableError):
            proc.process({"video_url": "http://example.com/in.mp4", "scale_factor": 2})

    def test_frame_interpolation_processor_validates_inputs(self, monkeypatch):
        monkeypatch.delenv("RIFE_BACKEND_URL", raising=False)
        proc = FrameInterpolationProcessor()
        with pytest.raises(ModelUnavailableError):
            proc.process({"video_url": "http://example.com/in.mp4", "multiplier": 2})

    def test_audio_enhancement_processor_validates_inputs(self, monkeypatch):
        monkeypatch.delenv("AUDIO_ENHANCE_BACKEND_URL", raising=False)
        proc = AudioEnhancementProcessor()
        with pytest.raises(ModelUnavailableError):
            proc.process({"audio_url": "http://example.com/in.wav", "mode": "denoise"})
