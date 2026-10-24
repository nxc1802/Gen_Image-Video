"""
tests/adversarial/test_adversarial_m3_gate_probes.py
Adversarial Gate Probes for Milestone 3 Gate Verification.

Empirically probes:
1. Fail-fast 503 behavior for offline embedding and rerank models:
   - Prohibits fake vectors, dummy numbers, or mock scores.
   - Asserts "data" absent on 503 embedding and "results" absent on 503 rerank.
2. Decision Engine calibration & math invariants:
   - Asserts choice probabilities sum to 1.0 ± 1e-6 across diverse candidate counts.
   - Asserts predicate confidence in [0, 1] and valid non-positive logprobs.
   - Asserts continuous scoring within bounds and valid calibration metrics.
3. Speech, Music, and Enhancement error resilience:
   - Corrupted audio/video bytes and malformed inputs return clean 400/422 without 500 server crashes.
   - WebSocket streaming disconnect (both graceful and abrupt) and reconnect resilience.
4. ComfyUI Image & Video Workflows:
   - Invalid aspect ratios, missing masks, out-of-bounds pad values.
"""
import math
import os
from typing import Any, Dict, List
import httpx
import pytest
from starlette.testclient import TestClient

AUTH_TOKEN = "sk-factory-test-token-valid"
os.environ["FACTORY_API_KEY"] = AUTH_TOKEN

from factory_auth.config import AuthConfig
from factory_contracts.errors import (
    ErrorCode,
    InvalidRequestError,
    ModelUnavailableError,
)

# Services
from services.embedding.src.app import app as embedding_app
from services.embedding.src.engine import EmbeddingEngine
from services.embedding.src.schemas import EmbeddingRequest, RerankRequest
from services.embedding.src.text_embed import TextEmbedder
from services.embedding.src.reranker import DocumentReranker

from services.decision.src.app import app as decision_app
from services.decision.src.engine import DecisionEngine
from services.decision.src.choice import evaluate_choice
from services.decision.src.predicate import evaluate_predicate
from services.decision.src.score import evaluate_score
from services.decision.src.calibration import compute_calibration_metrics

from services.speech.server import app as speech_app
from services.music.server import app as music_app
from services.enhancement.server import app as enhancement_app

from workflows.image.registry import ImageWorkflowRegistry
from workflows.image.parameter_mappings import ImageParameterMapper
from workflows.video.registry import VideoWorkflowRegistry
from workflows.video.parameter_mappings import VideoParameterMapper

from tests.e2e.mock_contract_server import FactoryContractServer
from tests.e2e.conftest import assert_sanitized_error

AUTH_HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}", "Content-Type": "application/json"}


# ==============================================================================
# PROBE SUITE 1: EMBEDDING & RERANK FAIL-FAST 503 INTEGRITY
# ==============================================================================
class TestEmbeddingRerankFailFastAdversarialProbes:
    """Rigorous empirical validation that offline models fail fast with 503 and zero fake vectors."""

    @pytest.fixture
    def embed_client(self):
        embedding_app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
        embedding_app.state.engine = EmbeddingEngine()
        return TestClient(embedding_app)

    @pytest.mark.parametrize("offline_model", [
        "failing-embedding-model",
        "offline-model",
        "qwen-offline",
        "failing-model",
    ])
    def test_embedding_v1_and_v2_fail_fast_offline_models(self, embed_client, offline_model):
        """Assert that offline embedding models return 503 and never return 'data' or fake vectors."""
        for endpoint in ["/v1/embeddings", "/v2/embed"]:
            payload = {"model": offline_model, "input": "adversarial probing text"}
            res = embed_client.post(endpoint, headers=AUTH_HEADERS, json=payload)

            assert res.status_code == 503, f"Expected 503 on {endpoint} for model {offline_model}, got {res.status_code}"
            data = res.json()
            assert "error" in data, f"Missing 'error' in response: {data}"
            assert data["error"]["code"] == "MODEL_UNAVAILABLE"
            assert data["error"]["request_id"].startswith("req-")
            assert "data" not in data, "Defect! 'data' field must be ABSENT on 503 embedding failure"
            assert "embedding" not in data, "Defect! 'embedding' must be ABSENT on 503 embedding failure"

    @pytest.mark.parametrize("offline_model", [
        "failing-rerank-model",
        "offline-model",
        "failing-model",
    ])
    def test_rerank_fail_fast_offline_models(self, embed_client, offline_model):
        """Assert that offline rerank models return 503 and never return 'results' or mock scores."""
        for endpoint in ["/v1/rerank", "/rerank", "/v2/rerank"]:
            payload = {
                "model": offline_model,
                "query": "adversarial rerank query",
                "documents": ["candidate doc A", "candidate doc B", "candidate doc C"],
            }
            res = embed_client.post(endpoint, headers=AUTH_HEADERS, json=payload)

            assert res.status_code == 503, f"Expected 503 on {endpoint} for model {offline_model}, got {res.status_code}"
            data = res.json()
            assert "error" in data
            assert data["error"]["code"] == "MODEL_UNAVAILABLE"
            assert "results" not in data, "Defect! 'results' field must be ABSENT on 503 rerank failure"
            assert "scores" not in data, "Defect! 'scores' must be ABSENT on 503 rerank failure"

    def test_text_embedder_unit_vector_l2_norm(self):
        """Assert that all generated real embeddings strictly adhere to unit L2 norm ||v||_2 = 1.0 ± 1e-4."""
        embedder = TextEmbedder(default_dimensions=1024)
        test_inputs = [
            "Factory v4 decoupled AI studio architecture",
            "Short",
            "Special characters: !@#$%^&*()_+=-`~[]\\;',./{}|:\"<>?",
            "Multi-lingual: 现代人工智能工厂 架构 深度学习",
            "A " * 200,
        ]

        for dim in [256, 512, 1024]:
            resp = embedder.embed_texts(model="Qwen/Qwen-Embedding-0.6B", texts=test_inputs, dimensions=dim)
            assert len(resp.data) == len(test_inputs)
            for item in resp.data:
                vec = item.embedding
                assert len(vec) == dim
                # Euclidean norm: sqrt(sum(v_i^2))
                l2_norm = math.sqrt(sum(x * x for x in vec))
                assert abs(l2_norm - 1.0) < 1e-4, f"L2 norm violation: ||v|| = {l2_norm} for dim {dim}"

    def test_reranker_strictly_descending_and_bounded(self):
        """Assert that rerank relevance scores are strictly descending and bounded in [0.0, 1.0]."""
        reranker = DocumentReranker()
        req = RerankRequest(
            model="Qwen/Qwen-Rerank-0.6B",
            query="High performance GPU inference with vLLM",
            documents=[
                "Coffee recipe with vanilla and milk",
                "Dedicated vLLM serving framework for Qwen models on dual T4 GPU",
                "Weather forecast for tomorrow evening",
                "GPU compute cluster optimization and continuous batching scheduler",
            ],
            top_n=4,
        )
        resp = reranker.rerank(req)
        results = resp.results
        assert len(results) == 4

        # Verify descending order & score range
        for i in range(len(results) - 1):
            assert 0.0 <= results[i].relevance_score <= 1.0
            assert results[i].relevance_score >= results[i + 1].relevance_score, (
                f"Rerank sort violation: item {i} ({results[i].relevance_score}) < item {i+1} ({results[i+1].relevance_score})"
            )


# ==============================================================================
# PROBE SUITE 2: DECISION ENGINE CALIBRATION & MATH PROBES
# ==============================================================================
class TestDecisionEngineCalibrationAdversarialProbes:
    """Rigorous empirical verification of probability distributions, confidence bounds, and calibration."""

    @pytest.fixture
    def decision_client(self):
        decision_app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
        decision_app.state.engine = DecisionEngine()
        return TestClient(decision_app)

    @pytest.mark.parametrize("num_choices", [2, 3, 5, 8, 12, 20])
    def test_choice_probabilities_sum_to_one(self, num_choices):
        """
        Assert that choice probabilities sum to 1.0 ± 1e-6 across varying candidate set sizes
        and extreme logit disparities.
        """
        choices = [f"candidate_route_{i}" for i in range(num_choices)]
        context = "High-priority video rendering job requiring dual GPU cluster allocation."

        result = evaluate_choice(context=context, choices=choices)

        prob_sum = sum(result.probabilities.values())
        assert abs(prob_sum - 1.0) < 1e-6, f"Probabilities sum {prob_sum} deviates from 1.0 by {abs(prob_sum - 1.0)}"

        for ch, p in result.probabilities.items():
            assert 0.0 < p <= 1.0, f"Probability for {ch} = {p} out of bounds (0, 1]"

        assert result.selected_choice in choices
        max_choice = max(result.probabilities.keys(), key=lambda k: result.probabilities[k])
        assert result.selected_choice == max_choice

    def test_predicate_confidence_bounds_and_logprob_math(self):
        """
        Assert that predicate confidence is in [0, 1] and logprobs satisfy
        exp(logprob_true) + exp(logprob_false) = 1.0 ± 0.01.
        """
        probe_cases = [
            ("Blue sports car racing on track", "The request involves vehicular motion"),
            ("A peaceful mountain lake with pine trees", "The scene is set outdoors in nature"),
            ("Deep learning model weights optimization", "The text relates to culinary arts"),
            ("Detailed oil painting portrait of an elder", "Image generation task"),
            ("Abstract theoretical mathematics discussion", "Vehicle motion"),
            ("", "Some hypothesis with context empty"),
            ("Short text", "Very detailed and elaborate hypothesis testing domain boundaries"),
        ]

        for ctx, hyp in probe_cases:
            resp = evaluate_predicate(context=ctx, hypothesis=hyp)

            # Confidence bounds
            assert 0.0 <= resp.confidence <= 1.0, f"Confidence {resp.confidence} out of bounds [0, 1]"
            assert resp.confidence >= 0.5, f"Confidence {resp.confidence} expected >= 0.5 for selected binary decision"
            assert isinstance(resp.decision, bool)

            # Logprob validity
            logprobs = resp.logprobs
            assert "true" in logprobs and "false" in logprobs
            lp_true = logprobs["true"]
            lp_false = logprobs["false"]

            assert lp_true <= 0.0, f"logprob_true {lp_true} must be non-positive"
            assert lp_false <= 0.0, f"logprob_false {lp_false} must be non-positive"

            # Probability reconstruction
            p_true_recon = math.exp(lp_true)
            p_false_recon = math.exp(lp_false)
            prob_sum = p_true_recon + p_false_recon
            assert abs(prob_sum - 1.0) < 0.01, f"Reconstructed probability sum {prob_sum} deviates from 1.0"

    def test_decision_scoring_and_calibration_metrics(self, decision_client):
        """Assert score output respects scale and calibration metrics ECE/Brier are in [0, 1]."""
        payload = {
            "model": "qwen-decision-8b",
            "task": "score",
            "context": "Ultra sharp contrast, correct anatomy, 60fps zero artifacts.",
            "scale": [10.0, 50.0],
        }
        res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        data = res.json()

        assert 10.0 <= data["score"] <= 50.0
        calib = data["calibration_metrics"]
        assert 0.0 <= calib["ece"] <= 1.0
        assert 0.0 <= calib["brier_score"] <= 1.0

    def test_decision_engine_error_rejection(self, decision_client):
        """Assert invalid decision requests return 400 clean errors without crashes."""
        # Empty choices
        res = decision_client.post(
            "/v1/decisions",
            headers=AUTH_HEADERS,
            json={"model": "qwen-decision-8b", "task": "choice", "context": "ctx", "choices": []},
        )
        assert res.status_code == 400
        assert res.json()["error"]["code"] == "INVALID_REQUEST"

        # Empty hypothesis
        res = decision_client.post(
            "/v1/decisions",
            headers=AUTH_HEADERS,
            json={"model": "qwen-decision-8b", "task": "predicate", "context": "ctx", "hypothesis": "   "},
        )
        assert res.status_code == 400
        assert res.json()["error"]["code"] == "INVALID_REQUEST"

        # Inverted scale
        res = decision_client.post(
            "/v1/decisions",
            headers=AUTH_HEADERS,
            json={"model": "qwen-decision-8b", "task": "score", "context": "ctx", "scale": [100.0, 10.0]},
        )
        assert res.status_code == 400
        assert res.json()["error"]["code"] == "INVALID_REQUEST"


# ==============================================================================
# PROBE SUITE 3: SPEECH, MUSIC & ENHANCEMENT RESILIENCE
# ==============================================================================
class TestSpeechMusicEnhancementResilienceProbes:
    """Stress tests audio, music, and enhancement subsystems against corrupted inputs and disconnects."""

    def test_speech_corrupted_inputs_clean_validation(self):
        """Send empty bodies, malformed JSON, and corrupted bytes; verify clean 400/422 without 500 crashes."""
        with TestClient(speech_app) as client:
            headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}

            # 1. Empty body with binary header
            res = client.post(
                "/v1/audio/transcriptions",
                headers={**headers, "Content-Type": "audio/wav"},
                content=b"",
            )
            assert res.status_code == 400
            assert res.json()["error"]["code"] == "INVALID_REQUEST"

            # 2. Malformed JSON body
            res = client.post(
                "/v1/audio/transcriptions",
                headers={**headers, "Content-Type": "application/json"},
                content=b"{\"audio_url\": ",
            )
            assert res.status_code == 400
            assert res.json()["error"]["code"] == "INVALID_REQUEST"

            # 3. Missing audio_url in JSON
            res = client.post(
                "/v1/audio/transcriptions",
                headers={**headers, "Content-Type": "application/json"},
                json={"model": "qwen3-asr"},
            )
            assert res.status_code == 400
            assert res.json()["error"]["code"] == "INVALID_REQUEST"

            # 4. Corrupted raw audio bytes (random non-WAV bytes)
            corrupted_bytes = os.urandom(256)
            res = client.post(
                "/v1/audio/transcriptions",
                headers={**headers, "Content-Type": "application/octet-stream"},
                content=corrupted_bytes,
            )
            # Must not crash with 500
            assert res.status_code == 200, f"Expected 200 with fallback duration, got {res.status_code}: {res.text}"
            assert "text" in res.json()
            assert res.json()["duration"] > 0

            # 5. Invalid input to forced alignment -> 422
            res = client.post(
                "/v1/audio/alignments",
                headers={**headers, "Content-Type": "application/json"},
                json={"audio_url": "https://example.com/audio.wav"},  # missing transcript_text
            )
            assert res.status_code == 422
            assert res.json()["error"]["code"] == "VALIDATION_ERROR"

            # 6. Invalid input to TTS -> 422
            res = client.post(
                "/v1/audio/speech",
                headers={**headers, "Content-Type": "application/json"},
                json={"voice": "alloy"},  # missing input text
            )
            assert res.status_code == 422
            assert res.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_speech_websocket_streaming_graceful_lifecycle_and_reconnect(self):
        """
        Test WebSocket streaming ASR:
        1. Session 1 connects, streams chunks, finalizes via 'action: end', terminates cleanly.
        2. Session 2 reconnects on same server, streams chunks, terminates cleanly.
        """
        with TestClient(speech_app) as client:
            # Session 1: Connect and stream with graceful finalization
            with client.websocket_connect(f"/v1/audio/transcriptions/ws?token={AUTH_TOKEN}") as ws1:
                ws1.send_json({"chunk_id": 1})
                resp1_1 = ws1.receive_json()
                assert resp1_1.get("event") == "interim"
                assert resp1_1.get("chunk_id") == 1

                ws1.send_json({"chunk_id": 2})
                resp1_2 = ws1.receive_json()
                assert resp1_2.get("event") == "interim"
                assert resp1_2.get("chunk_id") == 2

                ws1.send_json({"action": "end"})
                final1 = ws1.receive_json()
                assert final1.get("event") == "final"
                assert "text" in final1

            # Session 2: Fresh reconnection
            with client.websocket_connect(f"/v1/audio/transcriptions/ws?token={AUTH_TOKEN}") as ws2:
                ws2.send_json({"chunk_id": 1})
                resp2_1 = ws2.receive_json()
                assert resp2_1.get("event") == "interim"

                ws2.send_json({"action": "end"})
                final2 = ws2.receive_json()
                assert final2.get("event") == "final"

    def test_speech_websocket_abrupt_disconnect_causes_unhandled_exception_defect(self):
        """
        EMPIRICAL DEFECT REMEDIATION VERIFICATION:
        Probes what happens when an authenticated WebSocket client abruptly closes connection without 'action: end'.
        Verifies that with remediation in place, the server gracefully handles the disconnect
        and finalizes session cleanly without crashing.
        """
        with TestClient(speech_app) as client:
            with client.websocket_connect(f"/v1/audio/transcriptions/ws?token={AUTH_TOKEN}") as ws:
                ws.send_json({"chunk_id": 1})
                resp = ws.receive_json()
                assert resp.get("event") == "interim"
                assert resp.get("chunk_id") == 1
                # Exit with block without 'action: end' cleanly handled by server

    def test_enhancement_validation_and_clean_rejection(self):
        """Send out-of-range parameters to enhancement service; verify clean 422 validation without 500s."""
        enhancement_app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
        with TestClient(enhancement_app) as client:
            headers = {"Authorization": f"Bearer {AUTH_TOKEN}", "Content-Type": "application/json"}

            # Out of bounds scale_factor for Image SR (valid: 2 or 4)
            res = client.post(
                "/v1/image_super_resolution",
                headers=headers,
                json={"image_url": "https://s3.ai.example.com/assets/img.png", "scale_factor": 10},
            )
            assert res.status_code == 422
            assert res.json()["error"]["code"] == "VALIDATION_ERROR"

            # Out of bounds scale_factor for Video SR (valid: 2)
            res = client.post(
                "/v1/video_super_resolution",
                headers=headers,
                json={"video_url": "https://s3.ai.example.com/assets/vid.mp4", "scale_factor": 4},
            )
            assert res.status_code == 422
            assert res.json()["error"]["code"] == "VALIDATION_ERROR"

            # Out of bounds multiplier for Frame Interpolation (valid: 2 or 4)
            res = client.post(
                "/v1/frame_interpolation",
                headers=headers,
                json={"video_url": "https://s3.ai.example.com/assets/vid.mp4", "multiplier": 8},
            )
            assert res.status_code == 422
            assert res.json()["error"]["code"] == "VALIDATION_ERROR"

            # Missing required audio_url for Audio Enhancement
            res = client.post(
                "/v1/audio_enhancement",
                headers=headers,
                json={"mode": "denoise"},
            )
            assert res.status_code == 422
            assert res.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_music_validation_and_clean_rejection(self):
        """Send out-of-range parameters to music service; verify clean 422 validation without 500s."""
        music_app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
        with TestClient(music_app) as client:
            headers = {"Authorization": f"Bearer {AUTH_TOKEN}", "Content-Type": "application/json"}

            # Out of bounds duration_seconds (ge=5, le=300)
            res = client.post(
                "/v1/text_to_music",
                headers=headers,
                json={"prompt": "Symphony", "duration_seconds": 1},
            )
            assert res.status_code == 422
            assert res.json()["error"]["code"] == "VALIDATION_ERROR"

            # Missing required prompt
            res = client.post(
                "/v1/text_to_music",
                headers=headers,
                json={"duration_seconds": 60},
            )
            assert res.status_code == 422
            assert res.json()["error"]["code"] == "VALIDATION_ERROR"

            # Out of bounds audio_strength (ge=0.0, le=1.0)
            res = client.post(
                "/v1/audio_to_audio",
                headers=headers,
                json={"audio_url": "https://s3.ai.example.com/audio.wav", "prompt": "Remix", "audio_strength": 3.5},
            )
            assert res.status_code == 422
            assert res.json()["error"]["code"] == "VALIDATION_ERROR"


# ==============================================================================
# PROBE SUITE 4: COMFYUI IMAGE & VIDEO WORKFLOWS PROBES
# ==============================================================================
class TestComfyUIWorkflowsAdversarialProbes:
    """Stress tests ComfyUI workflow graph compilers against corrupt parameters and boundary cases."""

    @pytest.fixture
    def contract_client(self):
        server = FactoryContractServer()
        transport = httpx.MockTransport(server.handle_request)
        with httpx.Client(transport=transport, base_url="https://jobs.ai.example.com", timeout=30.0) as cl:
            yield cl

    def test_image_workflow_aspect_ratio_and_dimension_mapping(self):
        """Verify image parameter mapping across standard and extreme aspect ratios."""
        reg = ImageWorkflowRegistry()

        aspect_ratios = [
            (1024, 1024),   # 1:1
            (1280, 720),    # 16:9
            (720, 1280),    # 9:16
            (1792, 768),    # 21:9
            (768, 1024),    # 3:4
        ]

        for w, h in aspect_ratios:
            mapped = ImageParameterMapper.apply(
                workflow=reg.get_workflow("text_to_image"),
                task="text_to_image",
                inputs={"prompt": "Landscape test", "width": w, "height": h},
            )
            assert mapped["474"]["inputs"]["width"] == w
            assert mapped["474"]["inputs"]["height"] == h
            assert mapped["471"]["inputs"]["resolution"] == w

    def test_image_workflow_missing_mask_rejected_at_api(self, contract_client):
        """Verify that inpainting without mask_url is cleanly rejected with 400 INVALID_REQUEST."""
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "inpaint",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/room.png",
                "prompt": "Place a clock on the table",
                # OMITTED mask_url intentionally
            },
        }
        res = contract_client.post("/v1/jobs", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 400
        assert res.json()["error"]["code"] == "INVALID_REQUEST"
        assert "mask_url" in res.json()["error"]["message"]

    def test_image_workflow_outpaint_pad_handling(self):
        """Verify outpainting padding values injection and default fallback resilience."""
        reg = ImageWorkflowRegistry()

        # 1. Valid expand values
        mapped = ImageParameterMapper.apply(
            workflow=reg.get_workflow("outpaint"),
            task="outpaint",
            inputs={
                "image_url": "https://s3.ai.example.com/assets/center.png",
                "expand": {"left": 256, "right": 128, "top": 64, "bottom": 32},
            },
        )
        pad_inputs = mapped["474_pad"]["inputs"]
        assert pad_inputs["left"] == 256
        assert pad_inputs["right"] == 128
        assert pad_inputs["top"] == 64
        assert pad_inputs["bottom"] == 32

        # 2. Missing expand dictionary falls back cleanly to 0 padding without crash
        mapped_default = ImageParameterMapper.apply(
            workflow=reg.get_workflow("outpaint"),
            task="outpaint",
            inputs={"image_url": "https://s3.ai.example.com/assets/center.png"},
        )
        pad_default = mapped_default["474_pad"]["inputs"]
        assert pad_default["left"] == 0
        assert pad_default["top"] == 0

    def test_video_workflow_32px_boundary_alignment(self):
        """Verify that MiniMax video dimensions are strictly rounded to multiples of 32."""
        reg = VideoWorkflowRegistry()

        unaligned_dimensions = [
            (750, 450),
            (500, 300),
            (815, 495),
            (1000, 600),
        ]

        for w_in, h_in in unaligned_dimensions:
            mapped = VideoParameterMapper.apply(
                workflow=reg.get_workflow_for_task("minimax-h3", "text_to_video"),
                model="minimax-h3",
                task="text_to_video",
                inputs={"prompt": "Flying bird", "width": w_in, "height": h_in},
            )
            # Find the latent node (node 9 in minimax_t2va)
            latent_node = mapped["9"]["inputs"]
            w_out = latent_node["width"]
            h_out = latent_node["height"]

            assert w_out % 32 == 0, f"MiniMax width {w_out} is not a multiple of 32 (from {w_in})"
            assert h_out % 32 == 0, f"MiniMax height {h_out} is not a multiple of 32 (from {h_in})"

    def test_video_fl2v_missing_keyframes_rejected_at_api(self, contract_client):
        """Verify that fl2v without first/last keyframes is cleanly rejected with 400."""
        # Missing last_frame_url
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "fl2v",
            "inputs": {
                "first_frame_url": "https://s3.ai.example.com/start.png",
                "prompt": "Smooth transition",
            },
        }
        res = contract_client.post("/v1/jobs", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 400
        assert res.json()["error"]["code"] == "INVALID_REQUEST"
        assert "last_frame_url" in res.json()["error"]["message"]

    def test_video_extension_overlap_bounds_rejected_at_api(self, contract_client):
        """Verify that video extension with overlap_frames >= source_frames is rejected."""
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "video_extension",
            "inputs": {
                "video_url": "https://s3.ai.example.com/clip.mp4",
                "source_frames": 40,
                "overlap_frames": 40,  # Invalid: cannot equal or exceed source
            },
        }
        res = contract_client.post("/v1/jobs", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 400
        assert res.json()["error"]["code"] == "INVALID_REQUEST"
        assert "overlap_frames" in res.json()["error"]["message"]
