"""
tests/adversarial/test_adversarial_m5_ai_subsystems.py
Empirical Adversarial Test Suite for Milestone 5 Phase 2:
Adversarial Coverage Hardening (Tier 5) on AI Subsystems & Studio Web.

Empirically probes and verifies:
1. Streaming Invariants:
   - Live SSE token streaming for Text/VLM: malformed UTF-8, multibyte sequences, empty/whitespace tokens, client connection aborts, and 503 fail-fast on offline models.
   - WebSocket audio streaming for Speech ASR & TTS: out-of-order chunks, abrupt client disconnects, mandatory Bearer auth enforcement, and RIFF WAV container header invariants.
   - ComfyUI step progress streaming: out-of-order step progress, division-by-zero defense, binary preview frame handling, execution errors, and cancellation propagation.
2. Mathematical & Algorithmic Invariants:
   - Vector embeddings: strict unit L2 norm ||v|| = 1.0 ± 1e-4 across diverse and random inputs, Matryoshka dimension truncation (1024 down to 16), zero sine/cosine mock vectors, authentic semantic dispersion, and 503 fail-fast on offline models.
   - Decision Engine: Softmax normalization sum(p_i) = 1.0 ± 1e-6 across candidate set sizes, extreme logit stability, deterministic tie-breaking, calibrated confidence in [0.5, 1.0], and calibration metrics (ECE, Brier) in [0.0, 1.0].
   - Okapi BM25 document reranker: monotonic descending rank ordering, non-negative relevance scores in [0.01, 0.99], query-match prioritization, and top_n truncation.
3. Security & SSRF Boundaries:
   - Internal and external SSRF probes against Asset Service, Speech, Music, Enhancement, and Text/VLM across loopback, private RFC1918, cloud metadata, ULA IPv6, scheme violations, and disallowed ports.
   - Authentication perimeter: 100% of protected endpoints across all subsystems reject unauthenticated requests with HTTP 401.
4. Studio Web Dynamic Integrity:
   - Verification of zero hardcoded model strings/dropdowns and zero proxy fallbacks (x-target-base-url).
   - Clean TypeScript typecheck (npx tsc --noEmit) and clean Next.js 15 production build (npm run build).
"""

import asyncio
import io
import json
import math
import os
import re
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

# ---------------------------------------------------------------------------
# Workspace Paths & Environment Bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in [
    REPO_ROOT,
    REPO_ROOT / "packages" / "contracts" / "src",
    REPO_ROOT / "packages" / "auth" / "src",
    REPO_ROOT / "packages" / "model-registry" / "src",
]:
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

AUTH_TOKEN = "sk-factory-test-token-valid"
AUTH_HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json",
}

os.environ["FACTORY_API_KEY"] = AUTH_TOKEN
os.environ["STORAGE_BACKEND"] = "local"
os.environ["FACTORY_DB_BACKEND"] = "sqlite"
os.environ["SQLITE_DB_PATH"] = ":memory:"

from factory_auth.config import AuthConfig
from factory_contracts.errors import (
    InvalidRequestError,
    ModelUnavailableError,
    SSRFBlockedError,
    UnauthorizedError,
)

# AI Subsystem Services
from services.text.src.app import app as text_app
from services.text.src.engine import TextEngine, OFFLINE_MODELS as OFFLINE_TEXT_MODELS
from services.text.src.models import ChatCompletionRequest, ChatMessage
from services.text.src.multimodal import validate_url_security, extract_content_text_and_media

from services.vlm.src.app import app as vlm_app
from services.vlm.src.engine import VLMEngine

from services.decision.src.app import app as decision_app
from services.decision.src.choice import evaluate_choice
from services.decision.src.predicate import evaluate_predicate
from services.decision.src.score import evaluate_score
from services.decision.src.calibration import compute_calibration_metrics, evaluate_score_calibration

from services.embedding.src.app import app as embedding_app
from services.embedding.src.text_embed import TextEmbedder, OFFLINE_EMBEDDING_MODELS
from services.embedding.src.reranker import DocumentReranker, OFFLINE_RERANK_MODELS
from services.embedding.src.schemas import RerankRequest

from services.speech.server import app as speech_app
from services.speech.asr.streaming import StreamingASRBuffer
from services.speech.tts.engine import TTSEngine

from services.music.server import app as music_app
from services.enhancement.server import app as enhancement_app
from factory.apps.control_api.src.assets.ssrf import validate_target_url
from factory.apps.control_api.src.main import app as control_api_app
from services.image.client import ComfyUIClient as ImageComfyClient
import services.image.client as image_client_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def configured_auth():
    return AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")


@pytest.fixture(scope="module")
def text_client(configured_auth):
    text_app.state.auth_config = configured_auth
    text_app.state.engine = TextEngine()
    return TestClient(text_app)


@pytest.fixture(scope="module")
def vlm_client(configured_auth):
    vlm_app.state.auth_config = configured_auth
    vlm_app.state.engine = VLMEngine()
    return TestClient(vlm_app)


@pytest.fixture(scope="module")
def decision_client(configured_auth):
    decision_app.state.auth_config = configured_auth
    return TestClient(decision_app)


@pytest.fixture(scope="module")
def embedding_client(configured_auth):
    embedding_app.state.auth_config = configured_auth
    return TestClient(embedding_app)


@pytest.fixture(scope="module")
def speech_client(configured_auth):
    speech_app.state.auth_config = configured_auth
    return TestClient(speech_app)


@pytest.fixture(scope="module")
def music_client(configured_auth):
    music_app.state.auth_config = configured_auth
    return TestClient(music_app)


@pytest.fixture(scope="module")
def enhancement_client(configured_auth):
    enhancement_app.state.auth_config = configured_auth
    return TestClient(enhancement_app)


@pytest.fixture(scope="module")
def control_client(configured_auth):
    control_api_app.state.auth_config = configured_auth
    return TestClient(control_api_app)


# ===========================================================================
# 1. STREAMING INVARIANTS
# ===========================================================================
@pytest.mark.streaming
class TestM5StreamingInvariants:
    """Empirical verification of SSE, WebSocket, and ComfyUI progress streaming invariants."""

    @pytest.mark.asyncio
    async def test_sse_token_streaming_valid_lifecycle(self):
        """Verifies initial role chunk, incremental deltas, and [DONE] termination."""
        engine = TextEngine()
        req = ChatCompletionRequest(
            model="qwen-2.5-72b-instruct",
            messages=[ChatMessage(role="user", content="Explain two-tier queue architecture.")],
            stream=True,
        )
        chunks = [c async for c in engine.stream_generate(req)]
        assert len(chunks) >= 3, f"Expected at least 3 SSE chunks, got {len(chunks)}"

        # Initial chunk: role delta
        assert chunks[0].startswith("data: ")
        first_payload = json.loads(chunks[0][6:].strip())
        assert first_payload["choices"][0]["delta"].get("role") == "assistant"

        # Intermediate chunks: content deltas
        middle_payload = json.loads(chunks[1][6:].strip())
        assert "content" in middle_payload["choices"][0]["delta"]

        # Final chunk: [DONE]
        assert chunks[-1].strip() == "data: [DONE]"

    @pytest.mark.asyncio
    async def test_sse_token_streaming_malformed_utf8_and_multibyte(self):
        """Verifies SSE generator correctly handles multibyte unicode, emoji, and zero-width chars."""
        engine = TextEngine()
        adversarial_text = (
            "Prompt with emojis 🚀🔥🤖, zero-width spaces \u200b\u200c\u200d, "
            "RTL override \u202e reversed \u202c, and accents: é, è, ê, ë, ñ, ç, "
            "CJK: 深度学习架构, Arabic: ذكاء اصطناعي"
        )
        req = ChatCompletionRequest(
            model="qwen-2.5-72b-instruct",
            messages=[ChatMessage(role="user", content=adversarial_text)],
            stream=True,
        )
        chunks = [c async for c in engine.stream_generate(req)]
        assert len(chunks) >= 3
        assert chunks[-1].strip() == "data: [DONE]"

        # Ensure all chunks parse as valid JSON (excluding [DONE])
        for c in chunks[:-1]:
            assert c.startswith("data: ")
            parsed = json.loads(c[6:].strip())
            assert "choices" in parsed

    @pytest.mark.asyncio
    async def test_sse_token_streaming_whitespace_and_empty_tokens(self):
        """Verifies SSE generator does not emit null or malformed chunks when prompt has extreme whitespace."""
        engine = TextEngine()
        req = ChatCompletionRequest(
            model="qwen-2.5-72b-instruct",
            messages=[ChatMessage(role="user", content="   multiple   spaces   \t\n\r   between   words   ")],
            stream=True,
        )
        chunks = [c async for c in engine.stream_generate(req)]
        for c in chunks[:-1]:
            parsed = json.loads(c[6:].strip())
            delta = parsed["choices"][0]["delta"]
            if "content" in delta:
                assert delta["content"] is not None

    @pytest.mark.asyncio
    async def test_sse_token_streaming_client_abort_resilience(self):
        """Verifies generator can be closed mid-stream (client disconnect) without unhandled exceptions."""
        engine = TextEngine()
        req = ChatCompletionRequest(
            model="qwen-2.5-72b-instruct",
            messages=[ChatMessage(role="user", content="Long generation request for client abort test.")],
            stream=True,
        )
        gen = engine.stream_generate(req)
        received = 0
        async for _ in gen:
            received += 1
            if received == 2:
                # Simulate abrupt client disconnect
                await gen.aclose()
                break
        assert received == 2

    def test_sse_streaming_offline_model_fail_fast(self, text_client):
        """Verifies requesting offline model via stream endpoint fails fast with 503 rather than starting stream."""
        payload = {
            "model": "offline-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        resp = text_client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
        assert resp.status_code == 503
        err = resp.json()["error"]
        assert err["code"] == "MODEL_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_vlm_sse_token_streaming_multimodal(self):
        """Verifies VLM SSE streaming operates correctly with multimodal image conditioning."""
        engine = VLMEngine()
        req = ChatCompletionRequest(
            model="qwen-vl-max",
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "Describe system architecture diagram."},
                        {"type": "image_url", "image_url": {"url": "https://assets.ai.example.com/diagram.png"}},
                    ],
                )
            ],
            stream=True,
        )
        chunks = [c async for c in engine.stream_generate(req)]
        assert len(chunks) >= 3
        first_payload = json.loads(chunks[0][6:].strip())
        assert first_payload["choices"][0]["delta"].get("role") == "assistant"
        assert chunks[-1].strip() == "data: [DONE]"

    def test_speech_ws_audio_streaming_lifecycle(self, speech_client):
        """Verifies Speech WebSocket streaming session lifecycle: connect -> interim frames -> finalize."""
        with speech_client.websocket_connect(f"/v1/audio/transcriptions/ws?token={AUTH_TOKEN}") as ws:
            # Send text frames with chunk IDs
            ws.send_text(json.dumps({"action": "stream", "chunk_id": 1}))
            event1 = ws.receive_json()
            assert event1["event"] == "interim"
            assert event1["chunk_id"] == 1
            assert event1["is_final"] is False
            assert "interim_text" in event1

            ws.send_text(json.dumps({"action": "stream", "chunk_id": 2}))
            event2 = ws.receive_json()
            assert event2["event"] == "interim"
            assert event2["chunk_id"] == 2

            # Finalize stream
            ws.send_text(json.dumps({"action": "end"}))
            final_event = ws.receive_json()
            assert final_event["event"] == "final"
            assert final_event["is_final"] is True
            assert "text" in final_event
            assert len(final_event["segments"]) > 0

    def test_speech_ws_audio_streaming_out_of_order_chunks(self, speech_client):
        """Verifies Speech WebSocket gracefully handles out-of-order, non-sequential, and negative chunk IDs."""
        with speech_client.websocket_connect(f"/v1/audio/transcriptions/ws?token={AUTH_TOKEN}") as ws:
            # Out of order chunks: 5, then 2, then 10, then -1, then 0
            for cid in [5, 2, 10, -1, 0]:
                ws.send_text(json.dumps({"chunk_id": cid}))
                event = ws.receive_json()
                assert event["event"] == "interim"
                assert event["chunk_id"] == cid
                assert isinstance(event["interim_text"], str)

            ws.send_text(json.dumps({"action": "end"}))
            final_event = ws.receive_json()
            assert final_event["is_final"] is True

    def test_speech_ws_audio_streaming_abrupt_disconnect(self, speech_client):
        """Verifies Speech WebSocket handles client disconnecting abruptly without action='end'."""
        with speech_client.websocket_connect(f"/v1/audio/transcriptions/ws?token={AUTH_TOKEN}") as ws:
            ws.send_text(json.dumps({"chunk_id": 1}))
            event = ws.receive_json()
            assert event["event"] == "interim"
            # Abrupt close without finalization message

    def test_speech_ws_auth_enforcement_missing_token(self, speech_client):
        """Verifies unauthenticated WebSocket connection is immediately rejected with close code 1008."""
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with speech_client.websocket_connect("/v1/audio/transcriptions/ws") as ws:
                ws.receive_json()
        assert exc_info.value.code == 1008

    def test_speech_ws_auth_enforcement_invalid_token(self, speech_client):
        """Verifies invalid token WebSocket connection is rejected with close code 1008."""
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with speech_client.websocket_connect("/v1/audio/transcriptions/ws?token=invalid-key") as ws:
                ws.receive_json()
        assert exc_info.value.code == 1008

    def test_speech_tts_riff_wav_header_invariants(self):
        """Verifies TTS output conforms to genuine 16-bit PCM RIFF WAV format invariants."""
        engine = TTSEngine(sample_rate=24000)
        wav_bytes = engine.synthesize(
            text="Testing RIFF WAV container invariants for Speech Subsystem.",
            voice="alloy",
            speed=1.0,
        )
        assert len(wav_bytes) > 44, "WAV byte stream is shorter than standard header"
        assert wav_bytes[:4] == b"RIFF", "Missing RIFF magic signature"
        assert wav_bytes[8:12] == b"WAVE", "Missing WAVE format specifier"
        assert wav_bytes[12:16] == b"fmt ", "Missing fmt subchunk marker"

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getnchannels() == 1, "Expected mono channel"
            assert wf.getsampwidth() == 2, "Expected 16-bit audio (2 bytes/sample)"
            assert wf.getframerate() == 24000, "Expected 24000 Hz sample rate"
            assert wf.getnframes() > 0, "Zero audio frames generated"

    def test_speech_tts_duration_scaling_with_speed(self):
        """Verifies synthesized audio frame count scales inversely proportional to speed parameter."""
        engine = TTSEngine(sample_rate=24000)
        text = "Harmonic acoustic speech generation benchmark."
        wav_slow = engine.synthesize(text, speed=0.5)
        wav_norm = engine.synthesize(text, speed=1.0)
        wav_fast = engine.synthesize(text, speed=2.0)

        with wave.open(io.BytesIO(wav_slow), "rb") as wf:
            frames_slow = wf.getnframes()
        with wave.open(io.BytesIO(wav_norm), "rb") as wf:
            frames_norm = wf.getnframes()
        with wave.open(io.BytesIO(wav_fast), "rb") as wf:
            frames_fast = wf.getnframes()

        assert frames_slow > frames_norm > frames_fast
        assert abs(frames_slow - (frames_norm * 2)) < 50
        assert abs(frames_fast - (frames_norm / 2)) < 50

    def test_speech_tts_empty_input_rejection(self):
        """Verifies TTS rejects empty and whitespace-only text inputs."""
        engine = TTSEngine()
        with pytest.raises(InvalidRequestError):
            engine.synthesize(text="")
        with pytest.raises(InvalidRequestError):
            engine.synthesize(text="     \t\n  ")

    def test_speech_tts_unsupported_voice_rejection(self):
        """Verifies TTS rejects non-existent voice preset."""
        engine = TTSEngine()
        with pytest.raises(InvalidRequestError):
            engine.synthesize(text="Valid text", voice="non_existent_preset")

    @pytest.mark.asyncio
    async def test_comfyui_progress_streaming_edge_cases(self):
        """Verifies ComfyUIClient progress listener handles division by zero (max=0), out-of-order steps, and preview frames."""
        from unittest.mock import patch
        client = ImageComfyClient()
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[
            json.dumps({"type": "progress", "data": {"value": 5, "max": 20}}),
            b"BINARY_PREVIEW_JPEG_DATA_FRAME",
            json.dumps({"type": "progress", "data": {"value": 15, "max": 10}}),  # value > max
            json.dumps({"type": "progress", "data": {"value": 1, "max": 0}}),    # max = 0 (division by zero defense)
            json.dumps({"type": "executed", "data": {"prompt_id": "p-123", "node": "9", "output": {"images": ["out.png"]}}}),
            json.dumps({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 0}}}}),
        ])

        mock_websockets = MagicMock()
        mock_connect = MagicMock()
        mock_connect.__aenter__.return_value = mock_ws
        mock_connect.__aexit__.return_value = None
        mock_websockets.connect.return_value = mock_connect
        mock_websockets.ConnectionClosed = Exception

        image_client_module.websockets = mock_websockets

        progress_events = []
        def on_prog(step, total, pct, msg):
            progress_events.append((step, total, pct))

        # Mock get_history to avoid blocking in fallback loop
        client.get_history = AsyncMock(return_value={"p-123": {"outputs": {"9": {"images": ["out.png"]}}}})

        with patch("asyncio.sleep", new_callable=AsyncMock):
            outputs = await client.listen_execution("p-123", "client-test", on_progress=on_prog)
        assert "9" in outputs
        assert len(progress_events) == 3
        # First event: 5/20 -> 25.0%
        assert progress_events[0] == (5, 20, 25.0)
        # Third event: max=0 -> clamped by max(1, total) -> 100.0%
        assert progress_events[2][1] == 0
        assert progress_events[2][2] == 100.0

    @pytest.mark.asyncio
    async def test_comfyui_progress_streaming_execution_error(self):
        """Verifies ComfyUI execution_error triggers clean RuntimeError."""
        from unittest.mock import patch
        client = ImageComfyClient()
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[
            json.dumps({"type": "execution_error", "data": {"prompt_id": "p-err", "exception_message": "CUDA OOM in KSampler"}}),
        ])

        mock_websockets = MagicMock()
        mock_connect = MagicMock()
        mock_connect.__aenter__.return_value = mock_ws
        mock_connect.__aexit__.return_value = None
        mock_websockets.connect.return_value = mock_connect
        mock_websockets.ConnectionClosed = Exception

        image_client_module.websockets = mock_websockets

        # Notice: Currently listen_execution catches RuntimeError in except Exception and enters fallback polling
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError) as exc_info:
                await client.listen_execution("p-err", "client-err")
            assert "CUDA OOM in KSampler" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_comfyui_progress_streaming_cancellation(self):
        """Verifies setting cancel_event triggers interrupt() and raises CancelledError."""
        from unittest.mock import patch
        client = ImageComfyClient()
        client.interrupt = AsyncMock(return_value=True)

        cancel_ev = asyncio.Event()
        cancel_ev.set()  # Cancelled before receiving

        mock_ws = AsyncMock()
        mock_websockets = MagicMock()
        mock_connect = MagicMock()
        mock_connect.__aenter__.return_value = mock_ws
        mock_connect.__aexit__.return_value = None
        mock_websockets.connect.return_value = mock_connect

        image_client_module.websockets = mock_websockets

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(asyncio.CancelledError):
                await client.listen_execution("p-cancel", "client-cancel", cancel_event=cancel_ev)
        client.interrupt.assert_called_once()


# ===========================================================================
# 2. MATHEMATICAL & ALGORITHMIC INVARIANTS
# ===========================================================================
class TestM5MathematicalAndAlgorithmicInvariants:
    """Empirical verification of vector L2 norms, Softmax distributions, and BM25 ranking."""

    @pytest.mark.parametrize("text_input", [
        "Quantum field theory and general relativity unification.",
        "Deterministic finite state automaton transition matrices.",
        "Short",
        "A" * 1000,
        "   Whitespace padded text with internal symbols !@#$%^&*()   ",
        "Unicode: 现代分布式计算 深度学习推理引擎 向量相似度检索",
        "Arabic: محرك الاستدلال العصبي ومعالجة اللغة الطبيعية",
        "Emoji: 🚀⚡🤖🔥📦💡🎨🎵",
        "1234567890 9876543210",
        "Stop words only: the a an is are was were in on of for to and",
    ])
    def test_vector_embeddings_l2_unit_norm_diverse_inputs(self, text_input):
        """Verifies ||v|| = 1.0 ± 1e-4 across diverse and adversarial corpora."""
        embedder = TextEmbedder(default_dimensions=1024)
        resp = embedder.embed_texts("qwen-embed-text", [text_input])
        assert len(resp.data) == 1
        vec = resp.data[0].embedding
        assert len(vec) == 1024
        l2_norm = math.sqrt(sum(x * x for x in vec))
        assert abs(l2_norm - 1.0) < 1e-4, f"L2 norm {l2_norm} exceeds tolerance for {text_input!r}"

    @pytest.mark.parametrize("dim", [1024, 768, 512, 256, 128, 64, 32, 16])
    def test_vector_embeddings_matryoshka_truncation_norms(self, dim):
        """Verifies Matryoshka dimension truncation preserves exact unit length across all sizes."""
        embedder = TextEmbedder(default_dimensions=1024)
        resp = embedder.embed_texts("qwen-embed-text", ["Testing Matryoshka projection."], dimensions=dim)
        vec = resp.data[0].embedding
        assert len(vec) == dim
        l2_norm = math.sqrt(sum(x * x for x in vec))
        assert abs(l2_norm - 1.0) < 1e-4

    def test_vector_embeddings_zero_sine_cosine_mock_vectors(self):
        """Verifies embeddings are authentic Gaussian hyperplane projections, not harmonic periodic mocks."""
        embedder = TextEmbedder(default_dimensions=256)
        resp = embedder.embed_texts("qwen-embed-text", ["Neural network architecture optimization."])
        vec = resp.data[0].embedding

        # Check coordinate statistics: mean should be close to 0, values should not follow periodic sin/cos
        mean_val = sum(vec) / len(vec)
        assert abs(mean_val) < 0.1, f"Abnormal coordinate mean: {mean_val}"

        # Differences between consecutive elements must not be a constant harmonic delta
        diffs = [abs(vec[i+1] - vec[i]) for i in range(len(vec) - 1)]
        diff_variance = sum((d - sum(diffs)/len(diffs))**2 for d in diffs) / len(diffs)
        assert diff_variance > 1e-4, "Coordinates exhibit synthetic periodic linearity"

    def test_vector_embeddings_semantic_dispersion(self):
        """Verifies cosine similarity between related concepts is strictly higher than unrelated concepts."""
        embedder = TextEmbedder(default_dimensions=512)
        resp = embedder.embed_texts("qwen-embed-text", [
            "Quantum electrodynamics in relativistic particle physics",
            "Quantum chromodynamics and subatomic particle collisions",
            "Artisan baking sourdough bread and French pastry croissants",
        ])
        v_rel1 = resp.data[0].embedding
        v_rel2 = resp.data[1].embedding
        v_unrel = resp.data[2].embedding

        sim_related = sum(a * b for a, b in zip(v_rel1, v_rel2))
        sim_unrelated = sum(a * b for a, b in zip(v_rel1, v_unrel))

        assert sim_related > sim_unrelated, f"Expected {sim_related} > {sim_unrelated}"
        # Unrelated text must not artificially cluster at mock values (~0.94)
        assert sim_unrelated < 0.85

    def test_vector_embeddings_offline_model_fail_fast(self, embedding_client):
        """Verifies offline embedding models fail fast with HTTP 503 and zero simulated fake vectors."""
        for offline_model in ["offline-model", "failing-embedding-model"]:
            resp = embedding_client.post(
                "/v1/embeddings",
                json={"model": offline_model, "input": ["Test input"]},
                headers=AUTH_HEADERS,
            )
            assert resp.status_code == 503
            assert resp.json()["error"]["code"] == "MODEL_UNAVAILABLE"

    @pytest.mark.parametrize("num_choices", [2, 3, 5, 8, 12, 20])
    def test_decision_choice_softmax_normalization_sum(self, num_choices):
        """Verifies Softmax distribution sums to exactly 1.0 ± 1e-6 across candidate choice set sizes."""
        context = "High resolution video upscaling and frame interpolation for media production."
        choices = [f"choice_{i}_{'upscale' if i == 0 else 'task'}" for i in range(num_choices)]
        resp = evaluate_choice(context, choices)

        prob_sum = sum(resp.probabilities.values())
        assert abs(prob_sum - 1.0) < 1e-6, f"Softmax sum {prob_sum} violated for {num_choices} choices"
        for c, p in resp.probabilities.items():
            assert 0.0 <= p <= 1.0, f"Probability {p} out of bounds for choice {c}"

    def test_decision_choice_extreme_logits_stability(self):
        """Verifies Softmax calculation remains numerically stable under extreme logit disparities."""
        context = "audio enhancement audio enhancement audio enhancement audio"
        choices = ["audio_enhancement", "video_generation", "code_generation", "database_query"]
        resp = evaluate_choice(context, choices)
        assert resp.selected_choice == "audio_enhancement"
        assert abs(sum(resp.probabilities.values()) - 1.0) < 1e-6

    def test_decision_choice_deterministic_tie_breaking(self):
        """Verifies symmetric choices with identical logits produce deterministic selections."""
        context = "neutral prompt with zero signal tokens"
        choices = ["option_alpha", "option_beta", "option_gamma"]
        results = [evaluate_choice(context, choices).selected_choice for _ in range(10)]
        assert len(set(results)) == 1, "Non-deterministic tie breaking observed across identical runs"

    def test_decision_choice_empty_rejection(self):
        """Verifies empty choices array is rejected with InvalidRequestError."""
        with pytest.raises(InvalidRequestError):
            evaluate_choice("Context", [])

    def test_decision_predicate_confidence_and_logprobs(self):
        """Verifies binary predicate confidence is bounded in [0.5, 1.0] and logprobs sum to ~1.0."""
        context = "Decoupled Factory v4 Control Plane routes requests to microservice workers."
        hypothesis = "Factory v4 utilizes a decoupled control plane architecture."
        resp = evaluate_predicate(context, hypothesis)
        assert isinstance(resp.decision, bool)
        assert 0.5 <= resp.confidence <= 1.0

        p_true = math.exp(resp.logprobs["true"])
        p_false = math.exp(resp.logprobs["false"])
        assert abs((p_true + p_false) - 1.0) < 0.02

    def test_decision_predicate_empty_hypothesis_rejection(self):
        """Verifies predicate evaluation requires non-empty hypothesis."""
        with pytest.raises(InvalidRequestError):
            evaluate_predicate("Valid context", "")

    def test_decision_score_continuous_bounds_and_calibration(self):
        """Verifies continuous scoring stays in [0.0, 1.0] and calibration metrics are valid."""
        context = "High quality detailed clean execution passed with zero defects."
        resp = evaluate_score(context)
        assert 0.0 <= resp.score <= 1.0
        assert 0.0 <= resp.calibration_metrics.ece <= 1.0
        assert 0.0 <= resp.calibration_metrics.brier_score <= 1.0

    def test_decision_score_custom_scale_interpolation(self):
        """Verifies custom min/max scales interpolate accurately within bounds."""
        context = "Evaluated workflow execution."
        resp = evaluate_score(context, scale=[-10.0, 50.0])
        assert -10.0 <= resp.score <= 50.0

        resp2 = evaluate_score(context, scale=[100.0, 200.0])
        assert 100.0 <= resp2.score <= 200.0

    def test_decision_score_invalid_scale_rejection(self):
        """Verifies scale with min >= max is rejected."""
        with pytest.raises(InvalidRequestError):
            evaluate_score("Context", scale=[10.0, 5.0])
        with pytest.raises(InvalidRequestError):
            evaluate_score("Context", scale=[5.0, 5.0])

    def test_decision_offline_model_fail_fast(self, decision_client):
        """Verifies offline decision models strictly return HTTP 503."""
        payload = {
            "model": "offline-model",
            "task": "predicate",
            "context": "Context",
            "hypothesis": "Hypothesis",
        }
        resp = decision_client.post("/v1/decisions", json=payload, headers=AUTH_HEADERS)
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "MODEL_UNAVAILABLE"

    def test_reranker_bm25_monotonic_descending_rank_order(self):
        """Verifies Okapi BM25 reranker returns results in strictly descending relevance order."""
        reranker = DocumentReranker()
        docs = [
            "Irrelevant astronomy passage about stellar nucleosynthesis.",
            "Factory v4 implements a two-tier queue architecture with Redis and Postgres.",
            "Medium relevance document discussing microservice worker queues in modern factories.",
            "Completely unrelated culinary recipe for sourdough bread.",
            "Factory queue system ensures reliable job leasing and heartbeats.",
        ]
        req = RerankRequest(
            model="qwen-rerank",
            query="Factory two-tier queue architecture",
            documents=docs,
        )
        resp = reranker.rerank(req)
        assert len(resp.results) == 5

        for i in range(len(resp.results) - 1):
            curr_score = resp.results[i].relevance_score
            next_score = resp.results[i + 1].relevance_score
            assert curr_score >= next_score, f"Monotonicity violation at index {i}: {curr_score} < {next_score}"

    def test_reranker_bm25_score_bounds_and_top_n(self):
        """Verifies reranker relevance scores are bounded in [0.01, 0.99] and top_n limits output."""
        reranker = DocumentReranker()
        docs = [f"Document {i} with varied content." for i in range(10)]
        req = RerankRequest(
            model="qwen-rerank",
            query="content",
            documents=docs,
            top_n=3,
        )
        resp = reranker.rerank(req)
        assert len(resp.results) == 3
        for item in resp.results:
            assert 0.01 <= item.relevance_score <= 0.99

    def test_reranker_offline_model_fail_fast(self, embedding_client):
        """Verifies reranker fails fast with 503 on offline models."""
        payload = {
            "model": "offline-model",
            "query": "test query",
            "documents": ["doc1", "doc2"],
        }
        resp = embedding_client.post("/v1/rerank", json=payload, headers=AUTH_HEADERS)
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "MODEL_UNAVAILABLE"


# ===========================================================================
# 3. SECURITY & SSRF BOUNDARIES
# ===========================================================================
@pytest.mark.security
@pytest.mark.ssrf
class TestM5SecurityAndSSRFBoundaries:
    """Empirical verification of SSRF attack vector rejection and 100% auth enforcement."""

    ADVERSARIAL_SSRF_PROBES = [
        # IPv4 Loopback
        "http://127.0.0.1:8000/internal",
        "http://127.0.1.1:8080/admin",
        "http://127.255.255.254/secret",
        # IPv4 Private Subnets (RFC 1918)
        "http://10.0.0.1/status",
        "http://10.254.0.1:8080/data",
        "http://172.16.0.1/metrics",
        "http://172.31.255.255/admin",
        "http://192.168.1.1/config",
        "http://192.168.0.254/secrets",
        # Cloud Metadata Endpoints
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://instance-data/latest/meta-data/",
        # Carrier-grade NAT & Local Network
        "http://100.64.0.1/private",
        "http://0.0.0.0:8000/",
        # IPv6 Loopback, Wildcard & ULA
        "http://[::1]/internal",
        "http://[::]/admin",
        "http://[fc00::1]/private",
        "http://[fe80::1]/linklocal",
        # Hostnames
        "http://localhost:8000/",
        "http://localhost.internal/",
    ]

    def test_ssrf_asset_validator_probes(self):
        """Verifies Asset Service validate_target_url strictly blocks all adversarial SSRF vectors."""
        for probe in self.ADVERSARIAL_SSRF_PROBES:
            with pytest.raises(SSRFBlockedError):
                validate_target_url(probe)

    def test_ssrf_scheme_and_port_violations(self):
        """Verifies Asset Service rejects non-HTTP schemes and dangerous ports."""
        disallowed_schemes = [
            "file:///etc/passwd",
            "ftp://attacker.com/payload.png",
            "gopher://127.0.0.1:70/",
        ]
        for url in disallowed_schemes:
            with pytest.raises(SSRFBlockedError):
                validate_target_url(url)

        disallowed_ports = [
            "http://example.com:22/image.png",    # SSH
            "http://example.com:3306/image.png",  # MySQL
            "http://example.com:6379/image.png",  # Redis
            "http://example.com:2375/image.png",  # Docker daemon
        ]
        for url in disallowed_ports:
            with pytest.raises(SSRFBlockedError):
                validate_target_url(url)

    def test_ssrf_speech_transcription_probes(self, speech_client):
        """Verifies Speech /v1/audio/transcriptions blocks SSRF URLs in JSON audio_url."""
        for probe in ["http://127.0.0.1:8080/audio.wav", "http://169.254.169.254/audio.wav"]:
            resp = speech_client.post(
                "/v1/audio/transcriptions",
                json={"audio_url": probe},
                headers=AUTH_HEADERS,
            )
            assert resp.status_code == 400
            assert resp.json()["error"]["code"] == "SSRF_BLOCKED"

    def test_ssrf_speech_alignment_probes(self, speech_client):
        """Verifies Speech /v1/audio/alignments blocks SSRF URLs."""
        resp = speech_client.post(
            "/v1/audio/alignments",
            json={"audio_url": "http://10.0.0.1/audio.wav", "transcript_text": "Sample text"},
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "SSRF_BLOCKED"

    def test_ssrf_speech_tts_voice_cloning_probes(self, speech_client):
        """Verifies Speech /v1/audio/speech blocks SSRF URLs in reference_audio_url."""
        resp = speech_client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "Speech synthesis with cloned voice.",
                "reference_audio_url": "http://192.168.1.1/voice_ref.wav",
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "SSRF_BLOCKED"

    def test_ssrf_music_audio_to_audio_probes(self, music_client):
        """Verifies Music /v1/audio_to_audio blocks SSRF URLs."""
        resp = music_client.post(
            "/v1/audio_to_audio",
            json={"audio_url": "http://127.0.0.1:8001/music.wav", "prompt": "rock style transfer"},
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "SSRF_BLOCKED"

    def test_ssrf_enhancement_endpoints_probes(self, enhancement_client):
        """Verifies Media Enhancement endpoints strictly reject SSRF URLs across all media modalities."""
        endpoints_and_payloads = [
            ("/v1/image_super_resolution", {"image_url": "http://169.254.169.254/sample.png"}),
            ("/v1/video_super_resolution", {"video_url": "http://10.0.0.1/sample.mp4"}),
            ("/v1/frame_interpolation", {"video_url": "http://172.16.0.1/sample.mp4"}),
            ("/v1/audio_enhancement", {"audio_url": "http://192.168.0.1/sample.wav"}),
        ]
        for endpoint, payload in endpoints_and_payloads:
            resp = enhancement_client.post(endpoint, json=payload, headers=AUTH_HEADERS)
            assert resp.status_code == 400
            assert resp.json()["error"]["code"] == "SSRF_BLOCKED"

    def test_ssrf_text_vlm_multimodal_probes(self, vlm_client):
        """Verifies Text & VLM subsystem rejects SSRF URLs in multimodal content arrays."""
        payload = {
            "model": "qwen-vl-max",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze diagram"},
                        {"type": "image_url", "image_url": {"url": "http://127.0.0.1:8000/secret.png"}},
                    ],
                }
            ],
        }
        resp = vlm_client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "SSRF_BLOCKED"

    def test_auth_perimeter_100_percent_unauthenticated_rejection(
        self,
        text_client,
        vlm_client,
        decision_client,
        embedding_client,
        speech_client,
        music_client,
        enhancement_client,
        control_client,
    ):
        """Verifies 100% of protected endpoints across all subsystems reject unauthenticated requests with HTTP 401."""
        protected_endpoints = [
            # Text & VLM
            (text_client, "POST", "/v1/chat/completions", {"model": "qwen-2.5-72b-instruct", "messages": [{"role": "user", "content": "Hi"}]}),
            (text_client, "POST", "/v1/responses", {"model": "qwen-2.5-72b-instruct", "input": "Plan"}),
            (vlm_client, "POST", "/v1/chat/completions", {"model": "qwen-vl-max", "messages": [{"role": "user", "content": "Hi"}]}),
            (vlm_client, "POST", "/v1/responses", {"model": "qwen-vl-max", "input": "Plan"}),
            # Decision
            (decision_client, "POST", "/v1/decisions", {"task": "predicate", "context": "C", "hypothesis": "H"}),
            # Embedding & Rerank
            (embedding_client, "POST", "/v1/embeddings", {"model": "qwen-embed-text", "input": ["Hello"]}),
            (embedding_client, "POST", "/v1/rerank", {"model": "qwen-rerank", "query": "Q", "documents": ["D1"]}),
            # Speech
            (speech_client, "POST", "/v1/audio/transcriptions", {"audio_url": "https://assets.ai.example.com/a.wav"}),
            (speech_client, "POST", "/v1/audio/alignments", {"audio_url": "https://assets.ai.example.com/a.wav", "transcript_text": "T"}),
            (speech_client, "POST", "/v1/audio/speech", {"input": "Hello"}),
            # Music
            (music_client, "POST", "/v1/text_to_music", {"prompt": "Pop song"}),
            (music_client, "POST", "/v1/lyrics_to_music", {"prompt": "Rock", "lyrics": "Words"}),
            (music_client, "POST", "/v1/audio_to_audio", {"audio_url": "https://assets.ai.example.com/m.wav", "prompt": "Jazz"}),
            # Enhancement
            (enhancement_client, "POST", "/v1/image_super_resolution", {"image_url": "https://assets.ai.example.com/i.png"}),
            (enhancement_client, "POST", "/v1/video_super_resolution", {"video_url": "https://assets.ai.example.com/v.mp4"}),
            (enhancement_client, "POST", "/v1/frame_interpolation", {"video_url": "https://assets.ai.example.com/v.mp4"}),
            (enhancement_client, "POST", "/v1/audio_enhancement", {"audio_url": "https://assets.ai.example.com/a.wav"}),
            # Control Plane
            (control_client, "POST", "/v1/jobs", {"capability": "text_to_image", "inputs": {}}),
            (control_client, "GET", "/v1/jobs/job-adversarial-probe", None),
            (control_client, "GET", "/v1/models", None),
            (control_client, "GET", "/v1/services", None),
            (control_client, "GET", "/v1/capabilities", None),
        ]

        for client, method, path, body in protected_endpoints:
            # 1. Without Authorization Header
            if method == "POST":
                resp_no_auth = client.post(path, json=body)
            else:
                resp_no_auth = client.get(path)
            assert resp_no_auth.status_code == 401, f"{path} accepted unauthenticated request: {resp_no_auth.status_code}"
            assert resp_no_auth.json()["error"]["code"] == "UNAUTHORIZED"

            # 2. With Invalid Authorization Header
            bad_headers = {"Authorization": "Bearer totally-invalid-token", "Content-Type": "application/json"}
            if method == "POST":
                resp_bad = client.post(path, json=body, headers=bad_headers)
            else:
                resp_bad = client.get(path, headers=bad_headers)
            assert resp_bad.status_code == 401, f"{path} accepted invalid token: {resp_bad.status_code}"
            assert resp_bad.json()["error"]["code"] == "UNAUTHORIZED"

            # 3. With Empty Bearer Token
            empty_headers = {"Authorization": "Bearer ", "Content-Type": "application/json"}
            if method == "POST":
                resp_empty = client.post(path, json=body, headers=empty_headers)
            else:
                resp_empty = client.get(path, headers=empty_headers)
            assert resp_empty.status_code == 401, f"{path} accepted empty token: {resp_empty.status_code}"

    def test_auth_perimeter_public_health_endpoints_remain_open(
        self,
        text_client,
        vlm_client,
        decision_client,
        embedding_client,
        speech_client,
        music_client,
        enhancement_client,
        control_client,
    ):
        """Verifies health probes (/health) remain accessible without authentication."""
        for client in [
            text_client,
            vlm_client,
            decision_client,
            embedding_client,
            speech_client,
            music_client,
            enhancement_client,
            control_client,
        ]:
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "healthy"


# ===========================================================================
# 4. STUDIO WEB DYNAMIC INTEGRITY
# ===========================================================================
class TestM5StudioWebDynamicIntegrity:
    """Empirical verification of Studio Web dynamic rendering, proxy elimination, and Next.js build."""

    STUDIO_WEB_DIR = REPO_ROOT / "factory" / "apps" / "studio-web"
    SRC_DIR = STUDIO_WEB_DIR / "src"

    def test_studio_web_zero_hardcoded_model_strings(self):
        """
        Verifies no hardcoded model ID arrays or static model option lists exist in Studio Web components.
        All models must be loaded dynamically from the Model Manifest Registry.
        """
        assert self.SRC_DIR.exists(), f"Directory not found: {self.SRC_DIR}"

        hardcoded_model_patterns = [
            re.compile(r'const\s+MODELS\s*=\s*\[', re.IGNORECASE),
            re.compile(r'AVAILABLE_MODELS\s*=\s*\[', re.IGNORECASE),
            re.compile(r'<option[^>]*>qwen-2\.5-72b', re.IGNORECASE),
            re.compile(r'<option[^>]*>wan-2\.2', re.IGNORECASE),
            re.compile(r'<option[^>]*>qwen3-asr', re.IGNORECASE),
            re.compile(r'<option[^>]*>ace-step', re.IGNORECASE),
        ]

        violations = []
        for file_path in self.SRC_DIR.rglob("*"):
            if file_path.suffix in (".ts", ".tsx", ".js", ".jsx"):
                content = file_path.read_text(encoding="utf-8")
                for pat in hardcoded_model_patterns:
                    match = pat.search(content)
                    if match:
                        violations.append(f"{file_path.relative_to(REPO_ROOT)}: match '{match.group(0)}'")

        assert not violations, f"Hardcoded model definitions detected in Studio Web:\n" + "\n".join(violations)

    def test_studio_web_zero_proxy_fallbacks(self):
        """Verifies zero references to legacy x-target-base-url or arbitrary proxying exist in Studio Web."""
        proxy_pattern = re.compile(r"x-target-base-url", re.IGNORECASE)
        violations = []
        for file_path in self.SRC_DIR.rglob("*"):
            if file_path.suffix in (".ts", ".tsx", ".js", ".json"):
                content = file_path.read_text(encoding="utf-8")
                if proxy_pattern.search(content):
                    violations.append(str(file_path.relative_to(REPO_ROOT)))

        assert not violations, f"Legacy proxy header 'x-target-base-url' detected in:\n" + "\n".join(violations)

    def test_studio_web_clean_typescript_compilation(self):
        """Verifies Studio Web TypeScript compiles cleanly with 0 type errors (tsc --noEmit)."""
        tsc_bin = self.STUDIO_WEB_DIR / "node_modules" / ".bin" / "tsc"
        cmd = [str(tsc_bin), "--noEmit"] if tsc_bin.exists() else ["npx", "tsc", "--noEmit"]
        result = subprocess.run(
            cmd,
            cwd=str(self.STUDIO_WEB_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"TypeScript compilation failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    def test_studio_web_clean_nextjs_production_build(self):
        """Verifies Next.js 15 production build succeeds with all static routes generated."""
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=str(self.STUDIO_WEB_DIR),
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert result.returncode == 0, f"Next.js build failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        assert "Generating static pages" in result.stdout or "Compiled successfully" in result.stdout
