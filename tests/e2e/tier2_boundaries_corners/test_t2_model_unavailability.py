"""
Tier 2 Model Unavailability & Fail-Fast Integrity Tests
Validates strict 503 MODEL_UNAVAILABLE fail-fast behavior across all subsystems,
verifying zero deterministic fake vectors, zero heuristic mock decisions,
and capability-based task rejections.
"""

import pytest
import httpx
from tests.e2e.conftest import assert_sanitized_error


@pytest.mark.tier2
class TestModelUnavailabilityFailFast:
    # --- Embedding Subsystem Fail-Fast (Prohibition of fake vectors) ---
    def test_embeddings_fail_fast_offline_model(self, client: httpx.Client, auth_headers):
        payload = {"model": "failing-embedding-model", "input": "test query"}
        res = client.post("/v1/embeddings", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 503, "MODEL_UNAVAILABLE")

    def test_embeddings_fail_fast_v2_offline(self, client: httpx.Client, auth_headers):
        payload = {"model": "offline-model", "input": "multimodal test"}
        res = client.post("/v2/embed", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 503, "MODEL_UNAVAILABLE")

    def test_embeddings_zero_fake_vectors_on_error(self, client: httpx.Client, auth_headers):
        payload = {"model": "failing-embedding-model", "input": "Verify no deterministic hashes"}
        res = client.post("/v1/embeddings", headers=auth_headers, json=payload)
        assert res.status_code == 503
        # Response must not have "data" field containing fake vectors
        assert "data" not in res.json()

    # --- Rerank Subsystem Fail-Fast ---
    def test_rerank_fail_fast_offline_model(self, client: httpx.Client, auth_headers):
        payload = {"model": "failing-rerank-model", "query": "q", "documents": ["doc1", "doc2"]}
        res = client.post("/v1/rerank", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 503, "MODEL_UNAVAILABLE")

    def test_rerank_legacy_fail_fast(self, client: httpx.Client, auth_headers):
        payload = {"model": "failing-rerank-model", "query": "q", "documents": ["doc1"]}
        res = client.post("/rerank", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 503, "MODEL_UNAVAILABLE")

    def test_rerank_zero_mock_scores_on_error(self, client: httpx.Client, auth_headers):
        payload = {"model": "offline-model", "query": "q", "documents": ["doc"]}
        res = client.post("/v1/rerank", headers=auth_headers, json=payload)
        assert res.status_code == 503
        assert "results" not in res.json()

    # --- Decision Subsystem Fail-Fast (Prohibition of fake heuristics) ---
    def test_decision_predicate_fail_fast(self, client: httpx.Client, auth_headers):
        payload = {"model": "offline-decision-model", "task": "predicate", "context": "c", "hypothesis": "h"}
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 503, "MODEL_UNAVAILABLE")

    def test_decision_choice_fail_fast(self, client: httpx.Client, auth_headers):
        payload = {"model": "offline-decision-model", "task": "choice", "context": "c", "choices": ["A", "B"]}
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 503, "MODEL_UNAVAILABLE")

    def test_decision_score_fail_fast(self, client: httpx.Client, auth_headers):
        payload = {"model": "offline-decision-model", "task": "score", "context": "c"}
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 503, "MODEL_UNAVAILABLE")

    def test_decision_zero_heuristic_outputs_on_error(self, client: httpx.Client, auth_headers):
        payload = {"model": "offline-model", "task": "predicate", "context": "test", "hypothesis": "test"}
        res = client.post("/v1/decisions", headers=auth_headers, json=payload)
        assert res.status_code == 503
        assert "decision" not in res.json()

    # --- Text / VLM Fail-Fast ---
    def test_chat_completions_fail_fast(self, client: httpx.Client, auth_headers):
        payload = {"model": "offline-model", "messages": [{"role": "user", "content": "hi"}]}
        res = client.post("/v1/chat/completions", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 503, "MODEL_UNAVAILABLE")

    def test_text_responses_fail_fast(self, client: httpx.Client, auth_headers):
        payload = {"model": "offline-model", "input": "reasoning task"}
        res = client.post("/v1/responses", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 503, "MODEL_UNAVAILABLE")

    # --- Speech Subsystem Fail-Fast ---
    def test_speech_asr_fail_fast(self, client: httpx.Client, auth_headers):
        payload = {"model": "offline-model", "audio_url": "https://s3.ai.example.com/assets/audio.wav"}
        res = client.post("/v1/audio/transcriptions", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 503, "MODEL_UNAVAILABLE")

    # --- Job Queue Fail-Fast across Media Subsystems ---
    @pytest.mark.parametrize("service,task", [
        ("image", "text_to_image"),
        ("image", "image_to_image"),
        ("image", "inpaint"),
        ("image", "outpaint"),
        ("image", "instruction_edit"),
        ("image", "controlnet"),
        ("video", "text_to_video"),
        ("video", "image_to_video"),
        ("video", "fl2v"),
        ("video", "video_extension"),
        ("video", "video_with_audio"),
        ("music", "text_to_music"),
        ("music", "lyrics_to_music"),
        ("enhancement", "image_super_resolution"),
        ("enhancement", "frame_interpolation"),
    ])
    def test_jobs_fail_fast_on_offline_model(self, client: httpx.Client, auth_headers, service, task):
        payload = {
            "service": service,
            "model": "offline-model",
            "task": task,
            "inputs": {"prompt": "testing fail fast"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 503, "MODEL_UNAVAILABLE")

    # --- Capability Mismatch Rejection ---
    def test_partial_model_unsupported_capability_rejected(self, client: httpx.Client, auth_headers):
        # experimental-sketch-partial only supports t2i and i2i, NOT inpaint
        payload = {
            "service": "image",
            "model": "experimental-sketch-partial",
            "task": "inpaint",
            "inputs": {
                "image_url": "https://s3.ai.example.com/assets/img.png",
                "mask_url": "https://s3.ai.example.com/assets/mask.png",
                "prompt": "Fill"
            }
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "CAPABILITY_NOT_SUPPORTED")
        assert "supported_capabilities" in res.json()["error"]["details"]

    def test_unsupported_task_name_rejected(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "unknown_future_task",
            "inputs": {}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "CAPABILITY_NOT_SUPPORTED")
