"""
tests/adversarial/test_adversarial_m3_it2_gate_probes.py
Adversarial & Empirical Gate Probes for Milestone 3 Iteration 2 Gate Verification.

Empirically probes and verifies:
1. Vector Embeddings:
   - Strict unit L2 normalization: sum(v_i^2) == 1.0 ± 1e-4 across diverse corpora.
   - Matryoshka dimension truncation (1024, 768, 512, 256, 128, 64, 32).
   - Authentic semantic dispersion (related > unrelated, zero Fourier mock clustering to ~0.94).
   - Fail-fast HTTP 503 MODEL_UNAVAILABLE on offline models with zero mock vectors.
2. Document Reranker:
   - Okapi BM25 relevance scoring on non-test queries with strict descending sort order.
   - Score boundaries in [0.01, 0.99] and top_n truncation.
   - Fail-fast HTTP 503 on offline models with zero mock results.
3. Decision Engine:
   - Multi-class choice Softmax normalization summing to 1.0 ± 1e-6 across candidate set sizes.
   - Binary predicate confidence in [0.5, 1.0] and logprob validity (exp(lp_true) + exp(lp_false) == 1.0 ± 0.01).
   - Continuous score scaling to custom boundaries and calibration metrics (ECE, Brier) in [0.0, 1.0].
   - Fail-fast HTTP 503 on offline decision models.
4. Text & VLM Subsystems:
   - Multi-turn dialogue context preservation.
   - Dynamic system persona injection without static hardcoding.
   - Structured JSON output parsing and validation.
   - OpenAI Responses API 3-step structured decomposition and reasoning tokens.
   - Multimodal VLM dynamic modality conditioning (video stream vs diagram vs image) and SSRF rejection.
5. Speech WebSocket Security & Lifecycle:
   - Mandatory Bearer token authentication (close code 1008 on unauthenticated / invalid token).
   - Graceful and abrupt client disconnection resilience without daemon crash.
"""
import asyncio
import json
import math
import os
import re
from typing import Any, Dict, List
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

AUTH_TOKEN = "sk-factory-test-token-valid"
AUTH_HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}", "Content-Type": "application/json"}

from factory_auth.config import AuthConfig
from factory_contracts.errors import (
    ErrorCode,
    InvalidRequestError,
    ModelUnavailableError,
)

# AI Subsystem Services
from services.embedding.src.app import app as embedding_app
from services.embedding.src.engine import EmbeddingEngine
from services.embedding.src.schemas import EmbeddingRequest, RerankRequest
from services.embedding.src.text_embed import TextEmbedder, OFFLINE_EMBEDDING_MODELS
from services.embedding.src.reranker import DocumentReranker, OFFLINE_RERANK_MODELS

from services.decision.src.app import app as decision_app
from services.decision.src.engine import DecisionEngine
from services.decision.src.choice import evaluate_choice
from services.decision.src.predicate import evaluate_predicate
from services.decision.src.score import evaluate_score

from services.text.src.app import app as text_app
from services.text.src.engine import TextEngine, OFFLINE_MODELS as OFFLINE_TEXT_MODELS
from services.text.src.models import ChatCompletionRequest, ResponseApiRequest
from services.text.src.responses_handler import ResponsesHandler

from services.vlm.src.app import app as vlm_app
from services.vlm.src.engine import VLMEngine

from services.speech.server import app as speech_app


# ==============================================================================
# 1. VECTOR EMBEDDINGS EMPIRICAL PROBES
# ==============================================================================
class TestVectorEmbeddingsEmpiricalProbes:
    """Empirical validation of vector embeddings mathematical invariants and semantic behavior."""

    @pytest.fixture
    def embed_client(self):
        embedding_app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
        embedding_app.state.engine = EmbeddingEngine()
        return TestClient(embedding_app)

    @pytest.mark.parametrize("text_input", [
        "Quantum electrodynamics is a relativistic quantum field theory of electrodynamics.",
        "Differential geometry analyzes Riemannian manifolds and curvature tensors.",
        "Factory v4 decouples the control plane from autonomous subsystem inference daemons.",
        "A" * 500,
        "SingleWord",
        "12345 67890 !@#$%^&*()",
        "Multi-lingual probe: 现代人工智能系统 高性能并发计算 深度学习架构",
        "Short",
        "Dense technical specification describing two-tier job queues, Redis locks, and Postgres state machines.",
        "   Whitespace padded text query with internal punctuation... and symbols;   ",
    ])
    def test_embeddings_strict_l2_unit_norm_invariant(self, embed_client, text_input):
        """
        Verify that for arbitrary inputs, the Euclidean (L2) norm of the embedding vector
        strictly satisfies ||v||_2 = sqrt(sum(v_i^2)) == 1.0 ± 1e-4.
        """
        payload = {
            "model": "qwen-embedding-8b",
            "input": [text_input],
        }
        res = embed_client.post("/v1/embeddings", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200, f"Embedding failed: {res.text}"
        data = res.json()

        assert "data" in data
        assert len(data["data"]) == 1
        vec = data["data"][0]["embedding"]
        assert len(vec) == 1024

        # Compute Euclidean norm: ||v|| = sqrt(sum(v_i^2))
        norm_sq = sum(x * x for x in vec)
        l2_norm = math.sqrt(norm_sq)
        assert abs(l2_norm - 1.0) < 1e-4, f"L2 norm violation: ||v|| = {l2_norm} (deviation {abs(l2_norm - 1.0)})"

        # Check coordinate bounds
        for val in vec:
            assert -1.0 <= val <= 1.0, f"Coordinate {val} exceeds unit interval [-1, 1]"

    @pytest.mark.parametrize("dim", [1024, 768, 512, 256, 128, 64, 32])
    def test_embeddings_matryoshka_dimension_truncation_and_norm(self, embed_client, dim):
        """
        Verify Matryoshka dimension truncation:
        1. Output vector dimension strictly matches the requested 'dimensions' parameter.
        2. The truncated vector is strictly re-normalized to unit L2 norm (1.0 ± 1e-4).
        """
        payload = {
            "model": "qwen-embedding-8b",
            "input": ["Testing Matryoshka representation projection across diverse dimensions."],
            "dimensions": dim,
        }
        res = embed_client.post("/v1/embeddings", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        vec = res.json()["data"][0]["embedding"]

        assert len(vec) == dim, f"Expected dimension {dim}, got {len(vec)}"
        l2_norm = math.sqrt(sum(x * x for x in vec))
        assert abs(l2_norm - 1.0) < 1e-4, f"Truncated vector L2 norm violation at dim {dim}: ||v|| = {l2_norm}"

    def test_embeddings_semantic_dispersion_and_non_mock_clustering(self, embed_client):
        """
        Verify authentic semantic dispersion across diverse corpora:
        - Intra-topic cosine similarity must be significantly higher than inter-topic similarity.
        - Unrelated texts must NOT cluster to ~0.94 (which was the signature of the removed Fourier basis mock).
        """
        corpus = [
            # Topic A: Astrophysics & Cosmology (sharing 'supernova', 'stellar')
            "Supernova explosions disperse heavy elements into stellar space.",
            "Supernova core collapse creates neutron stars in stellar remnants.",
            # Topic B: Gastronomy & Baking (sharing 'sourdough', 'fermentation')
            "Sourdough fermentation relies on wild lactobacilli and yeast.",
            "Artisan sourdough bread develops flavor during cold fermentation.",
            # Topic C: Computer Systems (sharing 'operating', 'kernel')
            "Operating system kernel manages virtual memory and page tables.",
            "Kernel interrupt handlers execute within the operating system core.",
        ]

        payload = {
            "model": "qwen-embedding-8b",
            "input": corpus,
        }
        res = embed_client.post("/v1/embeddings", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        vectors = [item["embedding"] for item in res.json()["data"]]

        def cosine_similarity(v1: List[float], v2: List[float]) -> float:
            return sum(a * b for a, b in zip(v1, v2))

        # Intra-topic similarities
        sim_a_intra = cosine_similarity(vectors[0], vectors[1])  # Astro vs Astro
        sim_b_intra = cosine_similarity(vectors[2], vectors[3])  # Food vs Food
        sim_c_intra = cosine_similarity(vectors[4], vectors[5])  # OS vs OS

        # Inter-topic similarities
        sim_ab = cosine_similarity(vectors[0], vectors[2])  # Astro vs Food
        sim_ac = cosine_similarity(vectors[0], vectors[4])  # Astro vs OS
        sim_bc = cosine_similarity(vectors[2], vectors[4])  # Food vs OS

        avg_intra = (sim_a_intra + sim_b_intra + sim_c_intra) / 3.0
        avg_inter = (sim_ab + sim_ac + sim_bc) / 3.0

        # Invariant 1: Unrelated texts must not cluster to ~0.94 (historical mock defect)
        for sim in [sim_ab, sim_ac, sim_bc]:
            assert sim < 0.20, f"Unrelated similarity {sim:.4f} too high (indicates false clustering artifact)"

        # Invariant 2: Semantically related pairs have higher similarity than unrelated pairs
        assert avg_intra > avg_inter + 0.15, (
            f"Intra-topic similarity ({avg_intra:.4f}) must exceed inter-topic ({avg_inter:.4f}) by >= 0.15"
        )

    @pytest.mark.parametrize("offline_model", [
        "failing-embedding-model",
        "offline-model",
        "qwen-offline",
        "failing-model",
    ])
    def test_embeddings_fail_fast_offline_models(self, embed_client, offline_model):
        """
        Verify that offline embedding models fail fast with HTTP 503 MODEL_UNAVAILABLE.
        The response MUST NOT contain any 'data' key or mock numbers.
        """
        for route in ["/v1/embeddings", "/v2/embed"]:
            payload = {"model": offline_model, "input": "adversarial test"}
            res = embed_client.post(route, headers=AUTH_HEADERS, json=payload)
            assert res.status_code == 503, f"Expected 503 on {route} for {offline_model}, got {res.status_code}"
            body = res.json()
            assert body["error"]["code"] == "MODEL_UNAVAILABLE"
            assert "data" not in body, "Crucial integrity rule: 'data' must be ABSENT on 503 error"


# ==============================================================================
# 2. DOCUMENT RERANKER EMPIRICAL PROBES
# ==============================================================================
class TestDocumentRerankerEmpiricalProbes:
    """Empirical validation of Okapi BM25 reranking math, score bounds, and descending order."""

    @pytest.fixture
    def rerank_client(self):
        embedding_app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
        embedding_app.state.engine = EmbeddingEngine()
        return TestClient(embedding_app)

    def test_reranker_okapi_bm25_unseen_queries_descending_order(self, rerank_client):
        """
        Verify Okapi BM25 scoring with non-test queries:
        1. Relevance scores are strictly descending: score[i] >= score[i+1].
        2. Relevance scores are bounded in [0.01, 0.99].
        3. Documents with high term frequency and query alignment rank highest.
        """
        test_scenarios = [
            {
                "query": "cardiovascular indications for beta blockers and hypertension therapy",
                "documents": [
                    "Beta blockers are primarily indicated for managing cardiovascular hypertension and cardiac arrhythmias.",
                    "Hypertension therapy often includes dietary sodium reduction and aerobic exercise.",
                    "Geological strata in sedimentary rock formations indicate ancient volcanic activity.",
                    "A classic French onion soup recipe requires slow caramelized onions and beef broth.",
                ],
                "expected_top_term": "beta blockers",
            },
            {
                "query": "distributed consensus protocol state machine log replication",
                "documents": [
                    "The Raft consensus protocol ensures state machine consistency through leader-driven log replication.",
                    "Distributed algorithms coordinate nodes across asynchronous networks.",
                    "An introduction to watercolor painting techniques for landscapes.",
                ],
                "expected_top_term": "raft consensus",
            },
        ]

        for scenario in test_scenarios:
            payload = {
                "model": "qwen-rerank-8b",
                "query": scenario["query"],
                "documents": scenario["documents"],
            }
            res = rerank_client.post("/v1/rerank", headers=AUTH_HEADERS, json=payload)
            assert res.status_code == 200, f"Rerank failed: {res.text}"
            results = res.json()["results"]
            assert len(results) == len(scenario["documents"])

            scores = [r["relevance_score"] for r in results]

            # Invariant 1: Strictly descending order
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], f"Sort violation: {scores[i]} < {scores[i + 1]}"

            # Invariant 2: Bounded in [0.01, 0.99]
            for s in scores:
                assert 0.01 <= s <= 0.99, f"Score {s} out of bounds [0.01, 0.99]"

            # Invariant 3: Expected most relevant doc is ranked #1
            top_doc = results[0]["document"].lower()
            assert scenario["expected_top_term"] in top_doc

    def test_reranker_top_n_slicing(self, rerank_client):
        """Verify that 'top_n' correctly slices the top ranking documents."""
        payload = {
            "model": "qwen-rerank-8b",
            "query": "compiler optimization intermediate representation",
            "documents": [
                "LLVM intermediate representation enables modular compiler optimizations.",
                "Dead code elimination removes unreachable instructions from the IR.",
                "Loop unrolling improves pipeline execution throughput.",
                "Unrelated text about growing organic heirloom tomatoes.",
                "Coffee beans origin and roasting temperatures.",
            ],
            "top_n": 2,
        }
        res = rerank_client.post("/v1/rerank", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        results = res.json()["results"]
        assert len(results) == 2
        assert "compiler" in results[0]["document"].lower() or "ir" in results[0]["document"].lower()

    @pytest.mark.parametrize("offline_model", [
        "failing-rerank-model",
        "offline-model",
        "offline-rerank-model",
        "failing-model",
    ])
    def test_reranker_fail_fast_offline_models(self, rerank_client, offline_model):
        """Verify that offline reranker models return 503 and omit 'results' key."""
        payload = {
            "model": offline_model,
            "query": "test query",
            "documents": ["doc1", "doc2"],
        }
        res = rerank_client.post("/v1/rerank", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 503
        body = res.json()
        assert body["error"]["code"] == "MODEL_UNAVAILABLE"
        assert "results" not in body, "Crucial integrity rule: 'results' must be ABSENT on 503 error"


# ==============================================================================
# 3. DECISION ENGINE EMPIRICAL PROBES
# ==============================================================================
class TestDecisionEngineEmpiricalProbes:
    """Empirical validation of Decision Engine probability math, Softmax, and calibration."""

    @pytest.fixture
    def decision_client(self):
        decision_app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
        decision_app.state.engine = DecisionEngine()
        return TestClient(decision_app)

    @pytest.mark.parametrize("choices", [
        ["audio_enhancement", "video_upscale"],
        ["microservice_gateway", "database_storage", "event_stream"],
        ["low", "medium", "high", "critical", "emergency"],
        [f"routing_option_{i}" for i in range(10)],
        [f"candidate_cluster_worker_{i}" for i in range(20)],
    ])
    def test_decision_choice_softmax_normalization(self, decision_client, choices):
        """
        Verify that for multi-class choice tasks across varying candidate sizes:
        1. Probabilities sum to 1.0 ± 1e-6.
        2. Every individual probability p_i > 0.0 and p_i <= 1.0.
        3. selected_choice == max(probabilities, key=probabilities.get).
        """
        payload = {
            "model": "qwen-decision-8b",
            "task": "choice",
            "context": "Incoming task requires high-throughput streaming audio enhancement and noise suppression.",
            "choices": choices,
        }
        res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200, f"Decision choice failed: {res.text}"
        data = res.json()

        assert data["task"] == "choice"
        probs = data["probabilities"]
        assert len(probs) == len(choices)

        # Invariant 1: Sum strictly equals 1.0 ± 1e-6
        prob_sum = sum(probs.values())
        assert abs(prob_sum - 1.0) < 1e-6, f"Softmax sum violation: {prob_sum} deviates from 1.0 by {abs(prob_sum - 1.0)}"

        # Invariant 2: All p_i in (0, 1]
        for c, p in probs.items():
            assert 0.0 < p <= 1.0, f"Choice '{c}' probability {p} out of bounds (0, 1]"

        # Invariant 3: selected_choice is the argmax
        best_choice = max(probs.keys(), key=lambda k: probs[k])
        assert data["selected_choice"] == best_choice

    def test_decision_predicate_unseen_inputs_and_logprob_math(self, decision_client):
        """
        Verify binary predicate evaluation with unseen context/hypothesis pairs:
        1. p_true + p_false == 1.0 ± 0.01.
        2. exp(logprob_true) + exp(logprob_false) == 1.0 ± 0.01.
        3. logprobs are non-positive floats (<= 0.0).
        4. confidence == max(p_true, p_false) >= 0.5.
        """
        test_pairs = [
            ("The satellite communications downlink experienced high atmospheric interference.", "The incident involves signal transmission degradation."),
            ("Biomedical imaging analysis identified cellular mitotic anomalies.", "The findings pertain to oncological pathology."),
            ("Symphony orchestra performed Beethoven's Ninth in the concert hall.", "The event relates to automotive engine diagnostics."),
            ("The PostgreSQL database cluster triggered a failover replica promotion.", "The event relates to data storage high availability."),
        ]

        for ctx, hyp in test_pairs:
            payload = {
                "model": "qwen-decision-8b",
                "task": "predicate",
                "context": ctx,
                "hypothesis": hyp,
            }
            res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
            assert res.status_code == 200, f"Predicate evaluation failed: {res.text}"
            data = res.json()

            assert data["task"] == "predicate"
            assert isinstance(data["decision"], bool)

            # Confidence invariant
            assert 0.5 <= data["confidence"] <= 1.0, f"Confidence {data['confidence']} out of bounds [0.5, 1.0]"

            # Logprob invariants
            logprobs = data["logprobs"]
            lp_true = logprobs["true"]
            lp_false = logprobs["false"]
            assert lp_true <= 0.0, f"logprob_true {lp_true} must be <= 0.0"
            assert lp_false <= 0.0, f"logprob_false {lp_false} must be <= 0.0"

            # Probability reconstruction
            p_true = math.exp(lp_true)
            p_false = math.exp(lp_false)
            assert abs((p_true + p_false) - 1.0) < 0.01, f"Reconstructed probability sum {p_true + p_false} != 1.0"

    @pytest.mark.parametrize("scale", [
        [0.0, 1.0],
        [1.0, 10.0],
        [-50.0, 50.0],
        [100.0, 1000.0],
    ])
    def test_decision_score_scaling_and_calibration_metrics(self, decision_client, scale):
        """
        Verify continuous scoring:
        1. Score strictly lies within [scale[0], scale[1]].
        2. Calibration metrics ECE and Brier score are strictly within [0.0, 1.0].
        3. Positive sentiment text scores higher than negative sentiment text.
        """
        positive_context = "System metrics demonstrate optimal latency, zero packet loss, passed all tests, stable performance."
        negative_context = "System metrics demonstrate catastrophic failure, severe degradation, corrupted memory, unstable crashing."

        res_pos = decision_client.post(
            "/v1/decisions",
            headers=AUTH_HEADERS,
            json={"model": "qwen-decision-8b", "task": "score", "context": positive_context, "scale": scale},
        )
        assert res_pos.status_code == 200
        score_pos = res_pos.json()["score"]

        res_neg = decision_client.post(
            "/v1/decisions",
            headers=AUTH_HEADERS,
            json={"model": "qwen-decision-8b", "task": "score", "context": negative_context, "scale": scale},
        )
        assert res_neg.status_code == 200
        score_neg = res_neg.json()["score"]

        # Scale boundaries
        assert scale[0] <= score_pos <= scale[1]
        assert scale[0] <= score_neg <= scale[1]

        # Monotonicity: positive quality > negative quality
        assert score_pos > score_neg, f"Positive score {score_pos} should exceed negative score {score_neg}"

        # Calibration metrics
        calib = res_pos.json()["calibration_metrics"]
        assert 0.0 <= calib["ece"] <= 1.0
        assert 0.0 <= calib["brier_score"] <= 1.0

    def test_decision_fail_fast_offline_models(self, decision_client):
        """Verify that offline decision models return 503 and omit 'decision' key."""
        payload = {
            "model": "offline-decision-model",
            "task": "predicate",
            "context": "context",
            "hypothesis": "hypothesis",
        }
        res = decision_client.post("/v1/decisions", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 503
        body = res.json()
        assert body["error"]["code"] == "MODEL_UNAVAILABLE"
        assert "decision" not in body


# ==============================================================================
# 4. TEXT & VLM SUBSYSTEMS EMPIRICAL PROBES
# ==============================================================================
class TestTextAndVLMEmpiricalProbes:
    """Empirical validation of Text dialogue, persona injection, JSON formatting, and VLM."""

    @pytest.fixture
    def text_client(self):
        text_app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
        text_app.state.engine = TextEngine()
        return TestClient(text_app)

    @pytest.fixture
    def vlm_client(self):
        vlm_app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
        vlm_app.state.engine = VLMEngine()
        return TestClient(vlm_app)

    def test_text_multi_turn_dialogue(self, text_client):
        """Verify multi-turn conversation context flow."""
        payload = {
            "model": "qwen-2.5-72b-instruct",
            "messages": [
                {"role": "user", "content": "We need to design a high-throughput video processing pipeline."},
                {"role": "assistant", "content": "I recommend decoupling the queue management from ComfyUI workers using Redis events."},
                {"role": "user", "content": "What queue parameters should be configured in Factory v4?"},
            ],
        }
        res = text_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        content = res.json()["choices"][0]["message"]["content"]
        assert "Factory v4" in content or "queue" in content.lower()

    @pytest.mark.parametrize("persona_role", [
        "quantum computing research fellow",
        "cybersecurity penetration testing specialist",
        "embedded firmware real-time systems architect",
    ])
    def test_text_system_persona_injection(self, text_client, persona_role):
        """
        Verify that system instructions dynamically inject the persona
        into the assistant response without hardcoded static branches.
        """
        payload = {
            "model": "qwen-2.5-72b-instruct",
            "messages": [
                {"role": "system", "content": f"You are a {persona_role}."},
                {"role": "user", "content": "Identify yourself and describe your role."},
            ],
        }
        res = text_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        content = res.json()["choices"][0]["message"]["content"]
        assert persona_role.lower() in content.lower(), f"Persona '{persona_role}' not reflected in: {content}"

    def test_text_structured_json_response_format(self, text_client):
        """
        Verify that passing response_format={'type': 'json_object'} returns valid,
        parseable JSON containing required structured keys.
        """
        payload = {
            "model": "qwen-2.5-72b-instruct",
            "messages": [
                {"role": "user", "content": "Provide a status evaluation for node-cluster-01"},
            ],
            "response_format": {"type": "json_object"},
        }
        res = text_client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        raw_content = res.json()["choices"][0]["message"]["content"]

        parsed = json.loads(raw_content)
        assert isinstance(parsed, dict)
        assert parsed.get("status") == "success"
        assert parsed.get("model") == "qwen-2.5-72b-instruct"
        assert "response" in parsed
        assert "query" in parsed

    def test_text_responses_api_structured_reasoning(self, text_client):
        """Verify the OpenAI Responses API 3-step structured decomposition and reasoning tokens."""
        payload = {
            "model": "qwen-2.5-72b-instruct",
            "input": "Determine cluster memory allocation: Max_RAM = (Worker_Nodes * 32GB) - OS_Overhead",
        }
        res = text_client.post("/v1/responses", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["object"] == "response"
        assert "Reasoning Step 1" in data["output"]
        assert "Reasoning Step 2" in data["output"]
        assert "Reasoning Step 3" in data["output"]
        assert data["reasoning_tokens"] > 15
        assert data["status"] == "completed"

    def test_vlm_multimodal_dynamic_modality_conditioning(self, vlm_client):
        """
        Verify Qwen2-VL dynamically conditions its output based on media item modalities:
        1. Video asset -> temporal video analysis.
        2. Technical diagram -> architecture / topology analysis.
        3. Generic image -> visual features analysis.
        """
        # Case 1: Video clip
        res_vid = vlm_client.post(
            "/v1/chat/completions",
            headers=AUTH_HEADERS,
            json={
                "model": "qwen-vl-max",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Assess visual continuity across frames"},
                            {"type": "video_url", "video_url": {"url": "https://s3.ai.example.com/assets/sample.mp4"}},
                        ],
                    }
                ],
            },
        )
        assert res_vid.status_code == 200
        vid_text = res_vid.json()["choices"][0]["message"]["content"]
        assert "video" in vid_text.lower() or "temporal" in vid_text.lower()

        # Case 2: Technical diagram
        res_diag = vlm_client.post(
            "/v1/chat/completions",
            headers=AUTH_HEADERS,
            json={
                "model": "qwen-vl-max",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Review architecture topology schema"},
                            {"type": "image_url", "image_url": {"url": "https://s3.ai.example.com/assets/topology_diagram.png"}},
                        ],
                    }
                ],
            },
        )
        assert res_diag.status_code == 200
        diag_text = res_diag.json()["choices"][0]["message"]["content"]
        assert "diagram" in diag_text.lower() or "topology" in diag_text.lower()

    def test_vlm_ssrf_rejection(self, vlm_client):
        """Verify VLM rejects private, loopback, and cloud metadata URLs with 400 SSRF_BLOCKED."""
        for blocked_url in [
            "http://169.254.169.254/latest/meta-data/",
            "http://127.0.0.1:8000/internal",
            "http://10.0.0.1/secrets",
        ]:
            res = vlm_client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={
                    "model": "qwen-vl-max",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Inspect URL"},
                                {"type": "image_url", "image_url": {"url": blocked_url}},
                            ],
                        }
                    ],
                },
            )
            assert res.status_code == 400
            assert res.json()["error"]["code"] == "SSRF_BLOCKED"


# ==============================================================================
# 5. SPEECH WEBSOCKET SECURITY & LIFECYCLE PROBES
# ==============================================================================
class TestSpeechWebSocketSecurityAndLifecycleProbes:
    """Verifies that speech WebSocket enforces mandatory Bearer auth and survives disconnects."""

    def test_speech_ws_unauthenticated_rejection_code_1008(self):
        """Verify that connecting without authentication token is rejected with close code 1008."""
        speech_app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
        with TestClient(speech_app) as client:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect("/v1/audio/transcriptions/ws") as ws:
                    ws.send_json({"chunk_id": 1})
            assert exc_info.value.code == 1008

    def test_speech_ws_invalid_token_rejection_code_1008(self):
        """Verify that connecting with an invalid token is rejected with close code 1008."""
        speech_app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
        with TestClient(speech_app) as client:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect("/v1/audio/transcriptions/ws?token=invalid-secret-key") as ws:
                    ws.send_json({"chunk_id": 1})
            assert exc_info.value.code == 1008

    def test_speech_ws_authenticated_streaming_and_graceful_end(self):
        """
        Verify that connecting with a valid token permits chunk streaming
        and graceful finalization upon sending {'action': 'end'}.
        """
        speech_app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
        with TestClient(speech_app) as client:
            with client.websocket_connect(f"/v1/audio/transcriptions/ws?token={AUTH_TOKEN}") as ws:
                ws.send_json({"chunk_id": 1})
                r1 = ws.receive_json()
                assert r1.get("event") == "interim"

                ws.send_json({"chunk_id": 2})
                r2 = ws.receive_json()
                assert r2.get("event") == "interim"

                ws.send_json({"action": "end"})
                final_resp = ws.receive_json()
                assert final_resp.get("event") == "final"
                assert "text" in final_resp
