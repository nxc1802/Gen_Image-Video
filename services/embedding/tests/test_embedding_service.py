"""
services/embedding/tests/test_embedding_service.py
Unit and contract tests for the Vector Embedding & Rerank Daemon.
"""
import pytest
from starlette.testclient import TestClient
from factory_auth.config import AuthConfig
from services.embedding.src.app import app

AUTH_TOKEN = "sk-factory-test-token-valid"
AUTH_HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}", "Content-Type": "application/json"}


@pytest.fixture
def embed_client():
    app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
    return TestClient(app)


def test_embed_health(embed_client):
    res = embed_client.get("/health", headers=AUTH_HEADERS)
    assert res.status_code == 200
    assert res.json()["service"] == "embed-service"


def test_embeddings_single_and_batch(embed_client):
    payload = {
        "model": "qwen-embedding-8b",
        "input": ["First sentence", "Second sentence", "Third sentence"],
    }
    res = embed_client.post("/v1/embeddings", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 3
    for i, item in enumerate(data["data"]):
        assert item["index"] == i
        vec = item["embedding"]
        assert len(vec) == 1024
        # Verify L2 normalization
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 0.05
        for val in vec[:20]:
            assert -1.0 <= val <= 1.0


def test_embeddings_v2_compatibility(embed_client):
    payload = {
        "model": "qwen-embedding-8b",
        "input": "Testing v2 embed route",
    }
    res = embed_client.post("/v2/embed", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    assert len(res.json()["data"]) == 1


def test_embeddings_fail_fast_offline_model(embed_client):
    payload = {
        "model": "failing-embedding-model",
        "input": "test query",
    }
    res = embed_client.post("/v1/embeddings", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"
    # Mandatory Integrity check: zero fake vectors
    assert "data" not in res.json()


def test_embeddings_empty_input_rejected(embed_client):
    payload = {"model": "qwen-embedding-8b", "input": ""}
    res = embed_client.post("/v1/embeddings", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_REQUEST"


def test_rerank_standard_and_descending(embed_client):
    payload = {
        "model": "qwen-rerank-8b",
        "query": "How is queueing handled in Factory v4?",
        "documents": [
            "Factory v4 uses a two-tier queue with Postgres state and Redis events.",
            "Bananas are rich in potassium.",
            "ngrok static domains provide fixed API addresses.",
        ],
    }
    res = embed_client.post("/v1/rerank", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert len(data["results"]) == 3
    scores = [r["relevance_score"] for r in data["results"]]
    assert scores == sorted(scores, reverse=True)
    for score in scores:
        assert 0.0 <= score <= 1.0
    # Top result should be the queue document
    assert "two-tier queue" in data["results"][0]["document"]


def test_rerank_legacy_alias(embed_client):
    payload = {
        "model": "qwen-rerank-8b",
        "query": "GPU allocation policy",
        "documents": ["Dual T4 GPU setup on Kaggle", "Coffee roasting guide"],
    }
    res = embed_client.post("/rerank", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    assert len(res.json()["results"]) == 2


def test_rerank_fail_fast_offline_model(embed_client):
    payload = {
        "model": "failing-rerank-model",
        "query": "q",
        "documents": ["doc1", "doc2"],
    }
    res = embed_client.post("/v1/rerank", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"
    # Mandatory Integrity check: zero mock results
    assert "results" not in res.json()


def test_rerank_empty_query_rejected(embed_client):
    payload = {"model": "qwen-rerank-8b", "query": "", "documents": ["doc"]}
    res = embed_client.post("/v1/rerank", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_REQUEST"


def test_embeddings_matryoshka_dimensions_and_strict_l2(embed_client):
    """Verifies Matryoshka dimension truncation and strict unit L2 norm."""
    for dim in (256, 512, 1024):
        payload = {
            "model": "qwen-embedding-8b",
            "input": ["Testing Matryoshka dimension projection"],
            "dimensions": dim,
        }
        res = embed_client.post("/v1/embeddings", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        vec = res.json()["data"][0]["embedding"]
        assert len(vec) == dim
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-4


def test_embeddings_semantic_dispersion_not_clustering(embed_client):
    """Verifies that unrelated texts exhibit true semantic dispersion (do NOT cluster to ~0.94)."""
    payload = {
        "model": "qwen-embedding-8b",
        "input": [
            "The quick brown fox jumps over the lazy dog.",
            "Quantum electrodynamics is a relativistic quantum field theory of electrodynamics.",
            "The fast brown fox leaps over a sleepy hound.",
        ],
    }
    res = embed_client.post("/v1/embeddings", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    v_fox1 = res.json()["data"][0]["embedding"]
    v_quantum = res.json()["data"][1]["embedding"]
    v_fox2 = res.json()["data"][2]["embedding"]

    def cos_sim(a, b):
        return sum(x * y for x, y in zip(a, b))

    sim_unrelated = cos_sim(v_fox1, v_quantum)
    sim_related = cos_sim(v_fox1, v_fox2)

    # Invariant: Unrelated concepts must NOT cluster to 0.94, but have near-zero cosine similarity
    assert sim_unrelated < 0.35, f"Unrelated similarity {sim_unrelated} too high (indicates false clustering)"
    # Invariant: Semantically related texts must have significantly higher similarity than unrelated
    assert sim_related > sim_unrelated + 0.15


def test_rerank_top_n_truncation(embed_client):
    """Verifies that top_n parameter truncates rerank results properly."""
    payload = {
        "model": "qwen-rerank-8b",
        "query": "How is queueing handled in Factory v4?",
        "documents": [
            "Factory v4 uses a two-tier queue with Postgres state and Redis events.",
            "Random unrelated sentence about astronomy.",
            "Fixed ngrok networking addresses.",
        ],
        "top_n": 1,
    }
    res = embed_client.post("/v1/rerank", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 200
    results = res.json()["results"]
    assert len(results) == 1
    assert "two-tier queue" in results[0]["document"]
