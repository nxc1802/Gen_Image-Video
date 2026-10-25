"""
services/embedding/tests/test_embedding_service.py
Unit and contract tests for the Vector Embedding & Rerank Daemon.
"""
from unittest.mock import MagicMock, patch

import pytest
from factory_auth.config import AuthConfig
from services.embedding.src.app import app
from services.embedding.src.engine import EmbeddingEngine
from starlette.testclient import TestClient

AUTH_TOKEN = "sk-factory-test-token-valid"
AUTH_HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}", "Content-Type": "application/json"}


@pytest.fixture
def embed_client():
    app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")
    app.state.engine = EmbeddingEngine()
    return TestClient(app)


def test_embed_health(embed_client):
    res = embed_client.get("/health", headers=AUTH_HEADERS)
    assert res.status_code == 200
    assert res.json()["service"] == "embed-service"


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


def test_embeddings_unconfigured_fail_closed(embed_client):
    payload = {
        "model": "qwen-embedding-8b",
        "input": ["First sentence", "Second sentence", "Third sentence"],
    }
    res = embed_client.post("/v1/embeddings", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_embeddings_v2_unconfigured_fail_closed(embed_client):
    payload = {
        "model": "qwen-embedding-8b",
        "input": "Testing v2 embed route",
    }
    res = embed_client.post("/v2/embed", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_embeddings_configured_proxy(embed_client):
    app.state.engine.text_embedder.adapter.upstream_url = "http://mock-vllm-embed:8000"
    payload = {
        "model": "qwen-embedding-8b",
        "input": ["First sentence", "Second sentence"],
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "object": "list",
        "model": "qwen-embedding-8b",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1] * 1024},
            {"object": "embedding", "index": 1, "embedding": [0.2] * 1024},
        ],
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        res = embed_client.post("/v1/embeddings", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert len(data["data"][0]["embedding"]) == 1024


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


def test_rerank_unconfigured_fail_closed(embed_client):
    payload = {
        "model": "qwen-rerank-8b",
        "query": "How is queueing handled in Factory v4?",
        "documents": [
            "Factory v4 uses a two-tier queue with Postgres state and Redis events.",
            "Bananas are rich in potassium.",
        ],
    }
    res = embed_client.post("/v1/rerank", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_rerank_legacy_unconfigured_fail_closed(embed_client):
    payload = {
        "model": "qwen-rerank-8b",
        "query": "GPU allocation policy",
        "documents": ["Dual T4 GPU setup on Kaggle", "Coffee roasting guide"],
    }
    res = embed_client.post("/rerank", headers=AUTH_HEADERS, json=payload)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_rerank_configured_proxy_and_truncation(embed_client):
    app.state.engine.reranker.adapter.upstream_url = "http://mock-tei-rerank:8000"
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

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "qwen-rerank-8b",
        "results": [
            {"index": 0, "score": 0.95},
            {"index": 2, "score": 0.40},
            {"index": 1, "score": 0.05},
        ],
    }

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        res = embed_client.post("/v1/rerank", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["index"] == 0
        assert data["results"][0]["relevance_score"] == 0.95
        assert "two-tier queue" in data["results"][0]["document"]
