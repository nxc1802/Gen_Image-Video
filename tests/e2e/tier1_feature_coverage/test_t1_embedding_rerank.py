"""
Tier 1 Feature Coverage: Embedding & Rerank Subsystem (Features 19 - 20)
Validates happy path and contract adherence for:
- Feature 19: Vector Embeddings (POST /v1/embeddings, POST /v2/embed)
- Feature 20: Document Rerank (POST /rerank, POST /v1/rerank)
"""

import pytest
import httpx


@pytest.mark.tier1
class TestEmbeddingAndRerank:
    # --- Feature 19: Vector Embeddings ---
    def test_f19_embeddings_single_text(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-embedding-8b",
            "input": "Factory v4 decoupled AI architectures"
        }
        res = client.post("/v1/embeddings", headers=auth_headers, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert "embedding" in data["data"][0]

    def test_f19_embeddings_batch_text(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-embedding-8b",
            "input": ["First sentence", "Second sentence", "Third sentence"]
        }
        res = client.post("/v1/embeddings", headers=auth_headers, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert len(data["data"]) == 3
        for i, item in enumerate(data["data"]):
            assert item["index"] == i
            assert len(item["embedding"]) == 1024

    def test_f19_embeddings_vector_dimension(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-embedding-8b",
            "input": "Dimension check"
        }
        res = client.post("/v1/embeddings", headers=auth_headers, json=payload)
        assert res.status_code == 200
        vec = res.json()["data"][0]["embedding"]
        assert len(vec) == 1024

    def test_f19_embeddings_float_types(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-embedding-8b",
            "input": "Data integrity check"
        }
        res = client.post("/v1/embeddings", headers=auth_headers, json=payload)
        vec = res.json()["data"][0]["embedding"]
        for val in vec[:20]:
            assert isinstance(val, float)
            assert -1.0 <= val <= 1.0

    def test_f19_embeddings_v2_endpoint_compatibility(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-embedding-8b",
            "input": "Testing v2 embed route"
        }
        res = client.post("/v2/embed", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert len(res.json()["data"]) == 1

    # --- Feature 20: Document Rerank ---
    def test_f20_rerank_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-rerank-8b",
            "query": "How is queueing handled in Factory v4?",
            "documents": [
                "Factory v4 uses a two-tier queue with Postgres state and Redis events.",
                "Bananas are rich in potassium.",
                "ngrok static domains provide fixed API addresses."
            ]
        }
        res = client.post("/v1/rerank", headers=auth_headers, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "results" in data
        assert len(data["results"]) == 3

    def test_f20_rerank_scores_bounded(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-rerank-8b",
            "query": "GPU allocation policy",
            "documents": ["Dual T4 GPU setup on Kaggle", "Coffee roasting guide"]
        }
        res = client.post("/v1/rerank", headers=auth_headers, json=payload)
        assert res.status_code == 200
        for item in res.json()["results"]:
            assert "relevance_score" in item
            assert 0.0 <= item["relevance_score"] <= 1.0

    def test_f20_rerank_sorted_descending(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-rerank-8b",
            "query": "Optimal serving framework",
            "documents": ["vLLM for text", "Irrelevant snippet", "Random text"]
        }
        res = client.post("/v1/rerank", headers=auth_headers, json=payload)
        results = res.json()["results"]
        scores = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_f20_rerank_document_reflection(self, client: httpx.Client, auth_headers):
        doc = "ComfyUI on GPU 0 handles image generation workflows."
        payload = {
            "model": "qwen-rerank-8b",
            "query": "Image serving framework",
            "documents": [doc]
        }
        res = client.post("/v1/rerank", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert res.json()["results"][0]["document"] == doc

    def test_f20_rerank_legacy_endpoint_compatibility(self, client: httpx.Client, auth_headers):
        payload = {
            "model": "qwen-rerank-8b",
            "query": "Testing /rerank legacy alias",
            "documents": ["Target doc A", "Target doc B"]
        }
        res = client.post("/rerank", headers=auth_headers, json=payload)
        assert res.status_code == 200
        assert len(res.json()["results"]) == 2
