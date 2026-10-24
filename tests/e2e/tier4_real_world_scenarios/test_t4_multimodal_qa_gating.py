"""
Tier 4 Real-World Application Scenario: Enterprise Multimodal QA & Risk Gating
Validates an intelligence and document processing pipeline:
1. Ingest document images and audit logs (SSRF protected)
2. Generate dense vector embeddings for semantic search
3. Cross-encoder Rerank selects highest-relevance evidence clauses
4. VLM analyzes multimodal diagram inputs
5. Decision Engine scores compliance and legal risk
6. Structured reasoning output produced via Responses API
"""

import pytest
import httpx


@pytest.mark.tier4
class TestMultimodalQAGatingWorkflow:
    def test_full_enterprise_compliance_audit_pipeline(self, client: httpx.Client, auth_headers):
        # Step 1: Ingest document image via Asset Service
        asset_res = client.post("/v1/assets/upload", headers=auth_headers, json={"mime_type": "image/png"})
        assert asset_res.status_code == 201
        doc_s3_url = asset_res.json()["s3_url"]

        # Step 2: Dense Vector Embeddings for knowledge retrieval
        embed_res = client.post("/v1/embeddings", headers=auth_headers, json={
            "model": "qwen-embedding-8b",
            "input": [
                "Factory v4 mandatory edge Bearer token policy",
                "Two-tier queue architecture: Tier 1 native engine, Tier 2 Postgres/Redis",
                "Worker lease acquisition TTL 300 seconds and dead worker recovery sweeper"
            ]
        })
        assert embed_res.status_code == 200
        embeddings = embed_res.json()["data"]
        assert len(embeddings) == 3
        assert len(embeddings[0]["embedding"]) == 1024

        # Step 3: Cross-Encoder Document Rerank
        rerank_res = client.post("/v1/rerank", headers=auth_headers, json={
            "model": "qwen-rerank-8b",
            "query": "What happens when a GPU worker crashes mid-generation?",
            "documents": [
                "Dead-worker sweeper identifies expired leases and re-enqueues job with incremented retry count.",
                "Unauthenticated access is rejected with 401 Unauthorized.",
                "ACE-Step 1.5 runs natively in 4-6GB VRAM on an isolated port."
            ]
        })
        assert rerank_res.status_code == 200
        top_doc = rerank_res.json()["results"][0]
        assert top_doc["relevance_score"] >= 0.7
        assert "Dead-worker" in top_doc["document"]

        # Step 4: Decision Score evaluates system compliance risk
        score_res = client.post("/v1/decisions", headers=auth_headers, json={
            "model": "qwen-decision-8b",
            "task": "score",
            "context": f"Audit evaluation on worker crash handling and lease recovery: {top_doc['document']}"
        })
        assert score_res.status_code == 200
        compliance_score = score_res.json()["score"]
        assert compliance_score > 0.5

        # Step 5: Structured Responses API final report
        report_res = client.post("/v1/responses", headers=auth_headers, json={
            "model": "qwen-2.5-72b-instruct",
            "input": f"Generate formal compliance sign-off report based on score {compliance_score} and document {doc_s3_url}"
        })
        assert report_res.status_code == 200
        report = report_res.json()
        assert report["status"] == "completed"
        assert report["reasoning_tokens"] > 0

    def test_multimodal_chart_analysis_with_predicate_gate(self, client: httpx.Client, auth_headers):
        # Step 1: VLM Chat over multimodal image
        chat_res = client.post("/v1/chat/completions", headers=auth_headers, json={
            "model": "qwen-vl-max",
            "messages": [
                {"role": "user", "content": "Analyze GPU VRAM allocation telemetry chart: GPU 0 at 14GB, GPU 1 at 15GB."}
            ]
        })
        assert chat_res.status_code == 200
        vlm_analysis = chat_res.json()["choices"][0]["message"]["content"]

        # Step 2: Decision Predicate evaluates if VRAM threshold exceeded
        pred_res = client.post("/v1/decisions", headers=auth_headers, json={
            "model": "qwen-decision-8b",
            "task": "predicate",
            "context": vlm_analysis,
            "hypothesis": "Both GPUs are operating within safe memory thresholds under 16GB."
        })
        assert pred_res.status_code == 200
        assert pred_res.json()["decision"] is True

    def test_semantic_search_with_vector_similarity(self, client: httpx.Client, auth_headers):
        # Check embeddings batch consistency
        queries = ["SSRF protection in asset service", "Signed URLs for artifacts > 1MB"]
        res = client.post("/v1/embeddings", headers=auth_headers, json={
            "model": "qwen-embedding-8b",
            "input": queries
        })
        assert res.status_code == 200
        vectors = res.json()["data"]
        assert len(vectors) == 2
        assert len(vectors[0]["embedding"]) == len(vectors[1]["embedding"]) == 1024
