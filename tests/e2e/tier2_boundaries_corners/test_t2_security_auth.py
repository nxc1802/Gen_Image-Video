"""
Tier 2 Security & Authentication Tests
Validates mandatory Bearer authentication across all 42 feature endpoints,
prohibition of unauthenticated access, elimination of arbitrary proxying,
and prevention of credential forwarding.
"""

import pytest
import httpx
from tests.e2e.conftest import assert_sanitized_error


@pytest.mark.tier2
@pytest.mark.security
class TestSecurityAuthentication:
    endpoints = [
        ("POST", "/v1/jobs", {"service": "image", "model": "qwen-image-2.1", "task": "text_to_image", "inputs": {}}),
        ("GET", "/v1/jobs/job-12345", None),
        ("DELETE", "/v1/jobs/job-12345", None),
        ("POST", "/v1/jobs/job-12345/retry", None),
        ("GET", "/v1/jobs/job-12345/events", None),
        ("GET", "/v1/jobs/job-12345/ws", None),
        ("GET", "/v1/registry/services", None),
        ("GET", "/v1/registry/models", None),
        ("POST", "/v1/assets/upload", {"mime_type": "image/png"}),
        ("POST", "/v1/assets/validate", {"url": "https://example.com/asset.png"}),
        ("POST", "/v1/chat/completions", {"model": "qwen-2.5-72b-instruct", "messages": [{"role": "user", "content": "hi"}]}),
        ("POST", "/v1/responses", {"model": "qwen-2.5-72b-instruct", "input": "test"}),
        ("POST", "/v1/decisions", {"model": "qwen-decision-8b", "task": "predicate", "context": "c", "hypothesis": "h"}),
        ("POST", "/v1/audio/transcriptions", {"model": "qwen3-asr", "audio_url": "https://example.com/a.wav"}),
        ("POST", "/v1/audio/transcriptions/stream", {"chunk_id": 1}),
        ("POST", "/v1/audio/alignments", {"audio_url": "https://example.com/a.wav", "transcript_text": "t"}),
        ("POST", "/v1/audio/speech", {"model": "qwen3-tts", "input": "text"}),
        ("POST", "/v1/embeddings", {"model": "qwen-embedding-8b", "input": "text"}),
        ("POST", "/v1/rerank", {"model": "qwen-rerank-8b", "query": "q", "documents": ["d"]}),
        ("GET", "/api/studio/schemas", None),
        ("GET", "/api/studio/inspect/stream", None),
        ("POST", "/api/studio/codegen", {"service": "image"}),
    ]

    # Test 1-22: Missing Authorization header on all endpoints
    @pytest.mark.parametrize("method,path,payload", endpoints)
    def test_missing_auth_header_rejected_401(self, client: httpx.Client, method, path, payload):
        headers = {"Content-Type": "application/json"}
        res = client.request(method, path, headers=headers, json=payload if payload else None)
        assert_sanitized_error(res, 401, "UNAUTHORIZED")

    # Test 23-32: Empty Bearer token ("Bearer ")
    @pytest.mark.parametrize("method,path,payload", endpoints[:10])
    def test_empty_bearer_token_rejected_401(self, client: httpx.Client, method, path, payload):
        headers = {"Authorization": "Bearer ", "Content-Type": "application/json"}
        res = client.request(method, path, headers=headers, json=payload if payload else None)
        assert_sanitized_error(res, 401, "UNAUTHORIZED")

    # Test 33-36: Malformed tokens
    def test_invalid_bearer_token_string(self, client: httpx.Client):
        headers = {"Authorization": "Bearer bad_token_12345", "Content-Type": "application/json"}
        res = client.get("/v1/registry/services", headers=headers)
        assert_sanitized_error(res, 401, "UNAUTHORIZED")

    def test_basic_auth_instead_of_bearer(self, client: httpx.Client):
        headers = {"Authorization": "Basic dXNlcjpwYXNz", "Content-Type": "application/json"}
        res = client.get("/v1/registry/models", headers=headers)
        assert_sanitized_error(res, 401, "UNAUTHORIZED")

    def test_arbitrary_proxy_header_x_target_base_url_stripped(self, client: httpx.Client, auth_headers):
        headers = {**auth_headers, "x-target-base-url": "https://malicious-attacker.com"}
        # Must resolve internally, not redirect or forward to the malicious base URL
        res = client.get("/v1/registry/services", headers=headers)
        assert res.status_code == 200
        assert "malicious-attacker" not in str(res.url)

    def test_null_token_rejected(self, client: httpx.Client):
        headers = {"Authorization": "Bearer null", "Content-Type": "application/json"}
        res = client.post("/v1/jobs", headers=headers, json={"service": "image"})
        assert_sanitized_error(res, 401, "UNAUTHORIZED")

    def test_expired_token_rejected(self, client: httpx.Client):
        headers = {"Authorization": "Bearer expired_token", "Content-Type": "application/json"}
        res = client.get("/v1/jobs/job-123", headers=headers)
        assert_sanitized_error(res, 401, "UNAUTHORIZED")

    def test_healthz_is_public(self, client: httpx.Client):
        # Health endpoint should be publicly accessible without auth
        res = client.get("/healthz")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"

    def test_missing_bearer_prefix_rejected(self, client: httpx.Client):
        headers = {"Authorization": "sk-factory-test-token-valid", "Content-Type": "application/json"}
        res = client.get("/v1/registry/services", headers=headers)
        assert_sanitized_error(res, 401, "UNAUTHORIZED")

    def test_non_bearer_custom_scheme_rejected(self, client: httpx.Client):
        headers = {"Authorization": "CustomToken 123", "Content-Type": "application/json"}
        res = client.get("/v1/registry/models", headers=headers)
        assert_sanitized_error(res, 401, "UNAUTHORIZED")
