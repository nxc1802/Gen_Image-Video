"""
Tier 2 Error Sanitization & Information Leakage Defense Tests
Validates that all error responses strictly adhere to the Unified Error Contract
and that zero internal tracebacks, Python exception classes, or filesystem paths leak.
"""

import pytest
import httpx
from tests.e2e.conftest import assert_sanitized_error


@pytest.mark.tier2
class TestErrorSanitization:
    # Tests 1-10: 404 Not Found sanitization across resources
    @pytest.mark.parametrize("bad_path", [
        "/v1/jobs/job-non-existent-12345",
        "/v1/jobs/invalid-format",
        "/v1/registry/unknown-resource",
        "/v1/unknown-endpoint",
        "/api/nonexistent",
        "/v1/models/does-not-exist",
        "/v1/services/null",
        "/v1/admin/debug",
        "/internal/vram/dump",
        "/debug/trace",
    ])
    def test_404_error_sanitized(self, client: httpx.Client, auth_headers, bad_path):
        res = client.get(bad_path, headers=auth_headers)
        assert_sanitized_error(res, 404, "NOT_FOUND")

    # Tests 11-15: Malformed JSON syntax handling
    def test_malformed_json_body_on_jobs(self, client: httpx.Client, auth_headers):
        headers = {**auth_headers, "Content-Type": "application/json"}
        res = client.post("/v1/jobs", headers=headers, content=b"{malformed json,")
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    def test_malformed_json_body_on_chat(self, client: httpx.Client, auth_headers):
        headers = {**auth_headers, "Content-Type": "application/json"}
        res = client.post("/v1/chat/completions", headers=headers, content=b"['unclosed array")
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    def test_malformed_json_body_on_decisions(self, client: httpx.Client, auth_headers):
        headers = {**auth_headers, "Content-Type": "application/json"}
        res = client.post("/v1/decisions", headers=headers, content=b'{"unterminated string')
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    def test_malformed_json_body_on_assets(self, client: httpx.Client, auth_headers):
        headers = {**auth_headers, "Content-Type": "application/json"}
        res = client.post("/v1/assets/validate", headers=headers, content=b"###not-json###")
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    def test_malformed_json_body_on_embeddings(self, client: httpx.Client, auth_headers):
        headers = {**auth_headers, "Content-Type": "application/json"}
        res = client.post("/v1/embeddings", headers=headers, content=b"<xml></xml>")
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    # Tests 16-25: Structural integrity of error format and headers
    @pytest.mark.parametrize("status,code,test_call", [
        (401, "UNAUTHORIZED", lambda c: c.get("/v1/registry/services")),
        (404, "NOT_FOUND", lambda c, h: c.get("/v1/jobs/missing", headers=h)),
        (400, "INVALID_REQUEST", lambda c, h: c.post("/v1/assets/validate", headers=h, json={"url": ""})),
        (400, "SSRF_BLOCKED", lambda c, h: c.post("/v1/assets/validate", headers=h, json={"url": "http://127.0.0.1"})),
        (415, "UNSUPPORTED_MEDIA_TYPE", lambda c, h: c.post("/v1/assets/upload", headers=h, json={"mime_type": "text/html"})),
        (503, "MODEL_UNAVAILABLE", lambda c, h: c.post("/v1/embeddings", headers=h, json={"model": "offline-model", "input": "x"})),
        (400, "CAPABILITY_NOT_SUPPORTED", lambda c, h: c.post("/v1/jobs", headers=h, json={"service": "image", "model": "experimental-sketch-partial", "task": "inpaint", "inputs": {}})),
        (413, "PAYLOAD_TOO_LARGE", lambda c, h: c.post("/v1/assets/upload", headers={**h, "Content-Length": str(60*1024*1024)}, json={"mime_type": "image/png"})),
        (400, "INVALID_REQUEST", lambda c, h: c.post("/v1/decisions", headers=h, json={"model": "qwen-decision-8b", "task": "choice", "context": "c", "choices": []})),
        (400, "INVALID_REQUEST", lambda c, h: c.post("/v1/jobs", headers=h, json={"service": "image", "model": "qwen-image-2.1", "task": "text_to_image", "inputs": {"width": 5000}})),
    ])
    def test_error_headers_and_fields(self, client: httpx.Client, auth_headers, status, code, test_call):
        try:
            res = test_call(client, auth_headers)
        except TypeError:
            res = test_call(client)
        assert res.status_code == status
        body = res.json()
        assert "error" in body
        err = body["error"]
        assert err["code"] == code
        assert "message" in err
        assert "request_id" in err
        assert "timestamp" in err
        assert "details" in err
        # Header x-request-id correlation
        assert res.headers.get("x-request-id") == err["request_id"]

    # Tests 26-30: Strict regex assertion against internal leaking
    def test_no_file_paths_in_400(self, client: httpx.Client, auth_headers):
        res = client.post("/v1/jobs", headers=auth_headers, json={"service": "unknown"})
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    def test_no_stack_trace_in_503(self, client: httpx.Client, auth_headers):
        res = client.post("/v1/decisions", headers=auth_headers, json={"model": "offline-decision-model", "task": "score", "context": "c"})
        assert_sanitized_error(res, 503, "MODEL_UNAVAILABLE")

    def test_no_python_exception_names_in_404(self, client: httpx.Client, auth_headers):
        res = client.delete("/v1/jobs/missing-job-xyz", headers=auth_headers)
        assert_sanitized_error(res, 404, "NOT_FOUND")

    def test_no_os_user_directories_in_ssrf(self, client: httpx.Client, auth_headers):
        res = client.post("/v1/assets/validate", headers=auth_headers, json={"url": "http://10.0.0.1"})
        assert_sanitized_error(res, 400, "SSRF_BLOCKED")

    def test_no_traceback_on_unauthorized(self, client: httpx.Client):
        res = client.get("/v1/registry/models")
        assert_sanitized_error(res, 401, "UNAUTHORIZED")
