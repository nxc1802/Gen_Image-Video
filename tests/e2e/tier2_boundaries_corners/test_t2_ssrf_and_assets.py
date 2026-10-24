"""
Tier 2 SSRF Defense & Asset Validation Tests
Validates SSRF protection against loopback, private RFC1918 ranges, cloud metadata,
file scheme attacks, payload size ceilings, and media type filtering.
"""

import pytest
import httpx
from tests.e2e.conftest import assert_sanitized_error


@pytest.mark.tier2
@pytest.mark.ssrf
class TestSSRFAndAssetValidation:
    ssrf_malicious_urls = [
        "http://127.0.0.1:8000/internal",
        "http://127.0.0.2:8080/metrics",
        "http://127.1.2.3:8188/prompt",
        "http://localhost:8000/api",
        "http://localhost:8188/history",
        "http://[::1]:8080/debug",
        "http://10.0.0.1/admin",
        "http://10.10.1.5:8000/internal",
        "http://10.254.0.1/keys",
        "http://172.16.0.1/secrets",
        "http://172.20.10.1:9000/minio",
        "http://172.31.255.255/internal",
        "http://192.168.0.1/router",
        "http://192.168.1.1:8000/admin",
        "http://192.168.100.50/credentials",
        "http://169.254.169.254/latest/meta-data/",
        "http://169.254.169.254/computeMetadata/v1/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://0.0.0.0:8000/internal",
    ]

    # Tests 1-19: Parameterized SSRF URL rejection
    @pytest.mark.parametrize("target_url", ssrf_malicious_urls)
    def test_ssrf_url_blocked(self, client: httpx.Client, auth_headers, target_url):
        payload = {"url": target_url, "mime_type": "image/png"}
        res = client.post("/v1/assets/validate", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "SSRF_BLOCKED")

    # Tests 20-23: Invalid protocols & non-HTTP schemes
    @pytest.mark.parametrize("bad_scheme", [
        "file:///etc/passwd",
        "ftp://ftp.example.com/malware.exe",
        "gopher://gopher.floodgap.com/",
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA"
    ])
    def test_invalid_url_schemes_rejected(self, client: httpx.Client, auth_headers, bad_scheme):
        payload = {"url": bad_scheme, "mime_type": "image/png"}
        res = client.post("/v1/assets/validate", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    # Tests 24-28: Unsupported MIME types rejected with 415
    @pytest.mark.parametrize("bad_mime", [
        "text/html",
        "application/x-executable",
        "application/x-sh",
        "application/javascript",
        "application/zip"
    ])
    def test_unsupported_mime_types_rejected_415(self, client: httpx.Client, auth_headers, bad_mime):
        payload = {"mime_type": bad_mime}
        res = client.post("/v1/assets/upload", headers=auth_headers, json=payload)
        assert_sanitized_error(res, 415, "UNSUPPORTED_MEDIA_TYPE")

    # Tests 29-31: Oversized payload body ceilings (>50MB)
    def test_oversized_payload_header_rejected_413(self, client: httpx.Client, auth_headers):
        headers = {**auth_headers, "Content-Length": str(55 * 1024 * 1024)}
        res = client.post("/v1/assets/upload", headers=headers, json={"mime_type": "image/png"})
        assert_sanitized_error(res, 413, "PAYLOAD_TOO_LARGE")

    def test_oversized_payload_boundary_51mb(self, client: httpx.Client, auth_headers):
        headers = {**auth_headers, "Content-Length": str(51 * 1024 * 1024)}
        res = client.post("/v1/assets/upload", headers=headers, json={"mime_type": "video/mp4"})
        assert_sanitized_error(res, 413, "PAYLOAD_TOO_LARGE")

    def test_oversized_payload_extreme_100mb(self, client: httpx.Client, auth_headers):
        headers = {**auth_headers, "Content-Length": str(100 * 1024 * 1024)}
        res = client.post("/v1/assets/upload", headers=headers, json={"mime_type": "video/mp4"})
        assert_sanitized_error(res, 413, "PAYLOAD_TOO_LARGE")

    # Tests 32-35: Empty or malformed asset validation payloads
    def test_asset_validate_empty_url(self, client: httpx.Client, auth_headers):
        res = client.post("/v1/assets/validate", headers=auth_headers, json={"url": ""})
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    def test_asset_validate_missing_url(self, client: httpx.Client, auth_headers):
        res = client.post("/v1/assets/validate", headers=auth_headers, json={"mime_type": "image/png"})
        assert_sanitized_error(res, 400, "INVALID_REQUEST")

    def test_asset_upload_empty_mime(self, client: httpx.Client, auth_headers):
        res = client.post("/v1/assets/upload", headers=auth_headers, json={"mime_type": ""})
        assert_sanitized_error(res, 415, "UNSUPPORTED_MEDIA_TYPE")

    def test_asset_validate_whitespace_url(self, client: httpx.Client, auth_headers):
        res = client.post("/v1/assets/validate", headers=auth_headers, json={"url": "   "})
        assert_sanitized_error(res, 400, "INVALID_REQUEST")
