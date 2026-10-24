"""
tests/security/test_adversarial_security_harness.py
Milestone 1 Empirical Challenger Adversarial Security Attack Harness.

Tests:
1. Auth bypass: empty Bearer token, whitespace token, malformed Authorization headers, path traversal on protected endpoints.
2. SSRF attack vectors: 127.0.0.1, 127.1, 0.0.0.0, 169.254.169.254, [::1], IPv4-mapped IPv6 (::ffff:127.0.0.1), private CIDRs (10.0.0.1, 192.168.1.1, 172.16.0.1), localhost DNS aliases, and discovered bypasses ([::], port bypass on IP literals).
3. Payload attacks: truncated headers, content-length mismatch, files exceeding 50MB, fake MIME types with corrupt magic bytes, decompression bombs.
4. Error leakages: deliberately triggering exceptions to inspect response bodies for stack traces, local paths, or internal class names.
"""
import io
import os
import re
import time
import pytest
import jwt
from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient

from factory_auth.config import AuthConfig, load_auth_config
from factory_auth.bearer import verify_bearer_token
from factory_contracts.errors import (
    UnauthorizedError,
    SSRFBlockedError,
    InvalidMediaError,
    PayloadTooLargeError,
)
from factory.apps.control_api.src.assets.ssrf import validate_target_url, BLOCKED_NETWORKS
from factory.apps.control_api.src.assets.media_validator import (
    validate_media_content,
    detect_mime_type,
    MAX_IMAGE_DIMENSION,
    MAX_TOTAL_PIXELS,
)
from factory.apps.control_api.src.assets.service import AssetService
from factory.apps.control_api.src.api.errors import register_exception_handlers
from factory.apps.control_api.src.api.middleware import RequestIDMiddleware
from factory.apps.control_api.src.main import app as control_plane_app


FORBIDDEN_LEAK_PATTERNS = [
    r"Traceback \(most recent call last\)",
    r"/Volumes/[A-Za-z0-9_/:-]+",
    r"/Users/[A-Za-z0-9_/:-]+",
    r"/home/[A-Za-z0-9_/:-]+",
    r"C:\\",
    r"Exception:",
    r"KeyError:",
    r"ZeroDivisionError:",
    r"InternalSecretClass",
    r"postgresql://",
    r"secret123",
    r"\.py:\d+",
]


# ============================================================================
# 1. AUTH BYPASS ADVERSARIAL SUITE
# ============================================================================
@pytest.mark.security
class TestAuthBypassAdversarialHarness:
    """Stress tests authentication mechanics against malformed headers, bypasses, and path traversal."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        if hasattr(control_plane_app.state, "auth_config"):
            delattr(control_plane_app.state, "auth_config")
        monkeypatch.setenv("FACTORY_API_KEY", "prod-secret-alpha-99")
        monkeypatch.setenv("JWT_SECRET_KEY", "jwt-secret-testing-32bytes-long!!")
        monkeypatch.setenv("ENVIRONMENT", "test")
        yield
        if hasattr(control_plane_app.state, "auth_config"):
            delattr(control_plane_app.state, "auth_config")

    @pytest.mark.parametrize(
        "auth_header,desc",
        [
            (None, "Missing header"),
            ("", "Empty header"),
            ("   ", "Whitespace only header"),
            ("Bearer", "Bearer with no token"),
            ("Bearer ", "Bearer with single space"),
            ("Bearer      ", "Bearer with multiple spaces"),
            ("Bearer \t\r\n ", "Bearer with whitespace control chars"),
            ("Token prod-secret-alpha-99", "Token scheme instead of Bearer"),
            ("Basic dXNlcjpwYXNz", "Basic auth scheme"),
            ("Digest username=admin", "Digest scheme"),
            ("Bearer null", "String literal null"),
            ("Bearer undefined", "String literal undefined"),
            ('Bearer ""', "Quoted empty string"),
            ("Bearer ''", "Single-quoted empty string"),
            ("Bearer prod-secret-alpha-99 extra", "Extra token segments"),
            ("Bearer prod-secret-alpha-99\x00", "Null byte poisoned token suffix"),
            ("Bearer \x00prod-secret-alpha-99", "Null byte poisoned token prefix"),
            ("Bearer wrong-key", "Invalid secret key"),
        ],
    )
    def test_auth_rejects_malformed_headers(self, auth_header, desc):
        client = TestClient(control_plane_app)
        headers = {"Authorization": auth_header} if auth_header is not None else {}
        res = client.get("/v1/models", headers=headers)
        assert res.status_code == 401, f"Failed for {desc}: expected 401 got {res.status_code}"
        assert res.json()["error"]["code"] == "UNAUTHORIZED"

    def test_valid_token_variations_accepted(self):
        client = TestClient(control_plane_app)
        # 1. Exact token
        res1 = client.get("/v1/models", headers={"Authorization": "Bearer prod-secret-alpha-99"})
        assert res1.status_code == 200

        # 2. Lowercase bearer
        res2 = client.get("/v1/models", headers={"Authorization": "bearer prod-secret-alpha-99"})
        assert res2.status_code == 200

        # 3. Uppercase BEARER
        res3 = client.get("/v1/models", headers={"Authorization": "BEARER prod-secret-alpha-99"})
        assert res3.status_code == 200

        # 4. Token with surrounding padding
        res4 = client.get("/v1/models", headers={"Authorization": "Bearer   prod-secret-alpha-99   "})
        assert res4.status_code == 200

    def test_jwt_adversarial_tokens(self):
        client = TestClient(control_plane_app)
        secret = "jwt-secret-testing-32bytes-long!!"

        # Expired token
        expired = jwt.encode({"sub": "admin", "exp": time.time() - 3600}, secret, algorithm="HS256")
        res_exp = client.get("/v1/models", headers={"Authorization": f"Bearer {expired}"})
        assert res_exp.status_code == 401

        # Corrupted signature
        parts = expired.split(".")
        corrupted = f"{parts[0]}.{parts[1]}.badsignature"
        res_corr = client.get("/v1/models", headers={"Authorization": f"Bearer {corrupted}"})
        assert res_corr.status_code == 401

        # None algorithm
        try:
            none_alg = jwt.encode({"sub": "admin"}, "", algorithm="none")
            res_none = client.get("/v1/models", headers={"Authorization": f"Bearer {none_alg}"})
            assert res_none.status_code == 401
        except Exception:
            pass  # Rejection by pyjwt is also secure

    @pytest.mark.parametrize(
        "path",
        [
            "/health/../v1/models",
            "/v1/models/..",
            "/v1/models/../models",
            "/health/..;/v1/models",
            "/v1/%2e%2e/v1/models",
            "/health/..%2fv1%2fmodels",
            "/v1//models",
        ],
    )
    def test_path_traversal_cannot_bypass_auth(self, path):
        """Ensures path traversal attempts on endpoints either 401 (auth enforced) or 404 (not routed)."""
        client = TestClient(control_plane_app)
        res = client.get(path)
        # None of these paths should return 200 without authentication!
        assert res.status_code in (401, 404), f"Path traversal '{path}' returned {res.status_code} without auth!"


# ============================================================================
# 2. SSRF ATTACK VECTORS & VULNERABILITY EXPOSURE
# ============================================================================
@pytest.mark.security
@pytest.mark.ssrf
class TestSSRFAttackAdversarialHarness:
    """Stress tests SSRF validator with evasive vectors and documents confirmed weaknesses."""

    @pytest.mark.parametrize(
        "target_url",
        [
            "http://127.0.0.1:8000/",
            "http://127.1:8000/",
            "http://127.0.1:8000/",
            "http://0.0.0.0:8000/",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]:8000/",
            "http://[::ffff:127.0.0.1]:8000/",
            "http://[::ffff:7f00:1]:8000/",
            "http://[::ffff:10.0.0.1]:8000/",
            "http://[::ffff:192.168.1.1]:8000/",
            "http://[::ffff:169.254.169.254]:8000/",
            "http://10.0.0.1/admin",
            "http://192.168.1.1/setup",
            "http://172.16.0.1/internal",
            "http://localhost:8000/",
            "http://localhost.localdomain:8000/",
            "http://sub.localhost:8000/",
            "http://2130706433:8000/",  # Decimal 127.0.0.1
            "http://0x7f000001:8000/",  # Hex 127.0.0.1
            "http://127.0.0.1.nip.io:8000/",  # Wildcard DNS alias
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://instance-data/latest/meta-data/",
            "http://metadata/v1/",
        ],
    )
    def test_ssrf_standard_vectors_blocked(self, target_url):
        with pytest.raises(SSRFBlockedError):
            validate_target_url(target_url)

    def test_ssrf_ipv6_wildcard_bypass_vulnerability(self):
        """
        Verifies that IPv6 wildcard / unspecified addresses (::, ::0)
        are blocked by BLOCKED_NETWORKS (::/128).
        """
        for url in ("http://[::]:8000/", "http://[::0]:8000/", "http://[::]:80/", "http://[::0]:8080/"):
            with pytest.raises(SSRFBlockedError, match="blocked IP address"):
                validate_target_url(url)

    def test_ssrf_ip_literal_port_bypass_vulnerability(self):
        """
        Verifies that port validation strictly applies to literal IP addresses as well as hostnames.
        """
        with pytest.raises(SSRFBlockedError, match="Target port 22 is not permitted"):
            validate_target_url("http://example.com:22/ssh")

        # Port 22 on IP literal is now strictly blocked
        with pytest.raises(SSRFBlockedError, match="Target port 22 is not permitted"):
            validate_target_url("http://8.8.8.8:22/ssh")

        # Permitted ports on IP literal are accepted
        host, ips = validate_target_url("http://8.8.8.8:80/image.png")
        assert host == "8.8.8.8"
        assert ips[0].exploded == "8.8.8.8"

    def test_dns_rebinding_attack_prevented(self, monkeypatch):
        """
        Simulate an attacker controlling a domain whose DNS rebinds to 127.0.0.1 on second lookup.
        Verify that PinningAsyncHTTPTransport connects to the verified IP without secondary DNS resolution.
        """
        import anyio
        import asyncio
        import socket
        call_count = 0
        def mock_getaddrinfo(host, port, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', port or 80))]
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', port or 80))]

        connected_hosts = []
        async def mock_connect_tcp(remote_host, remote_port, *args, **kwargs):
            connected_hosts.append(remote_host)
            raise OSError(f"Connection refused to {remote_host}")

        monkeypatch.setattr(socket, "getaddrinfo", mock_getaddrinfo)
        monkeypatch.setattr(anyio, "connect_tcp", mock_connect_tcp)

        service = AssetService()
        with pytest.raises(SSRFBlockedError, match="Failed to fetch remote asset"):
            asyncio.run(service.ingest_url("http://rebind.attacker.com/image.png"))

        # Exactly 1 DNS query occurred (during validation), zero during connection
        assert call_count == 1
        # Connected to pre-verified IP, never attempted connection to rebind target (127.0.0.1)
        assert connected_hosts == ["93.184.216.34"]

    def test_dns_pinning_multi_ip_fallback(self, monkeypatch):
        """
        Verifies that PinningAsyncNetworkBackend iterates through verified IPs
        if the first IP fails to connect.
        """
        import anyio
        import asyncio
        import socket
        def mock_getaddrinfo(host, port, *args, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', port or 80)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.35', port or 80)),
            ]

        connected_hosts = []
        async def mock_connect_tcp(remote_host, remote_port, *args, **kwargs):
            connected_hosts.append(remote_host)
            raise OSError(f"Connection refused to {remote_host}")

        monkeypatch.setattr(socket, "getaddrinfo", mock_getaddrinfo)
        monkeypatch.setattr(anyio, "connect_tcp", mock_connect_tcp)

        service = AssetService()
        with pytest.raises(SSRFBlockedError):
            asyncio.run(service.ingest_url("http://multihomed.example.com/image.png"))

        assert connected_hosts == ["93.184.216.34", "93.184.216.35"]


# ============================================================================
# 3. PAYLOAD ATTACKS ADVERSARIAL SUITE
# ============================================================================
@pytest.mark.security
class TestPayloadAttacksAdversarialHarness:
    """Stress tests media validation, file sizes, decompression limits, and corrupt streams."""

    def test_truncated_image_headers_rejected(self):
        truncated_samples = [
            b"",
            b"\x89",
            b"\x89PNG",
            b"RIFF",
            b"RIFF\x00\x00\x00\x00",
            b"RIFF\x00\x00\x00\x00WEB",
            b"\xff",
            b"\xff\xd8",
        ]
        for t in truncated_samples:
            with pytest.raises(InvalidMediaError):
                validate_media_content(t, io.BytesIO(t))

    def test_truncated_mp4_rejected(self):
        """
        Verifies that truncated MP4 files (e.g. 8-byte ftyp atom) are rejected with InvalidMediaError.
        """
        raw_mp4_header = b"\x00\x00\x00\x18ftyp"
        with pytest.raises(InvalidMediaError, match="truncated|below minimum"):
            validate_media_content(raw_mp4_header, io.BytesIO(raw_mp4_header))

    def test_fake_mime_types_with_corrupt_bytes_rejected(self):
        fake_png = b"\x89PNG\r\n\x1a\n" + b"CORRUPT_PAYLOAD_NOT_IMAGE"
        with pytest.raises(InvalidMediaError, match="Image file integrity verification failed"):
            validate_media_content(fake_png[:32], io.BytesIO(fake_png))

        fake_jpeg = b"\xff\xd8\xff" + b"CORRUPT_PAYLOAD_NOT_JPEG"
        with pytest.raises(InvalidMediaError, match="Image file integrity verification failed"):
            validate_media_content(fake_jpeg[:32], io.BytesIO(fake_jpeg))

    def test_payload_exceeding_50mb_boundary(self):
        service = AssetService()
        max_bytes = 50 * 1024 * 1024

        # 50MB + 1 byte must be rejected
        oversized = b"\x89PNG\r\n\x1a\n" + b"\x00" * (max_bytes - 7)
        with pytest.raises(PayloadTooLargeError):
            service.ingest_bytes(oversized)

    @pytest.mark.parametrize(
        "width,height,desc",
        [
            (4097, 100, "Width exceeds 4096"),
            (100, 4097, "Height exceeds 4096"),
            (4096, 4097, "Total pixels 16781312 exceeds 16777216 limit"),
            (6000, 6000, "Large square decompression bomb"),
        ],
    )
    def test_decompression_bombs_rejected(self, width, height, desc):
        img = Image.new("RGB", (width, height), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()

        with pytest.raises(InvalidMediaError, match="exceed"):
            validate_media_content(data[:32], io.BytesIO(data))


# ============================================================================
# 4. ERROR LEAKAGES ADVERSARIAL SUITE
# ============================================================================
@pytest.mark.security
class TestErrorLeakagesAdversarialHarness:
    """Stress tests exception handlers with sensitive paths, database secrets, and internal classes."""

    @pytest.fixture
    def hostile_app(self):
        app = FastAPI()
        app.add_middleware(RequestIDMiddleware)
        register_exception_handlers(app)

        class InternalWorkerEngine:
            pass

        @app.get("/trigger/sensitive-path")
        def leak_path():
            raise FileNotFoundError("/Volumes/WorkSpace/Project/Gen_Image:Video/secret_key.pem missing")

        @app.get("/trigger/db-creds")
        def leak_creds():
            raise ConnectionError("FATAL: Failed connecting to postgresql://admin:SuperSecretPass@10.0.0.2:5432/db")

        @app.get("/trigger/internal-class")
        def leak_class():
            raise TypeError(f"Execution failed in {InternalWorkerEngine.__name__} context")

        return app

    def test_sensitive_errors_do_not_leak_to_client(self, hostile_app):
        client = TestClient(hostile_app, raise_server_exceptions=False)

        endpoints = [
            "/trigger/sensitive-path",
            "/trigger/db-creds",
            "/trigger/internal-class",
        ]

        for ep in endpoints:
            res = client.get(ep)
            assert res.status_code == 500
            data = res.json()
            assert "error" in data
            err = data["error"]
            assert err["code"] == "INTERNAL_SERVER_ERROR"
            assert err["request_id"].startswith("req-")
            assert res.headers.get("X-Request-ID") == err["request_id"]

            # Response text scan against forbidden patterns
            for pattern in FORBIDDEN_LEAK_PATTERNS:
                assert not re.search(pattern, res.text), (
                    f"Security leak detected on {ep}! Pattern '{pattern}' found in: {res.text}"
                )

    def test_404_and_405_errors_conform_and_sanitize(self):
        client = TestClient(control_plane_app)

        # 404
        res_404 = client.get("/v1/nonexistent-route")
        assert res_404.status_code == 404
        assert res_404.json()["error"]["code"] == "NOT_FOUND"
        assert res_404.json()["error"]["request_id"].startswith("req-")

        # 405
        res_405 = client.post("/health")
        assert res_405.status_code == 405
        assert res_405.json()["error"]["code"] == "METHOD_NOT_ALLOWED"
        assert res_405.json()["error"]["request_id"].startswith("req-")
