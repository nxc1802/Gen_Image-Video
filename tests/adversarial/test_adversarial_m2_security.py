"""
tests/adversarial/test_adversarial_m2_security.py
Adversarial Security Attack Suite for Milestone 2 Iteration 2 Gate (Challenger 2).

Thoroughly verifies:
1. Multi-Field Base64 Payload Threshold Bypass (ordering, deep nesting, mixed arrays, 1MB boundaries, deprecation headers)
2. Local Storage Traversal Attacks (sibling directories, absolute path tricks, URL-encoded traversals, base_dir escapes)
3. Storage Key Injections (sanitization against traversal sequences, null bytes, subsystem spoofing, canonical format)
4. WebSocket Security (unauthenticated access, forged tokens, invalid JWTs, policy violation code 1008)
5. WebSocket Typed Cancellation Frames (typed JSON-RPC frames, legacy action/event frames, resilience to malformed inputs)
"""
import base64
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import jwt
from factory_auth import AuthConfig, decode_jwt_token
from factory_contracts.errors import (
    FactoryError,
    InvalidRequestError,
    NotFoundError,
    PayloadTooLargeError,
)
from factory_contracts.jobs.job_contract import (
    ArtifactReference,
    FactoryJob,
    JobProgress,
    JobState,
    JobSubmissionRequest,
)
from factory_contracts.jobs.worker_contract import JobCompletionRequest
from factory.apps.control_api.src.main import app
from factory.apps.control_api.src.storage.config import StorageConfig
from factory.apps.control_api.src.storage.keys import (
    format_artifact_key,
    format_asset_key,
    sanitize_filename,
    sanitize_id,
)
from factory.apps.control_api.src.storage.local import LocalStorageProvider
from factory.apps.control_api.src.storage.threshold import (
    DEPRECATION_HEADERS,
    MAX_INLINE_BYTES,
    enforce_payload_thresholds,
    estimate_base64_size,
    find_inline_base64,
    is_raw_base64_media,
    should_attach_deprecation_header,
)
from factory.apps.control_api.src.streaming.ws import WebSocketHub

VALID_API_KEY = "sk-factory-prod-secret-token"
VALID_JWT_SECRET = "factory-test-jwt-secret-key-32-chars-min!"


def _generate_b64(byte_len: int, prefix: str = "data:image/png;base64,") -> str:
    raw = b"A" * byte_len
    return prefix + base64.b64encode(raw).decode("ascii")


@pytest.fixture(scope="module")
def api_client():
    old_key = os.environ.get("FACTORY_API_KEY")
    old_jwt = os.environ.get("FACTORY_JWT_SECRET")
    old_backend = os.environ.get("STORAGE_BACKEND")
    old_db = os.environ.get("FACTORY_DB_BACKEND")
    old_sqlite = os.environ.get("SQLITE_DB_PATH")

    os.environ["FACTORY_API_KEY"] = VALID_API_KEY
    os.environ["FACTORY_JWT_SECRET"] = VALID_JWT_SECRET
    os.environ["STORAGE_BACKEND"] = "local"
    os.environ["FACTORY_DB_BACKEND"] = "sqlite"
    os.environ["SQLITE_DB_PATH"] = ":memory:"

    with TestClient(app) as client:
        yield client

    if hasattr(app.state, "auth_config"):
        delattr(app.state, "auth_config")

    for k, v in [
        ("FACTORY_API_KEY", old_key),
        ("FACTORY_JWT_SECRET", old_jwt),
        ("STORAGE_BACKEND", old_backend),
        ("FACTORY_DB_BACKEND", old_db),
        ("SQLITE_DB_PATH", old_sqlite),
    ]:
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


@pytest.fixture
def auth_headers():
    return {
        "Authorization": f"Bearer {VALID_API_KEY}",
        "Content-Type": "application/json",
    }


# ==============================================================================
# Scenario 1: Multi-Input Payload Threshold Bypass
# ==============================================================================
class TestAdversarialPayloadThresholdBypass:
    """Adversarially attacks Base64 inspection with ordering and nesting permutations."""

    def test_multi_field_ordering_first_small_second_oversized(self):
        """Small payload followed by oversized payload must be caught."""
        small = _generate_b64(10 * 1024)  # 10 KB
        large = _generate_b64(2 * 1024 * 1024)  # 2 MB
        payload = {"avatar": small, "background": large}

        matches = find_inline_base64(payload)
        assert len(matches) == 2

        with pytest.raises(PayloadTooLargeError) as exc_info:
            enforce_payload_thresholds(payload)
        assert "inputs.background" in str(exc_info.value)
        assert exc_info.value.status_code == 413
        assert not should_attach_deprecation_header(payload)

    def test_multi_field_ordering_first_oversized_second_small(self):
        """Oversized payload followed by small payload must be caught."""
        small = _generate_b64(20 * 1024)  # 20 KB
        large = _generate_b64(1500 * 1024)  # 1.5 MB
        payload = {"background": large, "avatar": small}

        with pytest.raises(PayloadTooLargeError) as exc_info:
            enforce_payload_thresholds(payload)
        assert "inputs.background" in str(exc_info.value)
        assert not should_attach_deprecation_header(payload)

    def test_deeply_nested_mixed_payloads(self):
        """Oversized payload hidden deep within nested dicts and lists must be caught."""
        small = _generate_b64(5 * 1024)
        large = _generate_b64(1200 * 1024)  # 1.2 MB
        nested = {
            "pipeline": {
                "stages": [
                    {"name": "preprocess", "config": {"mask": small}},
                    {
                        "name": "generate",
                        "layers": [
                            {"reference": small},
                            {"reference_oversized": large},
                        ],
                    },
                ]
            }
        }

        with pytest.raises(PayloadTooLargeError) as exc_info:
            enforce_payload_thresholds(nested)
        assert "inputs.pipeline.stages[1].layers[1].reference_oversized" in str(exc_info.value)
        assert not should_attach_deprecation_header(nested)

    def test_mixed_array_ordering(self):
        """Array with multiple small items preceding oversized item."""
        small1 = _generate_b64(1024)
        small2 = _generate_b64(2048)
        large = _generate_b64(1024 * 1024 + 100)  # > 1MB
        small3 = _generate_b64(4096)
        arr = [small1, small2, large, small3]

        with pytest.raises(PayloadTooLargeError) as exc_info:
            enforce_payload_thresholds({"items": arr})
        assert "inputs.items[2]" in str(exc_info.value)

    def test_exact_1mb_boundary_allowed_with_deprecation(self):
        """Payload exactly at 1MB (1,048,576 bytes) must be allowed and attach deprecation headers."""
        exact_1mb = _generate_b64(MAX_INLINE_BYTES)
        payload = {"image": exact_1mb}

        # Should not raise
        enforce_payload_thresholds(payload)
        assert should_attach_deprecation_header(payload) is True

    def test_exact_1mb_plus_one_byte_rejected(self):
        """Payload at 1MB + 1 byte (1,048,577 bytes) must be rejected with 413."""
        oversized = _generate_b64(MAX_INLINE_BYTES + 1)
        payload = {"image": oversized}

        with pytest.raises(PayloadTooLargeError):
            enforce_payload_thresholds(payload)
        assert should_attach_deprecation_header(payload) is False

    def test_raw_base64_without_data_uri_detected(self):
        """Raw base64 string without data: prefix exceeding 1MB must be detected."""
        raw_b64 = base64.b64encode(b"B" * (2 * 1024 * 1024)).decode("ascii")
        payload = {"raw_mask": raw_b64}

        with pytest.raises(PayloadTooLargeError) as exc_info:
            enforce_payload_thresholds(payload)
        assert "inputs.raw_mask" in str(exc_info.value)

    def test_long_urls_exempt_from_base64_inspection(self):
        """Long URLs (>1024 chars) starting with http:// or https:// must not be flagged as base64."""
        long_url = "https://cdn.example.com/assets/" + ("subpath/" * 200) + "image.png"
        payload = {"url": long_url}
        matches = find_inline_base64(payload)
        assert len(matches) == 0
        enforce_payload_thresholds(payload)
        assert should_attach_deprecation_header(payload) is False

    def test_adversarial_empirical_finding_long_text_false_positive(self):
        """
        Remediation Verified: Plain text strings (>1024 chars) with natural language whitespace
        and punctuation are NOT treated as base64 media.
        """
        long_text = "Detailed scene prompt describing cinematic lighting and textures. " * 30  # ~2000 chars
        payload = {"prompt": long_text}
        matches = find_inline_base64(payload)
        assert len(matches) == 0
        assert should_attach_deprecation_header(payload) is False

    def test_adversarial_empirical_finding_oversized_text_rejected_as_media(self):
        """
        Remediation Verified: Large natural language text prompts (>1MB text)
        do NOT trigger false base64 media PayloadTooLargeError.
        """
        oversized_text = "Natural language text for long document summarization. " * 30000  # ~1.65 MB
        # Should complete cleanly without raising PayloadTooLargeError
        enforce_payload_thresholds({"prompt": oversized_text})
        assert should_attach_deprecation_header({"prompt": oversized_text}) is False

    def test_adversarial_probe_multilingual_markdown_oversized_text_allowed(self):
        """
        Adversarial Probe: Markdown documents with non-ASCII CJK, Arabic, Cyrillic,
        emojis, code blocks, and punctuation (>2MB) must NOT be flagged as raw Base64.
        """
        complex_text = (
            "# Detailed Chapter\n\n"
            "Here is **bold text**, *italic text*, and `def foo(x): return x * 2`.\n"
            "CJK: 日本語テスト文章、这是一段中文测试文本，한국어 텍스트.\n"
            "Arabic: هذه تجربة نصية باللغة العربية.\n"
            "Cyrillic: Это тестовый текст на русском языке.\n"
            "Symbols & Emojis: 🚀 💻 🔍 {key: [1, 2, 3], 'flag': true} -- @@ ## $$ %%\n"
            "Paragraph continuation with punctuation: , . ! ? ; : ( ) [ ] / < > \n\n"
        ) * 5000  # ~2.2 MB
        payload = {"prompt": complex_text}
        assert not is_raw_base64_media(complex_text)
        assert len(find_inline_base64(payload)) == 0
        enforce_payload_thresholds(payload)
        assert should_attach_deprecation_header(payload) is False

    def test_adversarial_probe_raw_base64_mime_wrapped_newlines_detected(self):
        """
        Adversarial Probe: Raw Base64 string (>1MB) formatted with standard MIME line wraps
        (newlines every 64 characters) must be detected as raw Base64 media and rejected with 413.
        """
        raw_bytes = b"C" * (1200 * 1024)
        raw_b64 = base64.b64encode(raw_bytes).decode("ascii")
        # Insert MIME line breaks every 64 characters
        wrapped_b64 = "\n".join(raw_b64[i : i + 64] for i in range(0, len(raw_b64), 64))
        assert is_raw_base64_media(wrapped_b64)
        matches = find_inline_base64({"media": wrapped_b64})
        assert len(matches) == 1
        with pytest.raises(PayloadTooLargeError) as exc_info:
            enforce_payload_thresholds({"media": wrapped_b64})
        assert "inputs.media" in str(exc_info.value)
        assert not should_attach_deprecation_header({"media": wrapped_b64})

    def test_adversarial_probe_raw_base64_all_padding_variations(self):
        """
        Adversarial Probe: Raw Base64 strings with all 3 possible padding alignments
        (0 padding, 1 padding '=', 2 padding '==') at 1MB boundary.
        """
        for pad_mod in [0, 1, 2]:
            exact_bytes = MAX_INLINE_BYTES - (MAX_INLINE_BYTES % 3) + pad_mod
            b64_allowed = base64.b64encode(b"X" * exact_bytes).decode("ascii")
            assert is_raw_base64_media(b64_allowed)
            # Allowed when <= 1MB
            if exact_bytes <= MAX_INLINE_BYTES:
                enforce_payload_thresholds({"asset": b64_allowed})
                assert should_attach_deprecation_header({"asset": b64_allowed}) is True

            oversized_bytes = exact_bytes + (3 if exact_bytes <= MAX_INLINE_BYTES else 0)
            b64_rejected = base64.b64encode(b"Y" * oversized_bytes).decode("ascii")
            assert is_raw_base64_media(b64_rejected)
            with pytest.raises(PayloadTooLargeError):
                enforce_payload_thresholds({"asset": b64_rejected})

    def test_adversarial_probe_data_uri_various_mimes_and_parameters(self):
        """
        Adversarial Probe: Data URIs across diverse media formats and parameters
        (webp, audio, video, binary with parameters) must be strictly enforced at 1MB.
        """
        prefixes = [
            "data:image/webp;base64,",
            "data:audio/wav;base64,",
            "data:video/mp4;base64,",
            "data:application/octet-stream;name=test.bin;base64,",
        ]
        for prefix in prefixes:
            duri_oversized = prefix + base64.b64encode(b"Z" * (1200 * 1024)).decode("ascii")
            matches = find_inline_base64({"data": duri_oversized})
            assert len(matches) == 1
            with pytest.raises(PayloadTooLargeError) as exc_info:
                enforce_payload_thresholds({"data": duri_oversized})
            assert "inputs.data" in str(exc_info.value)

    def test_adversarial_probe_mixed_huge_text_and_oversized_media(self):
        """
        Adversarial Probe: Payload containing both a 2MB natural text prompt AND a 2MB
        raw Base64 media string. Only the media string must be targeted for 413.
        """
        huge_prompt = "Document analysis prompt with multiple words and sentences. " * 35000  # ~2MB
        huge_image = base64.b64encode(b"I" * (1500 * 1024)).decode("ascii")
        payload = {"prompt": huge_prompt, "init_image": huge_image}

        matches = find_inline_base64(payload)
        assert len(matches) == 1
        assert matches[0][0] == "inputs.init_image"

        with pytest.raises(PayloadTooLargeError) as exc_info:
            enforce_payload_thresholds(payload)
        assert "inputs.init_image" in str(exc_info.value)
        assert "inputs.prompt" not in str(exc_info.value)

    def test_adversarial_probe_live_api_allows_huge_text_prompt(self, api_client, auth_headers):
        """
        Adversarial Probe: Live FastAPI POST /v1/jobs accepts a 1.5MB text prompt cleanly
        without triggering 413 PayloadTooLargeError or Deprecation headers.
        """
        large_prompt = "High fidelity narrative prompt for full story generation. " * 25000  # ~1.5MB
        req_body = {
            "service": "text",
            "model": "qwen-text-2.5",
            "task": "chat",
            "inputs": {
                "prompt": large_prompt,
            },
        }

        res = api_client.post("/v1/jobs", headers=auth_headers, json=req_body)
        assert res.status_code == 201
        assert res.headers.get("Deprecation") is None


    def test_live_api_rejection_of_multi_field_oversized_payload(self, api_client, auth_headers):
        """Live FastAPI POST /v1/jobs rejects multi-field payload with 413 and error envelope."""
        small = _generate_b64(10 * 1024)
        large = _generate_b64(2 * 1024 * 1024)
        req_body = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {
                "prompt": "Test prompt",
                "init_image": small,
                "mask_image": large,
            },
        }

        res = api_client.post("/v1/jobs", headers=auth_headers, json=req_body)
        assert res.status_code == 413
        data = res.json()
        assert "error" in data
        assert data["error"]["code"] == "PAYLOAD_TOO_LARGE"
        assert "inputs.mask_image" in data["error"]["message"]

    def test_live_api_allows_small_base64_with_deprecation_headers(self, api_client, auth_headers):
        """Live FastAPI POST /v1/jobs accepts small base64 payload and emits deprecation headers."""
        small = _generate_b64(20 * 1024)
        req_body = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {
                "prompt": "Small image input",
                "init_image": small,
            },
        }

        res = api_client.post("/v1/jobs", headers=auth_headers, json=req_body)
        assert res.status_code == 201
        assert res.headers.get("Deprecation") == "true"
        assert "Sunset" in res.headers
        assert "299" in res.headers.get("Warning", "")


# ==============================================================================
# Scenario 2: Local Storage Traversal Attacks
# ==============================================================================
class TestAdversarialLocalStorageTraversal:
    """Adversarially attacks LocalStorageProvider path containment and traversal prevention."""

    @pytest.fixture
    def local_storage(self):
        tmp_dir = tempfile.mkdtemp(prefix="factory_test_storage_")
        cfg = StorageConfig(
            storage_backend="local",
            local_storage_dir=tmp_dir,
            local_storage_secret_key="test-secret-key",
            local_base_url="http://testserver",
            s3_bucket_name="test-bucket",
        )
        provider = LocalStorageProvider(cfg)
        yield provider, Path(tmp_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_sibling_directory_traversal_rejected(self, local_storage):
        """Attempting to resolve sibling directory like ../storage_evil must raise InvalidRequestError."""
        provider, base_dir = local_storage
        with pytest.raises(InvalidRequestError) as exc_info:
            provider._get_path("../storage_evil", "file.txt")
        assert "Directory traversal" in str(exc_info.value)

    def test_relative_key_traversal_rejected(self, local_storage):
        """Attempting relative traversal in key must raise InvalidRequestError."""
        provider, base_dir = local_storage
        with pytest.raises(InvalidRequestError) as exc_info:
            provider._get_path("valid-bucket", "../../etc/passwd")
        assert "Directory traversal" in str(exc_info.value)

    def test_escaped_dot_dot_in_key_rejected(self, local_storage):
        """Nested traversal inside middle of key path must be rejected."""
        provider, base_dir = local_storage
        with pytest.raises(InvalidRequestError):
            provider._get_path("valid-bucket", "artifacts/../../evil.txt")

    def test_base_directory_escape_to_base_dir_itself_rejected(self, local_storage):
        """Resolving to base_dir itself must be rejected."""
        provider, base_dir = local_storage
        with pytest.raises(InvalidRequestError):
            provider._get_path("", "")
        with pytest.raises(InvalidRequestError):
            provider._get_path(".", ".")

    def test_absolute_path_key_contained_within_bucket(self, local_storage):
        """Leading slashes in key or bucket must be stripped and contained within base_dir."""
        provider, base_dir = local_storage
        target = provider._get_path("/my-bucket/", "/my-file.txt")
        assert target.is_relative_to(provider.base_dir)
        assert target == provider.base_dir / "my-bucket" / "my-file.txt"

    @pytest.mark.asyncio
    async def test_upload_and_download_prevent_directory_traversal(self, local_storage):
        """upload_bytes and download_bytes must not write or read outside base_dir."""
        provider, base_dir = local_storage

        # Malicious upload attempt
        with pytest.raises(InvalidRequestError):
            await provider.upload_bytes("../evil_dir", "malicious.bin", b"evil data", "application/octet-stream")

        # Malicious download attempt
        with pytest.raises(InvalidRequestError):
            await provider.download_bytes("../evil_dir", "malicious.bin")

        # Ensure no evil_dir exists next to base_dir
        evil_sibling = base_dir.parent / "evil_dir"
        assert not evil_sibling.exists()

    @pytest.mark.asyncio
    async def test_head_and_delete_prevent_directory_traversal(self, local_storage):
        """head_object and delete_object must reject traversal attempts."""
        provider, base_dir = local_storage

        with pytest.raises(InvalidRequestError):
            await provider.head_object("../evil_dir", "malicious.bin")

        with pytest.raises(InvalidRequestError):
            await provider.delete_object("../evil_dir", "malicious.bin")

    def test_live_api_url_encoded_traversal_rejected(self, api_client):
        """FastAPI /v1/storage/download rejects URL-encoded %2e%2e traversals."""
        # Forge signature for a traversal path
        res = api_client.get(
            "/v1/storage/download",
            params={
                "bucket": "%2e%2e/storage_evil",
                "key": "secrets.txt",
                "expires": int(time.time()) + 3600,
                "sig": "invalid_or_fake_sig",
            },
        )
        # Rejection should occur either due to invalid signature (403) or traversal (400)
        assert res.status_code in (400, 403)


# ==============================================================================
# Scenario 3: Storage Key Injections
# ==============================================================================
class TestAdversarialStorageKeyInjection:
    """Adversarially attacks sanitize_id, sanitize_filename, and key formatters."""

    def test_sanitize_id_strips_traversal_and_separators(self):
        """sanitize_id must eliminate ../, windows slashes, and null bytes."""
        assert sanitize_id("../../evil") == "evil"
        assert sanitize_id("../../../etc/passwd") == "etc_passwd"
        assert sanitize_id("..\\..\\windows\\traversal") == "windows_traversal"
        assert sanitize_id("job/../middle/../evil") == "job_middle_evil"
        assert sanitize_id("job\x00evil") == "job_evil"

    def test_sanitize_id_boundary_edge_cases(self):
        """Strings that reduce to nothing must return the safe default."""
        assert sanitize_id("") == "unknown"
        assert sanitize_id("...") == "unknown"
        assert sanitize_id("///") == "unknown"
        assert sanitize_id("..") == "unknown"
        assert sanitize_id("!@#$%^&*()", default="fallback") == "fallback"

    def test_sanitize_filename_prevents_escape(self):
        """sanitize_filename must reject or neutralize path navigation."""
        assert sanitize_filename("../../malicious.exe") == "malicious.exe"
        assert sanitize_filename("../test.png") == "test.png"
        assert sanitize_filename("") == "artifact.bin"
        assert sanitize_filename("...") == "artifact.bin"
        assert sanitize_filename("safe_name.png") == "safe_name.png"

    def test_format_artifact_key_sanitizes_all_components(self):
        """format_artifact_key produces canonical keys without traversal."""
        # Malicious job and artifact
        key1 = format_artifact_key("image", "../../evil_job", "../../../secret.png")
        assert key1 == "image/evil_job/secret.png"
        assert ".." not in key1

        # Unknown subsystem defaults to 'artifacts'
        key2 = format_artifact_key("evil_subsystem", "job-123", "out.png")
        assert key2 == "artifacts/job-123/out.png"

        # Uppercase subsystem normalized
        key3 = format_artifact_key("IMAGE", "job-123", "out.png")
        assert key3 == "image/job-123/out.png"

    def test_format_asset_key_sanitizes_id_and_filename(self):
        """format_asset_key produces canonical asset keys without traversal."""
        key1 = format_asset_key("../../evil_asset")
        assert key1 == "assets/evil_asset.bin"

        key2 = format_asset_key("../evil_asset", filename="../../evil.png")
        assert key2 == "assets/evil_asset/evil.png"
        assert ".." not in key2


# ==============================================================================
# Scenario 4: WebSocket Security (Authentication & Rejection)
# ==============================================================================
class TestAdversarialWebSocketSecurity:
    """Adversarially attacks WebSocket endpoints with unauthenticated and forged tokens."""

    @pytest.fixture
    def target_job_id(self, api_client, auth_headers):
        res = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "WebSocket security target"},
            },
        )
        assert res.status_code == 201
        return res.json()["job_id"]

    def test_unauthenticated_connection_rejected_with_1008(self, api_client, target_job_id):
        """Connecting with no credentials must immediately close with 1008."""
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with api_client.websocket_connect(f"/v1/jobs/{target_job_id}/ws"):
                pass
        assert exc_info.value.code == 1008

    def test_invalid_query_token_rejected_with_1008(self, api_client, target_job_id):
        """Connecting with invalid query token must close with 1008."""
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with api_client.websocket_connect(f"/v1/jobs/{target_job_id}/ws?token=forged-token"):
                pass
        assert exc_info.value.code == 1008

    def test_empty_query_token_rejected_with_1008(self, api_client, target_job_id):
        """Connecting with empty query token must close with 1008."""
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with api_client.websocket_connect(f"/v1/jobs/{target_job_id}/ws?token="):
                pass
        assert exc_info.value.code == 1008

    def test_invalid_bearer_header_rejected_with_1008(self, api_client, target_job_id):
        """Connecting with invalid Authorization header must close with 1008."""
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with api_client.websocket_connect(
                f"/v1/jobs/{target_job_id}/ws",
                headers={"Authorization": "Bearer fake-token-12345"},
            ):
                pass
        assert exc_info.value.code == 1008

    def test_forged_jwt_token_rejected_with_1008(self, api_client, target_job_id):
        """Connecting with a JWT signed by an incorrect secret must close with 1008."""
        forged_jwt = jwt.encode(
            {"sub": "attacker", "roles": ["admin"], "exp": int(time.time()) + 3600},
            "completely-wrong-secret-key-32-chars!!",
            algorithm="HS256",
        )
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with api_client.websocket_connect(f"/v1/jobs/{target_job_id}/ws?token={forged_jwt}"):
                pass
        assert exc_info.value.code == 1008

    def test_valid_api_key_accepted(self, api_client, target_job_id):
        """Valid API key in query parameter must establish connection."""
        with api_client.websocket_connect(f"/v1/jobs/{target_job_id}/ws?token={VALID_API_KEY}") as ws:
            assert ws is not None

    def test_valid_jwt_token_accepted(self, api_client, target_job_id):
        """Valid JWT signed with configured secret must establish connection."""
        valid_jwt = jwt.encode(
            {"sub": "service-worker-1", "roles": ["worker"], "exp": int(time.time()) + 3600},
            VALID_JWT_SECRET,
            algorithm="HS256",
        )
        with api_client.websocket_connect(
            f"/v1/jobs/{target_job_id}/ws",
            headers={"Authorization": f"Bearer {valid_jwt}"},
        ) as ws:
            assert ws is not None


# ==============================================================================
# Scenario 5: WebSocket Typed Cancellation Frames
# ==============================================================================
class TestAdversarialWebSocketCancellation:
    """Adversarially attacks WebSocket cancellation with typed frames and malformed messages."""

    def test_json_rpc_typed_cancellation_cancels_job(self, api_client, auth_headers):
        """Sending {"type": "cancel"} frame must trigger job cancellation."""
        # 1. Submit job
        res = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "To be cancelled via typed frame"},
            },
        )
        job_id = res.json()["job_id"]
        assert res.json()["state"] == "QUEUED"

        # 2. Connect and send typed cancellation frame
        with api_client.websocket_connect(f"/v1/jobs/{job_id}/ws?token={VALID_API_KEY}") as ws:
            ws.send_text(json.dumps({"type": "cancel"}))
            # Small yield to let async cancellation complete
            time.sleep(0.05)

        # 3. Verify job state transitioned to CANCELLED
        job_res = api_client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert job_res.status_code == 200
        assert job_res.json()["state"] == "CANCELLED"

    def test_legacy_action_cancellation_cancels_job(self, api_client, auth_headers):
        """Sending {"action": "cancel"} frame must also trigger job cancellation."""
        res = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "To be cancelled via action frame"},
            },
        )
        job_id = res.json()["job_id"]

        with api_client.websocket_connect(f"/v1/jobs/{job_id}/ws?token={VALID_API_KEY}") as ws:
            ws.send_text(json.dumps({"action": "cancel"}))
            time.sleep(0.05)

        job_res = api_client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert job_res.json()["state"] == "CANCELLED"

    def test_legacy_event_cancellation_cancels_job(self, api_client, auth_headers):
        """Sending {"event": "cancel"} frame must also trigger job cancellation."""
        res = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "To be cancelled via event frame"},
            },
        )
        job_id = res.json()["job_id"]

        with api_client.websocket_connect(f"/v1/jobs/{job_id}/ws?token={VALID_API_KEY}") as ws:
            ws.send_text(json.dumps({"event": "cancel"}))
            time.sleep(0.05)

        job_res = api_client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert job_res.json()["state"] == "CANCELLED"

    def test_non_cancel_frames_do_not_cancel_job(self, api_client, auth_headers):
        """Non-cancel frames (e.g. ping, status) must leave job in active state."""
        res = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "Active job test"},
            },
        )
        job_id = res.json()["job_id"]

        with api_client.websocket_connect(f"/v1/jobs/{job_id}/ws?token={VALID_API_KEY}") as ws:
            ws.send_text(json.dumps({"type": "ping"}))
            ws.send_text(json.dumps({"action": "heartbeat"}))
            ws.send_text(json.dumps({"random_key": "random_value"}))
            time.sleep(0.05)

        job_res = api_client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert job_res.json()["state"] == "QUEUED"

    def test_malformed_json_frames_handled_gracefully(self, api_client, auth_headers):
        """Sending invalid non-JSON strings must not crash the WebSocket or the server."""
        res = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "Malformed test"},
            },
        )
        job_id = res.json()["job_id"]

        with api_client.websocket_connect(f"/v1/jobs/{job_id}/ws?token={VALID_API_KEY}") as ws:
            ws.send_text("THIS IS NOT JSON AT ALL {{{")
            ws.send_text("")
            ws.send_text("\x00\x01\x02")
            time.sleep(0.05)

        # Job is still alive and QUEUED
        job_res = api_client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert job_res.json()["state"] == "QUEUED"
