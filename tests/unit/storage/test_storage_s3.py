"""
tests/unit/storage/test_storage_s3.py
Comprehensive unit tests for S3 & Local Signed Storage, HMAC signature validation,
strict 1MB threshold ceiling enforcement, SHA-256 integrity, and canonical key formatting.
"""
import base64
import os
import time
import pytest
from pathlib import Path

from factory_contracts.errors import InvalidRequestError, PayloadTooLargeError
from factory.apps.control_api.src.storage.config import StorageConfig, load_storage_config
from factory.apps.control_api.src.storage.integrity import (
    compute_bytes_sha256,
    compute_stream_sha256,
    verify_sha256,
)
from factory.apps.control_api.src.storage.keys import (
    format_artifact_key,
    format_asset_key,
    sanitize_filename,
    sanitize_id,
)
from factory.apps.control_api.src.storage.local import LocalStorageProvider
from factory.apps.control_api.src.storage.threshold import (
    MAX_INLINE_BYTES,
    enforce_payload_thresholds,
    estimate_base64_size,
    should_attach_deprecation_header,
)


class TestStorageConfig:
    def test_default_config(self):
        cfg = StorageConfig()
        assert cfg.storage_backend == "local"
        assert cfg.s3_bucket_name == "factory-artifacts"
        assert cfg.s3_download_url_expiration_seconds == 86400
        assert cfg.s3_upload_url_expiration_seconds == 3600

    def test_load_storage_config_env(self, monkeypatch):
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.setenv("S3_ENDPOINT_URL", "http://minio:9000")
        monkeypatch.setenv("S3_BUCKET_NAME", "custom-bucket")
        monkeypatch.setenv("S3_PUBLIC_ENDPOINT_URL", "https://s3.ai.example.com")
        cfg = load_storage_config()
        assert cfg.storage_backend == "s3"
        assert cfg.s3_endpoint_url == "http://minio:9000"
        assert cfg.s3_bucket_name == "custom-bucket"
        assert cfg.s3_public_endpoint_url == "https://s3.ai.example.com"


class TestCanonicalKeys:
    def test_format_artifact_key(self):
        key = format_artifact_key("image", "job-12345", "output_0.png")
        assert key == "image/job-12345/output_0.png"

        key2 = format_artifact_key("video", "job-999", "../sneaky/path/video.mp4")
        assert ".." not in key2
        assert key2.startswith("video/job-999/")

    def test_format_asset_key(self):
        key = format_asset_key("ast-abc-123")
        assert key == "assets/ast-abc-123.bin"

        key_custom = format_asset_key("ast-xyz", "sample.png")
        assert key_custom == "assets/ast-xyz/sample.png"

    def test_sanitize_filename(self):
        clean = sanitize_filename("test!@#$_image-1.png")
        assert clean == "test_____image-1.png"

    def test_format_artifact_key_sanitizes_job_id_traversal(self):
        key = format_artifact_key("image", "../../malicious", "output.png")
        assert key == "image/malicious/output.png"

    def test_format_artifact_key_sanitizes_job_id_backslashes(self):
        key = format_artifact_key("image", r"..\..\evil", "output.png")
        assert key == "image/evil/output.png"

    def test_format_asset_key_sanitizes_asset_id_traversal(self):
        key = format_asset_key("../../../asset-001", "file.png")
        assert key == "assets/asset-001/file.png"

    def test_format_artifact_key_all_dots_falls_back(self):
        key = format_artifact_key("image", "..", "out.png")
        assert key == "image/job-unknown/out.png"

    def test_sanitize_id_direct(self):
        assert sanitize_id("..") == "unknown"
        assert sanitize_id("../foo/../bar") == "foo_bar"
        assert sanitize_id(r"..\evil\path") == "evil_path"



class TestIntegritySHA256:
    def test_compute_bytes_and_verify(self):
        data = b"Factory v4 generative workflow data integrity"
        expected_hash = "baef35032049e2954a7374b341fbe148ec702e5b7c7be0255a73e617db8317a6"
        computed = compute_bytes_sha256(data)
        assert len(computed) == 64
        assert verify_sha256(data, computed)
        assert not verify_sha256(data, "0" * 64)

    def test_compute_stream_sha256(self):
        import io
        data = b"chunk1" + b"chunk2" + b"chunk3"
        stream = io.BytesIO(data)
        hash_stream = compute_stream_sha256(stream)
        assert hash_stream == compute_bytes_sha256(data)


class TestThresholdEnforcement:
    def test_small_base64_allowed_with_deprecation(self):
        # 100KB payload
        raw_bytes = b"x" * (100 * 1024)
        b64_str = "data:image/png;base64," + base64.b64encode(raw_bytes).decode("utf-8")
        inputs = {"prompt": "test", "image": b64_str}

        # Should NOT raise
        enforce_payload_thresholds(inputs)
        assert should_attach_deprecation_header(inputs) is True

    def test_large_base64_rejected_with_413(self):
        # 1.5MB payload (> 1MB threshold)
        raw_bytes = b"x" * int(1.5 * 1024 * 1024)
        b64_str = "data:image/png;base64," + base64.b64encode(raw_bytes).decode("utf-8")
        inputs = {"prompt": "test", "image": b64_str}

        with pytest.raises(PayloadTooLargeError) as exc_info:
            enforce_payload_thresholds(inputs)

        err = exc_info.value
        assert err.status_code == 413
        assert err.code == "PAYLOAD_TOO_LARGE"
        assert "exceeds the 1MB threshold" in err.message
        assert err.details["threshold_bytes"] == MAX_INLINE_BYTES

    def test_regular_text_no_deprecation(self):
        inputs = {"prompt": "A regular text prompt without media", "steps": 20}
        enforce_payload_thresholds(inputs)
        assert should_attach_deprecation_header(inputs) is False

    def test_multi_field_base64_small_first_large_second_rejected(self):
        small_bytes = b"x" * 10240
        large_bytes = b"x" * (2 * 1024 * 1024)
        b64_small = "data:image/png;base64," + base64.b64encode(small_bytes).decode("utf-8")
        b64_large = "data:image/png;base64," + base64.b64encode(large_bytes).decode("utf-8")
        inputs = {"small": b64_small, "large": b64_large}

        with pytest.raises(PayloadTooLargeError) as exc_info:
            enforce_payload_thresholds(inputs)
        assert exc_info.value.details["field"] == "inputs.large"

    def test_multi_field_base64_large_first_small_second_rejected(self):
        small_bytes = b"x" * 10240
        large_bytes = b"x" * (2 * 1024 * 1024)
        b64_small = "data:image/png;base64," + base64.b64encode(small_bytes).decode("utf-8")
        b64_large = "data:image/png;base64," + base64.b64encode(large_bytes).decode("utf-8")
        inputs = {"large": b64_large, "small": b64_small}

        with pytest.raises(PayloadTooLargeError) as exc_info:
            enforce_payload_thresholds(inputs)
        assert exc_info.value.details["field"] == "inputs.large"

    def test_multi_field_base64_all_small_deprecation_header(self):
        b64_1 = "data:image/png;base64," + base64.b64encode(b"x" * 10240).decode("utf-8")
        b64_2 = "data:image/png;base64," + base64.b64encode(b"x" * 20480).decode("utf-8")
        inputs = {"img1": b64_1, "img2": b64_2}
        enforce_payload_thresholds(inputs)
        assert should_attach_deprecation_header(inputs) is True

    def test_nested_list_base64_oversized_rejected(self):
        small_bytes = b"x" * 10240
        large_bytes = b"x" * (2 * 1024 * 1024)
        b64_small = "data:image/png;base64," + base64.b64encode(small_bytes).decode("utf-8")
        b64_large = "data:image/png;base64," + base64.b64encode(large_bytes).decode("utf-8")
        inputs = {"frames": [b64_small, b64_large]}

        with pytest.raises(PayloadTooLargeError) as exc_info:
            enforce_payload_thresholds(inputs)
        assert exc_info.value.details["field"] == "inputs.frames[1]"



@pytest.mark.asyncio
class TestLocalStorageProvider:
    async def test_hmac_signing_and_verification(self, tmp_path):
        cfg = StorageConfig(
            local_storage_dir=str(tmp_path),
            local_storage_secret_key="test-secret-key-for-hmac-test",
            local_base_url="http://testserver:8000",
        )
        provider = LocalStorageProvider(cfg)

        download_url = provider.generate_presigned_download_url("factory-artifacts", "image/test.png", expires_in_seconds=60)
        assert download_url.startswith("http://testserver:8000/v1/storage/download?")
        assert "sig=" in download_url
        assert "expires=" in download_url

        upload_presigned = provider.generate_presigned_upload_url("factory-artifacts", "image/upload.png", "image/png", 1024)
        assert upload_presigned.method == "PUT"
        assert upload_presigned.upload_url.startswith("http://testserver:8000/v1/storage/upload?")

    async def test_signature_expiry_and_tampering(self, tmp_path):
        cfg = StorageConfig(
            local_storage_dir=str(tmp_path),
            local_storage_secret_key="test-secret",
        )
        provider = LocalStorageProvider(cfg)

        # Valid signature
        expires_valid = int(time.time()) + 100
        sig = provider._sign("bucket1", "key1", expires_valid)
        assert provider.verify_signature("bucket1", "key1", expires_valid, sig) is True

        # Tampered key
        assert provider.verify_signature("bucket1", "tampered_key", expires_valid, sig) is False

        # Expired
        expires_past = int(time.time()) - 10
        sig_past = provider._sign("bucket1", "key1", expires_past)
        assert provider.verify_signature("bucket1", "key1", expires_past, sig_past) is False

    async def test_crud_and_traversal_protection(self, tmp_path):
        cfg = StorageConfig(
            local_storage_dir=str(tmp_path),
            local_storage_secret_key="test-secret",
        )
        provider = LocalStorageProvider(cfg)

        # Upload
        data = b"Binary artifact payload"
        meta = await provider.upload_bytes("test-bucket", "folder/file.bin", data, "application/octet-stream")
        assert meta.size_bytes == len(data)
        assert len(meta.sha256) == 64

        # Head
        head = await provider.head_object("test-bucket", "folder/file.bin")
        assert head is not None
        assert head.size_bytes == len(data)

        # Download
        downloaded = await provider.download_bytes("test-bucket", "folder/file.bin")
        assert downloaded == data

        # Delete
        deleted = await provider.delete_object("test-bucket", "folder/file.bin")
        assert deleted is True

        head_after = await provider.head_object("test-bucket", "folder/file.bin")
        assert head_after is None

        # Directory traversal protection
        with pytest.raises(InvalidRequestError, match="Directory traversal"):
            provider._get_path("test-bucket", "../../../etc/passwd")

    async def test_local_storage_sibling_directory_traversal_blocked(self, tmp_path):
        cfg = StorageConfig(
            local_storage_dir=str(tmp_path),
            local_storage_secret_key="test-secret",
        )
        provider = LocalStorageProvider(cfg)
        with pytest.raises(InvalidRequestError, match="Directory traversal"):
            provider._get_path("../storage_evil", "stolen.txt")

    async def test_local_storage_explicit_dot_dot_in_key_blocked(self, tmp_path):
        cfg = StorageConfig(
            local_storage_dir=str(tmp_path),
            local_storage_secret_key="test-secret",
        )
        provider = LocalStorageProvider(cfg)
        with pytest.raises(InvalidRequestError, match="Directory traversal"):
            provider._get_path("valid-bucket", "../evil.txt")

    async def test_local_storage_base_dir_root_access_blocked(self, tmp_path):
        cfg = StorageConfig(
            local_storage_dir=str(tmp_path),
            local_storage_secret_key="test-secret",
        )
        provider = LocalStorageProvider(cfg)
        with pytest.raises(InvalidRequestError, match="Directory traversal"):
            provider._get_path("", "")

    async def test_local_storage_valid_nested_key_allowed(self, tmp_path):
        cfg = StorageConfig(
            local_storage_dir=str(tmp_path),
            local_storage_secret_key="test-secret",
        )
        provider = LocalStorageProvider(cfg)
        target = provider._get_path("artifacts", "image/job-123/output.png")
        assert target.is_relative_to(provider.base_dir)
        assert target == provider.base_dir / "artifacts" / "image" / "job-123" / "output.png"

