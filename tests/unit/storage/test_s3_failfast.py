"""
tests/unit/storage/test_s3_failfast.py
Unit tests verifying production fail-fast behavior of S3StorageProvider and StorageConfig.
"""
import pytest
from factory.apps.control_api.src.storage.config import StorageConfig, load_storage_config
from factory.apps.control_api.src.storage.s3 import S3StorageProvider


@pytest.fixture(autouse=True, scope="module")
def cleanup_boto_modules():
    yield
    import sys
    for mod in list(sys.modules.keys()):
        if (
            mod == "boto3"
            or mod.startswith("boto3.")
            or mod == "botocore"
            or mod.startswith("botocore.")
            or "storage.s3" in mod
        ):
            sys.modules.pop(mod, None)


class TestS3StorageProviderFailFast:
    def test_s3_storage_provider_fails_fast_in_production(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        cfg = StorageConfig(
            storage_backend="s3",
            s3_endpoint_url="http://127.0.0.1:54321",  # Unreachable port
            s3_bucket_name="nonexistent-prod-bucket",
            s3_access_key="fake-key",
            s3_secret_key="fake-secret",
            environment="production",
        )
        with pytest.raises(RuntimeError) as exc_info:
            S3StorageProvider(cfg)
        assert "FATAL: S3 bucket" in str(exc_info.value)
        assert "inaccessible" in str(exc_info.value)

    def test_s3_storage_provider_tolerates_offline_in_development(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "development")
        cfg = StorageConfig(
            storage_backend="s3",
            s3_endpoint_url="http://127.0.0.1:54321",  # Unreachable port
            s3_bucket_name="offline-dev-bucket",
            s3_access_key="fake-key",
            s3_secret_key="fake-secret",
            environment="development",
        )
        # In development, S3StorageProvider does not crash on offline bucket creation
        provider = S3StorageProvider(cfg)
        assert provider.bucket == "offline-dev-bucket"

    def test_load_storage_config_s3_production_validation(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("STORAGE_BACKEND", "s3")

        # Missing credentials
        monkeypatch.delenv("S3_ACCESS_KEY", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        with pytest.raises(RuntimeError, match="FATAL: S3_ACCESS_KEY"):
            load_storage_config()

        # Default minioadmin forbidden in production
        monkeypatch.setenv("S3_ACCESS_KEY", "minioadmin")
        monkeypatch.setenv("S3_SECRET_KEY", "minioadmin")
        with pytest.raises(RuntimeError, match="FATAL: Default MinIO credentials"):
            load_storage_config()

    def test_load_storage_config_unsupported_backend_rejected(self, monkeypatch):
        monkeypatch.setenv("STORAGE_BACKEND", "ftp")
        with pytest.raises(RuntimeError, match="Unsupported STORAGE_BACKEND 'ftp'"):
            load_storage_config()
