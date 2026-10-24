"""
tests/security/test_ssrf_asset_validator.py
Security tests verifying SSRF protection, media validation, and 50MB payload limits.
"""
import io
import pytest
from PIL import Image

from factory_contracts.errors import (
    SSRFBlockedError,
    InvalidMediaError,
    PayloadTooLargeError,
)
from factory.apps.control_api.src.assets.ssrf import validate_target_url
from factory.apps.control_api.src.assets.media_validator import validate_media_content
from factory.apps.control_api.src.assets.service import AssetService


@pytest.fixture
def sample_png_bytes() -> bytes:
    img = Image.new("RGB", (10, 10), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (10, 10), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_webp_bytes() -> bytes:
    img = Image.new("RGB", (10, 10), color="red")
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


@pytest.mark.security
@pytest.mark.ssrf
class TestSSRFProtection:
    @pytest.mark.parametrize(
        "target_url",
        [
            "http://127.0.0.1:8000/v1/jobs",
            "http://127.0.0.2:8080/",
            "http://localhost:8000/health",
            "http://169.254.169.254/latest/meta-data/",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://instance-data/latest/meta-data/",
            "http://10.0.0.1/admin",
            "http://10.255.255.255/secrets",
            "http://192.168.1.1/router",
            "http://172.16.0.1/internal",
            "http://172.31.255.255/private",
            "http://0.0.0.0:8000/",
            "http://[::1]:8000/",
            "http://[::]:80/",
            "http://[::0]:80/",
            "http://[::]:8080/",
            "http://8.8.8.8:22/ssh",
            "http://8.8.8.8:3306/mysql",
            "ftp://files.example.com/data.png",
            "file:///etc/passwd",
            "gopher://example.com/",
            "http://example.com:22/ssh",
            "http://example.com:3306/mysql",
        ],
    )
    def test_ssrf_blocks_malicious_urls(self, target_url):
        with pytest.raises(SSRFBlockedError):
            validate_target_url(target_url)


@pytest.mark.security
class TestMediaValidation:
    def test_valid_png(self, sample_png_bytes):
        stream = io.BytesIO(sample_png_bytes)
        mime = validate_media_content(sample_png_bytes[:16], stream)
        assert mime == "image/png"

    def test_valid_jpeg(self, sample_jpeg_bytes):
        stream = io.BytesIO(sample_jpeg_bytes)
        mime = validate_media_content(sample_jpeg_bytes[:16], stream)
        assert mime == "image/jpeg"

    def test_valid_webp(self, sample_webp_bytes):
        stream = io.BytesIO(sample_webp_bytes)
        mime = validate_media_content(sample_webp_bytes[:16], stream)
        assert mime == "image/webp"

    def test_valid_mp4_header(self):
        mp4_bytes = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00" + b"\x00" * 32
        stream = io.BytesIO(mp4_bytes)
        mime = validate_media_content(mp4_bytes[:16], stream)
        assert mime == "video/mp4"

    def test_valid_wav_header(self):
        wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00" + b"\x00" * 32
        stream = io.BytesIO(wav_bytes)
        mime = validate_media_content(wav_bytes[:16], stream)
        assert mime == "audio/wav"

    def test_empty_and_truncated_riff_rejected(self):
        empty_riff = b"RIFF\x00\x00\x00\x00"
        with pytest.raises(InvalidMediaError):
            validate_media_content(empty_riff, io.BytesIO(empty_riff))

        empty_wave = b"RIFF\x04\x00\x00\x00WAVE"
        with pytest.raises(InvalidMediaError, match="truncated|below minimum"):
            validate_media_content(empty_wave, io.BytesIO(empty_wave))

    def test_corrupt_signature_rejected(self):
        bad_bytes = b"CORRUPT_NOT_MEDIA_HEADER" + b"\x00" * 32
        stream = io.BytesIO(bad_bytes)
        with pytest.raises(InvalidMediaError, match="Unsupported or unrecognized"):
            validate_media_content(bad_bytes[:16], stream)

    def test_decompression_bomb_rejected(self):
        # Image claiming 5000x5000 dimensions
        oversized_img = Image.new("RGB", (5000, 5000), color="white")
        buf = io.BytesIO()
        oversized_img.save(buf, format="PNG")
        oversized_bytes = buf.getvalue()

        stream = io.BytesIO(oversized_bytes)
        with pytest.raises(InvalidMediaError, match="exceed maximum permitted"):
            validate_media_content(oversized_bytes[:16], stream)


@pytest.mark.security
class TestAssetServiceLimits:
    def test_payload_exceeding_50mb_rejected(self):
        service = AssetService(max_size_bytes=50 * 1024 * 1024)
        oversized = b"\x89PNG\r\n\x1a\n" + (b"\x00" * (51 * 1024 * 1024))
        with pytest.raises(PayloadTooLargeError, match="exceeds maximum permitted limit"):
            service.ingest_bytes(oversized)

    def test_valid_payload_ingested_correctly(self, sample_png_bytes):
        service = AssetService()
        record = service.ingest_bytes(sample_png_bytes)
        assert record["artifact_id"].startswith("ast-")
        assert record["mime_type"] == "image/png"
        assert record["size_bytes"] == len(sample_png_bytes)
        assert len(record["sha256"]) == 64

    def test_ingest_url_malformed_content_length_rejected(self, monkeypatch):
        import asyncio
        import httpx
        from contextlib import asynccontextmanager

        service = AssetService()
        for bad_len in ["invalid", "1024, 1024", "NaN", "   "]:
            @asynccontextmanager
            async def mock_stream(self, method, url, **kwargs):
                yield httpx.Response(200, headers={"Content-Length": bad_len}, content=b"dummy")

            monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)
            with pytest.raises(InvalidMediaError, match="malformed Content-Length"):
                asyncio.run(service.ingest_url("http://93.184.216.34/asset.png"))

    def test_ingest_url_negative_content_length_rejected(self, monkeypatch):
        import asyncio
        import httpx
        from contextlib import asynccontextmanager

        service = AssetService()
        @asynccontextmanager
        async def mock_stream(self, method, url, **kwargs):
            yield httpx.Response(200, headers={"Content-Length": "-100"}, content=b"dummy")

        monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)
        with pytest.raises(InvalidMediaError, match="negative Content-Length"):
            asyncio.run(service.ingest_url("http://93.184.216.34/asset.png"))

    def test_ingest_url_redirect_rejected(self, monkeypatch):
        import asyncio
        import httpx
        from contextlib import asynccontextmanager

        service = AssetService()
        for redirect_code in (301, 302, 303, 307, 308):
            @asynccontextmanager
            async def mock_stream(self, method, url, **kwargs):
                yield httpx.Response(redirect_code, headers={"Location": "http://127.0.0.1"}, content=b"")

            monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)
            with pytest.raises(InvalidMediaError, match="Remote redirects are prohibited"):
                asyncio.run(service.ingest_url("http://93.184.216.34/asset.png"))

    def test_ingest_url_payload_too_large(self, monkeypatch):
        import asyncio
        import httpx
        from contextlib import asynccontextmanager

        service = AssetService()
        @asynccontextmanager
        async def mock_stream(self, method, url, **kwargs):
            yield httpx.Response(200, headers={"Content-Length": "60000000"}, content=b"dummy")

        monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)
        with pytest.raises(PayloadTooLargeError, match="exceeding"):
            asyncio.run(service.ingest_url("http://93.184.216.34/asset.png"))
