"""
Tier 1 Feature Coverage: Asset Service (Feature 9)
Validates happy path and contract adherence for:
- Feature 9: Asset Service Upload / SSRF Validation (POST /v1/assets/upload, POST /v1/assets/validate)
"""

import pytest
import httpx


VALID_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\n\x00\x00\x00\n\x08\x02\x00\x00\x00\x02PX\xea"
    b"\x00\x00\x00\x13IDATx\x9cc\xfc\xcf\x80\x0f0\xe1\x95e\x18\xa9\xd2\x00A,\x01\x13y\xed\xba&\x00\x00\x00\x00IEND\xaeB`\x82"
)
VALID_WAV_BYTES = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00" + b"\x00" * 32
VALID_MP4_BYTES = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42" + b"\x00" * 32
VALID_JPEG_BYTES = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07"
    b"\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c"
    b"(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\n\x00\n\x01\x01\x11\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x80\xff\xd9"
)


@pytest.mark.tier1
class TestAssetService:
    def test_f09_asset_upload_image(self, client: httpx.Client, auth_headers):
        files = {"file": ("sample.png", VALID_PNG_BYTES, "image/png")}
        headers = {k: v for k, v in auth_headers.items() if k.lower() != "content-type"}
        res = client.post("/v1/assets/upload", headers=headers, files=files)
        assert res.status_code == 201
        data = res.json()
        assert data["asset_id"].startswith("ast-")
        assert data["s3_url"].startswith("https://s3.ai.example.com/assets/")
        assert data["mime_type"] == "image/png"
        assert data["verified"] is True

    def test_f09_asset_upload_audio(self, client: httpx.Client, auth_headers):
        files = {"file": ("sample.wav", VALID_WAV_BYTES, "audio/wav")}
        headers = {k: v for k, v in auth_headers.items() if k.lower() != "content-type"}
        res = client.post("/v1/assets/upload", headers=headers, files=files)
        assert res.status_code == 201
        data = res.json()
        assert data["mime_type"] == "audio/wav"
        assert "sha256" in data

    def test_f09_asset_upload_video(self, client: httpx.Client, auth_headers):
        files = {"file": ("sample.mp4", VALID_MP4_BYTES, "video/mp4")}
        headers = {k: v for k, v in auth_headers.items() if k.lower() != "content-type"}
        res = client.post("/v1/assets/upload", headers=headers, files=files)
        assert res.status_code == 201
        data = res.json()
        assert data["mime_type"] == "video/mp4"

    def test_f09_asset_validate_public_url(self, client: httpx.Client, auth_headers):
        payload = {
            "url": "https://cdn.example.com/images/hero-character.png",
            "mime_type": "image/png"
        }
        res = client.post("/v1/assets/validate", headers=auth_headers, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["ssrf_safe"] is True
        assert data["source_url"] == payload["url"]
        assert data["s3_url"].startswith("https://s3.ai.example.com/assets/")

    def test_f09_asset_upload_checksum_verification(self, client: httpx.Client, auth_headers):
        files = {"file": ("sample.jpg", VALID_JPEG_BYTES, "image/jpeg")}
        headers = {k: v for k, v in auth_headers.items() if k.lower() != "content-type"}
        res = client.post("/v1/assets/upload", headers=headers, files=files)
        assert res.status_code == 201
        data = res.json()
        assert "sha256" in data
        assert len(data["sha256"]) == 64
        assert int(data["size_bytes"]) > 0


