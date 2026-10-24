"""
Tier 1 Feature Coverage: Asset Service (Feature 9)
Validates happy path and contract adherence for:
- Feature 9: Asset Service Upload / SSRF Validation (POST /v1/assets/upload, POST /v1/assets/validate)
"""

import pytest
import httpx


@pytest.mark.tier1
class TestAssetService:
    def test_f09_asset_upload_image(self, client: httpx.Client, auth_headers):
        payload = {"mime_type": "image/png"}
        headers = {**auth_headers, "Content-Type": "application/json"}
        res = client.post("/v1/assets/upload", headers=headers, json=payload)
        assert res.status_code == 201
        data = res.json()
        assert data["asset_id"].startswith("ast-")
        assert data["s3_url"].startswith("https://s3.ai.example.com/assets/")
        assert data["mime_type"] == "image/png"
        assert data["verified"] is True

    def test_f09_asset_upload_audio(self, client: httpx.Client, auth_headers):
        payload = {"mime_type": "audio/wav"}
        res = client.post("/v1/assets/upload", headers=auth_headers, json=payload)
        assert res.status_code == 201
        data = res.json()
        assert data["mime_type"] == "audio/wav"
        assert "sha256" in data

    def test_f09_asset_upload_video(self, client: httpx.Client, auth_headers):
        payload = {"mime_type": "video/mp4"}
        res = client.post("/v1/assets/upload", headers=auth_headers, json=payload)
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
        res = client.post("/v1/assets/upload", headers=auth_headers, json={"mime_type": "image/jpeg"})
        data = res.json()
        assert "sha256" in data
        assert len(data["sha256"]) == 64
        assert int(data["size_bytes"]) > 0
