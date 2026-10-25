"""
tests/unit/api/test_assets_upload.py
Unit tests verifying the Asset Upload, Download, and SSRF Validation API:
- Unauthenticated access rejection (401)
- Multipart/form-data media upload and validation (201)
- Binary download roundtrip and media-type header preservation (200)
- 404 Not Found for non-existent assets
- Legacy/contract JSON upload simulation (201)
- 415 Unsupported Media Type for invalid/empty MIME types
- 413 Payload Too Large for oversized uploads
- 400 Invalid Media Type for corrupted/invalid headers
- URL validation and SSRF defenses (400)
"""
import hashlib
import os
import pytest
from starlette.testclient import TestClient

from factory.apps.control_api.src.main import app

VALID_TOKEN = "sk-factory-prod-secret-token"
VALID_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\n\x00\x00\x00\n\x08\x02\x00\x00\x00\x02PX\xea"
    b"\x00\x00\x00\x13IDATx\x9cc\xfc\xcf\x80\x0f0\xe1\x95e\x18\xa9\xd2\x00A,\x01\x13y\xed\xba&\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture(scope="module")
def api_client():
    keys_to_preserve = ["FACTORY_API_KEY", "STORAGE_BACKEND", "FACTORY_DB_BACKEND", "SQLITE_DB_PATH"]
    old_env = {k: os.environ.get(k) for k in keys_to_preserve}
    os.environ["FACTORY_API_KEY"] = VALID_TOKEN
    os.environ["STORAGE_BACKEND"] = "local"
    os.environ["FACTORY_DB_BACKEND"] = "sqlite"
    os.environ["SQLITE_DB_PATH"] = ":memory:"
    try:
        with TestClient(app) as client:
            yield client
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if hasattr(app.state, "auth_config"):
            delattr(app.state, "auth_config")


@pytest.fixture
def auth_headers():
    return {
        "Authorization": f"Bearer {VALID_TOKEN}",
    }


class TestAssetsUploadRouter:
    def test_unauthenticated_asset_upload_rejected(self, api_client):
        res = api_client.post("/v1/assets/upload", json={"mime_type": "image/png"})
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "UNAUTHORIZED"

    def test_unauthenticated_asset_download_rejected(self, api_client):
        res = api_client.get("/v1/assets/ast-any/download")
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "UNAUTHORIZED"

    def test_valid_png_multipart_upload_and_download(self, api_client, auth_headers):
        # 1. Upload valid multipart file
        expected_sha256 = hashlib.sha256(VALID_PNG_BYTES).hexdigest()
        files = {"file": ("pixel.png", VALID_PNG_BYTES, "image/png")}
        res = api_client.post("/v1/assets/upload", headers=auth_headers, files=files)
        assert res.status_code == 201
        data = res.json()

        assert data["asset_id"].startswith("ast-")
        assert data["artifact_id"] == data["asset_id"]
        assert data["mime_type"] == "image/png"
        assert data["size_bytes"] == len(VALID_PNG_BYTES)
        assert data["sha256"] == expected_sha256
        assert data["download_url"].startswith("/v1/assets/ast-")
        assert data["url"] == data["download_url"]
        assert data["s3_path"].startswith("assets/")
        assert "s3_url" in data
        assert data["verified"] is True
        assert "created_at" in data

        asset_id = data["asset_id"]

        # 2. Download stored file
        dl_res = api_client.get(f"/v1/assets/{asset_id}/download", headers=auth_headers)
        assert dl_res.status_code == 200
        assert dl_res.headers["content-type"].startswith("image/png")
        assert dl_res.content == VALID_PNG_BYTES

    def test_asset_download_not_found(self, api_client, auth_headers):
        res = api_client.get("/v1/assets/ast-non-existent-99999/download", headers=auth_headers)
        assert res.status_code == 404
        err = res.json()["error"]
        assert err["code"] == "NOT_FOUND"

    def test_legacy_json_upload(self, api_client, auth_headers):
        headers = {**auth_headers, "Content-Type": "application/json"}
        res = api_client.post("/v1/assets/upload", headers=headers, json={"mime_type": "image/png"})
        assert res.status_code == 201
        data = res.json()
        assert data["asset_id"].startswith("ast-")
        assert data["mime_type"] == "image/png"
        assert data["s3_url"].startswith("https://s3.ai.example.com/assets/")
        assert data["verified"] is True
        assert len(data["sha256"]) == 64

    def test_unsupported_mime_rejected_415(self, api_client, auth_headers):
        headers = {**auth_headers, "Content-Type": "application/json"}
        res = api_client.post("/v1/assets/upload", headers=headers, json={"mime_type": "text/html"})
        assert res.status_code == 415
        err = res.json()["error"]
        assert err["code"] == "UNSUPPORTED_MEDIA_TYPE"

    def test_empty_mime_rejected_415(self, api_client, auth_headers):
        headers = {**auth_headers, "Content-Type": "application/json"}
        res = api_client.post("/v1/assets/upload", headers=headers, json={"mime_type": ""})
        assert res.status_code == 415
        err = res.json()["error"]
        assert err["code"] == "UNSUPPORTED_MEDIA_TYPE"

    def test_oversized_payload_rejected_413(self, api_client, auth_headers):
        headers = {
            **auth_headers,
            "Content-Length": str(55 * 1024 * 1024),
            "Content-Type": "application/json",
        }
        res = api_client.post("/v1/assets/upload", headers=headers, json={"mime_type": "image/png"})
        assert res.status_code == 413
        err = res.json()["error"]
        assert err["code"] == "PAYLOAD_TOO_LARGE"

    def test_corrupt_media_rejected_400(self, api_client, auth_headers):
        corrupt_bytes = b"NOT_A_VALID_IMAGE_JUST_PLAIN_TEXT_DATA"
        files = {"file": ("corrupt.png", corrupt_bytes, "image/png")}
        res = api_client.post("/v1/assets/upload", headers=auth_headers, files=files)
        assert res.status_code == 400
        err = res.json()["error"]
        assert err["code"] == "INVALID_MEDIA_TYPE"

    def test_asset_validate_empty_url_400(self, api_client, auth_headers):
        headers = {**auth_headers, "Content-Type": "application/json"}
        res = api_client.post("/v1/assets/validate", headers=headers, json={"url": ""})
        assert res.status_code == 400
        err = res.json()["error"]
        assert err["code"] == "INVALID_REQUEST"

    def test_asset_validate_invalid_scheme_400(self, api_client, auth_headers):
        headers = {**auth_headers, "Content-Type": "application/json"}
        res = api_client.post("/v1/assets/validate", headers=headers, json={"url": "file:///etc/passwd"})
        assert res.status_code == 400
        err = res.json()["error"]
        assert err["code"] == "INVALID_REQUEST"

    def test_asset_validate_ssrf_blocked_400(self, api_client, auth_headers):
        headers = {**auth_headers, "Content-Type": "application/json"}
        res = api_client.post("/v1/assets/validate", headers=headers, json={"url": "http://127.0.0.1:8000/internal"})
        assert res.status_code == 400
        err = res.json()["error"]
        assert err["code"] == "SSRF_BLOCKED"

    def test_asset_validate_public_url_200(self, api_client, auth_headers):
        headers = {**auth_headers, "Content-Type": "application/json"}
        payload = {
            "url": "https://cdn.example.com/images/hero-character.png",
            "mime_type": "image/png",
        }
        res = api_client.post("/v1/assets/validate", headers=headers, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["ssrf_safe"] is True
        assert data["source_url"] == payload["url"]
        assert data["s3_url"].startswith("https://s3.ai.example.com/assets/")
