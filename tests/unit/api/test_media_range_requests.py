"""
tests/unit/api/test_media_range_requests.py
Unit tests verifying HTTP Range requests and media delivery hardening (Requirement R3, P1.6):
- HTTP 206 Partial Content with Content-Range and Accept-Ranges: bytes
- Valid sliced bytes matching range specifications
- Video retention across multiple sequential range requests (no premature unlinking)
- Local storage download headers (Accept-Ranges and inline Content-Disposition for video)
"""
import io
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from factory.apps.control_api.src.main import app
from factory_contracts.jobs.job_contract import (
    FactoryJob,
    JobProgress,
    JobState,
    JobTelemetry,
    ArtifactReference,
    AssetRecord,
)

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


@pytest.mark.asyncio
async def test_asset_download_range_request_returns_206(api_client, auth_headers):
    # Upload valid PNG asset
    payload = VALID_PNG_BYTES
    total_len = len(payload)
    files = {"file": ("test_image.png", io.BytesIO(payload), "image/png")}
    up_resp = api_client.post("/v1/assets/upload", files=files, headers=auth_headers)
    assert up_resp.status_code == 201
    asset_id = up_resp.json()["artifact_id"]

    # 1. Full download (200 OK)
    full_resp = api_client.get(f"/v1/assets/{asset_id}/download", headers=auth_headers)
    assert full_resp.status_code == 200
    assert full_resp.headers["Accept-Ranges"] == "bytes"
    assert full_resp.headers["Content-Length"] == str(total_len)
    assert full_resp.content == payload

    # 2. Range request: bytes=10-29 (20 bytes)
    range_headers = dict(auth_headers)
    range_headers["Range"] = "bytes=10-29"
    range_resp = api_client.get(f"/v1/assets/{asset_id}/download", headers=range_headers)
    assert range_resp.status_code == 206
    assert range_resp.headers["Content-Range"] == f"bytes 10-29/{total_len}"
    assert range_resp.headers["Accept-Ranges"] == "bytes"
    assert range_resp.headers["Content-Length"] == "20"
    assert range_resp.content == payload[10:30]

    # 3. Open-ended range request: bytes=50-
    range_headers["Range"] = "bytes=50-"
    tail_resp = api_client.get(f"/v1/assets/{asset_id}/download", headers=range_headers)
    assert tail_resp.status_code == 206
    assert tail_resp.headers["Content-Range"] == f"bytes 50-{total_len - 1}/{total_len}"
    assert tail_resp.content == payload[50:]


@pytest.mark.asyncio
async def test_video_artifact_range_request_returns_206(api_client, auth_headers):
    # Setup completed video job on scratch disk
    job_id = "job-11111111-2222-3333-4444-555555555555"
    scratch = app.state.scratch_disk
    repo = app.state.repository

    payload = b"A" * 4096
    art_path = scratch.get_artifact_path(job_id, f"{job_id}.mp4")
    art_path.parent.mkdir(parents=True, exist_ok=True)
    art_path.write_bytes(payload)

    job = FactoryJob(
        job_id=job_id,
        request_id="req-11111111-2222-3333-4444-555555555555",
        service="video",
        model="minimax-h3",
        task="text_to_video",
        inputs={"prompt": "ocean waves"},
        state=JobState.SUCCEEDED,
        progress=JobProgress(stage="completed", percent=100.0),
        artifacts=[
            ArtifactReference(
                artifact_id="art-01",
                type="video",
                mime_type="video/mp4",
                size_bytes=4096,
                storage_path=f"artifacts/{job_id}/{job_id}.mp4",
                download_url=f"/v1/jobs/{job_id}/result",
                sha256="a" * 64,
            )
        ],
    )
    await repo.create_job(job)

    # Issue Range request: bytes=0-1023
    headers = dict(auth_headers)
    headers["Range"] = "bytes=0-1023"
    resp = api_client.get(f"/v1/jobs/{job_id}/result", headers=headers)
    assert resp.status_code == 206
    assert resp.headers["Accept-Ranges"] == "bytes"
    assert resp.headers["Content-Range"] == "bytes 0-1023/4096"
    assert len(resp.content) == 1024
    assert resp.content == payload[0:1024]
    assert "inline" in resp.headers["Content-Disposition"]


def test_video_retention_across_multiple_range_requests(api_client, auth_headers):
    # Sequential range requests must not delete the video artifact prematurely
    job_id = "job-22222222-3333-4444-5555-666666666666"
    scratch = app.state.scratch_disk
    repo = app.state.repository

    payload = b"VIDEO_CHUNK_" * 500  # 6000 bytes
    art_path = scratch.get_artifact_path(job_id, f"{job_id}.mp4")
    art_path.parent.mkdir(parents=True, exist_ok=True)
    art_path.write_bytes(payload)

    import asyncio
    asyncio.run(repo.create_job(FactoryJob(
        job_id=job_id,
        request_id="req-22222222-3333-4444-5555-666666666666",
        service="video",
        model="minimax-h3",
        task="text_to_video",
        inputs={"prompt": "clouds"},
        state=JobState.SUCCEEDED,
        progress=JobProgress(stage="completed", percent=100.0),
        artifacts=[
            ArtifactReference(
                artifact_id="art-02",
                type="video",
                mime_type="video/mp4",
                size_bytes=len(payload),
                storage_path=f"artifacts/{job_id}/{job_id}.mp4",
                download_url=f"/v1/jobs/{job_id}/result",
                sha256="b" * 64,
            )
        ],
    )))

    ranges = [
        ("bytes=0-999", 0, 1000),
        ("bytes=1000-1999", 1000, 2000),
        ("bytes=2000-2999", 2000, 3000),
        ("bytes=3000-3999", 3000, 4000),
        ("bytes=4000-4999", 4000, 5000),
    ]

    for rng, start, end in ranges:
        headers = dict(auth_headers)
        headers["Range"] = rng
        resp = api_client.get(f"/v1/jobs/{job_id}/result", headers=headers)
        assert resp.status_code == 206
        assert len(resp.content) == end - start
        assert resp.content == payload[start:end]

    # After 5 range requests, artifact must still exist on scratch disk
    assert art_path.exists(), "Artifact file was prematurely deleted after Range request!"


def test_local_storage_download_headers(api_client):
    storage = app.state.storage_provider
    key = "artifacts/video_download_test/sample.mp4"
    path = storage._get_path(storage.config.s3_bucket_name, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"dummy-mp4-stream-data")

    download_url = storage.generate_presigned_download_url(
        bucket=storage.config.s3_bucket_name,
        key=key,
        expires_in_seconds=3600,
        filename="custom_video.mp4",
    )
    rel_url = download_url.replace(storage.base_url, "")

    resp = api_client.get(rel_url)
    assert resp.status_code == 200
    assert resp.headers["Accept-Ranges"] == "bytes"
    assert "inline" in resp.headers["Content-Disposition"]
    assert "custom_video.mp4" in resp.headers["Content-Disposition"]
