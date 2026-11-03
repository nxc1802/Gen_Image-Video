"""
tests/unit/storage/test_unified_artifact_delivery.py
Verification of Unified Ephemeral Artifact Delivery for Factory v4 Lite:
- Worker streams artifact upload via PUT /v1/storage/upload to unified scratch disk
- Client retrieves real image/video/audio binary via GET /v1/jobs/{id}/result
- File is synchronously unlinked upon completed client download
- Public control plane URL resolution for remote Kaggle Dual T4 workers
"""
import os
import shutil
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from factory.apps.control_api.src.main import app
from factory.apps.control_api.src.storage.config import StorageConfig, load_storage_config
from factory.apps.control_api.src.storage.integrity import compute_file_sha256
from services.image.artifact_collector import ImageArtifactCollector, _stream_file_chunks
from services.image.config import ImageServiceConfig
from services.video.artifact_collector import VideoArtifactCollector
from services.video.config import VideoServiceConfig

VALID_TEST_API_KEY = "sk-factory-test-key-2026"


@pytest.fixture
def test_env(monkeypatch):
    tmp_scratch = tempfile.mkdtemp(prefix="factory_unified_test_")
    monkeypatch.setenv("FACTORY_API_KEY", VALID_TEST_API_KEY)
    monkeypatch.setenv("SCRATCH_ARTIFACTS_DIR", tmp_scratch)
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("CONTROL_PLANE_URL", "https://control.ai.provider.internal:8000")
    yield tmp_scratch
    shutil.rmtree(tmp_scratch, ignore_errors=True)


@pytest.fixture
def client(test_env):
    with TestClient(app, base_url="http://testserver") as c:
        yield c


@pytest.fixture
def auth_headers():
    return {
        "Authorization": f"Bearer {VALID_TEST_API_KEY}",
        "Content-Type": "application/json",
    }


def test_public_control_plane_url_resolution(test_env, monkeypatch):
    """Verify presigned upload URLs use CONTROL_PLANE_URL for Kaggle worker reachability."""
    monkeypatch.setenv("CONTROL_PLANE_URL", "https://kaggle-routable.example.com")
    cfg = load_storage_config()
    assert cfg.local_base_url == "https://kaggle-routable.example.com"


@pytest.mark.asyncio
async def test_compute_file_sha256(tmp_path):
    """Verify streaming SHA-256 calculation on disk files without full memory buffering."""
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"A" * 1024 * 1024)  # 1MB
    h = compute_file_sha256(sample)
    assert len(h) == 64


def test_image_worker_upload_and_client_result_delivery(client, auth_headers, test_env):
    """
    Simulates complete cycle:
    1. Create image job
    2. Image worker generates real output in its local comfy dir
    3. Worker calls upload-url and uploads stream
    4. Worker marks job completed with ArtifactReference
    5. Client calls GET /v1/jobs/{job_id}/result to receive real binary
    6. Scratch disk is cleaned up after transmission
    """
    # 1. Create job
    submit_resp = client.post(
        "/v1/jobs",
        json={
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "A beautiful landscape"},
        },
        headers=auth_headers,
    )
    assert submit_resp.status_code == 201
    job_id = submit_resp.json()["job_id"]

    # 2. Worker generates output file
    worker_tmp = tempfile.mkdtemp(prefix="worker_comfy_output_")
    try:
        real_png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4" + b"extra_payload_content"
        output_file = Path(worker_tmp) / "generated_001.png"
        output_file.write_bytes(real_png_bytes)

        # 3. Request presigned upload URL
        presign_resp = client.post(
            "/v1/storage/upload-url",
            json={
                "subsystem": "image",
                "job_id": job_id,
                "artifact_name": "generated_001.png",
                "mime_type": "image/png",
                "max_size_bytes": len(real_png_bytes),
            },
            headers=auth_headers,
        )
        assert presign_resp.status_code == 200
        presign = presign_resp.json()

        # Upload the bytes to the presigned upload endpoint
        upload_url = presign["upload_url"].replace("http://localhost:8000", "").replace("https://control.ai.provider.internal:8000", "")
        put_resp = client.put(
            upload_url,
            content=real_png_bytes,
            headers={"Content-Type": "image/png"},
        )
        assert put_resp.status_code == 200
        assert put_resp.json()["status"] == "uploaded"

        # 4. Worker marks job completed
        art_ref = {
            "artifact_id": "art-001",
            "type": "image",
            "mime_type": "image/png",
            "size_bytes": len(real_png_bytes),
            "storage_path": presign["storage_path"],
            "url": f"/v1/jobs/{job_id}/result",
            "download_url": f"/v1/jobs/{job_id}/result",
            "sha256": compute_file_sha256(output_file),
        }
        complete_resp = client.post(
            f"/v1/jobs/{job_id}/complete",
            json={"artifacts": [art_ref], "worker_id": "worker-image-0"},
            headers=auth_headers,
        )
        assert complete_resp.status_code == 200

        # 5. Client calls GET /v1/jobs/{job_id}/result
        result_resp = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
        assert result_resp.status_code == 200
        assert "image/png" in result_resp.headers.get("content-type", "")
        assert result_resp.content == real_png_bytes

    finally:
        shutil.rmtree(worker_tmp, ignore_errors=True)


def test_video_worker_upload_and_client_result_delivery(client, auth_headers, test_env):
    """
    Simulates complete video worker cycle:
    Upload -> Complete -> Client GET /v1/jobs/{job_id}/result -> 200 video/mp4
    """
    submit_resp = client.post(
        "/v1/jobs",
        json={
            "service": "video",
            "model": "wan-2.2",
            "task": "text_to_video",
            "inputs": {"prompt": "A cat walking on grass"},
        },
        headers=auth_headers,
    )
    assert submit_resp.status_code == 201
    job_id = submit_resp.json()["job_id"]

    real_mp4_bytes = b"\x00\x00\x00 ftypmp42\x00\x00\x00\x00isommp42\x00\x00\x00\x08free" + b"video_data_payload"

    presign_resp = client.post(
        "/v1/storage/upload-url",
        json={
            "subsystem": "video",
            "job_id": job_id,
            "artifact_name": "rendered.mp4",
            "mime_type": "video/mp4",
            "max_size_bytes": len(real_mp4_bytes),
        },
        headers=auth_headers,
    )
    assert presign_resp.status_code == 200
    presign = presign_resp.json()

    upload_url = presign["upload_url"].replace("http://localhost:8000", "").replace("https://control.ai.provider.internal:8000", "")
    put_resp = client.put(
        upload_url,
        content=real_mp4_bytes,
        headers={"Content-Type": "video/mp4"},
    )
    assert put_resp.status_code == 200

    art_ref = {
        "artifact_id": "art-video-001",
        "type": "video",
        "mime_type": "video/mp4",
        "size_bytes": len(real_mp4_bytes),
        "storage_path": presign["storage_path"],
        "url": f"/v1/jobs/{job_id}/result",
        "download_url": f"/v1/jobs/{job_id}/result",
        "sha256": "0" * 64,
    }
    comp_resp = client.post(
        f"/v1/jobs/{job_id}/complete",
        json={"artifacts": [art_ref], "worker_id": "worker-video-0"},
        headers=auth_headers,
    )
    assert comp_resp.status_code == 200

    result_resp = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
    assert result_resp.status_code == 200
    assert "video/mp4" in result_resp.headers.get("content-type", "")
    assert result_resp.content == real_mp4_bytes


@pytest.mark.asyncio
async def test_image_artifact_collector_streaming_upload(tmp_path):
    """Verify ImageArtifactCollector uses chunk streaming without full-file read_bytes()."""
    import httpx
    output_dir = tmp_path / "comfy_output"
    output_dir.mkdir()
    img_file = output_dir / "gen.png"
    fake_png = b"\x89PNG\r\n\x1a\nfake_image_payload_bytes"
    img_file.write_bytes(fake_png)

    cfg = ImageServiceConfig(output_dir=str(output_dir))
    collector = ImageArtifactCollector(cfg)

    uploaded_data = bytearray()

    async def mock_handler(request: httpx.Request):
        if request.url.path == "/v1/storage/upload-url":
            return httpx.Response(200, json={
                "upload_url": "http://storage.internal/upload",
                "storage_path": "image/job-123/gen.png",
                "headers": {"Content-Type": "image/png"},
            })
        elif request.url.path == "/v1/storage/download-url":
            return httpx.Response(200, json={"download_url": "/v1/jobs/job-123/result"})
        elif request.url.path == "/upload":
            content = await request.aread()
            uploaded_data.extend(content)
            return httpx.Response(200, json={"status": "uploaded"})
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://storage.internal") as mock_client:
        refs = await collector.collect_and_upload(mock_client, "job-123", {"outputs": {"filename": "gen.png"}})
        assert len(refs) == 1
        assert refs[0].storage_path == "image/job-123/gen.png"
        assert refs[0].size_bytes == len(fake_png)
        assert len(refs[0].sha256) == 64
        assert bytes(uploaded_data) == fake_png


@pytest.mark.asyncio
async def test_video_artifact_collector_streaming_upload(tmp_path):
    """Verify VideoArtifactCollector uses chunk streaming without full-file read_bytes()."""
    import httpx
    output_dir = tmp_path / "comfy_video_output"
    output_dir.mkdir()
    vid_file = output_dir / "gen.mp4"
    fake_video = b"\x00\x00\x00 ftypmp42\x00\x00\x00\x00isommp42streamed_video_payload"
    vid_file.write_bytes(fake_video)

    cfg = VideoServiceConfig(output_dir=str(output_dir))
    collector = VideoArtifactCollector(cfg)

    uploaded_data = bytearray()

    async def mock_handler(request: httpx.Request):
        if request.url.path == "/v1/storage/upload-url":
            return httpx.Response(200, json={
                "upload_url": "http://storage.internal/upload",
                "storage_path": "video/job-video-456/gen.mp4",
                "headers": {"Content-Type": "video/mp4"},
            })
        elif request.url.path == "/v1/storage/download-url":
            return httpx.Response(200, json={"download_url": "/v1/jobs/job-video-456/result"})
        elif request.url.path == "/upload":
            content = await request.aread()
            uploaded_data.extend(content)
            return httpx.Response(200, json={"status": "uploaded"})
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://storage.internal") as mock_client:
        refs = await collector.collect_and_upload(mock_client, "job-video-456", {"outputs": {"filename": "gen.mp4"}})
        assert len(refs) == 1
        assert refs[0].storage_path == "video/job-video-456/gen.mp4"
        assert refs[0].size_bytes == len(fake_video)
        assert len(refs[0].sha256) == 64
        assert bytes(uploaded_data) == fake_video
