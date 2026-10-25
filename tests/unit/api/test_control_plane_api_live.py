"""
tests/unit/api/test_control_plane_api_live.py
Unit tests executing against the production FastAPI Control Plane application
(factory.apps.control_api.src.main.app) verifying the end-to-end integration of
authentication, job lifecycle, worker leasing, storage, streaming, and telemetry.
"""
import base64
import os
import pytest
from starlette.testclient import TestClient

from factory.apps.control_api.src.main import app

VALID_TOKEN = "sk-factory-prod-secret-token"


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
        "Content-Type": "application/json",
    }


class TestControlPlaneLiveAPI:
    def test_health_check(self, api_client):
        res = api_client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["plane"] == "control"
        assert data["version"] == "4.0.0"

    def test_unauthenticated_request_rejected(self, api_client):
        res = api_client.post("/v1/jobs", json={"service": "image"})
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "UNAUTHORIZED"

    def test_job_submission_and_query(self, api_client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "A modern AI control center", "width": 1024, "height": 1024},
            "priority": 15,
        }
        res = api_client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        job = res.json()
        assert job["job_id"].startswith("job-")
        assert job["request_id"].startswith("req-")
        assert job["state"] == "QUEUED"
        assert job["priority"] == 15

        # Response headers check for tracing
        assert "traceparent" in res.headers
        assert "X-Trace-ID" in res.headers
        assert "X-Request-ID" in res.headers

        # Query status
        get_res = api_client.get(f"/v1/jobs/{job['job_id']}", headers=auth_headers)
        assert get_res.status_code == 200
        get_data = get_res.json()
        assert get_data["job_id"] == job["job_id"]
        assert get_data["state"] == "QUEUED"

    def test_job_submission_idempotency(self, api_client, auth_headers):
        idem_key = "test-live-idem-key-001"
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "text_to_video",
            "inputs": {"prompt": "Cinematic flying drone shot"},
            "idempotency_key": idem_key,
        }
        res1 = api_client.post("/v1/jobs", headers=auth_headers, json=payload)
        res2 = api_client.post("/v1/jobs", headers=auth_headers, json=payload)

        assert res1.status_code == 201
        assert res2.status_code == 200
        assert res1.json()["job_id"] == res2.json()["job_id"]

    def test_inline_payload_over_1mb_rejected_with_413(self, api_client, auth_headers):
        huge_raw = b"x" * int(1.2 * 1024 * 1024)
        huge_b64 = "data:image/png;base64," + base64.b64encode(huge_raw).decode("utf-8")
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "A sunset", "image_url": huge_b64},
        }
        res = api_client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 413
        err = res.json()["error"]
        assert err["code"] == "PAYLOAD_TOO_LARGE"
        assert "exceeds the 1MB threshold" in err["message"]

    def test_worker_leasing_heartbeat_and_completion_lifecycle(self, api_client, auth_headers):
        # 1. Submit job
        sub_res = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "robot arm assembling circuit board"},
                "priority": 20,
            },
        )
        assert sub_res.status_code == 201
        job_id = sub_res.json()["job_id"]

        # 2. Worker leases job
        lease_res = api_client.post(
            "/v1/jobs/lease",
            headers=auth_headers,
            json={
                "worker_id": "worker-gpu-image-0",
                "subsystem": "image",
                "supported_models": ["qwen-image-2.1"],
                "supported_tasks": ["text_to_image"],
                "gpu_device": 0,
                "lease_duration_seconds": 300,
            },
        )
        assert lease_res.status_code == 200
        lease_data = lease_res.json()
        assert lease_data["job"] is not None
        assert lease_data["lease_token"] is not None

        # 3. Worker heartbeat
        hb_res = api_client.post(
            "/v1/workers/worker-gpu-image-0/heartbeat",
            headers=auth_headers,
            json={
                "worker_id": "worker-gpu-image-0",
                "active_job_id": job_id,
                "lease_token": lease_data["lease_token"],
                "gpu_utilization_percent": 90.0,
                "vram_used_mb": 11500.0,
            },
        )
        assert hb_res.status_code == 200
        assert hb_res.json()["status"] == "acknowledged"

        # 4. Worker reports progress
        prog_res = api_client.post(
            f"/v1/jobs/{job_id}/progress",
            headers=auth_headers,
            json={
                "stage": "sampling",
                "step": 10,
                "total": 20,
                "percent": 50.0,
                "message": "Denoising step 10/20",
                "worker_id": "worker-gpu-image-0",
                "lease_token": lease_data["lease_token"],
            },
        )
        assert prog_res.status_code == 200

        # 5. Worker completes job with S3 artifact
        comp_res = api_client.post(
            f"/v1/jobs/{job_id}/complete",
            headers=auth_headers,
            json={
                "artifacts": [
                    {
                        "artifact_id": "art-live-01",
                        "type": "image",
                        "mime_type": "image/png",
                        "size_bytes": 102400,
                        "storage_path": f"image/{job_id}/output.png",
                        "download_url": f"https://s3.ai.example.com/artifacts/image/{job_id}/output.png",
                        "sha256": "f" * 64,
                    }
                ],
                "inference_duration_ms": 3200.0,
                "peak_vram_mb": 11500.0,
                "worker_id": "worker-gpu-image-0",
                "lease_token": lease_data["lease_token"],
            },
        )
        assert comp_res.status_code == 200

        # Verify job is SUCCEEDED
        final_job = api_client.get(f"/v1/jobs/{job_id}", headers=auth_headers).json()
        assert final_job["state"] == "SUCCEEDED"
        assert len(final_job["artifacts"]) == 1
        assert final_job["artifacts"][0]["url"] is not None

    def test_cancellation_and_retry(self, api_client, auth_headers):
        # 1. Submit job
        sub_res = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "service": "music",
                "model": "ace-step-1.5",
                "task": "text_to_music",
                "inputs": {"prompt": "ambient soundscape"},
            },
        )
        job_id = sub_res.json()["job_id"]

        # 2. Cancel
        cancel_res = api_client.delete(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert cancel_res.status_code == 200
        assert cancel_res.json()["state"] == "CANCELLED"

        # Idempotent cancel
        cancel_dup = api_client.post(f"/v1/jobs/{job_id}/cancel", headers=auth_headers)
        assert cancel_dup.status_code == 200

        # 3. Retry cancelled job
        retry_res = api_client.post(f"/v1/jobs/{job_id}/retry", headers=auth_headers)
        assert retry_res.status_code == 200
        retry_data = retry_res.json()
        assert retry_data["state"] == "QUEUED"
        assert retry_data["retry_count"] == 1

    def test_websocket_discovery_endpoint(self, api_client, auth_headers):
        sub_res = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "service": "speech",
                "model": "qwen3-tts",
                "task": "text_to_speech",
                "inputs": {"text": "Welcome to Factory v4"},
            },
        )
        job_id = sub_res.json()["job_id"]

        ws_res = api_client.get(f"/v1/jobs/{job_id}/ws", headers=auth_headers)
        assert ws_res.status_code == 200
        data = ws_res.json()
        assert "websocket_url" in data
        assert f"/v1/jobs/{job_id}/ws" in data["websocket_url"]
        assert data["status"] == "active"

    def test_storage_presigned_upload_url(self, api_client, auth_headers):
        res = api_client.post(
            "/v1/storage/upload-url",
            headers=auth_headers,
            json={
                "subsystem": "image",
                "job_id": "job-test-upload-01",
                "artifact_name": "output.png",
                "mime_type": "image/png",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert "upload_url" in data
        assert data["method"] == "PUT"
        assert data["storage_path"] == "image/job-test-upload-01/output.png"

    def test_websocket_stream_unauthenticated_closed_with_1008(self, api_client, auth_headers):
        sub_res = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "A cat"},
            },
        )
        job_id = sub_res.json()["job_id"]
        from starlette.websockets import WebSocketDisconnect
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with api_client.websocket_connect(f"/v1/jobs/{job_id}/ws"):
                pass
        assert exc_info.value.code == 1008

    def test_websocket_stream_invalid_token_closed_with_1008(self, api_client, auth_headers):
        sub_res = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "A cat"},
            },
        )
        job_id = sub_res.json()["job_id"]
        from starlette.websockets import WebSocketDisconnect
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with api_client.websocket_connect(f"/v1/jobs/{job_id}/ws?token=invalid-key"):
                pass
        assert exc_info.value.code == 1008

    def test_websocket_stream_authenticated_via_query_param(self, api_client, auth_headers):
        sub_res = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "A cat"},
            },
        )
        job_id = sub_res.json()["job_id"]
        with api_client.websocket_connect(f"/v1/jobs/{job_id}/ws?token={VALID_TOKEN}") as ws:
            assert ws is not None

    def test_websocket_stream_authenticated_via_auth_header(self, api_client, auth_headers):
        sub_res = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "A cat"},
            },
        )
        job_id = sub_res.json()["job_id"]
        with api_client.websocket_connect(
            f"/v1/jobs/{job_id}/ws",
            headers={"Authorization": f"Bearer {VALID_TOKEN}"},
        ) as ws:
            assert ws is not None

