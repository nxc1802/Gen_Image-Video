"""
Tier 1 Feature Coverage: Control Plane (Features 1 - 6)
Validates happy path and contract adherence for:
- Feature 1: Job Submission (POST /v1/jobs)
- Feature 2: Job Status Query (GET /v1/jobs/{job_id})
- Feature 3: Job Cancellation (DELETE /v1/jobs/{job_id})
- Feature 4: Job Retry (POST /v1/jobs/{job_id}/retry)
- Feature 5: Job SSE Events (GET /v1/jobs/{job_id}/events)
- Feature 6: Job WebSocket (GET /v1/jobs/{job_id}/ws)
"""

import pytest
import httpx
from tests.e2e.conftest import assert_valid_job


@pytest.mark.tier1
class TestControlPlaneJobs:
    # --- Feature 1: Job Submission ---
    def test_f01_job_submission_standard(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "A futuristic digital factory with robots assembling solar panels", "width": 1024, "height": 1024}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        data = res.json()
        assert_valid_job(data, "image", "text_to_image")
        assert data["state"] == "QUEUED"

    def test_f01_job_submission_with_priority(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "video",
            "model": "wan-2.2",
            "task": "text_to_video",
            "inputs": {"prompt": "Cinematic camera pan over misty mountains", "num_frames": 49},
            "priority": 10
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        data = res.json()
        assert data["priority"] == 10

    def test_f01_job_submission_idempotency(self, client: httpx.Client, auth_headers):
        idem_key = "test-idem-key-999"
        payload = {
            "service": "music",
            "model": "ace-step-1.5",
            "task": "text_to_music",
            "inputs": {"prompt": "Cyberpunk synthwave bassline", "duration_seconds": 30},
            "idempotency_key": idem_key
        }
        res1 = client.post("/v1/jobs", headers=auth_headers, json=payload)
        res2 = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res1.status_code == 201
        assert res2.status_code == 200
        assert res1.json()["job_id"] == res2.json()["job_id"]

    def test_f01_job_submission_metadata_preservation(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "enhancement",
            "model": "enhancement-suite-v1",
            "task": "image_super_resolution",
            "inputs": {"image_url": "https://s3.ai.example.com/assets/sample.png", "scale_factor": 4}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        data = res.json()
        assert data["service"] == "enhancement"
        assert data["task"] == "image_super_resolution"

    def test_f01_job_submission_generates_request_id(self, client: httpx.Client, auth_headers):
        payload = {
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Simple cube on white background"}
        }
        res = client.post("/v1/jobs", headers=auth_headers, json=payload)
        assert res.status_code == 201
        assert res.json()["request_id"].startswith("req-")

    # --- Feature 2: Job Status Query ---
    def test_f02_job_status_query_queued(self, client: httpx.Client, auth_headers):
        # Create job
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image", "model": "qwen-image-2.1", "task": "text_to_image",
            "inputs": {"prompt": "A majestic dragon"}
        })
        job_id = sub.json()["job_id"]

        # Query status
        res = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert data["job_id"] == job_id
        assert data["state"] == "QUEUED"

    def test_f02_job_status_query_progress_object(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image", "model": "qwen-image-2.1", "task": "text_to_image",
            "inputs": {"prompt": "Forest landscape"}
        })
        job_id = sub.json()["job_id"]
        res = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        prog = res.json()["progress"]
        assert "stage" in prog
        assert "percent" in prog
        assert isinstance(prog["percent"], (int, float))

    def test_f02_job_status_query_artifacts_list(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "video", "model": "wan-2.2", "task": "text_to_video",
            "inputs": {"prompt": "Ocean waves"}
        })
        job_id = sub.json()["job_id"]
        res = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        artifacts = res.json()["artifacts"]
        assert isinstance(artifacts, list)
        assert len(artifacts) >= 1
        assert artifacts[0]["url"].startswith("https://s3.ai.example.com/")

    def test_f02_job_status_query_worker_lease(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "music", "model": "ace-step-1.5", "task": "text_to_music",
            "inputs": {"prompt": "Acoustic folk song"}
        })
        job_id = sub.json()["job_id"]
        res = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        worker = res.json()["worker"]
        assert "worker_id" in worker
        assert "lease_expires_at" in worker

    def test_f02_job_status_query_timestamps(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image", "model": "qwen-image-2.1", "task": "text_to_image",
            "inputs": {"prompt": "Cybernetic city"}
        })
        job_id = sub.json()["job_id"]
        res = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        data = res.json()
        assert "created_at" in data
        assert "T" in data["created_at"]

    # --- Feature 3: Job Cancellation ---
    def test_f03_job_cancellation_active(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image", "model": "qwen-image-2.1", "task": "text_to_image",
            "inputs": {"prompt": "Cancel test prompt"}
        })
        job_id = sub.json()["job_id"]

        del_res = client.delete(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert del_res.status_code == 200
        assert del_res.json()["state"] == "CANCELLED"

    def test_f03_job_cancellation_state_reflected_in_query(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "video", "model": "wan-2.2", "task": "text_to_video",
            "inputs": {"prompt": "Running cheetah"}
        })
        job_id = sub.json()["job_id"]
        client.delete(f"/v1/jobs/{job_id}", headers=auth_headers)

        status_res = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert status_res.json()["state"] == "CANCELLED"

    def test_f03_job_cancellation_records_completed_at(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "music", "model": "ace-step-1.5", "task": "text_to_music",
            "inputs": {"prompt": "Orchestral overture"}
        })
        job_id = sub.json()["job_id"]
        client.delete(f"/v1/jobs/{job_id}", headers=auth_headers)

        status_res = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert status_res.json()["completed_at"] is not None

    def test_f03_job_cancellation_message(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image", "model": "qwen-image-2.1", "task": "text_to_image",
            "inputs": {"prompt": "Steampunk blimp"}
        })
        job_id = sub.json()["job_id"]
        del_res = client.delete(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert "message" in del_res.json()

    def test_f03_job_cancellation_idempotent_or_safe(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image", "model": "qwen-image-2.1", "task": "text_to_image",
            "inputs": {"prompt": "Double cancel test"}
        })
        job_id = sub.json()["job_id"]
        client.delete(f"/v1/jobs/{job_id}", headers=auth_headers)
        del_res2 = client.delete(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert del_res2.status_code == 200
        assert del_res2.json()["state"] == "CANCELLED"

    # --- Feature 4: Job Retry ---
    def test_f04_job_retry_cancelled_job(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image", "model": "qwen-image-2.1", "task": "text_to_image",
            "inputs": {"prompt": "Retry prompt"}
        })
        job_id = sub.json()["job_id"]
        client.delete(f"/v1/jobs/{job_id}", headers=auth_headers)

        retry_res = client.post(f"/v1/jobs/{job_id}/retry", headers=auth_headers)
        assert retry_res.status_code == 200
        assert retry_res.json()["state"] == "QUEUED"
        assert retry_res.json()["retry_count"] == 1

    def test_f04_job_retry_count_increments(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "video", "model": "wan-2.2", "task": "text_to_video",
            "inputs": {"prompt": "Incremental retry"}
        })
        job_id = sub.json()["job_id"]
        r1 = client.post(f"/v1/jobs/{job_id}/retry", headers=auth_headers)
        r2 = client.post(f"/v1/jobs/{job_id}/retry", headers=auth_headers)
        assert r1.json()["retry_count"] == 1
        assert r2.json()["retry_count"] == 2

    def test_f04_job_retry_resets_progress(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "music", "model": "ace-step-1.5", "task": "text_to_music",
            "inputs": {"prompt": "Progress reset check"}
        })
        job_id = sub.json()["job_id"]
        r = client.post(f"/v1/jobs/{job_id}/retry", headers=auth_headers)
        prog = r.json()["progress"]
        assert prog["percent"] == 0
        assert prog["stage"] == "queued_retry"

    def test_f04_job_retry_preserves_artifacts_contract(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "enhancement", "model": "enhancement-suite-v1", "task": "image_super_resolution",
            "inputs": {"image_url": "https://s3.ai.example.com/assets/sample.png", "scale_factor": 2}
        })
        job_id = sub.json()["job_id"]
        r = client.post(f"/v1/jobs/{job_id}/retry", headers=auth_headers)
        assert "artifacts" in r.json()

    def test_f04_job_retry_state_retrievable_via_get(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image", "model": "qwen-image-2.1", "task": "text_to_image",
            "inputs": {"prompt": "Retried retrieval check"}
        })
        job_id = sub.json()["job_id"]
        client.post(f"/v1/jobs/{job_id}/retry", headers=auth_headers)
        get_res = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert get_res.json()["state"] == "QUEUED"
        assert get_res.json()["retry_count"] == 1

    # --- Feature 5: Job SSE Events ---
    def test_f05_job_sse_events_status_code(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image", "model": "qwen-image-2.1", "task": "text_to_image",
            "inputs": {"prompt": "SSE test"}
        })
        job_id = sub.json()["job_id"]
        res = client.get(f"/v1/jobs/{job_id}/events", headers=auth_headers)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers.get("content-type", "")

    def test_f05_job_sse_events_contains_progress(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image", "model": "qwen-image-2.1", "task": "text_to_image",
            "inputs": {"prompt": "SSE progress check"}
        })
        job_id = sub.json()["job_id"]
        res = client.get(f"/v1/jobs/{job_id}/events", headers=auth_headers)
        text = res.text
        assert "event: progress" in text
        assert "stage" in text

    def test_f05_job_sse_events_contains_completed(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "video", "model": "wan-2.2", "task": "text_to_video",
            "inputs": {"prompt": "SSE completed check"}
        })
        job_id = sub.json()["job_id"]
        res = client.get(f"/v1/jobs/{job_id}/events", headers=auth_headers)
        assert "event: completed" in res.text
        assert "SUCCEEDED" in res.text

    def test_f05_job_sse_events_includes_job_id(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "music", "model": "ace-step-1.5", "task": "text_to_music",
            "inputs": {"prompt": "SSE job id correlation"}
        })
        job_id = sub.json()["job_id"]
        res = client.get(f"/v1/jobs/{job_id}/events", headers=auth_headers)
        assert job_id in res.text

    def test_f05_job_sse_events_artifacts_in_completion(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image", "model": "qwen-image-2.1", "task": "text_to_image",
            "inputs": {"prompt": "SSE artifact check"}
        })
        job_id = sub.json()["job_id"]
        res = client.get(f"/v1/jobs/{job_id}/events", headers=auth_headers)
        assert "https://s3.ai.example.com/" in res.text

    # --- Feature 6: Job WebSocket ---
    def test_f06_job_ws_endpoint_status(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image", "model": "qwen-image-2.1", "task": "text_to_image",
            "inputs": {"prompt": "WS test"}
        })
        job_id = sub.json()["job_id"]
        res = client.get(f"/v1/jobs/{job_id}/ws", headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["status"] == "active"

    def test_f06_job_ws_url_scheme(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "video", "model": "wan-2.2", "task": "text_to_video",
            "inputs": {"prompt": "WS scheme test"}
        })
        job_id = sub.json()["job_id"]
        res = client.get(f"/v1/jobs/{job_id}/ws", headers=auth_headers)
        ws_url = res.json()["websocket_url"]
        assert ws_url.startswith("wss://")

    def test_f06_job_ws_url_contains_job_id(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "music", "model": "ace-step-1.5", "task": "text_to_music",
            "inputs": {"prompt": "WS job id test"}
        })
        job_id = sub.json()["job_id"]
        res = client.get(f"/v1/jobs/{job_id}/ws", headers=auth_headers)
        assert job_id in res.json()["websocket_url"]

    def test_f06_job_ws_host_matches_static_domain(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image", "model": "qwen-image-2.1", "task": "text_to_image",
            "inputs": {"prompt": "WS host check"}
        })
        job_id = sub.json()["job_id"]
        res = client.get(f"/v1/jobs/{job_id}/ws", headers=auth_headers)
        assert "jobs.ai.example.com" in res.json()["websocket_url"]

    def test_f06_job_ws_path_structure(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "enhancement", "model": "enhancement-suite-v1", "task": "frame_interpolation",
            "inputs": {"video_url": "https://s3.ai.example.com/assets/video.mp4", "multiplier": 2}
        })
        job_id = sub.json()["job_id"]
        res = client.get(f"/v1/jobs/{job_id}/ws", headers=auth_headers)
        assert res.json()["websocket_url"].endswith(f"/v1/jobs/{job_id}/ws")
