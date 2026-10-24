"""
Tier 3 Cross-Feature Interaction Tests: Job Submission -> SSE Progress -> S3 Retrieval
Validates the complete asynchronous lifecycle:
1. Job submitted with idempotency key
2. Client streams SSE events to track execution steps
3. Completion event returns S3 download artifact
4. S3 download link verified
"""

import pytest
import httpx


@pytest.mark.tier3
@pytest.mark.streaming
class TestJobSSEStoragePipeline:
    def test_pipeline_job_submission_sse_progress_and_completion(self, client: httpx.Client, auth_headers):
        # Step 1: Submit job
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Sunset over a tranquil lake with mountains"},
            "idempotency_key": "sse-pipeline-test-01"
        })
        assert sub.status_code == 201
        job_id = sub.json()["job_id"]

        # Step 2: Connect to SSE stream
        sse_res = client.get(f"/v1/jobs/{job_id}/events", headers=auth_headers)
        assert sse_res.status_code == 200
        assert "text/event-stream" in sse_res.headers.get("content-type", "")
        text = sse_res.text

        # Step 3: Verify progress events and completion event
        assert "event: progress" in text
        assert "event: completed" in text
        assert "SUCCEEDED" in text

        # Step 4: Verify artifact in final job status matches S3 URL
        status_res = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert status_res.status_code == 200
        artifacts = status_res.json()["artifacts"]
        assert len(artifacts) >= 1
        s3_url = artifacts[0]["url"]
        assert "s3.ai.example.com" in s3_url

    def test_pipeline_job_reconnection_event_stream(self, client: httpx.Client, auth_headers):
        # Submit job
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "video",
            "model": "wan-2.2",
            "task": "text_to_video",
            "inputs": {"prompt": "Flying through clouds"}
        })
        job_id = sub.json()["job_id"]

        # First connection (simulating initial connect)
        stream1 = client.get(f"/v1/jobs/{job_id}/events", headers=auth_headers)
        assert stream1.status_code == 200

        # Second connection (simulating client network drop and reconnect with Last-Event-ID)
        headers = {**auth_headers, "Last-Event-ID": "5"}
        stream2 = client.get(f"/v1/jobs/{job_id}/events", headers=headers)
        assert stream2.status_code == 200
        assert "event: completed" in stream2.text

    def test_pipeline_job_cancellation_during_stream(self, client: httpx.Client, auth_headers):
        # Submit long running job
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "video",
            "model": "wan-2.2",
            "task": "text_to_video",
            "inputs": {"prompt": "Long video job to be cancelled", "num_frames": 81}
        })
        job_id = sub.json()["job_id"]

        # Cancel job
        cancel_res = client.delete(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert cancel_res.status_code == 200
        assert cancel_res.json()["state"] == "CANCELLED"

        # Verify job status
        get_res = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert get_res.json()["state"] == "CANCELLED"

    def test_pipeline_job_retry_and_re_streaming(self, client: httpx.Client, auth_headers):
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "music",
            "model": "ace-step-1.5",
            "task": "text_to_music",
            "inputs": {"prompt": "Music retry and re-stream test"}
        })
        job_id = sub.json()["job_id"]
        client.delete(f"/v1/jobs/{job_id}", headers=auth_headers)

        # Retry job
        retry_res = client.post(f"/v1/jobs/{job_id}/retry", headers=auth_headers)
        assert retry_res.status_code == 200
        assert retry_res.json()["state"] == "QUEUED"

        # Re-stream events
        sse_res = client.get(f"/v1/jobs/{job_id}/events", headers=auth_headers)
        assert sse_res.status_code == 200
        assert "event: completed" in sse_res.text
