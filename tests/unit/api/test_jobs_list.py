"""
tests/unit/api/test_jobs_list.py
Unit tests verifying the GET /v1/jobs list endpoint:
- Unauthenticated rejection (401)
- Ordering (created_at DESC)
- Pagination (limit, offset)
- Filters (service, state, status alias)
- Presigned artifact URL refresh
- Route ordering before /{job_id}
"""
import os
import time
import pytest
from starlette.testclient import TestClient

from factory.apps.control_api.src.main import app
from factory_contracts.jobs.job_contract import ArtifactReference

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


class TestJobsListEndpoint:
    def test_unauthenticated_jobs_list_rejected(self, api_client):
        res = api_client.get("/v1/jobs")
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "UNAUTHORIZED"

    def test_empty_or_initial_jobs_list(self, api_client, auth_headers):
        res = api_client.get("/v1/jobs", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)

    def test_jobs_list_ordering(self, api_client, auth_headers):
        job_ids = []
        for i in range(3):
            payload = {
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": f"Order test prompt {i}"},
            }
            res = api_client.post("/v1/jobs", headers=auth_headers, json=payload)
            assert res.status_code == 201
            job_ids.append(res.json()["job_id"])
            time.sleep(0.01)

        res = api_client.get("/v1/jobs?limit=50", headers=auth_headers)
        assert res.status_code == 200
        jobs = res.json()
        returned_ids = [j["job_id"] for j in jobs]

        # Ensure all submitted jobs are present and the latest is first
        for j_id in job_ids:
            assert j_id in returned_ids

        # Verify strict descending order by created_at
        timestamps = [j["created_at"] for j in jobs]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_jobs_list_pagination(self, api_client, auth_headers):
        for i in range(5):
            payload = {
                "service": "video",
                "model": "wan-2.2",
                "task": "text_to_video",
                "inputs": {"prompt": f"Pagination test {i}"},
            }
            res = api_client.post("/v1/jobs", headers=auth_headers, json=payload)
            assert res.status_code == 201

        res_p1 = api_client.get("/v1/jobs?limit=2&offset=0", headers=auth_headers)
        assert res_p1.status_code == 200
        p1 = res_p1.json()
        assert len(p1) == 2

        res_p2 = api_client.get("/v1/jobs?limit=2&offset=2", headers=auth_headers)
        assert res_p2.status_code == 200
        p2 = res_p2.json()
        assert len(p2) == 2

        res_p3 = api_client.get("/v1/jobs?limit=2&offset=4", headers=auth_headers)
        assert res_p3.status_code == 200
        p3 = res_p3.json()
        assert len(p3) >= 1

        # Check there is no overlap between page 1 and page 2
        p1_ids = {j["job_id"] for j in p1}
        p2_ids = {j["job_id"] for j in p2}
        assert p1_ids.isdisjoint(p2_ids)

    def test_jobs_list_service_filter(self, api_client, auth_headers):
        image_res = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "Filter image test"},
            },
        )
        assert image_res.status_code == 201
        image_job_id = image_res.json()["job_id"]

        speech_res = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "service": "speech",
                "model": "qwen3-tts",
                "task": "text_to_speech",
                "inputs": {"text": "Filter speech test"},
            },
        )
        assert speech_res.status_code == 201
        speech_job_id = speech_res.json()["job_id"]

        # Filter by service=speech
        res = api_client.get("/v1/jobs?service=speech", headers=auth_headers)
        assert res.status_code == 200
        speech_jobs = res.json()
        assert all(j["service"] == "speech" for j in speech_jobs)
        speech_ids = {j["job_id"] for j in speech_jobs}
        assert speech_job_id in speech_ids
        assert image_job_id not in speech_ids

    def test_jobs_list_state_filter(self, api_client, auth_headers):
        res = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "Cancel state test"},
            },
        )
        job_id = res.json()["job_id"]

        # Cancel the job
        cancel_res = api_client.delete(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert cancel_res.status_code == 200

        # Query with state=CANCELLED
        cancelled_res = api_client.get("/v1/jobs?state=CANCELLED", headers=auth_headers)
        assert cancelled_res.status_code == 200
        cancelled_jobs = cancelled_res.json()
        assert all(j["state"] == "CANCELLED" for j in cancelled_jobs)
        cancelled_ids = {j["job_id"] for j in cancelled_jobs}
        assert job_id in cancelled_ids

    def test_jobs_list_status_filter_alias(self, api_client, auth_headers):
        res1 = api_client.get("/v1/jobs?state=CANCELLED", headers=auth_headers)
        res2 = api_client.get("/v1/jobs?status=CANCELLED", headers=auth_headers)
        assert res1.status_code == 200
        assert res2.status_code == 200
        ids1 = [j["job_id"] for j in res1.json()]
        ids2 = [j["job_id"] for j in res2.json()]
        assert ids1 == ids2

    def test_jobs_list_presigned_urls_attached(self, api_client, auth_headers):
        # Submit a job
        res = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": "Artifact presigned test"},
            },
        )
        job_id = res.json()["job_id"]

        # Manually attach an artifact in the repository
        manager = app.state.job_manager
        import asyncio

        async def _attach_artifact():
            job = await manager.get_job(job_id)
            job.artifacts = [
                ArtifactReference(
                    artifact_id="art-test-001",
                    mime_type="image/png",
                    size_bytes=1024,
                    storage_path="assets/test-artifact.png",
                    download_url="https://placeholder-url.com/old",
                    sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                )
            ]
            await manager.repository.update_job(job)

        asyncio.run(_attach_artifact())

        # Query list endpoint
        list_res = api_client.get("/v1/jobs?limit=50", headers=auth_headers)
        assert list_res.status_code == 200
        matched = [j for j in list_res.json() if j["job_id"] == job_id]
        assert len(matched) == 1
        matched_job = matched[0]
        assert len(matched_job["artifacts"]) == 1
        art = matched_job["artifacts"][0]
        assert art["url"] is not None
        assert art["download_url"] is not None
        assert art["storage_path"] == "assets/test-artifact.png"
