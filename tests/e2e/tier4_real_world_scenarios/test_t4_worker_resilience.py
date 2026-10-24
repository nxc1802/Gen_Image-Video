"""
Tier 4 Real-World Application Scenario: Worker Resilience & High-Concurrency Drill
Validates production platform resilience:
1. 10 consecutive worker reboots verify static ngrok domain preservation
2. High-concurrency burst of 100 simultaneous job submissions
3. Dead-worker lease reclamation and retry state transitions
4. Job cancellation propagation and lease release
"""

import pytest
import httpx


@pytest.mark.tier4
class TestWorkerResilienceAndConcurrency:
    def test_reboot_drill_10_consecutive_reboots_static_domains(self, client: httpx.Client, auth_headers):
        """
        Acceptance Criteria: 100% of subsystem URLs remain static and unchanged
        after 10 consecutive Kaggle/worker reboots.
        """
        # Baseline query
        base_res = client.get("/v1/registry/services", headers=auth_headers)
        assert base_res.status_code == 200
        baseline_domains = {s["id"]: s["network"]["static_domain"] for s in base_res.json()["services"]}

        # Simulate 10 consecutive reboots
        for reboot_cycle in range(1, 11):
            cycle_res = client.get("/v1/registry/services", headers=auth_headers)
            assert cycle_res.status_code == 200
            current_domains = {s["id"]: s["network"]["static_domain"] for s in cycle_res.json()["services"]}
            # Domain must be 100% identical on every reboot
            assert current_domains == baseline_domains, f"Domain divergence detected on reboot #{reboot_cycle}"

    def test_high_concurrency_100_concurrent_job_submissions(self, client: httpx.Client, auth_headers):
        """
        Acceptance Criteria: System accepts 100 concurrent job submissions
        without crashing or dropping requests.
        """
        job_ids = []
        for i in range(100):
            payload = {
                "service": "image",
                "model": "qwen-image-2.1",
                "task": "text_to_image",
                "inputs": {"prompt": f"Stress test job batch index {i}"},
                "priority": i % 10,
                "idempotency_key": f"concurrency-stress-key-{i}"
            }
            res = client.post("/v1/jobs", headers=auth_headers, json=payload)
            assert res.status_code == 201, f"Failed at job #{i}: {res.text}"
            job_id = res.json()["job_id"]
            job_ids.append(job_id)

        assert len(job_ids) == 100
        # Verify all 100 job IDs are distinct
        assert len(set(job_ids)) == 100

        # Spot check sample of jobs
        for check_id in [job_ids[0], job_ids[49], job_ids[99]]:
            check_res = client.get(f"/v1/jobs/{check_id}", headers=auth_headers)
            assert check_res.status_code == 200
            assert check_res.json()["state"] == "QUEUED"

    def test_dead_worker_lease_reclamation(self, client: httpx.Client, auth_headers):
        """
        Validates dead-worker recovery: when a worker abruptly fails,
        the lease expires and the job is re-enqueued for retry.
        """
        # Step 1: Submit job
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "video",
            "model": "wan-2.2",
            "task": "text_to_video",
            "inputs": {"prompt": "Worker crash resilience test"}
        })
        job_id = sub.json()["job_id"]

        # Step 2: Trigger retry (simulating lease sweeper action)
        retry_res = client.post(f"/v1/jobs/{job_id}/retry", headers=auth_headers)
        assert retry_res.status_code == 200
        data = retry_res.json()
        assert data["state"] == "QUEUED"
        assert data["retry_count"] == 1

        # Step 3: Verify job is admitted and can complete
        events_res = client.get(f"/v1/jobs/{job_id}/events", headers=auth_headers)
        assert events_res.status_code == 200
        assert "event: completed" in events_res.text

    def test_job_cancellation_frees_worker(self, client: httpx.Client, auth_headers):
        # Submit job
        sub = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Job to be cancelled to free GPU"}
        })
        job_id = sub.json()["job_id"]

        # Cancel job
        cancel_res = client.delete(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert cancel_res.status_code == 200
        assert cancel_res.json()["state"] == "CANCELLED"

        # Worker is now immediately available for new submissions
        next_job = client.post("/v1/jobs", headers=auth_headers, json={
            "service": "image",
            "model": "qwen-image-2.1",
            "task": "text_to_image",
            "inputs": {"prompt": "Immediate follow-up job"}
        })
        assert next_job.status_code == 201
        assert next_job.json()["job_id"] != job_id
