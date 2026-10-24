"""
services/music/worker.py
Two-Tier Queue Worker for ACE-Step 1.5 Music Generation Subsystem.
Leases jobs from Factory Control Plane API for tasks:
- text_to_music (F33)
- lyrics_to_music (F34)
- audio_to_audio (F35)
"""
import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Bootstrap workspace package search paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for _subpath in [
    _PROJECT_ROOT,
    _PROJECT_ROOT / "packages" / "contracts" / "src",
    _PROJECT_ROOT / "packages" / "auth" / "src",
    _PROJECT_ROOT / "packages" / "model-registry" / "src",
]:
    if str(_subpath) not in sys.path:
        sys.path.insert(0, str(_subpath))

import httpx
from factory_contracts.errors.error_contract import ErrorCode, ErrorDetail
from factory_contracts.jobs.job_contract import FactoryJob
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    JobFailureRequest,
    ProgressUpdateRequest,
    WorkerHeartbeatRequest,
    WorkerLeaseRequest,
    WorkerLeaseResponse,
)
from services.music.acestep.adapter import ACEStepAdapter
from services.music.acestep.client import ACEStepClient

logger = logging.getLogger("services.music.worker")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [MusicWorker]: %(message)s")


class MusicWorker:
    """Worker daemon executing ACE-Step 1.5 generative music jobs."""

    def __init__(
        self,
        worker_id: str = "worker-music-0",
        control_plane_url: Optional[str] = None,
        api_key: Optional[str] = None,
        gpu_device: int = 1,
        lease_duration: int = 300,
    ):
        self.worker_id = worker_id
        self.control_plane_url = (
            control_plane_url
            or os.environ.get("FACTORY_BASE_URL")
            or "https://jobs.ai.example.com"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("FACTORY_API_KEY", "sk-factory-test-token-valid")
        self.gpu_device = gpu_device
        self.lease_duration = lease_duration
        self.client = ACEStepClient()
        self.running = False
        self._active_job_id: Optional[str] = None
        self._active_lease_token: Optional[str] = None

    @property
    def auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def lease_next_job(self, http_client: httpx.AsyncClient) -> Optional[WorkerLeaseResponse]:
        """Requests the next eligible music job from the Control Plane Queue."""
        req = WorkerLeaseRequest(
            worker_id=self.worker_id,
            subsystem="music",
            supported_models=["ace-step-1.5"],
            supported_tasks=["text_to_music", "lyrics_to_music", "audio_to_audio"],
            gpu_device=self.gpu_device,
            lease_duration_seconds=self.lease_duration,
        )
        try:
            res = await http_client.post(
                f"{self.control_plane_url}/v1/jobs/lease",
                headers=self.auth_headers,
                json=req.model_dump(),
                timeout=10.0,
            )
            if res.status_code == 200:
                data = res.json()
                return WorkerLeaseResponse.model_validate(data)
            logger.debug(f"Lease request returned status {res.status_code}")
        except Exception as e:
            logger.warning(f"Lease request failed: {e}")
        return None

    async def send_heartbeat(self, http_client: httpx.AsyncClient) -> None:
        """Sends periodic heartbeat to maintain job lease."""
        if not self._active_job_id:
            return
        hb = WorkerHeartbeatRequest(
            worker_id=self.worker_id,
            active_job_id=self._active_job_id,
            lease_token=self._active_lease_token,
            gpu_utilization_percent=60.0,
            vram_used_mb=4200.0,
        )
        try:
            await http_client.post(
                f"{self.control_plane_url}/v1/workers/{self.worker_id}/heartbeat",
                headers=self.auth_headers,
                json=hb.model_dump(),
                timeout=5.0,
            )
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")

    async def report_progress(
        self,
        http_client: httpx.AsyncClient,
        job_id: str,
        step: int,
        percent: float,
        stage: str,
    ) -> None:
        """Reports generation progress to Control Plane."""
        prog = ProgressUpdateRequest(
            stage=stage,
            step=step,
            total=20,
            percent=percent,
            message=f"ACE-Step {stage} at {percent:.0f}%",
        )
        try:
            await http_client.post(
                f"{self.control_plane_url}/v1/jobs/{job_id}/progress",
                headers=self.auth_headers,
                json=prog.model_dump(),
                timeout=5.0,
            )
        except Exception as e:
            logger.debug(f"Progress update failed: {e}")

    async def execute_job(self, http_client: httpx.AsyncClient, job: FactoryJob) -> bool:
        """Executes a single leased music job to completion."""
        self._active_job_id = job.job_id
        logger.info(f"Executing job {job.job_id} task={job.task} model={job.model}...")

        async def on_progress(step: int, pct: float, stage: str):
            await self.report_progress(http_client, job.job_id, step, pct, stage)

        try:
            params = ACEStepAdapter.map_inputs(job.task, job.inputs)
            result = await self.client.generate_music(
                task=job.task,
                params=params,
                progress_callback=on_progress,
            )

            artifact = ACEStepAdapter.create_artifact_reference(result)
            comp_req = JobCompletionRequest(
                artifacts=[artifact],
                inference_duration_ms=result.get("inference_duration_ms", 12500.0),
            )

            res = await http_client.post(
                f"{self.control_plane_url}/v1/jobs/{job.job_id}/complete",
                headers=self.auth_headers,
                json=comp_req.model_dump(mode="json"),
                timeout=10.0,
            )
            if res.status_code in (200, 201):
                logger.info(f"Job {job.job_id} completed successfully.")
                return True
            logger.error(f"Complete request failed with status {res.status_code}: {res.text}")
            return False

        except Exception as e:
            logger.error(f"Execution error on job {job.job_id}: {e}")
            fail_req = JobFailureRequest(
                error=ErrorDetail(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Music generation failed: {str(e)}",
                    request_id=job.request_id,
                ),
                retryable=True,
            )
            try:
                await http_client.post(
                    f"{self.control_plane_url}/v1/jobs/{job.job_id}/fail",
                    headers=self.auth_headers,
                    json=fail_req.model_dump(mode="json"),
                    timeout=5.0,
                )
            except Exception:
                pass
            return False
        finally:
            self._active_job_id = None
            self._active_lease_token = None

    async def run_lease_cycle(self, http_client: httpx.AsyncClient) -> bool:
        """Executes a single poll-and-lease cycle."""
        lease_resp = await self.lease_next_job(http_client)
        if lease_resp and lease_resp.job:
            self._active_lease_token = lease_resp.lease_token
            return await self.execute_job(http_client, lease_resp.job)
        return False

    async def run_forever(self) -> None:
        """Continuous polling worker loop."""
        self.running = True
        logger.info(f"Music worker {self.worker_id} started, polling {self.control_plane_url}...")
        async with httpx.AsyncClient() as client:
            while self.running:
                found = await self.run_lease_cycle(client)
                if not found:
                    await asyncio.sleep(2.0)

    def stop(self) -> None:
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="ACE-Step Music Worker")
    parser.add_argument("--worker-id", default="worker-music-0", help="Worker identifier")
    parser.add_argument("--control-api", default=None, help="Control Plane URL")
    args = parser.parse_args()

    worker = MusicWorker(worker_id=args.worker_id, control_plane_url=args.control_api)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def shutdown():
        worker.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(worker.run_forever())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
