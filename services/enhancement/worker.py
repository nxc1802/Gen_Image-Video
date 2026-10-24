"""
services/enhancement/worker.py
Two-Tier Queue Worker for Media Enhancement Subsystem.
Leases jobs from Factory Control Plane API for tasks:
- image_super_resolution (F36)
- video_super_resolution (F37)
- frame_interpolation (F38)
- audio_enhancement (F39)
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
from factory_contracts.jobs.job_contract import ArtifactReference, FactoryJob
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    JobFailureRequest,
    ProgressUpdateRequest,
    WorkerHeartbeatRequest,
    WorkerLeaseRequest,
    WorkerLeaseResponse,
)
from services.enhancement.processors.audio_enhance import AudioEnhancementProcessor
from services.enhancement.processors.image_sr import ImageSuperResolutionProcessor
from services.enhancement.processors.interpolation import FrameInterpolationProcessor
from services.enhancement.processors.video_sr import VideoSuperResolutionProcessor

logger = logging.getLogger("services.enhancement.worker")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [EnhanceWorker]: %(message)s")


class EnhancementWorker:
    """Worker daemon executing post-processing enhancement jobs."""

    def __init__(
        self,
        worker_id: str = "worker-enhance-0",
        control_plane_url: Optional[str] = None,
        api_key: Optional[str] = None,
        gpu_device: int = 1,
        lease_duration: int = 300,
    ):
        self.worker_id = worker_id
        self.control_plane_url = (
            control_plane_url
            or os.environ.get("CONTROL_PLANE_URL")
            or os.environ.get("FACTORY_BASE_URL")
            or "http://127.0.0.1:8000"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("FACTORY_API_KEY", "")
        self.gpu_device = gpu_device
        self.lease_duration = lease_duration

        # Processors
        self.image_sr = ImageSuperResolutionProcessor()
        self.video_sr = VideoSuperResolutionProcessor()
        self.interpolation = FrameInterpolationProcessor()
        self.audio_enhance = AudioEnhancementProcessor()

        self.running = False
        self._active_job_id: Optional[str] = None
        self._active_lease_token: Optional[str] = None

    @property
    def auth_headers(self) -> dict:
        if not self.api_key:
            raise RuntimeError("FACTORY_API_KEY is required for Enhancement worker authentication.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def lease_next_job(self, http_client: httpx.AsyncClient) -> Optional[WorkerLeaseResponse]:
        """Requests next enhancement job from the Control Plane queue."""
        req = WorkerLeaseRequest(
            worker_id=self.worker_id,
            subsystem="enhancement",
            supported_models=["enhancement-suite-v1"],
            supported_tasks=[
                "image_super_resolution",
                "video_super_resolution",
                "frame_interpolation",
                "audio_enhancement",
            ],
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
                return WorkerLeaseResponse.model_validate(res.json())
        except Exception as e:
            logger.warning(f"Enhancement lease request failed: {e}")
        return None

    async def send_heartbeat(self, http_client: httpx.AsyncClient) -> None:
        if not self._active_job_id:
            return
        hb = WorkerHeartbeatRequest(
            worker_id=self.worker_id,
            active_job_id=self._active_job_id,
            lease_token=self._active_lease_token,
        )
        try:
            await http_client.post(
                f"{self.control_plane_url}/v1/workers/{self.worker_id}/heartbeat",
                headers=self.auth_headers,
                json=hb.model_dump(),
                timeout=5.0,
            )
        except Exception:
            pass

    async def report_progress(
        self,
        http_client: httpx.AsyncClient,
        job_id: str,
        step: int,
        percent: float,
        stage: str,
    ) -> None:
        prog = ProgressUpdateRequest(
            stage=stage,
            step=step,
            total=10,
            percent=percent,
            message=f"Enhancement {stage} at {percent:.0f}%",
        )
        try:
            await http_client.post(
                f"{self.control_plane_url}/v1/jobs/{job_id}/progress",
                headers=self.auth_headers,
                json=prog.model_dump(),
                timeout=5.0,
            )
        except Exception:
            pass

    async def execute_job(self, http_client: httpx.AsyncClient, job: FactoryJob) -> bool:
        self._active_job_id = job.job_id
        t0 = time.time()
        logger.info(f"Enhancing media for job {job.job_id} task={job.task}...")

        try:
            await self.report_progress(http_client, job.job_id, 2, 20.0, "loading_media")

            if job.task == "image_super_resolution":
                res = self.image_sr.process(job.inputs)
                ext = "png"
            elif job.task == "video_super_resolution":
                res = self.video_sr.process(job.inputs)
                ext = "mp4"
            elif job.task == "frame_interpolation":
                res = self.interpolation.process(job.inputs)
                ext = "mp4"
            elif job.task == "audio_enhancement":
                res = self.audio_enhance.process(job.inputs)
                ext = "wav"
            else:
                raise ValueError(f"Unknown enhancement task: {job.task}")

            await self.report_progress(http_client, job.job_id, 8, 80.0, "encoding_output")

            artifact = ArtifactReference(
                artifact_id=res["artifact_id"],
                type=res["type"],
                mime_type=res["mime_type"],
                size_bytes=res["size_bytes"],
                storage_path=res["storage_path"],
                url=res["download_url"],
                download_url=res["download_url"],
                sha256=res["sha256"],
            )

            inference_duration_ms = round((time.time() - t0) * 1000, 2)
            comp_req = JobCompletionRequest(
                artifacts=[artifact],
                inference_duration_ms=inference_duration_ms,
            )

            res = await http_client.post(
                f"{self.control_plane_url}/v1/jobs/{job.job_id}/complete",
                headers=self.auth_headers,
                json=comp_req.model_dump(mode="json"),
                timeout=10.0,
            )
            return res.status_code in (200, 201)

        except Exception as e:
            logger.error(f"Enhancement failed on job {job.job_id}: {e}")
            fail_req = JobFailureRequest(
                error=ErrorDetail(
                    code=ErrorCode.SERVICE_UNAVAILABLE,
                    message="Enhancement backend failed. Inspect correlated worker logs.",
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
        lease_resp = await self.lease_next_job(http_client)
        if lease_resp and lease_resp.job:
            self._active_lease_token = lease_resp.lease_token
            return await self.execute_job(http_client, lease_resp.job)
        return False

    async def run_forever(self) -> None:
        self.running = True
        logger.info(f"Enhancement worker {self.worker_id} started, polling {self.control_plane_url}...")
        async with httpx.AsyncClient() as client:
            while self.running:
                found = await self.run_lease_cycle(client)
                if not found:
                    await asyncio.sleep(2.0)

    def stop(self) -> None:
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="Media Enhancement Worker")
    parser.add_argument("--worker-id", default="worker-enhance-0", help="Worker identifier")
    parser.add_argument("--control-api", default=None, help="Control Plane URL")
    args = parser.parse_args()

    worker = EnhancementWorker(worker_id=args.worker_id, control_plane_url=args.control_api)

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
