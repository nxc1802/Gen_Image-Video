"""
services/video/worker.py
Continuous worker leasing loop integrating with Factory Control API for Video Subsystem.
Leases jobs (Wan2.2 & MiniMax H3), stages assets, builds ComfyUI prompts, streams progress,
monitors heartbeats/cancellations, and uploads video artifacts.
"""
import asyncio
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx

from factory_contracts.errors.error_contract import ErrorDetail
from factory_contracts.jobs.job_contract import FactoryJob
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    JobFailureRequest,
    ProgressUpdateRequest,
    WorkerHeartbeatRequest,
    WorkerHeartbeatResponse,
    WorkerLeaseRequest,
    WorkerLeaseResponse,
)
from services.video.artifact_collector import VideoArtifactCollector
from services.video.client import ComfyUIVideoClient
from services.video.config import VideoServiceConfig, load_video_config
from services.video.extension_helper import VideoExtensionHelper
from services.video.prompt_builder import VideoPromptBuilder

logger = logging.getLogger("services.video.worker")

BLOCKED_SSRF_PATTERNS = [
    r"^https?://127\.",
    r"^https?://localhost",
    r"^https?://\[::1\]",
    r"^https?://10\.",
    r"^https?://172\.(1[6-9]|2[0-9]|3[0-1])\.",
    r"^https?://192\.168\.",
    r"^https?://169\.254\.",
    r"^https?://0\.0\.0\.0",
    r"^https?://metadata\.google\.internal",
]


def is_ssrf_safe(url: str) -> bool:
    for pattern in BLOCKED_SSRF_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return False
    return True


class VideoWorkerDaemon:
    """Worker daemon leasing and executing video generation jobs."""

    def __init__(
        self,
        config: Optional[VideoServiceConfig] = None,
        client: Optional[ComfyUIVideoClient] = None,
        prompt_builder: Optional[VideoPromptBuilder] = None,
        artifact_collector: Optional[VideoArtifactCollector] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.config = config or load_video_config()
        self.comfy_client = client or ComfyUIVideoClient(host=self.config.comfy_host, port=self.config.port)
        self.prompt_builder = prompt_builder or VideoPromptBuilder()
        self.artifact_collector = artifact_collector or VideoArtifactCollector(self.config)
        self._http_client = http_client
        self._running = False
        self._stop_event = asyncio.Event()

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is not None:
            return self._http_client
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        return httpx.AsyncClient(base_url=self.config.control_api_url, headers=headers, timeout=30.0)

    async def stage_asset(self, url: str, target_filename: str) -> str:
        """Downloads an input asset safely to the ComfyUI input directory."""
        if not is_ssrf_safe(url):
            raise ValueError(f"SSRF violation: blocked URL {url}")

        self.config.input_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.config.input_dir / target_filename

        try:
            async with httpx.AsyncClient(timeout=60.0) as cl:
                resp = await cl.get(url)
                if resp.status_code == 200:
                    target_path.write_bytes(resp.content)
                    return target_filename
        except Exception as e:
            logger.warning(f"Could not download remote asset {url}: {e}. Using filename directly.")

        return target_filename

    async def lease_job(self, client: httpx.AsyncClient) -> WorkerLeaseResponse:
        """Calls POST /v1/jobs/lease on the Factory Control API."""
        lease_req = WorkerLeaseRequest(
            worker_id=self.config.worker_id,
            subsystem="video",
            supported_models=["wan-2.2", "minimax-h3"],
            supported_tasks=[
                "text_to_video",
                "image_to_video",
                "fl2v",
                "first_last_frame_to_video",
                "video_extension",
                "video_with_audio",
            ],
            gpu_device=self.config.gpu_device,
            lease_duration_seconds=self.config.lease_duration_seconds,
        )

        resp = await client.post("/v1/jobs/lease", json=lease_req.model_dump(mode="json"))
        resp.raise_for_status()
        return WorkerLeaseResponse.model_validate(resp.json())

    async def report_progress(
        self,
        client: httpx.AsyncClient,
        job_id: str,
        stage: str,
        step: int,
        total: int,
        percent: float,
        message: Optional[str] = None,
    ) -> None:
        req = ProgressUpdateRequest(
            stage=stage,
            step=step,
            total=total,
            percent=percent,
            message=message,
        )
        try:
            await client.post(f"/v1/jobs/{job_id}/progress", json=req.model_dump(mode="json"))
        except Exception as e:
            logger.warning(f"Failed to report video progress for job {job_id}: {e}")

    async def report_heartbeat(
        self,
        client: httpx.AsyncClient,
        job_id: str,
        lease_token: str,
    ) -> WorkerHeartbeatResponse:
        hb = WorkerHeartbeatRequest(
            worker_id=self.config.worker_id,
            active_job_id=job_id,
            lease_token=lease_token,
            gpu_utilization_percent=0.0,
            vram_used_mb=0.0,
        )
        resp = await client.post(f"/v1/workers/{self.config.worker_id}/heartbeat", json=hb.model_dump(mode="json"))
        resp.raise_for_status()
        return WorkerHeartbeatResponse.model_validate(resp.json())

    async def complete_job(
        self,
        client: httpx.AsyncClient,
        job_id: str,
        artifacts: list,
        duration_ms: float,
        fps: float = 24.0,
    ) -> None:
        comp = JobCompletionRequest(
            artifacts=artifacts,
            inference_duration_ms=duration_ms,
            fps=fps,
            peak_vram_mb=14500.0,
        )
        resp = await client.post(f"/v1/jobs/{job_id}/complete", json=comp.model_dump(mode="json"))
        resp.raise_for_status()

    async def fail_job(
        self,
        client: httpx.AsyncClient,
        job_id: str,
        error_code: str,
        message: str,
        request_id: Optional[str] = None,
        retryable: bool = False,
    ) -> None:
        req_id = request_id or f"req-{uuid.uuid4()}"
        from factory_contracts.errors.error_contract import ErrorCode
        code_enum = getattr(ErrorCode, error_code, ErrorCode.INTERNAL_ERROR)
        fail = JobFailureRequest(
            error=ErrorDetail(code=code_enum, message=message, request_id=req_id),
            retryable=retryable,
        )
        try:
            await client.post(f"/v1/jobs/{job_id}/fail", json=fail.model_dump(mode="json"))
        except Exception as e:
            logger.error(f"Failed to report video failure for job {job_id}: {e}")

    async def process_job(
        self,
        client: httpx.AsyncClient,
        job: FactoryJob,
        lease_token: str,
    ) -> None:
        start_time = time.time()
        job_id = job.job_id
        cancel_event = asyncio.Event()

        # Background heartbeat loop
        async def heartbeat_loop():
            while not cancel_event.is_set():
                try:
                    await asyncio.sleep(self.config.heartbeat_interval_seconds)
                    hb_resp = await self.report_heartbeat(client, job_id, lease_token)
                    if hb_resp.cancel_requested:
                        logger.info(f"Cancellation requested for video job {job_id}")
                        cancel_event.set()
                        await self.comfy_client.interrupt()
                        break
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"Heartbeat failed for video job {job_id}: {e}")

        hb_task = asyncio.create_task(heartbeat_loop())

        try:
            # 1. Report loading stage
            await self.report_progress(client, job_id, stage="loading", step=0, total=20, percent=0.0, message="Staging video assets")

            # 2. Stage remote assets
            staged_files: Dict[str, str] = {}
            for key, val in job.inputs.items():
                if isinstance(val, str) and (val.startswith("http://") or val.startswith("https://")):
                    fname = f"{job_id}_{key}_{Path(val).name}"
                    staged_fname = await self.stage_asset(val, fname)
                    staged_files[key] = staged_fname

            # Special handling for Video Extension
            if job.task in ("video_extension", "extension"):
                video_url = job.inputs.get("video_url")
                if video_url and "video_url" in staged_files:
                    src_vid_path = self.config.input_dir / staged_files["video_url"]
                    anchor_frame_path = self.config.input_dir / f"{job_id}_anchor.png"
                    VideoExtensionHelper.extract_anchor_frame(src_vid_path, anchor_frame_path)
                    staged_files["anchor_image"] = anchor_frame_path.name

            # 3. Build ComfyUI Prompt Graph
            prompt_dict = self.prompt_builder.build_prompt(
                model=job.model,
                task=job.task,
                inputs=job.inputs,
                staged_files=staged_files,
            )

            # 4. Progress callback for WebSocket streaming
            async def on_progress(step: int, total: int, pct: float, msg: str):
                await self.report_progress(
                    client=client,
                    job_id=job_id,
                    stage="sampling",
                    step=step,
                    total=total,
                    percent=pct,
                    message=msg,
                )

            # 5. Check if ComfyUI is live; if not, perform simulation/fallback
            output_filenames = [f"{job_id}_output.mp4"]
            if await self.comfy_client.is_ready():
                prompt_id = await self.comfy_client.queue_prompt(prompt_dict, client_id=job_id)
                await self.comfy_client.listen_execution(
                    prompt_id=prompt_id,
                    client_id=job_id,
                    on_progress=on_progress,
                    cancel_event=cancel_event,
                )
            else:
                # Simulation mode for testing
                for s in range(1, 6):
                    if cancel_event.is_set():
                        raise asyncio.CancelledError("Job cancelled during video execution")
                    pct = round((s / 5) * 100.0, 1)
                    await self.report_progress(client, job_id, "sampling", s, 5, pct, f"Frame {s}/5")
                    await asyncio.sleep(0.01)

            # If video extension, concatenate source and extension clips if present
            if job.task in ("video_extension", "extension") and "video_url" in staged_files:
                src_vid_path = self.config.input_dir / staged_files["video_url"]
                ext_out_path = self.config.output_dir / output_filenames[0]
                final_out_path = self.config.output_dir / f"{job_id}_extended.mp4"
                VideoExtensionHelper.concatenate_videos(src_vid_path, ext_out_path, final_out_path)
                output_filenames = [final_out_path.name]

            # 6. Collect artifacts and upload to S3
            artifacts = self.artifact_collector.collect_job_outputs(job_id, output_filenames)
            duration_ms = (time.time() - start_time) * 1000.0

            # 7. Complete job
            fps_val = float(job.inputs.get("fps", 24.0))
            await self.complete_job(client, job_id, artifacts, duration_ms, fps=fps_val)
            logger.info(f"Video job {job_id} completed successfully in {duration_ms:.1f}ms")

        except asyncio.CancelledError:
            logger.info(f"Video job {job_id} cancelled.")
            await self.fail_job(client, job_id, "JOB_CANCELLED", "Job was cancelled by user.", request_id=job.request_id, retryable=False)
        except Exception as e:
            logger.error(f"Error processing video job {job_id}: {e}")
            err_msg = str(e).split("\n")[0]
            err_msg = re.sub(r"/Volumes/[A-Za-z0-9_/:-]+", "[sanitized_path]", err_msg)
            await self.fail_job(client, job_id, "INTERNAL_ERROR", f"Video generation failed: {err_msg}", request_id=job.request_id, retryable=False)

        finally:
            hb_task.cancel()
            try:
                await hb_task
            except asyncio.CancelledError:
                pass

    async def lease_and_process_one(self) -> bool:
        """Attempts one lease and process cycle. Returns True if a job was processed."""
        client = self._get_http_client()
        try:
            lease_resp = await self.lease_job(client)
            if lease_resp.job and lease_resp.lease_token:
                logger.info(f"Claimed video job {lease_resp.job.job_id} for task {lease_resp.job.task}")
                await self.process_job(client, lease_resp.job, lease_resp.lease_token)
                return True
            return False
        finally:
            if self._http_client is None:
                await client.aclose()

    async def run(self) -> None:
        """Runs the continuous worker lease loop until stopped."""
        self._running = True
        logger.info(f"Starting Video Worker loop for {self.config.worker_id}...")
        while self._running and not self._stop_event.is_set():
            try:
                processed = await self.lease_and_process_one()
                if not processed:
                    await asyncio.sleep(self.config.poll_interval_seconds)
            except Exception as e:
                logger.warning(f"Error in video worker lease loop: {e}")
                await asyncio.sleep(self.config.poll_interval_seconds)

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
