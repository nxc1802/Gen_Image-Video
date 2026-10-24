"""
services/image/worker.py
Continuous worker leasing loop integrating with Factory Control API.
Leases jobs, stages assets, builds ComfyUI prompts, streams WebSocket progress,
monitors heartbeats/cancellations, and uploads S3 artifacts.
"""
import asyncio
import logging
import os
import re
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional
import httpx

from factory_contracts.errors.error_contract import ErrorDetail
from factory.apps.control_api.src.assets.ssrf import validate_target_url
from factory.apps.control_api.src.assets.transport import PinningAsyncHTTPTransport
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
from services.image.artifact_collector import ImageArtifactCollector
from services.image.client import ComfyUIClient
from services.image.config import ImageServiceConfig, load_image_config
from services.image.prompt_builder import ImagePromptBuilder

logger = logging.getLogger("services.image.worker")

class ImageWorkerDaemon:
    """Worker daemon leasing and executing image generation jobs."""

    def __init__(
        self,
        config: Optional[ImageServiceConfig] = None,
        client: Optional[ComfyUIClient] = None,
        prompt_builder: Optional[ImagePromptBuilder] = None,
        artifact_collector: Optional[ImageArtifactCollector] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.config = config or load_image_config()
        self.comfy_client = client or ComfyUIClient(host=self.config.comfy_host, port=self.config.port)
        self.prompt_builder = prompt_builder or ImagePromptBuilder()
        self.artifact_collector = artifact_collector or ImageArtifactCollector(self.config)
        self._http_client = http_client
        self._running = False
        self._stop_event = asyncio.Event()

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is not None:
            return self._http_client
        if not self.config.api_key:
            raise RuntimeError("FACTORY_API_KEY is required for worker-to-Control-Plane authentication.")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        return httpx.AsyncClient(base_url=self.config.control_api_url, headers=headers, timeout=30.0)

    async def stage_asset(self, url: str, target_filename: str) -> str:
        """Stage only trusted Factory-owned assets with DNS-rebinding protection."""
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        allowed_hosts = {
            host.strip().lower()
            for host in os.environ.get("FACTORY_ASSET_ALLOWED_HOSTS", "").split(",")
            if host.strip()
        }
        configured_s3_host = (urlparse(self.config.s3_public_url).hostname or "").lower()
        if configured_s3_host:
            allowed_hosts.add(configured_s3_host)
        if not hostname or hostname not in allowed_hosts:
            raise ValueError(
                f"Asset host '{hostname or '<missing>'}' is not a trusted Factory asset host."
            )

        _, verified_ips = validate_target_url(url)
        transport = PinningAsyncHTTPTransport([str(ip) for ip in verified_ips])
        max_bytes = int(os.environ.get("ASSET_MAX_UPLOAD_MB", "50")) * 1024 * 1024

        self.config.input_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(target_filename).name
        target_path = (self.config.input_dir / safe_name).resolve()
        input_root = self.config.input_dir.resolve()
        target_path.relative_to(input_root)

        async with httpx.AsyncClient(
            timeout=60.0,
            transport=transport,
            follow_redirects=False,
        ) as cl:
            async with cl.stream("GET", url) as resp:
                resp.raise_for_status()
                content_type = (resp.headers.get("content-type") or "").split(";", 1)[0].lower()
                if content_type and not (
                    content_type.startswith("image/")
                    or content_type.startswith("video/")
                    or content_type.startswith("audio/")
                    or content_type == "application/octet-stream"
                ):
                    raise ValueError(f"Unsupported staged asset MIME type '{content_type}'.")
                content_length = resp.headers.get("content-length")
                if content_length and int(content_length) > max_bytes:
                    raise ValueError("Asset exceeds configured maximum size.")

                chunks = bytearray()
                async for chunk in resp.aiter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > max_bytes:
                        raise ValueError("Asset exceeds configured maximum size.")
                if not chunks:
                    raise ValueError("Asset download returned an empty body.")
                target_path.write_bytes(bytes(chunks))
        return safe_name

    async def lease_job(self, client: httpx.AsyncClient) -> WorkerLeaseResponse:
        """Calls POST /v1/jobs/lease on the Factory Control API."""
        lease_req = WorkerLeaseRequest(
            worker_id=self.config.worker_id,
            subsystem="image",
            supported_models=["qwen-image-2.1"],
            supported_tasks=[
                "text_to_image",
                "image_to_image",
                "inpaint",
                "outpaint",
                "instruction_edit",
                "reference_conditioning",
                "controlnet",
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
            logger.warning(f"Failed to report progress for job {job_id}: {e}")

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
    ) -> None:
        comp = JobCompletionRequest(
            artifacts=artifacts,
            inference_duration_ms=duration_ms,
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
            logger.error(f"Failed to report failure for job {job_id}: {e}")

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
                        logger.info(f"Cancellation requested for job {job_id}")
                        cancel_event.set()
                        await self.comfy_client.interrupt()
                        break
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"Heartbeat failed for job {job_id}: {e}")

        hb_task = asyncio.create_task(heartbeat_loop())

        try:
            # 1. Report loading stage
            await self.report_progress(client, job_id, stage="loading", step=0, total=20, percent=0.0, message="Staging assets")

            # 2. Stage remote assets
            staged_files: Dict[str, str] = {}
            for key, val in job.inputs.items():
                if isinstance(val, str) and (val.startswith("http://") or val.startswith("https://")):
                    fname = f"{job_id}_{key}_{Path(val).name}"
                    staged_fname = await self.stage_asset(val, fname)
                    staged_files[key] = staged_fname
                elif isinstance(val, list) and key == "reference_image_urls":
                    for idx, ref_url in enumerate(val):
                        if isinstance(ref_url, str) and (ref_url.startswith("http://") or ref_url.startswith("https://")):
                            fname = f"{job_id}_ref_{idx}_{Path(ref_url).name}"
                            staged_fname = await self.stage_asset(ref_url, fname)
                            staged_files[ref_url] = staged_fname

            # 3. Build ComfyUI Prompt Graph
            prompt_dict = self.prompt_builder.build_prompt(
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

            # 5. Execute only against a live ComfyUI backend.
            if not await self.comfy_client.is_ready():
                raise RuntimeError("ComfyUI image backend is unavailable; simulation fallback is disabled.")

            prompt_id = await self.comfy_client.queue_prompt(prompt_dict, client_id=job_id)
            outputs = await self.comfy_client.listen_execution(
                prompt_id=prompt_id,
                client_id=job_id,
                on_progress=on_progress,
                cancel_event=cancel_event,
            )
            if not outputs:
                raise RuntimeError("ComfyUI image execution completed without output metadata.")

            # 6. Resolve the exact files reported by ComfyUI and upload them to Factory Storage.
            artifacts = await self.artifact_collector.collect_and_upload(client, job_id, outputs)
            duration_ms = (time.time() - start_time) * 1000.0

            # 7. Complete job
            await self.complete_job(client, job_id, artifacts, duration_ms)
            logger.info(f"Job {job_id} completed successfully in {duration_ms:.1f}ms")

        except asyncio.CancelledError:
            logger.info(f"Job {job_id} cancelled.")
            await self.fail_job(client, job_id, "JOB_CANCELLED", "Job was cancelled by user.", request_id=job.request_id, retryable=False)
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            # Sanitize error detail strictly: no leaked paths or traces
            await self.fail_job(
                client,
                job_id,
                "SERVICE_UNAVAILABLE",
                "Image generation backend failed. Inspect correlated worker logs.",
                request_id=job.request_id,
                retryable=True,
            )

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
                logger.info(f"Claimed job {lease_resp.job.job_id} for task {lease_resp.job.task}")
                await self.process_job(client, lease_resp.job, lease_resp.lease_token)
                return True
            return False
        finally:
            if self._http_client is None:
                await client.aclose()

    async def run(self) -> None:
        """Runs the continuous worker lease loop until stopped."""
        self._running = True
        logger.info(f"Starting Image Worker loop for {self.config.worker_id}...")
        while self._running and not self._stop_event.is_set():
            try:
                processed = await self.lease_and_process_one()
                if not processed:
                    await asyncio.sleep(self.config.poll_interval_seconds)
            except Exception as e:
                logger.warning(f"Error in image worker lease loop: {e}")
                await asyncio.sleep(self.config.poll_interval_seconds)

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
