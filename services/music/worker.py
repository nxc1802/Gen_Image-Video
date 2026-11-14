"""
services/music/worker.py
Two-Tier Queue Worker for ACE-Step 1.5 Music Generation Subsystem.
Leases jobs from Factory Control Plane API for tasks:
- text_to_music (ACE-01)
- lyrics_to_music (ACE-02)
- reference_audio / reference_conditioning (ACE-07)
- cover / remix (ACE-08, ACE-10)
- repaint (ACE-09)
- extract (ACE-11, Base-only)
- lego (ACE-12, Base-only)
- complete (ACE-13, Base-only)
- music_extend (ACE-14, Base-only)
- audio_to_audio
"""
import argparse
import asyncio
import base64
import logging
import os
import shutil
import signal
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

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
from factory.apps.control_api.src.assets.ssrf import validate_target_url
from factory.apps.control_api.src.assets.transport import PinningAsyncHTTPTransport
from factory_contracts.errors.error_contract import (
    ErrorCode,
    ErrorDetail,
    InvalidRequestError,
    SSRFBlockedError,
    classify_worker_error,
)
from factory_contracts.jobs.job_contract import ArtifactReference, FactoryJob
from factory_contracts.jobs.worker_contract import (
    JobCompletionRequest,
    JobFailureRequest,
    ProgressUpdateRequest,
    WorkerHeartbeatRequest,
    WorkerLeaseRequest,
    WorkerLeaseResponse,
)
from services.music.acestep.adapter import ACEStepAdapter, ALL_SUPPORTED_TASKS as ACESTEP_TASKS
from services.music.acestep.client import ACEStepClient
from services.music.yue2.adapter import YuE2Adapter, ALL_SUPPORTED_TASKS as YUE2_TASKS
from services.music.yue2.client import YuE2Client

logger = logging.getLogger("services.music.worker")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [MusicWorker]: %(message)s")

SUPPORTED_TASKS = sorted(ACESTEP_TASKS | YUE2_TASKS)
SUPPORTED_MODELS = ["ace-step-1.5", "yue2"]


class MusicWorker:
    """Worker daemon executing generative music jobs (ACE-Step 1.5 & YuE2)."""

    def __init__(
        self,
        worker_id: str = "worker-music-0",
        control_plane_url: Optional[str] = None,
        api_key: Optional[str] = None,
        gpu_device: int = 1,
        lease_duration: int = 300,
        acestep_api_url: Optional[str] = None,
        yue2_api_url: Optional[str] = None,
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
        self.acestep_client = ACEStepClient(api_base_url=acestep_api_url)
        self.yue2_client = YuE2Client(api_base_url=yue2_api_url)
        self.client = self.acestep_client
        self.running = False
        self._active_job_id: Optional[str] = None
        self._active_lease_token: Optional[str] = None

    @property
    def auth_headers(self) -> dict:
        if not self.api_key:
            raise RuntimeError("FACTORY_API_KEY is required for Music worker authentication.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def lease_next_job(self, http_client: httpx.AsyncClient) -> Optional[WorkerLeaseResponse]:
        """Requests the next eligible music job from the Control Plane Queue."""
        req = WorkerLeaseRequest(
            worker_id=self.worker_id,
            subsystem="music",
            supported_models=SUPPORTED_MODELS,
            supported_tasks=SUPPORTED_TASKS,
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
            worker_id=self.worker_id,
            lease_token=self._active_lease_token,
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

    async def stage_asset(self, http_client: httpx.AsyncClient, asset_ref: str, suffix: str = ".wav") -> str:
        """
        Stages source or reference audio asset to local filesystem:
        - Decodes data: URI
        - Checks local path
        - Downloads remote URL or Control Plane asset ID
        """
        ref_str = str(asset_ref).strip()
        if ref_str.startswith("data:"):
            # Base64 data URI
            header, encoded = ref_str.split(",", 1)
            raw_bytes = base64.b64decode(encoded)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="staged_")
            tmp.write(raw_bytes)
            tmp.close()
            return tmp.name

        if ref_str.startswith("file://"):
            local_path = ref_str[len("file://"):]
            if Path(local_path).exists():
                src_p = Path(local_path)
                sfx = src_p.suffix or suffix
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=sfx, prefix="staged_local_")
                tmp.close()
                shutil.copyfile(local_path, tmp.name)
                return tmp.name
            raise InvalidRequestError(f"Local file URI target does not exist: {local_path}")

        if Path(ref_str).is_absolute() and Path(ref_str).exists():
            src_p = Path(ref_str)
            sfx = src_p.suffix or suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=sfx, prefix="staged_local_")
            tmp.close()
            shutil.copyfile(ref_str, tmp.name)
            return tmp.name

        # Remote URL or Storage Asset ID
        if ref_str.startswith(("http://", "https://")):
            download_url = ref_str
        else:
            # Resolve through Control Plane storage
            dl_resp = await http_client.post(
                f"{self.control_plane_url}/v1/storage/download-url",
                headers=self.auth_headers,
                json={"asset_id": ref_str, "storage_path": ref_str},
                timeout=10.0,
            )
            dl_resp.raise_for_status()
            download_url = dl_resp.json()["download_url"]

        # SSRF validation & IP pinning pre-flight check
        try:
            _, verified_ips = validate_target_url(download_url)
        except SSRFBlockedError as exc:
            raise InvalidRequestError(f"Asset URL is blocked by SSRF policy: {exc.message}") from exc
        except Exception as exc:
            raise InvalidRequestError(f"Invalid asset URL: {exc}") from exc

        transport = PinningAsyncHTTPTransport([str(ip) for ip in verified_ips])
        async with httpx.AsyncClient(transport=transport, timeout=60.0) as fetch_client:
            resp = await fetch_client.get(download_url)
            resp.raise_for_status()
            audio_bytes = resp.content

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="staged_")
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name

    async def execute_job(self, http_client: httpx.AsyncClient, job: FactoryJob) -> bool:
        """Executes a single leased music job to completion."""
        self._active_job_id = job.job_id
        logger.info(f"Executing job {job.job_id} task={job.task} model={job.model}...")

        async def on_progress(step: int, pct: float, stage: str):
            await self.report_progress(http_client, job.job_id, step, pct, stage)

        current_stage = "initializing"
        staged_files: List[str] = []
        heartbeat_task: Optional[asyncio.Task] = None

        async def heartbeat_loop():
            while self._active_job_id == job.job_id:
                await asyncio.sleep(5.0)
                await self.send_heartbeat(http_client)

        try:
            current_stage = "input_mapping"
            inputs = dict(job.inputs or {})
            is_yue2 = (job.model or "").strip().lower() in ("yue2", "yue-2", "yue-music")

            # Validate variant capability early before asset staging
            if is_yue2:
                YuE2Adapter.validate_variant_capability(job.task, inputs.get("variant"))
            else:
                ACEStepAdapter.validate_variant_capability(job.task, inputs.get("variant"))

            current_stage = "asset_staging"

            # Stage source audio if provided
            src_ref = (
                inputs.get("source_audio_asset_id")
                or inputs.get("src_audio_path")
                or inputs.get("source_audio_path")
                or inputs.get("audio_url")
            )
            if src_ref and not (Path(str(src_ref)).is_absolute() and Path(str(src_ref)).exists()):
                staged_src = await self.stage_asset(http_client, str(src_ref))
                staged_files.append(staged_src)
                inputs["src_audio_path"] = staged_src
                inputs["source_audio_path"] = staged_src

            # Stage reference audio if provided
            ref_ref = (
                inputs.get("reference_audio_asset_id")
                or inputs.get("reference_audio_path")
                or inputs.get("reference_audio_url")
            )
            if ref_ref and not (Path(str(ref_ref)).is_absolute() and Path(str(ref_ref)).exists()):
                staged_ref = await self.stage_asset(http_client, str(ref_ref))
                staged_files.append(staged_ref)
                inputs["reference_audio_path"] = staged_ref

            current_stage = "input_mapping"
            if is_yue2:
                params = YuE2Adapter.map_inputs(job.task, inputs)
            else:
                params = ACEStepAdapter.map_inputs(job.task, inputs)

            # Start heartbeat during generation
            heartbeat_task = asyncio.create_task(heartbeat_loop())

            current_stage = "generating_music"
            if is_yue2:
                if job.task == "music.symbolic_score" and inputs.get("plan_only", False):
                    result = await self.yue2_client.plan_score(params)
                elif job.task == "music.score_edit" and not inputs.get("render_audio", True):
                    result = await self.yue2_client.edit_score(params["base_score"], params["modifications"])
                else:
                    result = await self.yue2_client.generate_song(
                        task=job.task,
                        params=params,
                        progress_callback=on_progress,
                    )
            else:
                result = await self.client.generate_music(
                    task=job.task,
                    params=params,
                    progress_callback=on_progress,
                )

            current_stage = "uploading_artifacts"
            artifacts_to_upload = result.get("artifacts") or [result]
            persisted_artifacts: List[ArtifactReference] = []

            for art_item in artifacts_to_upload:
                content_bytes = art_item.get("audio_bytes") or art_item.get("bytes")
                filename = art_item["filename"]
                mime_type = art_item["mime_type"]
                sha256_hash = art_item["sha256"]
                size_bytes = art_item["size_bytes"]
                art_type = art_item.get("type") or (
                    "score" if mime_type == "text/vnd.abc" or filename.endswith(".abc") else "audio"
                )

                presign_resp = await http_client.post(
                    f"{self.control_plane_url}/v1/storage/upload-url",
                    headers=self.auth_headers,
                    json={
                        "subsystem": "music",
                        "job_id": job.job_id,
                        "artifact_name": filename,
                        "mime_type": mime_type,
                        "max_size_bytes": size_bytes,
                    },
                    timeout=10.0,
                )
                presign_resp.raise_for_status()
                presign = presign_resp.json()

                async with httpx.AsyncClient(timeout=300.0, follow_redirects=False) as upload_client:
                    put_resp = await upload_client.put(
                        presign["upload_url"],
                        content=content_bytes,
                        headers=presign.get("headers") or {"Content-Type": mime_type},
                    )
                    put_resp.raise_for_status()

                download_resp = await http_client.post(
                    f"{self.control_plane_url}/v1/storage/download-url",
                    headers=self.auth_headers,
                    json={"storage_path": presign["storage_path"], "filename": filename},
                    timeout=10.0,
                )
                download_resp.raise_for_status()
                download_url = download_resp.json()["download_url"]

                artifact = ArtifactReference(
                    artifact_id=f"art-{uuid.uuid4()}",
                    type=art_type,
                    mime_type=mime_type,
                    size_bytes=size_bytes,
                    storage_path=presign["storage_path"],
                    url=download_url,
                    download_url=download_url,
                    sha256=sha256_hash,
                )
                persisted_artifacts.append(artifact)

            current_stage = "completing_job"
            comp_req = JobCompletionRequest(
                artifacts=persisted_artifacts,
                worker_id=self.worker_id,
                lease_token=self._active_lease_token,
            )

            res = await http_client.post(
                f"{self.control_plane_url}/v1/jobs/{job.job_id}/complete",
                headers=self.auth_headers,
                json=comp_req.model_dump(mode="json"),
                timeout=10.0,
            )
            if res.status_code in (200, 201):
                logger.info(f"Job {job.job_id} completed successfully with {len(persisted_artifacts)} artifacts.")
                return True
            logger.error(f"Complete request failed with status {res.status_code}: {res.text}")
            return False

        except Exception as e:
            code, msg, retryable = classify_worker_error(e, subsystem="music", stage=current_stage)
            logger.error(
                f"Music worker job failure: job_id={job.job_id} request_id={job.request_id} "
                f"task={job.task} stage={current_stage} error_code={code.value} retryable={retryable}: {e}"
            )
            fail_req = JobFailureRequest(
                error=ErrorDetail(
                    code=code,
                    message=msg,
                    request_id=job.request_id,
                    details={"stage": current_stage},
                ),
                retryable=retryable,
                worker_id=self.worker_id,
                lease_token=self._active_lease_token,
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
            if heartbeat_task and not heartbeat_task.done():
                heartbeat_task.cancel()
            self._active_job_id = None
            self._active_lease_token = None
            # Clean up staged temporary files
            for st_file in staged_files:
                try:
                    p = Path(st_file)
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

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
    parser = argparse.ArgumentParser(description="Factory Music Worker (ACE-Step & YuE2)")
    parser.add_argument("--worker-id", default="worker-music-0", help="Worker identifier")
    parser.add_argument("--control-api", default=None, help="Control Plane URL")
    parser.add_argument("--acestep-url", default=None, help="Native ACE-Step API URL")
    parser.add_argument("--yue2-url", default=None, help="Native YuE2 API URL")
    args = parser.parse_args()

    worker = MusicWorker(
        worker_id=args.worker_id,
        control_plane_url=args.control_api,
        acestep_api_url=args.acestep_url,
        yue2_api_url=args.yue2_url,
    )

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
