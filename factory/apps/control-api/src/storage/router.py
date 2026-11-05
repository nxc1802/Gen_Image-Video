"""
factory/apps/control-api/src/storage/router.py
FastAPI router for storage presigned upload generation and local storage fallback endpoints.
"""
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional
import uuid
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from factory_auth import verify_bearer_token
from factory_contracts.errors import InvalidRequestError, UnauthorizedError
from factory.apps.control_api.src.storage.config import StorageConfig, load_storage_config
from factory.apps.control_api.src.storage.keys import format_artifact_key
from factory.apps.control_api.src.storage.local import LocalStorageProvider
from factory.apps.control_api.src.storage.provider import PresignedUpload, StorageProvider
from factory.apps.control_api.src.storage.scratch_disk import ScratchDiskManager, get_scratch_disk

logger = logging.getLogger("factory.storage.router")


class UploadURLRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subsystem: str = Field(default="artifacts")
    job_id: str = Field(..., description="Job identifier")
    artifact_name: str = Field(..., description="File name of artifact")
    mime_type: str = Field(default="application/octet-stream")
    max_size_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    expires_in_seconds: int = Field(default=3600, ge=60, le=86400)


class DownloadURLRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    storage_path: str = Field(..., min_length=1)
    filename: Optional[str] = None
    expires_in_seconds: int = Field(default=86400, ge=60, le=604800)


router = APIRouter(prefix="/v1/storage", tags=["Storage"])


def get_storage_provider(request: Request) -> StorageProvider:
    provider = getattr(request.app.state, "storage_provider", None)
    if provider is not None:
        return provider
    config = load_storage_config()
    if config.storage_backend == "s3":
        from factory.apps.control_api.src.storage.s3 import S3StorageProvider
        return S3StorageProvider(config)
    return LocalStorageProvider(config)


@router.post("/upload-url", response_model=PresignedUpload, dependencies=[Depends(verify_bearer_token)])
async def create_upload_url(
    req: UploadURLRequest,
    storage: StorageProvider = Depends(get_storage_provider),
):
    key = format_artifact_key(req.subsystem, req.job_id, req.artifact_name)
    config = load_storage_config()
    presigned = storage.generate_presigned_upload_url(
        bucket=config.s3_bucket_name,
        key=key,
        mime_type=req.mime_type,
        max_size_bytes=req.max_size_bytes,
        expires_in_seconds=req.expires_in_seconds,
    )
    return presigned




@router.post("/download-url", dependencies=[Depends(verify_bearer_token)])
async def create_download_url(
    req: DownloadURLRequest,
    storage: StorageProvider = Depends(get_storage_provider),
):
    config = load_storage_config()
    url = storage.generate_presigned_download_url(
        bucket=config.s3_bucket_name,
        key=req.storage_path,
        expires_in_seconds=req.expires_in_seconds,
        filename=req.filename,
    )
    return {
        "storage_path": req.storage_path,
        "download_url": url,
        "expires_in_seconds": req.expires_in_seconds,
    }


@router.get("/download")
async def local_storage_download(
    bucket: str = Query(...),
    key: str = Query(...),
    expires: int = Query(...),
    sig: str = Query(...),
    filename: Optional[str] = Query(default=None),
    storage: StorageProvider = Depends(get_storage_provider),
):
    if not isinstance(storage, LocalStorageProvider):
        raise HTTPException(status_code=400, detail="Local download endpoint only active for local storage backend.")

    if not storage.verify_signature(bucket, key, expires, sig):
        raise HTTPException(status_code=403, detail="Invalid or expired download signature.")

    path = storage._get_path(bucket, key)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    headers = {}
    if filename:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'

    return FileResponse(path=str(path), headers=headers)


@router.put("/upload")
async def local_storage_upload(
    request: Request,
    bucket: str = Query(...),
    key: str = Query(...),
    expires: int = Query(...),
    sig: str = Query(...),
    mime_type: str = Query(default="application/octet-stream"),
    storage: StorageProvider = Depends(get_storage_provider),
    scratch_disk: ScratchDiskManager = Depends(get_scratch_disk),
):
    if not isinstance(storage, LocalStorageProvider):
        raise HTTPException(status_code=400, detail="Local upload endpoint only active for local storage backend.")

    if not storage.verify_signature(bucket, key, expires, sig):
        raise HTTPException(status_code=403, detail="Invalid or expired upload signature.")

    path = storage._get_path(bucket, key)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Stream upload chunks atomically via temporary file to prevent partial-read races
    total_size = 0
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, prefix=".tmp_") as tmp_file:
            tmp_path = Path(tmp_file.name)
            async for chunk in request.stream():
                if chunk:
                    tmp_file.write(chunk)
                    total_size += len(chunk)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    # Sync artifact to scratch disk path: /tmp/factory_lite/artifacts/{job_id}/{filename}
    parts = [p for p in key.strip("/").split("/") if p]
    if len(parts) >= 2:
        job_id = parts[-2]
        filename = parts[-1]
        try:
            scratch_path = scratch_disk.get_artifact_path(job_id, filename)
            scratch_path.parent.mkdir(parents=True, exist_ok=True)
            if scratch_path.resolve() != path.resolve():
                if scratch_path.exists():
                    scratch_path.unlink()
                try:
                    os.link(str(path), str(scratch_path))
                except OSError:
                    scratch_tmp = scratch_path.parent / f".tmp_{uuid.uuid4().hex}"
                    shutil.copyfile(str(path), str(scratch_tmp))
                    os.replace(str(scratch_tmp), str(scratch_path))
        except Exception as exc:
            logger.warning(f"Failed to sync artifact to scratch disk for job {job_id}: {exc}")

    return {"status": "uploaded", "storage_path": key, "size_bytes": total_size}
