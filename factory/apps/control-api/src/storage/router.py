"""
factory/apps/control-api/src/storage/router.py
FastAPI router for storage presigned upload generation and local storage fallback endpoints.
"""
from typing import Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from factory_auth import verify_bearer_token
from factory_contracts.errors import InvalidRequestError, UnauthorizedError
from factory.apps.control_api.src.storage.config import StorageConfig, load_storage_config
from factory.apps.control_api.src.storage.keys import format_artifact_key
from factory.apps.control_api.src.storage.local import LocalStorageProvider
from factory.apps.control_api.src.storage.provider import PresignedUpload, StorageProvider
from factory.apps.control_api.src.storage.s3 import S3StorageProvider


class UploadURLRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subsystem: str = Field(default="artifacts")
    job_id: str = Field(..., description="Job identifier")
    artifact_name: str = Field(..., description="File name of artifact")
    mime_type: str = Field(default="application/octet-stream")
    max_size_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    expires_in_seconds: int = Field(default=3600, ge=60, le=86400)


router = APIRouter(prefix="/v1/storage", tags=["Storage"])


def get_storage_provider(request: Request) -> StorageProvider:
    provider = getattr(request.app.state, "storage_provider", None)
    if provider is not None:
        return provider
    config = load_storage_config()
    if config.storage_backend == "s3" and config.s3_endpoint_url:
        try:
            return S3StorageProvider(config)
        except Exception:
            return LocalStorageProvider(config)
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
):
    if not isinstance(storage, LocalStorageProvider):
        raise HTTPException(status_code=400, detail="Local upload endpoint only active for local storage backend.")

    if not storage.verify_signature(bucket, key, expires, sig):
        raise HTTPException(status_code=403, detail="Invalid or expired upload signature.")

    body = await request.body()
    await storage.upload_bytes(bucket, key, body, mime_type)
    return {"status": "uploaded", "storage_path": key, "size_bytes": len(body)}
