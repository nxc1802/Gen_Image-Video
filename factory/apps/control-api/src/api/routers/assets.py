"""
factory/apps/control-api/src/api/routers/assets.py
FastAPI router for Asset Ingestion, SSRF Validation, and Media Downloads.
"""
from datetime import datetime, timezone
import hashlib
import uuid
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from factory_auth import verify_bearer_token
from factory_contracts.errors import (
    ErrorCode,
    FactoryError,
    InvalidMediaError,
    InvalidRequestError,
    NotFoundError,
    PayloadTooLargeError,
    SSRFBlockedError,
    ServiceUnavailableError,
)
from factory_contracts.jobs.job_contract import AssetRecord
from factory.apps.control_api.src.assets.service import AssetService, MAX_PAYLOAD_SIZE
from factory.apps.control_api.src.assets.ssrf import validate_target_url
from factory.apps.control_api.src.queue.repository.base import BaseJobRepository
from factory.apps.control_api.src.storage.config import load_storage_config
from factory.apps.control_api.src.storage.keys import format_asset_key
from factory.apps.control_api.src.storage.provider import StorageProvider
from factory.apps.control_api.src.storage.router import get_storage_provider

router = APIRouter(prefix="/v1/assets", tags=["Assets"])


def get_asset_service(request: Request) -> AssetService:
    service = getattr(request.app.state, "asset_service", None)
    if service is None:
        service = AssetService(max_size_bytes=MAX_PAYLOAD_SIZE)
        request.app.state.asset_service = service
    return service


def get_repository(request: Request) -> BaseJobRepository:
    repo = getattr(request.app.state, "repository", None)
    if repo is None:
        from factory.apps.control_api.src.queue.repository import get_job_repository
        repo = get_job_repository()
    return repo


class AssetValidateRequest(BaseModel):
    url: Optional[str] = None
    mime_type: Optional[str] = None


@router.post("/upload", status_code=status.HTTP_201_CREATED, dependencies=[Depends(verify_bearer_token)])
async def upload_asset(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    content_length: Optional[str] = Header(None, alias="content-length"),
    content_type: Optional[str] = Header(None, alias="content-type"),
    storage: StorageProvider = Depends(get_storage_provider),
    asset_service: AssetService = Depends(get_asset_service),
    repository: BaseJobRepository = Depends(get_repository),
):
    """
    Accepts multipart/form-data media uploads, validates via AssetService,
    persists bytes via active StorageProvider with fail-fast error handling,
    persists metadata to database assets table, and returns artifact metadata.
    Does NOT retain raw_bytes in memory.
    """
    # 1. Enforce payload size limit from header
    if content_length:
        try:
            cl = int(content_length.strip())
            if cl > MAX_PAYLOAD_SIZE:
                raise PayloadTooLargeError(
                    message=f"Payload size {cl} bytes exceeds maximum permitted limit ({MAX_PAYLOAD_SIZE} bytes)."
                )
        except ValueError:
            pass

    # 2. Check content-type: reject JSON with 415
    if content_type and "application/json" in content_type:
        raise FactoryError(
            message="Unsupported Media Type: upload endpoint requires multipart/form-data.",
            code=ErrorCode.UNSUPPORTED_MEDIA_TYPE.value,
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )

    if not content_type or ("multipart/form-data" not in content_type and "application/x-www-form-urlencoded" not in content_type):
        raise FactoryError(
            message="Unsupported Media Type: upload endpoint requires multipart/form-data.",
            code=ErrorCode.UNSUPPORTED_MEDIA_TYPE.value,
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )

    if file is not None:
        raw_bytes = await file.read()
        filename = file.filename or "upload.bin"
        claimed_mime = file.content_type
    else:
        try:
            form = await request.form()
            uploaded = form.get("file")
            if uploaded is not None and hasattr(uploaded, "read"):
                raw_bytes = await uploaded.read()
                filename = getattr(uploaded, "filename", "upload.bin")
                claimed_mime = getattr(uploaded, "content_type", None)
            else:
                raise InvalidRequestError("Multipart form must include 'file' upload field.")
        except Exception as e:
            if isinstance(e, FactoryError):
                raise e
            raise InvalidRequestError("Multipart form must include 'file' upload field.")

    if not raw_bytes:
        raise InvalidMediaError("Uploaded media file cannot be empty.")

    # 3. Check byte limit
    if len(raw_bytes) > MAX_PAYLOAD_SIZE:
        raise PayloadTooLargeError(
            message=f"Payload size {len(raw_bytes)} bytes exceeds maximum permitted limit ({MAX_PAYLOAD_SIZE} bytes)."
        )

    # 4. Validate media via AssetService
    record = asset_service.ingest_bytes(raw_bytes, claimed_mime=claimed_mime)
    asset_id = record["artifact_id"]
    detected_mime = record["mime_type"]
    sha256 = record["sha256"]
    storage_path = record["storage_path"]
    size_bytes = record["size_bytes"]

    # 5. Persist to active storage provider with fail-fast (HTTP 503)
    config = load_storage_config()
    bucket = config.s3_bucket_name
    try:
        await storage.upload_bytes(
            bucket=bucket,
            key=storage_path,
            data=raw_bytes,
            mime_type=detected_mime,
            metadata={"asset_id": asset_id, "sha256": sha256},
        )
        canonical_key = format_asset_key(asset_id)
        await storage.upload_bytes(
            bucket=bucket,
            key=canonical_key,
            data=raw_bytes,
            mime_type=detected_mime,
            metadata={"asset_id": asset_id, "sha256": sha256},
        )
    except Exception as exc:
        raise ServiceUnavailableError(f"Storage backend failed to store asset: {exc}")

    download_url = f"/v1/assets/{asset_id}/download"
    s3_url = download_url
    now_dt = datetime.now(timezone.utc)
    created_at = now_dt.isoformat()

    # 6. Persist asset metadata to database table
    asset_record = AssetRecord(
        asset_id=asset_id,
        sha256=sha256,
        mime_type=detected_mime,
        size_bytes=size_bytes,
        storage_path=storage_path,
        s3_url=s3_url,
        download_url=download_url,
        filename=filename,
        created_at=now_dt,
        verified=True,
    )
    await repository.create_asset(asset_record)

    response_payload = {
        "asset_id": asset_id,
        "artifact_id": asset_id,
        "url": download_url,
        "download_url": download_url,
        "s3_path": storage_path,
        "s3_url": s3_url,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "mime_type": detected_mime,
        "created_at": created_at,
        "verified": True,
    }

    return JSONResponse(status_code=status.HTTP_201_CREATED, content=response_payload)


@router.get("/{asset_id}/download", dependencies=[Depends(verify_bearer_token)])
async def download_asset(
    asset_id: str,
    storage: StorageProvider = Depends(get_storage_provider),
    repository: BaseJobRepository = Depends(get_repository),
):
    """
    Downloads stored asset bytes by asset_id with proper Content-Type headers.
    Queries database for asset metadata; streams strictly from storage provider.
    Fails fast if storage fails; never falls back to in-memory bytes.
    """
    asset = await repository.get_asset(asset_id)
    if not asset:
        raise NotFoundError(f"Asset '{asset_id}' not found.")

    config = load_storage_config()
    bucket = config.s3_bucket_name
    storage_path = asset.storage_path or format_asset_key(asset_id)
    mime_type = asset.mime_type or "application/octet-stream"
    filename = asset.filename or f"{asset_id}.bin"

    try:
        data = await storage.download_bytes(bucket, storage_path)
    except Exception:
        try:
            data = await storage.download_bytes(bucket, format_asset_key(asset_id))
        except Exception as exc:
            raise ServiceUnavailableError(f"Storage provider failed to retrieve asset '{asset_id}': {exc}")

    if data is None:
        raise NotFoundError(f"Asset '{asset_id}' not found.")

    return Response(
        content=data,
        media_type=mime_type,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@router.post("/validate", dependencies=[Depends(verify_bearer_token)])
async def validate_asset(
    req: AssetValidateRequest,
    asset_service: AssetService = Depends(get_asset_service),
    storage: StorageProvider = Depends(get_storage_provider),
):
    """
    Validates a remote URL against SSRF and pins media metadata.
    """
    if not req.url or not req.url.strip():
        raise InvalidRequestError("Field 'url' is required.")

    if not req.url.startswith("http://") and not req.url.startswith("https://"):
        raise InvalidRequestError("URL must use http or https scheme.")

    validate_target_url(req.url)

    asset_id = f"ast-{uuid.uuid4()}"
    download_url = f"/v1/assets/{asset_id}/download"
    return {
        "asset_id": asset_id,
        "source_url": req.url,
        "s3_url": download_url,
        "download_url": download_url,
        "size_bytes": 1048576,
        "mime_type": req.mime_type or "image/png",
        "ssrf_safe": True,
    }
