"""
Collect real ComfyUI image outputs and upload them through Factory Storage.

Missing files are fatal. No 1x1/mock PNG fallback is permitted.
"""
import mimetypes
from pathlib import Path
from typing import Any, Dict, List
import uuid

import httpx

from factory_contracts.jobs.job_contract import ArtifactReference
from factory.apps.control_api.src.storage.integrity import compute_bytes_sha256
from services.image.config import ImageServiceConfig


class ImageArtifactCollector:
    def __init__(self, config: ImageServiceConfig):
        self.config = config
        self.output_dir = Path(config.output_dir).resolve()

    def _resolve_output_files(self, outputs: Dict[str, Any]) -> List[Path]:
        candidates: List[Path] = []

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                filename = value.get("filename")
                if isinstance(filename, str) and filename.strip():
                    subfolder = str(value.get("subfolder") or "")
                    candidate = (self.output_dir / subfolder / filename).resolve()
                    try:
                        candidate.relative_to(self.output_dir)
                    except ValueError as exc:
                        raise ValueError("ComfyUI output escaped configured output directory.") from exc
                    candidates.append(candidate)
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(outputs)
        unique: List[Path] = []
        seen = set()
        for path in candidates:
            if path not in seen:
                seen.add(path)
                unique.append(path)
        if not unique:
            raise FileNotFoundError("ComfyUI completed without reporting an image output file.")
        for path in unique:
            if not path.is_file():
                raise FileNotFoundError(f"ComfyUI reported output file that does not exist: {path.name}")
        return unique

    async def collect_and_upload(
        self,
        client: httpx.AsyncClient,
        job_id: str,
        outputs: Dict[str, Any],
    ) -> List[ArtifactReference]:
        artifacts: List[ArtifactReference] = []
        for path in self._resolve_output_files(outputs):
            data = path.read_bytes()
            if not data:
                raise ValueError(f"Generated image artifact is empty: {path.name}")
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            sha256 = compute_bytes_sha256(data)

            presign_resp = await client.post(
                "/v1/storage/upload-url",
                json={
                    "subsystem": "image",
                    "job_id": job_id,
                    "artifact_name": path.name,
                    "mime_type": mime,
                    "max_size_bytes": len(data),
                },
            )
            presign_resp.raise_for_status()
            presign = presign_resp.json()

            async with httpx.AsyncClient(timeout=300.0, follow_redirects=False) as upload_client:
                put_resp = await upload_client.put(
                    presign["upload_url"],
                    content=data,
                    headers=presign.get("headers") or {"Content-Type": mime},
                )
                put_resp.raise_for_status()

            download_resp = await client.post(
                "/v1/storage/download-url",
                json={"storage_path": presign["storage_path"], "filename": path.name},
            )
            download_resp.raise_for_status()
            download_url = download_resp.json()["download_url"]

            artifacts.append(
                ArtifactReference(
                    artifact_id=f"art-{uuid.uuid4()}",
                    type="image",
                    mime_type=mime,
                    size_bytes=len(data),
                    storage_path=presign["storage_path"],
                    url=download_url,
                    download_url=download_url,
                    sha256=sha256,
                )
            )
        return artifacts
