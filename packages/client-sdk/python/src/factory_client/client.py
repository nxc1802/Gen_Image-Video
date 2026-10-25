"""
factory_client/client.py
Official Python Client SDK for Factory v4 Control API.
"""
from typing import Any, Dict, List, Optional, Union
import httpx


class FactoryClient:
    """Official Python Client for Factory v4 Control Plane API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "sk-factory-studio",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    def close(self) -> None:
        if self._client and not self._client.is_closed:
            self._client.close()

    def __enter__(self) -> "FactoryClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def get_health(self) -> Dict[str, Any]:
        client = self._get_client()
        resp = client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def get_services(self) -> List[Dict[str, Any]]:
        client = self._get_client()
        resp = client.get("/v1/services")
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", data)

    def get_models(self) -> List[Dict[str, Any]]:
        client = self._get_client()
        resp = client.get("/v1/models")
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", data)

    def get_capabilities(self) -> List[Dict[str, Any]]:
        client = self._get_client()
        resp = client.get("/v1/capabilities")
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", data)

    def submit_job(
        self,
        service: str,
        model: str,
        task: str,
        inputs: Dict[str, Any],
        priority: int = 10,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        client = self._get_client()
        payload: Dict[str, Any] = {
            "service": service,
            "model": model,
            "task": task,
            "inputs": inputs,
            "priority": priority,
        }
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key
        resp = client.post("/v1/jobs", json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_job(self, job_id: str) -> Dict[str, Any]:
        client = self._get_client()
        resp = client.get(f"/v1/jobs/{job_id}")
        resp.raise_for_status()
        return resp.json()

    def list_jobs(
        self,
        limit: int = 50,
        offset: int = 0,
        service: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        client = self._get_client()
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if service:
            params["service"] = service
        if state:
            params["state"] = state
        resp = client.get("/v1/jobs", params=params)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("jobs", [])

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        client = self._get_client()
        resp = client.post(f"/v1/jobs/{job_id}/cancel")
        resp.raise_for_status()
        return resp.json()

    def retry_job(self, job_id: str) -> Dict[str, Any]:
        client = self._get_client()
        resp = client.post(f"/v1/jobs/{job_id}/retry")
        resp.raise_for_status()
        return resp.json()

    def upload_asset(
        self,
        file_content: Union[bytes, str],
        filename: str = "upload.png",
        content_type: str = "image/png",
    ) -> Dict[str, Any]:
        client = self._get_client()
        if isinstance(file_content, str):
            with open(file_content, "rb") as f:
                data = f.read()
        else:
            data = file_content
        files = {"file": (filename, data, content_type)}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = client.post("/v1/assets/upload", files=files, headers=headers)
        resp.raise_for_status()
        return resp.json()
