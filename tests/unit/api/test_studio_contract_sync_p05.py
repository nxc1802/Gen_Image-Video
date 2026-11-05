"""
tests/unit/api/test_studio_contract_sync_p05.py
Unit tests verifying Gate 2 (P0.5) Studio Web & Control Plane Contract Synchronization:
- Canonical task normalization in job submission (image.t2i, video.t2v, etc.)
- Studio capability resolution against model manifest capabilities
- Telemetry memory endpoint (/v1/memory) dual-T4 gauges
- Studio schema discovery endpoint (/api/studio/schemas)
- Asset download verification
"""
import os
import pytest
from starlette.testclient import TestClient

from factory.apps.control_api.src.main import app
from factory.apps.control_api.src.telemetry.metrics import metrics

VALID_TOKEN = "sk-factory-prod-secret-token"


@pytest.fixture(scope="module")
def api_client():
    keys_to_preserve = ["FACTORY_API_KEY", "STORAGE_BACKEND", "FACTORY_DB_BACKEND", "SQLITE_DB_PATH"]
    old_env = {k: os.environ.get(k) for k in keys_to_preserve}
    os.environ["FACTORY_API_KEY"] = VALID_TOKEN
    os.environ["STORAGE_BACKEND"] = "local"
    os.environ["FACTORY_DB_BACKEND"] = "sqlite"
    os.environ["SQLITE_DB_PATH"] = ":memory:"
    try:
        with TestClient(app) as client:
            yield client
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if hasattr(app.state, "auth_config"):
            delattr(app.state, "auth_config")


@pytest.fixture
def auth_headers():
    return {
        "Authorization": f"Bearer {VALID_TOKEN}",
        "Content-Type": "application/json",
    }


def test_studio_capability_task_submission_normalizes_image_t2i(api_client, auth_headers):
    """Submitting task='image.t2i' resolves canonically to 'text_to_image' and creates job."""
    payload = {
        "service": "image",
        "model": "qwen-image-2.1",
        "task": "image.t2i",
        "inputs": {"prompt": "Sunset over karst formations"},
    }
    res = api_client.post("/v1/jobs", headers=auth_headers, json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["task"] == "text_to_image"
    assert data["model"] == "qwen-image-2.1"
    assert data["service"] == "image"
    assert data["state"] == "QUEUED"


def test_studio_capability_task_submission_normalizes_video_t2v(api_client, auth_headers):
    """Submitting task='video.t2v' resolves canonically to 'text_to_video'."""
    payload = {
        "service": "video",
        "model": "minimax-h3",
        "task": "video.t2v",
        "inputs": {"prompt": "Majestic golden dragon flying over water"},
    }
    res = api_client.post("/v1/jobs", headers=auth_headers, json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["task"] == "text_to_video"
    assert data["model"] == "minimax-h3"
    assert data["service"] == "video"


def test_studio_capability_unsupported_task_fails_400(api_client, auth_headers):
    """Submitting unsupported task for model fails with CAPABILITY_NOT_SUPPORTED."""
    payload = {
        "service": "image",
        "model": "qwen-image-2.1",
        "task": "video.t2v",
        "inputs": {"prompt": "Invalid task for image service"},
    }
    res = api_client.post("/v1/jobs", headers=auth_headers, json=payload)
    assert res.status_code == 400
    err = res.json()["error"]
    assert err["code"] == "CAPABILITY_NOT_SUPPORTED"


def test_memory_endpoint_dual_t4_telemetry(api_client, auth_headers):
    """Verify /v1/memory returns dual-T4 GPU telemetry structure for Studio Web."""
    metrics.set_gpu_vram(gpu_0_gb=4.5, gpu_1_gb=8.2)

    res = api_client.get("/v1/memory", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert "gpus" in data
    gpus = data["gpus"]
    assert len(gpus) == 2

    # GPU 0
    assert gpus[0]["index"] == 0
    assert "Tesla T4 (GPU 0)" in gpus[0]["name"]
    assert gpus[0]["hardware_used_gb"] == 4.5
    assert gpus[0]["total_gb"] == 15.0
    assert gpus[0]["free_gb"] == 10.5

    # GPU 1
    assert gpus[1]["index"] == 1
    assert "Tesla T4 (GPU 1)" in gpus[1]["name"]
    assert gpus[1]["hardware_used_gb"] == 8.2
    assert gpus[1]["total_gb"] == 15.0
    assert gpus[1]["free_gb"] == 6.8


def test_studio_schemas_discovery(api_client, auth_headers):
    """Verify /api/studio/schemas returns complete schemas structure."""
    res = api_client.get("/api/studio/schemas", headers=auth_headers)
    assert res.status_code == 200
    schemas = res.json()["schemas"]
    assert "services" in schemas
    assert "models" in schemas
    assert len(schemas["services"]) > 0
    assert len(schemas["models"]) > 0
