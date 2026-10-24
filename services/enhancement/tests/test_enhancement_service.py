"""
services/enhancement/tests/test_enhancement_service.py
Service-level direct contract verification for Media Enhancement Subsystem.
"""
import pytest
import httpx
from factory_auth.config import AuthConfig
from services.enhancement.server import app

TEST_HEADERS = {
    "Authorization": "Bearer sk-factory-test-token-valid",
    "Content-Type": "application/json",
}


@pytest.fixture(autouse=True)
def setup_auth():
    app.state.auth_config = AuthConfig(api_key="sk-factory-test-token-valid", jwt_secret=None, environment="test")
    yield
    if hasattr(app.state, "auth_config"):
        delattr(app.state, "auth_config")


@pytest.mark.asyncio
async def test_enhancement_health():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        assert res.json()["subsystem"] == "enhancement"


@pytest.mark.asyncio
async def test_image_super_resolution():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/v1/image_super_resolution",
            headers=TEST_HEADERS,
            json={"image_url": "https://s3.ai.example.com/assets/sample.png", "scale_factor": 2},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "completed"
        assert data["artifact"]["mime_type"] == "image/png"


@pytest.mark.asyncio
async def test_video_super_resolution():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/v1/video_super_resolution",
            headers=TEST_HEADERS,
            json={"video_url": "https://s3.ai.example.com/assets/sample.mp4", "scale_factor": 2},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "completed"
        assert data["artifact"]["mime_type"] == "video/mp4"


@pytest.mark.asyncio
async def test_frame_interpolation():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/v1/frame_interpolation",
            headers=TEST_HEADERS,
            json={"video_url": "https://s3.ai.example.com/assets/sample.mp4", "multiplier": 2},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "completed"
        assert data["artifact"]["mime_type"] == "video/mp4"


@pytest.mark.asyncio
async def test_audio_enhancement():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/v1/audio_enhancement",
            headers=TEST_HEADERS,
            json={"audio_url": "https://s3.ai.example.com/assets/sample.wav", "mode": "denoise"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "completed"
        assert data["artifact"]["mime_type"] == "audio/wav"
