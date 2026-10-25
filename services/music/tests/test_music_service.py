"""
services/music/tests/test_music_service.py
Service-level direct contract verification for Music Subsystem.
"""
import pytest
import httpx
from factory_auth.config import AuthConfig
from services.music.server import app

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
async def test_music_health():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        assert res.json()["subsystem"] == "music"


@pytest.mark.asyncio
async def test_music_text_to_music():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/v1/text_to_music",
            headers=TEST_HEADERS,
            json={"prompt": "Catchy synthpop theme", "duration_seconds": 10, "genre": "synthpop"},
        )
        assert res.status_code == 410


@pytest.mark.asyncio
async def test_music_lyrics_to_music():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/v1/lyrics_to_music",
            headers=TEST_HEADERS,
            json={"prompt": "Pop song", "lyrics": "Hello world from Factory v4", "duration_seconds": 10},
        )
        assert res.status_code == 410


@pytest.mark.asyncio
async def test_music_audio_to_audio():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/v1/audio_to_audio",
            headers=TEST_HEADERS,
            json={"audio_url": "https://s3.ai.example.com/assets/sample.wav", "prompt": "EDM remix"},
        )
        assert res.status_code == 410
