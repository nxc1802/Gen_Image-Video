"""
services/speech/tests/test_speech_service.py
Service-level direct contract verification for Speech Subsystem.
"""
import pytest
import httpx
from factory_auth.config import AuthConfig
from services.speech.server import app

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
async def test_speech_health():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        assert res.json()["subsystem"] == "speech"


@pytest.mark.asyncio
async def test_speech_transcription():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer sk-factory-test-token-valid", "Content-Type": "audio/wav"},
            content=b"\x00\x01\x02\x03" * 50,
        )
        assert res.status_code == 503
        assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"


@pytest.mark.asyncio
async def test_speech_streaming_chunk():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/v1/audio/transcriptions/stream",
            headers=TEST_HEADERS,
            json={"chunk_id": 1, "audio_base64": "AAA="},
        )
        assert res.status_code == 503
        assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"


@pytest.mark.asyncio
async def test_speech_alignment():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/v1/audio/alignments",
            headers=TEST_HEADERS,
            json={"audio_url": "https://assets.ai.example.com/assets/aligned.wav", "transcript_text": "Welcome to Factory"},
        )
        assert res.status_code in (400, 503)


@pytest.mark.asyncio
async def test_speech_synthesis():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/v1/audio/speech",
            headers=TEST_HEADERS,
            json={"model": "qwen3-tts", "input": "Factory v4 decoupled speech", "voice": "alloy"},
        )
        assert res.status_code == 503
        assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"
