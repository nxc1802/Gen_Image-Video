"""Official SDK over real local HTTP; explicit doubles, never GPU inference proof."""

import asyncio
import base64
import io
import json
from pathlib import Path
import socket
import subprocess
import threading
import time
from types import SimpleNamespace
import wave
import pytest
import httpx
from fastapi import FastAPI
from openai import (
    OpenAI,
    AsyncOpenAI,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
)
import uvicorn
from factory_auth.config import AuthConfig
from factory_contracts.deployment import DeploymentConfig, DeploymentEntry
from factory.core.model_adapter import ModelAdapter, PreparedCall
from factory.core.media import artifact
from factory.engines.responses import ResponsesEngine
from factory.apps.control_api.src.api.routers import chat, inference, models
from factory.apps.control_api.src.api.errors import register_exception_handlers
from factory.apps.control_api.src.api.middleware import RequestIDMiddleware
from factory.apps.control_api.src.api.public.jobs import PublicMediaJobs
from factory.apps.control_api.src.storage.scratch_disk import ScratchDiskManager, ScratchDiskConfig
from model_registry.deployment import ModelCatalog, DeploymentRegistry

pytestmark = pytest.mark.contract_mock
ROOT = Path(__file__).resolve().parents[2]
KEY = "sk-factory-contract-only"


def png():
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (32, 32), (20, 80, 130)).save(out, format="PNG")
    return out.getvalue()


def wav():
    out = io.BytesIO()
    with wave.open(out, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x01\x00" * 1600)
    return out.getvalue()


@pytest.fixture(scope="module")
def mp4(tmp_path_factory):
    path = tmp_path_factory.mktemp("wire-video") / "fixture.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=32x32:r=4",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    return path.read_bytes()


@pytest.fixture
def server(monkeypatch, tmp_path, mp4):
    monkeypatch.setenv("FACTORY_API_KEY", KEY)
    ids = [
        "qwen-text",
        "qwen-vlm",
        "qwen-embedding",
        "qwen-reranker",
        "qwen-decision",
        "qwen-asr",
        "qwen-tts",
        "qwen-image-2.1",
        "wan-2.2",
        "ace-step-1.5",
        "enhancement-suite-v1",
    ]
    registry = DeploymentRegistry(
        DeploymentConfig(
            deployments=tuple(DeploymentEntry(model=mid, devices=(0,)) for mid in ids)
        ),
        ModelCatalog(ROOT / "manifests"),
    )
    state = SimpleNamespace(calls=[], authorized=True, broken=False, stream_broken=False)

    def authorize(*args):
        registry.assignment(args[0])
        if not state.authorized:
            from factory_contracts.errors import ModelNotLoadedError

            raise ModelNotLoadedError("Contract fixture denies authorization.")

    monkeypatch.setattr(registry, "require_active", authorize)
    session = SimpleNamespace(registry=registry, backends={}, healthy_models=lambda: tuple(ids))
    for mid in ids:
        session.backends[mid] = SimpleNamespace(
            registry=registry, binding=registry.assignment(mid), health=lambda: True
        )

    def native_response(model):
        return {
            "id": "resp_contract",
            "object": "response",
            "created_at": 1,
            "model": model,
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "output": [
                {
                    "type": "message",
                    "id": "msg_contract",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "contract response", "annotations": []}
                    ],
                }
            ],
            "parallel_tool_calls": False,
            "temperature": 1,
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1,
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        }

    class ContractAdapter(ModelAdapter):
        supported_backends = frozenset({b.backend for b in registry.bindings()})
        capabilities = frozenset(
            cid for m in registry.catalog.models().values() for cid in m.capabilities
        )

        def validate_request(self, model, request, capability):
            pass

        def prepare(self, context, request, capability):
            state.calls.append((context.model.engine, capability, request))
            return PreparedCall(capability, {"request": request})

        def infer(self, context, prepared):
            if state.broken:
                from factory_contracts.errors import ModelUnavailableError

                raise ModelUnavailableError("Contract injected backend failure.")
            request = prepared.payload["request"]
            model = context.model
            mid = model.metadata.id
            if model.engine == "responses":
                if prepared.capability == "responses.create":
                    return native_response(mid)
                return {
                    "id": "chatcmpl_contract",
                    "object": "chat.completion",
                    "created": 1,
                    "model": mid,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "contract response"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                }
            if prepared.capability == "vector.text_embedding":
                texts = [request.input] if isinstance(request.input, str) else request.input
                return {
                    "object": "list",
                    "model": mid,
                    "data": [
                        {"index": i, "object": "embedding", "embedding": [0.25, 0.75]}
                        for i in range(len(texts))
                    ],
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                }
            if prepared.capability == "vector.reranking":
                return {
                    "model": mid,
                    "results": [
                        {"index": 0, "document": request.documents[0], "relevance_score": 0.8}
                    ],
                }
            if model.engine == "decision":
                return {
                    "model": mid,
                    "answers": [
                        {"type": "predicate", "name": q.name, "probability": 0.8}
                        for q in request.questions
                    ],
                    "usage": {
                        "input_tokens": 3,
                        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                        "output_tokens": 0,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 3,
                    },
                }
            if prepared.capability == "speech.asr":
                return {"model": mid, "text": "contract transcript"}
            raw, kind, mime = (
                (png(), "image", "image/png")
                if model.engine in {"image", "enhancement"}
                else (mp4, "video", "video/mp4")
                if model.engine == "video"
                else (wav(), "audio", "audio/wav")
            )
            return {"model": mid, "artifacts": [artifact(raw, kind, mime)]}

        def normalize_output(self, context, prepared, output):
            return output

        async def stream(self, context, prepared):
            if prepared.capability == "responses.create":
                snapshot = native_response(context.binding.model)
                for n, event in enumerate(
                    ["response.created", "response.output_text.delta", "response.completed"]
                ):
                    if state.stream_broken and n == 2:
                        from factory_contracts.errors import ModelUnavailableError

                        raise ModelUnavailableError("Injected stream truncation")
                    body = {"type": event, "sequence_number": n}
                    if n in (0, 2):
                        body["response"] = (
                            {**snapshot, "status": "in_progress"} if n == 0 else snapshot
                        )
                    else:
                        body.update(
                            item_id="msg_contract",
                            output_index=0,
                            content_index=0,
                            delta="contract response",
                        )
                    yield "event: " + event + "\ndata: " + json.dumps(body) + "\n\n"
            else:
                for finish in (None, "stop"):
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": "chatcmpl_contract",
                                "object": "chat.completion.chunk",
                                "created": 1,
                                "model": context.binding.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": "contract response"}
                                        if finish is None
                                        else {},
                                        "finish_reason": finish,
                                    }
                                ],
                            }
                        )
                        + "\n\n"
                    )
                yield "data: [DONE]\n\n"

    def make(request, cls):
        return cls(registry, session, adapter_loader=lambda name: ContractAdapter())

    monkeypatch.setattr(inference, "engine", make)
    monkeypatch.setattr(chat, "_get_text_engine", lambda request: make(request, ResponsesEngine))
    app = FastAPI()
    app.state.auth_config = AuthConfig(api_key=KEY, environment="test")
    app.state.deployment_registry = registry
    app.state.backend_session = session
    app.state.scratch_disk = ScratchDiskManager(
        ScratchDiskConfig(root_dir=str(tmp_path / "scratch"))
    )
    app.state.public_media_jobs = PublicMediaJobs(app.state.scratch_disk)
    app.add_middleware(RequestIDMiddleware)
    register_exception_handlers(app)
    for router in (chat.router, inference.router, models.router):
        app.include_router(router)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(32)
    address = "http://127.0.0.1:" + str(sock.getsockname()[1]) + "/v1"
    config = uvicorn.Config(app, log_level="error", lifespan="off")
    worker = uvicorn.Server(config)
    thread = threading.Thread(target=worker.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not worker.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert worker.started
    state.app = app
    state.base_url = address
    state.registry = registry
    with OpenAI(api_key=KEY, base_url=address, max_retries=0, timeout=10) as sdk:
        state.sdk = sdk
        yield state
    worker.should_exit = True
    thread.join(timeout=10)
    sock.close()
    assert not thread.is_alive()


def test_sdk_chat_and_responses(server):
    chat_result = server.sdk.chat.completions.create(
        model="qwen-text",
        messages=[{"role": "user", "content": "contract input"}],
        max_completion_tokens=16,
    )
    assert chat_result.choices[0].message.content == "contract response"
    result = server.sdk.responses.create(model="qwen-text", input="contract input", store=False)
    assert result.output_text == "contract response" and result.id == "resp_contract"
    assert all(not hasattr(call, "service") for _, _, call in server.calls)


def test_sdk_chat_and_responses_streams(server):
    chunks = list(
        server.sdk.chat.completions.create(
            model="qwen-text", messages=[{"role": "user", "content": "contract input"}], stream=True
        )
    )
    assert chunks[-1].choices[0].finish_reason == "stop"
    events = list(
        server.sdk.responses.create(model="qwen-text", input="contract input", stream=True)
    )
    assert events[-1].type == "response.completed" and events[1].delta == "contract response"


def test_sdk_failed_stream_never_emits_completion(server):
    server.stream_broken = True
    seen = []
    with pytest.raises(Exception):
        for event in server.sdk.responses.create(
            model="qwen-text", input="contract input", stream=True
        ):
            seen.append(event.type)
    assert "response.completed" not in seen


def test_sdk_embeddings_float_and_default_base64(server):
    for kwargs in ({"encoding_format": "float"}, {}):
        result = server.sdk.embeddings.create(
            model="qwen-embedding", input=["one", "two"], dimensions=2, **kwargs
        )
        assert result.data[0].embedding == [0.25, 0.75] and result.data[1].index == 1
        assert result.usage.total_tokens == 3


def test_sdk_audio_upload_and_waveform(server):
    result = server.sdk.audio.transcriptions.create(
        model="qwen-asr", file=("input.wav", wav(), "audio/wav")
    )
    assert result.text == "contract transcript"
    text = server.sdk.audio.transcriptions.create(
        model="qwen-asr", file=("input.wav", wav()), response_format="text"
    )
    assert text == "contract transcript"
    out = server.sdk.audio.speech.create(
        model="qwen-tts",
        voice="default",
        input="contract input",
        response_format="wav",
        extra_body={
            "reference_audio": "data:audio/wav;base64," + base64.b64encode(wav()).decode(),
            "reference_text": "contract reference",
        },
    )
    assert out.content == wav()
    assert server.calls[-1][2].inputs["text"] == "contract input"


def test_sdk_image_generate_and_multipart_edits(server):
    result = server.sdk.images.generate(
        model="qwen-image-2.1", prompt="contract input", response_format="b64_json", size="512x512"
    )
    assert base64.b64decode(result.data[0].b64_json) == png()
    result = server.sdk.images.edit(
        model="qwen-image-2.1",
        prompt="contract change",
        image=("source.png", png()),
        mask=("mask.png", png()),
    )
    assert result.data[0].b64_json
    assert server.calls[-1][1] == "image.inpaint" and server.calls[-1][2].inputs["image"] == png()
    server.sdk.images.edit(
        model="qwen-image-2.1",
        prompt="contract change",
        image=[("one.png", png()), ("two.png", png())],
    )
    assert server.calls[-1][1] == "image.multi_reference"


def completed_video(server, **kwargs):
    created = server.sdk.videos.create(
        model="wan-2.2", prompt="contract video", seconds="4", size="512x512", **kwargs
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = server.sdk.videos.retrieve(created.id)
        if result.status in ("completed", "failed"):
            break
        time.sleep(0.02)
    assert result.status == "completed", result.error
    return result


def test_sdk_video_create_retrieve_content_extension(server, mp4):
    result = completed_video(server, input_reference=("source.png", png()))
    assert result.object == "video" and result.seconds == "1" and result.progress == 100
    assert server.sdk.videos.download_content(result.id).content == mp4
    assert server.calls[-1][1] == "video.image_to_video"
    extended = server.sdk.videos.extend(
        video=("source.mp4", mp4),
        prompt="contract extension",
        seconds="4",
        extra_body={"model": "wan-2.2", "size": "512x512"},
    )
    assert extended.model == "wan-2.2"
    with pytest.raises(BadRequestError):
        server.sdk.videos.edit(
            video=("source.mp4", mp4), prompt="contract edit", extra_body={"model": "wan-2.2"}
        )
    response = httpx.post(
        server.base_url + "/videos/extensions",
        headers={"Authorization": "Bearer " + KEY},
        json={"video": {"id": result.id}, "prompt": "contract continuation", "seconds": "4"},
    )
    assert response.status_code == 200 and response.json()["model"] == "wan-2.2"


def test_sdk_decisions_and_models(server):
    result = server.sdk.decisions.create(
        model="qwen-decision",
        input="contract input",
        questions=[{"type": "predicate", "name": "test", "instructions": "contract question"}],
    )
    assert result.answers[0].probability == 0.8 and result.answers[0].name == "test"
    assert server.sdk.models.list().data == []


@pytest.mark.asyncio
async def test_async_sdk(server):
    async with AsyncOpenAI(api_key=KEY, base_url=server.base_url, max_retries=0) as sdk:
        assert (
            await sdk.responses.create(model="qwen-text", input="contract input")
        ).output_text == "contract response"


@pytest.mark.parametrize(
    "resource,kwargs",
    [
        ("responses", {"model": "qwen-text", "input": "text"}),
        ("embeddings", {"model": "qwen-embedding", "input": "text"}),
        (
            "speech",
            {"model": "qwen-tts", "input": "text", "voice": "default", "response_format": "wav"},
        ),
        ("images", {"model": "qwen-image-2.1", "prompt": "text"}),
        ("videos", {"model": "wan-2.2", "prompt": "text"}),
        (
            "decisions",
            {
                "model": "qwen-decision",
                "input": "text",
                "questions": [{"type": "predicate", "instructions": "text"}],
            },
        ),
    ],
)
def test_unvalidated_session_fails_closed_through_sdk(server, resource, kwargs):
    server.authorized = False
    method = {
        "responses": server.sdk.responses.create,
        "embeddings": server.sdk.embeddings.create,
        "speech": server.sdk.audio.speech.create,
        "images": server.sdk.images.generate,
        "videos": server.sdk.videos.create,
        "decisions": server.sdk.decisions.create,
    }[resource]
    with pytest.raises(APIStatusError) as caught:
        method(**kwargs)
    assert caught.value.status_code == 503 and caught.value.body["type"] == "server_error"
    assert server.calls == []


@pytest.mark.parametrize("field", ["service", "backend", "devices", "checkpoint", "variant"])
def test_public_rejects_deployment_overrides(server, field):
    with pytest.raises(APIStatusError) as caught:
        server.sdk.images.generate(
            model="qwen-image-2.1", prompt="text", extra_body={field: "forbidden"}
        )
    assert caught.value.status_code == 400 and caught.value.body["param"] == field
    assert server.calls == []


def test_auth_and_error_objects(server):
    with OpenAI(api_key="wrong", base_url=server.base_url, max_retries=0) as sdk:
        with pytest.raises(AuthenticationError) as caught:
            sdk.models.list()
        assert caught.value.body["type"] == "invalid_request_error"
        assert caught.value.request_id.startswith("req-")
    server.broken = True
    with pytest.raises(InternalServerError) as caught:
        server.sdk.responses.create(model="qwen-text", input="text")
    assert caught.value.status_code == 503 and caught.value.body["param"] is None


def test_remote_media_ssrf_rejected_before_adapter(server):
    with pytest.raises(BadRequestError):
        server.sdk.chat.completions.create(
            model="qwen-vlm",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "http://127.0.0.1/private"}}
                    ],
                }
            ],
        )
    assert server.calls == []


def test_video_unknown_and_not_completed_content(server):
    with pytest.raises(APIStatusError) as caught:
        server.sdk.videos.retrieve("video_unknown")
    assert caught.value.status_code == 404


def test_custom_music_jobs_and_disabled_enhancement(server):
    headers = {"Authorization": "Bearer " + KEY}
    response = httpx.post(
        server.base_url + "/music",
        headers=headers,
        json={
            "model": "ace-step-1.5",
            "capability": "music.text_to_music",
            "inputs": {"prompt": "contract music"},
        },
    )
    assert response.status_code == 200, response.text
    identifier = response.json()["id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = httpx.get(server.base_url + "/music/" + identifier, headers=headers)
        if response.json()["status"] in ("completed", "failed"):
            break
        time.sleep(0.02)
    assert response.json()["status"] == "completed"
    assert (
        httpx.get(server.base_url + "/music/" + identifier + "/content", headers=headers).content
        == wav()
    )
    assert "service" not in response.json()
    response = httpx.post(
        server.base_url + "/enhancements",
        headers=headers,
        json={
            "model": "enhancement-suite-v1",
            "capability": "enhancement.image_super_resolution",
            "inputs": {"image": "data:image/png;base64," + base64.b64encode(png()).decode()},
        },
    )
    assert response.status_code == 400  # Declared false even with fixture authorization.


def test_job_failure_has_no_content_or_false_completion(server):
    server.broken = True
    result = server.sdk.videos.create(model="wan-2.2", prompt="contract failure", seconds="4")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = server.sdk.videos.retrieve(result.id)
        if result.status == "failed":
            break
        time.sleep(0.02)
    assert result.status == "failed" and result.progress == 0 and result.completed_at is None
    with pytest.raises(APIStatusError) as caught:
        server.sdk.videos.download_content(result.id)
    assert caught.value.status_code == 409


def test_video_content_is_bound_to_owner_and_kind(server):
    result = completed_video(server)
    row = server.app.state.public_media_jobs.rows[result.id]
    row["owner"] = "different-user"
    with pytest.raises(APIStatusError) as caught:
        server.sdk.videos.retrieve(result.id)
    assert caught.value.status_code == 404
    with pytest.raises(APIStatusError) as caught:
        server.sdk.videos.download_content(result.id)
    assert caught.value.status_code == 404
    row["owner"] = ":system"
    row["kind"] = "music"
    with pytest.raises(APIStatusError) as caught:
        server.sdk.videos.retrieve(result.id)
    assert caught.value.status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"size": "bad"},
        {"steps": 1.5},
        {"cfg": "NaN"},
        {"size": "0x512"},
        {"n": 2},
        {"output_format": "svg"},
        {"quality": "hd"},
        {"response_format": "url"},
    ],
)
def test_unsupported_or_malformed_image_parameters_fail_closed(server, body):
    with pytest.raises(APIStatusError) as caught:
        server.sdk.images.generate(model="qwen-image-2.1", prompt="contract input", extra_body=body)
    assert caught.value.status_code == 400 and server.calls == []


def test_main_application_mounts_all_required_routes():
    from factory.apps.control_api.src.main import app

    paths = app.openapi()["paths"]
    assert {
        "/v1/responses",
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/audio/transcriptions",
        "/v1/audio/speech",
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/videos",
        "/v1/videos/edits",
        "/v1/videos/extensions",
        "/v1/models",
        "/v1/decisions",
        "/v1/rerank",
        "/v1/music",
        "/v1/enhancements",
    } <= set(paths)


@pytest.mark.asyncio
async def test_body_limit_measures_actual_bytes(monkeypatch):
    import factory.apps.control_api.src.api.public.body_limit as limits

    monkeypatch.setattr(limits, "MAX_BODY", 8)
    reached = []
    sent = []

    async def app(scope, receive, send):
        reached.append(True)

    messages = iter(
        [
            {"type": "http.request", "body": b"12345", "more_body": True},
            {"type": "http.request", "body": b"6789", "more_body": False},
        ]
    )

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    await limits.PublicBodyLimit(app)(
        {"type": "http", "path": "/v1/responses", "method": "POST"}, receive, send
    )
    assert reached == [] and sent[0]["status"] == 413


def test_custom_rerank_uses_canonical_engine(server):
    response = httpx.post(
        server.base_url + "/rerank",
        headers={"Authorization": "Bearer " + KEY},
        json={"model": "qwen-reranker", "query": "contract query", "documents": ["contract doc"]},
    )
    assert response.status_code == 200 and response.json()["results"][0]["relevance_score"] == 0.8
    assert server.calls[-1][1] == "vector.reranking"
