"""Phase 7 software/process contracts; native calls and GPU metadata are doubles."""

import asyncio
import io
from pathlib import Path
import threading
from types import SimpleNamespace
import wave
import pytest
from factory.core.media import MediaRequest, artifact
from factory.core.model_adapter import ModelAdapter, PreparedCall
from factory.core.execution import async_execution_scope, device_locks
from factory.engines.music import MusicEngine
from factory.apps.control_api.src.api.public.jobs import PublicMediaJobs
from factory.apps.control_api.src.storage.scratch_disk import ScratchDiskManager, ScratchDiskConfig
from factory_contracts.errors import ConflictError, NotFoundError, ModelNotLoadedError

pytestmark = pytest.mark.contract_mock


def wave_bytes():
    out = io.BytesIO()
    with wave.open(out, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(b"\x01\0" * 80)
    return out.getvalue()


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.fixture
def runtime(routing_registry, monkeypatch, tmp_path):
    registry = routing_registry([("ace-step-1.5", [0])])
    monkeypatch.setattr(registry, "require_active", lambda *args: None)
    binding = registry.assignment("ace-step-1.5")
    backend = SimpleNamespace(registry=registry, binding=binding, health=lambda: True)
    session = SimpleNamespace(
        registry=registry,
        backends={binding.model: backend},
        healthy_models=lambda: (binding.model,),
    )
    state = SimpleNamespace(
        calls=0, running=0, peak=0, started=threading.Event(), release=threading.Event(), fail=False
    )

    class ContractAdapter(ModelAdapter):
        capabilities = frozenset({"music.text_to_music"})
        supported_backends = frozenset({"acestep_native"})

        def validate_request(self, model, request, capability):
            pass

        def prepare(self, context, request, capability):
            return PreparedCall(capability, {})

        def infer(self, context, prepared):
            state.calls += 1
            state.running += 1
            state.peak = max(state.running, state.peak)
            state.started.set()
            try:
                assert state.release.wait(5), "Contract native call timed out"
                if state.fail:
                    raise RuntimeError("Explicit contract backend failure")
                return {
                    "model": binding.model,
                    "artifacts": [artifact(wave_bytes(), "audio", "audio/wav")],
                }
            finally:
                state.running -= 1

        def normalize_output(self, context, prepared, output):
            return output

    state.engine = MusicEngine(registry, session, adapter_loader=lambda name: ContractAdapter())
    state.call = MediaRequest(
        model=binding.model,
        capability="music.text_to_music",
        inputs={"prompt": "software-only contract"},
    )
    state.jobs = PublicMediaJobs(
        ScratchDiskManager(ScratchDiskConfig(root_dir=str(tmp_path / "scratch")))
    )
    yield state
    state.release.set()


@pytest.mark.asyncio
async def test_waiting_cancel_never_reaches_native_backend(runtime):
    r = runtime
    first = await r.jobs.submit(r.engine, r.call, "owner", "music", {})
    await until(lambda: r.started.is_set())
    second = await r.jobs.submit(r.engine, r.call, "owner", "music", {})
    await asyncio.sleep(0.03)
    assert r.jobs.get(second["id"], "owner")["status"] == "queued"
    assert r.jobs.cancel(second["id"], "owner")["status"] == "cancelled"
    r.release.set()
    await until(lambda: not r.jobs.tasks)
    assert r.calls == 1 and r.jobs.get(first["id"], "owner")["status"] == "completed"
    with pytest.raises(ConflictError):
        await r.jobs.content(second["id"], "owner")
    await r.jobs.close()


@pytest.mark.asyncio
async def test_running_cancel_keeps_gpu_reserved_until_native_call_returns(runtime):
    r = runtime
    first = await r.jobs.submit(r.engine, r.call, "owner", "music", {})
    await until(lambda: r.started.is_set())
    second = await r.jobs.submit(r.engine, r.call, "owner", "music", {})
    assert r.jobs.cancel(first["id"], "owner")["status"] == "cancelling"
    await asyncio.sleep(0.05)
    assert r.calls == 1 and r.running == 1
    assert r.jobs.get(second["id"], "owner")["status"] == "queued"
    assert r.jobs.cancel(first["id"], "owner")["status"] == "cancelling"  # idempotent
    r.release.set()
    await until(lambda: not r.jobs.tasks)
    assert r.peak == 1 and r.calls == 2
    assert r.jobs.get(first["id"], "owner")["status"] == "cancelled"
    assert r.jobs.get(second["id"], "owner")["status"] == "completed"
    with pytest.raises(ConflictError):
        await r.jobs.content(first["id"], "owner")
    progress = [event["progress"] for event in r.jobs.get(second["id"], "owner")["events"]]
    assert set(progress) <= {0, 100} and progress[-1] == 100
    await r.jobs.close()


@pytest.mark.asyncio
async def test_repeated_task_cancellation_cannot_release_gpu_early(runtime):
    r = runtime
    first = await r.jobs.submit(r.engine, r.call, "owner", "music", {})
    await until(lambda: r.started.is_set())
    task = r.jobs.by_id[first["id"]]
    r.jobs.cancel(first["id"], "owner")
    await asyncio.sleep(0.02)
    task.cancel()
    await asyncio.sleep(0.02)
    assert not task.done() and r.running == 1
    r.release.set()
    await until(lambda: not r.jobs.tasks)
    assert r.jobs.get(first["id"], "owner")["status"] == "cancelled"
    await r.jobs.close()


@pytest.mark.asyncio
async def test_shutdown_stops_backend_before_draining_native_jobs(runtime):
    r = runtime
    job = await r.jobs.submit(r.engine, r.call, "owner", "music", {})
    await until(lambda: r.started.is_set())
    order = []

    async def stop():
        order.append("stop-owned-backend")
        r.release.set()

    await r.jobs.close(stop_backends=stop)
    assert order == ["stop-owned-backend"] and not r.jobs.tasks and r.running == 0
    assert r.jobs.get(job["id"], "owner")["status"] == "cancelled"
    with pytest.raises(Exception, match="shutting down"):
        await r.jobs.submit(r.engine, r.call, "owner", "music", {})


@pytest.mark.asyncio
async def test_partial_dual_device_lock_is_released_on_cancel(routing_registry):
    registry = routing_registry(
        [("qwen-text", [0, 1]), ("qwen-embedding", [0]), ("qwen-decision", [1])]
    )
    dual = SimpleNamespace(registry=registry, binding=registry.assignment("qwen-text"))
    single = SimpleNamespace(registry=registry, binding=registry.assignment("qwen-embedding"))
    lock = device_locks(dual)[1]
    lock.acquire()
    reached = []

    async def hold():
        async with async_execution_scope(dual):
            reached.append("dual")

    task = asyncio.create_task(hold())
    try:
        await asyncio.sleep(0.03)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with asyncio.timeout(1):
            async with async_execution_scope(single):
                reached.append("single")
        assert reached == ["single"]
    finally:
        lock.release()


@pytest.mark.asyncio
async def test_artifact_expiration_integrity_owner_and_cancel_terminal(runtime):
    r = runtime
    r.release.set()
    job = await r.jobs.submit(r.engine, r.call, "owner", "music", {})
    await until(lambda: not r.jobs.tasks)
    assert (await r.jobs.content(job["id"], "owner"))[0] == wave_bytes()
    with pytest.raises(NotFoundError):
        r.jobs.cancel(job["id"], "other-owner")
    assert r.jobs.cancel(job["id"], "owner")["status"] == "completed"
    row = r.jobs.get(job["id"], "owner")
    path = r.jobs.scratch.get_artifact_path(job["id"], row["files"][0]["filename"])
    path.write_bytes(b"corrupt")
    with pytest.raises(Exception, match="integrity"):
        await r.jobs.content(job["id"], "owner")
    row["expires_at"] = 0
    with pytest.raises(NotFoundError):
        r.jobs.get(job["id"], "owner")
    assert r.jobs.list("owner") == []
    await r.jobs.close()


@pytest.mark.asyncio
async def test_backend_failure_never_completes_job(runtime):
    r = runtime
    r.fail = True
    r.release.set()
    job = await r.jobs.submit(r.engine, r.call, "owner", "music", {})
    await until(lambda: not r.jobs.tasks)
    row = r.jobs.get(job["id"], "owner")
    assert row["status"] == "failed" and row["progress"] == 0 and row["files"] == []
    with pytest.raises(ConflictError):
        await r.jobs.content(job["id"], "owner")
    await r.jobs.close()


@pytest.mark.asyncio
async def test_unvalidated_registry_is_not_authorized_by_local_runtime(runtime, monkeypatch):
    r = runtime

    def deny(*args):
        raise ModelNotLoadedError("No native V3/V4 evidence")

    monkeypatch.setattr(r.engine.registry, "require_active", deny)
    with pytest.raises(ModelNotLoadedError):
        await r.jobs.submit(r.engine, r.call, "owner", "music", {})
    assert r.calls == 0 and r.jobs.rows == {}
    await r.jobs.close()


@pytest.mark.asyncio
async def test_shutdown_during_admission_cannot_create_late_job(runtime, monkeypatch):
    r = runtime
    entered = threading.Event()
    release = threading.Event()

    def blocked(*args):
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr("factory.apps.control_api.src.api.public.jobs.admit", blocked)
    task = asyncio.create_task(r.jobs.submit(r.engine, r.call, "owner", "music", {}))
    await until(entered.is_set)
    await r.jobs.close()
    release.set()
    with pytest.raises(Exception, match="shutting down"):
        await task
    assert not r.jobs.rows and not r.jobs.tasks and r.calls == 0


@pytest.mark.asyncio
async def test_local_job_http_management_and_events(runtime, monkeypatch, tmp_path):
    import httpx
    from factory.apps.control_api.src.main import create_app

    r = runtime
    monkeypatch.setenv("FACTORY_API_KEY", "local-job-contract-key")
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("SCRATCH_ARTIFACTS_DIR", str(tmp_path / "app-scratch"))
    app = create_app()
    async with app.router.lifespan_context(app):
        app.state.public_media_jobs = r.jobs
        job = await r.jobs.submit(r.engine, r.call, ":system", "music", {})
        await until(r.started.is_set)
        headers = {"Authorization": "Bearer local-job-contract-key"}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test", headers=headers
        ) as client:
            listing = (await client.get("/v1/jobs")).json()["data"]
            assert (
                len(listing) == 1 and listing[0]["id"] == job["id"] and "service" not in listing[0]
            )
            response = await client.post("/v1/jobs/" + job["id"] + "/cancel")
            assert response.status_code == 200 and response.json()["status"] == "cancelling"
            r.release.set()
            await until(lambda: not r.jobs.tasks)
            assert (await client.get("/v1/jobs/" + job["id"])).json()["status"] == "cancelled"
            assert (await client.get("/v1/jobs/" + job["id"] + "/content")).status_code == 409
            events = await client.get("/v1/jobs/" + job["id"] + "/events")
            assert (
                events.status_code == 200
                and "event: job.status" in events.text
                and "cancelled" in events.text
            )
            assert (await client.post("/v1/workers/lease", json={})).status_code == 404
        await r.jobs.close()


def test_retained_legacy_host_refuses_production(monkeypatch):
    from fastapi.testclient import TestClient
    from factory.apps.control_api.src.legacy import app

    monkeypatch.setenv("FACTORY_API_KEY", "legacy-rejection-contract-key")
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(RuntimeError, match="migration"):
        with TestClient(app):
            pass
