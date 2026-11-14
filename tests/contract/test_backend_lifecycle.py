"""Lifecycle contracts. GPU checks are mocked explicitly; no inference evidence."""
import asyncio
import hashlib
import os
from pathlib import Path
import socket
import sys
import time
from unittest.mock import MagicMock

import psutil
import pytest
import yaml

from factory_contracts.deployment import BackendOptions, DeploymentConfig, DeploymentEntry
from model_registry.deployment import DeploymentRegistry, ModelCatalog
from model_registry.backends.process import BackendError, LaunchPlan, ProcessBackend
from model_registry.backends.session import BackendSession

pytestmark = pytest.mark.contract_mock
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def process_backend(tmp_path, monkeypatch):
    data = yaml.safe_load((ROOT / "manifests/models/qwen-embedding.yaml").read_text())
    data["metadata"]["id"] = "lifecycle-contract"
    data["aliases"] = []
    spec = data["weights"]["files"]["primary"]
    contents = b"contract file, never model weights"
    spec.update(path="contract.dat", destination="model/contract.dat", sha256=hashlib.sha256(contents).hexdigest(), size_bytes=len(contents))
    data["weights"]["files"] = {"primary":spec}
    catalog = tmp_path / "catalog/models"
    catalog.mkdir(parents=True)
    (catalog / "contract.yaml").write_text(yaml.safe_dump(data))
    store = tmp_path / "models/lifecycle-contract/model"
    store.mkdir(parents=True)
    (store / "contract.dat").write_bytes(contents)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = DeploymentConfig(models_dir=tmp_path / "models", deployments=(DeploymentEntry(model="lifecycle-contract", backend="vllm", devices=(0,), port=port,
        backend_options=BackendOptions(startup_timeout_seconds=5, stop_timeout_seconds=0.2)),))
    registry = DeploymentRegistry(config, ModelCatalog(catalog.parent), config_path=tmp_path / "config.yaml")
    class Adapter:
        stubborn = False
        def plan(self, backend, view):
            args = [sys.executable, str(ROOT / "tests/fixtures/backend_process_fixture.py"), "--port", str(port),
                "--model", backend.binding.model, "--nonce", backend.boot_nonce, "--graph", backend.binding.checkpoint_graph_sha256,
                "--token_file", str(backend.directory / "backend.key")]
            if self.stubborn:
                args += ["--stubborn-child"]
            return LaunchPlan(tuple(args), backend.directory, health_kind="native")
    backend = ProcessBackend(registry, registry.bindings()[0], Adapter(), tmp_path / "session")
    # CPU process-control test only. This must never enter a GPU evidence report.
    monkeypatch.setattr(backend, "_gpu_ownership", lambda owned: True)
    yield backend
    backend.stop()


def wait_health(backend):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if backend.health():
            return
        time.sleep(0.02)
    raise AssertionError("CPU contract fixture did not become healthy")


def test_owned_process_prepare_launch_health_stop_does_not_certify(process_backend):
    backend = process_backend
    backend.prepare()
    backend.launch()
    wait_health(backend)
    info = backend.runtime_info()
    assert info["pid"] == backend.process.pid and info["state"] == "HEALTHY"
    assert info["weights_verified"] and info["executable_sha256"]
    assert not info["production_ready"] and not info["inference_verified"]
    assert not backend.registry.active_models()
    backend.stop()
    assert backend.process.poll() is not None and not backend.health()


def test_checkpoint_integrity_failure_never_launches(process_backend):
    backend = process_backend
    path = backend.binding.weights_dir / "model/contract.dat"
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(BackendError, match="integrity"):
        backend.prepare()
    assert backend.process is None


def test_http_200_without_listener_ownership_is_not_health(process_backend, monkeypatch):
    backend = process_backend
    backend.prepare()
    backend.launch()
    wait_health(backend)
    monkeypatch.setattr(backend, "_listener_owned", lambda owned: False)
    assert not backend.health()
    assert not backend.runtime_info()["model_loaded"]


def test_stop_before_prepare_cannot_launch_late(process_backend):
    backend = process_backend
    backend.stop()
    with pytest.raises(RuntimeError, match="cancelled"):
        backend.prepare()
    assert backend.process is None
    assert not backend.health()


def test_checkpoint_mutation_revokes_health(process_backend):
    backend = process_backend
    backend.prepare()
    backend.launch()
    wait_health(backend)
    path = backend.binding.weights_dir / "model/contract.dat"
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert not backend.health()
    assert backend.state != "HEALTHY"


def test_stop_reaps_stubborn_owned_child(process_backend):
    backend = process_backend
    backend.adapter.stubborn = True
    backend.prepare()
    backend.launch()
    wait_health(backend)
    children = psutil.Process(backend.process.pid).children(recursive=True)
    assert children
    backend.stop()
    assert backend.process.poll() is not None
    assert all(not child.is_running() or child.status() == psutil.STATUS_ZOMBIE for child in children)


@pytest.mark.asyncio
async def test_startup_failure_isolated_and_health_loss_never_falls_back(routing_registry, monkeypatch):
    registry = routing_registry([("qwen-embedding", [0]), ("qwen-decision", [1])])
    calls = []
    class FakeBackend:
        def __init__(self, registry, binding, adapter, directory):
            self.binding, self.state, self.error = binding, "CONFIGURED", None
            self.process = None
            self.ok = True
        def prepare(self):
            calls.append((self.binding.model, self.binding.backend))
            if self.binding.model == "qwen-decision":
                raise BackendError("selected backend load failure")
        def launch(self):
            self.process = MagicMock()
            self.process.poll.return_value = None
        def health(self):
            return self.ok
        def stop(self):
            self.state = "STOPPED"
            if self.process:
                self.process.poll.return_value = 0
        def runtime_info(self):
            return {"state":self.state, "production_ready":False, "inference_verified":False}
    monkeypatch.setattr("model_registry.backends.session.selected_plugin", lambda group, name, **kwargs: FakeBackend if group == "factory.backends" else lambda: object())
    session = BackendSession(registry)
    try:
        await session.start()
        assert session.healthy_models() == ("qwen-embedding",)
        assert "qwen-decision" in session.failures
        assert calls == [("qwen-embedding", "vllm"), ("qwen-decision", "vllm")]
        session.backends["qwen-embedding"].ok = False
        await session.check_health()
        assert not session.healthy_models()
        assert len(calls) == 2  # No retry, restart, swap, or alternate backend.
        assert not registry.active_models()
    finally:
        await session.stop()
    import json
    receipt = json.loads((session.directory / "shutdown.json").read_text())
    assert receipt["stop_completed"] and not receipt["errors"]
    assert receipt["session_id"] == registry.session_id
    assert all(row["state"] == "STOPPED" for row in receipt["backends"].values())
    assert not receipt["production_ready"]


def test_dead_process_is_not_reported_as_loaded_before_monitor_tick(process_backend):
    backend = process_backend
    backend.prepare()
    backend.launch()
    wait_health(backend)
    session = BackendSession(backend.registry)
    session.backends[backend.binding.model] = backend
    session._healthy.add(backend.binding.model)
    backend.process.kill()
    backend.process.wait(timeout=5)
    status = session.status()
    assert status["healthy_backends"] == []
    assert status["backends"][backend.binding.model]["state"] == "FAILED"
    assert not status["backends"][backend.binding.model]["model_loaded"]
    asyncio.run(session.check_health())
    assert backend.state == "STOPPED"
    assert not session._healthy


@pytest.mark.parametrize("path,payload", [
    ("/v1/chat/completions", {"model":"qwen-embedding", "messages":[{"role":"user", "content":"authorization test"}]}),
    ("/v1/responses", {"model":"qwen-embedding", "input":"authorization test"}),
])
def test_production_chat_cannot_bypass_session_authorization(routing_registry, monkeypatch, path, payload):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from factory.apps.control_api.src.api.routers import chat
    from factory.apps.control_api.src.api.errors import register_exception_handlers
    from factory_auth import verify_bearer_token
    app = FastAPI()
    app.state.auth_config = SimpleNamespace(environment="production")
    app.state.deployment_registry = routing_registry([("qwen-embedding", [0])])
    app.dependency_overrides[verify_bearer_token] = lambda: True
    app.include_router(chat.router)
    register_exception_handlers(app)
    def forbidden(*args, **kwargs):
        pytest.fail("Unvalidated request reached a legacy upstream engine")
    for name in ["_get_text_engine", "_get_vlm_engine"]:
        monkeypatch.setattr(chat, name, forbidden)
    with TestClient(app) as client:
        response = client.post(path, json=payload)
    assert response.status_code == 503


@pytest.mark.parametrize("device_type,device_index,dtype", [("cpu",None,"fp16"),("cuda",1,"fp16"),("cuda",0,"fp32")])
def test_native_asr_refuses_sdk_device_or_precision_substitution(monkeypatch, device_type, device_index, dtype):
    # Deliberately mocked SDK/tensor metadata; no tensor, GPU or inference runs.
    from types import SimpleNamespace
    from model_registry.backends.native_server import load_sdk
    parameter = SimpleNamespace(device=SimpleNamespace(type=device_type,index=device_index),dtype=dtype,is_floating_point=lambda:True)
    loaded = SimpleNamespace(backend="transformers",model=SimpleNamespace(parameters=lambda:iter([parameter])))
    cuda = SimpleNamespace(is_available=lambda:True,set_device=lambda value:None,init=lambda:None)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=cuda,float16="fp16",bfloat16="bf16",float32="fp32"))
    monkeypatch.setitem(sys.modules,"qwen_asr",SimpleNamespace(Qwen3ASRModel=SimpleNamespace(from_pretrained=lambda *args,**kwargs:loaded)))
    with pytest.raises(RuntimeError,match="resident|precision"):
        load_sdk({"sdk":"qwen_asr","adapter":"qwen_asr","model_path":"/contract-only/not-weights","precision":"fp16"})


@pytest.mark.asyncio
async def test_resident_budget_violation_stops_backend_with_precise_reason(routing_registry):
    """Mock residency only; proves fail-closed supervision, never GPU inference."""
    from types import SimpleNamespace
    registry = routing_registry([('qwen-embedding',[0])])
    session = BackendSession(registry)
    backend = SimpleNamespace(error=None,health=MagicMock(return_value=True),stop=MagicMock(),
                              runtime_info=lambda:{'gpu_processes':[{'pid':123,'device':0,'used_memory_mb':14541.0}]})
    session.backends['qwen-embedding'] = backend
    session._healthy.add('qwen-embedding')
    await session.check_health()
    assert not session._healthy
    backend.stop.assert_called_once()
    assert session.failures['qwen-embedding']['type'] == 'BACKEND_UNHEALTHY'
    assert session.failures['qwen-embedding']['message'] == 'Observed resident GPU memory exceeds the selected hardware budget'
    assert not registry.active_models()
