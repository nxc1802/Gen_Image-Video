"""Phase 3 configuration/software checks. No GPU inference or readiness proof."""
import hashlib
import json
from pathlib import Path
import socket
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError
import yaml

from factory_contracts.deployment import DeploymentConfig, DeploymentEntry
from factory_contracts.errors import InvalidRequestError, ModelNotLoadedError
from factory_contracts.jobs.job_contract import JobSubmissionRequest
from factory_contracts.jobs.worker_contract import WorkerLeaseRequest
from model_registry.cli import factory_main
from model_registry.deployment import ModelCatalog, DeploymentRegistry, read_config
from model_registry.deployment_validator import DeploymentValidator, EnvironmentInventory, verify_weights

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("example", ["dual-t4-shared", "dual-t4-exclusive"])
def test_example_placements_are_configuration_only(example):
    path = ROOT / f"deployments/examples/{example}.yaml"
    registry = DeploymentRegistry(read_config(path), ModelCatalog(ROOT / "manifests"), config_path=path)
    report = DeploymentValidator(registry).validate(config_only=True)
    assert report["configuration_valid"]
    assert not report["preflight_passed"] and not report["production_ready"]
    assert not registry.active_models()


def test_cli_materializes_both_exit_gate_layouts_without_python_changes(tmp_path, capsys):
    path = tmp_path / "session.yaml"
    base = ["--manifests-dir", str(ROOT / "manifests"), "--deployment", str(path)]
    def run(*args):
        code = factory_main([*base, *args])
        captured = capsys.readouterr()
        return code, json.loads(captured.out) if captured.out else json.loads(captured.err)
    assert run("deployment", "init")[0] == 0
    for model, gpu in [("qwen-decision", "0"), ("qwen-embedding", "0"), ("qwen-reranker", "1")]:
        assert run("deployment", "add", model, "--gpu", gpu)[0] == 0
    assert run("deployment", "validate", "--config-only")[1]["configuration_valid"]
    before = path.read_bytes()
    # Oversubscribing GPU 0 cannot partially modify the configuration.
    assert run("deployment", "add", "qwen-image-2.1", "--gpu", "0")[0] == 2
    assert path.read_bytes() == before
    assert run("deployment", "init")[0] == 1  # No overwrite.
    for model in ["qwen-decision", "qwen-embedding", "qwen-reranker"]:
        assert run("deployment", "remove", model)[0] == 0
    assert run("deployment", "add", "qwen-text", "--gpu", "0,1", "--exclusive")[0] == 0
    config = read_config(path)
    assert config.deployments[0].devices == (0, 1)
    assert config.deployments[0].backend == "llama_cpp"
    assert run("deployment", "validate", "--config-only")[1]["configuration_valid"]
    assert not run("status")[1]["running_session_observed"]


@pytest.mark.parametrize("entries,failed_check", [
    ([("qwen-image-2.1", [2], None, False)], "gpu_ids"),
    ([("qwen-image-2.1", [0], "vllm", False)], "backend_selection"),
    ([("qwen-text", [0], None, True)], "gpu_count"),
    ([("qwen-text", [0, 1], None, True), ("qwen-embedding", [0], None, False)], "exclusive_conflict"),
    ([("qwen-image-2.1", [0], None, False), ("qwen-embedding", [0], None, False)], "vram_budget"),
    ([("yue2", [0], None, False)], "model_disposition"),
    ([("enhancement-suite-v1", [0], None, False)], "model_disposition"),
])
def test_invalid_layouts_are_rejected(tmp_path, entries, failed_check):
    config = DeploymentConfig(deployments=tuple(DeploymentEntry(model=mid, devices=tuple(devices), backend=backend,
        exclusive=exclusive, port=9400+i) for i, (mid, devices, backend, exclusive) in enumerate(entries)))
    registry = DeploymentRegistry(config, ModelCatalog(ROOT / "manifests"), config_path=tmp_path / "config.yaml")
    report = DeploymentValidator(registry).validate(config_only=True)
    assert not report["configuration_valid"]
    assert any(item["check"] == failed_check for item in report["errors"])


@pytest.mark.parametrize("data", [
    {"model": "qwen-text", "devices": [0, 1]},
    {"model": "qwen-embedding", "devices": [True]},
    {"model": "qwen-embedding", "devices": [0, 0]},
    {"model": "qwen-embedding", "devices": [0], "ready": True},
])
def test_configuration_cannot_claim_activation_or_implicit_devices(data):
    with pytest.raises(ValidationError):
        DeploymentEntry.model_validate(data)


def test_ports_and_resolved_aliases_cannot_collide(tmp_path):
    catalog = ModelCatalog(ROOT / "manifests")
    config = DeploymentConfig(deployments=(DeploymentEntry(model="qwen-embedding", devices=(0,), port=9000),
        DeploymentEntry(model="qwen-reranker", devices=(1,), port=9000)))
    registry = DeploymentRegistry(config, catalog, config_path=tmp_path / "config.yaml")
    assert any(error["check"] == "backend_port" for error in DeploymentValidator(registry).validate(config_only=True)["errors"])
    config = DeploymentConfig(deployments=(DeploymentEntry(model="qwen-image", devices=(0,)),
        DeploymentEntry(model="qwen-image-2.1", devices=(1,))))
    with pytest.raises(ValueError, match="Duplicate"):
        DeploymentRegistry(config, catalog, config_path=tmp_path / "config.yaml")


@pytest.fixture
def tiny_contract_registry(tmp_path):
    # These bytes are a test file, not model weights/inference evidence.
    contents = b"contract-only file identity"
    data = yaml.safe_load((ROOT / "manifests/models/qwen-embedding.yaml").read_text())
    data["metadata"]["id"] = "contract-model"
    data["aliases"] = []
    spec = data["weights"]["files"]["primary"]
    spec.update(path="contract.dat", destination="contract.dat", sha256=hashlib.sha256(contents).hexdigest(), size_bytes=len(contents))
    data["weights"]["files"] = {"primary": spec}
    directory = tmp_path / "catalog/models"
    directory.mkdir(parents=True)
    (directory / "contract.yaml").write_text(yaml.safe_dump(data))
    topology = tmp_path / "catalog/hardware"
    topology.mkdir()
    (topology / "kaggle.yaml").write_bytes((ROOT / "manifests/hardware/kaggle-dual-t4.yaml").read_bytes())
    store = tmp_path / "store/contract-model"
    store.mkdir(parents=True)
    (store / "contract.dat").write_bytes(contents)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        api_port = sock.getsockname()[1]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = DeploymentConfig(models_dir=tmp_path / "store", api_port=api_port,
        deployments=(DeploymentEntry(model="contract-model", devices=(0,), port=port),))
    return DeploymentRegistry(config, ModelCatalog(directory.parent), config_path=tmp_path / "config.yaml")


def test_preflight_integrity_is_not_inference_or_activation(tiny_contract_registry):
    registry = tiny_contract_registry
    inventory = EnvironmentInventory(gpus={0:{"name":"test Tesla T4", "total_vram_gb":15}, 1:{"name":"test Tesla T4", "total_vram_gb":15}},
        backends=frozenset({"vllm"}), adapters=frozenset({"qwen_embedding"}))
    report = DeploymentValidator(registry).validate(inventory=inventory)
    assert report["preflight_passed"]  # Mocked inventory: software contract only.
    assert not report["runtime_load_validated"] and not report["production_ready"]
    assert not registry.active_models()
    with pytest.raises(ModelNotLoadedError):
        registry.require_active("contract-model")
    binding = registry.assignment("contract-model")
    path = binding.weights_dir / "contract.dat"
    path.write_bytes(b"x" * path.stat().st_size)
    assert not verify_weights(binding, registry.catalog.get("contract-model"))["passed"]
    path.unlink()
    assert not verify_weights(binding, registry.catalog.get("contract-model"))["passed"]


def test_missing_environment_and_providers_cannot_pass(tiny_contract_registry):
    report = DeploymentValidator(tiny_contract_registry).validate(inventory=EnvironmentInventory())
    assert not report["preflight_passed"]
    assert {item["check"] for item in report["errors"]} >= {"gpu_inventory", "backend_installed", "adapter_available"}


@pytest.mark.asyncio
async def test_queue_uses_configured_gpu_and_worker_cannot_override(routing_registry):
    from factory.apps.control_api.src.queue.repository.memory import InMemoryJobStore
    from factory.apps.control_api.src.queue.scheduler import InMemoryScheduler
    from factory.apps.control_api.src.queue.manager import JobLifecycleManager
    from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
    from factory.apps.control_api.src.streaming.manager import StreamManager
    registry = routing_registry([("qwen-image-2.1", [1])])
    repo, scheduler = InMemoryJobStore(), InMemoryScheduler()
    await repo.initialize()
    await scheduler.initialize()
    manager = JobLifecycleManager(repo, scheduler, StreamManager(repository=repo), deployment_registry=registry)
    coordinator = LeaseCoordinator(repo, scheduler, manifest_loader=registry.catalog.loader, deployment_registry=registry)
    try:
        job, _ = await manager.submit_job(JobSubmissionRequest(service="image", model="qwen-image-2.1", task="t2i", inputs={}), "req-placement")
        assert job.devices == [1] and scheduler._gpu_queues[1]
        request = dict(worker_id="placement-contract", subsystem="image", supported_models=[job.model], supported_tasks=[job.task])
        assert (await coordinator.lease_job(WorkerLeaseRequest(**request, gpu_device=0))).job is None
        lease = await coordinator.lease_job(WorkerLeaseRequest(**request, gpu_device=1))
        assert lease.job.worker.gpu_device == 1
        assert scheduler.is_gpu_busy(1) and not scheduler.is_gpu_busy(0)
        await manager.cancel_job(job.job_id)
        assert not scheduler.is_gpu_busy(1)
    finally:
        await repo.close()
        await scheduler.close()


def test_empty_registry_and_stale_session_fail_closed(routing_registry):
    from factory.apps.control_api.src.queue.policy import binding_for_job
    from factory_contracts.jobs.job_contract import FactoryJob
    first = routing_registry([("qwen-embedding", [0])])
    second = routing_registry([("qwen-embedding", [0])])
    job = FactoryJob(job_id="job-00000000-0000-0000-0000-000000000001", request_id="req-00000000-0000-0000-0000-000000000001",
        service="embedding", model="qwen-embedding", task="embedding", inputs={}, devices=[0],
        deployment_session_id=first.session_id, deployment_config_sha256=first.config_sha256)
    with pytest.raises(ModelNotLoadedError, match="earlier runtime session"):
        binding_for_job(second, job)
    with pytest.raises(ModelNotLoadedError):
        first.assignment("qwen-text")


def test_public_discovery_never_promotes_configured_catalog_models(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from factory.apps.control_api.src.main import app
    from model_registry.deployment import write_config
    path = tmp_path / "config.yaml"
    write_config(path, DeploymentConfig(deployments=(DeploymentEntry(model="qwen-image-2.1", devices=(1,), port=9500),)), create=True)
    monkeypatch.setenv("FACTORY_MANIFESTS_DIR", str(ROOT / "manifests"))
    monkeypatch.setenv("FACTORY_DEPLOYMENT_FILE", str(path))
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("FACTORY_API_KEY", "deployment-contract-key")
    headers = {"Authorization": "Bearer deployment-contract-key"}
    with TestClient(app) as client:
        assert client.get("/v1/models", headers=headers).json() == {"object": "list", "data": []}
        catalog = client.get("/v1/registry/models", headers=headers).json()
        assert catalog["total"] == 20
        status = client.get("/v1/registry/deployment", headers=headers).json()
        assert status["configured"]["deployments"][0]["devices"] == [1]
        assert not status["active_models"] and not status["production_ready"]
        payload = {"model": "qwen-image-2.1", "prompt": "a cat"}
        result = client.post("/v1/images/generations", headers=headers, json=payload)
        assert result.status_code == 503
        assert result.json()["error"]["code"] == "MODEL_NOT_LOADED"
        payload["gpu"] = 0
        assert client.post("/v1/images/generations", headers=headers, json=payload).status_code == 400


def test_serve_cannot_skip_lifecycle_even_after_software_preflight(tmp_path, monkeypatch, capsys):
    from model_registry.deployment import write_config
    import uvicorn
    path = tmp_path / "config.yaml"
    write_config(path, DeploymentConfig(deployments=(DeploymentEntry(model="qwen-embedding", devices=(0,)),)), create=True)
    monkeypatch.setattr(DeploymentValidator, "validate", lambda *a, **k: {"mode": "local_environment", "preflight_passed": True, "production_ready": False})
    monkeypatch.setenv("FACTORY_API_KEY", "lifecycle-contract-key")
    def startup_only(app, **kwargs):
        import asyncio
        async def run():
            async with app.router.lifespan_context(app):
                raise AssertionError("API entered lifespan with no healthy backend")
        asyncio.run(run())
    monkeypatch.setattr(uvicorn, "run", startup_only)
    with pytest.raises(RuntimeError, match="No configured backend"):
        factory_main(["--deployment", str(path), "serve"])


@pytest.mark.asyncio
async def test_dual_gpu_binding_is_reserved_atomically(routing_registry):
    from factory.apps.control_api.src.queue.repository.memory import InMemoryJobStore
    from factory.apps.control_api.src.queue.scheduler import InMemoryScheduler
    from factory.apps.control_api.src.queue.manager import JobLifecycleManager
    from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
    from factory.apps.control_api.src.streaming.manager import StreamManager
    registry = routing_registry([("qwen-text", [0, 1])])
    repo, scheduler = InMemoryJobStore(), InMemoryScheduler()
    await repo.initialize()
    await scheduler.initialize()
    manager = JobLifecycleManager(repo, scheduler, StreamManager(repository=repo), deployment_registry=registry)
    coordinator = LeaseCoordinator(repo, scheduler, manifest_loader=registry.catalog.loader, deployment_registry=registry)
    try:
        job, _ = await manager.submit_job(JobSubmissionRequest(service="text", model="qwen-text", task="chat", inputs={}), "req-dual-contract")
        assert job.devices == [0, 1]
        request = WorkerLeaseRequest(worker_id="dual-contract", subsystem="text", supported_models=[job.model], supported_tasks=[job.task], gpu_device=0)
        assert scheduler.acquire_gpu(1, "other-contract-job")
        assert (await coordinator.lease_job(request)).job is None
        assert not scheduler.is_gpu_busy(0)
        scheduler.release_gpu(1, "other-contract-job")
        assert (await coordinator.lease_job(request)).job is not None
        assert scheduler.is_gpu_busy(0) and scheduler.is_gpu_busy(1)
        assert not registry.active_models()  # Locks are software, not GPU/model evidence.
        await manager.cancel_job(job.job_id)
        assert not scheduler.is_gpu_busy(0) and not scheduler.is_gpu_busy(1)
    finally:
        await repo.close()
        await scheduler.close()


@pytest.mark.asyncio
async def test_idempotency_key_cannot_return_a_different_model(routing_registry):
    from factory.apps.control_api.src.queue.repository.memory import InMemoryJobStore
    from factory.apps.control_api.src.queue.broker.memory_broker import MemoryQueueBroker
    from factory.apps.control_api.src.queue.manager import JobLifecycleManager
    from factory.apps.control_api.src.streaming.manager import StreamManager
    registry = routing_registry([("qwen-embedding", [0]), ("qwen-decision", [1])])
    repo, broker = InMemoryJobStore(), MemoryQueueBroker()
    await repo.initialize()
    await broker.initialize()
    manager = JobLifecycleManager(repo, broker, StreamManager(repository=repo), deployment_registry=registry)
    try:
        await manager.submit_job(JobSubmissionRequest(service="embedding", model="qwen-embedding", task="embedding", inputs={}, idempotency_key="same-key"), "req-first")
        with pytest.raises(InvalidRequestError, match="different explicit model"):
            await manager.submit_job(JobSubmissionRequest(service="decision", model="qwen-decision", task="predicate", inputs={}, idempotency_key="same-key"), "req-second")
    finally:
        await repo.close()
        await broker.close()


def test_direct_production_startup_requires_explicit_deployment(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from factory.apps.control_api.src.main import app
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("FACTORY_API_KEY", "startup-contract-key")
    monkeypatch.setenv("FACTORY_DEPLOYMENT_FILE", str(tmp_path / "missing.yaml"))
    monkeypatch.setenv("FACTORY_MANIFESTS_DIR", str(ROOT / "manifests"))
    with pytest.raises(RuntimeError, match="Explicit deployment"):
        with TestClient(app):
            pass


def test_legacy_runtime_contract_is_not_a_production_readiness_path(legacy_routing, monkeypatch):
    import asyncio
    from factory.apps.control_api.src.queue.model_runtime import ModelRuntimeManager
    monkeypatch.setenv("ENVIRONMENT", "production")
    manager = ModelRuntimeManager()
    model = ModelCatalog(legacy_routing).get("qwen-image-2.1")
    with pytest.raises(ModelNotLoadedError, match="Legacy load contracts"):
        asyncio.run(manager.ensure_loaded(model, gpu_ids=[0]))
