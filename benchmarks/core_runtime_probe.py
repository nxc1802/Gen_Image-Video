"""CPU-only local HTTP composition probe, with explicit fake adapter/authorization.

Executed in a minimal core-only venv by the Phase 7 gate. It cannot certify any
model/capability. Production modules contain no fixture or activation switch.
"""

import argparse
import asyncio
import hashlib
import importlib.abc
import importlib.metadata
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import urllib.request
import urllib.error
import wave

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FORBIDDEN = {
    "redis",
    "boto3",
    "botocore",
    "sqlalchemy",
    "alembic",
    "aiosqlite",
    "asyncpg",
    "psycopg",
    "psycopg2",
    "jwt",
    "cryptography",
}


class BlockInfrastructure(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise AssertionError("Core imported optional infrastructure: " + fullname)


def run():
    sys.meta_path.insert(0, BlockInfrastructure())
    from factory.apps.control_api.src.main import create_app
    from factory.apps.control_api.src.api.routers import inference
    from factory.core.model_adapter import ModelAdapter, PreparedCall
    from factory.core.media import artifact
    from factory_contracts.deployment import DeploymentConfig, DeploymentEntry
    from model_registry.deployment import ModelCatalog, DeploymentRegistry
    import uvicorn

    installed = {
        d.metadata["Name"].lower().replace("_", "-"): d.version
        for d in importlib.metadata.distributions()
    }
    assert not {
        "boto3",
        "botocore",
        "redis",
        "sqlalchemy",
        "alembic",
        "pyjwt",
        "cryptography",
        "aiosqlite",
    }.intersection(installed)
    with tempfile.TemporaryDirectory(prefix="factory-core-runtime-") as directory:
        os.environ.update(
            ENVIRONMENT="test",
            FACTORY_API_KEY="core-software-contract-key",
            FACTORY_MANIFESTS_DIR=str(ROOT / "manifests"),
            SCRATCH_ARTIFACTS_DIR=directory,
            REDIS_URL="redis://127.0.0.1:1",
            DATABASE_URL="postgresql://127.0.0.1:1/unreachable",
            S3_ENDPOINT_URL="http://127.0.0.1:1",
        )
        config = DeploymentConfig(
            deployments=(DeploymentEntry(model="ace-step-1.5", devices=(0,)),)
        )
        registry = DeploymentRegistry(
            config, ModelCatalog(ROOT / "manifests"), config_path=Path(directory) / "explicit.yaml"
        )
        app = create_app()
        app.state.launch_registry = registry
        binding = registry.assignment("ace-step-1.5")
        session = SimpleNamespace(
            registry=registry,
            backends={
                binding.model: SimpleNamespace(
                    registry=registry, binding=binding, health=lambda: True
                )
            },
            healthy_models=lambda: (binding.model,),
        )
        output = io.BytesIO()
        with wave.open(output, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(8000)
            f.writeframes(b"\x01\0" * 80)
        raw = output.getvalue()
        fixture_calls = []

        class ContractAdapter(ModelAdapter):
            capabilities = frozenset({"music.text_to_music"})
            supported_backends = frozenset({"acestep_native"})

            def validate_request(self, model, request, capability):
                pass

            def prepare(self, context, request, capability):
                return PreparedCall(capability, {})

            def infer(self, context, prepared):
                fixture_calls.append("explicit-software-double")
                return {
                    "model": context.binding.model,
                    "artifacts": [artifact(raw, "audio", "audio/wav")],
                }

            def normalize_output(self, context, prepared, result):
                return result

        original = inference.engine
        inference.engine = lambda request, cls: cls(
            registry, session, adapter_loader=lambda name: ContractAdapter()
        )
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(16)
        port = listener.getsockname()[1]
        native_connect = socket.socket.connect

        def local_only(sock, address):
            if not isinstance(address, tuple) or address[0] != "127.0.0.1" or address[1] != port:
                raise AssertionError("Unexpected infrastructure/network connection")
            return native_connect(sock, address)

        socket.socket.connect = local_only
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        checks = {}
        try:
            deadline = time.monotonic() + 10
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started

            def request(path, body=None):
                data = json.dumps(body).encode() if body is not None else None
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}" + path,
                    data=data,
                    headers={
                        "Authorization": "Bearer core-software-contract-key",
                        "Content-Type": "application/json",
                    },
                )
                try:
                    with urllib.request.urlopen(req, timeout=10) as response:
                        return response.status, response.read()
                except urllib.error.HTTPError as exc:
                    return exc.code, exc.read()

            status, body = request("/v1/registry/runtime")
            runtime = json.loads(body)
            checks["real_core_startup_without_optional_packages"] = (
                status == 200
                and runtime["legacy_components"]
                == runtime["loaded_optional_infrastructure"]
                == runtime["external_services_required"]
                == []
            )
            checks["legacy_worker_routes_not_mounted"] = request("/v1/workers/lease", {})[0] == 404
            checks["public_discovery_stays_empty"] = json.loads(request("/v1/models")[1]) == {
                "object": "list",
                "data": [],
            }
            call = {
                "model": binding.model,
                "capability": "music.text_to_music",
                "inputs": {"prompt": "Explicit software probe only"},
            }
            checks["unvalidated_inference_rejected"] = (
                request("/v1/music", call)[0] == 503 and fixture_calls == []
            )
            # Local test-only replacement, never a production configuration flag.
            registry.require_active = lambda *args: None
            status, body = request("/v1/music", call)
            assert status == 200, body
            identifier = json.loads(body)["id"]
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                row = json.loads(request("/v1/jobs/" + identifier)[1])
                if row["status"] in {"failed", "completed"}:
                    break
                time.sleep(0.01)
            checks["canonical_async_job_path_software_only"] = (
                row["status"] == "completed"
                and row["progress"] == 100
                and fixture_calls == ["explicit-software-double"]
            )
            status, content = request("/v1/jobs/" + identifier + "/content")
            checks["local_scratch_integrity"] = (
                status == 200
                and content == raw
                and hashlib.sha256(content).hexdigest() == hashlib.sha256(raw).hexdigest()
            )
            checks["no_legacy_state_initialized"] = all(
                not hasattr(app.state, name)
                for name in (
                    "broker",
                    "repository",
                    "job_manager",
                    "lease_coordinator",
                    "model_runtime_manager",
                )
            )
            checks["no_optional_imports"] = not FORBIDDEN.intersection(sys.modules)
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            listener.close()
            inference.engine = original
            socket.socket.connect = native_connect
        checks["clean_shutdown"] = (
            not thread.is_alive()
            and not app.state.public_media_jobs.tasks
            and app.state.scratch_disk._sweeper_task is None
        )
        assert all(checks.values()), checks
        return {
            "classification": "SOFTWARE_ONLY_CORE_RUNTIME_AND_EXPLICIT_ADAPTER_DOUBLES",
            "checks": checks,
            "installed_packages": installed,
            "runtime": runtime,
            "native_inference_executed": False,
            "semantic_validation_executed": False,
            "full_factory_e2e_executed": False,
            "production_ready": False,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checks": report["checks"], "classification": report["classification"]}))


if __name__ == "__main__":
    main()
