"""Owned subprocess lifecycle with bounded health and process-group cleanup."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import time
import threading
import urllib.request

import psutil
from model_registry.deployment_validator import verify_weights


class BackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class LaunchPlan:
    argv: tuple[str, ...]
    cwd: Path
    environment: dict[str, str] = field(default_factory=dict)
    health_kind: str = "models"
    model_loaded_on_health: bool = True


class Backend(ABC):
    @abstractmethod
    def prepare(self): ...
    @abstractmethod
    def launch(self): ...
    @abstractmethod
    def health(self): ...
    @abstractmethod
    def stop(self): ...
    @abstractmethod
    def runtime_info(self): ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class ProcessBackend(Backend):
    def __init__(self, registry, binding, adapter, session_dir):
        self.registry, self.binding = registry, binding
        self.model = registry.catalog.get(binding.model)
        self.adapter = adapter
        self.directory = Path(session_dir) / binding.model
        self.process = None
        self.plan = None
        self._log = None
        self._birth = None
        self._files = {}
        self.error = None
        self.state = "CONFIGURED"
        self.boot_nonce = secrets.token_hex(24)
        self.token = secrets.token_urlsafe(32)
        self.verified = False
        self._gpu_processes = []
        self._stopping = threading.Event()
        self.source_revision = None
        self._executable_sha256 = None
        self._package_version = None
        self._sdk_runtime = None
        self.adapter_assets = {}

    @classmethod
    def available(cls):
        return True

    @classmethod
    def available_for(cls, binding, config_path):
        selected = binding.backend_options.executable
        if selected is not None and hasattr(cls, "executable_name"):
            path = selected if selected.is_absolute() else Path(config_path).parent / selected
            return path.is_file() and os.access(path, os.X_OK)
        return cls.available()

    def prepare(self):
        self.state = "PREPARING"
        profile = self.model.runtime_profiles[self.registry.config.hardware]
        if profile.extra_vram_headroom_gb and not getattr(self.adapter, "supports_extra_vram_headroom", False):
            raise BackendError("Selected launch Adapter does not support extra VRAM headroom")
        if profile.component_tiling:
            from factory.core.model_adapter import load_adapter

            inference_adapter = load_adapter(self.binding.adapter)
            if set(profile.component_tiling) - inference_adapter.supported_tiling_components:
                raise BackendError("Selected Adapter does not support the declared component tiling policy")
        result = verify_weights(self.binding, self.model, cancelled=self._stopping.is_set)
        if not result["passed"]:
            raise BackendError("Exact checkpoint bundle failed integrity verification")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.directory.chmod(0o700)
        token_file = self.directory / "backend.key"
        token_file.write_text(self.token)
        token_file.chmod(0o600)
        view = self.directory / "weights"
        for spec in self.model.weights.files.values():
            relative = spec.destination or spec.path
            source = self.binding.weights_dir / relative
            target = view / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source.resolve())
            stat = source.stat()
            self._files[str(source)] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        self.verified = True
        self.plan = self.adapter.plan(self, view)
        executable = Path(self.plan.argv[0])
        digest = hashlib.sha256()
        with executable.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        self._executable_sha256 = digest.hexdigest()
        distribution = getattr(self.adapter, "distribution", None)
        if distribution:
            try:
                self._package_version = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                pass
        self.state = "PREPARED"

    def launch(self):
        if self._stopping.is_set() or self.state != "PREPARED" or self.process is not None:
            raise BackendError("Backend must be prepared exactly once before launch")
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", self.binding.port))
            except OSError as exc:
                raise BackendError(
                    "Selected backend port is already occupied; refusing to attach to another daemon"
                ) from exc
        env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
        env.update(self.plan.environment)
        env.update(
            CUDA_VISIBLE_DEVICES=",".join(map(str, self.binding.devices)),
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            PYTHONUNBUFFERED="1",
        )
        self._log = (self.directory / "backend.log").open("wb")
        self.process = subprocess.Popen(
            list(self.plan.argv),
            cwd=self.plan.cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._birth = psutil.Process(self.process.pid).create_time()
        self.state = "STARTING"

    def _owned_pids(self):
        if self.process is None or self.process.poll() is not None:
            return set()
        try:
            process = psutil.Process(self.process.pid)
            if process.create_time() != self._birth:
                return set()
            return {process.pid, *(child.pid for child in process.children(recursive=True))}
        except psutil.Error:
            return set()

    def _listener_owned(self, owned):
        for pid in owned:
            try:
                if any(
                    c.status == psutil.CONN_LISTEN and c.laddr.port == self.binding.port
                    for c in psutil.Process(pid).net_connections(kind="tcp")
                ):
                    return True
            except (psutil.Error, OSError):
                continue
        return False

    def _json(self, path):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.binding.port}{path}",
            headers={"Authorization": "Bearer " + self.token},
        )
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=2) as response:
            body = response.read(2 * 1024 * 1024 + 1)
            if len(body) > 2 * 1024 * 1024:
                raise BackendError("Backend health response exceeded size limit")
            return json.loads(body) if body else {}

    def _gpu_ownership(self, owned):
        try:
            index = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            devices = {
                uuid.strip(): int(idx)
                for idx, uuid in (row.split(",") for row in index.stdout.splitlines())
            }
            usage = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            records = []
            aliases = set(owned)
            for pid in owned:
                try:
                    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                        if line.startswith("NSpid:"):
                            aliases.update(map(int, line.split()[1:]))
                except (OSError, ValueError):
                    pass
            for row in usage.stdout.splitlines():
                pid, uuid, memory = (item.strip() for item in row.split(","))
                if int(pid) in aliases:
                    records.append(
                        {"pid": int(pid), "device": devices[uuid], "used_memory_mb": float(memory)}
                    )
            self._gpu_processes = records
            observed = {item["device"] for item in records if item["used_memory_mb"] > 0}
            return observed == set(self.binding.devices)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            return False

    def health(self):
        if self._stopping.is_set():
            return False
        owned = self._owned_pids()
        if not owned:
            self.state = "FAILED"
            self.error = "Owned backend process exited"
            return False
        try:
            for filename, expected in self._files.items():
                stat = Path(filename).stat()
                if (stat.st_ino, stat.st_size, stat.st_mtime_ns) != expected:
                    raise BackendError(
                        "Verified checkpoint or adapter asset changed after preparation"
                    )
            if not self._listener_owned(owned):
                raise BackendError("Backend listener does not belong to the owned process tree")
            if self.plan.health_kind == "native":
                body = self._json("/_factory/health")
                ok = (
                    body.get("boot_nonce") == self.boot_nonce
                    and body.get("model") == self.binding.model
                    and body.get("checkpoint_graph_sha256") == self.binding.checkpoint_graph_sha256
                    and body.get("model_loaded") is True
                )
                if ok:
                    self._sdk_runtime = body.get("sdk_runtime")
            elif self.plan.health_kind == "comfyui":
                body = self._json("/system_stats")
                ok = any(device.get("type") == "cuda" for device in body.get("devices", []))
            else:
                self._json("/health")
                body = self._json("/v1/models")
                ok = {item.get("id") for item in body.get("data", [])} == {self.binding.model}
            if (
                ok
                and self._gpu_ownership(owned)
                and not self._stopping.is_set()
                and self.process.poll() is None
            ):
                self.state, self.error = "HEALTHY", None
                return True
        except (OSError, ValueError, BackendError):
            pass
        if self.state == "HEALTHY":
            self.state, self.error = "UNHEALTHY", "Current-session health/identity/GPU check failed"
        return False

    def stop(self):
        self._stopping.set()
        if self.process is not None:
            # The process group was created by us. Children can outlive their
            # parent; terminate the entire group, then reap the owned parent.
            try:
                leader = psutil.Process(self.process.pid)
                if leader.create_time() != self._birth:
                    raise BackendError("Refusing to signal a reused, unowned process ID")
                children = leader.children(recursive=True)
            except psutil.NoSuchProcess:
                children = []
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=self.binding.backend_options.stop_timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
            _, alive = psutil.wait_procs(
                children, timeout=self.binding.backend_options.stop_timeout_seconds
            )
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=5)
        if self._log is not None:
            self._log.close()
            self._log = None
        self.state = "STOPPED"

    def runtime_info(self):
        executable = Path(self.plan.argv[0]).resolve() if self.plan is not None else None
        return {
            "model": self.binding.model,
            "backend": self.binding.backend,
            "adapter": self.binding.adapter,
            "session_id": self.registry.session_id,
            "checkpoint_graph_sha256": self.binding.checkpoint_graph_sha256,
            "devices": list(self.binding.devices),
            "state": self.state,
            "pid": self.process.pid if self.process else None,
            "pid_started_at": self._birth,
            "port": self.binding.port,
            "weights_verified": self.verified,
            "model_loaded": self.state == "HEALTHY" and self.plan.model_loaded_on_health,
            "gpu_processes": list(self._gpu_processes),
            "log_path": str(self.directory / "backend.log"),
            "executable": str(executable) if executable else None,
            "source_revision": self.source_revision,
            "sdk_runtime": self._sdk_runtime,
            "adapter_assets": dict(self.adapter_assets),
            "executable_sha256": self._executable_sha256,
            "backend_package_version": self._package_version,
            "error": self.error,
            "production_ready": False,
            "inference_verified": False,
        }
