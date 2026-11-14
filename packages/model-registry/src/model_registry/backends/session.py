"""Startup supervision, fail-closed health invalidation and reverse-order stop."""
import asyncio
from importlib import metadata
import logging
import json
from datetime import datetime, timezone
from pathlib import Path
import time

from .process import BackendError

logger = logging.getLogger(__name__)


def selected_plugin(group, name, *, binding=None, config_path=None):
    entries = [entry for entry in metadata.entry_points(group=group) if entry.name == name]
    if len(entries) != 1:
        raise BackendError(f"Exactly one installed {group} provider named {name!r} is required")
    provider = entries[0].load()
    available = provider.available_for(binding, config_path) if group == "factory.backends" and binding is not None else provider.available()
    if not available:
        raise BackendError(f"Selected provider {name!r} is unavailable; fallback is forbidden")
    return provider


class BackendSession:
    def __init__(self, registry):
        self.registry = registry
        self.backends = {}
        self.failures = {}
        self._healthy = set()
        root = registry.config.runtime_dir
        if not root.is_absolute():
            root = registry.config_path.parent / root
        self.directory = root.resolve() / registry.session_id
        self._monitor_task = None
        self._owns_directory = False
        registry.attach_backend_session(self)

    async def start(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        self._owns_directory = True
        self.directory.chmod(0o700)
        self._monitor_task = asyncio.create_task(self._monitor())
        for binding in self.registry.bindings():
            backend = None
            try:
                provider = selected_plugin("factory.backends", binding.backend, binding=binding, config_path=self.registry.config_path)
                adapter = selected_plugin("factory.adapters", binding.adapter)()
                backend = provider(self.registry, binding, adapter, self.directory)
                self.backends[binding.model] = backend
                await asyncio.to_thread(backend.prepare)
                await asyncio.to_thread(backend.launch)
                deadline = time.monotonic() + binding.backend_options.startup_timeout_seconds
                while time.monotonic() < deadline:
                    if await asyncio.to_thread(backend.health):
                        if not self._runtime_budget_ok(binding.model):
                            raise BackendError("Observed resident GPU memory exceeds the selected hardware budget")
                        self._healthy.add(binding.model)
                        break
                    if backend.process is None or backend.process.poll() is not None or backend.state == "FAILED":
                        raise BackendError("Selected backend exited or failed before health passed")
                    await asyncio.sleep(0.5)
                else:
                    raise BackendError("Selected backend startup timed out")
            except asyncio.CancelledError:
                await self.stop()
                raise
            except Exception as exc:
                self.failures[binding.model] = {"type": type(exc).__name__, "message": str(exc)}
                logger.error("Selected backend startup failed for %s (%s): %s: %s", binding.model, binding.backend, type(exc).__name__, str(exc))
                self._healthy.discard(binding.model)
                if backend is not None:
                    backend.error = str(exc)
                    await asyncio.to_thread(backend.stop)

    async def _monitor(self):
        while True:
            await asyncio.sleep(2)
            await self.check_health()

    async def check_health(self):
        for model in tuple(self._healthy):
            backend = self.backends[model]
            try:
                healthy = await asyncio.to_thread(backend.health)
                if healthy and not self._runtime_budget_ok():
                    healthy = False
                    backend.error = "Observed resident GPU memory exceeds the selected hardware budget"
            except Exception as exc:
                healthy = False
                backend.error = f"Health failed: {type(exc).__name__}"
            if not healthy:
                self._healthy.discard(model)
                self.failures[model] = {"type": "BACKEND_UNHEALTHY", "message": backend.error or "Health/identity/runtime memory check failed"}
                try:
                    await asyncio.to_thread(backend.stop)
                except Exception as exc:
                    self.failures[model]["stop_error"] = type(exc).__name__

    def _runtime_budget_ok(self, extra=None):
        topology = self.registry.catalog.loader.load_hardware_topologies()[self.registry.config.hardware]
        limits = {gpu.index:gpu.max_allocatable_vram_gb * 1024 for gpu in topology.gpus}
        usage = {}
        for model in self._healthy | ({extra} if extra else set()):
            for row in self.backends[model].runtime_info().get("gpu_processes", []):
                usage[(row["pid"], row["device"])] = row["used_memory_mb"]
        return all(sum(memory for (_, index), memory in usage.items() if index == device) <= limit for device, limit in limits.items())

    def healthy_models(self):
        # Filter dead processes synchronously; retain the pending monitor entry
        # so it still reaps owned descendants after the leader has exited.
        alive = []
        for model in tuple(self._healthy):
            process = self.backends[model].process
            if process is None or process.poll() is not None:
                self.backends[model].state = "FAILED"
                self.backends[model].error = "Owned backend process exited"
            else:
                alive.append(model)
        return tuple(sorted(alive))

    def status(self):
        return {"healthy_backends": list(self.healthy_models()), "failures": dict(self.failures),
                "backends": {model:backend.runtime_info() for model, backend in self.backends.items()},
                "production_ready": False, "native_inference_verified": False}

    async def stop(self):
        task, self._monitor_task = self._monitor_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._healthy.clear()
        errors = []
        for backend in reversed(tuple(self.backends.values())):
            try:
                await asyncio.to_thread(backend.stop)
            except Exception as exc:
                errors.append(type(exc).__name__)
        stopped = not errors and all(backend.process is None or backend.process.poll() is not None for backend in self.backends.values())
        if self._owns_directory:
            receipt = {"session_id":self.registry.session_id, "config_sha256":self.registry.config_sha256,
                "stopped_at_utc":datetime.now(timezone.utc).isoformat(), "stop_completed":stopped, "errors":errors,
                "backends":{model:backend.runtime_info() for model,backend in self.backends.items()}, "production_ready":False}
            (self.directory / "shutdown.json").write_text(json.dumps(receipt,indent=2)+"\n")
        if errors:
            raise BackendError("One or more owned backends failed cleanup: " + ", ".join(errors))
