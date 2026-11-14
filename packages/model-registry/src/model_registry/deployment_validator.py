"""Two distinct preflights: configuration plausibility and local environment.

Neither stage certifies load, semantic inference, or production capabilities.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import shutil
import socket
import subprocess

from factory_contracts.capabilities import require_canonical


@dataclass
class EnvironmentInventory:
    gpus: dict = field(default_factory=dict)
    backends: frozenset[str] = frozenset()
    adapters: frozenset[str] = frozenset()
    error: str | None = None


def _installed_plugins(group):
    installed = set()
    for entry in metadata.entry_points(group=group):
        try:
            provider = entry.load()
            if callable(getattr(provider, "available", None)) and provider.available():
                installed.add(entry.name)
        except Exception:
            continue
    return frozenset(installed)


def inspect_environment():
    inventory = EnvironmentInventory(backends=_installed_plugins("factory.backends"), adapters=_installed_plugins("factory.adapters"))
    executable = shutil.which("nvidia-smi")
    if not executable:
        inventory.error = "nvidia-smi is unavailable; no GPU hardware attested"
        return inventory
    try:
        result = subprocess.run([executable, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True)
        for row in result.stdout.splitlines():
            index, name, memory = row.split(",")
            inventory.gpus[int(index)] = {"name": name.strip(), "total_vram_gb": float(memory) / 1024}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        inventory.gpus.clear()
        inventory.error = f"GPU inventory unavailable: {type(exc).__name__}"
    return inventory


def verify_weights(binding, model, *, cancelled=None):
    files = []
    for key, spec in model.weights.files.items():
        path = binding.weights_dir / (spec.destination or spec.path)
        passed, reason = False, "file missing"
        try:
            before = path.stat()
            if not path.is_file():
                reason = "not a regular file"
            elif before.st_size != spec.size_bytes:
                reason = "byte size mismatch"
            else:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(8 * 1024 * 1024):
                        if cancelled is not None and cancelled():
                            raise RuntimeError("Checkpoint verification cancelled")
                        digest.update(chunk)
                after = path.stat()
                stable = (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
                passed = stable and digest.hexdigest() == spec.sha256
                reason = "exact SHA256/size match" if passed else "SHA256 mismatch or file changed during verification"
        except OSError as exc:
            reason = type(exc).__name__
        files.append({"key": key, "path": str(path), "expected_sha256": spec.sha256, "passed": passed, "reason": reason})
    return {"model": binding.model, "checkpoint_graph_sha256": binding.checkpoint_graph_sha256,
            "passed": bool(files) and all(item["passed"] for item in files), "files": files,
            "model_inference_verified": False}


class DeploymentValidator:
    def __init__(self, registry):
        self.registry = registry

    def validate(self, *, config_only=False, inventory=None):
        errors, checks = [], []
        def check(name, passed, message, model=None, stage="configuration"):
            item = {"check": name, "passed": bool(passed), "message": message, "stage": stage}
            if model:
                item["model"] = model
            checks.append(item)
            if not passed:
                errors.append(item)

        registry = self.registry
        topologies = registry.catalog.loader.load_hardware_topologies()
        topology = topologies.get(registry.config.hardware)
        check("hardware_profile", topology is not None, "Selected hardware topology must exist")
        gpu_limits = {gpu.index: gpu.max_allocatable_vram_gb for gpu in topology.gpus} if topology else {}
        allocated, residents, ports = defaultdict(float), defaultdict(list), {registry.config.api_port: "api"}
        for binding in registry.bindings():
            model = registry.catalog.get(binding.model)
            canonical = model.apiVersion == "factory.ai/v2"
            check("canonical_model", canonical, "Deployment requires a canonical v2 model", binding.model)
            if not canonical:
                continue
            profile = model.runtime_profiles.get(registry.config.hardware)
            check("runtime_profile", profile is not None, "Model must declare requirements for selected hardware", binding.model)
            if profile is None:
                continue
            check("model_disposition", model.validation.state.value not in {"BLOCKED", "DISABLED", "DEPRECATED"},
                  "Blocked/disabled/deprecated models cannot be served", binding.model)
            check("backend_selection", binding.backend in model.backend.supported, "Select one manifest-supported backend before serving", binding.model)
            check("gpu_count", len(binding.devices) == profile.gpu_count, "GPU count must match the selected runtime profile", binding.model)
            check("exclusive_requirement", not profile.exclusive or binding.exclusive, "Runtime exclusivity must be preserved", binding.model)
            check("gpu_ids", all(device in gpu_limits for device in binding.devices), "GPU IDs must belong to selected hardware", binding.model)
            check("backend_port", binding.port not in ports, f"Backend port {binding.port} must not conflict with another backend/API", binding.model)
            ports[binding.port] = binding.model
            for cid in model.capabilities:
                require_canonical(cid, engine=model.engine)
            for device in binding.devices:
                allocated[device] += profile.min_vram_gb
                residents[device].append(binding)
        for device, bindings in residents.items():
            check("exclusive_conflict", len(bindings) == 1 or not any(binding.exclusive for binding in bindings),
                  f"GPU {device} cannot share a resident exclusive deployment")
            check("vram_budget", allocated[device] <= gpu_limits.get(device, 0),
                  f"GPU {device}: sum of declared per-device requirements {allocated[device]:g} GB; limit {gpu_limits.get(device, 0):g} GB. Metadata plausibility only.")
        configuration_valid = not errors
        weights = []
        if not config_only and configuration_valid:
            observed = inventory if inventory is not None else inspect_environment()
            check("gpu_inventory", set(observed.gpus) == {0, 1} and all("t4" in gpu["name"].lower() for gpu in observed.gpus.values()),
                  observed.error or "Real current-host Dual-T4 inventory is required", stage="environment")
            for binding in registry.bindings():
                model = registry.catalog.get(binding.model)
                profile = model.runtime_profiles[registry.config.hardware]
                check("observed_vram", all(observed.gpus.get(device, {}).get("total_vram_gb", 0) >= allocated[device] for device in binding.devices),
                      "Observed device memory must cover configured resident requirements", binding.model, "environment")
                installed = binding.backend in observed.backends
                if inventory is None:
                    entries = [e for e in metadata.entry_points(group="factory.backends") if e.name == binding.backend]
                    try:
                        installed = len(entries) == 1 and entries[0].load().available_for(binding, registry.config_path)
                    except Exception:
                        installed = False
                check("backend_installed", installed, "A loadable factory.backends provider for the selected implementation is required", binding.model, "environment")
                check("adapter_available", binding.adapter in observed.adapters, "A loadable factory.adapters provider for the selected adapter is required", binding.model, "environment")
                result = verify_weights(binding, model)
                weights.append(result)
                check("checkpoint_integrity", result["passed"], "Every pinned primary/support file must match SHA256 and size", binding.model, "environment")
            for port, owner in ports.items():
                try:
                    family = socket.AF_INET6 if registry.config.api_host == "::1" else socket.AF_INET
                    with socket.socket(family) as sock:
                        sock.bind((registry.config.api_host, port))
                    available = True
                except OSError:
                    available = False
                check("port_available", available, f"Port {port} for {owner} must be available at preflight", stage="environment")
        return {"mode": "configuration_only" if config_only else "local_environment", "configuration_valid": configuration_valid,
            "observed_at_utc": datetime.now(timezone.utc).isoformat(), "classification": "PREFLIGHT_ONLY",
            "preflight_passed": not config_only and configuration_valid and not errors,
            "config_sha256": registry.config_sha256, "resolved": registry.snapshot(), "checks": checks, "errors": errors,
            "weight_verification": weights, "runtime_load_validated": False, "production_ready": False,
            "active_models": [], "native_inference_executed": False}
