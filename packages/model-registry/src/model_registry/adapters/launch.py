"""Native launch bindings. They cannot certify or fabricate inference output."""

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml
from model_registry.backends.process import BackendError, LaunchPlan


def executable(backend, default):
    selected = backend.binding.backend_options.executable
    if selected is not None:
        selected = (
            selected if selected.is_absolute() else backend.registry.config_path.parent / selected
        )
        path = str(selected.resolve())
    else:
        path = shutil.which(default)
    if not path or not Path(path).is_file() or not os.access(path, os.X_OK):
        raise BackendError(f"Required selected backend executable is unavailable: {default}")
    return path


def source_directory(backend, env_name):
    selected = backend.binding.backend_options.source_dir
    if selected is None:
        value = os.environ.get(env_name)
        if not value:
            raise BackendError(f"Explicit backend source_dir (or {env_name}) is required")
        selected = Path(value)
    if not selected.is_absolute():
        selected = backend.registry.config_path.parent / selected
    selected = selected.resolve()
    revision = subprocess.run(
        ["git", "-C", str(selected), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    expected = backend.binding.backend_options.source_revision
    if expected and revision != expected:
        raise BackendError("Backend checkout differs from its selected immutable source revision")
    backend.source_revision = revision
    return selected


class LaunchAdapter:
    inference_implemented = False
    supports_extra_vram_headroom = False

    @classmethod
    def available(cls):
        return True  # Launch binding code only, never an inference readiness claim.


class VLLMLaunchAdapter(LaunchAdapter):
    distribution = "vllm"

    def extra_args(self, backend):
        return []

    def plan(self, backend, view):
        model, binding = backend.model, backend.binding
        profile = model.runtime_profiles[backend.registry.config.hardware]
        # T4's target memory budget is defined by the selected hardware manifest.
        topology = backend.registry.catalog.loader.load_hardware_topologies()[
            backend.registry.config.hardware
        ]
        memory = min(g.total_vram_gb for g in topology.gpus if g.index in binding.devices)
        args = [
            executable(backend, "vllm"),
            "serve",
            str(view / "model"),
            "--host",
            "127.0.0.1",
            "--port",
            str(binding.port),
            "--served-model-name",
            binding.model,
            "--dtype",
            {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}[
                model.weights.default_precision.value
            ],
            "--tensor-parallel-size",
            str(len(binding.devices)),
            "--gpu-memory-utilization",
            str(round(min(profile.min_vram_gb / memory, 0.9), 4)),
            "--enforce-eager",
            "--max-num-seqs",
            "1",
        ]
        if "max_context_tokens" in model.parameters:
            args += ["--max-model-len", str(model.parameters["max_context_tokens"]["default"])]
        if model.weights.quantization.value != "none":
            args += ["--quantization", model.weights.quantization.value]
        args += self.extra_args(backend)
        return LaunchPlan(tuple(args), backend.directory, {"VLLM_API_KEY": backend.token}, "models")


class DecisionLaunchAdapter(VLLMLaunchAdapter):
    def plan(self, backend, view):
        from factory.core.model_adapter import load_adapter

        plan = super().plan(backend, view)
        environment = load_adapter(backend.model.backend.adapter).backend_environment(backend)
        return LaunchPlan(
            plan.argv,
            plan.cwd,
            {**plan.environment, **environment},
            plan.health_kind,
            plan.model_loaded_on_health,
        )


class EmbeddingLaunchAdapter(VLLMLaunchAdapter):
    def extra_args(self, backend):
        return [
            "--runner",
            "pooling",
            "--convert",
            "embed",
            "--hf-overrides",
            json.dumps({"is_matryoshka": True}),
        ]


class RerankerLaunchAdapter(VLLMLaunchAdapter):
    def extra_args(self, backend):
        # Original checkpoint, using the upstream vLLM documented token-head
        # override. Never replace it with a converted seq-cls checkpoint.
        overrides = {
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        }
        template = backend.directory / "qwen3_reranker.jinja"
        template.write_bytes(Path(__file__).with_name("qwen3_reranker.jinja").read_bytes())
        digest = hashlib.sha256(template.read_bytes()).hexdigest()
        backend.adapter_assets[template.name] = digest
        stat = template.stat()
        backend._files[str(template)] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        return [
            "--runner",
            "pooling",
            "--hf-overrides",
            json.dumps(overrides),
            "--chat-template",
            str(template),
        ]


class OmniLaunchAdapter(VLLMLaunchAdapter):
    distribution = "vllm-omni"

    def plan(self, backend, view):
        from factory.core.model_adapter import load_adapter

        plan = super().plan(backend, view)
        selected = load_adapter(backend.model.backend.adapter)
        environment = selected.backend_environment(backend)
        return LaunchPlan(
            tuple(plan.argv),
            plan.cwd,
            {**plan.environment, **environment},
            plan.health_kind,
            plan.model_loaded_on_health,
        )

    def extra_args(self, backend):
        from factory.core.model_adapter import load_adapter

        selected = load_adapter(backend.model.backend.adapter)
        if not hasattr(selected, "backend_deploy_config"):
            raise BackendError("Selected Omni adapter must declare its stage deployment config")
        config = selected.backend_deploy_config(backend)
        return ["--omni", "--deploy-config", str(config)]


class GGUFLaunchAdapter(LaunchAdapter):
    def plan(self, backend, view):
        binding, model = backend.binding, backend.model
        spec = model.weights.files["primary"]
        args = [
            executable(backend, "llama-server"),
            "--model",
            str(view / (spec.destination or spec.path)),
            "--alias",
            binding.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(binding.port),
            "--api-key-file",
            str(backend.directory / "backend.key"),
            "--ctx-size",
            str(model.parameters["max_context_tokens"]["default"]),
            "--n-gpu-layers",
            "all",
            "--fit",
            "off",
            "--parallel",
            "1",
        ]
        args += ["--jinja"]
        if len(binding.devices) > 1:
            args += [
                "--split-mode",
                "layer",
                "--tensor-split",
                ",".join("1" for _ in binding.devices),
            ]
        return LaunchPlan(tuple(args), backend.directory, health_kind="models")


class ComfyLaunchAdapter(LaunchAdapter):
    supports_extra_vram_headroom = True

    def plan(self, backend, view):
        source = source_directory(backend, "COMFYUI_DIR")
        if not (source / "main.py").is_file():
            raise BackendError("Selected ComfyUI checkout has no main.py")
        folders = {}
        for spec in backend.model.weights.files.values():
            relative = Path(spec.destination or spec.path)
            parent = relative.parent
            if backend.model.engine == "enhancement":
                categories = ("upscale_models",)
            elif relative.suffix == ".gguf":
                categories = ("diffusion_models", "unet")
            else:
                categories = (parent.name,)
            for category in categories:
                folders.setdefault(category, set()).add(str(parent))
        paths = {key: "\n".join(sorted(values)) for key, values in folders.items()}
        paths["base_path"] = str(view)
        config = backend.directory / "extra_model_paths.yaml"
        config.write_text(yaml.safe_dump({"factory_pinned": paths}))
        base = backend.directory / "comfy_base"
        base.mkdir()
        for directory in ("custom_nodes", "input", "output", "user", "models"):
            (base / directory).mkdir()
        from factory.core.model_adapter import load_adapter

        inference_adapter = load_adapter(backend.binding.adapter)
        if hasattr(inference_adapter, "install_nodes"):
            inference_adapter.install_nodes(base, backend=backend, source_dir=source)
        args = [
            sys.executable,
            str(source / "main.py"),
            "--listen",
            "127.0.0.1",
            "--port",
            str(backend.binding.port),
            "--base-directory",
            str(base),
            "--extra-model-paths-config",
            str(config),
            "--disable-auto-launch",
        ]
        if not backend.model.runtime_profiles[backend.registry.config.hardware].cpu_offload:
            args.append("--gpu-only")
        topology = backend.registry.catalog.loader.load_hardware_topologies()[
            backend.registry.config.hardware
        ]
        headroom = max(
            g.total_vram_gb - g.max_allocatable_vram_gb
            for g in topology.gpus
            if g.index in backend.binding.devices
        )
        headroom += backend.model.runtime_profiles[backend.registry.config.hardware].extra_vram_headroom_gb
        args += ["--reserve-vram", f"{headroom:.3f}"]
        extra = backend.model.runtime_profiles[backend.registry.config.hardware].extra_vram_headroom_gb
        if extra:
            args += ["--vram-headroom", f"{extra:.3f}"]
        return LaunchPlan(tuple(args), source, health_kind="comfyui", model_loaded_on_health=False)


class NativeLaunchAdapter(LaunchAdapter):
    sdk = None

    def plan(self, backend, view):
        model = backend.model
        primary = Path(
            model.weights.files["primary"].destination or model.weights.files["primary"].path
        )
        payload = {
            "sdk": self.sdk,
            "adapter": backend.binding.adapter,
            "model": backend.binding.model,
            "model_path": str(view / "model"),
            "bundle_path": str(view),
            "primary_directory": str(primary.parent),
            "precision": model.weights.default_precision.value,
            "component_precisions": {name: precision.value for name, precision in model.weights.component_precisions.items()},
            "component_parameter_precisions": {component: {name: precision.value for name, precision in parameters.items()} for component, parameters in model.weights.component_parameter_precisions.items()},
            "cpu_offload": model.runtime_profiles[backend.registry.config.hardware].cpu_offload,
            "offload_strategy": model.runtime_profiles[backend.registry.config.hardware].offload_strategy,
            "component_tiling": {name: policy.model_dump() for name, policy in model.runtime_profiles[backend.registry.config.hardware].component_tiling.items()},
            "checkpoint_graph_sha256": backend.binding.checkpoint_graph_sha256,
            "boot_nonce": backend.boot_nonce,
            "token_file": str(backend.directory / "backend.key"),
            "capabilities": [cid for cid, enabled in model.capabilities.items() if enabled],
            "output_dir": str(backend.directory / "outputs"),
            "input_dir": str(backend.directory / "inputs"),
        }
        config = backend.directory / "native.json"
        config.write_text(json.dumps(payload))
        config.chmod(0o600)
        return LaunchPlan(
            (
                sys.executable,
                "-m",
                "model_registry.backends.native_server",
                "--config",
                str(config),
                "--port",
                str(backend.binding.port),
            ),
            backend.directory,
            health_kind="native",
        )


class ASRLaunchAdapter(NativeLaunchAdapter):
    sdk = "qwen_asr"
    distribution = "qwen-asr"


class DiffusersLaunchAdapter(NativeLaunchAdapter):
    sdk = "diffusers"
    distribution = "diffusers"


class ACEStepLaunchAdapter(NativeLaunchAdapter):
    sdk = "acestep_native"
    distribution = "acestep"


class YuELaunchAdapter(NativeLaunchAdapter):
    sdk = "yue2_native"
    distribution = "transformers"
