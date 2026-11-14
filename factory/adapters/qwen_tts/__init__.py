import base64
from factory.core.model_adapter import ModelAdapter, PreparedCall
from factory.core.media import data_bytes, check_parameters, artifact
from factory_contracts.errors import InvalidRequestError


class Adapter(ModelAdapter):
    capabilities = frozenset({"speech.tts", "speech.voice_cloning"})
    supported_backends = frozenset({"vllm_omni"})

    def backend_environment(self, backend):
        from .native_compat import stage_environment

        return stage_environment(backend)

    def backend_deploy_config(self, backend):
        import hashlib
        from importlib.util import find_spec
        from pathlib import Path
        import yaml

        spec = find_spec("vllm_omni")
        if spec is None or spec.origin is None:
            raise RuntimeError("Selected vLLM-Omni stage template is unavailable")
        source = Path(spec.origin).parent / "deploy/qwen3_tts.yaml"
        config = yaml.safe_load(source.read_text())
        stages = config.get("stages", [])
        if [s.get("stage_id") for s in stages] != [0, 1]:
            raise RuntimeError("Unsupported Qwen TTS native stage layout")
        if (
            len(backend.binding.devices) != 1
            or backend.model.weights.default_precision.value != "fp16"
        ):
            raise RuntimeError("This Qwen TTS deployment requires explicit single-GPU FP16")
        profile = backend.model.runtime_profiles[backend.registry.config.hardware]
        topology = backend.registry.catalog.loader.load_hardware_topologies()[
            backend.registry.config.hardware
        ]
        memory = next(
            g.total_vram_gb for g in topology.gpus if g.index == backend.binding.devices[0]
        )
        total_fraction = min(profile.min_vram_gb / memory, 0.9)
        config["model_runner"] = "v1"
        for stage, share in zip(stages, (0.67, 0.33)):
            stage.update(
                dtype="float16",
                devices="0",
                max_num_seqs=1,
                enforce_eager=True,
                async_scheduling=False,
                gpu_memory_utilization=round(total_fraction * share, 4),
            )
        # Platform overlays cannot re-enable unsupported precision/graphs.
        config.pop("platforms", None)
        path = backend.directory / "qwen3_tts_deploy.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        upstream = backend.directory / "qwen3_tts_upstream.yaml"
        upstream.write_bytes(source.read_bytes())
        for tracked in (path, upstream):
            backend.adapter_assets[tracked.name] = hashlib.sha256(tracked.read_bytes()).hexdigest()
            stat = tracked.stat()
            backend._files[str(tracked)] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        return path

    def required_capabilities(self, request, capability):
        return {capability, "speech.tts"}

    def resolve_capability(self, request, operation):
        return "speech.voice_cloning" if request.inputs.get("reference_audio") else operation

    def validate_request(self, model, request, capability):
        check_parameters(
            model,
            request.inputs,
            {"text", "reference_audio", "reference_text", "voice", "speed", "response_format"},
        )
        if not isinstance(request.inputs.get("text"), str) or not request.inputs["text"].strip():
            raise InvalidRequestError("TTS text is required.")
        if not request.inputs.get("reference_audio") or not request.inputs.get("reference_text"):
            raise InvalidRequestError(
                "Selected Base checkpoint requires reference audio and its transcript; no preset/VoiceDesign fallback."
            )
        data_bytes(request.inputs["reference_audio"], "audio")
        if request.inputs.get("voice") not in (None, "default"):
            raise InvalidRequestError("Preset voices require a separately selected checkpoint.")
        if request.inputs.get("response_format", "wav") != "wav":
            raise InvalidRequestError("This adapter currently validates WAV output only.")

    def prepare(self, context, request, capability):
        ref = (
            "data:audio/wav;base64,"
            + base64.b64encode(data_bytes(request.inputs["reference_audio"], "audio")).decode()
        )
        payload = {
            "model": context.binding.model,
            "input": request.inputs["text"],
            "task_type": "Base",
            "ref_audio": ref,
            "ref_text": request.inputs["reference_text"],
            "response_format": "wav",
            "non_streaming_mode": True,
            "speed": request.inputs.get("speed", 1.0),
        }
        return PreparedCall(capability, payload)

    def infer(self, context, prepared):
        return context.post_bytes("/v1/audio/speech", prepared.payload)

    def normalize_output(self, context, prepared, output):
        return {
            "model": context.binding.model,
            "artifacts": [artifact(output, "audio", "audio/wav")],
        }
