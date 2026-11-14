"""Real private-backend Wave A runner. No public authorization bypass or READY."""

import argparse
import asyncio
import copy
import hashlib
import json
import math
import shutil
import statistics
import time
import traceback
from pathlib import Path

from factory_contracts.deployment import BackendOptions, DeploymentConfig, DeploymentEntry
from model_registry.backends.session import BackendSession
from model_registry.deployment import DeploymentRegistry, ModelCatalog
from model_registry.deployment_validator import inspect_environment

from factory.core.model_adapter import ExecutionContext, load_adapter
from factory.engines.decision.contracts import PublicDecisionRequest
from factory.engines.responses.contracts import ChatCompletionRequest, ResponseApiRequest
from factory.engines.vector.contracts import EmbeddingRequest, RerankRequest

ROOT = Path(__file__).resolve().parents[2]


def encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def plain(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def memory_observation():
    import psutil

    result = {"host_available_bytes": psutil.virtual_memory().available}
    for name in ("memory.current", "memory.max", "memory.events"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            result[name] = path.read_text().strip()
    return result


def semantic(case, output):
    expected = case["expected"]
    cap = case["capability"]
    checks = {}
    metrics = {}
    if cap == "decision.predicate_evaluation":
        answers = output["answers"]
        positive = answers[expected["positive_question"]]["probability"]
        negative = answers[expected["negative_question"]]["probability"]
        checks = {
            "positive_expected_true": positive > expected["threshold"],
            "negative_expected_false": negative < expected["threshold"],
        }
        metrics = {
            "positive_probability": positive,
            "negative_probability": negative,
            "brier_score": ((positive - 1) ** 2 + negative**2) / 2,
        }
    elif cap == "decision.choice_routing":
        checks = {"expected_choice": output["answers"][0]["choice"] == expected["choice"]}
        metrics = {"actual_choice": output["answers"][0]["choice"]}
    elif cap == "decision.score_ranking":
        low = output["answers"][expected["lower_question"]]["score"]
        high = output["answers"][expected["higher_question"]]["score"]
        margin = high - low
        checks = {"expected_ordering": margin >= expected["minimum_margin"]}
        metrics = {"lower_score": low, "higher_score": high, "margin": margin}
    elif cap == "vector.text_embedding":
        vectors = [row["embedding"] for row in output["data"]]
        norms = [math.sqrt(math.fsum(x * x for x in v)) for v in vectors]

        def cosine(a, b):
            return math.fsum(x * y for x, y in zip(a, b, strict=True)) / (
                math.sqrt(math.fsum(x * x for x in a)) * math.sqrt(math.fsum(x * x for x in b))
            )

        pos = cosine(vectors[expected["query"]], vectors[expected["positive"]])
        neg = cosine(vectors[expected["query"]], vectors[expected["negative"]])
        checks = {
            "exact_dimension": all(len(v) == expected["dimension"] for v in vectors),
            "finite": all(math.isfinite(x) for v in vectors for x in v),
            "normalization": all(abs(n - 1) <= expected["norm_tolerance"] for n in norms),
            "positive_above_negative": pos - neg >= expected["minimum_margin"],
        }
        metrics = {
            "positive_cosine": pos,
            "negative_cosine": neg,
            "margin": pos - neg,
            "norms": norms,
        }
    elif cap == "vector.reranking":
        rows = output["results"]
        scores = {r["index"]: r["relevance_score"] for r in rows}
        best = expected["best_index"]
        margin = scores[best] - max(s for i, s in scores.items() if i != best)
        checks = {
            "expected_top_document": rows[0]["index"] == best,
            "positive_above_distractors": margin >= expected["minimum_margin"],
        }
        metrics = {"scores": scores, "margin": margin}
    else:
        if case["operation"] == "response":
            text = "".join(
                part.get("text", "")
                for item in output["output"]
                if item.get("type") == "message"
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            )
        elif case["operation"] == "stream":
            text = "".join(
                choice["delta"].get("content", "") or ""
                for event in output["events"]
                for choice in event.get("choices", [])
            )
            checks["native_done"] = output["native_done"]
        else:
            text = output["choices"][0]["message"].get("content") or ""
        if "tool_name" in expected:
            calls = output["choices"][0]["message"].get("tool_calls") or []
            checks["one_native_function_call"] = len(calls) == 1
            if calls:
                function = calls[0]["function"]
                arguments = json.loads(function["arguments"])
                checks.update(
                    expected_function=function["name"] == expected["tool_name"],
                    expected_city=isinstance(arguments.get("city"), str)
                    and expected["city_contains"].casefold() in arguments["city"].casefold(),
                    no_extra_arguments=set(arguments) == {"city"},
                )
        elif "json" in expected:
            actual = json.loads(text)
            checks["structured_json_and_expected_fact"] = actual == expected["json"]
            metrics["actual_json"] = actual
        else:
            checks["exact_meaningful_answer"] = text.strip() == expected["text"]
            metrics["actual_text"] = text
    return {
        "status": "PASS" if checks and all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": metrics,
    }


def request_for(case):
    cls = {
        "decision": PublicDecisionRequest,
        "embedding": EmbeddingRequest,
        "rerank": RerankRequest,
        "chat": ChatCompletionRequest,
        "response": ResponseApiRequest,
        "stream": ChatCompletionRequest,
    }[case["operation"]]
    return cls(model=case["model"], **case["request"])


async def execute_case(context, adapter, case):
    call = request_for(case)
    operation = {
        "decision": call.task if hasattr(call, "task") else None,
        "embedding": "vector.text_embedding",
        "rerank": "vector.reranking",
        "chat": "responses.chat_completion",
        "response": "responses.create",
        "stream": "responses.chat_completion",
    }[case["operation"]]
    cap = adapter.resolve_capability(call, operation)
    required = adapter.required_capabilities(call, cap) | {cap, case["capability"]}
    if any(
        not context.model.capabilities.get(cid) or cid not in adapter.capabilities
        for cid in required
    ):
        raise ValueError("Golden requests an undeclared capability")
    adapter.validate_request(context.model, call, cap)
    prepared = await asyncio.to_thread(adapter.prepare, context, call, cap)
    if case["operation"] == "stream":
        events = []
        done = False
        async for message in adapter.stream(context, prepared):
            for line in message.splitlines():
                if line.startswith("data: "):
                    if line[6:] == "[DONE]":
                        done = True
                    else:
                        events.append(json.loads(line[6:]))
        return {"events": events, "native_done": done}
    raw = await asyncio.to_thread(adapter.infer, context, prepared)
    result = await asyncio.to_thread(adapter.normalize_output, context, prepared, raw)
    return plain(result)


async def model_run(
    model_id, root, output, cases, *, executor=None, validator=None, backend_options=None
):
    executor = executor or execute_case
    validator = validator or semantic
    catalog = ModelCatalog(ROOT / "manifests")
    model = catalog.get(model_id)
    profile = model.runtime_profiles["kaggle-dual-t4"]
    record = {
        "model": model_id,
        "engine": str(model.engine),
        "backend": profile.preferred_backend,
        "checkpoint": {k: s.model_dump(mode="json") for k, s in model.weights.files.items()},
        "cases": [],
        "production_ready": False,
        "full_factory_e2e_executed": False,
        "public_models": [],
        "fallback": False,
        "checkpoint_files": [],
        "status": "NOT_MET",
    }
    session = None
    try:
        from huggingface_hub import hf_hub_download

        for key, spec in model.weights.files.items():
            path = Path(
                hf_hub_download(spec.repository, filename=spec.path, revision=spec.revision)
            )
            sha = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    sha.update(chunk)
            matched = sha.hexdigest() == spec.sha256 and path.stat().st_size == spec.size_bytes
            record["checkpoint_files"].append(
                {
                    "key": key,
                    "repository": spec.repository,
                    "revision": spec.revision,
                    "file": spec.path,
                    "sha256": sha.hexdigest(),
                    "size_bytes": path.stat().st_size,
                    "matched": matched,
                }
            )
            if not matched:
                raise ValueError("Exact checkpoint integrity failed")
            target = root / "models" / model_id / (spec.destination or spec.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(path.resolve())
        devices = (0, 1) if profile.gpu_count == 2 else (0,)
        binary = (
            root / "llama.cpp/build/bin/llama-server"
            if profile.preferred_backend == "llama_cpp"
            else None
        )
        config = DeploymentConfig(
            models_dir=root / "models",
            runtime_dir=root / "runtime",
            deployments=(
                DeploymentEntry(
                    model=model_id,
                    devices=devices,
                    exclusive=len(devices) > 1,
                    port=18990,
                    backend_options=BackendOptions(
                        executable=binary,
                        startup_timeout_seconds=600,
                        stop_timeout_seconds=10,
                        **(backend_options or {}),
                    ),
                ),
            ),
        )
        registry = DeploymentRegistry(config, catalog, config_path=root / "native.yaml")
        session = BackendSession(registry)
        record["startup_memory_before"] = memory_observation()
        startup_active = {"value": True}
        record["startup_memory_samples"] = []

        async def startup_memory():
            import psutil

            while startup_active["value"]:
                processes = []
                for candidate in session.backends.values():
                    for pid in candidate._owned_pids():
                        try:
                            process = psutil.Process(pid)
                            processes.append({"pid": pid, "rss_bytes": process.memory_info().rss})
                        except psutil.Error:
                            continue
                record["startup_memory_samples"].append(
                    {
                        "monotonic_ns": time.monotonic_ns(),
                        "memory": memory_observation(),
                        "owned_processes": processes,
                    }
                )
                await asyncio.sleep(1)

        memory_task = asyncio.create_task(startup_memory())
        try:
            await session.start()
        finally:
            startup_active["value"] = False
            await memory_task
            record["startup_memory_after"] = memory_observation()
            selected = session.backends.get(model_id)
            record["startup_exit_code"] = (
                selected.process.poll() if selected and selected.process else None
            )
        record["lifecycle"] = session.status()
        if session.healthy_models() != (model_id,):
            raise RuntimeError("Selected backend failed startup: " + json.dumps(session.failures))
        backend = session.backends[model_id]
        if backend.binding.backend == "comfyui":
            import psutil

            assert backend.process.pid in backend._owned_pids()
            record["owned_comfy_launch_argv"] = psutil.Process(backend.process.pid).cmdline()
        context = ExecutionContext(registry, session, registry.assignment(model_id), model)
        adapter = load_adapter(model.backend.adapter)
        for case in cases:
            result = {
                "case_id": case["id"],
                "model": model_id,
                "capability": case["capability"],
                "input_sha256": digest(case),
                "fallback": False,
                "weights_verified": backend.verified,
                "status": "NOT_MET",
                "runs": [],
                "memory_before": memory_observation(),
            }
            samples = []
            executed_case = case
            iteration = None
            try:
                for iteration in range(4):
                    executed_case = copy.deepcopy(case)
                    if "seed_sequence" in case:
                        seeds = case["seed_sequence"]
                        if len(seeds) != 4 or len(set(seeds)) != 4:
                            raise ValueError(
                                "Native cached backends require four distinct declared seeds"
                            )
                        executed_case["request"]["seed"] = seeds[iteration]
                    samples = []
                    observing = {"active": True}

                    async def telemetry(samples=samples, observing=observing):
                        while observing["active"]:
                            await asyncio.to_thread(backend._gpu_ownership, backend._owned_pids())
                            samples.append(
                                {
                                    "monotonic_ns": time.monotonic_ns(),
                                    "gpu_processes": backend.runtime_info()["gpu_processes"],
                                }
                            )
                            await asyncio.sleep(0.1)

                    await asyncio.to_thread(backend._gpu_ownership, backend._owned_pids())
                    samples.append(
                        {
                            "monotonic_ns": time.monotonic_ns(),
                            "gpu_processes": backend.runtime_info()["gpu_processes"],
                        }
                    )
                    watcher = asyncio.create_task(telemetry())
                    started = time.monotonic_ns()
                    try:
                        actual = await executor(context, adapter, executed_case)
                    finally:
                        finished = time.monotonic_ns()
                        observing["active"] = False
                        await watcher
                    elapsed = (finished - started) / 1e6
                    actual_bytes = encoded(actual)
                    filename = case["id"] + f"-{iteration}.json"
                    (output / filename).write_bytes(actual_bytes)
                    semantics = validator(case, actual)
                    result["runs"].append(
                        {
                            "role": "warmup" if iteration == 0 else "measured",
                            "iteration": iteration,
                            **(
                                {
                                    "executed_input_sha256": digest(executed_case),
                                    "seed": seeds[iteration],
                                }
                                if "seed_sequence" in case
                                else {}
                            ),
                            "latency_ms": elapsed,
                            "timing_scope": "private_adapter_roundtrip_including_health_and_output_validation",
                            "output_file": filename,
                            "output_sha256": hashlib.sha256(actual_bytes).hexdigest(),
                            "semantic_validation": semantics,
                            "gpu_samples": samples,
                        }
                    )
                measured = result["runs"][1:]
                latencies = [r["latency_ms"] for r in measured]
                result["latency_ms"] = {
                    "median": statistics.median(latencies),
                    "min": min(latencies),
                    "max": max(latencies),
                }
                result["peak_vram_mb"] = max(
                    p["used_memory_mb"]
                    for r in result["runs"]
                    for s in r["gpu_samples"]
                    for p in s["gpu_processes"]
                )
                result["status"] = (
                    "PASS"
                    if all(
                        r["semantic_validation"]["status"] == "PASS"
                        and r["gpu_samples"]
                        and all(s["gpu_processes"] for s in r["gpu_samples"])
                        for r in result["runs"]
                    )
                    else "FAIL"
                )
                result["native_inference_executed"] = True
            except Exception as exc:
                result["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "details": getattr(exc, "details", {}),
                    "traceback": traceback.format_exc(),
                    "backend_exit_code": backend.process.poll() if backend.process else None,
                }
                result["native_inference_executed"] = bool(result["runs"])
                if iteration is not None and samples:
                    result["failed_attempt"] = {
                        "iteration": iteration,
                        "executed_input_sha256": digest(executed_case),
                        "gpu_samples": samples,
                        "session_failures": copy.deepcopy(session.failures),
                    }
            result["memory_after"] = memory_observation()
            record["cases"].append(result)
            (output / "observation.json").write_text(json.dumps(record, indent=2) + "\n")
        record["runtime_after_inference"] = backend.runtime_info()
        record["lifecycle_after_inference"] = session.status()
        record["backend_exit_code_before_shutdown"] = (
            backend.process.poll() if backend.process else None
        )
        record["status"] = (
            "PASS" if all(c["status"] == "PASS" for c in record["cases"]) else "NOT_MET"
        )
        assert registry.active_models() == ()
    except Exception as exc:
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if session is not None:
            await session.stop()
            record["shutdown"] = json.loads((session.directory / "shutdown.json").read_text())
            record["stop_verified"] = (
                record["shutdown"]["stop_completed"] and not record["shutdown"]["errors"]
            )
            for log in session.directory.glob("*/backend.log"):
                shutil.copyfile(log, output / "backend.log")
            backend = session.backends.get(model_id)
            if backend is not None:
                for name in backend.adapter_assets:
                    source = (backend.directory / name).resolve()
                    if source.is_relative_to(backend.directory.resolve()) and source.is_file():
                        target = output / "adapter_assets" / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, target)
        if not record.get("stop_verified"):
            record["status"] = "NOT_MET"
        (output / "observation.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    from importlib.metadata import version

    suite = json.loads((ROOT / "benchmarks/native/golden_wave_a.json").read_text())
    report = {
        "classification": "REAL_NATIVE_VALIDATION_ATTEMPT",
        "hardware": inspect_environment().gpus,
        "cases_sha256": digest(suite),
        "runtime_versions": {
            name: version(name)
            for name in (
                ["vllm", "torch", "transformers", "huggingface-hub"]
                if args.models != ["qwen-text"]
                else ["huggingface-hub", "psutil", "httpx"]
            )
        },
        "models": [],
        "fallback": False,
        "production_ready": False,
        "full_factory_e2e_executed": False,
    }
    assert set(report["hardware"]) == {0, 1} and all(
        "t4" in v["name"].lower() for v in report["hardware"].values()
    ), "Exact Dual-T4 unavailable"
    for model in args.models:
        directory = args.output / model
        directory.mkdir(parents=True, exist_ok=True)
        result = asyncio.run(
            model_run(
                model, args.root, directory, [c for c in suite["cases"] if c["model"] == model]
            )
        )
        report["models"].append(result)
    report["status"] = "PASS" if all(m["status"] == "PASS" for m in report["models"]) else "NOT_MET"
    (args.output / "wave_a.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "models": {r["model"]: r["status"] for r in report["models"]},
            }
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
