"""Image validation software contracts, explicitly not model/GPU evidence."""

import copy
import hashlib
import io
import json
from pathlib import Path

import pytest
from model_registry.deployment import ModelCatalog
from PIL import Image

from benchmarks.native.image_inputs import derive, png
from benchmarks.native.image_semantic import cosine, history_checks, pixel_metrics
from benchmarks.native.wave_a import digest
from deployments.kaggle.native_wave_a import render_kernel

pytestmark = pytest.mark.contract_mock
ROOT = Path(__file__).resolve().parents[2]
SUITE = json.loads((ROOT / "benchmarks/native/golden_wave_c.json").read_text())


def test_exact_declared_image_coverage_and_four_fresh_seeds():
    catalog = ModelCatalog(ROOT / "manifests")
    expected = {
        (m, c)
        for m in ("qwen-image-2.1", "qwen-image-2.1-turbo")
        for c, v in catalog.get(m).capabilities.items()
        if v
    }
    assert {(c["model"], c["capability"]) for c in SUITE["cases"]} == expected
    pins = digest(json.loads((ROOT / "benchmarks/native/image_semantic_pins.json").read_text()))
    for case in SUITE["cases"]:
        assert len(case["seed_sequence"]) == len(set(case["seed_sequence"])) == 4
        assert case["expected"]["verifier_pins_sha256"] == pins
        assert case["asset_identities"] == {n: SUITE["assets"][n]["sha256"] for n in case["assets"]}


def test_canonical_png_and_alpha_mask_are_real_decodable_technical_inputs():
    recipe = SUITE["assets"]["inpaint_mask"]["recipe"]
    raw = derive(recipe, {})
    assert hashlib.sha256(raw).hexdigest() == SUITE["assets"]["inpaint_mask"]["sha256"]
    image = Image.open(io.BytesIO(raw))
    assert (
        image.size == (512, 512)
        and image.getpixel((32, 240))[3] == 0
        and image.getpixel((31, 240))[3] == 255
    )
    assert png(image) == raw


def fixture():
    case = SUITE["cases"][0]
    graph = {
        "sample": {"inputs": {"seed": case["seed_sequence"][0], "model": ["prefix_cache", 0]}},
        "prefix_cache": {
            "class_type": "QwenImage21Cache",
            "inputs": {"model": ["model", 0], "device": "off", "dtype": "default"},
        },
    }
    return case, {
        "graph": graph,
        "seed": case["seed_sequence"][0],
        "prompt_id": "mock-prompt",
        "native_history": {
            "prompt": [0, "mock-prompt", copy.deepcopy(graph)],
            "status": {
                "completed": True,
                "status_str": "success",
                "messages": [
                    ["execution_start", {}],
                    ["execution_cached", {"nodes": ["clip"]}],
                    ["execution_success", {}],
                ],
            },
        },
    }


def test_cache_of_conditioning_allowed_sampling_replay_rejected():
    case, output = fixture()
    assert all(history_checks(output, case).values())
    output["native_history"]["status"]["messages"][1][1]["nodes"].append("sample")
    assert history_checks(output, case)["sampling_not_cached"] is False


@pytest.mark.parametrize("tamper", ["seed", "graph", "completion", "success"])
def test_native_execution_identity_tampering_fails(tamper):
    case, output = fixture()
    if tamper == "seed":
        output["seed"] = 999
    elif tamper == "graph":
        output["graph"]["extra"] = {}
    elif tamper == "completion":
        output["native_history"]["status"]["completed"] = False
    else:
        output["native_history"]["status"]["messages"].pop()
    assert not all(history_checks(output, case).values())


def test_pixel_oracles_reject_wrong_color_and_changed_preservation_region():
    original = png(Image.new("RGB", (32, 32), "orange"))
    changed = png(Image.new("RGB", (32, 32), "blue"))
    spec = {"color": {"name": "blue"}, "preserve": {"asset": "source", "region": [0, 0, 32, 32]}}
    assert pixel_metrics(original, spec, {"source": original})["color_fraction"] == 0
    metrics = pixel_metrics(changed, spec, {"source": original})
    assert metrics["color_fraction"] == 1 and metrics["preserved_mean_absolute_error"] > 0.5


def test_image_embedding_rejects_zero_nonfinite_wrong_dimension():
    for value in ([0.0] * 512, [float("nan")] * 512, [1.0]):
        with pytest.raises(ValueError):
            cosine(value, value)


@pytest.mark.parametrize("kind", ["image", "image_fp32"])
def test_image_kernel_is_explicitly_pinned_and_compiles(kind):
    code = render_kernel(kind)
    compile(code, "image-kernel", "exec")
    assert "IMAGE_MODELS=['qwen-image-2.1', 'qwen-image-2.1-turbo']" in code
    assert "'--torch-backend','cu130'" in code and "use_safetensors" not in code


@pytest.mark.parametrize("extra,expected_reserve", [(0, "0.800"), (0.2, "1.000")])
def test_comfy_gpu1_launch_preserves_backend_physical_visibility(
    tmp_path, monkeypatch, extra, expected_reserve
):
    """Mock launch planning only; actual model/GPU behavior remains unverified here."""
    import threading
    from types import SimpleNamespace

    import model_registry.adapters.launch as launch
    from model_registry.backends.process import ProcessBackend

    import factory.core.model_adapter

    source = tmp_path / "comfy-source"
    source.mkdir()
    (source / "main.py").write_text("# contract fixture only\n")
    directory = tmp_path / "owned"
    directory.mkdir()
    model = ModelCatalog(ROOT / "manifests").get("qwen-image-2.1")
    model.runtime_profiles["kaggle-dual-t4"].extra_vram_headroom_gb = extra
    backend = SimpleNamespace(
        directory=directory,
        model=model,
        registry=SimpleNamespace(
            config=SimpleNamespace(hardware="kaggle-dual-t4"),
            catalog=ModelCatalog(ROOT / "manifests"),
        ),
        binding=SimpleNamespace(adapter=model.backend.adapter, devices=(1,), port=0),
        token="contract",
        session_id="contract",
        environment={},
    )
    monkeypatch.setattr(launch, "source_directory", lambda *args: source)
    monkeypatch.setattr(factory.core.model_adapter, "load_adapter", lambda *args: SimpleNamespace())
    plan = launch.ComfyLaunchAdapter().plan(backend, tmp_path / "weights")
    assert "--cuda-device" not in plan.argv and "--default-device" not in plan.argv
    assert plan.argv[plan.argv.index("--reserve-vram") + 1] == expected_reserve
    if extra:
        assert plan.argv[plan.argv.index("--vram-headroom") + 1] == f"{extra:.3f}"
    else:
        assert "--vram-headroom" not in plan.argv
    # The common process launcher is the authority for the immutable physical map.
    backend.plan = plan
    backend._stopping = threading.Event()
    backend.state = "PREPARED"
    backend.process = None
    captured = {}

    def capture_launch(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("contract stops before process creation")

    monkeypatch.setattr("model_registry.backends.process.subprocess.Popen", capture_launch)
    try:
        with pytest.raises(RuntimeError, match="before process creation"):
            ProcessBackend.launch(backend)
    finally:
        backend._log.close()
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"


@pytest.mark.parametrize("precision", ["fp16", "bf16", "auto", "fp32"])
def test_image_native_precision_rejects_unsupported_or_unknown_source(precision):
    from factory.adapters.qwen_image.native_nodes import native_node_overlay

    with pytest.raises(RuntimeError, match="fallback|pinned dependency"):
        native_node_overlay(b"# contract fixture, not pinned upstream source\n", precision)


def test_image_manifest_explicitly_selects_supported_diffusion_precision():
    catalog = ModelCatalog(ROOT / "manifests")
    for model in ["qwen-image-2.1", "qwen-image-2.1-turbo"]:
        selected = catalog.get(model)
        assert selected.weights.default_precision == "fp32"
        assert selected.weights.quantization == "q4_k_m"
        assert selected.runtime_profiles["kaggle-dual-t4"].cpu_offload is True


def test_image_native_batches_cover_each_predeclared_case_once():
    from deployments.kaggle.native_wave_a import IMAGE_BATCHES

    identifiers = [name for batch in IMAGE_BATCHES.values() for name in batch]
    assert len(identifiers) == len(set(identifiers)) == len(SUITE["cases"])
    assert set(identifiers) == {c["id"] for c in SUITE["cases"]}
    for kind in IMAGE_BATCHES:
        compile(render_kernel(kind), kind, "exec")


def test_reference_prefix_cache_policy_cannot_silently_change():
    case, output = fixture()
    output["graph"]["prefix_cache"]["inputs"]["device"] = "auto"
    assert history_checks(output, case)["explicit_lossless_prefix_cache_policy"] is False


def test_image_adapter_submits_lossless_cache_off_policy_to_native_sampler():
    from types import SimpleNamespace

    from factory.adapters.qwen_image import Adapter
    from factory.core.media import MediaRequest

    case = SUITE["cases"][0]
    context = SimpleNamespace(model=ModelCatalog(ROOT / "manifests").get(case["model"]))
    request = MediaRequest(
        model=case["model"],
        capability=case["capability"],
        inputs={**case["request"], "seed": case["seed_sequence"][0]},
    )
    selected = Adapter()
    selected.validate_request(context.model, request, request.capability)
    graph = selected.prepare(context, request, request.capability).payload["graph"]
    assert graph["prefix_cache"]["class_type"] == "QwenImage21Cache"
    assert graph["prefix_cache"]["inputs"] == {
        "model": ["model", 0],
        "device": "off",
        "dtype": "default",
    }
    assert graph["sample"]["inputs"]["model"] == ["prefix_cache", 0]
