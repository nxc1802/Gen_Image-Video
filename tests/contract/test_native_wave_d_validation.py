"""Explicit software-only video oracle controls; never model/GPU inference evidence."""

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from benchmarks.native.video_semantic import motion, sampled_indices
from deployments.kaggle.native_wave_a import render_kernel

pytestmark = pytest.mark.contract_mock
ROOT = Path(__file__).resolve().parents[2]


def frames(positions):
    result = []
    for x in positions:
        image = Image.new("RGB", (64, 32), "white")
        ImageDraw.Draw(image).ellipse((x, 8, x + 12, 20), fill="red")
        result.append(image)
    return result


def test_motion_oracle_distinguishes_static_leftward_and_rightward():
    right = motion(frames([5, 10, 15, 20, 25]))
    left = motion(frames([25, 20, 15, 10, 5]))
    still = motion(frames([15] * 5))
    assert (
        right["rightward_displacement_fraction"] > 0.025 and right["distinct_decoded_frames"] == 5
    )
    assert left["rightward_displacement_fraction"] < 0
    assert still["rightward_displacement_fraction"] == 0 and still["distinct_decoded_frames"] == 1


def test_colour_oracle_requires_visible_target_object_in_every_frame():
    scene = frames([5, 10, 15, 20, 25])
    scene[2] = Image.new("RGB", (64, 32), "white")
    assert min(motion(scene)["red_fraction_by_frame"]) == 0


def test_frame_sampling_covers_actual_video_endpoints():
    assert sampled_indices(17) == [0, 4, 8, 12, 16]
    assert sampled_indices(5) == [0, 1, 2, 3, 4]


def test_wan_baseline_kernel_compiles_with_explicit_native_versions():
    code = render_kernel("wan_t2v")
    compile(code, "wan-native-kernel", "exec")
    assert (
        "VIDEO_MODELS=['wan-2.2']" in code
        and "VIDEO_CASES=['wan-t2v-red-ball-screen-right-v3']" in code
    )
    assert "'--torch-backend','cu130'" in code
    assert "0.43.0" in code


def test_video_goldens_cover_exact_declared_model_set_with_distinct_native_seeds():
    suite = json.loads((ROOT / "benchmarks/native/golden_wave_d.json").read_text())
    assert len(suite["cases"]) == 13 and {c["model"] for c in suite["cases"]} == {
        "wan-2.2",
        "minimax-h3",
        "minimax-h3-turbo",
        "minimax-h3-reference",
    }
    assert len(set(suite["cases"][0]["seed_sequence"])) == 4


@pytest.mark.parametrize(
    "capability,fields",
    [
        ("video.text_to_video", {"image": b"contract fixture"}),
        ("video.image_to_video", {"image": b"contract fixture", "last_image": b"contract fixture"}),
    ],
)
def test_minimax_rejects_keyframes_that_would_change_explicit_capability(
    routing_registry, capability, fields
):
    from factory_contracts.errors import InvalidRequestError

    from factory.core.media import MediaRequest
    from factory.core.model_adapter import load_adapter

    model = routing_registry([("minimax-h3", [0])]).catalog.get("minimax-h3")
    request = MediaRequest(
        model="minimax-h3", capability=capability, inputs={"prompt": "contract", **fields}
    )
    with pytest.raises(InvalidRequestError, match="conflict"):
        load_adapter("minimax").validate_request(model, request, capability)


def test_minimax_reference_uses_native_autogrow_dependency_links(routing_registry, monkeypatch):
    from factory.core.media import MediaRequest
    from factory.core.model_adapter import load_adapter
    from tests.contract.test_phase5_response_and_media import context_for

    context = context_for(routing_registry([("minimax-h3-reference", [0])]), "minimax-h3-reference")
    adapter = load_adapter("minimax")
    monkeypatch.setattr(adapter, "stage", lambda context, value: value)
    request = MediaRequest(
        model=context.binding.model,
        capability="video.reference_conditioning",
        inputs={
            "prompt": "<Picture 1> and <Picture 2> in motion",
            "reference_images": ["first-contract.png", "second-contract.png"],
        },
    )
    adapter.validate_request(context.model, request, request.capability)
    graph = adapter.prepare(context, request, request.capability).payload["graph"]
    inputs = graph["condition"]["inputs"]
    assert "ref_images" not in inputs
    assert inputs["ref_images.ref_image_1"] == ["ref1", 0]
    assert inputs["ref_images.ref_image_2"] == ["ref2", 0]
    assert graph["ref1"]["inputs"]["image"] == "first-contract.png"
    assert graph["ref2"]["inputs"]["image"] == "second-contract.png"


def test_extension_oracle_rejects_static_generated_suffix_even_when_input_moves(monkeypatch):
    """Adversarial software control: a copied moving prefix cannot prove native extension."""
    import base64
    import hashlib

    from benchmarks.native import video_semantic

    prefix = frames([5, 10, 15, 20])
    suffix = frames([25] * 5)
    monkeypatch.setattr(
        video_semantic,
        "decode",
        lambda raw: (prefix + suffix, {"width": 64, "height": 32, "frames": 9, "fps": 16.0}),
    )
    vector = [1.0] + [0.0] * 511
    opposite = [-1.0] + [0.0] * 511
    raw = b"explicit adversarial contract control, never a model output"
    case = {
        "model": "wan-2.2",
        "capability": "video.video_extension",
        "seed_sequence": [1, 2, 3, 4],
        "request": {"width": 64, "height": 32, "num_frames": 5, "fps": 16},
        "expected": {
            "generated_segment_start_frame": 4,
            "output_num_frames": 9,
            "positive_text": "contract positive",
            "negative_texts": ["contract negative"],
            "verifier_pins_sha256": "contract-pins",
            "minimum_mean_text_cosine": 0.25,
            "minimum_frame_text_cosine": 0.22,
            "minimum_text_margin": 0.03,
            "minimum_distinct_frames": 5,
            "minimum_mean_frame_change": 0.002,
        },
    }
    sha = hashlib.sha256(raw).hexdigest()
    output = {
        "model": "wan-2.2",
        "seed": 1,
        "artifacts": [
            {
                "type": "video",
                "mime_type": "video/mp4",
                "size_bytes": len(raw),
                "sha256": sha,
                "data_base64": base64.b64encode(raw).decode(),
            }
        ],
        "verification": {
            "input_sha256": sha,
            "pins_sha256": "contract-pins",
            "frame_indices": [4, 5, 6, 7, 8],
            "frame_embeddings": [vector] * 5,
            "text_embeddings": [vector, opposite],
            "texts": ["contract positive", "contract negative"],
        },
    }
    result = video_semantic.semantic(case, output)
    assert result["status"] == "FAIL"
    assert result["checks"]["expected_object_and_scene"] is True
    assert result["checks"]["exact_frame_sampling"] is True
    assert result["checks"]["nonstatic_video"] is False
    assert result["checks"]["generated_segment_has_real_pixel_motion"] is False


def test_native_video_decoder_rejects_corrupt_container():
    import subprocess

    from benchmarks.native.video_semantic import decode

    with pytest.raises(subprocess.CalledProcessError):
        decode(b"explicit corrupt container control, never a model output")


def test_wan_fl2v_rejects_native_branch_that_ignores_mandatory_last_image(tmp_path):
    from types import SimpleNamespace

    from factory.adapters.wan import Adapter

    # Software negative control only. No inference is executed or certified.
    loaded = {"image": SimpleNamespace(config=SimpleNamespace(expand_timesteps=True))}
    with pytest.raises(RuntimeError, match="ignores last_image"):
        Adapter().native_infer(
            loaded,
            "video.first_last_frame_to_video",
            {"prompt": "contract", "image": "invalid", "last_image": "invalid"},
            {"output_dir": str(tmp_path / "must-not-be-created")},
        )
    assert not (tmp_path / "must-not-be-created").exists()


def test_wan_endpoint_override_rejects_unknown_upstream_before_tensor_runtime():
    from types import SimpleNamespace

    from factory.adapters.wan.endpoint_conditioning import endpoint_pipeline_class

    def unrelated_implementation():
        raise AssertionError("Software source-identity negative control must not execute")

    unknown = SimpleNamespace(prepare_latents=unrelated_implementation, __call__=unrelated_implementation)
    with pytest.raises(RuntimeError, match="Unverified Wan"):
        endpoint_pipeline_class(unknown)


def test_wan_endpoint_evidence_survives_default_warning_root_logging(monkeypatch, capsys):
    import logging

    from factory.adapters.wan import emit_endpoint_evidence

    monkeypatch.setattr(logging.getLogger(), "level", logging.WARNING)
    control = {"classification": "SOFTWARE_ONLY_EVIDENCE_TRANSPORT_CONTROL", "inference": False}
    emit_endpoint_evidence(control)
    line = capsys.readouterr().out.strip()
    prefix = "FACTORY_WAN_ENDPOINT_CONDITIONING "
    assert line.startswith(prefix) and json.loads(line[len(prefix):]) == control


@pytest.mark.parametrize("corrupt", ["seed", "source_last_image_sha256", "conditioned_latent_indices", "output_mp4_sha256", "free_interior_latent_frames"])
def test_wan_endpoint_bindings_reject_wrong_input_output_or_mask(corrupt):
    """Software-only predicate controls, no native worker/output/receipt is created."""
    from benchmarks.native.wan_endpoint_evidence import verify
    from factory.adapters.wan.endpoint_conditioning import runtime_identity

    suite = json.loads((ROOT / "benchmarks/native/golden_wave_d.json").read_text())
    case = next(c for c in suite["cases"] if c["model"] == "wan-2.2" and c["capability"] == "video.first_last_frame_to_video")
    request = case["request"]
    temporal = (request["num_frames"] - 1) // 4 + 1
    row = {
        **runtime_identity(), "seed": 314159, "output_mp4_sha256": "d" * 64,
        "source_first_image_sha256": suite["assets"][request["image"][1:]]["sha256"],
        "source_last_image_sha256": suite["assets"][request["last_image"][1:]]["sha256"],
        "condition_shape": [1, 48, temporal, request["height"] // 16, request["width"] // 16],
        "context_pixel_frames": request["num_frames"],
        "conditioned_latent_indices": [0, temporal - 1], "free_interior_latent_frames": temporal - 2,
        **dict.fromkeys(["normalized_first_image_sha256", "normalized_last_image_sha256", "first_condition_sha256", "last_condition_sha256", "condition_sha256", "mask_sha256"], "a" * 64),
    }
    sample, output = {"seed": 314159}, {"artifacts": [{"sha256": "d" * 64}]}
    assert verify(case, sample, output, [row], suite["assets"]) == row
    row[corrupt] = {"seed": 271828, "source_last_image_sha256": "0" * 64, "conditioned_latent_indices": [0], "output_mp4_sha256": "0" * 64, "free_interior_latent_frames": 0}[corrupt]
    with pytest.raises(AssertionError):
        verify(case, sample, output, [row], suite["assets"])


def test_declared_video_fixtures_match_original_input_identities():
    import hashlib

    suite = json.loads((ROOT / "benchmarks/native/golden_wave_d.json").read_text())
    for name, spec in suite["assets"].items():
        raw = (ROOT / spec["fixture_path"]).read_bytes()
        assert (
            len(raw) == spec["size_bytes"] and hashlib.sha256(raw).hexdigest() == spec["sha256"]
        ), name
    wan = {(c["model"], c["capability"]) for c in suite["cases"] if c["model"] == "wan-2.2"}
    assert len(wan) == 4


@pytest.mark.parametrize(
    "compute,manual,device",
    [("fp16", None, "cuda"), ("fp32", "fp16", "cuda"), ("fp32", None, "cpu")],
)
def test_minimax_native_loader_fails_closed_without_dtype_or_device_fallback(
    monkeypatch, compute, manual, device
):
    import runpy
    import sys
    from types import ModuleType, SimpleNamespace

    calls = []
    sd = ModuleType("comfy.sd")
    model = SimpleNamespace(
        model=SimpleNamespace(get_dtype=lambda: compute, manual_cast_dtype=manual),
        load_device=SimpleNamespace(type=device),
    )
    sd.load_diffusion_model = lambda path, **kwargs: calls.append((path, kwargs)) or model
    comfy = ModuleType("comfy")
    comfy.sd = sd
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.sd", sd)
    monkeypatch.setitem(
        sys.modules,
        "folder_paths",
        SimpleNamespace(
            get_full_path_or_raise=lambda category, name: "/verified-checkpoint/" + name
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="fp32"))
    node = runpy.run_path(str(ROOT / "factory/adapters/minimax/loader_node.py"))[
        "MiniMaxManifestLoader"
    ]()
    with pytest.raises(RuntimeError, match="FP32|CUDA"):
        node.load("contract-original-int8.safetensors")
    assert calls == [
        (
            "/verified-checkpoint/contract-original-int8.safetensors",
            {"model_options": {"dtype": "fp32"}},
        )
    ]


@pytest.mark.parametrize(
    "tiling",
    [
        {},
        {"vae": {"tile_height": 512, "tile_width": 512, "stride_height": 384, "stride_width": 384}},
    ],
)
def test_wan_wrappers_preserve_explicit_component_dtypes_without_global_recast(monkeypatch, tiling):
    """Software reproduction: native from_pipe would recast a separately loaded FP32 VAE."""
    import sys
    from types import ModuleType, SimpleNamespace

    from factory.adapters.wan import Adapter

    # Isolate wrapper/precision plumbing; this software control is not inference.
    monkeypatch.setattr(
        "factory.adapters.wan.endpoint_conditioning.endpoint_pipeline_class", lambda native: native
    )

    offloads = []

    overrides = {
        "transformer": {"time_embedder.weight": "fp32"},
        "text_encoder": {"wo.weight": "fp32"},
    }

    def component(name):
        params = {
            "weight": SimpleNamespace(
                dtype="fp32" if name == "vae" else "fp16",
                is_floating_point=lambda: True,
                numel=lambda: 10,
            )
        }
        params.update(
            {
                key: SimpleNamespace(dtype="fp32", is_floating_point=lambda: True, numel=lambda: 10)
                for key in overrides.get(name, {})
            }
        )
        return SimpleNamespace(named_parameters=lambda: iter(params.items()))

    modules = {name: component(name) for name in ["transformer", "text_encoder", "vae"]}

    def enable_tiling(**kwargs):
        modules["vae"].use_tiling = True
        for key, value in kwargs.items():
            setattr(modules["vae"], key, value)

    modules["vae"].enable_tiling = enable_tiling
    shared = {**modules, "tokenizer": object(), "scheduler": object()}
    text = SimpleNamespace(
        **shared,
        components=shared,
        config=SimpleNamespace(boundary_ratio=None, expand_timesteps=True),
        enable_sequential_cpu_offload=lambda **kw: offloads.append(kw),
    )

    def load_vae(path, **kwargs):
        assert path == "/verified/model" and kwargs == {
            "subfolder": "vae",
            "dtype": "fp32",
            "local_files_only": True,
        }
        return modules["vae"]

    def from_pretrained(path, **kwargs):
        assert path == "/verified/model" and kwargs == {
            "dtype": "fp16",
            "vae": modules["vae"],
            "local_files_only": True,
        }
        return text

    def image_wrapper(**kwargs):
        assert all(kwargs[name] is module for name, module in shared.items())
        assert kwargs["expand_timesteps"] is True and kwargs["boundary_ratio"] is None
        return SimpleNamespace(
            **shared, enable_sequential_cpu_offload=lambda **kw: offloads.append(kw)
        )

    module = ModuleType("diffusers")
    module.AutoencoderKLWan = SimpleNamespace(from_pretrained=load_vae)
    module.WanPipeline = SimpleNamespace(from_pretrained=from_pretrained)
    module.WanImageToVideoPipeline = image_wrapper
    monkeypatch.setitem(sys.modules, "diffusers", module)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(float16="fp16", float32="fp32", bfloat16="bf16")
    )
    loaded = Adapter().load_native(
        {
            "model_path": "/verified/model",
            "precision": "fp16",
            "component_precisions": {"vae": "fp32"},
            "component_parameter_precisions": overrides,
            "cpu_offload": True,
            "offload_strategy": "sequential",
            "component_tiling": tiling,
        },
        "fp16",
    )
    assert loaded["text"].vae is loaded["image"].vae
    assert offloads == [{"gpu_id": 0}, {"gpu_id": 0}]
    assert loaded._factory_runtime_identity["component_precisions"] == {"vae": "fp32"}
    assert loaded._factory_runtime_identity["component_parameter_precisions"] == overrides
    assert loaded._factory_runtime_identity["component_tiling"] == tiling
    assert all(
        spec["dtypes"] == (["fp32"] if name == "vae" else ["fp16", "fp32"])
        and spec["shared_between_wrappers"]
        for name, spec in loaded._factory_runtime_identity["components"].items()
    )


@pytest.mark.parametrize(
    "kind",
    [
        "wan_t2v",
        "minimax_t2v",
        "wan_conditioned",
        "wan_i2v",
        "wan_fl2v",
        "wan_extension",
        "minimax_conditioned",
        "minimax_turbo",
        "minimax_reference",
        "minimax_audio",
        "minimax_reference_audio",
    ],
)
def test_native_video_script_fits_kaggle_source_limit_with_exact_selected_fixtures(kind):
    import ast
    import base64
    import io
    import zipfile

    from deployments.kaggle.native_wave_a import VIDEO_BATCHES

    code = render_kernel(kind)
    assert len(code.encode()) < 1_000_000
    compile(code, kind, "exec")
    encoded = ast.literal_eval(ast.parse(code).body[0].value)
    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded)))
    suite = json.loads((ROOT / "benchmarks/native/golden_wave_d.json").read_text())
    wanted = {
        suite["assets"][n]["fixture_path"]
        for c in suite["cases"]
        if c["id"] in VIDEO_BATCHES[kind]
        for n in c["assets"]
    }
    assert {n for n in archive.namelist() if n.startswith("tests/fixtures/native/")} == wanted


def test_video_goldens_pass_manifest_request_admission_before_gpu_setup(routing_registry):
    from factory.core.media import MediaRequest
    from factory.core.model_adapter import load_adapter

    suite = json.loads((ROOT / "benchmarks/native/golden_wave_d.json").read_text())
    for case in suite["cases"]:
        registry = routing_registry([(case["model"], [0])])
        model = registry.catalog.get(case["model"])
        inputs = {
            k: (ROOT / suite["assets"][v[1:]]["fixture_path"]).read_bytes()
            if isinstance(v, str) and v.startswith("@")
            else v
            for k, v in case["request"].items()
        }
        request = MediaRequest(model=case["model"], capability=case["capability"], inputs=inputs)
        adapter = load_adapter(model.backend.adapter)
        assert model.capabilities[case["capability"]] and case["capability"] in adapter.capabilities
        adapter.validate_request(model, request, case["capability"])


def test_minimax_savevideo_ui_metadata_does_not_misclassify_real_mp4_or_change_history(monkeypatch):
    from types import SimpleNamespace

    from factory.adapters.comfy import ComfyAdapter
    from factory.adapters.minimax import Adapter

    history = {
        "outputs": {
            "save": {"images": [{"filename": "original.mp4", "type": "output"}], "animated": [True]}
        }
    }
    output = {"model": "minimax-h3", "outputs": history["outputs"], "native_history": history}
    prepared = SimpleNamespace(payload={"graph": {"save": {"class_type": "SaveVideo"}}})
    monkeypatch.setattr(
        ComfyAdapter, "normalize_output", lambda self, context, prepared, output: output
    )
    actual = Adapter().normalize_output(None, prepared, output)
    assert actual["outputs"]["save"] == {"videos": [{"filename": "original.mp4", "type": "output"}]}
    assert history["outputs"]["save"]["images"][0]["filename"] == "original.mp4"
    assert actual["native_history"] is history


@pytest.mark.parametrize(
    "transcript,passed", [("Factory makes useful tools", True), ("This is unrelated speech", False)]
)
def test_video_audio_oracle_requires_actual_spoken_words_and_matching_decoded_identity(
    monkeypatch, transcript, passed
):
    import hashlib
    import struct

    from benchmarks.native import video_audio

    samples = [0.02, -0.02] * 40
    decoded = struct.pack("<" + "f" * len(samples), *samples)
    raw = b"explicit software audio-verifier control, never a model output"
    monkeypatch.setattr(video_audio, "decode", lambda raw: (decoded, samples))
    expected = {
        "transcript": "Factory makes useful tools",
        "maximum_word_error_rate": 0.25,
        "required_phrases": ["factory", "useful tools"],
        "minimum_rms": 0.003,
        "maximum_clipped_fraction": 0.01,
        "maximum_av_duration_difference_seconds": 0.2,
        "verifier_pins_sha256": "software-control",
    }
    verification = {
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "decoded_samples_sha256": hashlib.sha256(decoded).hexdigest(),
        "sample_rate": 16000,
        "duration_seconds": len(samples) / 16000,
        "transcript": transcript,
        "verifier_pins_sha256": "software-control",
    }
    result = video_audio.semantic(expected, verification, raw, len(samples) / 16000)
    assert all(result["checks"].values()) == passed
    verification["decoded_samples_sha256"] = "different decoded data"
    assert (
        video_audio.semantic(expected, verification, raw, len(samples) / 16000)["checks"][
            "exact_generated_audio_identity"
        ]
        is False
    )


def test_video_audio_decoder_rejects_real_recorded_input_without_audio_track():
    import subprocess

    from benchmarks.native.video_audio import decode

    with pytest.raises(subprocess.CalledProcessError):
        decode((ROOT / "tests/fixtures/native/vtest-first2s.mp4").read_bytes())


@pytest.mark.parametrize(
    "decision,hashes,reviewed,passed",
    [
        ("PASS", ["a", "b", "c", "d"], True, True),
        ("FAIL", ["a", "b", "c", "d"], True, False),
        ("PASS", ["different"] * 4, True, False),
        ("PASS", ["a", "b", "c", "d"], False, False),
    ],
)
def test_video_content_review_is_bound_to_all_four_actual_collected_outputs(
    tmp_path, decision, hashes, reviewed, passed
):
    from benchmarks.native.video_content_review import CLASSIFICATION, apply

    (tmp_path / "validation").mkdir()
    (tmp_path / "validation/native_video_content_review.json").write_text(
        json.dumps(
            {
                "classification": CLASSIFICATION,
                "cases": {
                    "contract-run": {
                        "decision": decision,
                        "output_json_sha256": hashes,
                        "all_decoded_frames_reviewed": reviewed,
                        "observation": "Explicit software review control, never real inference evidence.",
                    }
                },
            }
        )
    )
    result = {
        "exit_gate": "PASS",
        "findings": [],
        "cases": [
            {
                "run_id": "contract-run",
                "status": "PASS",
                "native_inference_verified": True,
                "output_sha256": ["a", "b", "c", "d"],
                "checks": [],
            }
        ],
    }
    actual = apply(result, tmp_path)
    assert (actual["exit_gate"] == "PASS") == passed
    assert actual["cases"][0]["native_inference_verified"] == passed


def test_wan_protected_parameter_precisions_are_exact_manifest_data_from_pinned_meta_schema(
    routing_registry,
):
    model = routing_registry([("wan-2.2", [0])]).catalog.get("wan-2.2")
    report = json.loads((ROOT / "validation/native_wan_precision_schema.json").read_text())
    assert report["classification"] == "META_CONFIGURATION_AND_PARAMETER_SCHEMA_ONLY_NOT_INFERENCE"
    assert (
        report["weights_loaded"]
        is report["gpu_inference_executed"]
        is report["production_ready"]
        is False
    )
    expected = {
        name: spec["parameter_precision_overrides"] for name, spec in report["components"].items()
    }
    assert model.weights.model_dump(mode="json")["component_parameter_precisions"] == expected
    assert len(expected["transformer"]) == 95 and len(expected["text_encoder"]) == 24
    assert all(value == "fp32" for values in expected.values() for value in values.values())
    assert report["components"]["transformer"]["parameter_count"] == 4999787712
    assert report["components"]["text_encoder"]["parameter_count"] == 5680910336


def test_extension_anchor_is_actual_last_decoded_frame_even_for_low_fps_input(tmp_path):
    from benchmarks.native.video_semantic import decode
    from factory.adapters.wan import extract_last_frame

    source = ROOT / "tests/fixtures/native/vtest-first2s.mp4"
    anchor = tmp_path / "last.png"
    extract_last_frame(source, anchor)
    decoded, info = decode(source.read_bytes())
    assert info["frames"] == 4 and info["fps"] == 2
    assert Image.open(anchor).convert("RGB").tobytes() == decoded[-1].tobytes()


@pytest.mark.parametrize("field,value", [("width", 513), ("height", 720)])
def test_wan_rejects_dimensions_that_native_backend_would_silently_round(
    routing_registry, field, value
):
    from factory_contracts.errors import InvalidRequestError

    from factory.core.media import MediaRequest
    from factory.core.model_adapter import load_adapter

    model = routing_registry([("wan-2.2", [0])]).catalog.get("wan-2.2")
    request = MediaRequest(
        model="wan-2.2",
        capability="video.text_to_video",
        inputs={"prompt": "contract", field: value},
    )
    with pytest.raises(InvalidRequestError, match="multiples of 32"):
        load_adapter("wan").validate_request(model, request, request.capability)


def test_reference_fixture_lists_are_real_inline_image_bytes_before_native_staging():
    from benchmarks.native.wave_d import inline_fixture_inputs
    from factory.core.media import data_bytes

    spec = json.loads((ROOT / "benchmarks/native/golden_wave_d.json").read_text())["assets"][
        "astronaut_first"
    ]
    raw = (ROOT / spec["fixture_path"]).read_bytes()
    request = {"reference_images": ["@astronaut_first"], "seed": 7, "prompt": "contract"}
    result = inline_fixture_inputs(request, {"astronaut_first": spec}, {"astronaut_first": raw})
    assert data_bytes(result["reference_images"][0], "image") == raw
    assert request["reference_images"] == ["@astronaut_first"]
    assert result["seed"] == 7 and result["prompt"] == "contract"


def test_heavy_video_session_budget_comes_from_explicit_manifest_deadlines():
    from deployments.kaggle.native_wave_a import execution_budget

    assert execution_budget("vllm") == (4500, 7200)
    assert execution_budget("wan_t2v") == (9600, 13200)
    assert execution_budget("minimax_t2v") == (7200, 10800)
    assert execution_budget("minimax_conditioned") == (12000, 15600)


@pytest.mark.parametrize(
    "override",
    [
        {"inference_timeout_seconds": 999999},
        {"component_tiling": {"vae": {"tile_height": 16}}},
        {"extra_vram_headroom_gb": 0.2},
    ],
)
def test_inference_request_cannot_override_declared_runtime_policy(routing_registry, override):
    from factory_contracts.errors import InvalidRequestError

    from factory.core.media import MediaRequest
    from factory.core.model_adapter import load_adapter

    model = routing_registry([("minimax-h3", [0])]).catalog.get("minimax-h3")
    request = MediaRequest(
        model="minimax-h3",
        capability="video.text_to_video",
        inputs={"prompt": "contract", **override},
    )
    with pytest.raises(InvalidRequestError, match="Unsupported inference parameter"):
        load_adapter("minimax").validate_request(model, request, request.capability)


@pytest.mark.parametrize(
    "policy",
    [
        {"tile_height": 512, "tile_width": 512, "stride_height": 512, "stride_width": 384},
        {"tile_height": 512, "tile_width": 512, "stride_height": 0, "stride_width": 384},
        {"tile_height": 512, "tile_width": 512, "stride_height": 384, "stride_width": 384.5},
        {
            "tile_height": 512,
            "tile_width": 512,
            "stride_height": 384,
            "stride_width": 384,
            "automatic_retry": True,
        },
    ],
)
def test_component_tiling_rejects_unsafe_or_ignored_policy(policy):
    from factory_contracts.manifests import ComponentTilingSpec
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ComponentTilingSpec.model_validate(policy)


def test_unsupported_adapter_tiling_fails_before_weight_access(monkeypatch):
    from types import SimpleNamespace

    from factory_contracts.manifests import ModelRuntimeProfile
    from model_registry.backends.process import BackendError, ProcessBackend

    profile = ModelRuntimeProfile(
        gpu_count=1,
        min_vram_gb=4,
        preferred_backend="vllm",
        notes="Explicit software control",
        component_tiling={
            "vae": {
                "tile_height": 512,
                "tile_width": 512,
                "stride_height": 384,
                "stride_width": 384,
            }
        },
    )
    backend = ProcessBackend.__new__(ProcessBackend)
    backend.registry = SimpleNamespace(config=SimpleNamespace(hardware="kaggle-dual-t4"))
    backend.model = SimpleNamespace(runtime_profiles={"kaggle-dual-t4": profile})
    backend.binding = SimpleNamespace(adapter="qwen_embedding")

    def forbidden_weight_access(*args, **kwargs):
        pytest.fail("Unsupported component tiling reached checkpoint access")

    monkeypatch.setattr("model_registry.backends.process.verify_weights", forbidden_weight_access)
    with pytest.raises(BackendError, match="does not support"):
        backend.prepare()


@pytest.mark.parametrize("extra", [-0.001, float("nan"), float("inf"), 0.0001])
def test_extra_vram_reservation_rejects_invalid_or_unrepresentable_values(extra):
    from factory_contracts.manifests import ModelRuntimeProfile
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ModelRuntimeProfile(
            gpu_count=1,
            min_vram_gb=4,
            preferred_backend="comfyui",
            notes="Software control",
            extra_vram_headroom_gb=extra,
        )


def test_unsupported_backend_extra_headroom_fails_before_checkpoint_access(monkeypatch):
    from types import SimpleNamespace

    from factory_contracts.manifests import ModelRuntimeProfile
    from model_registry.backends.process import BackendError, ProcessBackend

    profile = ModelRuntimeProfile(
        gpu_count=1,
        min_vram_gb=4,
        preferred_backend="vllm",
        notes="Software control",
        extra_vram_headroom_gb=0.2,
    )
    backend = ProcessBackend.__new__(ProcessBackend)
    backend.registry = SimpleNamespace(config=SimpleNamespace(hardware="kaggle-dual-t4"))
    backend.model = SimpleNamespace(runtime_profiles={"kaggle-dual-t4": profile})
    backend.adapter = SimpleNamespace(supports_extra_vram_headroom=False)

    def forbidden_weight_access(*args, **kwargs):
        pytest.fail("Unsupported backend headroom reached checkpoint access")

    monkeypatch.setattr("model_registry.backends.process.verify_weights", forbidden_weight_access)
    with pytest.raises(BackendError, match="does not support extra VRAM headroom"):
        backend.prepare()
