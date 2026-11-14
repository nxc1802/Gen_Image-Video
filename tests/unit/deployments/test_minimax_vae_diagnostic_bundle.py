"""Software packaging controls only; never evidence of GPU/model inference."""

import ast
import base64
import io
import json
import zipfile

import pytest
import yaml

from deployments.kaggle import native_wave_a
from deployments.kaggle.minimax_vae_diagnostic import KIND, render


@pytest.mark.parametrize("native_canvas,sampler_control", [(False, False), (True, False), (False, True)])
def test_diagnostic_is_bounded_explicit_and_does_not_mutate_certification_batches(native_canvas, sampler_control):
    batches = native_wave_a.VIDEO_BATCHES.copy()
    source = render(native_canvas=native_canvas, sampler_control=sampler_control)
    assert len(source.encode()) < 1_000_000
    compile(source, KIND, "exec")
    assert native_wave_a.VIDEO_BATCHES == batches
    assert "benchmarks.native.minimax_vae_diagnostic" in source
    classification = "REAL_NATIVE_SAMPLER_GPU_DIAGNOSIS_NOT_V3" if sampler_control else "REAL_NATIVE_CANVAS_GPU_DIAGNOSIS_NOT_V3" if native_canvas else "REAL_GPU_SAME_LATENT_VAE_DIAGNOSIS_NOT_V3"
    assert classification in source
    assert ("'--native-canvas'" in source) is native_canvas
    assert ("'--sampler-control'" in source) is sampler_control
    data = ast.literal_eval(ast.parse(source).body[0].value)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(data))) as archive:
        names = set(archive.namelist())
        assert "benchmarks/native/minimax_vae_diagnostic.py" in names
        assert "benchmarks/native/minimax_vae_diagnostic_nodes.py" in names
        assert {n for n in names if n.startswith("tests/fixtures/native/")} == {
            "tests/fixtures/native/astronaut_first.png"
        }
    with zipfile.ZipFile(io.BytesIO(native_wave_a.bundle("minimax_audio"))) as archive:
        assert not any("minimax_vae_diagnostic" in n for n in archive.namelist())


def test_native_canvas_is_valid_only_in_the_explicit_diagnostic_manifest():
    from factory_contracts.errors import InvalidRequestError
    from model_registry.deployment import ModelCatalog

    from benchmarks.native.minimax_vae_diagnostic import diagnostic_manifest, native_canvas_case
    from factory.adapters.minimax import Adapter
    from factory.core.media import MediaRequest

    root = native_wave_a.ROOT
    base = next(c for c in json.loads((root / "benchmarks/native/golden_wave_d.json").read_text())["cases"] if c["id"] == "minimax-i2v-astronaut-native-temporal-v2")
    candidate = native_canvas_case(base)
    assert candidate["expected"] == base["expected"] and candidate["assets"] == base["assets"]
    assert candidate["seed_sequence"] == base["seed_sequence"]
    assert {k: v for k, v in candidate["request"].items() if k not in {"width", "height"}} == {k: v for k, v in base["request"].items() if k not in {"width", "height"}}
    selected = ModelCatalog(root / "manifests").get("minimax-h3")
    request = MediaRequest(model="minimax-h3", capability=base["capability"], inputs=candidate["request"])
    with pytest.raises(InvalidRequestError):
        Adapter().validate_request(selected, request, request.capability)
    original = yaml.safe_load((root / "manifests/models/minimax-h3.yaml").read_text())
    configured = diagnostic_manifest(original, "fp16", native_canvas=True)
    assert original["parameters"]["max_height"]["default"] == 720
    selected = type(selected).model_validate(configured)
    Adapter().validate_request(selected, request, request.capability)
    assert selected.runtime_profiles["kaggle-dual-t4"].inference_timeout_seconds == 5400
    assert selected.weights.files == ModelCatalog(root / "manifests").get("minimax-h3").weights.files


def test_two_diagnostic_changes_cannot_be_silently_combined():
    batches = native_wave_a.VIDEO_BATCHES.copy()
    with pytest.raises(AssertionError):
        render(native_canvas=True, sampler_control=True)
    assert native_wave_a.VIDEO_BATCHES == batches
