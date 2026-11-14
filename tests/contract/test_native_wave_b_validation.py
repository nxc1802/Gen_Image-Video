"""Software-only native validation contracts; no model or GPU is executed."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from benchmarks.native.wave_b import semantic, word_error_rate
from deployments.kaggle.native_wave_a import render_kernel
from factory.adapters.qwen_tts import Adapter
from model_registry.deployment import ModelCatalog

pytestmark = pytest.mark.contract_mock
ROOT = Path(__file__).resolve().parents[2]
SUITE = json.loads((ROOT / "benchmarks/native/golden_wave_b.json").read_text())
CASES = {c["id"]: c for c in SUITE["cases"]}


@pytest.mark.parametrize(
    "text,passed",
    [
        (CASES["asr-jfk-real-speech"]["expected"]["transcript"], True),
        ("A completely unrelated transcript from a different recording.", False),
        ("", False),
    ],
)
def test_asr_requires_real_ground_truth_meaning(text, passed):
    assert (semantic(CASES["asr-jfk-real-speech"], {"text": text})["status"] == "PASS") is passed


def test_word_error_rate_counts_insertions_deletions_and_substitutions():
    assert word_error_rate("one two three", "one two three") == 0
    assert word_error_rate("one two three", "one four") == pytest.approx(2 / 3)
    assert word_error_rate("one two", "zero one two") == 0.5


@pytest.mark.parametrize(
    "case_id,text,passed",
    [
        ("vlm-astronaut-suit", "Orange.", True),
        ("vlm-astronaut-suit", "Blue", False),
        ("vlm-document-title", "Region-based segmentation", True),
        ("vlm-document-title", "Image classification", False),
        ("vlm-head-grounding-v2", '{"bbox":[285,20,600,365]}', True),
        ("vlm-head-grounding-v2", '```json\n{"bbox":[285,20,600,365]}\n```', True),
        ("vlm-head-grounding-v2", '{"bbox":[0,0,1000,1000]}', False),
        ("vlm-head-grounding-v2", '```json\n{"bbox":[0,0,1000,1000]}\n```', False),
        ("vlm-pedestrian-video", "People are walking outdoors.", True),
        ("vlm-pedestrian-video", "Cars are driving on the road.", False),
        ("vlm-pedestrian-video", "A person is standing still.", False),
    ],
)
def test_visual_oracles_reject_semantically_wrong_outputs(case_id, text, passed):
    output = {"choices": [{"message": {"content": text}}]}
    assert (semantic(CASES[case_id], output)["status"] == "PASS") is passed


def test_actual_video_fixture_is_pinned_input_not_inference_evidence():
    spec = SUITE["assets"]["pedestrians"]
    raw = (ROOT / spec["fixture_path"]).read_bytes()
    assert len(raw) == spec["size_bytes"]
    assert hashlib.sha256(raw).hexdigest() == spec["sha256"]
    assert spec["frames"] == 4


@pytest.mark.parametrize(
    "failure", ["unverified", "wrong_words", "wrong_speaker", "wrong_waveform_hash"]
)
def test_generated_audio_requires_actual_semantics_and_reference_identity(failure):
    import base64
    import io
    import wave

    case = CASES["tts-base-voice-cloning"]
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(bytes(64000))
    raw = stream.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    output = {
        "model": "qwen-tts",
        "artifacts": [
            {
                "type": "audio",
                "mime_type": "audio/wav",
                "size_bytes": len(raw),
                "sha256": sha,
                "data_base64": base64.b64encode(raw).decode(),
            }
        ],
    }
    if failure != "unverified":
        output["verification"] = {
            "input_sha256": "0" * 64 if failure == "wrong_waveform_hash" else sha,
            "transcript": "A different sentence."
            if failure == "wrong_words"
            else case["expected"]["transcript"],
            "duration_seconds": 2,
            "reference_sha256": case["expected"]["reference_audio_sha256"],
            "verifier_pins_sha256": case["expected"]["verifier_pins_sha256"],
            "speaker_reference_embedding": [1.0] + [0.0] * 511,
            "speaker_output_embedding": ([0.0, 1.0] + [0.0] * 510)
            if failure == "wrong_speaker"
            else [1.0] + [0.0] * 511,
        }
    assert semantic(case, output)["status"] != "PASS"


def test_wave_b_kernel_uses_explicit_consistent_asr_stack():
    for kind in ["asr", "vlm"]:
        code = render_kernel(kind)
        compile(code, "native_" + kind, "exec")
        assert "venv_args+=['--system-site-packages']" not in code
        assert "'--torch-backend','cu130'" in code
        assert "import torch,torchvision,torchaudio" in code


def test_tts_stage_config_is_fp16_fixed_gpu_and_manifest_budgeted(tmp_path, monkeypatch):
    package = tmp_path / "fake_omni"
    (package / "deploy").mkdir(parents=True)
    source = {
        "model_runner": "v2",
        "stages": [
            {"stage_id": 0, "dtype": "bfloat16", "max_num_seqs": 64},
            {"stage_id": 1, "dtype": "bfloat16", "max_num_seqs": 64},
        ],
        "platforms": {"rocm": {"model_runner": "v1"}},
    }
    (package / "deploy/qwen3_tts.yaml").write_text(yaml.safe_dump(source))
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: SimpleNamespace(origin=str(package / "__init__.py")),
    )
    catalog = ModelCatalog(ROOT / "manifests")
    backend = SimpleNamespace(
        model=catalog.get("qwen-tts"),
        binding=SimpleNamespace(devices=(0,)),
        registry=SimpleNamespace(
            config=SimpleNamespace(hardware="kaggle-dual-t4"), catalog=catalog
        ),
        directory=tmp_path,
        adapter_assets={},
        _files={},
    )
    result = yaml.safe_load(Adapter().backend_deploy_config(backend).read_text())
    assert result["model_runner"] == "v1" and "platforms" not in result
    assert all(
        s["dtype"] == "float16"
        and s["devices"] == "0"
        and s["max_num_seqs"] == 1
        and s["enforce_eager"]
        and not s["async_scheduling"]
        for s in result["stages"]
    )
    total_memory = catalog.loader.load_hardware_topologies()["kaggle-dual-t4"].gpus[0].total_vram_gb
    assert sum(s["gpu_memory_utilization"] for s in result["stages"]) * total_memory <= 8.002
    for name, digest in backend.adapter_assets.items():
        path = tmp_path / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert str(path) in backend._files
    source["stages"] = [copy.deepcopy(source["stages"][0])]
    (package / "deploy/qwen3_tts.yaml").write_text(yaml.safe_dump(source))
    with pytest.raises(RuntimeError, match="stage layout"):
        Adapter().backend_deploy_config(backend)


@pytest.mark.parametrize("failure", [None, "source_hash", "patch_count"])
def test_native_fp16_overlays_are_private_and_reject_source_drift(tmp_path, monkeypatch, failure):
    from importlib.metadata import distributions
    from factory.adapters.qwen_tts import native_compat

    package = tmp_path / "fake_omni"
    source = package / "model_executor/models/qwen3_tts"
    source.mkdir(parents=True)
    raw = b"DTYPE = 'bf16'\n"
    (source / "sample.py").write_bytes(raw)
    monkeypatch.setattr(
        native_compat,
        "SOURCES",
        {
            "sample.py": {
                "sha256": "0" * 64 if failure == "source_hash" else hashlib.sha256(raw).hexdigest(),
                "replacements": [("'bf16'", "'fp16'", 2 if failure == "patch_count" else 1)],
            }
        },
    )
    monkeypatch.setattr(
        native_compat,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(package / "__init__.py")),
    )
    directory = tmp_path / "backend"
    directory.mkdir()
    backend = SimpleNamespace(directory=directory, adapter_assets={}, _files={})
    if failure:
        with pytest.raises(RuntimeError):
            native_compat.stage_environment(backend)
        return
    env = native_compat.stage_environment(backend)
    assert env["VLLM_PLUGINS"] == "factory_qwen_tts_fp16,vllm_omni_register_models"
    assert (source / "sample.py").read_bytes() == raw
    assert (directory / "native_plugins/overlays/sample.py").read_text() == "DTYPE = 'fp16'\n"
    entries = [
        entry
        for dist in distributions(path=[str(directory / "native_plugins")])
        for entry in dist.entry_points
    ]
    assert len(entries) == 1 and entries[0].group == "vllm.general_plugins"
    for name, sha in backend.adapter_assets.items():
        path = directory / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha and str(path) in backend._files
