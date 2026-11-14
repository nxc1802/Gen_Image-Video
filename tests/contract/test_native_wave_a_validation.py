"""Semantic validator/harness contracts only. No real model or GPU is executed."""

import copy
import json
from pathlib import Path
import pytest
from benchmarks.native.wave_a import semantic, request_for, digest
from deployments.kaggle.native_wave_a import bundle, render_kernel

pytestmark = pytest.mark.contract_mock
ROOT = Path(__file__).resolve().parents[2]
SUITE = json.loads((ROOT / "benchmarks/native/golden_wave_a.json").read_text())
CASES = {case["id"]: case for case in SUITE["cases"]}


def test_wave_a_has_exact_one_warmup_three_measured_and_all_declared_targets():
    assert SUITE["performance"] == {"warmup": 1, "measured": 3}
    assert len(CASES) == 10
    for case in CASES.values():
        request = request_for(case)
        assert request.model == case["model"]
        assert digest(case) == digest(json.loads(json.dumps(case)))


@pytest.mark.parametrize(
    "positive,negative,passed",
    [(0.9, 0.1, True), (0.9, 0.9, False), (0.1, 0.1, False), (0.5, 0.5, False)],
)
def test_predicate_requires_known_true_and_false_answers(positive, negative, passed):
    output = {"answers": [{"probability": positive}, {"probability": negative}]}
    result = semantic(CASES["decision-predicate-facts"], output)
    assert (result["status"] == "PASS") is passed
    assert result["metrics"]["brier_score"] == pytest.approx(
        ((1 - positive) ** 2 + negative**2) / 2
    )


@pytest.mark.parametrize("choice,passed", [("billing", True), ("shipping", False)])
def test_choice_checks_expected_department(choice, passed):
    assert (
        semantic(CASES["decision-choice-billing"], {"answers": [{"choice": choice}]})["status"]
        == "PASS"
    ) is passed


@pytest.mark.parametrize(
    "scores,passed", [([0.1, 1.9], True), ([1.9, 0.1], False), ([1, 1], False)]
)
def test_score_checks_semantic_ordering(scores, passed):
    result = semantic(CASES["decision-score-damage"], {"answers": [{"score": x} for x in scores]})
    assert (result["status"] == "PASS") is passed


@pytest.mark.parametrize("reverse", [False, True])
def test_embedding_checks_real_query_relevance_not_just_shape(reverse):
    case = copy.deepcopy(CASES["embedding-france-retrieval"])
    case["expected"]["dimension"] = 2
    positive, negative = ([1.0, 0.0], [0.0, 1.0]) if not reverse else ([0.0, 1.0], [1.0, 0.0])
    output = {"data": [{"embedding": v} for v in [[1.0, 0.0], positive, negative]]}
    assert (semantic(case, output)["status"] == "PASS") is (not reverse)


@pytest.mark.parametrize("best,passed", [(1, True), (2, False)])
def test_reranker_rejects_wrong_relevant_document(best, passed):
    output = {
        "results": [
            {"index": i, "relevance_score": 0.9 if i == best else 0.1}
            for i in [best, *[n for n in range(3) if n != best]]
        ]
    }
    assert (semantic(CASES["rerank-france-capital"], output)["status"] == "PASS") is passed


@pytest.mark.parametrize(
    "case_id,output,passed",
    [
        ("text-arithmetic", {"choices": [{"message": {"content": "42"}}]}, True),
        ("text-arithmetic", {"choices": [{"message": {"content": "43"}}]}, False),
        ("text-structured-city", {"choices": [{"message": {"content": '{"city":"Paris"}'}}]}, True),
        (
            "text-structured-city",
            {"choices": [{"message": {"content": '{"city":"Berlin"}'}}]},
            False,
        ),
        (
            "text-tool-weather",
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "lookup_weather",
                                        "arguments": '{"city":"Paris"}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            True,
        ),
        (
            "text-tool-weather",
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "lookup_weather",
                                        "arguments": '{"city":"Berlin"}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            False,
        ),
        (
            "text-streaming",
            {
                "events": [{"choices": [{"delta": {"content": "FACTORY_STREAM_OK"}}]}],
                "native_done": True,
            },
            True,
        ),
        (
            "text-streaming",
            {
                "events": [{"choices": [{"delta": {"content": "FACTORY_STREAM_OK"}}]}],
                "native_done": False,
            },
            False,
        ),
    ],
)
def test_text_checks_meaning_tool_arguments_schema_and_real_stream_end(case_id, output, passed):
    assert (semantic(CASES[case_id], output)["status"] == "PASS") is passed


def test_kernel_bundle_and_scripts_are_reproducible_and_use_selected_venv_path():
    assert bundle() == bundle()
    for kind in ["vllm", "text"]:
        script = render_kernel(kind)
        compile(script, "native_kernel", "exec")
        assert "PATH=str(python.parent)+os.pathsep" in script
        assert (
            "NvidiaTeslaT4" not in script
        )  # Accelerator is selected by push; driver checks actual inventory.
        assert "FULL_FACTORY_E2E_PASS" not in script


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "undeclared"])
def test_native_gate_requires_complete_unambiguous_manifest_coverage(tmp_path, mutation):
    import shutil
    from benchmarks.native.wave_a_gate import evaluate

    shutil.copytree(ROOT / "manifests", tmp_path / "manifests")
    suite = copy.deepcopy(SUITE)
    if mutation == "missing":
        suite["cases"].pop()
    elif mutation == "duplicate":
        suite["cases"].append(copy.deepcopy(suite["cases"][0]))
    else:
        suite["cases"][0]["capability"] = "responses.image_chat"
    destination = tmp_path / "benchmarks/native/golden_wave_a.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(suite))
    report = evaluate(tmp_path)
    expected = "Duplicate golden" if mutation == "duplicate" else "Golden coverage differs"
    assert any(expected in finding for finding in report["findings"])
    assert report["exit_gate"] == "NOT_MET"
    assert report["production_ready"] is False


def test_qwen2_plugin_registers_only_native_strategy(monkeypatch):
    import sys
    from types import SimpleNamespace
    from factory.adapters.qwen_decision.vllm_plugin import register

    native = object()
    registry = {}
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.openai.decisions.strategies",
        SimpleNamespace(READ_STRATEGIES=registry, NextTokenStrategy=native),
    )
    register()
    register()
    assert registry == {"Qwen2ForCausalLM": native}
    registry["Qwen2ForCausalLM"] = object()
    with pytest.raises(RuntimeError, match="Conflicting"):
        register()


def test_qwen2_backend_extension_is_private_discoverable_and_hash_tracked(tmp_path):
    import hashlib
    from types import SimpleNamespace
    from importlib.metadata import distributions
    from factory.adapters.qwen_decision import Adapter

    backend = SimpleNamespace(directory=tmp_path, adapter_assets={}, _files={})
    env = Adapter().backend_environment(backend)
    assert env["VLLM_PLUGINS"] == "factory_qwen2_decision"
    folder = tmp_path / "native_plugins"
    entries = [e for dist in distributions(path=[str(folder)]) for e in dist.entry_points]
    assert len(entries) == 1 and entries[0].group == "vllm.general_plugins"
    assert entries[0].value == "factory_qwen2_decision:register"
    for relative, sha in backend.adapter_assets.items():
        path = tmp_path / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha and str(path) in backend._files


def test_embedding_matryoshka_config_comes_from_selected_launch_adapter():
    from model_registry.adapters.launch import EmbeddingLaunchAdapter

    args = EmbeddingLaunchAdapter().extra_args(None)
    assert json.loads(args[args.index("--hf-overrides") + 1]) == {"is_matryoshka": True}
