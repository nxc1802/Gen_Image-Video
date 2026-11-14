"""Decision architecture/protocol software tests; no model, GPU or inference proof."""
import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import pytest
import yaml

from factory_contracts.deployment import DeploymentConfig, DeploymentEntry
from factory_contracts.errors import InvalidRequestError, ModelNotLoadedError, ModelUnavailableError
from factory.core.model_adapter import ExecutionContext, load_adapter
from factory.engines.decision import DecisionEngine
from factory.engines.decision.contracts import DecisionRequest, DecisionTask
from model_registry.deployment import DeploymentRegistry, ModelCatalog

pytestmark = pytest.mark.contract_mock
ROOT = Path(__file__).resolve().parents[2]


def mocked_session(registry):
    backends = {
        binding.model: SimpleNamespace(registry=registry, binding=binding, token="decision-contract-key", health=lambda: True)
        for binding in registry.bindings()
    }
    return SimpleNamespace(registry=registry, backends=backends, healthy_models=lambda: tuple(backends))


@pytest.fixture
def decision_runtime(routing_registry):
    registry = routing_registry([("qwen-decision", [1])])
    session = mocked_session(registry)
    context = ExecutionContext(registry, session, registry.assignment("qwen-decision"), registry.catalog.get("qwen-decision"))
    return registry, session, context


def request_for(task, **kwargs):
    fields = {"model": "qwen-decision", "task": task, "context": "Contract-only context"}
    if task == DecisionTask.PREDICATE:
        fields["hypothesis"] = "Contract-only hypothesis"
    elif task == DecisionTask.CHOICE:
        fields["choices"] = ["billing", "returns"]
    else:
        fields["scale"] = [50.0, 150.0]
    return DecisionRequest(**{**fields, **kwargs})


def usage():
    # Explicit fixture values, never an inference/token-count observation.
    return {"input_tokens": 20, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 20}


def native_output(prepared, model="qwen-decision"):
    question = prepared.metadata["question"]
    if question["type"] == "predicate":
        answer = {"type": "predicate", "name": "decision", "probability": 0.73}
    elif question["type"] == "choice":
        values = [row["value"] for row in question["choices"]]
        probabilities = [0.8] + [0.2 / (len(values) - 1)] * (len(values) - 1)
        answer = {"type": "choice", "name": "decision", "choice": values[0], "confidence": 0.6,
                  "probabilities": [{"value": value, "probability": probability} for value, probability in zip(values, probabilities)]}
    else:
        rows = [{"value": index, "label": level["label"], "probability": 0.35 if index == 7 else 0.65 if index == 8 else 0.0}
                for index, level in enumerate(question["levels"])]
        answer = {"type": "score", "name": "decision", "score": 7.65, "probabilities": rows, "confidence": 0.6}
    return {"model": model, "answers": [answer], "usage": usage()}


def prepared_for(context, task):
    adapter = load_adapter(context.binding.adapter)
    request = request_for(task)
    return adapter, adapter.prepare(context, request, task)


def test_healthy_decision_backend_does_not_authorize_inference(decision_runtime, monkeypatch):
    registry, session, _ = decision_runtime
    def forbidden(*args, **kwargs):
        pytest.fail("Uncertified decision request reached inference")
    monkeypatch.setattr(ExecutionContext, "post_json", forbidden)
    with pytest.raises(ModelNotLoadedError):
        DecisionEngine(registry, session).evaluate(request_for(DecisionTask.PREDICATE))


@pytest.mark.parametrize("task", list(DecisionTask))
def test_native_answers_are_preserved_without_invented_calibration(decision_runtime, task):
    _, _, context = decision_runtime
    adapter, prepared = prepared_for(context, task)
    output = native_output(prepared)
    result = adapter.normalize_output(context, prepared, output)
    assert result.answer.model_dump() == output["answers"][0]
    assert result.usage.model_dump() == output["usage"]
    assert not {"logprobs", "calibration_metrics", "decision", "selected_choice"}.intersection(result.answer.model_dump())
    if task == DecisionTask.SCORE:
        assert result.answer.score == 7.65  # Native expected level, not its label.
        assert math.isclose(result.score_mapping.value, 135.0)
        assert result.score_mapping.method == "linear_map_of_native_expected_level"
    else:
        assert result.score_mapping is None


def test_native_protocol_and_transport_are_fixed_by_the_deployment(decision_runtime, monkeypatch):
    registry, session, _ = decision_runtime
    monkeypatch.setenv("VLLM_DECISION_URL", "http://unselected-backend.invalid:9999")
    monkeypatch.setenv("DECISION_UPSTREAM_URL", "http://another-unselected-backend.invalid:9998")
    monkeypatch.setattr(registry, "require_active", lambda *args: None)  # Mock authorization, software only.
    calls = []
    actual_client = httpx.Client
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"model": "qwen-decision", "answers": [{"type": "predicate", "name": "decision", "probability": 0.73}], "usage": usage()})
    def client(**kwargs):
        assert kwargs["follow_redirects"] is False and kwargs["trust_env"] is False
        return actual_client(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr("factory.core.model_adapter.httpx.Client", client)
    result = DecisionEngine(registry, session).evaluate(request_for(DecisionTask.PREDICATE))
    assert result.answer.probability == 0.73 and len(calls) == 1
    assert str(calls[0].url) == "http://127.0.0.1:9200/v1/decisions"
    assert calls[0].headers["authorization"] == "Bearer decision-contract-key"
    assert json.loads(calls[0].content) == {
        "model": "qwen-decision", "input": "Contract-only context",
        "questions": [{"type": "predicate", "name": "decision", "instructions": "Contract-only hypothesis"}],
    }
    assert registry.assignment("qwen-decision").devices == (1,)


@pytest.mark.parametrize("change", ["wrong_model", "wrong_count", "wrong_question", "wrong_type", "missing_usage", "fake_logprobs",
                                  "nan_probability", "string_probability", "boolean_probability", "invalid_probability", "boolean_token_count", "inconsistent_usage"])
def test_malformed_predicate_result_fails_closed(decision_runtime, change):
    _, _, context = decision_runtime
    adapter, prepared = prepared_for(context, DecisionTask.PREDICATE)
    output = native_output(prepared)
    answer = output["answers"][0]
    if change == "wrong_model": output["model"] = "another-checkpoint"
    elif change == "wrong_count": output["answers"].append(copy.deepcopy(answer))
    elif change == "wrong_question": answer["name"] = "other-question"
    elif change == "wrong_type": answer["type"] = "choice"
    elif change == "missing_usage": output.pop("usage")
    elif change == "fake_logprobs": answer["logprobs"] = {"true": -0.02, "false": -3.91}
    elif change == "nan_probability": answer["probability"] = float("nan")
    elif change == "string_probability": answer["probability"] = "0.73"
    elif change == "boolean_probability": answer["probability"] = True
    elif change == "invalid_probability": answer["probability"] = 1.1
    elif change == "boolean_token_count": output["usage"]["input_tokens"] = True
    elif change == "inconsistent_usage": output["usage"]["total_tokens"] = 25
    with pytest.raises(ModelUnavailableError): adapter.normalize_output(context, prepared, output)


@pytest.mark.parametrize("change", ["missing_option", "duplicate_option", "wrong_option", "wrong_winner", "bad_sum", "bad_confidence"])
def test_invalid_choice_distribution_is_not_repaired(decision_runtime, change):
    _, _, context = decision_runtime
    adapter, prepared = prepared_for(context, DecisionTask.CHOICE)
    output = native_output(prepared)
    answer = output["answers"][0]
    if change == "missing_option": answer["probabilities"].pop()
    elif change == "duplicate_option": answer["probabilities"][1]["value"] = "billing"
    elif change == "wrong_option": answer["choice"] = "other-option"
    elif change == "wrong_winner": answer["choice"] = "returns"
    elif change == "bad_sum": answer["probabilities"][0]["probability"] = 0.9
    elif change == "bad_confidence": answer["confidence"] = 0.99
    with pytest.raises(ModelUnavailableError): adapter.normalize_output(context, prepared, output)


@pytest.mark.parametrize("change", ["wrong_label", "duplicate_level", "out_of_range_level", "fake_scaled_score", "wrong_expectation", "nan_score", "bad_sum", "bad_confidence"])
def test_invalid_score_distribution_is_not_repaired(decision_runtime, change):
    _, _, context = decision_runtime
    adapter, prepared = prepared_for(context, DecisionTask.SCORE)
    output = native_output(prepared)
    answer = output["answers"][0]
    if change == "wrong_label": answer["probabilities"][0]["label"] = "another-label"
    elif change == "duplicate_level": answer["probabilities"][1]["value"] = 0
    elif change == "out_of_range_level": answer["probabilities"][0]["value"] = 10
    elif change == "fake_scaled_score": answer["score"] = 135.0
    elif change == "wrong_expectation": answer["score"] = 5.0
    elif change == "nan_score": answer["score"] = float("nan")
    elif change == "bad_sum": answer["probabilities"][0]["probability"] = 0.1
    elif change == "bad_confidence": answer["confidence"] = 0.99
    with pytest.raises(ModelUnavailableError): adapter.normalize_output(context, prepared, output)


@pytest.mark.parametrize("task,fields", [
    (DecisionTask.PREDICATE, {"hypothesis": ""}),
    (DecisionTask.PREDICATE, {"choices": ["unexpected", "choices"]}),
    (DecisionTask.CHOICE, {"choices": ["same", "same"]}),
    (DecisionTask.CHOICE, {"choices": ["", "choice"]}),
    (DecisionTask.CHOICE, {"choices": [str(i) for i in range(27)]}),
    (DecisionTask.SCORE, {"scale": [0.0, float("inf")]}),
    (DecisionTask.SCORE, {"scale": [5.0, 5.0]}),
    (DecisionTask.SCORE, {"scale": [1.0, math.nextafter(1.0, 2.0)]}),
])
def test_invalid_task_fields_fail_before_inference(decision_runtime, task, fields):
    registry, session, _ = decision_runtime
    with pytest.raises(InvalidRequestError):
        DecisionEngine(registry, session).evaluate(request_for(task, **fields))


@pytest.mark.parametrize("task", list(DecisionTask))
def test_refusal_is_preserved_and_has_no_derived_decision(decision_runtime, task):
    _, _, context = decision_runtime
    adapter, prepared = prepared_for(context, task)
    output = {"model": "qwen-decision", "answers": [{"type": "refusal", "name": "decision"}], "usage": usage()}
    result = adapter.normalize_output(context, prepared, output)
    assert result.answer.type == "refusal" and result.score_mapping is None


def test_request_alias_is_normalized_only_at_legacy_service_ingress():
    from services.decision.src.schemas import DecisionRequest as IngressRequest
    from pydantic import ValidationError
    request = IngressRequest(model="qwen-decision", task="predicate", context="context", hypothesis="hypothesis")
    assert request.task == DecisionTask.PREDICATE
    with pytest.raises(ValidationError):
        DecisionRequest(model="qwen-decision", task="predicate", context="context", hypothesis="hypothesis")


@pytest.mark.parametrize("field,value", [("devices", [0]), ("backend", "another_backend"), ("scale", [False, True]), ("scale", ["0", "1"])])
def test_request_cannot_choose_deployment_or_coerce_scale_types(field, value):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        request_for(DecisionTask.SCORE, **{field: value})


def test_large_finite_scale_does_not_overflow_during_level_preparation(decision_runtime):
    _, _, context = decision_runtime
    request = request_for(DecisionTask.SCORE, scale=[0, 1e308])
    adapter = load_adapter(context.binding.adapter)
    adapter.validate_request(context.model, request, request.task)
    prepared = adapter.prepare(context, request, request.task)
    assert len({row["label"] for row in prepared.metadata["question"]["levels"]}) == 10


def test_new_decision_model_and_adapter_need_no_engine_core_edits(tmp_path, monkeypatch):
    data = yaml.safe_load((ROOT / "manifests/models/qwen-decision.yaml").read_text())
    data["metadata"]["id"] = "contract-new-decision"
    data["aliases"] = []
    data["backend"]["adapter"] = "contract_decision_adapter"
    directory = tmp_path / "catalog/models"
    directory.mkdir(parents=True)
    (directory / "model.yaml").write_text(yaml.safe_dump(data))
    from factory.adapters.qwen_decision import Adapter
    monkeypatch.setitem(sys.modules, "factory.adapters.contract_decision_adapter", SimpleNamespace(Adapter=Adapter))
    registry = DeploymentRegistry(DeploymentConfig(deployments=(DeploymentEntry(model="contract-new-decision", devices=(1,)),)), ModelCatalog(directory.parent))
    monkeypatch.setattr(registry, "require_active", lambda *args: None)  # Software authorization only.
    def post(context, path, payload):
        assert path == "/v1/decisions" and payload["model"] == "contract-new-decision"
        return {"model": payload["model"], "answers": [{"type": "predicate", "name": "decision", "probability": 0.4}], "usage": usage()}
    monkeypatch.setattr(ExecutionContext, "post_json", post)
    request = request_for(DecisionTask.PREDICATE, model="contract-new-decision")
    result = DecisionEngine(registry, mocked_session(registry)).evaluate(request)
    assert result.model == request.model and result.answer.probability == 0.4


def test_wrong_session_cannot_supply_a_decision_backend(decision_runtime, monkeypatch):
    registry, session, _ = decision_runtime
    monkeypatch.setattr(registry, "require_active", lambda *args: None)
    session.registry = object()
    with pytest.raises(ModelUnavailableError, match="exact deployment"):
        DecisionEngine(registry, session).evaluate(request_for(DecisionTask.PREDICATE))


def test_missing_native_endpoint_never_falls_back_to_chat(decision_runtime, monkeypatch):
    registry, session, _ = decision_runtime
    monkeypatch.setattr(registry, "require_active", lambda *args: None)
    actual_client = httpx.Client
    seen = []
    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(404, json={"error": "contract fixture endpoint unavailable"})
    monkeypatch.setattr("factory.core.model_adapter.httpx.Client", lambda **kwargs: actual_client(transport=httpx.MockTransport(handler), **kwargs))
    with pytest.raises(ModelUnavailableError):
        DecisionEngine(registry, session).evaluate(request_for(DecisionTask.PREDICATE))
    assert seen == ["http://127.0.0.1:9200/v1/decisions"]


def test_retained_service_uses_canonical_engine_and_denies_unvalidated_model(decision_runtime, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from services.decision.src.app import app
    registry, _, _ = decision_runtime
    path = tmp_path / "deployment.yaml"
    path.write_text(yaml.safe_dump(registry.config.model_dump(mode="json")))
    monkeypatch.setenv("FACTORY_DEPLOYMENT_FILE", str(path))
    monkeypatch.setenv("FACTORY_MANIFESTS_DIR", str(ROOT / "manifests"))
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("FACTORY_API_KEY", "decision-contract-only-key")
    with TestClient(app) as client:
        assert isinstance(app.state.engine, DecisionEngine)
        response = client.post("/v1/decisions", json={"model": "qwen-decision", "task": "predicate", "context": "context", "hypothesis": "hypothesis"},
                               headers={"Authorization": "Bearer decision-contract-only-key"})
    assert response.status_code == 503 and "answer" not in response.json()
