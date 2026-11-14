"""New public contracts with actual adapters and explicit HTTP/native doubles."""

import asyncio
import json
from types import SimpleNamespace
import httpx
import pytest
from factory.core.model_adapter import ExecutionContext, load_adapter
from factory.engines.responses.contracts import (
    ResponseApiRequest,
    ChatCompletionRequest,
    ChatMessage,
)
from factory.engines.decision.contracts import PublicDecisionRequest
from factory_contracts.errors import (
    InvalidRequestError,
    ModelUnavailableError,
    UnsupportedCapabilityError,
)

pytestmark = pytest.mark.contract_mock


def context(registry, mid):
    binding = registry.assignment(mid)
    backend = SimpleNamespace(
        registry=registry, binding=binding, health=lambda: True, token="contract-private"
    )
    session = SimpleNamespace(
        registry=registry, backends={mid: backend}, healthy_models=lambda: (mid,)
    )
    return ExecutionContext(registry, session, binding, registry.catalog.get(mid))


def test_public_decision_batch_preserves_question_semantics_and_one_usage(routing_registry):
    ctx = context(routing_registry([("qwen-decision", [0])]), "qwen-decision")
    adapter = load_adapter(ctx.binding.adapter)
    request = PublicDecisionRequest(
        model=ctx.binding.model,
        input="contract input",
        questions=[
            {"type": "predicate", "name": "p", "instructions": "contract predicate"},
            {
                "type": "choice",
                "name": "c",
                "instructions": "contract choice",
                "choices": [{"value": "a"}, {"value": "b"}],
            },
            {
                "type": "score",
                "name": "s",
                "instructions": "contract score",
                "levels": [{"label": "low"}, {"label": "high"}],
            },
        ],
    )
    assert adapter.required_capabilities(request, request.task) == {
        "decision.predicate_evaluation",
        "decision.choice_routing",
        "decision.score_ranking",
    }
    adapter.validate_request(ctx.model, request, request.task)
    prepared = adapter.prepare(ctx, request, request.task)
    assert prepared.payload["questions"][2]["levels"] == [{"label": "low"}, {"label": "high"}]
    raw = {
        "model": ctx.binding.model,
        "answers": [
            {"type": "predicate", "name": "p", "probability": 0.7},
            {
                "type": "choice",
                "name": "c",
                "choice": "b",
                "confidence": 0.7,
                "probabilities": [
                    {"value": "a", "probability": 0.3},
                    {"value": "b", "probability": 0.7},
                ],
            },
            {
                "type": "score",
                "name": "s",
                "score": 0.6,
                "confidence": 0.6,
                "probabilities": [
                    {"value": 0, "label": "low", "probability": 0.4},
                    {"value": 1, "label": "high", "probability": 0.6},
                ],
            },
        ],
        "usage": {
            "input_tokens": 9,
            "output_tokens": 0,
            "total_tokens": 9,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    assert adapter.normalize_output(ctx, prepared, raw).model_dump() == raw
    raw["answers"][2]["score"] = 0.5
    with pytest.raises(ModelUnavailableError):
        adapter.normalize_output(ctx, prepared, raw)


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "x", "choices": [{"value": True}, {"value": False}]},
        {"type": "score", "instructions": "x", "levels": [{"label": "a"}]},
        {"type": "predicate", "instructions": "x", "choices": [{"value": "a"}, {"value": "b"}]},
        {"type": "predicate", "instructions": " "},
    ],
)
def test_decision_rejects_unsupported_native_subset(routing_registry, question):
    ctx = context(routing_registry([("qwen-decision", [0])]), "qwen-decision")
    adapter = load_adapter(ctx.binding.adapter)
    call = PublicDecisionRequest(
        model=ctx.binding.model, input="contract input", questions=[question]
    )
    with pytest.raises(InvalidRequestError):
        adapter.validate_request(ctx.model, call, call.task)


@pytest.mark.parametrize("limit", [0, -1, 999999])
def test_native_adapter_checks_max_completion_tokens(routing_registry, limit):
    ctx = context(routing_registry([("qwen-text", [0, 1])]), "qwen-text")
    adapter = load_adapter(ctx.binding.adapter)
    request = ChatCompletionRequest(
        model=ctx.binding.model,
        messages=[ChatMessage(role="user", content="contract")],
        max_completion_tokens=limit,
    )
    with pytest.raises(InvalidRequestError):
        adapter.validate_request(ctx.model, request, "responses.chat_completion")


def test_native_adapter_maps_max_completion_tokens_exactly(routing_registry):
    ctx = context(routing_registry([("qwen-text", [0, 1])]), "qwen-text")
    adapter = load_adapter(ctx.binding.adapter)
    request = ChatCompletionRequest(
        model=ctx.binding.model,
        messages=[ChatMessage(role="user", content="contract")],
        max_completion_tokens=16,
    )
    adapter.validate_request(ctx.model, request, "responses.chat_completion")
    prepared = adapter.prepare(ctx, request, "responses.chat_completion")
    assert prepared.payload["max_tokens"] == 16 and "max_completion_tokens" not in prepared.payload


def test_text_response_media_never_reaches_native_text_backend(routing_registry):
    ctx = context(routing_registry([("qwen-text", [0, 1])]), "qwen-text")
    adapter = load_adapter(ctx.binding.adapter)
    request = ResponseApiRequest(
        model=ctx.binding.model,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": "data:image/png;base64,aQ=="}],
            }
        ],
    )
    with pytest.raises(UnsupportedCapabilityError):
        adapter.validate_request(ctx.model, request, "responses.create")


@pytest.mark.parametrize(
    "corruption",
    ["none", "truncated", "wrong_model", "wrong_id", "sequence", "missing_usage", "empty_text"],
)
def test_native_response_sse_is_verified_not_fabricated(routing_registry, monkeypatch, corruption):
    ctx = context(routing_registry([("qwen-text", [0, 1])]), "qwen-text")
    adapter = load_adapter(ctx.binding.adapter)
    call = ResponseApiRequest(model=ctx.binding.model, input="contract", stream=True)
    prepared = adapter.prepare(ctx, call, "responses.create")
    snapshot = {
        "id": "resp_native",
        "created_at": 123,
        "object": "response",
        "model": ctx.binding.model,
        "status": "completed",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "native contract"}]}
        ],
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    }
    events = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**snapshot, "status": "in_progress"},
        },
        {"type": "response.completed", "sequence_number": 1, "response": snapshot},
    ]
    if corruption == "truncated":
        events = events[:1]
    if corruption == "wrong_model":
        snapshot["model"] = "other"
    if corruption == "wrong_id":
        events[1]["response"] = {**snapshot, "id": "changed"}
    if corruption == "sequence":
        events[1]["sequence_number"] = 0
    if corruption == "missing_usage":
        snapshot.pop("usage")
    if corruption == "empty_text":
        snapshot["output"][0]["content"][0]["text"] = ""
    body = "".join(
        "event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n" for event in events
    )
    actual = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    monkeypatch.setattr(
        "factory.adapters.response_protocol.httpx.AsyncClient",
        lambda **kwargs: actual(transport=transport, **kwargs),
    )
    seen = []

    async def run():
        async for event in adapter.stream(ctx, prepared):
            seen.append(event)

    if corruption == "none":
        asyncio.run(run())
        assert "response.completed" in seen[-1]
    else:
        with pytest.raises(ModelUnavailableError):
            asyncio.run(run())
        assert not any("event: response.completed" in event for event in seen)
