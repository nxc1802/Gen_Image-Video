"""Native response transport shared by selected model adapters, not by Engines."""

import json
import math
import httpx
from factory_contracts.errors import (
    InvalidRequestError,
    ModelUnavailableError,
    UnsupportedCapabilityError,
)
from factory.core.model_adapter import ModelAdapter, PreparedCall


class ResponseAdapter(ModelAdapter):
    capabilities = frozenset(
        {
            "responses.chat_completion",
            "responses.create",
            "responses.structured_output",
            "responses.tool_calling",
            "responses.streaming",
        }
    )

    def required_capabilities(self, request, capability):
        required = {capability}
        if getattr(request, "stream", False):
            required.add("responses.streaming")
        if getattr(request, "tools", None) or getattr(request, "tool_choice", None):
            required.add("responses.tool_calling")
        if getattr(request, "response_format", None) or getattr(request, "text", None):
            required.add("responses.structured_output")
        return required

    def validate_request(self, model, request, capability):
        if getattr(request, "messages", None) is not None:
            if not request.messages or any(
                message.role not in {"system", "developer", "user", "assistant", "tool"}
                for message in request.messages
            ):
                raise InvalidRequestError("Invalid response messages.")
            if not any(message.content or message.tool_calls for message in request.messages):
                raise InvalidRequestError("Messages contain no input.")
        elif not request.input or isinstance(request.input, str) and not request.input.strip():
            raise InvalidRequestError("Response input must be nonempty.")
        for name, high in [("temperature", 2), ("top_p", 1)]:
            value = getattr(request, name, None)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= high):
                raise InvalidRequestError(f"Invalid {name}.")
        if (
            getattr(request, "max_tokens", None) is not None
            and getattr(request, "max_completion_tokens", None) is not None
        ):
            raise InvalidRequestError("Use only one output token limit.")
        if getattr(request, "store", False):
            raise UnsupportedCapabilityError("Stored responses are not implemented.")
        if (getattr(request, "reasoning", None) or {}).get("effort") not in (None, "none"):
            raise UnsupportedCapabilityError(
                "This checkpoint has no declared reasoning-effort implementation."
            )
        value = (
            (
                request.max_completion_tokens
                if request.max_completion_tokens is not None
                else request.max_tokens
            )
            if hasattr(request, "max_tokens")
            else getattr(request, "max_output_tokens", None)
        )
        maximum = model.parameters.get("max_output_tokens", {}).get("default")
        if value is not None and (
            type(value) is not int or value <= 0 or maximum is not None and value > maximum
        ):
            raise InvalidRequestError("Output token limit exceeds the manifest profile.")
        if getattr(request, "reasoning_effort", None) not in (None, "none"):
            raise UnsupportedCapabilityError(
                "This checkpoint has no declared reasoning-effort implementation."
            )

    def prepare(self, context, request, capability):
        payload = request.model_dump(mode="json", exclude_none=True)
        payload["model"] = context.binding.model
        payload.pop("reasoning", None)
        payload.pop("reasoning_effort", None)
        limit = payload.pop("max_completion_tokens", None)
        if limit is not None:
            payload["max_tokens"] = limit
        is_response = capability == "responses.create"
        field = "max_output_tokens" if is_response else "max_tokens"
        if field not in payload and "max_output_tokens" in context.model.parameters:
            payload[field] = context.model.parameters["max_output_tokens"]["default"]
        return PreparedCall(
            capability,
            payload,
            {
                "path": "/v1/responses" if is_response else "/v1/chat/completions",
                "tools": {
                    tool.get("name", tool.get("function", {}).get("name"))
                    for tool in (getattr(request, "tools", None) or [])
                },
            },
        )

    def infer(self, context, prepared):
        if prepared.payload.get("stream"):
            raise InvalidRequestError("Use the streaming engine operation for a stream request.")
        return context.post_json(prepared.metadata["path"], prepared.payload)

    @staticmethod
    def check_usage(usage, response=False):
        names = (
            ("input_tokens", "output_tokens", "total_tokens")
            if response
            else ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        if (
            not isinstance(usage, dict)
            or any(type(usage.get(key)) is not int or usage[key] < 0 for key in names)
            or usage[names[0]] + usage[names[1]] != usage["total_tokens"]
        ):
            raise ValueError("Invalid native token usage")

    def normalize_output(self, context, prepared, output):
        try:
            response = prepared.metadata["path"] == "/v1/responses"
            if (
                output["model"] != context.binding.model
                or not isinstance(output["id"], str)
                or not output["id"]
            ):
                raise ValueError("Native identity missing or changed")
            self.check_usage(output["usage"], response)
            if response:
                if type(output.get("created_at")) is not int or output["created_at"] < 0:
                    raise ValueError("Native response timestamp missing")
                for item in output.get("output", []):
                    if not isinstance(item, dict):
                        raise ValueError("Invalid response output item")
                    if item.get("type") == "message":
                        parts = item.get("content")
                        if (
                            not isinstance(parts, list)
                            or not parts
                            or any(
                                not isinstance(p, dict)
                                or p.get("type") not in {"output_text", "refusal"}
                                or not isinstance(
                                    p.get("text" if p["type"] == "output_text" else "refusal"), str
                                )
                                or not p["text" if p["type"] == "output_text" else "refusal"]
                                for p in parts
                            )
                        ):
                            raise ValueError("Native message contains no text/refusal")
                    elif item.get("type") == "function_call":
                        if (
                            item.get("name") not in prepared.metadata["tools"]
                            or not item.get("call_id")
                            or not isinstance(json.loads(item["arguments"]), dict)
                        ):
                            raise ValueError("Invalid native function call")
                if (
                    output["object"] != "response"
                    or output.get("status") != "completed"
                    or not isinstance(output["output"], list)
                    or not output["output"]
                ):
                    raise ValueError("Incomplete native response")
                if not any(
                    item.get("type") in {"message", "function_call"} for item in output["output"]
                ):
                    raise ValueError("Native response contains no supported output")
            else:
                if (
                    output["object"] != "chat.completion"
                    or type(output["created"]) is not int
                    or not output["choices"]
                ):
                    raise ValueError("Invalid native completion")
                if len({row["index"] for row in output["choices"]}) != len(
                    output["choices"]
                ) or any(
                    type(row["index"]) is not int or row["index"] < 0 for row in output["choices"]
                ):
                    raise ValueError("Invalid completion indices")
                for row in output["choices"]:
                    message = row["message"]
                    if (
                        message.get("role") != "assistant"
                        or not (
                            message.get("content")
                            or message.get("tool_calls")
                            or message.get("refusal")
                        )
                        or row.get("finish_reason")
                        not in {"stop", "length", "tool_calls", "content_filter"}
                    ):
                        raise ValueError("Native completion contains no answer/refusal/tool call")
                    for tool in message.get("tool_calls") or []:
                        if (
                            not tool.get("id")
                            or tool.get("type") != "function"
                            or tool["function"]["name"] not in prepared.metadata["tools"]
                            or not isinstance(json.loads(tool["function"]["arguments"]), dict)
                        ):
                            raise ValueError("Invalid native tool call")
            return output  # No generated IDs, timestamps, usage, reasoning or answer text.
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ModelUnavailableError(
                "Native response failed validation; no replacement was generated."
            ) from exc

    async def stream(self, context, prepared):
        if not prepared.payload.get("stream"):
            raise InvalidRequestError("Streaming requires stream=true.")
        if prepared.metadata["path"] == "/v1/responses":
            async for event in self.stream_response(context, prepared):
                yield event
            return
        backend = context.backend
        if not backend.health():
            raise ModelUnavailableError("Selected backend is unhealthy.")
        identity, finished = None, False
        usage_seen = False
        try:
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{context.binding.port}",
                timeout=120,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                async with client.stream(
                    "POST",
                    prepared.metadata["path"],
                    json=prepared.payload,
                    headers={"Authorization": "Bearer " + backend.token},
                ) as response:
                    if response.status_code != 200:
                        raise ModelUnavailableError("Selected backend rejected the stream.")
                    async for line in response.aiter_lines():
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data: "):
                            raise ValueError("Invalid native SSE field")
                        raw = line[6:]
                        if raw == "[DONE]":
                            if not finished:
                                raise ValueError("Native stream ended without a finish reason")
                            if (
                                prepared.payload.get("stream_options", {}).get("include_usage")
                                and not usage_seen
                            ):
                                raise ValueError("Requested native stream usage missing")
                            yield "data: [DONE]\n\n"
                            return
                        if len(raw) > 2 * 1024 * 1024:
                            raise ValueError("SSE record too large")
                        body = json.loads(raw)
                        if (
                            body.get("model") != context.binding.model
                            or body.get("object") != "chat.completion.chunk"
                            or not body.get("id")
                        ):
                            raise ValueError("Invalid stream identity")
                        if identity is not None and body["id"] != identity:
                            raise ValueError("Stream identity changed")
                        identity = body["id"]
                        if type(body.get("created")) is not int:
                            raise ValueError("Missing native chunk timestamp")
                        if body.get("usage") is not None:
                            self.check_usage(body["usage"])
                            usage_seen = True
                        for choice in body.get("choices", []):
                            if not isinstance(choice.get("delta"), dict):
                                raise ValueError("Invalid stream delta")
                            finished |= choice.get("finish_reason") in {
                                "stop",
                                "length",
                                "tool_calls",
                                "content_filter",
                            }
                        yield "data: " + json.dumps(body) + "\n\n"
            raise ValueError("Native stream was truncated")
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ModelUnavailableError(
                "Native stream failed; no completion event was fabricated."
            ) from exc

    async def stream_response(self, context, prepared):
        backend = context.backend
        if not backend.health():
            raise ModelUnavailableError("Selected backend is unhealthy.")
        identity = None
        completed = False
        sequence = -1
        event_name = None
        lines = []
        try:
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{context.binding.port}",
                timeout=120,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                async with client.stream(
                    "POST",
                    prepared.metadata["path"],
                    json=prepared.payload,
                    headers={"Authorization": "Bearer " + backend.token},
                ) as response:
                    if response.status_code != 200:
                        raise ModelUnavailableError("Selected backend rejected response stream.")
                    async for line in response.aiter_lines():
                        if line.startswith(":"):
                            continue
                        if line.startswith("event:"):
                            event_name = line[6:].strip()
                            continue
                        if line.startswith("data:"):
                            lines.append(line[5:].lstrip())
                            if sum(map(len, lines)) > 2 * 1024 * 1024:
                                raise ValueError("SSE event too large")
                            continue
                        if line:
                            raise ValueError("Unknown SSE field")
                        if not lines:
                            continue
                        body = json.loads("\n".join(lines))
                        lines = []
                        kind = body.get("type")
                        if (
                            not isinstance(kind, str)
                            or not kind.startswith("response.")
                            or event_name is not None
                            and kind != event_name
                        ):
                            raise ValueError("Invalid native response event")
                        if (
                            type(body.get("sequence_number")) is not int
                            or body["sequence_number"] <= sequence
                        ):
                            raise ValueError("Invalid native sequence")
                        sequence = body["sequence_number"]
                        native = body.get("response")
                        if native is not None:
                            if native.get("model") != context.binding.model or not native.get("id"):
                                raise ValueError("Invalid native response identity")
                            if identity is not None and native["id"] != identity:
                                raise ValueError("Response identity changed")
                            identity = native["id"]
                        if identity is None:
                            raise ValueError("Response stream has no native identity")
                        if kind == "response.completed":
                            self.normalize_output(context, prepared, native)
                            completed = True
                        if kind in {"response.failed", "response.incomplete", "response.error"}:
                            raise ValueError("Native response did not complete")
                        yield "event: " + kind + "\ndata: " + json.dumps(body) + "\n\n"
                        event_name = None
                        if completed:
                            return
            raise ValueError("Native response stream truncated")
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ModelUnavailableError(
                "Native response stream failed; no success event was fabricated."
            ) from exc
