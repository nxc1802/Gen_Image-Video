"""Shared Text/Vision engine requests. Public API expansion belongs to Phase 6."""

from typing import Any
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream_options: dict[str, Any] | None = None
    stream: bool = False
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    reasoning_effort: str | None = None
    reasoning: dict[str, Any] | None = None

    @model_validator(mode="after")
    def reasoning_fields(self):
        nested = (self.reasoning or {}).get("effort")
        if nested is not None:
            if not isinstance(nested, str) or nested.lower() not in {
                "none",
                "low",
                "medium",
                "high",
            }:
                raise ValueError("Invalid reasoning.effort")
            if self.reasoning_effort and self.reasoning_effort != nested.lower():
                raise ValueError("Conflicting reasoning effort fields")
            self.reasoning_effort = nested.lower()
        return self

    @field_validator("reasoning_effort")
    @classmethod
    def effort(cls, value):
        if value is not None and value.lower() not in {"none", "low", "medium", "high"}:
            raise ValueError("Invalid reasoning_effort")
        return value.lower() if value else value


class ResponseApiRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    top_p: float | None = None
    tool_choice: str | dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    store: bool = False
    temperature: float | None = None
    max_output_tokens: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    text: dict[str, Any] | None = None

    @field_validator("input")
    @classmethod
    def input_items(cls, value):
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    raise ValueError("Invalid response input item")
                kind = item.get("type", "message")
                if kind == "message":
                    if item.get("role") not in {
                        "system",
                        "developer",
                        "user",
                        "assistant",
                    } or not isinstance(item.get("content"), (str, list)):
                        raise ValueError("Invalid response message")
                    if isinstance(item["content"], list) and any(
                        not isinstance(part, dict) for part in item["content"]
                    ):
                        raise ValueError("Invalid response content part")
                elif kind == "function_call":
                    if not all(
                        isinstance(item.get(k), str) for k in ("call_id", "name", "arguments")
                    ):
                        raise ValueError("Invalid function call")
                elif kind == "function_call_output":
                    if not isinstance(item.get("call_id"), str) or not isinstance(
                        item.get("output"), str
                    ):
                        raise ValueError("Invalid function output")
                else:
                    raise ValueError("Unsupported response input type")
        return value
