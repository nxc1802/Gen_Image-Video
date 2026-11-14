"""
services/text/src/models.py
Request and Response schemas for OpenAI-compatible Chat Completions and Responses API.
"""
import time
import uuid
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")
    type: str = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    reasoning_effort: Optional[str] = Field(default=None, description="Reasoning effort: 'none', 'low', 'medium', 'high'")
    reasoning: Optional[Dict[str, Any]] = Field(default=None, description="Reasoning configuration object, e.g. {'effort': 'medium'}")

    @model_validator(mode="after")
    def resolve_reasoning_effort(self) -> "ChatCompletionRequest":
        if self.reasoning is not None:
            if not isinstance(self.reasoning, dict):
                raise ValueError("Field 'reasoning' must be an object.")
            if "effort" in self.reasoning:
                effort = self.reasoning.get("effort")
                if not isinstance(effort, str) or not effort.strip():
                    raise ValueError("Field 'reasoning.effort' must be a non-empty string: 'none', 'low', 'medium', 'high'.")
                if not self.reasoning_effort:
                    self.reasoning_effort = effort.lower()
        if self.reasoning_effort is not None:
            if not isinstance(self.reasoning_effort, str) or not self.reasoning_effort.strip():
                raise ValueError("Field 'reasoning_effort' must be a non-empty string: 'none', 'low', 'medium', 'high'.")
            norm = self.reasoning_effort.lower().strip()
            if norm not in {"none", "low", "medium", "high"}:
                raise ValueError("Field 'reasoning_effort' must be one of: 'none', 'low', 'medium', 'high'.")
            self.reasoning_effort = norm
        return self


class ChatCompletionChoiceMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    reasoning_content: Optional[str] = Field(default=None, description="Decoupled <think> reasoning tokens")
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionChoiceMessage
    finish_reason: str = "stop"


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int = 0
    accepted_prediction_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[CompletionTokensDetails] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = Field(default=None, description="Streaming reasoning token delta")
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]


class ResponseApiRequest(BaseModel):
    model: str
    input: str
    temperature: Optional[float] = 0.2
    max_output_tokens: Optional[int] = 2048


class ResponseApiResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"resp-{uuid.uuid4()}")
    object: str = "response"
    model: str
    output: str
    reasoning_tokens: int = 42
    status: str = "completed"
