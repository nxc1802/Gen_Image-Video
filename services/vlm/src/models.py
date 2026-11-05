"""
services/vlm/src/models.py
Request and Response schemas for Multimodal Vision-Language completions.
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from services.text.src.models import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionTokensDetails,
    ResponseApiRequest,
    ResponseApiResponse,
    UsageInfo,
)

__all__ = [
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionChoiceMessage",
    "ChatCompletionChoice",
    "ChatCompletionChunkDelta",
    "ChatCompletionChunkChoice",
    "ChatCompletionChunk",
    "CompletionTokensDetails",
    "UsageInfo",
    "ChatCompletionResponse",
    "ResponseApiRequest",
    "ResponseApiResponse",
]
