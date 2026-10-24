"""
services/vlm/src/models.py
Request and Response schemas for Multimodal Vision-Language completions.
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from services.text.src.models import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ResponseApiRequest,
    ResponseApiResponse,
    UsageInfo,
)

__all__ = [
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionChoiceMessage",
    "ChatCompletionChoice",
    "UsageInfo",
    "ChatCompletionResponse",
    "ResponseApiRequest",
    "ResponseApiResponse",
]
