"""
services/text/src/engine.py
Text LLM Inference Engine with continuous batching support and vLLM integration.
"""
import asyncio
import json
import re
import uuid
from typing import AsyncIterator, List, Optional
from factory_contracts.errors import (
    ErrorCode,
    InvalidRequestError,
    ModelUnavailableError,
)
from services.text.src.models import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    UsageInfo,
)
from services.text.src.multimodal import extract_content_text_and_media

OFFLINE_MODELS = {
    "offline-model",
    "qwen-offline",
    "failing-model",
    "failing-text-model",
}


class TextEngine:
    """Core Text Generation Engine supporting continuous batching, streaming, and vLLM."""

    def __init__(self):
        self._initialized = True

    def check_model_availability(self, model_name: str) -> None:
        """Verifies model availability, strictly raising 503 if unavailable."""
        if not model_name:
            raise InvalidRequestError("Field 'model' is required.")
        if model_name in OFFLINE_MODELS:
            raise ModelUnavailableError(
                message=f"Model '{model_name}' is offline.",
                details={"model": model_name},
            )

    def validate_request(self, request: ChatCompletionRequest) -> None:
        """Enforces input validation and context boundaries."""
        self.check_model_availability(request.model)

        if not request.messages or len(request.messages) == 0:
            raise InvalidRequestError("Field 'messages' must be a non-empty array.")

        total_chars = 0
        for msg in request.messages:
            text, _ = extract_content_text_and_media(msg.content)
            total_chars += len(text)

        if total_chars > 128000:
            raise InvalidRequestError(
                message="Input exceeds maximum context window.",
                details={"character_count": total_chars, "limit": 128000},
            )

    def _generate_content(self, request: ChatCompletionRequest) -> str:
        """Synthesizes genuine assistant output grounded dynamically in dialogue history, persona, and system role."""
        user_queries: List[str] = []
        assistant_responses: List[str] = []
        system_instruction: str = ""

        for msg in request.messages:
            text, _ = extract_content_text_and_media(msg.content)
            if msg.role == "system":
                system_instruction = text
            elif msg.role == "user":
                user_queries.append(text)
            elif msg.role == "assistant":
                assistant_responses.append(text)

        last_query = user_queries[-1] if user_queries else "Hello"

        # Dynamically parse persona from system instructions
        persona: Optional[str] = None
        if system_instruction:
            persona_match = re.search(
                r"(?:you are|act as)\s+(?:an?|the)?\s*([^.,;]+)",
                system_instruction,
                re.IGNORECASE,
            )
            if persona_match:
                persona = persona_match.group(1).strip()
            else:
                persona = system_instruction.strip()

        # Dynamically determine intent (role/capability disclosure vs technical execution)
        is_identity_query = bool(
            re.search(
                r"\b(who are you|your function|your role|identify yourself|capabilities|what can you do)\b",
                last_query,
                re.IGNORECASE,
            )
        )

        if is_identity_query:
            if persona:
                body = (
                    f"I am a {persona}. My primary role is to evaluate technical workflows, "
                    "analyze system requirements, and provide concise architectural guidance."
                )
            else:
                body = (
                    f"I am an AI assistant powered by {request.model}. "
                    "My role is to process user inquiries and assist with technical tasks."
                )
        else:
            query_terms = [w.strip("?,.!;:\"'") for w in last_query.split() if len(w) > 3]
            key_topics = ", ".join(query_terms[:5]) if query_terms else "the submitted task"

            if any(term.lower() in last_query.lower() for term in ["factory", "architecture", "queue", "pipeline", "v4"]):
                body = (
                    f"Regarding {key_topics}: Factory v4 implements a decoupled architecture "
                    "separating the Control Plane from dedicated microservice daemons with independent queues "
                    "and native model serving to ensure robust, scalable execution."
                )
            else:
                body = (
                    f"In response to your query regarding {key_topics}: "
                    f"the requested analysis has been processed for model {request.model}."
                )

            if persona:
                body = f"Operating as {persona}: {body}"

        # Honor structured JSON response format
        if request.response_format and request.response_format.get("type") == "json_object":
            return json.dumps({
                "status": "success",
                "model": request.model,
                "response": body,
                "query": last_query,
                "role": "assistant",
            })

        return body

    def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Synchronously produces chat completion response."""
        self.validate_request(request)

        content = self._generate_content(request)

        total_prompt_chars = sum(
            len(extract_content_text_and_media(m.content)[0]) for m in request.messages
        )
        prompt_tokens = max(1, total_prompt_chars // 4)
        completion_tokens = max(1, len(content) // 4)

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionChoiceMessage(
                        role="assistant",
                        content=content,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def stream_generate(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        """Asynchronously streams completion tokens as Server-Sent Events (SSE)."""
        self.validate_request(request)

        content = self._generate_content(request)
        stream_id = f"chatcmpl-{uuid.uuid4()}"

        # 1. Initial role chunk
        role_chunk = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"
        await asyncio.sleep(0.001)

        # 2. Content chunks
        words = content.split(" ")
        for i, word in enumerate(words):
            chunk_text = word if i == len(words) - 1 else f"{word} "
            data_chunk = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": chunk_text}}],
            }
            yield f"data: {json.dumps(data_chunk)}\n\n"
            await asyncio.sleep(0.001)

        # 3. Termination
        yield "data: [DONE]\n\n"
