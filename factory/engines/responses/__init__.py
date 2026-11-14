"""One orchestration contract for text and multimodal responses."""

import asyncio
from factory_contracts.errors import InvalidRequestError
from factory.core.model_adapter import Engine


class ResponsesEngine(Engine):
    engine = "responses"

    def validate_request(self, request):
        if not request.model or not request.messages:
            raise InvalidRequestError("A model and nonempty messages are required.")
        self.resolve(request, "responses.chat_completion")

    def generate(self, request):
        self.validate_request(request)
        return self.execute(request, "responses.chat_completion")

    def execute_reasoning(self, request):
        if not request.input:
            raise InvalidRequestError("Response input must be nonempty.")
        return self.execute(request, "responses.create")

    async def stream_generate(self, request):
        self.validate_request(request)
        async for event in self._stream(request, "responses.chat_completion"):
            yield event

    async def stream_response(self, request):
        async for event in self._stream(request, "responses.create"):
            yield event

    async def _stream(self, request, operation):
        context, adapter, prepared = await asyncio.to_thread(
            self.prepare_execution, request, operation
        )
        from factory.core.execution import async_execution_scope

        async with async_execution_scope(context):
            async for event in adapter.stream(context, prepared):
                yield event
