from factory.adapters.response_protocol import ResponseAdapter
from factory_contracts.errors import InvalidRequestError


class Adapter(ResponseAdapter):
    supported_backends = frozenset({"vllm"})
    capabilities = ResponseAdapter.capabilities | {
        "responses.image_chat",
        "responses.video_chat",
        "responses.visual_grounding",
        "responses.document_qa",
    }

    def resolve_capability(self, request, operation):
        if operation != "responses.chat_completion":
            return operation
        parts = [
            part
            for message in request.messages
            if isinstance(message.content, list)
            for part in message.content
        ]
        return (
            "responses.video_chat"
            if any(part.get("type") == "video_url" for part in parts)
            else "responses.image_chat"
        )

    def validate_request(self, model, request, capability):
        super().validate_request(model, request, capability)
        for message in getattr(request, "messages", []):
            for part in message.content if isinstance(message.content, list) else []:
                kind = part.get("type")
                if kind in {"image_url", "video_url"}:
                    value = part.get(kind)
                    url = value.get("url") if isinstance(value, dict) else value
                    if not isinstance(url, str) or not url.startswith("data:"):
                        raise InvalidRequestError(
                            "Media must be ingested as inline data by the request normalizer before adapter dispatch."
                        )
                elif kind != "text":
                    raise InvalidRequestError("Unsupported multimodal content part.")

        for item in (
            getattr(request, "input", [])
            if isinstance(getattr(request, "input", None), list)
            else []
        ):
            for part in item.get("content", []) if isinstance(item.get("content"), list) else []:
                if part.get("type") == "input_image":
                    if not isinstance(part.get("image_url"), str) or not part[
                        "image_url"
                    ].startswith("data:"):
                        raise InvalidRequestError(
                            "Response media must be normalized inline before adapter dispatch."
                        )
                elif part.get("type") != "input_text":
                    raise InvalidRequestError("Unsupported response input part.")

    def required_capabilities(self, request, capability):
        required = super().required_capabilities(request, capability)
        if capability == "responses.create" and isinstance(request.input, list):
            if any(
                part.get("type") == "input_image"
                for item in request.input
                for part in (item.get("content") if isinstance(item.get("content"), list) else [])
                if isinstance(part, dict)
            ):
                required.add("responses.image_chat")
        return required
