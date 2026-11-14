from factory.adapters.response_protocol import ResponseAdapter
from factory_contracts.errors import UnsupportedCapabilityError


class Adapter(ResponseAdapter):
    supported_backends = frozenset({"llama_cpp"})

    def validate_request(self, model, request, capability):
        super().validate_request(model, request, capability)
        for message in getattr(request, "messages", []):
            if isinstance(message.content, list) and any(
                part.get("type") != "text" for part in message.content
            ):
                raise UnsupportedCapabilityError("Selected text checkpoint does not accept media.")

        for item in (
            getattr(request, "input", [])
            if isinstance(getattr(request, "input", None), list)
            else []
        ):
            for part in item.get("content", []) if isinstance(item.get("content"), list) else []:
                if part.get("type") != "input_text":
                    raise UnsupportedCapabilityError(
                        "Selected text checkpoint does not accept media."
                    )
