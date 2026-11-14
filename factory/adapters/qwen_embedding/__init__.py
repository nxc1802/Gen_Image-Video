"""Qwen embedding protocol and output validation for the selected vLLM backend."""

import math

from factory_contracts.errors import InvalidRequestError, ModelUnavailableError
from factory.core.model_adapter import ModelAdapter, PreparedCall
from factory.engines.vector.contracts import EmbeddingResponse


class Adapter(ModelAdapter):
    capabilities = frozenset({"vector.text_embedding"})
    supported_backends = frozenset({"vllm"})

    def resolve_capability(self, request, operation):
        if (
            operation == "vector.text_embedding"
            and isinstance(request.input, list)
            and any(not isinstance(item, str) for item in request.input)
        ):
            return "vector.multimodal_embedding"
        return operation

    def validate_request(self, model, request, capability):
        texts = [request.input] if isinstance(request.input, str) else request.input
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise InvalidRequestError("Embedding inputs must be nonempty text strings.")
        dimension = (
            request.dimensions
            if request.dimensions is not None
            else model.parameters["dimensions"]["default"]
        )
        maximum = model.parameters["dimensions"].get(
            "maximum",
            model.parameters["dimensions"].get("max", model.parameters["dimensions"]["default"]),
        )
        minimum = model.parameters["dimensions"].get("minimum", 1)
        if type(dimension) is not int or not minimum <= dimension <= maximum:
            raise InvalidRequestError(
                "Requested dimensions exceed the model's declared embedding size."
            )

    def prepare(self, context, request, capability):
        texts = [request.input] if isinstance(request.input, str) else request.input
        dimensions = (
            request.dimensions
            if request.dimensions is not None
            else context.model.parameters["dimensions"]["default"]
        )
        return PreparedCall(
            capability,
            {"model": context.binding.model, "input": texts, "dimensions": dimensions},
            {"count": len(texts), "dimensions": dimensions},
        )

    def infer(self, context, prepared):
        return context.post_json("/v1/embeddings", prepared.payload)

    def normalize_output(self, context, prepared, output):
        try:
            if output.get("model") != context.binding.model or output.get("object") != "list":
                raise ValueError("Backend returned another model identity or object")
            rows = output["data"]
            count, dimensions = prepared.metadata["count"], prepared.metadata["dimensions"]
            if (
                len(rows) != count
                or any(type(row["index"]) is not int for row in rows)
                or {row["index"] for row in rows} != set(range(count))
            ):
                raise ValueError("Backend returned missing/duplicate embedding indices")
            for row in rows:
                vector = row["embedding"]
                if (
                    row.get("object") != "embedding"
                    or len(vector) != dimensions
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)
                    or not any(vector)
                ):
                    raise ValueError("Backend returned an invalid embedding vector")
            usage = output["usage"]
            if (
                any(
                    type(usage[key]) is not int or usage[key] < 0
                    for key in ["prompt_tokens", "total_tokens"]
                )
                or usage["total_tokens"] < usage["prompt_tokens"]
            ):
                raise ValueError("Backend returned invalid token usage")
            return EmbeddingResponse.model_validate(
                {**output, "data": sorted(rows, key=lambda row: row["index"])}
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelUnavailableError(
                "Embedding backend returned an invalid result; no vector or usage was fabricated."
            ) from exc
