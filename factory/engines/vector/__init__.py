"""Vector orchestration. Checkpoints, native protocols and outputs belong to adapters."""
from factory_contracts.errors import InvalidRequestError
from factory.core.model_adapter import Engine


class VectorEngine(Engine):
    engine = "vector"

    def generate_embeddings(self, request):
        if not request.input or isinstance(request.input, str) and not request.input.strip():
            raise InvalidRequestError("Field 'input' is required.")
        return self.execute(request, "vector.text_embedding")

    def rerank_documents(self, request):
        if not request.query.strip() or not request.documents:
            raise InvalidRequestError("A query and candidate documents are required.")
        return self.execute(request, "vector.reranking")
