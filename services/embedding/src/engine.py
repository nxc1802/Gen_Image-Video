"""
services/embedding/src/engine.py
Unified Embedding and Rerank Engine manager.
"""
from typing import List, Union
from factory_contracts.errors import InvalidRequestError
from services.embedding.src.multimodal_embed import MultimodalEmbedder
from services.embedding.src.reranker import DocumentReranker
from services.embedding.src.schemas import EmbeddingRequest, EmbeddingResponse, RerankRequest, RerankResponse
from services.embedding.src.text_embed import TextEmbedder


class EmbeddingEngine:
    """Coordinates Vector Embeddings and Cross-Encoder Document Reranking."""

    def __init__(self):
        self.text_embedder = TextEmbedder(default_dimensions=1024)
        self.multimodal_embedder = MultimodalEmbedder(self.text_embedder)
        self.reranker = DocumentReranker()

    def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Processes single or batch input into normalized embeddings."""
        if request.input is None:
            raise InvalidRequestError("Field 'input' is required.")

        # Check for empty string or empty list
        if isinstance(request.input, str):
            if not request.input.strip():
                raise InvalidRequestError("Field 'input' is required.")
            raw_inputs: List[str] = [request.input]
        elif isinstance(request.input, list):
            if len(request.input) == 0:
                raise InvalidRequestError("Field 'input' is required.")
            raw_inputs = request.input
        else:
            raw_inputs = [str(request.input)]

        # Separate text from multimodal
        if all(isinstance(x, str) for x in raw_inputs):
            return self.text_embedder.embed_texts(
                model=request.model,
                texts=raw_inputs,
                dimensions=request.dimensions,
            )

        # Handle mixed/multimodal elements
        dim = request.dimensions or 1024
        self.text_embedder.check_model_availability(request.model)
        items = [
            self.multimodal_embedder.embed_multimodal_item(item, idx, dim)
            for idx, item in enumerate(raw_inputs)
        ]
        from services.embedding.src.schemas import EmbeddingUsage

        return EmbeddingResponse(
            object="list",
            data=items,
            model=request.model,
            usage=EmbeddingUsage(
                prompt_tokens=len(raw_inputs) * 5,
                total_tokens=len(raw_inputs) * 5,
            ),
        )

    def rerank_documents(self, request: RerankRequest) -> RerankResponse:
        """Processes cross-encoder rerank request."""
        return self.reranker.rerank(request)
