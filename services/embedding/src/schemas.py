"""
services/embedding/src/schemas.py
Pydantic schemas for Vector Embeddings (/v1/embeddings, /v2/embed) and Reranking (/v1/rerank, /rerank).
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="Embedding model name or alias")
    input: Union[str, List[str], List[Dict[str, Any]]] = Field(
        ..., description="Input text or multimodal elements to embed"
    )
    dimensions: Optional[int] = Field(
        default=1024, description="Target Matryoshka embedding dimensions (default: 1024)"
    )


class EmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int = Field(..., description="0-indexed position corresponding to input array")
    embedding: List[float] = Field(..., description="Dense float vector, L2 normalized")


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingItem] = Field(..., description="List of generated embedding items")
    model: str
    usage: EmbeddingUsage


class RerankRequest(BaseModel):
    model: str = Field(..., description="Reranking model name or alias")
    query: str = Field(..., description="Search query string")
    documents: List[Union[str, Dict[str, Any]]] = Field(
        ..., description="List of candidate document strings or structured records"
    )
    top_n: Optional[int] = Field(
        default=None, description="Optional limit on returned reranked results"
    )


class RerankResultItem(BaseModel):
    index: int = Field(..., description="Original index of the document in the input list")
    document: Union[str, Dict[str, Any]] = Field(..., description="Document content preserved")
    relevance_score: float = Field(..., description="Calibrated cross-encoder score in [0.0, 1.0]")


class RerankResponse(BaseModel):
    model: str
    results: List[RerankResultItem] = Field(
        ..., description="Results sorted in strictly descending order of relevance_score"
    )
