"""
Strict adapter for a real cross-encoder/vLLM reranker.

BM25/lexical fallbacks are intentionally forbidden because returning them under
an AI reranker model identifier silently corrupts downstream ranking quality.
"""
import os
from typing import Any, Dict, List, Optional

import httpx

from factory_contracts.errors import InvalidRequestError, ModelUnavailableError
from services.embedding.src.schemas import RerankRequest, RerankResponse, RerankResultItem

OFFLINE_RERANK_MODELS = {
    "failing-rerank-model", "offline-model", "failing-model", "offline-rerank-model"
}


class VLLMRerankAdapter:
    def __init__(self, upstream_url: Optional[str] = None, timeout: float = 60.0):
        self.upstream_url = (
            upstream_url
            or os.environ.get("RERANK_UPSTREAM_URL")
            or os.environ.get("VLLM_RERANK_URL")
            or os.environ.get("TEI_RERANK_URL")
            or ""
        ).rstrip("/")
        self.api_key = os.environ.get("RERANK_API_KEY") or os.environ.get("FACTORY_API_KEY", "")
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.upstream_url)

    def rerank(self, request: RerankRequest) -> RerankResponse:
        if not self.is_configured:
            raise ModelUnavailableError(
                message="Real rerank backend is not configured. Lexical fallback is disabled.",
                details={"required_env": "RERANK_UPSTREAM_URL or VLLM_RERANK_URL or TEI_RERANK_URL"},
            )
        payload: Dict[str, Any] = {
            "model": request.model,
            "query": request.query,
            "documents": request.documents,
        }
        if request.top_n is not None:
            payload["top_n"] = request.top_n
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.upstream_url}/rerank", headers=headers, json=payload)
                if response.status_code == 404:
                    response = client.post(f"{self.upstream_url}/v1/rerank", headers=headers, json=payload)
                if response.status_code >= 400:
                    raise ModelUnavailableError(
                        message="Real rerank backend rejected the request.",
                        details={"status_code": response.status_code, "model": request.model},
                    )
                data = response.json()
        except ModelUnavailableError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                message="Real rerank backend is unavailable.",
                details={"error_type": type(exc).__name__, "model": request.model},
            ) from exc

        raw_results = data.get("results") or data.get("data") or []
        results: List[RerankResultItem] = []
        for item in raw_results:
            idx = int(item["index"])
            results.append(
                RerankResultItem(
                    index=idx,
                    document=item.get("document", request.documents[idx]),
                    relevance_score=float(item.get("relevance_score", item.get("score"))),
                )
            )
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        if request.top_n:
            results = results[: request.top_n]
        return RerankResponse(model=data.get("model", request.model), results=results)


class DocumentReranker:
    def __init__(self, upstream_url: Optional[str] = None):
        self.adapter = VLLMRerankAdapter(upstream_url=upstream_url)

    def rerank(self, request: RerankRequest) -> RerankResponse:
        if not request.model:
            raise InvalidRequestError("Field 'model' is required.")
        if request.model in OFFLINE_RERANK_MODELS:
            raise ModelUnavailableError(message=f"Rerank model '{request.model}' is offline.")
        if not request.query or not request.query.strip():
            raise InvalidRequestError("Field 'query' is required.")
        if not request.documents:
            raise InvalidRequestError("Field 'documents' must be a non-empty list.")
        return self.adapter.rerank(request)
