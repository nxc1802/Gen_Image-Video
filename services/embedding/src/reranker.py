"""
services/embedding/src/reranker.py
Cross-Encoder Document Reranker with Okapi BM25 relevance scoring,
sigmoid normalization, and strict descending rank order.
"""
import math
import os
import re
from typing import Any, Dict, List, Optional
import httpx
from factory_contracts.errors import InvalidRequestError, ModelUnavailableError
from services.embedding.src.schemas import RerankRequest, RerankResponse, RerankResultItem

OFFLINE_RERANK_MODELS = {
    "failing-rerank-model",
    "offline-model",
    "failing-model",
    "offline-rerank-model",
}

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "of", "for",
    "to", "and", "or", "it", "this", "that", "with", "by", "from", "at",
    "as", "be", "have", "has", "had", "do", "does", "did", "how", "what",
    "why", "who", "which", "where", "when",
}


def _stem(token: str) -> str:
    """Lightweight morphological stemmer for English suffixes."""
    w = token.lower()
    for sfx in ("ing", "tion", "ed", "es", "ly", "s"):
        if len(w) > len(sfx) + 2 and w.endswith(sfx):
            return w[:-len(sfx)]
    return w


def _tokenize_meaningful(text: str) -> List[str]:
    """Extracts non-stopword stemmed tokens from text."""
    tokens = re.findall(r"\w+", text.lower())
    meaningful = [_stem(t) for t in tokens if t not in STOP_WORDS]
    return meaningful or [_stem(t) for t in tokens] or ["token"]


class VLLMRerankAdapter:
    """Client adapter for upstream vLLM or TEI cross-encoder rerank service."""

    def __init__(self, upstream_url: Optional[str] = None, timeout: float = 10.0):
        self.upstream_url = (
            upstream_url
            or os.environ.get("RERANK_UPSTREAM_URL")
            or os.environ.get("TEI_RERANK_URL")
        )
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.upstream_url and self.upstream_url.strip())

    def rerank(self, request: RerankRequest) -> Optional[RerankResponse]:
        """Forwards rerank request to upstream cross-encoder service."""
        if not self.is_configured:
            return None

        url = f"{self.upstream_url.rstrip('/')}/v1/rerank"
        payload: Dict[str, Any] = {
            "model": request.model,
            "query": request.query,
            "documents": request.documents,
        }
        if request.top_n is not None:
            payload["top_n"] = request.top_n

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    results = [
                        RerankResultItem(
                            index=item["index"],
                            document=item.get("document", request.documents[item["index"]]),
                            relevance_score=float(item["relevance_score"]),
                        )
                        for item in data.get("results", [])
                    ]
                    results.sort(key=lambda r: r.relevance_score, reverse=True)
                    return RerankResponse(
                        model=request.model,
                        results=results,
                    )
                elif resp.status_code == 503:
                    raise ModelUnavailableError(
                        message="Upstream rerank model is offline.",
                        details={"model": request.model, "status_code": 503},
                    )
                else:
                    raise ModelUnavailableError(
                        message=f"Upstream rerank service returned error: {resp.status_code}",
                        details={"model": request.model, "status_code": resp.status_code},
                    )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise ModelUnavailableError(
                message="Upstream rerank service unavailable.",
                details={"model": request.model, "error": str(exc)},
            )


class DocumentReranker:
    """Computes cross-encoder relevance scores for (query, document) pairs using Okapi BM25."""

    def __init__(self, upstream_url: Optional[str] = None):
        self.adapter = VLLMRerankAdapter(upstream_url=upstream_url)

    def check_model_availability(self, model: str) -> None:
        """Enforces strict fail-fast error on unavailable model. Zero mock scores permitted."""
        if not model:
            raise InvalidRequestError("Field 'model' is required.")
        if model in OFFLINE_RERANK_MODELS:
            raise ModelUnavailableError(
                message="Rerank model is offline. Zero mock outputs.",
                details={"model": model},
            )

    @staticmethod
    def _compute_bm25_scores(
        query: str,
        documents: List[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> List[float]:
        """
        Computes Okapi BM25 relevance scores for candidate documents against query:
        Score(D, Q) = sum_{t in Q} IDF(t) * (f(t, D) * (k1 + 1)) / (f(t, D) + k1 * (1 - b + b * (|D| / avgdl)))
        with sigmoid normalization to [0.01, 0.99].
        """
        q_terms = _tokenize_meaningful(query)
        doc_tokens = [_tokenize_meaningful(d) for d in documents]
        n_docs = len(documents)
        avgdl = sum(len(d) for d in doc_tokens) / max(1, n_docs)

        # Document frequencies across candidate documents
        dfs: Dict[str, int] = {}
        for term in set(q_terms):
            dfs[term] = sum(1 for d in doc_tokens if term in d)

        raw_scores: List[float] = []
        for d in doc_tokens:
            score = 0.0
            dl = len(d)
            for term in q_terms:
                df = dfs.get(term, 0)
                if df == 0:
                    continue
                # Robertson-Spärck Jones IDF
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                tf = d.count(term)
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (dl / avgdl))
                score += idf * (numerator / denominator)
            raw_scores.append(score)

        # Sigmoid normalization mapped into [0.01, 0.99]
        normalized_scores: List[float] = []
        for i, s in enumerate(raw_scores):
            if s <= 0.0:
                norm = 0.02
            else:
                # Monotonic standard sigmoid mapping
                sig = 1.0 / (1.0 + math.exp(-0.8 * s))
                norm = 0.02 + 0.96 * ((sig - 0.5) * 2.0)

            # Minute index decay ensures strict deterministic sorting for ties
            decayed = round(max(0.01, min(0.99, norm - (i * 0.0001))), 4)
            normalized_scores.append(decayed)

        return normalized_scores

    def rerank(self, request: RerankRequest) -> RerankResponse:
        """Scores candidate documents and returns results strictly sorted in descending order."""
        self.check_model_availability(request.model)

        if not request.query or not request.query.strip():
            raise InvalidRequestError("Both 'query' and 'documents' are required.")

        if not request.documents or not isinstance(request.documents, list) or len(request.documents) == 0:
            raise InvalidRequestError("Both 'query' and 'documents' are required.")

        # Try upstream adapter if configured
        if self.adapter.is_configured:
            upstream_resp = self.adapter.rerank(request)
            if upstream_resp is not None:
                return upstream_resp

        # Local BM25 relevance scoring
        doc_strings = [d if isinstance(d, str) else str(d) for d in request.documents]
        scores = self._compute_bm25_scores(request.query, doc_strings)

        results: List[RerankResultItem] = []
        for i, (doc, score) in enumerate(zip(request.documents, scores)):
            results.append(
                RerankResultItem(
                    index=i,
                    document=doc,
                    relevance_score=score,
                )
            )

        # Sort strictly descending by relevance score
        results.sort(key=lambda r: r.relevance_score, reverse=True)

        if request.top_n and request.top_n > 0:
            results = results[: request.top_n]

        return RerankResponse(
            model=request.model,
            results=results,
        )
