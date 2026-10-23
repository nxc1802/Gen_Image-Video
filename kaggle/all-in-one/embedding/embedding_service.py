"""
🧬 Embedding & Rerank Service Manager (Dual-GPU Adaptive Orchestrator)
================================================================================
Điều phối các model Embedding và Reranking theo phân bổ phần cứng Dual Tesla T4:
- GPU 0 (cuda:0) - 3 Model Dòng Text: Qwen3-0.6B, Qwen3-4B, Qwen3-8B
- GPU 1 (cuda:1) - 2 Model Dòng Vision-Language: Qwen3-VL-2B, Qwen3-VL-8B
Tương thích hoàn toàn chuẩn OpenAI POST /v1/embeddings và Cohere/Jina POST /v1/rerank.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Union

from .embedding_engine import QwenEmbeddingEngine, QwenRerankEngine

logger = logging.getLogger("EmbeddingService")

# Danh mục định nghĩa cấu hình model chuẩn
EMBEDDING_CATALOG = {
    # GPU 0: 3 Model Dòng Text
    "qwen3-0.6b": {
        "id": "Qwen/Qwen3-Embedding-0.6B",
        "hf_repo": "mradermacher/Qwen3-Embedding-0.6B-GGUF",
        "model_file": "Qwen3-Embedding-0.6B.Q4_K_M.gguf",
        "device": "cuda:0",
        "dimensions": 1024,
    },
    "qwen3-4b": {
        "id": "Qwen/Qwen3-Embedding-4B",
        "hf_repo": "Qwen/Qwen3-Embedding-4B-GGUF",
        "model_file": "Qwen3-Embedding-4B-Q4_K_M.gguf",
        "device": "cuda:0",
        "dimensions": 2560,
    },
    "qwen3-8b": {
        "id": "Qwen/Qwen3-Embedding-8B",
        "hf_repo": "Qwen/Qwen3-Embedding-8B-GGUF",
        "model_file": "Qwen3-Embedding-8B-Q4_K_M.gguf",
        "device": "cuda:0",
        "dimensions": 4096,
    },
    # GPU 1: 2 Model Dòng Vision-Language (VL)
    "qwen3-vl-2b": {
        "id": "Qwen/Qwen3-VL-Embedding-2B",
        "hf_repo": "mradermacher/Qwen3-VL-Embedding-2B-GGUF",
        "model_file": "Qwen3-VL-Embedding-2B.Q4_K_M.gguf",
        "mmproj_file": "Qwen3-VL-Embedding-2B.mmproj-Q8_0.gguf",
        "device": "cuda:1",
        "dimensions": 1536,
    },
    "qwen3-vl-8b": {
        "id": "Qwen/Qwen3-VL-Embedding-8B",
        "hf_repo": "mradermacher/Qwen3-VL-Embedding-8B-GGUF",
        "model_file": "Qwen3-VL-Embedding-8B.Q4_K_M.gguf",
        "mmproj_file": "Qwen3-VL-Embedding-8B.mmproj-Q8_0.gguf",
        "device": "cuda:1",
        "dimensions": 3584,
    },
}

RERANK_CATALOG = {
    # GPU 0: 3 Model Dòng Text
    "qwen3-0.6b": {
        "id": "Qwen/Qwen3-Reranker-0.6B",
        "hf_repo": "mradermacher/Qwen3-Reranker-0.6B-GGUF",
        "model_file": "Qwen3-Reranker-0.6B.Q4_K_M.gguf",
        "device": "cuda:0",
    },
    "qwen3-4b": {
        "id": "Qwen/Qwen3-Reranker-4B",
        "hf_repo": "mradermacher/Qwen3-Reranker-4B-GGUF",
        "model_file": "Qwen3-Reranker-4B.Q4_K_M.gguf",
        "device": "cuda:0",
    },
    "qwen3-8b": {
        "id": "Qwen/Qwen3-Reranker-8B",
        "hf_repo": "mradermacher/Qwen3-Reranker-8B-GGUF",
        "model_file": "Qwen3-Reranker-8B.Q4_K_M.gguf",
        "device": "cuda:0",
    },
    # GPU 1: 2 Model Dòng Vision-Language (VL)
    "qwen3-vl-2b": {
        "id": "Qwen/Qwen3-VL-Reranker-2B",
        "hf_repo": "mradermacher/Qwen3-VL-Reranker-2B-GGUF",
        "model_file": "Qwen3-VL-Reranker-2B.Q4_K_M.gguf",
        "mmproj_file": "Qwen3-VL-Reranker-2B.mmproj-Q8_0.gguf",
        "device": "cuda:1",
    },
    "qwen3-vl-8b": {
        "id": "Qwen/Qwen3-VL-Reranker-8B",
        "hf_repo": "mradermacher/Qwen3-VL-Reranker-8B-GGUF",
        "model_file": "Qwen3-VL-Reranker-8B.Q4_K_M.gguf",
        "mmproj_file": "Qwen3-VL-Reranker-8B.mmproj-Q8_0.gguf",
        "device": "cuda:1",
    },
}


class EmbeddingServiceManager:
    """Singleton điều phối toàn bộ dịch vụ Embedding & Rerank."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EmbeddingServiceManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.embedding_engines: Dict[str, QwenEmbeddingEngine] = {}
        self.rerank_engines: Dict[str, QwenRerankEngine] = {}
        self.default_embed_key = "qwen3-0.6b"
        self.default_rerank_key = "qwen3-0.6b"
        logger.info("🧬 EmbeddingServiceManager sẵn sàng điều phối Dual-GPU.")

    def _normalize_key(self, model_name: Optional[str], catalog: Dict[str, Any], default_key: str) -> str:
        if not model_name:
            return default_key
        name_lower = model_name.lower().replace("_", "-")
        # Check direct exact key matching first
        for key in catalog:
            if key == name_lower or key.replace("-", "") == name_lower.replace("-", ""):
                return key
        # Check explicit multimodal flag
        if "vl" in name_lower:
            if "8b" in name_lower:
                return "qwen3-vl-8b"
            if "2b" in name_lower:
                return "qwen3-vl-2b"
        # Pure text line
        if "8b" in name_lower:
            return "qwen3-8b"
        if "4b" in name_lower:
            return "qwen3-4b"
        if "2b" in name_lower:
            return "qwen3-vl-2b"
        if "0.6b" in name_lower or "06b" in name_lower:
            return "qwen3-0.6b"
        for key in catalog:
            if key in name_lower:
                return key
        return default_key

    def get_catalog(self) -> Dict[str, Any]:
        """Trả về toàn bộ danh mục model phân bổ trên Dual-GPU."""
        return {
            "gpu_0_text_models": {
                "embedding": ["qwen3-0.6b", "qwen3-4b", "qwen3-8b"],
                "reranker": ["qwen3-0.6b", "qwen3-4b", "qwen3-8b"],
                "target_device": "cuda:0",
            },
            "gpu_1_vl_models": {
                "embedding": ["qwen3-vl-2b", "qwen3-vl-8b"],
                "reranker": ["qwen3-vl-2b", "qwen3-vl-8b"],
                "target_device": "cuda:1",
            },
            "embeddings": EMBEDDING_CATALOG,
            "rerankers": RERANK_CATALOG,
        }

    def get_embedding_engine(self, model_name: Optional[str] = None) -> QwenEmbeddingEngine:
        """Lấy hoặc khởi tạo Embedding Engine theo model requested."""
        key = self._normalize_key(model_name, EMBEDDING_CATALOG, self.default_embed_key)
        if key not in self.embedding_engines:
            cfg = EMBEDDING_CATALOG[key]
            self.embedding_engines[key] = QwenEmbeddingEngine(cfg)
        return self.embedding_engines[key]

    def get_rerank_engine(self, model_name: Optional[str] = None) -> QwenRerankEngine:
        """Lấy hoặc khởi tạo Rerank Engine theo model requested."""
        key = self._normalize_key(model_name, RERANK_CATALOG, self.default_rerank_key)
        if key not in self.rerank_engines:
            cfg = RERANK_CATALOG[key]
            self.rerank_engines[key] = QwenRerankEngine(cfg)
        return self.rerank_engines[key]

    def embed(
        self,
        inputs: Union[str, List[str]],
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Tạo embeddings chuẩn OpenAI API response."""
        engine = self.get_embedding_engine(model)
        t0 = time.time()
        vectors = engine.embed(inputs, dimensions=dimensions)
        elapsed = time.time() - t0

        data = []
        total_tokens = 0
        for idx, vec in enumerate(vectors):
            data.append({
                "object": "embedding",
                "index": idx,
                "embedding": vec,
            })
            total_tokens += len(vec) // 4  # Ước lượng tokens

        return {
            "object": "list",
            "data": data,
            "model": engine.model_id,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
            "elapsed_seconds": round(elapsed, 4),
            "target_device": engine.target_device,
        }

    def rerank(
        self,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        model: Optional[str] = None,
        top_n: Optional[int] = None,
        return_documents: bool = True,
    ) -> Dict[str, Any]:
        """Sắp xếp lại văn bản chuẩn Cohere/Jina response."""
        engine = self.get_rerank_engine(model)
        t0 = time.time()
        ranked = engine.rerank(query, documents, top_n=top_n)
        elapsed = time.time() - t0

        results = []
        for r in ranked:
            item = {
                "index": r["index"],
                "relevance_score": r["relevance_score"],
            }
            if return_documents:
                item["document"] = r["document"]
            results.append(item)

        return {
            "model": engine.model_id,
            "results": results,
            "usage": {
                "total_documents": len(documents),
                "ranked_documents": len(results),
            },
            "elapsed_seconds": round(elapsed, 4),
            "target_device": engine.target_device,
        }

    def preload_all_models(self):
        """Khởi động nhanh các model mặc định."""
        try:
            logger.info("⚡ [Preload Embedding] Khởi động Qwen3-Embedding-0.6B...")
            self.get_embedding_engine("qwen3-0.6b").load()
        except Exception as e:
            logger.warning(f"Preload embedding error: {e}")

        try:
            logger.info("⚡ [Preload Reranker] Khởi động Qwen3-Reranker-0.6B...")
            self.get_rerank_engine("qwen3-0.6b").load()
        except Exception as e:
            logger.warning(f"Preload reranker error: {e}")

    def free_vram(self):
        """Dọn sạch toàn bộ VRAM của nhánh Embedding."""
        for e in self.embedding_engines.values():
            e.release_from_gpu()
        for r in self.rerank_engines.values():
            r.release_from_gpu()
        self.embedding_engines.clear()
        self.rerank_engines.clear()
        logger.info("✨ [Free VRAM] Đã giải phóng toàn bộ VRAM của phân hệ Embedding!")


_manager_instance = None

def get_embedding_service_manager() -> EmbeddingServiceManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = EmbeddingServiceManager()
    return _manager_instance
