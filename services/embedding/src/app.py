"""
services/embedding/src/app.py
FastAPI application for the Factory v4 Vector Embedding & Rerank Subsystem (Port 8083).
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from factory_auth import verify_bearer_token
from services.embedding.src.config import load_embedding_config
from services.embedding.src.engine import EmbeddingEngine
from services.embedding.src.schemas import EmbeddingRequest, RerankRequest
from services.text.src.app import RequestIDMiddleware, register_error_handlers

logger = logging.getLogger("services.embedding")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from factory_auth import load_auth_config
    app.state.auth_config = load_auth_config()
    app.state.config = load_embedding_config()
    app.state.engine = EmbeddingEngine()
    yield


app = FastAPI(
    title="Factory v4 Vector Embedding & Rerank Daemon",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get("FACTORY_CORS_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_error_handlers(app)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "embed-service",
        "version": "1.0.0",
        "port": 8083,
    }


# --- Vector Embeddings Endpoints ---
@app.post("/v1/embeddings", dependencies=[Depends(verify_bearer_token)])
async def create_embeddings_v1(request: EmbeddingRequest):
    """OpenAI-compatible vector embeddings endpoint."""
    engine: EmbeddingEngine = getattr(app.state, "engine", None) or EmbeddingEngine()
    return engine.generate_embeddings(request)


@app.post("/v2/embed", dependencies=[Depends(verify_bearer_token)])
async def create_embeddings_v2(request: EmbeddingRequest):
    """High-performance v2 vector embedding route for multimodal content."""
    engine: EmbeddingEngine = getattr(app.state, "engine", None) or EmbeddingEngine()
    return engine.generate_embeddings(request)


# --- Document Rerank Endpoints ---
@app.post("/v1/rerank", dependencies=[Depends(verify_bearer_token)])
async def rerank_documents_v1(request: RerankRequest):
    """Cross-encoder document relevance reranking endpoint."""
    engine: EmbeddingEngine = getattr(app.state, "engine", None) or EmbeddingEngine()
    return engine.rerank_documents(request)


@app.post("/rerank", dependencies=[Depends(verify_bearer_token)])
async def rerank_documents_legacy(request: RerankRequest):
    """Legacy compatibility route for /rerank."""
    engine: EmbeddingEngine = getattr(app.state, "engine", None) or EmbeddingEngine()
    return engine.rerank_documents(request)


@app.post("/v2/rerank", dependencies=[Depends(verify_bearer_token)])
async def rerank_documents_v2(request: RerankRequest):
    """Extended v2 route for document reranking."""
    engine: EmbeddingEngine = getattr(app.state, "engine", None) or EmbeddingEngine()
    return engine.rerank_documents(request)
