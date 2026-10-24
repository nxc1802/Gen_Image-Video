"""
services/decision/src/app.py
FastAPI application for the Factory v4 Decision Engine Subsystem (Port 8082).
"""
import logging
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from factory_auth import verify_bearer_token
from services.decision.src.config import load_decision_config
from services.decision.src.engine import DecisionEngine
from services.decision.src.schemas import DecisionRequest
from services.text.src.app import RequestIDMiddleware, register_error_handlers

logger = logging.getLogger("services.decision")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from factory_auth import load_auth_config
    app.state.auth_config = load_auth_config()
    app.state.config = load_decision_config()
    app.state.engine = DecisionEngine()
    yield


app = FastAPI(
    title="Factory v4 Decision Routing Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_error_handlers(app)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "decision-service",
        "version": "1.0.0",
        "port": 8082,
    }


@app.post("/v1/decisions", dependencies=[Depends(verify_bearer_token)])
async def evaluate_decision(request: DecisionRequest):
    """Evaluates decision tasks: predicate, choice, or continuous calibrated score."""
    engine: DecisionEngine = getattr(app.state, "engine", None) or DecisionEngine()
    return engine.evaluate(request)
