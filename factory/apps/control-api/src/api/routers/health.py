"""
factory/apps/control-api/src/api/routers/health.py
Public Healthcheck endpoints for Factory v4 Control Plane API.
"""
from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
@router.get("/healthz")
async def health():
    return {
        "status": "healthy",
        "plane": "control",
        "version": "4.0.0",
    }
