"""
factory/apps/control-api/src/api/routers/models.py
Declarative Model and Service Registry endpoints for Factory v4 Lite.
"""
from fastapi import APIRouter, Depends, Request
from factory_auth import verify_bearer_token
from model_registry.loader import ManifestLoader

router = APIRouter(prefix="/v1", tags=["Registry"], dependencies=[Depends(verify_bearer_token)])


@router.get("/models")
async def list_models(request: Request):
    """Lists all registered models dynamically from manifests."""
    loader = getattr(request.app.state, "manifest_loader", None) or ManifestLoader()
    models = loader.load_models()
    return {"data": [m.model_dump() for m in models.values()]}


@router.get("/services")
async def list_services(request: Request):
    """Lists all registered services dynamically from manifests."""
    loader = getattr(request.app.state, "manifest_loader", None) or ManifestLoader()
    services = loader.load_services()
    return {"data": [s.model_dump() for s in services.values()]}


@router.get("/capabilities")
async def list_capabilities(request: Request):
    """Lists all declared capabilities dynamically from manifests."""
    loader = getattr(request.app.state, "manifest_loader", None) or ManifestLoader()
    capabilities = loader.load_capabilities()
    caps_list = [c.model_dump(mode="json") for c in capabilities.values()]
    return {"capabilities": caps_list, "data": caps_list}
