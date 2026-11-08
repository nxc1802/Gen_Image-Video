"""
factory/apps/control-api/src/api/routers/models.py
Declarative Model and Service Registry endpoints for Factory v4 Lite.
"""
from typing import Optional
from fastapi import APIRouter, Depends, Query, Request
from factory_auth import verify_bearer_token
from model_registry.loader import ManifestLoader

router = APIRouter(prefix="/v1", tags=["Registry"], dependencies=[Depends(verify_bearer_token)])


@router.get("/models")
@router.get("/registry/models")
async def list_models(
    request: Request,
    family: Optional[str] = Query(default=None),
    service_id: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    validation_status: Optional[str] = Query(default=None),
):
    """Lists all registered models dynamically from manifests with capability truth metadata."""
    loader = getattr(request.app.state, "manifest_loader", None) or ManifestLoader()
    models = loader.load_models()
    result = list(models.values())

    if family:
        norm_fam = family.strip().lower()
        result = [m for m in result if m.metadata.family.lower() == norm_fam]
    if service_id:
        norm_svc = service_id.strip().lower()
        result = [m for m in result if m.metadata.service_id.lower() == norm_svc]
    if status:
        norm_st = status.strip().lower()
        result = [m for m in result if m.status.value.lower() == norm_st]
    if validation_status:
        norm_vs = validation_status.strip().lower()
        result = [m for m in result if m.validation_status.value.lower() == norm_vs]

    dumped = [m.model_dump(mode="json") for m in result]
    return {
        "models": dumped,
        "data": dumped,
        "total": len(dumped),
    }


@router.get("/services")
@router.get("/registry/services")
async def list_services(
    request: Request,
    subsystem: Optional[str] = Query(default=None),
):
    """Lists all registered services dynamically from manifests."""
    loader = getattr(request.app.state, "manifest_loader", None) or ManifestLoader()
    services = loader.load_services()
    result = list(services.values())

    if subsystem:
        norm_sub = subsystem.strip().lower()
        result = [s for s in result if getattr(s.metadata, "subsystem", "").lower() == norm_sub]

    dumped = [s.model_dump(mode="json") for s in result]
    return {
        "services": dumped,
        "data": dumped,
        "total": len(dumped),
    }


@router.get("/capabilities")
@router.get("/registry/capabilities")
async def list_capabilities(request: Request):
    """Lists all declared capabilities dynamically from manifests."""
    loader = getattr(request.app.state, "manifest_loader", None) or ManifestLoader()
    capabilities = loader.load_capabilities()
    caps_list = [c.model_dump(mode="json") for c in capabilities.values()]
    return {
        "capabilities": caps_list,
        "data": caps_list,
        "total": len(caps_list),
    }
