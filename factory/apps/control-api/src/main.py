"""
factory/apps/control-api/src/main.py
Factory v4 Control Plane API Server Entrypoint
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from factory_auth import load_auth_config, verify_bearer_token
from factory.apps.control_api.src.api.middleware import RequestIDMiddleware
from factory.apps.control_api.src.api.errors import register_exception_handlers
from factory.apps.control_api.src.api.routers.jobs import router as jobs_router
from factory.apps.control_api.src.api.routers.workers import router as workers_router
from factory.apps.control_api.src.api.routers.inspector import router as inspector_router
from factory.apps.control_api.src.storage.router import router as storage_router
from factory.apps.control_api.src.storage.config import load_storage_config
from factory.apps.control_api.src.storage.s3 import S3StorageProvider
from factory.apps.control_api.src.storage.local import LocalStorageProvider
from factory.apps.control_api.src.queue.repository import get_job_repository
from factory.apps.control_api.src.queue.broker import get_queue_broker
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.reaper import DeadJobReaper
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.streaming.manager import StreamManager
from factory.apps.control_api.src.telemetry.middleware import TelemetryTracingMiddleware
from factory.apps.control_api.src.telemetry.logging import configure_telemetry_logging
from model_registry.loader import ManifestLoader


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Enforces fail-fast startup guard if FACTORY_API_KEY is empty in production
    auth_config = load_auth_config()
    app.state.auth_config = auth_config
    app.state.manifest_loader = ManifestLoader()

    # 1. Initialize Storage Provider
    storage_config = load_storage_config()
    if storage_config.storage_backend == "s3" and storage_config.s3_endpoint_url:
        try:
            storage_provider = S3StorageProvider(storage_config)
        except Exception:
            storage_provider = LocalStorageProvider(storage_config)
    else:
        storage_provider = LocalStorageProvider(storage_config)
    app.state.storage_provider = storage_provider

    # 2. Initialize Two-Tier Queue & Lifecycle Engine
    repository = get_job_repository()
    await repository.initialize()
    app.state.repository = repository

    broker = get_queue_broker()
    await broker.initialize()
    app.state.broker = broker

    stream_manager = StreamManager()
    app.state.stream_manager = stream_manager

    job_manager = JobLifecycleManager(
        repository=repository,
        broker=broker,
        stream_manager=stream_manager,
    )
    app.state.job_manager = job_manager

    lease_coordinator = LeaseCoordinator(repository=repository, broker=broker)
    app.state.lease_coordinator = lease_coordinator

    reaper = DeadJobReaper(
        repository=repository,
        broker=broker,
        stream_manager=stream_manager,
    )
    reaper.start()
    app.state.reaper = reaper

    yield

    # Graceful shutdown
    await reaper.stop()
    await broker.close()
    await repository.close()


app = FastAPI(
    title="Factory v4 Control Plane API",
    version="4.0.0",
    description="Dedicated Control Plane for Factory v4 generative workflows",
    lifespan=lifespan,
)

# 1. Middlewares
app.add_middleware(RequestIDMiddleware)
app.add_middleware(TelemetryTracingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://studio.factory.domain", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Sanitized Exception Handlers
register_exception_handlers(app)

# 3. Router Inclusions
app.include_router(jobs_router)
app.include_router(workers_router)
app.include_router(storage_router)
app.include_router(inspector_router)


# 4. Public Health Endpoints
@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "healthy",
        "plane": "control",
        "version": "4.0.0",
    }


# 5. Protected Endpoints (Enforces Mandatory Bearer Token)
@app.get("/v1/models", tags=["Registry"], dependencies=[Depends(verify_bearer_token)])
async def list_models(request: Request):
    loader = getattr(request.app.state, "manifest_loader", ManifestLoader())
    models = loader.load_models()
    return {"data": [m.model_dump() for m in models.values()]}


@app.get("/v1/services", tags=["Registry"], dependencies=[Depends(verify_bearer_token)])
async def list_services(request: Request):
    loader = getattr(request.app.state, "manifest_loader", ManifestLoader())
    services = loader.load_services()
    return {"data": [s.model_dump() for s in services.values()]}


@app.get("/v1/capabilities", tags=["Registry"], dependencies=[Depends(verify_bearer_token)])
async def list_capabilities(request: Request):
    loader = getattr(request.app.state, "manifest_loader", ManifestLoader())
    capabilities = loader.load_capabilities()
    caps_list = [c.model_dump(mode="json") for c in capabilities.values()]
    return {"capabilities": caps_list, "data": caps_list}


@app.get("/v1/me", tags=["Auth"])
async def get_current_user(user: dict = Depends(verify_bearer_token)):
    return {"user": user}
