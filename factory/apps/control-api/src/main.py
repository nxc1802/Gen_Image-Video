"""
factory/apps/control-api/src/main.py
Factory v4 Control Plane API Server Entrypoint
"""
import asyncio
from contextlib import asynccontextmanager, suppress
import os
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from factory_auth import load_auth_config, verify_bearer_token
from factory.apps.control_api.src.api.middleware import (
    RequestIDMiddleware,
    RateLimitMiddleware,
    RequestTimeoutMiddleware,
)
from factory.apps.control_api.src.api.errors import register_exception_handlers
from factory.apps.control_api.src.api.routers.health import router as health_router
from factory.apps.control_api.src.api.routers.models import router as models_router
from factory.apps.control_api.src.api.routers.jobs import router as jobs_router
from factory.apps.control_api.src.api.routers.workers import router as workers_router
from factory.apps.control_api.src.api.routers.inspector import router as inspector_router
from factory.apps.control_api.src.api.routers.assets import router as assets_router
from factory.apps.control_api.src.assets.service import AssetService, MAX_PAYLOAD_SIZE
from factory.apps.control_api.src.storage.router import router as storage_router
from factory.apps.control_api.src.storage.config import load_storage_config
from factory.apps.control_api.src.storage.local import LocalStorageProvider
from factory.apps.control_api.src.storage.scratch_disk import ScratchDiskManager
from factory.apps.control_api.src.queue.repository import get_job_repository
from factory.apps.control_api.src.queue.broker import get_queue_broker
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from factory.apps.control_api.src.queue.watchdog import watchdog_loop
from factory.apps.control_api.src.streaming.manager import StreamManager
from factory.apps.control_api.src.telemetry.middleware import TelemetryTracingMiddleware
from factory.apps.control_api.src.telemetry.logging import configure_telemetry_logging
from model_registry.loader import ManifestLoader


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Actively configure structured JSON logging
    configure_telemetry_logging()

    # Enforces fail-fast startup guard if FACTORY_API_KEY is empty in production
    auth_config = load_auth_config()
    app.state.auth_config = auth_config
    app.state.manifest_loader = ManifestLoader()

    # 1. Initialize Scratch Disk Storage & Background Sweeper
    scratch_disk = ScratchDiskManager()
    await scratch_disk.initialize()
    app.state.scratch_disk = scratch_disk

    # Storage Provider (Local scratch disk for Factory v4 Lite compatibility)
    storage_config = load_storage_config()
    storage_provider = LocalStorageProvider(storage_config)
    app.state.storage_provider = storage_provider

    # 2. Initialize In-Memory Queue & Lifecycle Engine
    repository = get_job_repository()
    await repository.initialize()
    app.state.repository = repository

    broker = get_queue_broker()
    await broker.initialize()
    app.state.broker = broker
    app.state.scheduler = broker

    stream_manager = StreamManager(
        repository=repository,
    )
    app.state.stream_manager = stream_manager

    job_manager = JobLifecycleManager(
        repository=repository,
        broker=broker,
        stream_manager=stream_manager,
    )
    app.state.job_manager = job_manager

    lease_coordinator = LeaseCoordinator(repository=repository, broker=broker)
    app.state.lease_coordinator = lease_coordinator

    app.state.asset_service = AssetService(max_size_bytes=MAX_PAYLOAD_SIZE)

    # Background watchdog running every 10s for deterministic failure handling
    watchdog_task = asyncio.create_task(watchdog_loop(app, interval_seconds=10.0))

    yield

    # Graceful shutdown
    watchdog_task.cancel()
    with suppress(asyncio.CancelledError):
        await watchdog_task
    await scratch_disk.shutdown()
    await broker.close()
    await repository.close()


app = FastAPI(
    title="Factory v4 Control Plane API",
    version="4.0.0",
    description="Dedicated Control Plane for Factory v4 generative workflows",
    lifespan=lifespan,
)

# 1. Middlewares
app.add_middleware(RequestTimeoutMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(TelemetryTracingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get(
            "FACTORY_CORS_ORIGINS",
            "http://localhost:3000",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Sanitized Exception Handlers
register_exception_handlers(app)

# 3. Router Inclusions
app.include_router(health_router)
app.include_router(models_router)
app.include_router(jobs_router)
app.include_router(workers_router)
app.include_router(storage_router)
app.include_router(inspector_router)
app.include_router(assets_router)


# 4. User identity
@app.get("/v1/me", tags=["Auth"])
async def get_current_user(user: dict = Depends(verify_bearer_token)):
    return {"user": user}
