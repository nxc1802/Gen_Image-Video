"""Retained worker/lease/database migration host. Not imported by core startup."""

import asyncio
from contextlib import asynccontextmanager, suppress, AsyncExitStack
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
from factory.apps.control_api.src.api.routers.chat import router as chat_router
from factory.apps.control_api.src.api.routers.inference import router as inference_router
from factory.apps.control_api.src.api.routers.workers import router as workers_router
from factory.apps.control_api.src.api.routers.inspector import (
    router as inspector_router,
    memory_router,
)
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
from model_registry.deployment import ModelCatalog, DeploymentRegistry, deployment_path, read_config
from model_registry.deployment_validator import DeploymentValidator
from factory_contracts.deployment import DeploymentConfig


async def _stop_task(task):
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


@asynccontextmanager
async def legacy_lifespan(app: FastAPI):
    # Actively configure structured JSON logging
    configure_telemetry_logging()

    # Enforces fail-fast startup guard if FACTORY_API_KEY is empty in production
    auth_config = load_auth_config()
    app.state.auth_config = auth_config
    app.state.manifest_loader = ManifestLoader()
    carried = getattr(app.state, "launch_registry", None)
    config_path = (
        carried.config_path if isinstance(carried, DeploymentRegistry) else deployment_path()
    )
    config = (
        carried.config
        if isinstance(carried, DeploymentRegistry)
        else read_config(config_path)
        if config_path.is_file()
        else DeploymentConfig()
    )
    deployment_registry = (
        carried
        if isinstance(carried, DeploymentRegistry)
        else DeploymentRegistry(
            config, ModelCatalog(app.state.manifest_loader.manifests_dir), config_path=config_path
        )
    )
    app.state.deployment_registry = deployment_registry
    async with AsyncExitStack() as cleanup:
        backend_session = None
        app.state.backend_session = None
        if app.state.auth_config.environment == "production":
            if not config_path.is_file():
                raise RuntimeError("Explicit deployment configuration is required in production")
            report = (
                getattr(app.state, "launch_preflight", None)
                if carried is deployment_registry
                else None
            )
            if (
                not isinstance(report, dict)
                or report.get("config_sha256") != deployment_registry.config_sha256
            ):
                report = DeploymentValidator(deployment_registry).validate()
            if not report["preflight_passed"]:
                raise RuntimeError(
                    "Deployment environment preflight failed; run factory deployment validate"
                )
            from model_registry.backends.session import BackendSession

            backend_session = BackendSession(deployment_registry)
            app.state.backend_session = backend_session
            cleanup.push_async_callback(backend_session.stop)
            await backend_session.start()
            if deployment_registry.bindings() and not backend_session.healthy_models():
                raise RuntimeError(
                    "No configured backend passed current-session launch/identity/health checks"
                )

        # 1. Initialize Scratch Disk Storage & Background Sweeper
        scratch_disk = ScratchDiskManager()
        cleanup.push_async_callback(scratch_disk.shutdown)
        await scratch_disk.initialize()
        app.state.scratch_disk = scratch_disk
        from factory.apps.control_api.src.api.public.jobs import PublicMediaJobs

        app.state.public_media_jobs = PublicMediaJobs(scratch_disk)
        cleanup.push_async_callback(app.state.public_media_jobs.close)

        # Storage Provider (S3 when configured, Local scratch disk otherwise)
        storage_config = load_storage_config()
        if storage_config.storage_backend == "s3":
            from factory.apps.control_api.src.storage.s3 import S3StorageProvider

            storage_provider = S3StorageProvider(storage_config)
        else:
            storage_provider = LocalStorageProvider(storage_config)
        app.state.storage_provider = storage_provider

        # 2. Initialize In-Memory Queue & Lifecycle Engine
        auto_migrate = os.environ.get("FACTORY_AUTO_MIGRATE", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        db_backend = os.environ.get("FACTORY_DB_BACKEND", "memory").lower()
        if auto_migrate and db_backend in ("sqlite", "postgres"):
            from factory.apps.control_api.src.queue.repository.migration_runner import (
                run_migrations,
            )

            db_url = os.environ.get("DATABASE_URL") or os.environ.get("SQLITE_DB_PATH")
            if db_url and db_url != ":memory:":
                resolved_url = (
                    db_url if db_url.startswith(("sqlite:", "postgres")) else f"sqlite:///{db_url}"
                )
                await asyncio.to_thread(run_migrations, db_url=resolved_url)

        repository = get_job_repository()
        await repository.initialize()
        cleanup.push_async_callback(repository.close)
        app.state.repository = repository

        broker = get_queue_broker()
        await broker.initialize()
        cleanup.push_async_callback(broker.close)
        app.state.broker = broker
        app.state.scheduler = broker

        from factory.apps.control_api.src.queue.model_runtime import ModelRuntimeManager

        model_runtime_manager = ModelRuntimeManager(
            scheduler=broker, num_gpus=2, deployment_registry=deployment_registry
        )
        app.state.model_runtime_manager = model_runtime_manager
        app.state.runtime_manager = model_runtime_manager

        stream_manager = StreamManager(
            repository=repository,
        )
        app.state.stream_manager = stream_manager

        job_manager = JobLifecycleManager(
            repository=repository,
            broker=broker,
            stream_manager=stream_manager,
            deployment_registry=deployment_registry,
        )
        app.state.job_manager = job_manager

        lease_coordinator = LeaseCoordinator(
            repository=repository,
            broker=broker,
            manifest_loader=app.state.manifest_loader,
            deployment_registry=deployment_registry,
        )
        app.state.lease_coordinator = lease_coordinator

        app.state.asset_service = AssetService(max_size_bytes=MAX_PAYLOAD_SIZE)

        # Background watchdog running every 10s for deterministic failure handling
        watchdog_task = asyncio.create_task(watchdog_loop(app, interval_seconds=10.0))
        cleanup.push_async_callback(_stop_task, watchdog_task)

        yield


from factory.apps.control_api.src.main import create_app


@asynccontextmanager
async def migration_lifespan(app):
    if load_auth_config().environment == "production":
        raise RuntimeError(
            "Legacy worker control plane is retained only for migration tests; use the canonical factory serve entrypoint in production"
        )
    async with legacy_lifespan(app):
        yield


def create_legacy_app():
    """Explicit legacy fixtures/Studio migration host; never production inference."""
    result = create_app(lifespan_handler=migration_lifespan, include_job_management=False)
    for router in (
        jobs_router,
        workers_router,
        storage_router,
        inspector_router,
        memory_router,
        assets_router,
    ):
        result.include_router(router)
    return result


app = create_legacy_app()
