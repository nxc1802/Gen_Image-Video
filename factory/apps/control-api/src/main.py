"""Factory single-host Control Plane: selected backends, local jobs and artifacts."""

import os
from contextlib import AsyncExitStack, asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from factory_auth import load_auth_config, verify_bearer_token
from factory.apps.control_api.src.api.errors import register_exception_handlers
from factory.apps.control_api.src.api.middleware import (
    RequestIDMiddleware,
    RateLimitMiddleware,
    RequestTimeoutMiddleware,
)
from factory.apps.control_api.src.api.public.body_limit import PublicBodyLimit
from factory.apps.control_api.src.api.public.jobs import PublicMediaJobs
from factory.apps.control_api.src.api.routers.health import router as health_router
from factory.apps.control_api.src.api.routers.models import router as models_router
from factory.apps.control_api.src.api.routers.chat import router as chat_router
from factory.apps.control_api.src.api.routers.inference import router as inference_router
from factory.apps.control_api.src.api.routers.local_jobs import router as local_jobs_router
from factory.apps.control_api.src.storage.scratch_disk import ScratchDiskManager
from factory.apps.control_api.src.telemetry.middleware import TelemetryTracingMiddleware
from factory.apps.control_api.src.telemetry.logging import configure_telemetry_logging
from model_registry.loader import ManifestLoader
from model_registry.deployment import ModelCatalog, DeploymentRegistry, deployment_path, read_config
from model_registry.deployment_validator import DeploymentValidator
from factory_contracts.deployment import DeploymentConfig


@asynccontextmanager
async def lifespan(app):
    configure_telemetry_logging()
    app.state.auth_config = load_auth_config()
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
    registry = (
        carried
        if isinstance(carried, DeploymentRegistry)
        else DeploymentRegistry(
            config, ModelCatalog(app.state.manifest_loader.manifests_dir), config_path=config_path
        )
    )
    app.state.deployment_registry = registry
    app.state.backend_session = None
    async with AsyncExitStack() as cleanup:
        session = None
        if app.state.auth_config.environment == "production":
            if not config_path.is_file():
                raise RuntimeError("Explicit deployment configuration is required in production")
            report = getattr(app.state, "launch_preflight", None) if carried is registry else None
            if (
                not isinstance(report, dict)
                or report.get("config_sha256") != registry.config_sha256
            ):
                report = DeploymentValidator(registry).validate()
            if not report["preflight_passed"]:
                raise RuntimeError(
                    "Deployment environment preflight failed; run factory deployment validate"
                )
            from model_registry.backends.session import BackendSession

            session = BackendSession(registry)
            app.state.backend_session = session
            cleanup.push_async_callback(session.stop)
            await session.start()
            if registry.bindings() and not session.healthy_models():
                raise RuntimeError(
                    "No configured backend passed current-session launch/identity/health checks"
                )
        scratch = ScratchDiskManager()
        cleanup.push_async_callback(scratch.shutdown)
        await scratch.initialize()
        app.state.scratch_disk = scratch
        jobs = PublicMediaJobs(scratch)
        app.state.public_media_jobs = jobs
        # On shutdown cancel admission, stop owned backends to release native
        # RPCs, then drain jobs while their physical GPU locks remain held.
        cleanup.push_async_callback(jobs.close, stop_backends=session.stop if session else None)
        yield


def create_app(*, lifespan_handler=lifespan, include_job_management=True):
    result = FastAPI(
        title="Factory Control Plane API",
        version="4.0.0",
        description="Single-host inference on explicit model deployments",
        lifespan=lifespan_handler,
    )
    result.add_middleware(PublicBodyLimit)
    result.add_middleware(RequestTimeoutMiddleware)
    result.add_middleware(RateLimitMiddleware)
    result.add_middleware(TelemetryTracingMiddleware)
    result.add_middleware(RequestIDMiddleware)
    result.add_middleware(
        CORSMiddleware,
        allow_origins=[
            v.strip()
            for v in os.environ.get("FACTORY_CORS_ORIGINS", "http://localhost:3000").split(",")
            if v.strip()
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(result)
    for router in (health_router, models_router, chat_router, inference_router):
        result.include_router(router)
    if include_job_management:
        result.include_router(local_jobs_router)

    @result.get("/v1/me", tags=["Auth"])
    async def current_user(user: dict = Depends(verify_bearer_token)):
        return {"user": user}

    return result


app = create_app()
