"""Explicit deployment configuration commands; no backend fallback."""
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

from factory_contracts.deployment import DeploymentConfig, DeploymentEntry, BackendOptions
from .deployment import ModelCatalog, DeploymentRegistry, deployment_path, read_config, write_config
from .deployment_validator import DeploymentValidator


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward session credentials to a redirected origin.


def add_commands(groups):
    deployment = groups.add_parser("deployment", aliases=["deploy"])
    commands = deployment.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--models-dir", type=Path, default=Path("models"))
    commands.add_parser("show")
    add = commands.add_parser("add")
    add.add_argument("model")
    add.add_argument("--gpu", required=True, help="Explicit physical GPU IDs, e.g. 0 or 0,1")
    add.add_argument("--backend")
    add.add_argument("--port", type=int)
    add.add_argument("--weights-dir", type=Path)
    add.add_argument("--exclusive", action="store_true")
    add.add_argument("--backend-executable", type=Path)
    add.add_argument("--backend-dir", type=Path)
    add.add_argument("--backend-revision")
    remove = commands.add_parser("remove")
    remove.add_argument("model")
    validate = commands.add_parser("validate")
    validate.add_argument("--config-only", action="store_true", help="Check configuration only; does not verify the host, weights or installed providers")
    serve = groups.add_parser("serve")
    serve.add_argument("--check", action="store_true", help="Run full local preflight without launching backends/API")
    status = groups.add_parser("status")
    status.add_argument("--live", action="store_true", help="Read the running local API session rather than configuration only")


def handle(args):
    path = deployment_path(args.deployment)
    catalog = ModelCatalog(args.manifests_dir)
    if args.group in {"deployment", "deploy"} and args.command == "init":
        config = DeploymentConfig(models_dir=args.models_dir)
        write_config(path, config, create=True)
        return {"configuration_file": str(path), "configuration": config.model_dump(mode="json"), "production_ready": False}, 0
    config = read_config(path)
    if args.group in {"deployment", "deploy"} and args.command in {"add", "remove"}:
        mid = catalog.resolve(args.model)
        entries = list(config.deployments)
        if args.command == "remove":
            remaining = [entry for entry in entries if catalog.resolve(entry.model) != mid]
            if len(remaining) == len(entries):
                raise ValueError(f"Model is not configured: {mid}")
            config = config.model_copy(update={"deployments": tuple(remaining)})
        else:
            if any(catalog.resolve(entry.model) == mid for entry in entries):
                raise ValueError(f"Model already configured: {mid}; remove it before replacing its binding")
            model = catalog.get(mid)
            profile = model.runtime_profiles.get(config.hardware)
            if not profile:
                raise ValueError(f"No runtime profile for {mid} on {config.hardware}")
            used_ports = {config.api_port, *(entry.port for entry in entries if entry.port)}
            port = args.port or next(port for port in range(9000, 65536) if port not in used_ports)
            if any(not piece.strip() for piece in args.gpu.split(",")):
                raise ValueError("GPU IDs must be explicit comma-separated integers")
            devices = tuple(int(piece) for piece in args.gpu.split(","))
            entry = DeploymentEntry(model=mid, devices=devices, backend=args.backend or profile.preferred_backend,
                port=port, exclusive=args.exclusive, weights_dir=args.weights_dir,
                backend_options=BackendOptions(executable=args.backend_executable, source_dir=args.backend_dir, source_revision=args.backend_revision))
            config = config.model_copy(update={"deployments": tuple([*entries, entry])})
            report = DeploymentValidator(DeploymentRegistry(config, catalog, config_path=path)).validate(config_only=True)
            if not report["configuration_valid"]:
                return report, 2
        write_config(path, config)
    registry = DeploymentRegistry(config, catalog, config_path=path)
    if args.group == "status":
        if args.live:
            key = os.environ.get("FACTORY_API_KEY", "").strip()
            if not key:
                raise ValueError("FACTORY_API_KEY is required to query the running session")
            host = "[::1]" if config.api_host == "::1" else "127.0.0.1"
            request = urllib.request.Request(f"http://{host}:{config.api_port}/v1/registry/deployment",
                headers={"Authorization": f"Bearer {key}"})
            try:
                with urllib.request.build_opener(_NoRedirect()).open(request, timeout=5) as response:
                    return {"observation": "running_session", "status": json.load(response)}, 0
            except (OSError, ValueError) as exc:
                return {"error": {"code": "SESSION_UNAVAILABLE", "message": type(exc).__name__}, "production_ready": False}, 2
        return {"observation": "configuration_only", "configuration_file": str(path), "configured": registry.snapshot(),
                "running_session_observed": False, "production_ready": False}, 0
    if args.group == "serve" or (args.group in {"deployment", "deploy"} and args.command == "validate"):
        report = DeploymentValidator(registry).validate(config_only=getattr(args, "config_only", False))
        passed = report["configuration_valid"] if report["mode"] == "configuration_only" else report["preflight_passed"]
        if not passed:
            return report, 2
        if args.group == "serve" and not args.check:
            if not os.environ.get("FACTORY_API_KEY", "").strip():
                return {"error": {"code": "API_KEY_REQUIRED", "message": "FACTORY_API_KEY is required before serving."}}, 2
            import uvicorn
            from factory_control_api import app
            # Uvicorn does not expose the API until lifespan startup finishes.
            # The exact preflight registry is carried into that lifespan.
            app.state.launch_registry = registry
            app.state.launch_preflight = report
            previous = os.environ.get("ENVIRONMENT")
            os.environ["ENVIRONMENT"] = "production"
            try:
                uvicorn.run(app, host=config.api_host, port=config.api_port)
            finally:
                for name in ("launch_registry", "launch_preflight"):
                    if hasattr(app.state, name):
                        delattr(app.state, name)
                if previous is None:
                    os.environ.pop("ENVIRONMENT", None)
                else:
                    os.environ["ENVIRONMENT"] = previous
            return {"session_id": registry.session_id, "state": "STOPPED", "production_ready": False}, 0
        return report, 0
    return {"configuration_file": str(path), "configured": registry.snapshot(), "active_models": [], "production_ready": False}, 0
