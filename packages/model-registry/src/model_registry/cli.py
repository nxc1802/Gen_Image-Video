"""
packages/model-registry/src/model_registry/cli.py
CLI entrypoint wrapper for factory-manifest
"""
import argparse
import json
import sys
import yaml

from factory_contracts.cli import main
from model_registry.loader import ManifestLoader
from factory_contracts.manifests.catalog import model_identity_index


def factory_main(argv=None):
    """Catalog inspection and explicit deployment configuration."""
    parser = argparse.ArgumentParser(prog="factory")
    parser.add_argument("--manifests-dir", default=None)
    parser.add_argument("--deployment", default=None, help="Deployment YAML (or FACTORY_DEPLOYMENT_FILE)")
    groups = parser.add_subparsers(dest="group", required=True)
    models_parser = groups.add_parser("models")
    commands = models_parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("model")
    from model_registry.deployment_cli import add_commands
    add_commands(groups)
    args = parser.parse_args(argv)
    try:
        if args.group != "models":
            from model_registry.deployment_cli import handle
            result, code = handle(args)
            print(json.dumps(result, indent=2))
            return code
        loader = ManifestLoader(args.manifests_dir)
        models = loader.load_models()
        if not models:
            raise ValueError("No model manifests found in the configured catalog")
        if args.command == "inspect":
            aliases, _ = model_identity_index(loader.manifests_dir)
            model_id = aliases.get(args.model.strip().lower(), args.model)
            if model_id not in models:
                for path in (loader.manifests_dir / "retired").glob("*.yaml"):
                    retired = yaml.safe_load(path.read_text())
                    if args.model.strip().lower() in [retired["id"], *retired.get("aliases", [])]:
                        print(json.dumps({"error": {"code": "model_retired", "message": retired["reason"], "replacements": retired["replacements"]}}), file=sys.stderr)
                        return 1
                print(json.dumps({"error": {"code": "model_not_found", "message": "Model is not in the configured catalog."}}), file=sys.stderr)
                return 1
            model = models[model_id]
            complete = model.apiVersion == "factory.ai/v2"
            result = {
                "model": model.metadata.id,
                "metadata": model.metadata.model_dump(mode="json"),
                "engine": model.engine.value if model.engine else None,
                "checkpoint": model.weights.model_dump(mode="json", include={"files", "format", "quantization", "default_precision"}),
                "backend": model.backend.model_dump(mode="json") if model.backend else None,
                "aliases": model.aliases, "capabilities": model.capabilities,
                "capability_truth": {key: value.model_dump(mode="json") for key, value in model.capabilities_truth.items()},
                "parameters": model.parameters,
                "runtime_profiles": {key: value.model_dump(mode="json") for key, value in model.runtime_profiles.items()},
                "validation": model.validation.model_dump(mode="json") if model.validation else None,
                "current_session": {"attested": False, "production_ready": False},
                "canonical_truth": {
                    "complete": complete,
                    "reason": "Pinned static model truth; no runtime or inference certification." if complete else "Legacy manifest requires canonical migration.",
                },
            }
        else:
            result = {"object": "catalog", "deployment_registry": False, "models": [
                {"id": model.metadata.id, "checkpoint": model.weights.primary_checkpoint,
                 "revision": model.weights.checkpoint_revision, "engine": model.engine.value if model.engine else None,
                 "backend": model.backend.model_dump(mode="json") if model.backend else None,
                 "required_hardware": {key: value.model_dump(mode="json") for key, value in model.runtime_profiles.items()},
                 "capability_claims": model.capabilities,
                 "validation_status": model.validation.state.value if model.validation else model.validation_status.value,
                 "session_attested": False, "production_ready": False}
                for model in models.values()
            ]}
        print(json.dumps(result, indent=2))
        if args.command == "inspect" and not result["canonical_truth"]["complete"]:
            return 2
        return 0
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(json.dumps({"error": {"code": "invalid_catalog" if args.group == "models" else "invalid_deployment", "message": str(exc)}}), file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(factory_main())
