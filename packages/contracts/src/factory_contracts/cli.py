"""
factory_contracts/cli.py
Command-Line Interface for Manifest Validation and TypeScript Code Generation
"""
import sys
from pathlib import Path
from typing import Dict, List
import yaml

from factory_contracts.manifests.service import ServiceManifest
from factory_contracts.manifests.model import ModelManifest, ModelStatus
from factory_contracts.manifests.hardware import HardwareTopologyManifest
from factory_contracts.manifests.capability import CapabilityManifest


def validate_manifests(path: str = "manifests") -> tuple[bool, List[str], Dict[str, int]]:
    manifest_dir = Path(path)
    if not manifest_dir.exists():
        return False, [f"Manifest directory not found: {manifest_dir.resolve()}"], {}

    errors: List[str] = []
    services: Dict[str, ServiceManifest] = {}
    models: Dict[str, ModelManifest] = {}
    capabilities: Dict[str, CapabilityManifest] = {}
    used_ports: Dict[int, str] = {}
    used_domains: Dict[str, str] = {}

    # 1. Load Services
    service_dir = manifest_dir / "services"
    service_files = list(service_dir.glob("*.yaml")) + list(service_dir.glob("*.yml"))
    for sf in sorted(service_files):
        try:
            with open(sf, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not data:
                continue
            sm = ServiceManifest.model_validate(data)

            # Port collision check
            port = sm.network.internal_port
            if port in used_ports:
                errors.append(f"Port collision: Port {port} used by both '{used_ports[port]}' and '{sm.metadata.id}'")
            else:
                used_ports[port] = sm.metadata.id

            # Domain collision check
            domain = sm.network.static_domain
            if domain in used_domains:
                errors.append(f"Domain collision: '{domain}' used by '{used_domains[domain]}' and '{sm.metadata.id}'")
            else:
                used_domains[domain] = sm.metadata.id

            services[sm.metadata.id] = sm
        except Exception as e:
            errors.append(f"Service manifest error in {sf.name}: {e}")

    # 2. Load Capabilities
    cap_dir = manifest_dir / "capabilities"
    if cap_dir.exists():
        cap_files = list(cap_dir.glob("*.yaml")) + list(cap_dir.glob("*.yml"))
        for cf in sorted(cap_files):
            try:
                with open(cf, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if not data:
                    continue
                cm = CapabilityManifest.model_validate(data)
                capabilities[cm.id] = cm
            except Exception as e:
                errors.append(f"Capability manifest error in {cf.name}: {e}")

    # 3. Load Models
    model_dir = manifest_dir / "models"
    model_files = list(model_dir.glob("*.yaml")) + list(model_dir.glob("*.yml"))
    for mf in sorted(model_files):
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not data:
                continue
            mm = ModelManifest.model_validate(data)

            # Referential integrity: service_id must exist in loaded services
            if services and mm.metadata.service_id not in services:
                errors.append(
                    f"Model '{mm.metadata.id}' references unknown service_id '{mm.metadata.service_id}'"
                )

            # Completeness enforcement
            if mm.status == ModelStatus.FULL:
                if mm.coverage_percent < 100.0:
                    errors.append(
                        f"Model '{mm.metadata.id}' marked FULL but coverage is {mm.coverage_percent}%"
                    )
                if len(mm.missing_capabilities) > 0:
                    errors.append(
                        f"Model '{mm.metadata.id}' marked FULL but lists missing capabilities: {mm.missing_capabilities}"
                    )
            elif mm.status == ModelStatus.PARTIAL:
                false_caps = [k for k, v in mm.capabilities.items() if not v]
                if mm.coverage_percent >= 100.0 and len(mm.missing_capabilities) == 0 and not false_caps:
                    errors.append(
                        f"Model '{mm.metadata.id}' has 100% coverage and no missing capabilities; must be status: 'full'"
                    )

            models[mm.metadata.id] = mm
        except Exception as e:
            errors.append(f"Model manifest error in {mf.name}: {e}")

    # 4. Load Hardware Topologies
    hw_dir = manifest_dir / "hardware"
    hw_files = list(hw_dir.glob("*.yaml")) + list(hw_dir.glob("*.yml"))
    for hf in sorted(hw_files):
        try:
            with open(hf, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not data:
                continue
            hm = HardwareTopologyManifest.model_validate(data)

            # Verify assigned services exist
            if services:
                for gpu in hm.gpus:
                    for s_id in gpu.assigned_services:
                        if s_id not in services:
                            errors.append(f"Hardware '{hm.id}' GPU {gpu.index} references unknown service '{s_id}'")
        except Exception as e:
            errors.append(f"Hardware manifest error in {hf.name}: {e}")

    counts = {
        "services": len(services),
        "models": len(models),
        "capabilities": len(capabilities),
        "hardware": len(hw_files),
    }

    return (len(errors) == 0), errors, counts


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Factory v4 Manifest Tooling & Validation Suite")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    val_parser = subparsers.add_parser("validate", help="Validate all manifests")
    val_parser.add_argument("--path", "-p", default="manifests", help="Path to manifests directory")

    args = parser.parse_args()

    # Default to validate if no command provided
    cmd = args.command or "validate"
    path = getattr(args, "path", "manifests")

    if cmd == "validate":
        print(f"Validating manifests in: {Path(path).resolve()}")
        ok, errors, counts = validate_manifests(path)
        print("Manifest Inventory Summary:")
        for k, v in counts.items():
            print(f"  {k.capitalize()}: {v}")

        if not ok:
            print(f"\n❌ Validation Failures Detected ({len(errors)}):")
            for err in errors:
                print(f"  • {err}")
            sys.exit(1)
        else:
            print("\n✓ All manifests validated successfully with 0 errors.")
            sys.exit(0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
