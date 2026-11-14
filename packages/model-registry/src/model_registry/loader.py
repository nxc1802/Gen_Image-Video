"""
packages/model-registry/src/model_registry/loader.py
Manifest Loader for Factory v4
"""
from pathlib import Path
import os
from typing import Dict, List, Optional
import yaml

from factory_contracts.manifests.service import ServiceManifest
from factory_contracts.manifests.model import ModelManifest
from factory_contracts.manifests.capability import CapabilityManifest
from factory_contracts.manifests.hardware import HardwareTopologyManifest
from factory_contracts.manifests.catalog import manifest_root, load_model_manifests


class ManifestLoader:
    """Loads and caches declarative manifests from the filesystem."""

    def __init__(self, manifests_dir: str | Path | None = None):
        self.manifests_dir = manifest_root(manifests_dir)

    def load_services(self) -> Dict[str, ServiceManifest]:
        services: Dict[str, ServiceManifest] = {}
        svc_dir = self.manifests_dir / "services"
        if not svc_dir.exists():
            return services
        for p in sorted(list(svc_dir.glob("*.yaml")) + list(svc_dir.glob("*.yml"))):
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(os.path.expandvars(f.read()))
            if data:
                sm = ServiceManifest.model_validate(data)
                services[sm.metadata.id] = sm
        return services

    def load_models(self) -> Dict[str, ModelManifest]:
        return load_model_manifests(self.manifests_dir)

    def load_capabilities(self) -> Dict[str, CapabilityManifest]:
        caps: Dict[str, CapabilityManifest] = {}
        cap_dir = self.manifests_dir / "capabilities"
        if not cap_dir.exists():
            return caps
        for p in sorted(list(cap_dir.glob("*.yaml")) + list(cap_dir.glob("*.yml"))):
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(os.path.expandvars(f.read()))
            if data:
                cm = CapabilityManifest.model_validate(data)
                if cm.id in caps:
                    raise ValueError(f"Duplicate capability parameter specification: {cm.id}")
                caps[cm.id] = cm
        return caps

    def load_hardware_topologies(self) -> Dict[str, HardwareTopologyManifest]:
        topologies: Dict[str, HardwareTopologyManifest] = {}
        hw_dir = self.manifests_dir / "hardware"
        if not hw_dir.exists():
            return topologies
        for p in sorted(list(hw_dir.glob("*.yaml")) + list(hw_dir.glob("*.yml"))):
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(os.path.expandvars(f.read()))
            if data:
                hm = HardwareTopologyManifest.model_validate(data)
                topologies[hm.id] = hm
        return topologies
