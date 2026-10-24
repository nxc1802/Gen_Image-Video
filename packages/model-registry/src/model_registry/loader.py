"""
packages/model-registry/src/model_registry/loader.py
Manifest Loader for Factory v4
"""
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from factory_contracts.manifests.service import ServiceManifest
from factory_contracts.manifests.model import ModelManifest
from factory_contracts.manifests.capability import CapabilityManifest
from factory_contracts.manifests.hardware import HardwareTopologyManifest


class ManifestLoader:
    """Loads and caches declarative manifests from the filesystem."""

    def __init__(self, manifests_dir: str | Path = "manifests"):
        self.manifests_dir = Path(manifests_dir)

    def load_services(self) -> Dict[str, ServiceManifest]:
        services: Dict[str, ServiceManifest] = {}
        svc_dir = self.manifests_dir / "services"
        if not svc_dir.exists():
            return services
        for p in sorted(list(svc_dir.glob("*.yaml")) + list(svc_dir.glob("*.yml"))):
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data:
                sm = ServiceManifest.model_validate(data)
                services[sm.metadata.id] = sm
        return services

    def load_models(self) -> Dict[str, ModelManifest]:
        models: Dict[str, ModelManifest] = {}
        mod_dir = self.manifests_dir / "models"
        if not mod_dir.exists():
            return models
        for p in sorted(list(mod_dir.glob("*.yaml")) + list(mod_dir.glob("*.yml"))):
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data:
                mm = ModelManifest.model_validate(data)
                models[mm.metadata.id] = mm
        return models

    def load_capabilities(self) -> Dict[str, CapabilityManifest]:
        caps: Dict[str, CapabilityManifest] = {}
        cap_dir = self.manifests_dir / "capabilities"
        if not cap_dir.exists():
            return caps
        for p in sorted(list(cap_dir.glob("*.yaml")) + list(cap_dir.glob("*.yml"))):
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data:
                cm = CapabilityManifest.model_validate(data)
                caps[cm.id] = cm
        return caps

    def load_hardware_topologies(self) -> Dict[str, HardwareTopologyManifest]:
        topologies: Dict[str, HardwareTopologyManifest] = {}
        hw_dir = self.manifests_dir / "hardware"
        if not hw_dir.exists():
            return topologies
        for p in sorted(list(hw_dir.glob("*.yaml")) + list(hw_dir.glob("*.yml"))):
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data:
                hm = HardwareTopologyManifest.model_validate(data)
                topologies[hm.id] = hm
        return topologies
