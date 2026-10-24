"""
packages/model-registry/src/model_registry/validator.py
Comprehensive Manifest Validation and Integrity Verifier
"""
from typing import Dict, List, Tuple
from factory_contracts.manifests.service import ServiceManifest
from factory_contracts.manifests.model import ModelManifest, ModelStatus
from factory_contracts.manifests.capability import CapabilityManifest
from factory_contracts.manifests.hardware import HardwareTopologyManifest
from model_registry.loader import ManifestLoader


class ManifestValidator:
    """Validates structural and semantic invariants across all manifests."""

    def __init__(self, loader: ManifestLoader):
        self.loader = loader

    def validate_all(self) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        services = self.loader.load_services()
        models = self.loader.load_models()
        capabilities = self.loader.load_capabilities()
        hardware = self.loader.load_hardware_topologies()

        # Check port and domain collisions
        ports: Dict[int, str] = {}
        domains: Dict[str, str] = {}
        for s_id, s in services.items():
            if s.network.internal_port in ports:
                errors.append(
                    f"Port collision: {s.network.internal_port} claimed by {ports[s.network.internal_port]} and {s_id}"
                )
            else:
                ports[s.network.internal_port] = s_id

            if s.network.static_domain in domains:
                errors.append(
                    f"Domain collision: {s.network.static_domain} claimed by {domains[s.network.static_domain]} and {s_id}"
                )
            else:
                domains[s.network.static_domain] = s_id

        # Check model referential integrity and completeness
        for m_id, m in models.items():
            if m.metadata.service_id not in services:
                errors.append(f"Model '{m_id}' references unknown service '{m.metadata.service_id}'")

            if m.status == ModelStatus.FULL:
                if m.coverage_percent < 100.0 or len(m.missing_capabilities) > 0:
                    errors.append(f"Model '{m_id}' marked FULL but completeness invariant violated")
            elif m.status == ModelStatus.PARTIAL:
                false_caps = [k for k, v in m.capabilities.items() if not v]
                if m.coverage_percent >= 100.0 and len(m.missing_capabilities) == 0 and not false_caps:
                    errors.append(f"Model '{m_id}' has 100% coverage and no missing capabilities; must be status: 'full'")

        # Check hardware service references
        for hw_id, hw in hardware.items():
            for gpu in hw.gpus:
                for assigned in gpu.assigned_services:
                    if assigned not in services:
                        errors.append(f"Hardware '{hw_id}' GPU {gpu.index} references unknown service '{assigned}'")

        return (len(errors) == 0), errors
