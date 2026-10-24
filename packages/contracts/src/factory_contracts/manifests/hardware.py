"""
factory_contracts/manifests/hardware.py
Hardware Topology Manifest Schema
"""
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class GPUDeviceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., ge=0, le=7)
    name: str = Field(..., description="GPU model name, e.g. 'Tesla T4'")
    total_vram_gb: float = Field(..., ge=1.0)
    max_allocatable_vram_gb: float = Field(..., ge=1.0, description="Usable VRAM after CUDA runtime reservation")
    assigned_services: List[str] = Field(default_factory=list, description="IDs of services bound to this GPU")


class HostResourcesSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cpu_cores: int = Field(..., ge=1)
    system_ram_gb: float = Field(..., ge=2.0)
    storage_temp_gb: float = Field(..., ge=5.0)


class HardwareTopologyManifest(BaseModel):
    """Declarative specification of host hardware profile (Kaggle Dual-T4, Colab, Local)."""
    model_config = ConfigDict(extra="forbid")

    apiVersion: str = Field(default="factory.ai/v1", pattern=r"^factory\.ai/v\d+$")
    kind: str = Field(default="HardwareTopologyManifest")
    id: str = Field(..., pattern=r"^[a-z0-9-]+$", description="Profile ID, e.g. 'kaggle-dual-t4'")
    name: str
    environment: str = Field(..., description="kaggle | colab | local | cloud")
    host: HostResourcesSpec
    gpus: List[GPUDeviceSpec]
