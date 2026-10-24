"""
Hardware topology contracts.

assigned_services are services eligible to run on a GPU, not services assumed
to be resident at the same time. Factory defaults to exclusive admission on
memory-constrained worker profiles such as Kaggle T4.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class GPUDeviceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., ge=0, le=7)
    name: str
    total_vram_gb: float = Field(..., ge=1.0)
    max_allocatable_vram_gb: float = Field(..., ge=1.0)
    assigned_services: List[str] = Field(
        default_factory=list,
        description="Services eligible for this GPU under the declared admission policy.",
    )
    admission_mode: Literal["exclusive", "shared"] = Field(
        default="exclusive",
        description="exclusive = one resident GPU service at a time; shared = concurrent residency permitted.",
    )
    max_concurrent_services: int = Field(default=1, ge=1, le=16)


class HostResourcesSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cpu_cores: int = Field(..., ge=1)
    system_ram_gb: float = Field(..., ge=2.0)
    storage_temp_gb: float = Field(..., ge=5.0)


class HardwareTopologyManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    apiVersion: str = Field(default="factory.ai/v1", pattern=r"^factory\.ai/v\d+$")
    kind: str = Field(default="HardwareTopologyManifest")
    id: str = Field(..., pattern=r"^[a-z0-9-]+$")
    name: str
    environment: str
    host: HostResourcesSpec
    gpus: List[GPUDeviceSpec]
