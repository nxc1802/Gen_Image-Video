"""Explicit session configuration. Never a runtime or inference certificate."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class BackendOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    executable: Path | None = None
    source_dir: Path | None = None
    source_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    startup_timeout_seconds: float = Field(default=600, gt=0, le=3600)
    stop_timeout_seconds: float = Field(default=10, gt=0, le=30)


class DeploymentEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str = Field(min_length=1)
    devices: tuple[StrictInt, ...] = Field(min_length=1)
    backend: str | None = None
    port: int | None = Field(default=None, ge=1024, le=65535)
    exclusive: bool = False
    weights_dir: Path | None = None
    backend_options: BackendOptions = Field(default_factory=BackendOptions)

    @model_validator(mode="after")
    def explicit_devices(self):
        if len(set(self.devices)) != len(self.devices) or any(d < 0 for d in self.devices):
            raise ValueError("Deployment devices must be distinct nonnegative GPU IDs")
        if len(self.devices) > 1 and not self.exclusive:
            raise ValueError("Multi-GPU deployments require exclusive: true")
        return self


class DeploymentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    apiVersion: Literal["factory.ai/v1"] = "factory.ai/v1"
    kind: Literal["DeploymentConfig"] = "DeploymentConfig"
    hardware: Literal["kaggle-dual-t4"] = "kaggle-dual-t4"
    models_dir: Path = Path("models")
    runtime_dir: Path = Path(".factory_storage/runtime")
    api_host: Literal["127.0.0.1", "0.0.0.0", "::1"] = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1024, le=65535)
    deployments: tuple[DeploymentEntry, ...] = ()

    @model_validator(mode="after")
    def unique_models(self):
        names = [entry.model for entry in self.deployments]
        if len(names) != len(set(names)):
            raise ValueError("A model may have only one explicit deployment in a session")
        return self
