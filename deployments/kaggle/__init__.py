"""
deployments.kaggle package for Kaggle Dual-T4 deployment and live model certification.
"""
from deployments.kaggle.runner import (
    KaggleAPIClient,
    KaggleDualT4Runner,
    KaggleAPIError,
    KaggleAuthError,
    KaggleTimeoutError,
    KaggleKernelFailedError,
)
from deployments.kaggle.certify_kernel import (
    DualT4HardwareTopology,
    CertificationGateResult,
    ModelCertificationProfile,
    Tier1ModelCertifier,
)
from deployments.kaggle.smoke_runner import (
    ModelSmokeRunner,
    SmokeReport,
    StageResult,
)
from deployments.kaggle.bootstrap import (
    KaggleBootstrap,
    BootstrapResult,
    HardwareReport,
)

__all__ = [
    "KaggleAPIClient",
    "KaggleDualT4Runner",
    "KaggleAPIError",
    "KaggleAuthError",
    "KaggleTimeoutError",
    "KaggleKernelFailedError",
    "DualT4HardwareTopology",
    "CertificationGateResult",
    "ModelCertificationProfile",
    "Tier1ModelCertifier",
    "ModelSmokeRunner",
    "SmokeReport",
    "StageResult",
    "KaggleBootstrap",
    "BootstrapResult",
    "HardwareReport",
]

