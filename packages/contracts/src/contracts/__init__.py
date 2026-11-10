"""
contracts alias module redirecting to factory_contracts.
"""
import factory_contracts
from factory_contracts import *
from factory_contracts import errors, jobs, manifests, models, resilience, tasks, telemetry

__all__ = factory_contracts.__all__ + [
    "errors",
    "jobs",
    "manifests",
    "models",
    "resilience",
    "tasks",
    "telemetry",
]

