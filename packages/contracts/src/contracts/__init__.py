"""
contracts alias module redirecting to factory_contracts.
"""
from factory_contracts import *
from factory_contracts import errors, manifests, jobs

__all__ = factory_contracts.__all__ + ["errors", "manifests", "jobs"]
