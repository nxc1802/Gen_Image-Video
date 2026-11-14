"""Migration boundary: retained standalone proxies cannot authorize production."""
import os
from factory_contracts.errors import ModelNotLoadedError


def forbid_legacy_production():
    if os.environ.get("ENVIRONMENT",os.environ.get("FACTORY_ENV","development")).lower()=="production":
        raise ModelNotLoadedError("Retained legacy entrypoint cannot authorize production inference. Use the canonical deployment Engine.")
