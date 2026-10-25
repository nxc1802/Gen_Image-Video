"""
factory_control_api: Factory v4 Control Plane API package entrypoint.
"""
from typing import Any

__version__ = "0.1.0"

try:
    from factory.apps.control_api.src.main import app as _app
except ImportError:
    from ..main import app as _app


def get_application() -> Any:
    """Returns the FastAPI application instance."""
    return _app


app = get_application()

__all__ = ["app", "get_application", "__version__"]
