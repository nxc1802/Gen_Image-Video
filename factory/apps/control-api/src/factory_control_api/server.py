"""
factory_control_api.server: Module exporting FastAPI application instance for Uvicorn entrypoint compatibility.
"""
from . import app

__all__ = ["app"]
