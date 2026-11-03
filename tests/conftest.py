# Root tests conftest
import os
import sys
from pathlib import Path
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factory.apps.control_api.src.main import app
from factory_auth.config import load_auth_config


@pytest.fixture(autouse=True)
def reset_auth_config():
    """Ensure hermetic auth configuration isolation across all tests."""
    auth_keys = ("FACTORY_API_KEY", "ENVIRONMENT", "FACTORY_ENV", "JWT_SECRET_KEY", "JWT_AUDIENCE", "JWT_ISSUER")
    orig_env = {k: os.environ[k] for k in auth_keys if k in os.environ}
    app.state.auth_config = load_auth_config()
    yield
    for k in auth_keys:
        if k in orig_env:
            os.environ[k] = orig_env[k]
        else:
            os.environ.pop(k, None)
    app.state.auth_config = load_auth_config()
