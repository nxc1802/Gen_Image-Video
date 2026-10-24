"""
tests/security/test_proxy_elimination.py
Security tests verifying complete removal of arbitrary proxying and fallback domain leakage.
"""
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
STUDIO_WEB_DIR = ROOT_DIR / "factory" / "apps" / "studio-web"


@pytest.mark.security
class TestProxyElimination:
    def test_proxy_route_directory_deleted(self):
        proxy_dir = STUDIO_WEB_DIR / "src" / "app" / "api" / "proxy"
        assert not proxy_dir.exists(), f"Arbitrary proxy directory must be deleted: {proxy_dir}"

    def test_no_x_target_base_url_in_studio_web(self):
        violations = []
        for path in STUDIO_WEB_DIR.rglob("*"):
            if not path.is_file() or path.suffix not in {".ts", ".tsx", ".js", ".mjs", ".json"}:
                continue
            if any(part in {".next", "node_modules"} for part in path.parts):
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            if "x-target-base-url" in content.lower():
                violations.append(str(path.relative_to(ROOT_DIR)))
        assert not violations, f"x-target-base-url detected in files: {violations}"

    def test_no_trycloudflare_fallback_in_studio_web(self):
        violations = []
        for path in STUDIO_WEB_DIR.rglob("*"):
            if not path.is_file() or path.suffix not in {".ts", ".tsx", ".js", ".mjs", ".json"}:
                continue
            if any(part in {".next", "node_modules"} for part in path.parts):
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            if "trycloudflare.com" in content.lower():
                violations.append(str(path.relative_to(ROOT_DIR)))
        assert not violations, f"trycloudflare.com detected in studio-web files: {violations}"
