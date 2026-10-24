"""
tests/security/test_no_legacy_imports.py
Automated boundary isolation test ensuring zero legacy imports or leaked tunnel URLs in Factory v4.
"""
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

LEGACY_IMPORT_PATTERNS = [
    re.compile(r"^\s*(from|import)\s+legacy(\.|\s|$)", re.MULTILINE),
    re.compile(r"^\s*(from|import)\s+kaggle(\.|\s|$)", re.MULTILINE),
]

LEAK_PATTERNS = [
    re.compile(r"x-target-base-url", re.IGNORECASE),
    re.compile(r"trycloudflare\.com", re.IGNORECASE),
]

SCAN_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".mjs"}
EXCLUDED_GLOBAL = {".git", "legacy", "node_modules", ".next", ".agents", ".venv"}
PRODUCTION_DIRS = ["factory", "packages", "services", "workflows", "infrastructure"]


@pytest.mark.security
def test_zero_legacy_imports_across_workspace():
    """Ensures no file outside legacy/ imports from legacy or kaggle."""
    violations = []
    for path in ROOT_DIR.rglob("*"):
        if not path.is_file() or path.suffix not in SCAN_EXTENSIONS:
            continue
        if any(part in EXCLUDED_GLOBAL for part in path.parts):
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in LEGACY_IMPORT_PATTERNS:
            matches = pattern.findall(content)
            if matches:
                violations.append(f"{path.relative_to(ROOT_DIR)}: pattern '{pattern.pattern}'")

    assert not violations, "Forbidden legacy imports detected:\n" + "\n".join(violations)


@pytest.mark.security
def test_zero_proxy_or_fallback_in_production_code():
    """Ensures production codebase has 0 references to x-target-base-url or trycloudflare.com."""
    violations = []
    for prod_dir_name in PRODUCTION_DIRS:
        prod_dir = ROOT_DIR / prod_dir_name
        if not prod_dir.exists():
            continue
        for path in prod_dir.rglob("*"):
            if not path.is_file() or path.suffix not in SCAN_EXTENSIONS:
                continue
            if any(part in EXCLUDED_GLOBAL for part in path.parts):
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in LEAK_PATTERNS:
                matches = pattern.findall(content)
                if matches:
                    violations.append(f"{path.relative_to(ROOT_DIR)}: pattern '{pattern.pattern}'")

    assert not violations, "Forbidden proxy or tunnel fallback references in production code:\n" + "\n".join(violations)
