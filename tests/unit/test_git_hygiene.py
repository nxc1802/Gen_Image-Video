"""Unit tests for Feature 31 (P2.5): Artifact Git Hygiene.

Verifies:
1. Zero tracked media binaries, weights, or database files in Git.
2. Comprehensive .gitignore rule coverage across root and submodules.
3. Regression test fixture whitelisting preservation while ephemeral directories are ignored.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_zero_media_binaries_and_weights_tracked() -> None:
    """Assert that no media, models, or binary databases are tracked in Git."""
    res = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_files = [line.strip() for line in res.stdout.splitlines() if line.strip()]

    prohibited_extensions = {
        # Video
        ".mp4", ".webm", ".mkv", ".mov", ".avi", ".flv", ".wmv",
        # Audio
        ".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a",
        # Images (excluding SVG source files)
        ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".ico",
        # Models & Weights
        ".safetensors", ".gguf", ".pt", ".pth", ".bin", ".onnx", ".ckpt", ".h5", ".tflite", ".engine",
        # Databases
        ".sqlite", ".sqlite3", ".db",
    }

    violations = []
    for f in tracked_files:
        ext = Path(f).suffix.lower()
        if ext in prohibited_extensions:
            violations.append(f)

    assert not violations, f"Forbidden binary files tracked in Git: {violations}"


def test_gitignore_rule_coverage() -> None:
    """Assert that root .gitignore covers all mandatory categories."""
    gitignore_path = REPO_ROOT / ".gitignore"
    assert gitignore_path.exists(), "Root .gitignore must exist"
    content = gitignore_path.read_text(encoding="utf-8")

    mandatory_rules = [
        "*.mp4", "*.wav", "*.png", "*.safetensors", "*.gguf",
        "*.sqlite", "*.db", ".factory_storage/", "artifacts/",
        "storage/", "deployments/kaggle/*_outputs/",
        "!tests/**/fixtures/**", "!tests/fixtures/**",
    ]

    missing = [rule for rule in mandatory_rules if rule not in content]
    assert not missing, f"Missing critical .gitignore rules: {missing}"


def test_submodule_gitignores_exist() -> None:
    """Assert submodule .gitignore files exist for studio-web and kaggle."""
    studio_gitignore = REPO_ROOT / "factory" / "apps" / "studio-web" / ".gitignore"
    kaggle_gitignore = REPO_ROOT / "deployments" / "kaggle" / ".gitignore"

    assert studio_gitignore.exists(), "factory/apps/studio-web/.gitignore must exist"
    assert kaggle_gitignore.exists(), "deployments/kaggle/.gitignore must exist"


def test_test_fixtures_unignored_preservation() -> None:
    """Assert that test fixtures are unignored by .gitignore while ephemeral outputs are ignored."""
    fixtures_sample = REPO_ROOT / "tests" / "fixtures" / "mock_sample.png"
    ephemeral_sample = REPO_ROOT / "storage" / "job_out.png"

    # Evaluate git check-ignore
    res_fixture = subprocess.run(
        ["git", "check-ignore", "-q", str(fixtures_sample)],
        cwd=REPO_ROOT,
    )
    # Return code 1 means NOT ignored (correct for fixtures)
    assert res_fixture.returncode == 1, "Test fixtures in tests/fixtures/ must NOT be ignored!"

    res_ephemeral = subprocess.run(
        ["git", "check-ignore", "-q", str(ephemeral_sample)],
        cwd=REPO_ROOT,
    )
    # Return code 0 means IGNORED (correct for storage/)
    assert res_ephemeral.returncode == 0, "Artifacts in storage/ must be ignored by .gitignore!"
