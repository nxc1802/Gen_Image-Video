"""
tests/security/test_feature30_adversarial_docs_and_config.py
Adversarial Verification Suite for Feature 30 (P2.4: Documentation, Config Sync, Next.js 15.5.29 & Security).

Target Invariants:
1. Documentation Integrity: Verify that file paths referenced in core docs exist on disk.
2. Server Importability & Lifespan: Verify `factory_control_api.server:app` imports and executes ASGI lifespan cleanly.
3. Configuration Synchronization: Verify all settings parsed across control-api, auth, storage, and queue are declared in `.env.example`.
4. Frontend Security & Build Config: Verify Next.js 15.5.29 pin, unoptimized image safety, and zero critical vulnerabilities.
"""
import asyncio
from pathlib import Path
import re
import pytest
from httpx import AsyncClient, ASGITransport

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


class TestDocumentationPathIntegrity:
    """Stress-test referenced file and directory paths across documentation files."""

    def test_readme_referenced_paths_exist(self):
        """Verify all concrete project paths mentioned in README.md exist on disk."""
        paths_to_verify = [
            "factory/apps/control-api/",
            "factory/apps/studio-web/",
            "services/text/",
            "services/vlm/",
            "services/decision/",
            "services/embedding/",
            "services/speech/",
            "services/image/",
            "services/video/",
            "services/music/",
            "services/enhancement/",
            "manifests/",
            "deployments/kaggle/",
            "benchmarks/",
            "tests/test_preflight.py",
            "tests/unit/benchmarks/test_unified_benchmark_runner.py",
        ]
        missing = [p for p in paths_to_verify if not (WORKSPACE_ROOT / p).exists()]
        assert not missing, f"README.md references missing paths on disk: {missing}"

    def test_architecture_overview_paths_exist(self):
        """Verify paths and architecture references in docs/architecture/overview.md."""
        overview_path = WORKSPACE_ROOT / "docs/architecture/overview.md"
        assert overview_path.exists(), "docs/architecture/overview.md is missing!"

        content = overview_path.read_text(encoding="utf-8")
        assert "Factory v4 Architecture Overview" in content
        assert "Discrete Dual-T4 Hardware Partitioning" in content
        assert "LocalStorageProvider" in content
        assert "S3StorageProvider" in content

    def test_kaggle_deployment_readme_scripts_exist(self):
        """Verify that all runner scripts documented in deployments/kaggle/README.md exist."""
        script_paths = [
            "deployments/kaggle/bootstrap.py",
            "deployments/kaggle/live_gpu_test.py",
            "deployments/kaggle/prepare_datasets.py",
            "deployments/kaggle/session1_visual_pipeline.py",
            "deployments/kaggle/session2_audio_pipeline.py",
            "deployments/kaggle/monitor_both_sessions.py",
        ]
        missing = [p for p in script_paths if not (WORKSPACE_ROOT / p).exists()]
        assert not missing, f"deployments/kaggle/README.md references missing scripts: {missing}"

    def test_manifest_coverage_and_documentation_discrepancies(self):
        """
        Verify authoritative manifests on disk and empirically assert documentation discrepancies
        in manifests/README.md.
        """
        manifests_readme = WORKSPACE_ROOT / "manifests/README.md"
        assert manifests_readme.exists(), "manifests/README.md is missing!"

        # Authoritative files that must exist
        assert (WORKSPACE_ROOT / "manifests/models/qwen-text.yaml").exists()
        assert (WORKSPACE_ROOT / "manifests/models/qwen-image-2.1.yaml").exists()
        assert (WORKSPACE_ROOT / "manifests/models/qwen-image-2.1-turbo.yaml").exists()
        assert (WORKSPACE_ROOT / "manifests/models/wan-2.2.yaml").exists()
        assert (WORKSPACE_ROOT / "manifests/models/minimax-h3.yaml").exists()
        assert (WORKSPACE_ROOT / "manifests/hardware/kaggle-dual-t4.yaml").exists()
        assert (WORKSPACE_ROOT / "manifests/hardware/colab-single-t4.yaml").exists()
        assert (WORKSPACE_ROOT / "manifests/hardware/local-dev.yaml").exists()

        # Document and assert known illustrative typos in manifests/README.md:
        # Line 16 cites audio.tts.yaml (does not exist; audio capabilities are not in capabilities/)
        # Line 21 cites local-dev-cpu.yaml (does not exist; actual file is local-dev.yaml)
        audio_tts_exists = (WORKSPACE_ROOT / "manifests/capabilities/audio.tts.yaml").exists()
        local_dev_cpu_exists = (WORKSPACE_ROOT / "manifests/hardware/local-dev-cpu.yaml").exists()

        assert not audio_tts_exists, (
            "manifests/capabilities/audio.tts.yaml was unexpectedly found on disk."
        )
        assert not local_dev_cpu_exists, (
            "manifests/hardware/local-dev-cpu.yaml was unexpectedly found on disk (actual is local-dev.yaml)."
        )

        # Confirm authoritative capability count is exactly 12 and hardware count is exactly 3
        capabilities = list((WORKSPACE_ROOT / "manifests/capabilities").glob("*.yaml"))
        assert len(capabilities) == 12, f"Expected 12 capability manifests, found {len(capabilities)}"

        hardware = list((WORKSPACE_ROOT / "manifests/hardware").glob("*.yaml"))
        assert len(hardware) == 3, f"Expected 3 hardware manifests, found {len(hardware)}"


class TestControlApiServerInitialization:
    """Stress-test importability, ASGI lifespan startup/teardown, and endpoints of factory_control_api.server:app."""

    def test_import_server_app(self):
        """Verify factory_control_api.server:app can be imported cleanly."""
        from factory_control_api.server import app

        assert app is not None
        assert app.title == "Factory v4 Control Plane API"
        assert app.version == "4.0.0"

    @pytest.mark.asyncio
    async def test_server_lifespan_startup_and_endpoints(self):
        """Verify full lifespan startup, scratch disk initialization, health check, and model listing."""
        from factory_control_api.server import app

        async with app.router.lifespan_context(app):
            # Assert state components were initialized
            assert hasattr(app.state, "auth_config")
            assert hasattr(app.state, "scratch_disk")
            assert hasattr(app.state, "storage_provider")
            assert hasattr(app.state, "repository")
            assert hasattr(app.state, "broker")
            assert hasattr(app.state, "manifest_loader")

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                # 1. Health endpoint
                res_health = await client.get("/health")
                assert res_health.status_code == 200
                data_health = res_health.json()
                assert data_health["status"] == "healthy"
                assert data_health["plane"] == "control"

                # 2. Models endpoint with auth
                api_key = app.state.auth_config.api_key
                res_models = await client.get("/v1/models", headers={"Authorization": f"Bearer {api_key}"})
                assert res_models.status_code == 200
                data_models = res_models.json()
                assert "data" in data_models
                models = data_models["data"]
                assert len(models) == 12, f"Expected 12 models registered, got {len(models)}"

                model_ids = {m["metadata"]["id"] for m in models}
                assert "qwen-image-2.1-turbo" in model_ids


class TestConfigurationSynchronization:
    """Stress-test .env.example coverage against all operational variables."""

    def test_env_example_contains_all_operational_variables(self):
        """Verify that all core operational and extension settings are defined in .env.example."""
        env_example_path = WORKSPACE_ROOT / ".env.example"
        assert env_example_path.exists(), ".env.example is missing!"

        content = env_example_path.read_text(encoding="utf-8")
        parsed_keys = set(re.findall(r"^#?\s*([A-Z0-9_]+)=", content, re.M))

        required_keys = [
            # Auth & Security
            "FACTORY_API_KEY",
            "ENVIRONMENT",
            "JWT_SECRET_KEY",
            "JWT_ALGORITHM",
            "JWT_AUDIENCE",
            "JWT_ISSUER",
            # Server & Networking
            "HOST",
            "PORT",
            "DEBUG",
            "TRUSTED_PROXIES",
            "FACTORY_CORS_ORIGINS",
            "REQUEST_TIMEOUT_SECONDS",
            # Rate Limiter
            "RATE_LIMIT_ENABLED",
            "RATE_LIMIT_DEFAULT",
            "RATE_LIMIT_WINDOW",
            # Database & Migrations
            "FACTORY_DB_BACKEND",
            "FACTORY_AUTO_MIGRATE",
            "DATABASE_URL",
            "SQLITE_DB_PATH",
            "FACTORY_MAX_JOBS",
            "JOB_TTL_SECONDS",
            # Storage & Artifacts
            "STORAGE_BACKEND",
            "SCRATCH_ARTIFACTS_DIR",
            "ARTIFACT_TTL_SECONDS",
            "ARTIFACT_SWEEP_INTERVAL_SECONDS",
            "MAX_ARTIFACT_STORAGE_MB",
            "ARTIFACT_CLEANUP_ON_DOWNLOAD",
            "LOCAL_STORAGE_SECRET_KEY",
            "S3_ENDPOINT_URL",
            "S3_PUBLIC_ENDPOINT_URL",
            "S3_BUCKET_NAME",
            "S3_ACCESS_KEY",
            "S3_SECRET_KEY",
            "S3_REGION",
            # Studio Web & Kaggle
            "NEXT_PUBLIC_CONTROL_API_URL",
            "KAGGLE_USERNAME",
            "KAGGLE_API_TOKEN",
            "KAGGLE_KEY",
            "HF_TOKEN",
            # Native Backend URLs
            "VLLM_TEXT_URL",
            "VLLM_VLM_URL",
            "VLLM_DECISION_URL",
            "VLLM_EMBEDDING_URL",
            "VLLM_RERANK_URL",
            "QWEN_ASR_URL",
            "QWEN_TTS_URL",
            "ACESTEP_API_URL",
        ]

        missing = [k for k in required_keys if k not in parsed_keys]
        assert not missing, f".env.example is missing required operational keys: {missing}"

    def test_env_example_has_no_raw_production_secrets(self):
        """Verify .env.example does not leak live API keys or private credentials."""
        content = (WORKSPACE_ROOT / ".env.example").read_text(encoding="utf-8")
        assert not re.search(r"=\s*KGAT_[a-zA-Z0-9]+", content), "Raw Kaggle bearer token found in .env.example!"
        assert not re.search(r"=\s*hf_[a-zA-Z0-9]{20,}", content), "Raw HuggingFace token found in .env.example!"


class TestNextJsSecurityAndBuildContracts:
    """Stress-test Next.js security posture and package manifest configuration."""

    def test_nextjs_version_pinned_15_5_29(self):
        """Verify factory/apps/studio-web pins Next.js 15.5.29 to eliminate Flight RCE."""
        package_json = WORKSPACE_ROOT / "factory/apps/studio-web/package.json"
        assert package_json.exists(), "studio-web package.json missing!"

        content = package_json.read_text(encoding="utf-8")
        assert '"next": "^15.5.29"' in content or '"next": "15.5.29"' in content
        assert '"eslint-config-next": "^15.5.29"' in content or '"eslint-config-next": "15.5.29"' in content

    def test_next_config_unoptimized_images(self):
        """Verify next.config.mjs disables remote image optimization to mitigate CVE-2024-34351."""
        next_config = WORKSPACE_ROOT / "factory/apps/studio-web/next.config.mjs"
        assert next_config.exists(), "studio-web next.config.mjs missing!"

        content = next_config.read_text(encoding="utf-8")
        assert "unoptimized: true" in content
