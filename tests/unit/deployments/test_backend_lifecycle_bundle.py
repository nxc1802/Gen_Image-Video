"""Deployment packaging regression, without Kaggle credentials or GPU use."""
import io
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from deployments.kaggle.backend_lifecycle_probe import source_bundle


def test_lifecycle_bundle_is_deterministic_and_imports_in_isolation(tmp_path):
    bundle = source_bundle()
    assert bundle == source_bundle()
    archive = zipfile.ZipFile(io.BytesIO(bundle))
    assert all(Path(name).suffix in {".py",".toml",".md",".yaml",".jinja"} or Path(name).name == "LICENSE.vllm" for name in archive.namelist())
    assert not any({"tests","__pycache__",".venv"}.intersection(Path(name).parts) for name in archive.namelist())
    archive.extractall(tmp_path)
    (tmp_path / "factory/apps/control_api").symlink_to("control-api",target_is_directory=True)
    paths = [str(tmp_path)] + [str(tmp_path / "packages" / name / "src") for name in ["contracts","model-registry","auth"]]
    env = {**os.environ,"PYTHONPATH":os.pathsep.join(paths)}
    code = "from pathlib import Path; from factory.apps.control_api.src import main; from services.text.src import models; assert Path(main.__file__).is_relative_to(Path.cwd()); assert Path(models.__file__).is_relative_to(Path.cwd())"
    result = subprocess.run([sys.executable,"-c",code],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode == 0, result.stderr
