"""File integrity checks only. These fixtures never count as model inference."""
import hashlib

import pytest

from deployments.kaggle.weight_identity_probe import inspect_weights


def spec(**updates):
    return {"model": "test-model", "role": "weights", "dataset": "owner/test-dataset",
            "filename": "weights.bin", "expected_sha256": hashlib.sha256(b"weights").hexdigest(),
            "expected_size_bytes": 7, **updates}


def test_weight_identity_requires_exact_digest_not_matching_size(tmp_path):
    dataset = tmp_path / "test-dataset"
    dataset.mkdir()
    (dataset / "weights.bin").write_bytes(b"garbage")
    result = inspect_weights([spec()], tmp_path, tmp_path / "report.json")["results"][0]
    assert result["status"] == "BLOCKED"
    assert not result["weights_verified"]


def test_weight_identity_does_not_choose_first_duplicate(tmp_path):
    for directory in (tmp_path / "test-dataset/a", tmp_path / "test-dataset/b"):
        directory.mkdir(parents=True)
        (directory / "weights.bin").write_bytes(b"weights")
    result = inspect_weights([spec()], tmp_path, tmp_path / "report.json")["results"][0]
    assert result["status"] == "BLOCKED"
    assert "found 2" in result["reason"]


def test_observed_hash_without_expected_digest_is_not_verified(tmp_path):
    dataset = tmp_path / "test-dataset"
    dataset.mkdir()
    (dataset / "weights.bin").write_bytes(b"weights")
    report = inspect_weights([spec(expected_sha256=None)], tmp_path, tmp_path / "report.json")
    assert report["results"][0]["status"] == "OBSERVED_UNBOUND"
    assert not report["results"][0]["weights_verified"]
    assert not report["model_inference_executed"]


@pytest.mark.parametrize("specs", [[], [spec(filename="*.bin")]])
def test_empty_or_wildcard_spec_fails_closed(tmp_path, specs):
    with pytest.raises(ValueError):
        inspect_weights(specs, tmp_path, tmp_path / "report.json")
