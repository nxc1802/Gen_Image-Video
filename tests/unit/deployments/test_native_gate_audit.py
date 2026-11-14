"""Software integrity checks only; never native model evidence."""

import hashlib
import io
import zipfile

import pytest

from benchmarks.native.gate_audit_kernel import unpack_native
from deployments.kaggle.native_gate_audit import prepare


def archive_of(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in entries:
            archive.writestr(name, raw)
    return zipfile.ZipFile(io.BytesIO(stream.getvalue()))


def test_unpack_requires_every_original_receipt_byte_before_writing(tmp_path):
    marker = b"explicit software archive integrity control, never inference evidence"
    spec = {"folder": "validation/native_control_v1", "artifacts": [
        {"file": "validation/native_control_v1/setup.json", "sha256": hashlib.sha256(marker).hexdigest()},
        {"file": "validation/native_control_v1/output.json", "sha256": hashlib.sha256(marker).hexdigest()},
    ]}
    with archive_of([("setup.json", marker), ("output.json", b"changed bytes")]) as archive:
        with pytest.raises(ValueError, match="digest mismatch"):
            unpack_native(archive, spec, tmp_path)
    assert not list(tmp_path.rglob("*.json"))
    with archive_of([("setup.json", marker), ("output.json", marker)]) as archive:
        unpack_native(archive, spec, tmp_path)
    assert (tmp_path / spec["folder"] / "output.json").read_bytes() == marker


@pytest.mark.parametrize("filename", ["../escape.json", "/absolute.json"])
def test_archive_path_escape_is_rejected_even_with_a_matching_digest(tmp_path, filename):
    raw = b"explicit software path control, never model output"
    spec = {"folder": "validation/native_control_v1", "artifacts": [
        {"file": "validation/native_control_v1/" + filename, "sha256": hashlib.sha256(raw).hexdigest()},
    ]}
    with archive_of([(filename, raw)]) as archive:
        with pytest.raises(ValueError, match="path"):
            unpack_native(archive, spec, tmp_path)


def test_cpu_gate_audit_fits_real_submission_bound_and_keeps_original_gate():
    code, inputs, sources = prepare()
    compile(code, "cpu_gate_audit", "exec")
    assert len(code.encode()) < 1_000_000
    assert inputs["native_attempts"] and len(sources) == len(inputs["native_attempts"])
    assert all(set(a) == {"kind", "version", "folder", "artifacts"} for a in inputs["native_attempts"])
    assert "enable_gpu" not in code and "torch" not in code
