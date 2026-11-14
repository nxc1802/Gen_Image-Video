"""Pinned meta-device schema enumeration only; no weights, GPU, or inference."""

import hashlib
import inspect
import json
import urllib.request
from importlib.metadata import version
from pathlib import Path

import torch
import yaml
from diffusers import WanTransformer3DModel
from transformers import UMT5Config, UMT5EncoderModel

repo = Path(__file__).resolve().parents[2]
manifest = yaml.safe_load((repo / "manifests/models/wan-2.2.yaml").read_text())
specs = {v["path"]: v for v in manifest["weights"]["files"].values()}
configs = {}
verified = {}
for file in [
    "transformer/config.json",
    "text_encoder/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "text_encoder/model.safetensors.index.json",
]:
    spec = specs[file]
    raw = urllib.request.urlopen(
        f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/{file}"
    ).read()
    assert len(raw) == spec["size_bytes"] and hashlib.sha256(raw).hexdigest() == spec["sha256"]
    configs[file] = json.loads(raw)
    verified[file] = {"sha256": spec["sha256"], "size_bytes": len(raw), "matched": True}
with torch.device("meta"):
    transformer = WanTransformer3DModel.from_config(configs["transformer/config.json"])
    text_encoder = UMT5EncoderModel(UMT5Config.from_dict(configs["text_encoder/config.json"]))
report = {
    "classification": "META_CONFIGURATION_AND_PARAMETER_SCHEMA_ONLY_NOT_INFERENCE",
    "weights_loaded": False,
    "gpu_inference_executed": False,
    "production_ready": False,
    "runtime_versions": {n: version(n) for n in ["torch", "transformers", "diffusers"]},
    "verified_config_and_indexes": verified,
    "components": {},
}
for name, model, index in [
    ("transformer", transformer, "transformer/diffusion_pytorch_model.safetensors.index.json"),
    ("text_encoder", text_encoder, "text_encoder/model.safetensors.index.json"),
]:
    params = dict(model.named_parameters())
    protected = sorted(model._keep_in_fp32_modules)
    overrides = {
        key: "fp32" for key in params if any(label in key.split(".") for label in protected)
    }
    assert overrides and set(overrides) <= set(configs[index]["weight_map"])
    assert all(p.device.type == "meta" for p in params.values())
    source = Path(inspect.getfile(type(model))).read_bytes()
    report["components"][name] = {
        "parameter_count": sum(p.numel() for p in params.values()),
        "native_keep_in_fp32_modules": protected,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "parameter_precision_overrides": overrides,
        "all_parameters_meta": True,
    }
print(json.dumps(report, indent=2))
