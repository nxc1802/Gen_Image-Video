"""Pinned CPU CLIP plus decoded-pixel oracles for actual native image bytes."""

import base64
import hashlib
import io
import json
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

from benchmarks.native.speech_semantic import pinned_download
from benchmarks.native.wave_a import digest

ROOT = Path(__file__).resolve().parents[2]


def rgb(raw):
    image = Image.open(io.BytesIO(raw)).convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    return Image.alpha_composite(background, image).convert("RGB")


def cosine(a, b):
    if (
        len(a) != 512
        or len(b) != 512
        or not all(type(x) in (float, int) and math.isfinite(x) for x in a + b)
    ):
        raise ValueError("Invalid actual CLIP embedding")
    norms = [math.sqrt(math.fsum(x * x for x in v)) for v in (a, b)]
    if any(abs(n - 1) > 0.001 for n in norms):
        raise ValueError("CLIP embeddings must be normalized")
    return math.fsum(x * y for x, y in zip(a, b, strict=True)) / (norms[0] * norms[1])


def pixel_metrics(raw, expected, assets):
    image = rgb(raw)
    rgba = Image.open(io.BytesIO(raw)).convert("RGBA")
    metrics = {
        "width": image.width,
        "height": image.height,
        "mean_alpha": ImageStat.Stat(rgba.getchannel("A")).mean[0] / 255,
        "pixel_stddev": sum(ImageStat.Stat(image).stddev) / 3 / 255,
    }
    if "color" in expected:
        region = image.crop(
            tuple(expected["color"].get("region", (0, 0, image.width, image.height)))
        )

        def matches(p):
            r, g, b = p
            return {
                "red": r > 100 and r > 1.35 * g and r > 1.35 * b,
                "blue": b > 70 and b > 1.2 * r and b > 1.1 * g,
                "green": g > 60 and g > 1.15 * r and g > 1.1 * b,
                "beige": r > 120 and g > 100 and b > 70 and r >= g >= b and max(p) - min(p) < 100,
            }[expected["color"]["name"]]

        metrics["color_fraction"] = sum(matches(p) for p in region.getdata()) / (
            region.width * region.height
        )
    if "maximum_saturation" in expected:
        metrics["mean_saturation"] = (
            ImageStat.Stat(image.convert("HSV").getchannel("S")).mean[0] / 255
        )
    if "preserve" in expected:
        spec = expected["preserve"]
        reference = rgb(assets[spec["asset"]])
        if reference.size != image.size:
            raise ValueError("Preservation oracle requires an exact input canvas")
        if "region" in spec:
            delta = ImageChops.difference(
                image.crop(tuple(spec["region"])), reference.crop(tuple(spec["region"]))
            )
            metrics["preserved_mean_absolute_error"] = sum(ImageStat.Stat(delta).mean) / 3 / 255
        else:
            mask = Image.open(io.BytesIO(assets[spec["mask"]])).convert("RGBA").getchannel("A")
            metrics["preserved_mean_absolute_error"] = (
                sum(ImageStat.Stat(ImageChops.difference(image, reference), mask).mean) / 3 / 255
            )
    return metrics


def history_checks(output, case):
    history = output["native_history"]
    messages = history["status"]["messages"]
    kinds = [m[0] for m in messages]
    cached = {n for kind, data in messages if kind == "execution_cached" for n in data["nodes"]}
    graph = output["graph"]
    submitted = history["prompt"][2]
    return {
        "explicit_lossless_prefix_cache_policy": graph.get("prefix_cache")
        == {
            "class_type": "QwenImage21Cache",
            "inputs": {"model": ["model", 0], "device": "off", "dtype": "default"},
        }
        and graph["sample"]["inputs"].get("model") == ["prefix_cache", 0],
        "actual_native_execution": history["status"]["completed"] is True
        and history["status"]["status_str"] == "success"
        and "execution_start" in kinds
        and "execution_success" in kinds,
        "sampling_not_cached": "sample" not in cached,
        "exact_submitted_graph": submitted == graph and history["prompt"][1] == output["prompt_id"],
        "declared_sampling_seed": graph["sample"]["inputs"]["seed"] == output["seed"]
        and output["seed"] in case["seed_sequence"],
    }


def semantic(case, output, *, assets=None):
    if "verification" not in output:
        return {
            "status": "NOT_VERIFIED",
            "checks": {},
            "reason": "Await pinned CPU semantic verification after owned GPU backend shutdown",
        }
    expected = case["expected"]
    verification = output["verification"]
    artifact = output["artifacts"][0]
    raw = base64.b64decode(artifact["data_base64"], validate=True)
    vector = verification["image_embedding"]
    similarities = [cosine(vector, v) for v in verification["text_embeddings"]]
    metrics = pixel_metrics(raw, expected, assets or {})
    metrics.update(
        positive_text_cosine=similarities[0],
        strongest_negative_cosine=max(similarities[1:]),
        text_margin=similarities[0] - max(similarities[1:]),
    )
    checks = history_checks(output, case)
    checks.update(
        exact_native_image_identity=len(output["artifacts"]) == 1
        and output["model"] == case["model"]
        and artifact["type"] == "image"
        and artifact["mime_type"] == "image/png"
        and len(raw) == artifact["size_bytes"]
        and hashlib.sha256(raw).hexdigest() == artifact["sha256"] == verification["input_sha256"],
        exact_verifier=verification["pins_sha256"] == expected["verifier_pins_sha256"],
        exact_text_oracle=verification["texts"]
        == [expected["positive_text"], *expected["negative_texts"]],
        requested_canvas=[metrics["width"], metrics["height"]]
        == [case["request"]["width"], case["request"]["height"]],
        visible_nonconstant_image=metrics["mean_alpha"] >= 0.95 and metrics["pixel_stddev"] >= 0.03,
        expected_visual_content=similarities[0] >= expected["minimum_text_cosine"]
        and metrics["text_margin"] >= expected["minimum_text_margin"],
    )
    if "color" in expected:
        checks["expected_color_in_region"] = (
            metrics["color_fraction"] >= expected["color"]["minimum_fraction"]
        )
    if "maximum_saturation" in expected:
        checks["requested_monochrome_transformation"] = (
            metrics["mean_saturation"] <= expected["maximum_saturation"]
        )
    if "preserve" in expected:
        checks["preserve_unedited_content"] = (
            metrics["preserved_mean_absolute_error"]
            <= expected["preserve"]["maximum_mean_absolute_error"]
        )
    for name, minimum in expected.get("reference_cosines", {}).items():
        reference = verification["reference_embeddings"][name]
        checks["exact_reference_" + name] = (
            verification["reference_sha256"][name] == case["asset_identities"][name]
        )
        metrics["reference_cosine_" + name] = cosine(vector, reference)
        checks["reference_content_" + name] = metrics["reference_cosine_" + name] >= minimum
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": metrics,
    }


class ImageVerifier:
    def __init__(self, cache, assets):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        torch.set_num_threads(2)
        pins = json.loads((ROOT / "benchmarks/native/image_semantic_pins.json").read_text())
        self.evidence = {"scope": pins["scope"], "pins_sha256": digest(pins), "verified_files": {}}
        for name, spec in pins["files"].items():
            url = f"https://huggingface.co/{pins['repository']}/resolve/{pins['revision']}/{name}"
            self.evidence["verified_files"][name] = pinned_download(url, spec, cache / name)
        self.processor = CLIPProcessor.from_pretrained(cache, local_files_only=True)
        self.model = (
            CLIPModel.from_pretrained(cache, local_files_only=True, use_safetensors=False)
            .float()
            .eval()
            .to("cpu")
        )
        params = list(self.model.parameters())
        loaded = {
            "parameter_count": sum(p.numel() for p in params),
            "devices": sorted({str(p.device) for p in params}),
            "dtypes": sorted({str(p.dtype) for p in params if p.is_floating_point()}),
        }
        if loaded["devices"] != ["cpu"] or loaded["dtypes"] != ["torch.float32"]:
            raise ValueError("Semantic image verifier must use explicitly selected CPU/F32")
        self.evidence["loaded"] = loaded
        self.assets = assets

    def evaluate(self, raw, expected):
        texts = [expected["positive_text"], *expected["negative_texts"]]
        names = list(expected.get("reference_cosines", {}))
        images = [rgb(raw)] + [rgb(self.assets[n]) for n in names]
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        with self.torch.inference_mode():
            result = self.model(**inputs)
        return {
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "pins_sha256": self.evidence["pins_sha256"],
            "texts": texts,
            "image_embedding": result.image_embeds[0].cpu().tolist(),
            "text_embeddings": result.text_embeds.cpu().tolist(),
            "reference_embeddings": {
                n: result.image_embeds[i + 1].cpu().tolist() for i, n in enumerate(names)
            },
            "reference_sha256": {n: hashlib.sha256(self.assets[n]).hexdigest() for n in names},
        }
