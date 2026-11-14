"""Diagnostic-only transport of an actual sampled video latent, never creates one."""

import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def verify_capture(path, expected):
    raw = Path(path).read_bytes()
    if len(raw) > 64 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError("Actual captured latent digest mismatch; no replacement")


class CaptureVideoLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "path": ("STRING",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "capture"
    CATEGORY = "factory/diagnostic"

    def capture(self, samples, path):
        video = samples["samples"]
        if video.is_nested:
            video = video.unbind()[0]
        if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[1] != 24:
            raise RuntimeError("Missing actual MiniMax sampled video latent")
        if not torch.isfinite(video).all():
            raise RuntimeError("Actual sampled video latent is non-finite")
        path = Path(path)
        if not path.is_absolute() or path.name != "actual_sampled_video.safetensors" or not path.parent.is_dir():
            raise RuntimeError("Explicit owned diagnostic output path required")
        if path.exists():
            raise RuntimeError("Diagnostic capture must not overwrite another sample")
        save_file({"video_samples": video.detach().cpu().contiguous()}, path)
        return (samples,)


class LoadCapturedVideoLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING",), "expected_sha256": ("STRING",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "factory/diagnostic"

    def load(self, path, expected_sha256):
        verify_capture(path, expected_sha256)
        data = load_file(path, device="cpu")
        if set(data) != {"video_samples"}:
            raise RuntimeError("Unexpected captured tensor graph")
        video = data["video_samples"]
        if video.ndim != 5 or video.shape[1] != 24 or not torch.isfinite(video).all():
            raise RuntimeError("Invalid actual captured video tensor")
        return ({"samples": video},)


NODE_CLASS_MAPPINGS = {
    "FactoryDiagnosticCaptureVideoLatent": CaptureVideoLatent,
    "FactoryDiagnosticLoadCapturedVideoLatent": LoadCapturedVideoLatent,
}
