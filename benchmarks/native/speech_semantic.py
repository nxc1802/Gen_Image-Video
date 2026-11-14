"""Real CPU semantic verifiers for generated speech; never production fallback."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import urllib.request

from benchmarks.native.wave_a import digest

ROOT = Path(__file__).resolve().parents[2]


def pinned_download(url, spec, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        with urllib.request.urlopen(url, timeout=120) as source, path.open("wb") as target:
            received = 0
            while chunk := source.read(8 * 1024 * 1024):
                received += len(chunk)
                if received > spec["size_bytes"]:
                    raise ValueError("Oversized semantic-verifier checkpoint")
                target.write(chunk)
    sha = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            sha.update(chunk)
    if path.stat().st_size != spec["size_bytes"] or sha.hexdigest() != spec["sha256"]:
        raise ValueError("Semantic-verifier checkpoint integrity failed")
    return {"sha256": sha.hexdigest(), "size_bytes": path.stat().st_size, "matched": True}


def decode_audio(raw):
    import numpy as np

    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            "pipe:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "pipe:1",
        ],
        input=raw,
        capture_output=True,
        check=True,
        timeout=30,
    )
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Native generated audio has no finite decoded samples")
    return samples


class SpeechVerifier:
    def __init__(self, cache, reference):
        import torch
        import whisper
        from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor

        self.torch = torch
        torch.set_num_threads(2)
        pins = json.loads((ROOT / "benchmarks/native/semantic_pins.json").read_text())
        if version("openai-whisper") != pins["whisper"]["version"]:
            raise ValueError("Semantic ASR SDK version drift")
        self.evidence = {
            "scope": "EVALUATION_ONLY_NOT_FACTORY_TARGET_MODEL_OR_FALLBACK",
            "pins_sha256": digest(pins),
            "verified_files": {},
            "loaded": {},
            "runtime_versions": {
                p: version(p) for p in ["openai-whisper", "transformers", "torch"]
            },
        }
        whisper_path = cache / "whisper/base.en.pt"
        self.evidence["verified_files"]["whisper/base.en.pt"] = pinned_download(
            pins["whisper"]["url"], pins["whisper"], whisper_path
        )
        self.asr = whisper.load_model(str(whisper_path), device="cpu")
        speaker = pins["speaker"]
        speaker_path = cache / "speaker"
        for name, spec in speaker["files"].items():
            url = f"https://huggingface.co/{speaker['repository']}/resolve/{speaker['revision']}/{name}"
            self.evidence["verified_files"]["speaker/" + name] = pinned_download(
                url, spec, speaker_path / name
            )
        self.features = Wav2Vec2FeatureExtractor.from_pretrained(
            speaker_path, local_files_only=True
        )
        self.speaker = (
            WavLMForXVector.from_pretrained(
                speaker_path, local_files_only=True, use_safetensors=False
            )
            .float()
            .eval()
            .to("cpu")
        )
        for name, model in [("whisper", self.asr), ("speaker", self.speaker)]:
            parameters = list(model.parameters())
            devices = sorted({str(p.device) for p in parameters})
            dtypes = sorted({str(p.dtype) for p in parameters if p.is_floating_point()})
            if devices != ["cpu"] or dtypes != ["torch.float32"]:
                raise ValueError(
                    "Semantic verifier is not on its explicitly selected CPU/FP32 runtime"
                )
            self.evidence["loaded"][name] = {
                "parameter_count": sum(p.numel() for p in parameters),
                "devices": devices,
                "dtypes": dtypes,
            }
        self.reference_sha256 = hashlib.sha256(reference).hexdigest()
        self.reference_embedding = self.embedding(decode_audio(reference))

    def embedding(self, samples):
        features = self.features(samples, sampling_rate=16000, return_tensors="pt", padding=True)
        with self.torch.inference_mode():
            vector = self.speaker(**features).embeddings[0]
            vector = self.torch.nn.functional.normalize(vector, dim=0)
        return vector.cpu().tolist()

    def evaluate(self, raw):
        samples = decode_audio(raw)
        with self.torch.inference_mode():
            result = self.asr.transcribe(
                samples, language="en", fp16=False, temperature=0, condition_on_previous_text=False
            )
        return {
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "decoded_samples_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
            "sample_rate": 16000,
            "duration_seconds": len(samples) / 16000,
            "transcript": result["text"],
            "reference_sha256": self.reference_sha256,
            "speaker_reference_embedding": self.reference_embedding,
            "speaker_output_embedding": self.embedding(samples),
            "verifier_pins_sha256": self.evidence["pins_sha256"],
        }
