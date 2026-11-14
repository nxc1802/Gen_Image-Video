"""Pinned independent CPU speech validation of audio actually decoded from native MP4."""

import base64
import hashlib
import json
import math
import shutil
import struct
import subprocess
from importlib.metadata import version
from pathlib import Path

from benchmarks.native.speech_semantic import pinned_download
from benchmarks.native.wave_a import digest
from benchmarks.native.wave_b import word_error_rate, words

ROOT = Path(__file__).resolve().parents[2]


def decode(raw):
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            "pipe:0",
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "pipe:1",
        ],
        input=raw,
        check=True,
        capture_output=True,
        timeout=60,
    )
    samples = [v[0] for v in struct.iter_unpack("<f", result.stdout)]
    if not samples or not all(math.isfinite(v) for v in samples):
        raise ValueError("Native video has no finite decoded audio samples")
    return result.stdout, samples


def semantic(expected, verification, raw, video_duration):
    decoded, samples = decode(raw)
    rms = math.sqrt(math.fsum(v * v for v in samples) / len(samples))
    clipped = sum(abs(v) >= 0.9995 for v in samples) / len(samples)
    duration = len(samples) / 16000
    wer = word_error_rate(expected["transcript"], verification["transcript"])
    normalized = " ".join(words(verification["transcript"]))
    checks = {
        "exact_generated_audio_identity": verification["input_sha256"]
        == hashlib.sha256(raw).hexdigest()
        and verification["decoded_samples_sha256"] == hashlib.sha256(decoded).hexdigest()
        and verification["sample_rate"] == 16000
        and verification["duration_seconds"] == duration,
        "exact_independent_audio_verifier": verification["verifier_pins_sha256"]
        == expected["verifier_pins_sha256"],
        "audible_finite_unclipped_audio": rms >= expected["minimum_rms"]
        and clipped <= expected["maximum_clipped_fraction"],
        "audio_video_duration_matches": abs(duration - video_duration)
        <= expected["maximum_av_duration_difference_seconds"],
        "actual_spoken_words_match": wer <= expected["maximum_word_error_rate"]
        and all(p in normalized for p in expected["required_phrases"]),
    }
    return {
        "checks": checks,
        "metrics": {
            "audio_rms": rms,
            "audio_clipped_fraction": clipped,
            "audio_duration_seconds": duration,
            "audio_word_error_rate": wer,
            "actual_audio_transcript": verification["transcript"],
        },
    }


class VideoSpeechVerifier:
    def __init__(self, cache):
        import torch
        import whisper

        self.torch = torch
        torch.set_num_threads(2)
        pins = json.loads((ROOT / "benchmarks/native/video_audio_semantic_pins.json").read_text())
        pin = pins["whisper"]
        if version("openai-whisper") != pin["version"]:
            raise ValueError("Pinned video semantic ASR runtime drift")
        target = cache / "base.en.pt"
        verified = pinned_download(pin["url"], pin, target)
        self.model = whisper.load_model(str(target), device="cpu").float().eval()
        params = list(self.model.parameters())
        loaded = {
            "parameter_count": sum(p.numel() for p in params),
            "devices": sorted({str(p.device) for p in params}),
            "dtypes": sorted({str(p.dtype) for p in params if p.is_floating_point()}),
        }
        if loaded != {"parameter_count": 71825408, "devices": ["cpu"], "dtypes": ["torch.float32"]}:
            raise ValueError(
                "Actual video semantic ASR parameters do not match pinned CPU/F32 verifier"
            )
        self.evidence = {
            "scope": pins["scope"],
            "pins_sha256": digest(pins),
            "verified_files": {"base.en.pt": verified},
            "loaded": loaded,
            "runtime_versions": {
                "openai-whisper": version("openai-whisper"),
                "torch": version("torch"),
            },
        }
        decoder = Path(shutil.which("ffmpeg") or "").resolve()
        if not decoder.is_file():
            raise ValueError("Actual native audio decoder is unavailable")
        self.evidence["audio_decoder"] = {
            "executable_sha256": hashlib.sha256(decoder.read_bytes()).hexdigest(),
            "version_output": subprocess.check_output(
                [str(decoder), "-version"], text=True, timeout=10
            ),
            "sample_format": "mono_16000hz_f32le",
            "classification": "OBSERVED_DECODER_IDENTITY_NOT_A_PORTABLE_DECODE_CERTIFICATE",
        }

    def evaluate(self, raw):
        import numpy as np

        decoded, samples = decode(raw)
        with self.torch.inference_mode():
            result = self.model.transcribe(
                np.frombuffer(decoded, dtype="<f4").copy(),
                language="en",
                fp16=False,
                temperature=0,
                condition_on_previous_text=False,
            )
        return {
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "decoded_samples_sha256": hashlib.sha256(decoded).hexdigest(),
            # Preserve the exact independent verifier input for decoder-drift diagnosis.
            # The gate still independently decodes the MP4 and checks the exact digest.
            "decoded_samples_base64": base64.b64encode(decoded).decode("ascii"),
            "sample_rate": 16000,
            "duration_seconds": len(samples) / 16000,
            "transcript": result["text"],
            "verifier_pins_sha256": self.evidence["pins_sha256"],
        }
