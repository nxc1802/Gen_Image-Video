"""
services/music/yue2/adapter.py
Task and parameter mapping, validation, and error translation for YuE2 subsystem.
Handles tasks: text_to_music, lyrics_to_music, symbolic_score, custom_score, score_edit, song_variation.
"""
import math
import re
from typing import Any, Dict, List, Optional, Set

from factory_contracts.errors import (
    InvalidRequestError,
    UnsupportedCapabilityError,
)
from services.music.yue2.score import validate_abc_score

ALL_SUPPORTED_TASKS: Set[str] = {
    "music.text_to_music",
    "music.lyrics_to_music",
    "music.symbolic_score",
    "music.custom_score",
    "music.score_edit",
    "music.song_variation",
}

VALID_VARIANTS: Set[str] = {
    "standard",
    "default",
    "7b",
    "3b",
    "track_a",
    "track_b",
    "quantized",
    "q8_0",
    "q6_k",
    "q5_k_m",
}

MAX_DURATION_SECONDS: float = 300.0
DEFAULT_DURATION_SECONDS: float = 30.0


class YuE2Adapter:
    """Validates and maps Factory music requests to YuE2 native parameters."""

    @classmethod
    def normalize_variant(cls, variant: Optional[str]) -> str:
        norm = str(variant or "").strip().lower()
        if not norm or norm in ("standard", "default"):
            return "standard"
        return norm

    @classmethod
    def is_task_supported(cls, task: str) -> bool:
        return task in ALL_SUPPORTED_TASKS

    @classmethod
    def is_task_supported_for_variant(cls, task: str, variant: Optional[str] = None) -> bool:
        canonical_task = task
        if canonical_task not in ALL_SUPPORTED_TASKS:
            return False
        norm_var = cls.normalize_variant(variant)
        return norm_var in VALID_VARIANTS

    @classmethod
    def validate_variant_capability(cls, task: str, variant: Optional[str] = None) -> None:
        canonical_task = task
        if canonical_task not in ALL_SUPPORTED_TASKS:
            raise UnsupportedCapabilityError(
                f"YuE2 does not support task '{task}'. "
                f"Supported tasks: {sorted(list(ALL_SUPPORTED_TASKS))}"
            )
        norm_var = cls.normalize_variant(variant)
        if norm_var not in VALID_VARIANTS:
            raise UnsupportedCapabilityError(
                f"YuE2 variant '{variant}' is not recognized. "
                f"Supported variants: {sorted(list(VALID_VARIANTS))}"
            )

    @classmethod
    def validate_duration(cls, raw_val: Any) -> float:
        if raw_val is None:
            return DEFAULT_DURATION_SECONDS

        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            raise InvalidRequestError(f"audio_duration must be a numeric value, got {type(raw_val).__name__}")

        if math.isnan(val):
            raise InvalidRequestError("audio_duration cannot be NaN")
        if math.isinf(val):
            raise InvalidRequestError("audio_duration cannot be infinite")

        if val <= 0.0 or val > MAX_DURATION_SECONDS:
            raise InvalidRequestError(
                f"audio_duration must be > 0 and <= {MAX_DURATION_SECONDS}s, got {val}"
            )
        return val

    @classmethod
    def validate_seed(cls, raw_val: Any) -> Optional[int]:
        if raw_val is None:
            return None

        # Check float NaN or Inf before converting
        if isinstance(raw_val, float):
            if math.isnan(raw_val) or math.isinf(raw_val):
                raise InvalidRequestError("seed cannot be NaN or infinite")

        try:
            val = int(raw_val)
        except (ValueError, TypeError):
            raise InvalidRequestError(f"seed must be an integer, got {type(raw_val).__name__}")

        if val < 0 or val > 2**32 - 1:
            raise InvalidRequestError(f"seed must be between 0 and 2^32 - 1, got {val}")
        return val

    @classmethod
    def validate_sampling_config(cls, raw_config: Any) -> Dict[str, Any]:
        if not raw_config:
            return {"temperature": 1.0, "top_p": 0.95, "top_k": 50}
        if not isinstance(raw_config, dict):
            raise InvalidRequestError("sampling_config must be a dictionary")

        config = dict(raw_config)
        temp = config.get("temperature", 1.0)
        try:
            temp_f = float(temp)
        except (ValueError, TypeError):
            raise InvalidRequestError("sampling_config.temperature must be a float")
        if math.isnan(temp_f) or math.isinf(temp_f) or temp_f <= 0.0 or temp_f > 2.0:
            raise InvalidRequestError(f"sampling_config.temperature must be between 0.0 and 2.0, got {temp_f}")
        config["temperature"] = temp_f

        top_p = config.get("top_p", 0.95)
        try:
            top_p_f = float(top_p)
        except (ValueError, TypeError):
            raise InvalidRequestError("sampling_config.top_p must be a float")
        if math.isnan(top_p_f) or math.isinf(top_p_f) or top_p_f <= 0.0 or top_p_f > 1.0:
            raise InvalidRequestError(f"sampling_config.top_p must be between 0.0 and 1.0, got {top_p_f}")
        config["top_p"] = top_p_f

        if "top_k" in config and config["top_k"] is not None:
            try:
                top_k_i = int(config["top_k"])
            except (ValueError, TypeError):
                raise InvalidRequestError("sampling_config.top_k must be an integer")
            if top_k_i < 0:
                raise InvalidRequestError("sampling_config.top_k must be >= 0")
            config["top_k"] = top_k_i
        else:
            config["top_k"] = 50

        return config

    @classmethod
    def map_inputs(cls, task: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates input arguments and maps them to clean parameters for YuE2 backend.
        Raises InvalidRequestError or UnsupportedCapabilityError if inputs are invalid.
        """
        inputs = dict(inputs or {})
        canonical_task = task
        variant = cls.normalize_variant(inputs.get("variant"))
        cls.validate_variant_capability(canonical_task, variant)

        # Duration validation
        raw_dur = inputs.get("audio_duration") if "audio_duration" in inputs else inputs.get("duration")
        duration = cls.validate_duration(raw_dur)

        # Seed validation
        seed = cls.validate_seed(inputs.get("seed"))

        # Prompt & Lyrics validation
        prompt = str(inputs.get("prompt") or "").strip()
        lyrics = str(inputs.get("lyrics") or "").strip()
        style = inputs.get("style") or inputs.get("genres") or ""
        if isinstance(style, list):
            genres_str = ", ".join(str(s).strip() for s in style if s)
        else:
            genres_str = str(style).strip()

        # Custom score / ABC prompt
        custom_score = (
            inputs.get("custom_score")
            or inputs.get("abc_score")
            or inputs.get("score_prompt")
            or inputs.get("score")
        )
        if custom_score is not None:
            if not isinstance(custom_score, str) or not custom_score.strip():
                raise InvalidRequestError("custom_score must be a non-empty string of ABC notation")
            # Enforce validation
            validate_abc_score(custom_score)
            custom_score = custom_score.strip()

        # Task specific requirements
        if canonical_task == "music.text_to_music":
            if not prompt:
                raise InvalidRequestError("prompt is required for text_to_music")

        elif canonical_task == "music.lyrics_to_music":
            if not lyrics:
                raise InvalidRequestError("lyrics are required for lyrics_to_music")

        elif canonical_task == "music.custom_score":
            if not custom_score:
                raise InvalidRequestError("custom_score (valid ABC score) is required for custom_score task")

        elif canonical_task == "music.symbolic_score":
            if not prompt and not lyrics and not custom_score:
                raise InvalidRequestError(
                    "symbolic_score requires at least one of prompt, lyrics, or existing score"
                )

        elif canonical_task == "music.score_edit":
            base_score = (
                inputs.get("base_score")
                or inputs.get("abc_score")
                or inputs.get("custom_score")
                or custom_score
            )
            if not base_score:
                raise InvalidRequestError("score_edit requires base_score in ABC notation")
            validate_abc_score(base_score)
            modifications = inputs.get("modifications") or inputs.get("edits")
            if not modifications or not isinstance(modifications, dict):
                raise InvalidRequestError("score_edit requires modifications dictionary")

        elif canonical_task == "music.song_variation":
            if not prompt and not lyrics and not custom_score:
                raise InvalidRequestError("song_variation requires prompt, lyrics, or base score")

        sampling_config = cls.validate_sampling_config(inputs.get("sampling_config"))

        return {
            "task": canonical_task,
            "variant": variant,
            "prompt": prompt,
            "lyrics": lyrics,
            "genres": genres_str,
            "audio_duration": duration,
            "seed": seed,
            "score_prompt": custom_score,
            "base_score": inputs.get("base_score") or custom_score,
            "modifications": inputs.get("modifications") or inputs.get("edits"),
            "sampling_config": sampling_config,
            "raw_inputs": inputs,
        }
