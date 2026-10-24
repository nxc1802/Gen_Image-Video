"""
services/music/acestep/adapter.py
Translates Factory Job payloads to ACE-Step task specifications and maps outputs to ArtifactReferences.
"""
from typing import Any, Dict
from factory_contracts.jobs.job_contract import ArtifactReference


class ACEStepAdapter:
    """Translates job inputs to ACE-Step parameters and formats artifacts."""

    @staticmethod
    def map_inputs(task: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Maps Factory job inputs dictionary into ACE-Step task arguments."""
        params: Dict[str, Any] = {}

        if task == "text_to_music":
            params["prompt"] = inputs.get("prompt", "")
            params["duration_seconds"] = inputs.get("duration_seconds", 60)
            params["genre"] = inputs.get("genre", "pop")
            params["tempo"] = inputs.get("tempo", 120)

        elif task == "lyrics_to_music":
            params["prompt"] = inputs.get("prompt", "")
            params["lyrics"] = inputs.get("lyrics", "")
            params["vocal_style"] = inputs.get("vocal_style", "default")
            params["duration_seconds"] = inputs.get("duration_seconds", 60)

        elif task == "audio_to_audio":
            params["audio_url"] = inputs.get("audio_url", "")
            params["prompt"] = inputs.get("prompt", "")
            params["audio_strength"] = inputs.get("audio_strength", 0.5)
            params["seed"] = inputs.get("seed", 42)

        return params

    @staticmethod
    def create_artifact_reference(result: Dict[str, Any], base_s3_url: str = "https://s3.ai.example.com") -> ArtifactReference:
        """Constructs an ArtifactReference from synthesis output."""
        art_id = result["artifact_id"]
        storage_path = f"artifacts/music/{art_id}.wav"
        s3_url = f"{base_s3_url.rstrip('/')}/assets/{art_id}.wav"

        return ArtifactReference(
            artifact_id=art_id,
            type="audio",
            mime_type="audio/wav",
            size_bytes=result["size_bytes"],
            storage_path=storage_path,
            url=s3_url,
            download_url=s3_url,
            sha256=result["sha256"],
        )
