"""
factory/apps/control-api/src/queue/policy.py
Authoritative Dual-T4 GPU subsystem placement policy for Factory v4 Lite.
Single source of truth for Subsystem -> GPU partition mapping.
"""
from typing import Dict, Optional

# GPU 0: Text, Decision, VLM, Image, Speech
# GPU 1: Video, Music, Enhancement, Embedding, Rerank
SUBSYSTEM_GPU_MAP: Dict[str, int] = {
    "text": 0,
    "vlm": 0,
    "vision": 0,
    "decision": 0,
    "image": 0,
    "speech": 0,
    "video": 1,
    "music": 1,
    "enhancement": 1,
    "embedding": 1,
    "embed": 1,
    "rerank": 1,
}


def get_subsystem_gpu(service: Optional[str]) -> int:
    """Returns the authoritative preferred GPU partition (0 or 1) for a given service."""
    if not service:
        return 0
    return SUBSYSTEM_GPU_MAP.get(str(service).lower().strip(), 0)


__all__ = ["SUBSYSTEM_GPU_MAP", "get_subsystem_gpu"]
