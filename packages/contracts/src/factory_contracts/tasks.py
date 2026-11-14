"""
factory_contracts/tasks.py
Authoritative Canonical Identifier Resolution and Normalization Maps for Models and Tasks.
Provides bidirectional resolution between capability IDs, aliases, and canonical names.
"""
from typing import List, Optional

from factory_contracts.capabilities import (
    CANONICAL_TASK_MAP, CANONICAL_TO_CAPABILITY_MAP,
    normalize_capability, capability_aliases,
)

# =========================================================================
# Model Resolution Maps
# =========================================================================

from factory_contracts.manifests.catalog import CatalogIdentityMapping

# Read-only compatibility views. Model aliases/families are authored only in
# ModelManifest; capability names/aliases come from the canonical registry.
CANONICAL_MODEL_MAP = CatalogIdentityMapping()
MODEL_TO_FAMILY_MAP = CatalogIdentityMapping(families=True)


# =========================================================================
# Resolution & Normalization Helper Functions
# =========================================================================

def normalize_task(task: Optional[str], *, engine=None, inputs=None) -> str:
    """Compatibility name for the sole ingress capability normalizer."""
    return normalize_capability(task, engine=engine, inputs=inputs)


def task_to_capability_id(canonical_task: Optional[str]) -> Optional[str]:
    normalized = normalize_task(canonical_task)
    return CANONICAL_TO_CAPABILITY_MAP.get(normalized)


def get_task_aliases(canonical_task: str) -> List[str]:
    """Boundary/introspection helper; dispatch must never expand aliases."""
    return capability_aliases(normalize_task(canonical_task))


def normalize_model(model: Optional[str]) -> str:
    """
    Normalizes any model family, short alias, or checkpoint variant
    into its authoritative canonical model ID (e.g. 'qwen-image' -> 'qwen-image-2.1').
    If not found in CANONICAL_MODEL_MAP, returns trimmed, lowercased string.
    """
    if not model:
        return ""
    key = str(model).strip().lower()
    return CANONICAL_MODEL_MAP.get(key, key)


def model_to_family(canonical_model: Optional[str]) -> Optional[str]:
    """
    Resolves a canonical model ID (e.g. 'qwen-image-2.1') to its manifest
    model family (e.g. 'qwen-image').
    """
    if not canonical_model:
        return None
    normalized = normalize_model(canonical_model)
    return MODEL_TO_FAMILY_MAP.get(normalized)


def get_model_aliases(canonical_model: str) -> List[str]:
    """Returns all aliases and family names that resolve to the given canonical model."""
    normalized = normalize_model(canonical_model)
    return [alias for alias, target in CANONICAL_MODEL_MAP.items() if target == normalized]
