"""Deprecated predicate compatibility shim.

Factory v4 Decisions are served by native vLLM /v1/decisions. Local heuristic
scoring is forbidden because it can masquerade as model inference.
"""
from factory_contracts.errors import ModelUnavailableError

def evaluate_predicate(*_args, **_kwargs):
    raise ModelUnavailableError(
        message="Local Decision heuristics are disabled. Use the native vLLM Decisions backend."
    )
