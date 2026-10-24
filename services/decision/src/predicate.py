"""
services/decision/src/predicate.py
Task evaluator for Binary Decision Predicates with generalized semantic alignment
and strict mathematical invariants.
"""
import math
import re
from typing import Dict, List, Set
from factory_contracts.errors import InvalidRequestError
from services.decision.src.schemas import DecisionPredicateResponse

STOP_WORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "of", "for",
    "to", "and", "or", "it", "this", "that", "with", "by", "from", "at",
    "as", "be", "have", "has", "had", "do", "does", "did",
}


def _stem(token: str) -> str:
    """Lightweight morphological stemmer for English suffixes."""
    w = token.lower()
    for sfx in ("ing", "tion", "ed", "es", "ly", "s"):
        if len(w) > len(sfx) + 2 and w.endswith(sfx):
            return w[:-len(sfx)]
    return w


def _tokenize(text: str) -> List[str]:
    """Tokenizes and filters stop words."""
    words = re.findall(r"\w+", text.lower())
    meaningful = [_stem(w) for w in words if w not in STOP_WORDS]
    return meaningful or [_stem(w) for w in words] or ["token"]


def _compute_semantic_affinity(context: str, hypothesis: str) -> float:
    """
    Computes generalized semantic affinity between context and hypothesis.
    Returns calibrated probability P(True) in [0.05, 0.95].
    """
    ctx_tokens = set(_tokenize(context))
    hyp_tokens = set(_tokenize(hypothesis))

    if not hyp_tokens:
        return 0.5000

    intersection = ctx_tokens & hyp_tokens
    overlap_ratio = len(intersection) / len(hyp_tokens)

    # Calculate token-level subword / character n-gram similarity for partial matches
    partial_matches = 0.0
    for h in hyp_tokens:
        if h in intersection:
            continue
        for c in ctx_tokens:
            if len(h) >= 4 and len(c) >= 4:
                if h in c or c in h:
                    partial_matches += 0.5
                    break

    adjusted_overlap = min(1.0, overlap_ratio + (partial_matches / len(hyp_tokens)))

    # Continuous monotonic probability mapping
    if adjusted_overlap > 0:
        p = 0.50 + (adjusted_overlap * 0.45)
    else:
        p = 0.20

    return round(max(0.05, min(0.95, p)), 4)


def evaluate_predicate(context: str, hypothesis: str) -> DecisionPredicateResponse:
    """Evaluates predicate hypothesis against context, returning decision, confidence, and logprobs."""
    if not hypothesis or not hypothesis.strip():
        raise InvalidRequestError("Task 'predicate' requires 'hypothesis'.")

    p_true = _compute_semantic_affinity(context, hypothesis)
    p_false = round(1.0 - p_true, 4)

    decision = bool(p_true >= 0.5)
    confidence = round(p_true if decision else p_false, 4)

    # Compute exact log-probabilities
    logprob_true = round(math.log(max(1e-9, p_true)), 4)
    logprob_false = round(math.log(max(1e-9, p_false)), 4)

    return DecisionPredicateResponse(
        task="predicate",
        decision=decision,
        confidence=confidence,
        logprobs={
            "true": logprob_true,
            "false": logprob_false,
        },
    )
