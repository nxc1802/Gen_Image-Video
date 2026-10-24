"""
services/decision/src/choice.py
Task evaluator for Multi-Class Decision Routing with generalized semantic alignment
and strict Softmax probability distributions.
"""
import math
import re
from typing import Dict, List, Set
from factory_contracts.errors import InvalidRequestError
from services.decision.src.schemas import DecisionChoiceResponse

STOP_WORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "of", "for",
    "to", "and", "or", "it", "this", "that", "with", "by", "from", "at",
}


def _stem(token: str) -> str:
    """Lightweight morphological stemmer for English suffixes."""
    w = token.lower()
    for sfx in ("ing", "tion", "ed", "es", "ly", "s"):
        if len(w) > len(sfx) + 2 and w.endswith(sfx):
            return w[:-len(sfx)]
    return w


def _tokenize(text: str) -> List[str]:
    """Tokenizes text into words and stemmed units."""
    words = re.findall(r"\w+", text.lower())
    meaningful = [_stem(w) for w in words if w not in STOP_WORDS]
    return meaningful or [_stem(w) for w in words] or ["token"]


def _score_choice_relevance(context_tokens: Set[str], context_raw: str, choice: str) -> float:
    """
    Computes generalized relevance logit for candidate choice given context tokens.
    Awards weights for exact token matches, stem matches, and subword substrings.
    """
    # Choice can be compound like 'audio_enhancement' or 'text_generation'
    choice_words = re.findall(r"\w+", choice.lower())
    choice_stems = [_stem(w) for w in choice_words if w not in STOP_WORDS] or choice_words

    match_score = 0.0
    for w, s in zip(choice_words, choice_stems):
        if w in context_raw.lower():
            match_score += 2.5
        elif s in context_tokens:
            match_score += 1.8
        else:
            # Check partial substring in context tokens
            for ct in context_tokens:
                if len(s) >= 4 and (s in ct or ct in s):
                    match_score += 1.0
                    break

    return 1.0 + match_score


def evaluate_choice(context: str, choices: List[str]) -> DecisionChoiceResponse:
    """Selects best candidate choice and computes normalized Softmax probability distribution."""
    if not choices or not isinstance(choices, list) or len(choices) == 0:
        raise InvalidRequestError("Task 'choice' requires non-empty list of 'choices'.")

    ctx_tokens = set(_tokenize(context))
    logits = [_score_choice_relevance(ctx_tokens, context, c) for c in choices]

    # Numerically stable Softmax calculation
    max_logit = max(logits)
    exp_logits = [math.exp(z - max_logit) for z in logits]
    sum_exp = sum(exp_logits)

    raw_probs = [e / sum_exp for e in exp_logits]

    # Round to 6 decimal places while strictly enforcing sum == 1.0
    probs_rounded = [round(p, 6) for p in raw_probs]
    diff = round(1.0 - sum(probs_rounded), 6)
    best_idx = int(max(range(len(probs_rounded)), key=lambda i: probs_rounded[i]))
    probs_rounded[best_idx] = round(probs_rounded[best_idx] + diff, 6)

    prob_dict: Dict[str, float] = {}
    for c, p in zip(choices, probs_rounded):
        prob_dict[c] = p

    selected_choice = choices[best_idx]

    return DecisionChoiceResponse(
        task="choice",
        selected_choice=selected_choice,
        probabilities=prob_dict,
    )
