"""
services/decision/src/score.py
Task evaluator for Continuous Scoring and Calibration Evaluation with generalized quality assessment.
"""
import re
from typing import List, Optional
from factory_contracts.errors import InvalidRequestError
from services.decision.src.calibration import evaluate_score_calibration
from services.decision.src.schemas import DecisionScoreResponse

POSITIVE_SIGNALS = {
    "high", "correct", "good", "fine", "clean", "smooth", "clear",
    "rich", "detailed", "valid", "success", "priority", "urgent", "optimum",
    "optimal", "accurate", "stable", "passed",
}

NEGATIVE_SIGNALS = {
    "low", "poor", "bad", "defect", "fail", "broken", "corrupt",
    "distorted", "blurry", "degraded", "unstable", "invalid",
}


def _evaluate_base_quality(context: str) -> float:
    """
    Evaluates intrinsic quality or priority score in range [0.05, 0.95]
    using generalized lexical diversity and sentiment signal density.
    """
    tokens = re.findall(r"\w+", context.lower())
    if not tokens:
        return 0.5000

    unique_tokens = len(set(tokens))
    lexical_ratio = unique_tokens / len(tokens)

    pos_count = sum(1 for t in tokens if t in POSITIVE_SIGNALS)
    neg_count = sum(1 for t in tokens if t in NEGATIVE_SIGNALS)

    sentiment_delta = (pos_count - neg_count) * 0.08
    base = 0.65 + (lexical_ratio * 0.15) + sentiment_delta
    return round(max(0.05, min(0.95, base)), 4)


def evaluate_score(context: str, scale: Optional[List[float]] = None) -> DecisionScoreResponse:
    """Computes continuous score scaled to custom boundaries and evaluated against calibration curve."""
    if scale is None:
        scale = [0.0, 1.0]

    if len(scale) != 2 or scale[0] >= scale[1]:
        raise InvalidRequestError("Scale min must be strictly less than max.")

    base_score = _evaluate_base_quality(context)

    # Scale interpolation: mapped strictly within [scale[0], scale[1]]
    s_min, s_max = scale[0], scale[1]
    scaled_score = s_min + (base_score * (s_max - s_min))
    final_score = round(max(s_min, min(s_max, scaled_score)), 4)

    calib = evaluate_score_calibration(base_score)

    return DecisionScoreResponse(
        task="score",
        score=final_score,
        calibration_metrics=calib,
    )
