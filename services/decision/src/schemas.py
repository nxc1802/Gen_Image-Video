"""
services/decision/src/schemas.py
Pydantic Schemas for the Decision Engine Subsystem.
"""
from typing import Dict
from pydantic import BaseModel, Field, field_validator
from factory_contracts.capabilities import normalize_capability
from factory.engines.decision.contracts import DecisionRequest as CanonicalDecisionRequest, DecisionTask, DecisionResult


class DecisionRequest(CanonicalDecisionRequest):
    """Legacy service ingress: normalize aliases once before canonical dispatch."""

    @field_validator("task", mode="before")
    @classmethod
    def normalize_request_task(cls, value):
        return normalize_capability(value, engine="decision")


# Historical response DTOs are retained for ordered migration only. Canonical
# execution returns DecisionResult without requiring per-request calibration metrics.
class DecisionPredicateResponse(BaseModel):
    task: str = "decision.predicate_evaluation"
    decision: bool = Field(..., description="Binary evaluation decision (True/False)")
    confidence: float = Field(..., description="Confidence probability (0.0 to 1.0)")
    logprobs: Dict[str, float] = Field(..., description="Log-probabilities for 'true' and 'false' tokens")


class DecisionChoiceResponse(BaseModel):
    task: str = "decision.choice_routing"
    selected_choice: str = Field(..., description="Selected routing candidate choice")
    probabilities: Dict[str, float] = Field(..., description="Full probability distribution over choices")


class CalibrationMetrics(BaseModel):
    ece: float = Field(..., description="Expected Calibration Error (ECE)")
    brier_score: float = Field(..., description="Brier score against calibrated benchmark")


class DecisionScoreResponse(BaseModel):
    task: str = "decision.score_ranking"
    score: float = Field(..., description="Continuous calibrated score")
    calibration_metrics: CalibrationMetrics = Field(..., description="Reliability and calibration metrics")
