"""
services/decision/src/schemas.py
Pydantic Schemas for the Decision Engine Subsystem.
"""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DecisionTask(str, Enum):
    PREDICATE = "predicate"
    CHOICE = "choice"
    SCORE = "score"


class DecisionRequest(BaseModel):
    model: str = Field(..., description="Model identifier for decision routing")
    task: str = Field(..., description="Decision task type ('predicate', 'choice', or 'score')")
    context: str = Field(..., description="Context text upon which decision is made")
    hypothesis: Optional[str] = Field(default=None, description="Hypothesis to test for 'predicate' task")
    choices: Optional[List[str]] = Field(default=None, description="Candidate choices for 'choice' task")
    scale: Optional[List[float]] = Field(default=None, description="Score boundaries [min, max] for 'score' task")


class DecisionPredicateResponse(BaseModel):
    task: str = "predicate"
    decision: bool = Field(..., description="Binary evaluation decision (True/False)")
    confidence: float = Field(..., description="Confidence probability (0.0 to 1.0)")
    logprobs: Dict[str, float] = Field(..., description="Log-probabilities for 'true' and 'false' tokens")


class DecisionChoiceResponse(BaseModel):
    task: str = "choice"
    selected_choice: str = Field(..., description="Selected routing candidate choice")
    probabilities: Dict[str, float] = Field(..., description="Full probability distribution over choices")


class CalibrationMetrics(BaseModel):
    ece: float = Field(..., description="Expected Calibration Error (ECE)")
    brier_score: float = Field(..., description="Brier score against calibrated benchmark")


class DecisionScoreResponse(BaseModel):
    task: str = "score"
    score: float = Field(..., description="Continuous calibrated score")
    calibration_metrics: CalibrationMetrics = Field(..., description="Reliability and calibration metrics")
