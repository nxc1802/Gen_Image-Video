"""
services/decision/src/engine.py
Decision Engine coordinator enforcing fail-fast model availability,
vLLM adapter integration, and generalized task routing.
"""
from typing import Optional, Union
from factory_contracts.errors import InvalidRequestError, ModelUnavailableError
from services.decision.src.choice import evaluate_choice
from services.decision.src.predicate import evaluate_predicate
from services.decision.src.schemas import (
    DecisionChoiceResponse,
    DecisionPredicateResponse,
    DecisionRequest,
    DecisionScoreResponse,
    DecisionTask,
)
from services.decision.src.score import evaluate_score
from services.decision.src.vllm_adapter import VLLMDecisionAdapter

OFFLINE_DECISION_MODELS = {
    "offline-decision-model",
    "offline-model",
    "failing-model",
    "failing-decision-model",
}


class DecisionEngine:
    """Core Decision Engine supporting predicate, choice, and score routing."""

    def __init__(self, upstream_url: Optional[str] = None):
        self.adapter = VLLMDecisionAdapter(upstream_url=upstream_url)

    def check_model_availability(self, model: str) -> None:
        """Verifies model availability, strictly raising 503 if unavailable."""
        if not model:
            raise InvalidRequestError("Field 'model' is required for decisions.")
        if model in OFFLINE_DECISION_MODELS:
            raise ModelUnavailableError(
                message="Decision model is offline. Heuristic fake outputs are prohibited.",
                details={"model": model},
            )

    def evaluate(
        self, request: DecisionRequest
    ) -> Union[DecisionPredicateResponse, DecisionChoiceResponse, DecisionScoreResponse]:
        """Dispatches decision request to vLLM adapter or generalized task evaluator."""
        self.check_model_availability(request.model)

        if not request.context or not request.context.strip():
            raise InvalidRequestError("Field 'context' is required for decisions.")

        # Try upstream vLLM adapter if configured
        if self.adapter.is_configured:
            upstream_resp = self.adapter.evaluate(request)
            if upstream_resp is not None:
                return upstream_resp

        task = (request.task or "").strip().lower()

        if task == DecisionTask.PREDICATE.value:
            return evaluate_predicate(request.context, request.hypothesis or "")

        elif task == DecisionTask.CHOICE.value:
            return evaluate_choice(request.context, request.choices or [])

        elif task == DecisionTask.SCORE.value:
            return evaluate_score(request.context, request.scale)

        else:
            raise InvalidRequestError(f"Unknown decision task '{request.task}'.")
