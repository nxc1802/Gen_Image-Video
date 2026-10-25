"""
Strict compatibility proxy for native vLLM Decisions.

No lexical/heuristic fallback is permitted.  If the native decision backend is
not configured or unavailable the request fails closed with MODEL_UNAVAILABLE.
"""
from factory_contracts.errors import InvalidRequestError, ModelUnavailableError
from services.decision.src.schemas import DecisionRequest
from services.decision.src.vllm_adapter import VLLMDecisionAdapter

OFFLINE_DECISION_MODELS = {
    "offline-decision-model", "offline-model", "failing-model", "failing-decision-model"
}


class DecisionEngine:
    def __init__(self, upstream_url: str | None = None):
        self.adapter = VLLMDecisionAdapter(upstream_url=upstream_url)

    def evaluate(self, request: DecisionRequest):
        if not request.model:
            raise InvalidRequestError("Field 'model' is required for decisions.")
        if request.model in OFFLINE_DECISION_MODELS:
            raise ModelUnavailableError(message=f"Decision model '{request.model}' is offline.")
        if not request.context or not request.context.strip():
            raise InvalidRequestError("Field 'context' is required for decisions.")
        self.adapter._question(request)
        if not self.adapter.is_configured:
            raise ModelUnavailableError(
                message="Native vLLM decision backend is not configured. Heuristic fallback is disabled.",
                details={"required_env": "VLLM_DECISION_URL"},
            )
        result = self.adapter.evaluate(request)
        if result is None:
            raise InvalidRequestError(f"Unknown or unsupported decision task '{request.task}'.")
        return result
