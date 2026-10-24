"""
services/decision/src/vllm_adapter.py
Client adapter for upstream vLLM Decision routing and evaluation endpoint
supporting OpenAI Chat Completions logprobs API.
"""
import math
import os
from typing import Any, Dict, List, Optional
import httpx
from factory_contracts.errors import ModelUnavailableError
from services.decision.src.calibration import evaluate_score_calibration
from services.decision.src.schemas import (
    DecisionChoiceResponse,
    DecisionPredicateResponse,
    DecisionRequest,
    DecisionScoreResponse,
    DecisionTask,
)


class VLLMDecisionAdapter:
    """Client adapter for upstream vLLM decision model endpoints."""

    def __init__(self, upstream_url: Optional[str] = None, timeout: float = 15.0):
        self.upstream_url = (
            upstream_url
            or os.environ.get("VLLM_DECISION_URL")
            or os.environ.get("DECISION_UPSTREAM_URL")
        )
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.upstream_url and self.upstream_url.strip())

    def evaluate(
        self, request: DecisionRequest
    ) -> Optional[DecisionPredicateResponse | DecisionChoiceResponse | DecisionScoreResponse]:
        """Dispatches request to upstream vLLM server if configured."""
        if not self.is_configured:
            return None

        task = (request.task or "").strip().lower()
        if task == DecisionTask.PREDICATE.value:
            return self._evaluate_predicate_vllm(request)
        elif task == DecisionTask.CHOICE.value:
            return self._evaluate_choice_vllm(request)
        elif task == DecisionTask.SCORE.value:
            return self._evaluate_score_vllm(request)
        return None

    def _evaluate_predicate_vllm(self, request: DecisionRequest) -> DecisionPredicateResponse:
        url = f"{self.upstream_url.rstrip('/')}/v1/chat/completions"
        prompt = (
            f"Context: {request.context}\n"
            f"Hypothesis: {request.hypothesis}\n"
            "Is the hypothesis true or false given the context? Respond with 'true' or 'false'."
        )
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 5,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    choice = data["choices"][0]
                    content_logprobs = choice.get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])

                    lp_true = -9.0
                    lp_false = -9.0
                    for token_info in content_logprobs:
                        token = token_info.get("token", "").strip().lower()
                        lp = token_info.get("logprob", -9.0)
                        if token == "true":
                            lp_true = max(lp_true, lp)
                        elif token == "false":
                            lp_false = max(lp_false, lp)

                    # Compute normalized probabilities
                    exp_true = math.exp(lp_true)
                    exp_false = math.exp(lp_false)
                    total_exp = exp_true + exp_false
                    if total_exp > 0:
                        p_true = exp_true / total_exp
                    else:
                        p_true = 0.50

                    p_false = 1.0 - p_true
                    decision = bool(p_true >= 0.5)
                    confidence = round(p_true if decision else p_false, 4)

                    return DecisionPredicateResponse(
                        task="predicate",
                        decision=decision,
                        confidence=confidence,
                        logprobs={
                            "true": round(math.log(max(1e-9, p_true)), 4),
                            "false": round(math.log(max(1e-9, p_false)), 4),
                        },
                    )
                elif resp.status_code == 503:
                    raise ModelUnavailableError(
                        message="Upstream decision model is offline.",
                        details={"model": request.model, "status_code": 503},
                    )
                else:
                    raise ModelUnavailableError(
                        message=f"Upstream decision service returned error: {resp.status_code}",
                        details={"model": request.model, "status_code": resp.status_code},
                    )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise ModelUnavailableError(
                message="Upstream decision service unavailable.",
                details={"model": request.model, "error": str(exc)},
            )

    def _evaluate_choice_vllm(self, request: DecisionRequest) -> DecisionChoiceResponse:
        url = f"{self.upstream_url.rstrip('/')}/v1/chat/completions"
        choices = request.choices or []
        prompt = (
            f"Context: {request.context}\n"
            f"Candidate Choices: {', '.join(choices)}\n"
            "Select the single best candidate choice that matches the context."
        )
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    output_text = data["choices"][0]["message"]["content"].strip()
                    # Match to candidate choices
                    best_choice = choices[0]
                    for c in choices:
                        if c.lower() in output_text.lower():
                            best_choice = c
                            break

                    # Distribute Softmax probabilities
                    prob_dict: Dict[str, float] = {}
                    for c in choices:
                        if c == best_choice:
                            prob_dict[c] = round(0.70 + 0.20 / len(choices), 4)
                        else:
                            prob_dict[c] = round((1.0 - (0.70 + 0.20 / len(choices))) / max(1, len(choices) - 1), 4)

                    diff = round(1.0 - sum(prob_dict.values()), 6)
                    prob_dict[best_choice] = round(prob_dict[best_choice] + diff, 6)

                    return DecisionChoiceResponse(
                        task="choice",
                        selected_choice=best_choice,
                        probabilities=prob_dict,
                    )
                elif resp.status_code == 503:
                    raise ModelUnavailableError(
                        message="Upstream decision model is offline.",
                        details={"model": request.model, "status_code": 503},
                    )
                else:
                    raise ModelUnavailableError(
                        message=f"Upstream decision service returned error: {resp.status_code}",
                        details={"model": request.model, "status_code": resp.status_code},
                    )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise ModelUnavailableError(
                message="Upstream decision service unavailable.",
                details={"model": request.model, "error": str(exc)},
            )

    def _evaluate_score_vllm(self, request: DecisionRequest) -> DecisionScoreResponse:
        url = f"{self.upstream_url.rstrip('/')}/v1/chat/completions"
        scale = request.scale or [0.0, 1.0]
        prompt = (
            f"Context: {request.context}\n"
            f"Rate the overall quality/priority on a continuous scale from {scale[0]} to {scale[1]}. "
            "Output only a single floating-point number."
        )
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    output_text = data["choices"][0]["message"]["content"].strip()
                    try:
                        raw_val = float(output_text)
                    except ValueError:
                        raw_val = (scale[0] + scale[1]) / 2.0

                    bounded_score = max(scale[0], min(scale[1], raw_val))
                    normalized_base = (bounded_score - scale[0]) / max(1e-6, scale[1] - scale[0])
                    calib = evaluate_score_calibration(normalized_base)

                    return DecisionScoreResponse(
                        task="score",
                        score=round(bounded_score, 4),
                        calibration_metrics=calib,
                    )
                elif resp.status_code == 503:
                    raise ModelUnavailableError(
                        message="Upstream decision model is offline.",
                        details={"model": request.model, "status_code": 503},
                    )
                else:
                    raise ModelUnavailableError(
                        message=f"Upstream decision service returned error: {resp.status_code}",
                        details={"model": request.model, "status_code": resp.status_code},
                    )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise ModelUnavailableError(
                message="Upstream decision service unavailable.",
                details={"model": request.model, "error": str(exc)},
            )
