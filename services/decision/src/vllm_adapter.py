"""
Strict adapter to vLLM's native OpenAI-compatible Decisions API.

This module translates the legacy single-task Factory request into one native
/v1/decisions question. All probabilities/scores come from vLLM; there are no
hand-written lexical, probability, or midpoint fallbacks.
"""
import os
from typing import Any, Dict, Optional

import httpx

from factory_contracts.errors import InvalidRequestError, ModelUnavailableError
from services.decision.src.schemas import DecisionRequest, DecisionTask


class VLLMDecisionAdapter:
    def __init__(self, upstream_url: Optional[str] = None, timeout: float = 60.0):
        self.upstream_url = (
            upstream_url
            or os.environ.get("VLLM_DECISION_URL")
            or os.environ.get("DECISION_UPSTREAM_URL")
            or ""
        ).rstrip("/")
        self.api_key = os.environ.get("VLLM_DECISION_API_KEY") or os.environ.get("FACTORY_API_KEY", "")
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.upstream_url)

    def _question(self, request: DecisionRequest) -> Dict[str, Any]:
        task = (request.task or "").strip().lower()
        if task == DecisionTask.PREDICATE.value:
            if not request.hypothesis or not request.hypothesis.strip():
                raise InvalidRequestError("Task 'predicate' requires 'hypothesis'.")
            return {
                "type": "predicate",
                "name": "decision",
                "instructions": request.hypothesis.strip(),
            }

        if task == DecisionTask.CHOICE.value:
            choices = request.choices or []
            if len(choices) < 2:
                raise InvalidRequestError("Task 'choice' requires at least two choices.")
            if len(choices) > 26:
                raise InvalidRequestError("Native vLLM Decisions supports at most 26 choices.")
            return {
                "type": "choice",
                "name": "decision",
                "instructions": "Select the choice that best matches the input context.",
                "choices": [{"value": choice} for choice in choices],
            }

        if task == DecisionTask.SCORE.value:
            scale = request.scale or [0.0, 1.0]
            if len(scale) != 2 or scale[0] >= scale[1]:
                raise InvalidRequestError("Task 'score' requires scale [min, max] with min < max.")
            # Native Decisions scores ordered discrete levels. Ten evenly spaced
            # labels preserve the requested numeric range without inventing a score.
            lo, hi = float(scale[0]), float(scale[1])
            levels = [
                {"label": f"{lo + (hi - lo) * i / 9.0:.6g}"}
                for i in range(10)
            ]
            return {
                "type": "score",
                "name": "decision",
                "instructions": f"Score the input on the requested range [{lo}, {hi}].",
                "levels": levels,
            }

        raise InvalidRequestError(f"Unknown decision task '{request.task}'.")

    def evaluate(self, request: DecisionRequest) -> Dict[str, Any]:
        if not self.is_configured:
            raise ModelUnavailableError(
                message="Native vLLM Decisions backend is not configured.",
                details={"required_env": "VLLM_DECISION_URL"},
            )

        payload = {
            "model": request.model,
            "input": request.context,
            "questions": [self._question(request)],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.upstream_url}/v1/decisions",
                    headers=headers,
                    json=payload,
                )
                if response.status_code >= 400:
                    raise ModelUnavailableError(
                        message="Native vLLM Decisions backend rejected the request.",
                        details={"status_code": response.status_code, "model": request.model},
                    )
                data = response.json()
        except ModelUnavailableError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                message="Native vLLM Decisions backend is unavailable.",
                details={"error_type": type(exc).__name__, "model": request.model},
            ) from exc

        answers = data.get("answers")
        if not isinstance(answers, list) or not answers:
            raise ModelUnavailableError(
                message="Native vLLM Decisions backend returned no answers.",
                details={"model": request.model},
            )
        return {
            "task": request.task,
            "answer": answers[0],
            "usage": data.get("usage"),
            "id": data.get("id"),
            "model": data.get("model", request.model),
        }
