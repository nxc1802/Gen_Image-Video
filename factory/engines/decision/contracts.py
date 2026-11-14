"""Canonical internal decision contracts, without a backend protocol dependency.

The single-task request is retained for ordered service migration. Phase 6 owns
the public multi-question API. No response requires invented calibration data.
"""

from enum import Enum
from factory_contracts.capabilities import normalize_capability
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    PrivateAttr,
    model_validator,
)


class DecisionTask(str, Enum):
    PREDICATE = "decision.predicate_evaluation"
    CHOICE = "decision.choice_routing"
    SCORE = "decision.score_ranking"


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: StrictStr
    task: Literal[
        "decision.predicate_evaluation", "decision.choice_routing", "decision.score_ranking"
    ]
    context: StrictStr
    hypothesis: StrictStr | None = None
    choices: list[StrictStr] | None = None
    scale: list[Annotated[float, Field(strict=True)]] | None = None


class NativeObject(BaseModel):
    model_config = ConfigDict(extra="forbid")


Probability = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]
TokenCount = Annotated[StrictInt, Field(ge=0)]


class PredicateAnswer(NativeObject):
    type: Literal["predicate"]
    name: StrictStr | None
    probability: Probability


class ChoiceProbability(NativeObject):
    value: StrictStr
    probability: Probability


class ChoiceAnswer(NativeObject):
    type: Literal["choice"]
    name: StrictStr | None
    choice: StrictStr
    probabilities: list[ChoiceProbability]
    confidence: Probability


class ScoreProbability(NativeObject):
    value: StrictInt
    label: StrictStr
    probability: Probability


class ScoreAnswer(NativeObject):
    type: Literal["score"]
    name: StrictStr | None
    score: FiniteNumber
    probabilities: list[ScoreProbability]
    confidence: Probability


class RefusalAnswer(NativeObject):
    type: Literal["refusal"]
    name: StrictStr | None


Answer = Annotated[
    PredicateAnswer | ChoiceAnswer | ScoreAnswer | RefusalAnswer, Field(discriminator="type")
]


class InputTokenDetails(NativeObject):
    cached_tokens: TokenCount
    cache_write_tokens: TokenCount


class OutputTokenDetails(NativeObject):
    reasoning_tokens: TokenCount


class DecisionUsage(NativeObject):
    input_tokens: TokenCount
    input_tokens_details: InputTokenDetails
    output_tokens: TokenCount
    output_tokens_details: OutputTokenDetails
    total_tokens: TokenCount


class NativeDecisionResponse(NativeObject):
    model: StrictStr
    answers: list[Answer]
    usage: DecisionUsage


class ScoreMapping(NativeObject):
    method: Literal["linear_map_of_native_expected_level"] = "linear_map_of_native_expected_level"
    minimum: FiniteNumber
    maximum: FiniteNumber
    value: FiniteNumber


class DecisionResult(NativeObject):
    model: StrictStr
    task: DecisionTask
    answer: Answer
    usage: DecisionUsage
    score_mapping: ScoreMapping | None = None


class PublicDecisionQuestion(NativeObject):
    type: Literal["predicate", "choice", "score"]
    name: StrictStr | None = None
    instructions: StrictStr
    choices: list[dict] | None = None
    levels: list[dict] | None = None
    _capability: str = PrivateAttr()

    @model_validator(mode="after")
    def canonical_capability(self):
        self._capability = normalize_capability(self.type, engine="decision")
        return self

    @property
    def capability(self):
        return self._capability


class PublicDecisionRequest(NativeObject):
    model: StrictStr
    input: StrictStr
    questions: list[PublicDecisionQuestion]

    @property
    def task(self):
        return self.questions[0].capability if self.questions else "decision.predicate_evaluation"
