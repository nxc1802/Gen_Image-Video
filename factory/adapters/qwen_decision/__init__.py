"""Native typed decisions on the explicitly selected generation backend.

Probabilities, confidence and token usage come from the backend. A requested
numeric scale is derived separately from its expected zero-based level index.
"""

import math

from factory_contracts.errors import InvalidRequestError, ModelUnavailableError
from factory.core.model_adapter import ModelAdapter, PreparedCall
from factory.engines.decision.contracts import (
    DecisionResult,
    DecisionTask,
    NativeDecisionResponse,
    ScoreMapping,
)


class Adapter(ModelAdapter):
    capabilities = frozenset(task.value for task in DecisionTask)
    supported_backends = frozenset({"vllm"})
    max_choices = 26
    score_levels = 10

    def _question(self, request, capability):
        if not request.context.strip():
            raise InvalidRequestError("Decision context must be nonempty.")
        if capability == DecisionTask.PREDICATE:
            if (
                not request.hypothesis
                or not request.hypothesis.strip()
                or request.choices is not None
                or request.scale is not None
            ):
                raise InvalidRequestError(
                    "Predicate requires a hypothesis and no choice/scale fields."
                )
            return {"type": "predicate", "name": "decision", "instructions": request.hypothesis}
        if capability == DecisionTask.CHOICE:
            choices = request.choices or []
            if (
                not 2 <= len(choices) <= self.max_choices
                or len(set(choices)) != len(choices)
                or any(not choice.strip() for choice in choices)
            ):
                raise InvalidRequestError("Choice requires 2–26 distinct nonempty candidates.")
            if request.hypothesis is not None or request.scale is not None:
                raise InvalidRequestError("Choice does not accept hypothesis/scale fields.")
            return {
                "type": "choice",
                "name": "decision",
                "instructions": "Select the candidate that best matches the supplied context.",
                "choices": [{"value": choice} for choice in choices],
            }
        if capability == DecisionTask.SCORE:
            scale = request.scale if request.scale is not None else [0.0, 1.0]
            if (
                len(scale) != 2
                or any(not math.isfinite(v) for v in scale)
                or scale[0] >= scale[1]
                or not math.isfinite(scale[1] - scale[0])
            ):
                raise InvalidRequestError(
                    "Score requires a finite scale [minimum, maximum] with minimum < maximum."
                )
            if request.hypothesis is not None or request.choices is not None:
                raise InvalidRequestError("Score does not accept hypothesis/choice fields.")
            lo, hi = scale
            values = [
                lo + (hi - lo) * (index / (self.score_levels - 1))
                for index in range(self.score_levels)
            ]
            if any(not math.isfinite(v) for v in values) or len(set(values)) != self.score_levels:
                raise InvalidRequestError(
                    "Scale cannot represent the selected native score levels safely."
                )
            return {
                "type": "score",
                "name": "decision",
                "instructions": f"Rate the context against the ordered scale from {lo} to {hi}.",
                "levels": [{"label": format(value, ".17g")} for value in values],
            }
        raise InvalidRequestError("Unsupported canonical decision capability.")

    def required_capabilities(self, request, capability):
        if not hasattr(request, "questions"):
            return {capability}
        return {q.capability for q in request.questions}

    def _public_questions(self, request):
        if not request.input.strip() or not 1 <= len(request.questions) <= 400:
            raise InvalidRequestError("Expected nonempty text and 1–400 questions.")
        questions = []
        for question in request.questions:
            if not question.instructions.strip():
                raise InvalidRequestError("Question instructions must be nonempty text.")
            data = question.model_dump(exclude_none=True)
            data["name"] = question.name
            if question.type == "choice":
                choices = question.choices or []
                if (
                    question.levels is not None
                    or not 2 <= len(choices) <= self.max_choices
                    or any(
                        set(c) - {"value", "description"}
                        or not isinstance(c.get("value"), str)
                        or not c["value"].strip()
                        or "description" in c
                        and not isinstance(c["description"], str)
                        for c in choices
                    )
                    or len({c["value"] for c in choices}) != len(choices)
                ):
                    raise InvalidRequestError("Invalid native choice candidates.")
            elif question.type == "score":
                levels = question.levels or []
                if (
                    question.choices is not None
                    or not 2 <= len(levels) <= self.score_levels
                    or any(
                        set(v) != {"label"}
                        or not isinstance(v["label"], str)
                        or not v["label"].strip()
                        for v in levels
                    )
                ):
                    raise InvalidRequestError("Score requires 2–10 ordered labels.")
            elif question.choices is not None or question.levels is not None:
                raise InvalidRequestError("Predicate does not accept choices or levels.")
            questions.append(data)
        return questions

    def validate_request(self, model, request, capability):
        self._public_questions(request) if hasattr(request, "questions") else self._question(
            request, capability
        )

    def prepare(self, context, request, capability):
        if hasattr(request, "questions"):
            questions = self._public_questions(request)
            return PreparedCall(
                capability,
                {"model": context.binding.model, "input": request.input, "questions": questions},
                {
                    "questions": questions,
                    "capabilities": [q.capability for q in request.questions],
                    "public": True,
                },
            )
        question = self._question(request, capability)
        metadata = {"question": question}
        if question["type"] == "score":
            metadata["scale"] = request.scale if request.scale is not None else [0.0, 1.0]
        return PreparedCall(
            capability,
            {"model": context.binding.model, "input": request.context, "questions": [question]},
            metadata,
        )

    def infer(self, context, prepared):
        # One selected protocol. An unavailable Decisions endpoint fails closed;
        # it does not fall back to chat, lexical heuristics or another model.
        return context.post_json("/v1/decisions", prepared.payload)

    @staticmethod
    def _distribution(probabilities):
        if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Native probabilities do not sum to one")

    def normalize_output(self, context, prepared, output):
        try:
            if prepared.metadata.get("public"):
                response = NativeDecisionResponse.model_validate(output)
                questions = prepared.metadata["questions"]
                if len(response.answers) != len(questions):
                    raise ValueError("Wrong answer count")
                for answer, question, capability in zip(
                    response.answers, questions, prepared.metadata["capabilities"]
                ):
                    single = {**output, "answers": [answer.model_dump()]}
                    self.normalize_output(
                        context,
                        PreparedCall(capability, {}, {"question": question, "scale": [0.0, 1.0]}),
                        single,
                    )
                return response
            response = NativeDecisionResponse.model_validate(output)
            if response.model != context.binding.model or len(response.answers) != 1:
                raise ValueError("Native response changed model or question count")
            answer, question = response.answers[0], prepared.metadata["question"]
            if answer.name != question["name"] or answer.type not in (question["type"], "refusal"):
                raise ValueError("Native response changed question identity/type")
            usage = response.usage
            if (
                usage.total_tokens != usage.input_tokens + usage.output_tokens
                or usage.input_tokens_details.cached_tokens > usage.input_tokens
                or usage.input_tokens_details.cache_write_tokens > usage.input_tokens
                or usage.output_tokens_details.reasoning_tokens > usage.output_tokens
            ):
                raise ValueError("Native token usage is inconsistent")
            mapping = None
            if answer.type == "choice":
                expected = [choice["value"] for choice in question["choices"]]
                values = [row.value for row in answer.probabilities]
                if (
                    len(values) != len(expected)
                    or set(values) != set(expected)
                    or answer.choice not in expected
                ):
                    raise ValueError("Native choice candidates are missing, duplicated or changed")
                probabilities = {row.value: row.probability for row in answer.probabilities}
                self._distribution(probabilities.values())
                winner = max(probabilities.values())
                if (
                    not math.isclose(probabilities[answer.choice], winner, abs_tol=1e-6)
                    or answer.confidence > winner + 1e-6
                ):
                    raise ValueError("Native choice/confidence disagrees with its distribution")
            elif answer.type == "score":
                rows, levels = answer.probabilities, question["levels"]
                if len(rows) != len(levels) or {row.value for row in rows} != set(
                    range(len(levels))
                ):
                    raise ValueError("Native score levels are missing, duplicated or changed")
                if any(row.label != levels[row.value]["label"] for row in rows):
                    raise ValueError("Native score level labels changed")
                self._distribution(row.probability for row in rows)
                expected = math.fsum(row.value * row.probability for row in rows)
                if (
                    not math.isclose(answer.score, expected, rel_tol=1e-7, abs_tol=1e-6)
                    or not 0 <= answer.score <= len(levels) - 1
                    or answer.confidence > max(row.probability for row in rows) + 1e-6
                ):
                    raise ValueError("Native score/confidence disagrees with its distribution")
                lo, hi = prepared.metadata["scale"]
                mapped = lo + (hi - lo) * (answer.score / (len(levels) - 1))
                if not math.isfinite(mapped) or not lo <= mapped <= hi:
                    raise ValueError("Derived numeric score is invalid")
                mapping = ScoreMapping(minimum=lo, maximum=hi, value=mapped)
            return DecisionResult(
                model=response.model,
                task=prepared.capability,
                answer=answer,
                usage=usage,
                score_mapping=mapping,
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise ModelUnavailableError(
                "Native decision result failed validation; no probabilities, usage or calibration were fabricated."
            ) from exc

    def backend_environment(self, backend):
        """Stage a selected, hash-attested native plugin; no Engine core change."""
        import hashlib, os
        from pathlib import Path

        root = backend.directory / "native_plugins"
        root.mkdir()
        name = "factory_qwen2_decision"
        metadata = root / (name + "-1.0.dist-info")
        metadata.mkdir()
        files = {
            root / (name + ".py"): Path(__file__).with_name("vllm_plugin.py").read_bytes(),
            metadata
            / "METADATA": b"Metadata-Version: 2.1\nName: factory-qwen2-decision\nVersion: 1.0\n",
            metadata
            / "entry_points.txt": b"[vllm.general_plugins]\nfactory_qwen2_decision = factory_qwen2_decision:register\n",
        }
        for target, data in files.items():
            target.write_bytes(data)
            backend.adapter_assets[str(target.relative_to(backend.directory))] = hashlib.sha256(
                data
            ).hexdigest()
            stat = target.stat()
            backend._files[str(target)] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        return {
            "VLLM_PLUGINS": name,
            "PYTHONPATH": str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
