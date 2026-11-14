# Decision service migration

The service delegates to canonical `factory.engines.decision.DecisionEngine`. ModelManifest selects the model adapter; deployment selects the backend, GPU devices and port before serving. The Engine has no backend protocol, environment URL routing or list of model names.

Current state: **software implementation only, no production-authorized decision capability**. No Decision GPU inference, semantic validation, Factory E2E or official SDK acceptance has run during this migration.

## Engine and adapter

- The existing service ingress normalizes `predicate`, `choice` and `score` once into canonical capability IDs. Internal requests accept canonical IDs only.
- `factory.adapters.qwen_decision.Adapter` maps the retained single-task request to the selected native `/v1/decisions` protocol. Failure does not try chat, another backend or another checkpoint.
- Predicate preserves native `probability`. It does not invent a Boolean threshold, confidence or token logprobs.
- Choice preserves native `choice`, typed candidate probabilities and backend confidence. The adapter rejects missing/duplicate candidates, inconsistent distributions and invalid winners.
- Score preserves the native expected zero-based level index and per-level probabilities. A legacy request's numeric `scale` produces a separate `score_mapping` with method `linear_map_of_native_expected_level`; `answer.score` remains unchanged. This transformation is not calibration.
- Genuine usage and refusals are preserved. Missing usage, malformed probabilities, changed model/question identity or fabricated extra fields fail closed. ECE/Brier metrics require a separate labeled evaluation and are not inference response defaults.

The native answer schema is described in the [vLLM Decisions documentation](https://docs.vllm.ai/en/latest/models/structured_decisions_models/). Confidence is not an application calibration measurement.

## Public API boundary

The retained service endpoint still accepts the migration request (`model`, `task`, `context`, plus task-specific fields). Its current session must authorize a capability before inference. Model/backend health alone cannot authorize it.

OpenAI now defines [POST /v1/decisions](https://developers.openai.com/api/reference/resources/decisions/methods/create). The final public endpoint must follow that contract under Phase 6. [PLAN_FINDINGS.md](../../docs/PLAN_FINDINGS.md) records the plan's outdated custom-endpoint classification and the proposed compatibility migration. No public compatibility gate has been passed.

## Retained migration code

`legacy_engine.py` and the old `vllm_adapter.py` remain for explicit `contract_mock` historical tests. The service application imports the canonical engine. Those fixtures, proxy responses and artifact existence cannot certify model inference.
