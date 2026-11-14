# Legacy contract fixtures

This catalog preserves v1 shapes for existing queue, lease and preflight algorithm
tests during the ordered migration. It is selected explicitly by `legacy_catalog`
or `legacy_workspace` pytest fixtures. Readiness flags and backend results are
simulated. No fixture proves weights, native inference, GPU execution, semantic
correctness or production readiness.

Production identity and aliases live in the root `manifests/models` v2 catalog.
Canonical model truth and fail-closed admission tests always use that catalog.
Do not configure this directory as a production manifest source.

`deployment.yaml` is a contract-only routing fixture preserving old test placements.
Its combined resident models/exclusivity are intentionally not a valid production
session. Canonical deployment preflight rejects this v1 catalog. Queue tests use
these declared bindings to test software locks and transitions, not GPU residency.
