# Declarative Manifests Specification & Registry

## 1. Overview

The `manifests/` directory serves as the authoritative, declarative Single Source of Truth (SSoT) for Factory v4. It contains declarative YAML configurations specifying models, services, capabilities, and hardware targets. All manifests are validated by Pydantic v2 models in `packages/contracts/src/factory_contracts/manifests/` and dynamically checked at startup and during CI/CD.

---

## 2. Manifest Directory Structure

```text
manifests/
├── capabilities/         # Standardized capability specifications (12 files)
│   ├── image.t2i.yaml
│   ├── video.t2v.yaml
│   ├── qwen-speech.yaml
│   └── ...
├── hardware/             # Physical and virtual hardware profiles (3 files)
│   ├── kaggle-dual-t4.yaml
│   ├── colab-single-t4.yaml
│   └── kaggle-dual-t4.yaml
├── models/               # Model manifests (12 files)
│   ├── qwen-text.yaml
│   ├── qwen-image-2.1.yaml
│   ├── qwen-image-2.1-turbo.yaml  # Explicit 8-step fast inference manifest
│   ├── wan-2.2.yaml
│   ├── minimax-h3.yaml
│   └── ...
└── services/             # Service daemon manifests (11 files)
    ├── text.yaml
    ├── image.yaml
    ├── video.yaml
    └── ...
```

---

## 3. Schema & Validation Invariants

### 3.1 Model Completeness Invariants (`ModelManifest`)
- **Status `full`**:
  - Requires `coverage_percent == 100.0`.
  - Requires `missing_capabilities == []`.
  - Requires all entries in `capabilities` to evaluate to `true`.
- **Status `partial`**:
  - Requires at least one capability in `capabilities` evaluating to `false`.
  - Requires `coverage_percent < 100.0`.
  - Requires non-empty `missing_capabilities` listing every unsupported task.

### 3.2 Weights & Provenance Requirements
Every model manifest must declare genuine checkpoint provenance:
- `primary_checkpoint`: Hugging Face repo path or local checkpoint identifier.
- `repository`: Canonical Hugging Face or ModelScope repository.
- `checkpoint_revision`: Exact Git commit SHA (at least 7 characters) or immutable release tag.
- `weights_file`: Exact filename of the primary model weights (e.g. `*.gguf`, `*.safetensors`).
- `license`: Explicit legal license (e.g. `Tongyi Qianwen License`, `MiniMax Non-Commercial License`).
- `commercial_use`: Explicit boolean (`true` or `false`). Generic undeclared MIT licenses on proprietary model weights fail validation.

### 3.3 Hardware Constraints & Discrete GPU Allocation
Hardware specs must fit within target physical envelopes:
- On `kaggle-dual-t4`: Allocatable ceiling is **14.2 GB (14,540 MB)** per GPU.
- `preferred_gpu_index`: Must be discrete device `0` or `1`.
- `allocation_policy`: `single_gpu`, `dual_gpu_split`, or `offload_ram`.

---

## 4. Validating Manifests via CLI

Validate all manifests across the repository at any time:

```bash
# Validate using Python module CLI
python -m model_registry.cli validate --path manifests

# Or run contract test suite
pytest tests/contract/test_manifest_schemas.py -v
pytest tests/unit/manifests/test_manifest_coverage.py -v
```

Expected output on success:
```text
✓ Validated 11 services
✓ Validated 3 hardware topologies
✓ Validated 12 capabilities
✓ Validated 12 model manifests
Manifest validation: SUCCESS (0 errors)
```
