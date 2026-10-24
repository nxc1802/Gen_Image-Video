# Factory v4: Decoupled AI Generative Factory

Factory v4 is a production-grade, decoupled AI Generative Factory monorepo where a dedicated Control Plane manages authentication, job lifecycle, service/model registry, storage, and telemetry without executing inference, while autonomous subsystem workers (Text, VLM, Decision, Embedding, Speech, Image, Video, Music, Enhancement) run natively on their optimal serving frameworks behind fixed ngrok static domains.

## Architecture

- **`factory/apps/control-api/`**: FastAPI Control Plane managing authentication, registries, two-tier job queues, storage, and telemetry.
- **`factory/apps/studio-web/`**: Next.js 15 / React 19 Operations Console & API Explorer (reads dynamic schemas and capability manifests).
- **`packages/contracts/`**: Dual-language type definitions (Pydantic v2 & TypeScript) defining `ServiceManifest`, `ModelManifest`, `CapabilityManifest`, `HardwareTopologyManifest`, `UnifiedErrorResponse`, and `FactoryJob`.
- **`packages/auth/`**: Core authentication libraries and token verification.
- **`packages/model-registry/`**: Manifest loader and `factory-manifest` CLI validator.
- **`services/`**: Subsystem service daemons and launchers (vLLM, ComfyUI, ACE-Step 1.5).
- **`workflows/`**: Versioned ComfyUI workflows for image and video generation.
- **`manifests/`**: Declarative YAML manifests for services, models, and hardware topology.
- **`infrastructure/`**: ngrok edge routing, PostgreSQL schemas, Redis configurations, MinIO S3 setup.
- **`legacy/v3/`**: Archived Kaggle AI Studio v3.5 runtime and artifacts.
