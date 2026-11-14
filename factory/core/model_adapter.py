"""Adapter contract and exact-session backend transport. Never authorizes READY."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
import re
from typing import Any

import httpx

from factory_contracts.capabilities import require_canonical
from factory_contracts.errors import ModelUnavailableError, UnsupportedCapabilityError


@dataclass(frozen=True)
class PreparedCall:
    capability: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionContext:
    registry: Any
    session: Any
    binding: Any
    model: Any

    @property
    def backend(self):
        if self.session is None or self.session.registry is not self.registry:
            raise ModelUnavailableError(
                "Inference requires this exact deployment's backend session."
            )
        backend = self.session.backends.get(self.binding.model)
        if (
            backend is None
            or backend.registry is not self.registry
            or backend.binding != self.binding
        ):
            raise ModelUnavailableError(
                "Selected backend does not belong to the configured model/session."
            )
        return backend

    def post_json(self, path, payload, *, timeout=60):
        # Paths are adapter-owned constants. Requests cannot select a host, port,
        # backend, GPU or checkpoint, nor redirect the private backend credential.
        if not path.startswith("/") or path.startswith("//") or "://" in path:
            raise ValueError("Backend transport requires an adapter-owned local path")
        if payload.get("model") != self.binding.model:
            raise ModelUnavailableError(
                "Adapter payload attempted to change the admitted model identity."
            )
        backend = self.backend
        if not backend.health():
            raise ModelUnavailableError("Selected backend lost current-session health.")
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{self.binding.port}",
                timeout=timeout,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                response = client.post(
                    path, json=payload, headers={"Authorization": "Bearer " + backend.token}
                )
                if response.status_code != 200:
                    raise ModelUnavailableError(
                        "Selected backend rejected inference.",
                        details={"status_code": response.status_code},
                    )
                body = response.json()
            if not isinstance(body, dict):
                raise ValueError("Backend JSON must be an object")
            return body
        except ModelUnavailableError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                "Selected backend returned no valid inference result.",
                details={"error_type": type(exc).__name__},
            ) from exc

    def post_bytes(self, path, payload):
        if (
            payload.get("model") != self.binding.model
            or not path.startswith("/")
            or path.startswith("//")
        ):
            raise ModelUnavailableError("Invalid bound backend request.")
        backend = self.backend
        if not backend.health():
            raise ModelUnavailableError("Selected backend lost health.")
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{self.binding.port}",
                timeout=600,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                response = client.post(
                    path, json=payload, headers={"Authorization": "Bearer " + backend.token}
                )
                if response.status_code != 200 or not response.content:
                    raise ModelUnavailableError("Native backend returned no media.")
                return response.content
        except httpx.HTTPError as exc:
            raise ModelUnavailableError("Native media backend unavailable.") from exc


class ModelAdapter(ABC):
    capabilities = frozenset()
    supported_backends = frozenset()
    supported_tiling_components = frozenset()

    def resolve_capability(self, request, operation):
        return operation

    def required_capabilities(self, request, capability):
        return {capability}

    @abstractmethod
    def validate_request(self, model, request, capability): ...
    @abstractmethod
    def prepare(self, context, request, capability): ...
    @abstractmethod
    def infer(self, context, prepared): ...
    @abstractmethod
    def normalize_output(self, context, prepared, output): ...

    def health(self, context):
        return (
            context.binding.model in context.session.healthy_models() and context.backend.health()
        )


def load_adapter(name):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ModelUnavailableError("Manifest adapter name is invalid.")
    try:
        provider = import_module(f"factory.adapters.{name}").Adapter
        if not isinstance(provider, type) or not issubclass(provider, ModelAdapter):
            raise TypeError("Provider does not implement ModelAdapter")
        return provider()
    except (ImportError, AttributeError, TypeError) as exc:
        raise ModelUnavailableError(
            "The selected manifest adapter is unavailable; fallback is forbidden.",
            details={"adapter": name},
        ) from exc


class Engine:
    engine = None

    def __init__(self, registry=None, session=None, *, adapter_loader=load_adapter):
        if registry is None:
            from model_registry.deployment import DeploymentRegistry

            registry = DeploymentRegistry.from_environment()
        self.registry, self.session, self.adapter_loader = registry, session, adapter_loader

    def resolve(self, request, operation):
        binding = self.registry.assignment(request.model)
        model = self.registry.catalog.get(binding.model)
        if model.engine != self.engine:
            raise UnsupportedCapabilityError("Configured model belongs to another engine.")
        adapter = self.adapter_loader(binding.adapter)
        capability = adapter.resolve_capability(request, operation)
        require_canonical(capability, engine=self.engine)
        required = adapter.required_capabilities(request, capability)
        for cid in required | {capability}:
            require_canonical(cid, engine=self.engine)
            if not model.capabilities.get(cid) or cid not in adapter.capabilities:
                raise UnsupportedCapabilityError(
                    "The selected model/adapter does not implement this capability.",
                    details={"capability": cid},
                )
        if binding.backend not in adapter.supported_backends:
            raise ModelUnavailableError(
                "Adapter does not support the explicitly selected backend; no alternate implementation was selected."
            )
        adapter.validate_request(model, request, capability)
        return binding, model, adapter, capability, required

    def prepare_execution(self, request, operation):
        binding, model, adapter, capability, required = self.resolve(request, operation)
        # There is no bypass boolean here. Native validation can exercise an
        # adapter internally, but public engine execution requires authorization.
        for cid in required | {capability}:
            self.registry.require_active(binding.model, cid)
        context = ExecutionContext(self.registry, self.session, binding, model)
        context.backend  # Validate session identity before invoking adapter hooks.
        if not adapter.health(context):
            raise ModelUnavailableError("Configured backend is not currently healthy.")
        prepared = adapter.prepare(context, request, capability)
        if prepared.capability != capability:
            raise UnsupportedCapabilityError("Adapter preparation changed the admitted capability.")
        return context, adapter, prepared

    def execute(self, request, operation):
        context, adapter, prepared = self.prepare_execution(request, operation)
        from factory.core.execution import execution_scope

        with execution_scope(context):
            return adapter.normalize_output(context, prepared, adapter.infer(context, prepared))

    async def execute_async(self, request, operation, *, on_started=None):
        """Canonical local-job execution; no worker lease or request routing."""
        import asyncio
        from factory.core.execution import async_execution_scope, finish_native_call

        context, adapter, prepared = await asyncio.to_thread(
            self.prepare_execution, request, operation
        )
        async with async_execution_scope(context):
            if on_started is not None:
                on_started()

            def infer():
                return adapter.normalize_output(context, prepared, adapter.infer(context, prepared))

            return await finish_native_call(infer)
