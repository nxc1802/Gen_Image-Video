"""
factory/apps/control-api/src/api/routers/inspector.py
FastAPI router for Studio Operations Console live telemetry inspector (/api/studio/inspect/stream),
schema discovery (/api/studio/schemas), and executable client code generation (/api/studio/codegen).
"""
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import StreamingResponse

from factory_auth import verify_bearer_token
from model_registry.loader import ManifestLoader
from factory.apps.control_api.src.streaming.inspector import stream_studio_inspector

router = APIRouter(
    prefix="/api/studio",
    tags=["Studio Inspector"],
    dependencies=[Depends(verify_bearer_token)],
)


def _get_manifest_loader(request: Request) -> ManifestLoader:
    loader = getattr(request.app.state, "manifest_loader", None)
    if loader is not None:
        return loader
    if Path("manifests").exists():
        return ManifestLoader("manifests")
    repo_manifests = Path(__file__).resolve().parents[6] / "manifests"
    if repo_manifests.exists():
        return ManifestLoader(repo_manifests)
    return ManifestLoader()


def _dump_service(s: Any) -> Dict[str, Any]:
    d = s.model_dump(mode="json") if hasattr(s, "model_dump") else dict(s)
    for k, v in d.get("metadata", {}).items():
        if k not in d:
            d[k] = v
    return d


def _dump_model(m: Any) -> Dict[str, Any]:
    d = m.model_dump(mode="json") if hasattr(m, "model_dump") else dict(m)
    for k, v in d.get("metadata", {}).items():
        if k not in d:
            d[k] = v
    return d


@router.get("/schemas")
async def get_studio_schemas(request: Request):
    """
    Returns active services and models inventory for dynamic Studio Web forms.
    """
    loader = _get_manifest_loader(request)
    services = loader.load_services()
    models = loader.load_models()
    return {
        "schemas": {
            "services": [_dump_service(s) for s in services.values()],
            "models": [_dump_model(m) for m in models.values()],
        }
    }


@router.post("/codegen")
async def studio_codegen(payload: Optional[Dict[str, Any]] = Body(default_factory=dict)):
    """
    Generates production-grade client code snippets (cURL, Python httpx, TypeScript fetch)
    for submitting jobs with the provided service, model, task, and inputs.
    """
    payload = payload or {}
    service = payload.get("service") or "image"
    model = payload.get("model") or "qwen-image-2.1"
    task = payload.get("task") or "text_to_image"
    inputs = payload.get("inputs") or {}

    job_payload = {
        "service": service,
        "model": model,
        "task": task,
        "inputs": inputs,
    }
    body_json = json.dumps(job_payload)

    curl_snippet = (
        f'curl -X POST https://jobs.ai.example.com/v1/jobs \\\n'
        f'  -H "Authorization: Bearer $FACTORY_API_KEY" \\\n'
        f'  -H "Content-Type: application/json" \\\n'
        f"  -d '{body_json}'"
    )

    python_snippet = (
        f"import httpx\n"
        f"import os\n\n"
        f'API_KEY = os.getenv("FACTORY_API_KEY", "sk-factory-...")\n\n'
        f'response = httpx.post(\n'
        f'    "https://jobs.ai.example.com/v1/jobs",\n'
        f'    headers={{"Authorization": f"Bearer {{API_KEY}}", "Content-Type": "application/json"}},\n'
        f'    json={json.dumps(job_payload, indent=4)},\n'
        f')\n'
        f'job = response.json()\n'
        f'print(f"Submitted job: {{job.get(\'job_id\')}}")\n'
    )

    typescript_snippet = (
        f'const apiKey = process.env.FACTORY_API_KEY || "sk-factory-...";\n\n'
        f'const response = await fetch("https://jobs.ai.example.com/v1/jobs", {{\n'
        f'  method: "POST",\n'
        f'  headers: {{\n'
        f'    "Authorization": `Bearer ${{apiKey}}`,\n'
        f'    "Content-Type": "application/json",\n'
        f'  }},\n'
        f'  body: JSON.stringify({json.dumps(job_payload, indent=4)}),\n'
        f'}});\n'
        f'const job = await response.json();\n'
        f'console.log("Submitted job:", job.job_id);\n'
    )

    return {
        "curl": curl_snippet,
        "python": python_snippet,
        "typescript": typescript_snippet,
    }


@router.get("/inspect/stream")
async def studio_telemetry_stream():
    """
    Live Server-Sent Events stream for Studio Web telemetry gauges.
    Emits 'telemetry' events with GPU VRAM, queue depth, active jobs, and keepalive 'ping'.
    Handles disconnection cleanly without errors.
    """
    async def _safe_stream():
        try:
            async for chunk in stream_studio_inspector():
                yield chunk
                if "event: telemetry" in chunk:
                    yield chunk.replace("event: telemetry", "event: metrics", 1)
        except (asyncio.CancelledError, GeneratorExit):
            return

    return StreamingResponse(
        _safe_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
