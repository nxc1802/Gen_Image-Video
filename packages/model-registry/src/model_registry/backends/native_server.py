"""Internal adapter-hosted native SDK runtime. No fabricated output.

Phase 4 proves process/load/health, not semantic inference. Phase 5 adapters own
the inference operations; V3/V4 gates own production authorization.
"""
import argparse
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import secrets


def load_sdk(config):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    dtype = {"fp16":torch.float16, "bf16":torch.bfloat16, "fp32":torch.float32}[config["precision"]]
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected BF16 precision is unsupported on this GPU; no precision/model fallback")
    torch.cuda.set_device(0)
    torch.cuda.init()
    from factory.core.model_adapter import load_adapter
    adapter=load_adapter(config["adapter"])
    if config["sdk"] not in adapter.supported_backends:
        raise RuntimeError("Native adapter/backend identity mismatch")
    return adapter.load_native(config,dtype)


def create_app(config):
    from fastapi import FastAPI, HTTPException, Request
    token = Path(config["token_file"]).read_text().strip()
    from factory.core.model_adapter import load_adapter
    from threading import Lock
    adapter=load_adapter(config["adapter"])
    inference_lock=Lock()
    @asynccontextmanager
    async def lifespan(app):
        app.state.loaded = load_sdk(config)
        yield
        app.state.loaded = None
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    @app.get("/_factory/health")
    def health(request: Request):
        authorization = request.headers.get("authorization", "")
        if not secrets.compare_digest(authorization, "Bearer " + token):
            raise HTTPException(401)
        if getattr(app.state, "loaded", None) is None:
            raise HTTPException(503)
        return {"model": config["model"], "boot_nonce": config["boot_nonce"],
            "checkpoint_graph_sha256": config["checkpoint_graph_sha256"], "model_loaded": True,
            "sdk_runtime": getattr(app.state.loaded, "_factory_runtime_identity", None),
            "inference_verified": False, "production_ready": False}
    @app.post("/_factory/infer")
    async def infer(request: Request):
        if not secrets.compare_digest(request.headers.get("authorization",""),"Bearer "+token):
            raise HTTPException(401)
        body=await request.json()
        if body.get("model")!=config["model"] or body.get("capability") not in config["capabilities"] or not isinstance(body.get("inputs"),dict):
            raise HTTPException(400,"Request does not match selected model/capability")
        import asyncio
        def execute():
            with inference_lock:
                return adapter.native_infer(app.state.loaded,body["capability"],body["inputs"],config)
        try: return await asyncio.to_thread(execute)
        except Exception as exc: raise HTTPException(503,"Selected native inference failed: "+type(exc).__name__) from exc
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    import uvicorn
    config = json.loads(args.config.read_text())
    uvicorn.run(create_app(config), host="127.0.0.1", port=args.port, log_level="info", workers=1)


if __name__ == "__main__":
    main()
