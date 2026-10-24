"""
Compatibility authenticated reverse proxy for an internal ComfyUI image daemon.

Factory production exposes native ComfyUI directly behind the ngrok raw-API
policy. This gateway remains for deployments that explicitly choose an
application-auth proxy. It never fabricates upstream state or success.
"""
import asyncio
import os
import secrets
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect

try:
    import websockets
except ImportError:
    websockets = None

from factory_auth import decode_jwt_token, load_auth_config, verify_bearer_token
from factory.apps.control_api.src.api.errors import register_exception_handlers
from factory.apps.control_api.src.api.middleware import RequestIDMiddleware
from services.image.config import ImageServiceConfig, load_image_config

app = FastAPI(title="ComfyUI Image Compatibility Gateway", version="1.0.0")
app.add_middleware(RequestIDMiddleware)
register_exception_handlers(app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get("FACTORY_CORS_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config: ImageServiceConfig = load_image_config()


def _internal(path: str) -> str:
    return f"http://{config.comfy_host}:{config.internal_comfy_port}{path}"


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(_internal("/system_stats"))
            response.raise_for_status()
        return {"status": "ok", "service": "image-gateway", "upstream": "ready"}
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Internal ComfyUI service unavailable.") from exc


@app.get("/system_stats", dependencies=[Depends(verify_bearer_token)])
async def system_stats():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(_internal("/system_stats"))
            response.raise_for_status()
            return JSONResponse(status_code=response.status_code, content=response.json())
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="Internal ComfyUI service unavailable.") from exc


@app.post("/prompt", dependencies=[Depends(verify_bearer_token)])
async def proxy_prompt(request: Request):
    try:
        body = await request.json()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(_internal("/prompt"), json=body)
            response.raise_for_status()
            return JSONResponse(status_code=response.status_code, content=response.json())
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Internal ComfyUI service unavailable.") from exc


@app.post("/interrupt", dependencies=[Depends(verify_bearer_token)])
async def proxy_interrupt():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(_internal("/interrupt"))
            response.raise_for_status()
            if response.content:
                try:
                    return JSONResponse(status_code=response.status_code, content=response.json())
                except ValueError:
                    pass
            return {"status": "interrupted"}
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Internal ComfyUI service unavailable.") from exc


def _ws_token(websocket: WebSocket) -> Optional[str]:
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    protocols = [
        item.strip()
        for item in websocket.headers.get("sec-websocket-protocol", "").split(",")
        if item.strip()
    ]
    protocol = next((p for p in protocols if p.startswith("factory-bearer.")), None)
    return protocol[len("factory-bearer."):] if protocol else None


@app.websocket("/ws")
async def websocket_proxy(websocket: WebSocket):
    auth_config = load_auth_config()
    token = _ws_token(websocket)
    authenticated = bool(token and secrets.compare_digest(token, auth_config.api_key))
    if not authenticated and token and auth_config.jwt_secret and token.count(".") == 2:
        try:
            decode_jwt_token(token, auth_config.jwt_secret)
            authenticated = True
        except Exception:
            authenticated = False
    if not authenticated:
        await websocket.close(code=1008)
        return
    if websockets is None:
        await websocket.close(code=1011)
        return

    await websocket.accept()
    target_ws = f"ws://{config.comfy_host}:{config.internal_comfy_port}/ws?clientId=factory-gateway"
    try:
        async with websockets.connect(target_ws, ping_interval=20, ping_timeout=20) as backend_ws:
            async def to_client():
                async for message in backend_ws:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)

            async def to_backend():
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        return
                    if message.get("bytes") is not None:
                        await backend_ws.send(message["bytes"])
                    elif message.get("text") is not None:
                        await backend_ws.send(message["text"])

            await asyncio.gather(to_client(), to_backend())
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
