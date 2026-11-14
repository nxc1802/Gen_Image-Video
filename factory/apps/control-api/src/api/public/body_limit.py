"""Bound actual public request bytes before FastAPI/Starlette decoders allocate."""

import json
from . import MAX_BODY


class PublicBodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        public = path in {
            "/v1/responses",
            "/v1/chat/completions",
            "/v1/embeddings",
            "/v1/decisions",
            "/v1/rerank",
            "/v1/music",
            "/v1/enhancements",
        } or path.startswith(("/v1/audio/", "/v1/images/", "/v1/videos"))
        if scope["type"] != "http" or scope.get("method") != "POST" or not public:
            return await self.app(scope, receive, send)
        records = []
        total = 0
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            total += len(event.get("body", b""))
            if total > MAX_BODY:
                body = json.dumps(
                    {
                        "error": {
                            "message": "Request body exceeds 64 MiB.",
                            "type": "invalid_request_error",
                            "param": None,
                            "code": "PAYLOAD_TOO_LARGE",
                        }
                    }
                ).encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
            records.append(event)
            if not event.get("more_body", False):
                break
        iterator = iter(records)

        async def replay():
            try:
                return next(iterator)
            except StopIteration:
                return await receive()

        return await self.app(scope, replay, send)
