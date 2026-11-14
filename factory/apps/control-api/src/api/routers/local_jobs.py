"""Owner-scoped management for the single-host Engine job runtime."""

import asyncio
import json
from fastapi import APIRouter, Depends, Request, Query
from fastapi.responses import Response, StreamingResponse
from factory_auth import verify_bearer_token
from factory.apps.control_api.src.api.public.jobs import owner

router = APIRouter(
    prefix="/v1/jobs", tags=["Local jobs"], dependencies=[Depends(verify_bearer_token)]
)


@router.get("")
async def list_jobs(request: Request, limit: int = Query(default=50, ge=1, le=100)):
    jobs = request.app.state.public_media_jobs
    return {"object": "list", "data": jobs.list(owner(request), limit=limit)}


@router.get("/{identifier}")
async def job_status(identifier: str, request: Request):
    jobs = request.app.state.public_media_jobs
    return jobs.view(jobs.get(identifier, owner(request)), canonical=True)


@router.post("/{identifier}/cancel")
async def cancel_job(identifier: str, request: Request):
    return request.app.state.public_media_jobs.cancel(identifier, owner(request))


@router.get("/{identifier}/content")
async def job_content(identifier: str, request: Request):
    raw, mime = await request.app.state.public_media_jobs.content(identifier, owner(request))
    return Response(raw, media_type=mime)


@router.get("/{identifier}/events")
async def job_events(identifier: str, request: Request):
    jobs = request.app.state.public_media_jobs
    row = jobs.get(identifier, owner(request))

    async def events():
        sent = 0
        while True:
            for event in row["events"][sent:]:
                sent += 1
                yield (
                    "id: "
                    + str(event["sequence"])
                    + "\nevent: job.status\ndata: "
                    + json.dumps(event)
                    + "\n\n"
                )
            if row["status"] in {"completed", "failed", "cancelled"}:
                return
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.1)

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )
