"""Bounded in-process public media jobs over canonical Engines and scratch storage.

IDs/time/status describe Factory's actual job lifecycle, never fabricated native
model output. Legacy worker queues are retained for the ordered Phase 7 migration.
"""

import asyncio
import hashlib
import json
import subprocess
import math
import time
import uuid
from factory_contracts.errors import (
    FactoryError,
    NotFoundError,
    ConflictError,
    ModelUnavailableError,
)
from factory.core.media import check_media
from . import admit


class PublicMediaJobs:
    def __init__(self, scratch):
        self.scratch = scratch
        self.rows = {}
        self.tasks = set()
        self.closed = False
        self.by_id = {}

    def get(self, identifier, owner, kind=None):
        row = self.rows.get(identifier)
        if row is None or row["owner"] != owner or kind is not None and row["kind"] != kind:
            raise NotFoundError("Media job not found.")
        if row["expires_at"] is not None and row["expires_at"] <= time.time():
            self.rows.pop(identifier, None)
            raise NotFoundError("Media job expired.")
        return row

    async def submit(self, instance, call, owner, kind, metadata):
        if self.closed:
            raise ModelUnavailableError("Server is shutting down.")
        await asyncio.to_thread(admit, instance, call, call.capability)
        if self.closed:
            raise ModelUnavailableError("Server is shutting down.")
        now = int(time.time())
        for identifier, row in list(self.rows.items()):
            if row["expires_at"] is not None and row["expires_at"] <= now:
                self.rows.pop(identifier, None)
        if len(self.tasks) >= 32 or len(self.rows) >= 256:
            raise ConflictError("Public media job capacity reached.")
        identifier = kind + "_" + uuid.uuid4().hex
        row = {
            "id": identifier,
            "object": kind,
            "model": instance.registry.assignment(call.model).model,
            "status": "queued",
            "progress": 0,
            "created_at": now,
            "completed_at": None,
            "expires_at": None,
            "error": None,
            **metadata,
            "owner": owner,
            "kind": kind,
            "files": [],
            "events": [],
            "cancel_requested": False,
        }
        self.rows[identifier] = row
        self._event(row)
        task = asyncio.create_task(self._run(row, instance, call))
        self.tasks.add(task)
        self.by_id[identifier] = task

        def finished(task):
            self.tasks.discard(task)
            self.by_id.pop(identifier, None)
            if task.cancelled() and row["status"] not in {"completed", "failed", "cancelled"}:
                self._cancelled(row)

        task.add_done_callback(finished)
        return self.view(row)

    async def _run(self, row, instance, call):
        try:

            def started():
                row["status"] = "in_progress"
                self._event(row)

            output = await instance.execute_async(call, call.capability, on_started=started)
            if output.get("model") != row["model"] or not output.get("artifacts"):
                raise ModelUnavailableError("Native media result has no matching output.")
            for index, item in enumerate(output["artifacts"]):
                raw = item["data"]
                kind = item["type"]
                mime = item["mime_type"]
                if (
                    not isinstance(raw, bytes)
                    or len(raw) != item["size_bytes"]
                    or hashlib.sha256(raw).hexdigest() != item["sha256"]
                ):
                    raise ModelUnavailableError("Native media output failed integrity validation.")
                if row["kind"] == "video" and (kind != "video" or mime != "video/mp4"):
                    raise ModelUnavailableError("Native video result is not an MP4.")
                await asyncio.to_thread(check_media, raw, kind)
                suffix = {"image": ".png", "audio": ".wav", "video": ".mp4"}[kind]
                filename = str(index) + suffix
                await self.scratch.save_artifact(row["id"], filename, raw, mime)
                if row["kind"] == "video":
                    path = self.scratch.get_artifact_path(row["id"], filename)
                    probe = await asyncio.to_thread(
                        subprocess.run,
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-select_streams",
                            "v:0",
                            "-show_entries",
                            "stream=width,height:format=duration",
                            "-of",
                            "json",
                            str(path),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    metadata = json.loads(probe.stdout)
                    duration = float(metadata["format"]["duration"])
                    stream = metadata["streams"][0]
                    if not math.isfinite(duration) or duration <= 0:
                        raise ModelUnavailableError("Native video duration is invalid.")
                    row["seconds"] = format(duration, ".8g")
                    row["size"] = str(stream["width"]) + "x" + str(stream["height"])
                row["files"].append(
                    {"filename": filename, "mime_type": mime, "sha256": item["sha256"]}
                )
            row.update(status="completed", progress=100, completed_at=int(time.time()))
        except asyncio.CancelledError:
            self._cancelled(row)
            raise
        except Exception as exc:
            code = exc.code if isinstance(exc, FactoryError) else "BACKEND_UNAVAILABLE"
            row.update(
                status="failed",
                error={
                    "code": code,
                    "message": "Selected backend did not produce a validated media result.",
                },
                files=[],
            )
        finally:
            row["expires_at"] = int(time.time()) + self.scratch.config.ttl_seconds
            self._event(row)

    @staticmethod
    def view(row, canonical=False):
        result = {
            k: v
            for k, v in row.items()
            if k not in {"owner", "kind", "files", "events", "cancel_requested"}
        }
        if (
            not canonical
            and row["kind"] == "video"
            and result["status"] in {"cancelled", "cancelling"}
        ):
            result["status"] = "failed" if result["status"] == "cancelled" else "in_progress"
        return result

    def _event(self, row):
        row["events"].append(
            {
                "sequence": len(row["events"]),
                "observed_at": time.time(),
                **self.view(row, canonical=True),
            }
        )

    def _cancelled(self, row):
        row.update(
            status="cancelled",
            error={
                "code": "JOB_CANCELLED",
                "message": "Media job was cancelled; no output was published.",
            },
            files=[],
            expires_at=int(time.time()) + self.scratch.config.ttl_seconds,
        )
        self._event(row)

    def list(self, owner, *, limit=50):
        now = time.time()
        rows = [
            row
            for row in self.rows.values()
            if row["owner"] == owner and (row["expires_at"] is None or row["expires_at"] > now)
        ]
        return [
            self.view(row, canonical=True)
            for row in sorted(rows, key=lambda row: row["created_at"], reverse=True)[:limit]
        ]

    def cancel(self, identifier, owner):
        row = self.get(identifier, owner)
        if row["status"] in {"completed", "failed", "cancelled"} or row["cancel_requested"]:
            return self.view(row, canonical=True)
        row["cancel_requested"] = True
        running = row["status"] == "in_progress"
        row["status"] = "cancelling" if running else "cancelled"
        if not running:
            self._cancelled(row)
        else:
            self._event(row)
        task = self.by_id.get(identifier)
        if task is not None:
            task.cancel()
        return self.view(row, canonical=True)

    async def content(self, identifier, owner, kind=None):
        row = self.get(identifier, owner, kind)
        if row["status"] != "completed" or not row["files"]:
            raise ConflictError("Media job has no completed content.")
        record = row["files"][0]
        path = self.scratch.get_artifact_path(identifier, record["filename"])
        try:
            raw = await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            raise NotFoundError("Media content expired or is unavailable.") from exc
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ModelUnavailableError("Stored media content failed integrity validation.")
        return raw, record["mime_type"]

    async def close(self, *, stop_backends=None):
        self.closed = True
        tasks = list(self.tasks)
        for identifier in list(self.by_id):
            row = self.rows[identifier]
            self.cancel(identifier, row["owner"])
        try:
            if stop_backends is not None:
                await stop_backends()
        finally:
            await asyncio.gather(*tasks, return_exceptions=True)


def owner(request):
    identity = getattr(request.state, "user", {})
    return str(identity.get("iss", "")) + ":" + str(identity.get("sub", ""))
