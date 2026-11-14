"""Owned ComfyUI execution: exact manifest selectors, real history and bytes."""

from pathlib import PurePosixPath
import time
import uuid
import httpx
from factory.core.model_adapter import ModelAdapter
from factory.core.media import data_bytes, artifact, check_parameters
from factory_contracts.errors import InvalidRequestError, ModelUnavailableError


def checkpoint(model, key):
    spec = model.weights.files[key]
    return PurePosixPath(spec.destination or spec.path).name


class ComfyAdapter(ModelAdapter):
    supported_backends = frozenset({"comfyui"})

    def validate_request(self, model, request, capability):
        check_parameters(model, request.inputs, getattr(self, "allowed_inputs", None))

    def client(self, context):
        if not context.backend.health():
            raise ModelUnavailableError("Owned ComfyUI daemon is unhealthy.")
        return httpx.Client(
            base_url=f"http://127.0.0.1:{context.binding.port}",
            timeout=30,
            trust_env=False,
            follow_redirects=False,
            headers={"Authorization": "Bearer " + context.backend.token},
        )

    def stage(self, context, value):
        data = data_bytes(value, "image")
        from PIL import Image
        import io

        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
        except Exception as exc:
            raise InvalidRequestError("Input image cannot be decoded.") from exc
        with self.client(context) as client:
            response = client.post(
                "/upload/image",
                files={"image": (uuid.uuid4().hex + ".png", data, "image/png")},
                data={"type": "input", "overwrite": "false"},
            )
            if response.status_code != 200:
                raise ModelUnavailableError("ComfyUI input staging failed.")
            body = response.json()
        path = PurePosixPath(body.get("subfolder", "")) / body["name"]
        if path.is_absolute() or ".." in path.parts:
            raise ModelUnavailableError("Invalid staged input identity.")
        return str(path)

    def infer(self, context, prepared):
        graph = prepared.payload["graph"]
        allowed = {checkpoint(context.model, key) for key in context.model.weights.files}
        selectors = {"unet_name", "clip_name", "vae_name", "lora_name", "model_name", "ckpt_name"}
        for node in graph.values():
            for key, value in node["inputs"].items():
                if key in selectors and value not in allowed:
                    raise ModelUnavailableError(
                        "Workflow attempted an unpinned checkpoint selector."
                    )
        with self.client(context) as client:
            info = client.get("/object_info")
            if info.status_code != 200:
                raise ModelUnavailableError("Cannot inspect selected backend nodes.")
            installed = info.json()
            if any(node["class_type"] not in installed for node in graph.values()):
                raise ModelUnavailableError(
                    "Selected backend is missing a required adapter node; no graph fallback."
                )
            response = client.post("/prompt", json={"prompt": graph, "client_id": uuid.uuid4().hex})
            if response.status_code != 200:
                raise ModelUnavailableError("Native workflow rejected.")
            prompt_id = response.json().get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise ModelUnavailableError("Native workflow has no execution ID.")
            deadline = time.monotonic() + context.model.runtime_profiles[context.registry.config.hardware].inference_timeout_seconds
            while time.monotonic() < deadline:
                response = client.get("/history/" + prompt_id)
                if response.status_code != 200:
                    raise ModelUnavailableError("Native history unavailable.")
                entry = response.json().get(prompt_id)
                if entry:
                    status = entry.get("status", {})
                    if status.get("status_str") == "error":
                        raise ModelUnavailableError("Native workflow execution failed.")
                    if status.get("completed") is True:
                        if not entry.get("outputs"):
                            raise ModelUnavailableError(
                                "Native workflow completed without outputs."
                            )
                        return {
                            "model": context.binding.model,
                            "prompt_id": prompt_id,
                            "outputs": entry["outputs"],
                            "native_history": entry,
                        }
                time.sleep(0.25)
        raise ModelUnavailableError("Native workflow timed out; no artifact fabricated.")

    def normalize_output(self, context, prepared, output):
        if output.get("model") != context.binding.model:
            raise ModelUnavailableError("Workflow identity changed.")
        results = []
        with self.client(context) as client:
            for node in output.get("outputs", {}).values():
                for field, kind, mime in [
                    ("images", "image", "image/png"),
                    ("videos", "video", "video/mp4"),
                    ("gifs", "video", "video/mp4"),
                    ("audio", "audio", "audio/wav"),
                ]:
                    for record in node.get(field, []):
                        filename = PurePosixPath(record["filename"])
                        sub = PurePosixPath(record.get("subfolder", ""))
                        if (
                            filename.is_absolute()
                            or len(filename.parts) != 1
                            or sub.is_absolute()
                            or ".." in sub.parts
                            or record.get("type", "output") != "output"
                        ):
                            raise ModelUnavailableError("Invalid native output location.")
                        response = client.get(
                            "/view",
                            params={
                                "filename": str(filename),
                                "subfolder": str(sub),
                                "type": "output",
                            },
                        )
                        if response.status_code != 200:
                            raise ModelUnavailableError("Native output bytes unavailable.")
                        results.append(artifact(response.content, kind, mime))
        if not results:
            raise ModelUnavailableError("No native media artifacts returned.")
        return {"model": context.binding.model, "artifacts": results}
