"""Canonical media requests and content-checked native artifacts.

Container/hash checks are software checks, never semantic inference evidence.
"""
import base64
import hashlib
import io
from pathlib import Path
import wave
from pydantic import BaseModel, ConfigDict
from factory_contracts.errors import InvalidRequestError, ModelUnavailableError


class MediaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    capability: str
    inputs: dict


def data_bytes(value, kind):
    if isinstance(value,bytes): data=value
    elif isinstance(value,str) and value.startswith("data:") and ";base64," in value:
        try: data=base64.b64decode(value.split(",",1)[1],validate=True)
        except (ValueError,TypeError) as exc: raise InvalidRequestError("Invalid inline media encoding.") from exc
    else: raise InvalidRequestError("Media must be pre-ingested inline data, never a request-supplied filesystem path or remote URL.")
    if not data or len(data)>64*1024*1024: raise InvalidRequestError("Empty or oversized media input.")
    return data


def check_media(data, kind):
    if kind=="audio":
        try:
            with wave.open(io.BytesIO(data),"rb") as audio:
                if audio.getnframes()<=0 or audio.getnchannels() not in (1,2) or audio.getframerate()<=0 or len(audio.readframes(audio.getnframes()))!=audio.getnframes()*audio.getnchannels()*audio.getsampwidth(): raise ValueError("Invalid WAV")
        except (wave.Error,EOFError,ValueError) as exc: raise ModelUnavailableError("Native audio is not a valid nonempty WAV.") from exc
    elif kind=="image":
        from PIL import Image
        try:
            with Image.open(io.BytesIO(data)) as image: image.verify()
        except Exception as exc: raise ModelUnavailableError("Native image cannot be decoded.") from exc
    elif kind=="video":
        if len(data)<32 or data[4:8]!=b"ftyp": raise ModelUnavailableError("Native output is not an MP4 container.")
        import tempfile,subprocess,json
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
                video.write(data);video.flush()
                result=subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-count_frames","-show_entries","stream=width,height,nb_read_frames","-of","json",video.name],check=True,capture_output=True,text=True,timeout=60)
            stream=json.loads(result.stdout)["streams"][0]
            if stream["width"]<=0 or stream["height"]<=0 or int(stream["nb_read_frames"])<=0: raise ValueError("No decoded frames")
        except (OSError,ValueError,KeyError,IndexError,subprocess.SubprocessError) as exc: raise ModelUnavailableError("Native video has no decodable frames.") from exc
    return data


def artifact(data,kind,mime):
    check_media(data,kind)
    return {"type":kind,"mime_type":mime,"size_bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"data":data}


def check_parameters(model, inputs, allowed=None):
    forbidden={"model","variant","checkpoint","backend","devices","gpu","gpu_device","weights_dir","lora","lora_path","model_name"}
    if forbidden.intersection(inputs) or any(key.endswith(("_path","_url")) for key in inputs): raise InvalidRequestError("Deployment/model identity or asset paths cannot be overridden by inference parameters.")
    if allowed is not None and set(inputs)-set(allowed): raise InvalidRequestError("Unsupported inference parameter.",details={"parameters":sorted(set(inputs)-set(allowed))})
    if "seed" in inputs and (type(inputs["seed"]) is not int or not 0<=inputs["seed"]<2**32): raise InvalidRequestError("Seed must be a uint32 integer.")
    for name,spec in model.parameters.items():
        if name not in inputs: continue
        value=inputs[name]
        if spec.get("type") in ("integer","number"):
            import math
            if type(value) not in (int,float) or not math.isfinite(value) or spec["type"]=="integer" and type(value) is not int:
                raise InvalidRequestError(f"Invalid numeric parameter {name}.")
            if value<spec.get("minimum",float("-inf")) or value>spec.get("maximum",float("inf")):
                raise InvalidRequestError(f"Parameter {name} is outside manifest bounds.")
    for field,maximum in [("width","max_width"),("height","max_height"),("num_frames","max_frames"),("duration_seconds","max_duration_seconds")]:
        if field in inputs:
            value=inputs[field]
            limit=model.parameters.get(maximum,{}).get("default")
            if type(value) not in (int,float) or not __import__('math').isfinite(value) or value<=0 or limit and value>limit:
                raise InvalidRequestError(f"Invalid or excessive {field}.")


def defaults(model,inputs,names):
    result=dict(inputs)
    for name in names:
        if name not in result:
            if name not in model.parameters or "default" not in model.parameters[name]: raise ModelUnavailableError(f"Manifest lacks inference default: {name}")
            result[name]=model.parameters[name]["default"]
    return result


def native_artifacts(context, output):
    try:
        if output["model"]!=context.binding.model or not output["artifacts"]: raise ValueError("Missing native artifacts/identity")
        results=[]
        root=(context.backend.directory/"outputs").resolve()
        for record in output["artifacts"]:
            path=(root/record["filename"]).resolve()
            if not path.is_relative_to(root) or not path.is_file(): raise ValueError("Native artifact escaped session output directory")
            results.append(artifact(path.read_bytes(),record["type"],record["mime_type"]))
        return {"model":context.binding.model,"artifacts":results}
    except (KeyError,TypeError,ValueError,OSError) as exc:
        raise ModelUnavailableError("Native backend returned invalid artifact metadata.") from exc
