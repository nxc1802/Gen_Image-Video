import base64
import hashlib
import io
import json
import logging
from pathlib import Path
import subprocess
import uuid
from factory.core.model_adapter import ModelAdapter, PreparedCall
from factory.core.media import data_bytes, check_parameters, native_artifacts, defaults
from factory_contracts.errors import InvalidRequestError


def emit_endpoint_evidence(evidence):
    # Certification evidence must survive the server's root WARNING log level.
    print("FACTORY_WAN_ENDPOINT_CONDITIONING " + json.dumps(evidence,sort_keys=True),flush=True)


def extract_last_frame(source, anchor):
    subprocess.run(["ffmpeg","-nostdin","-y","-v","error","-i",str(source),"-map","0:v:0","-an","-sws_flags","bitexact+accurate_rnd","-pix_fmt","rgb24","-fps_mode","passthrough","-update","1",str(anchor)],check=True,capture_output=True,timeout=60)
    if not anchor.is_file():
        raise RuntimeError("Input video has no decoded final frame; no substitute anchor")


class LoadedPipelines(dict):
    """Shared native components plus private load identity; not inference certification."""


class Adapter(ModelAdapter):
    capabilities=frozenset({"video.text_to_video","video.image_to_video","video.first_last_frame_to_video","video.video_extension"})
    supported_backends=frozenset({"diffusers"})
    supported_tiling_components=frozenset({"vae"})

    def validate_request(self,model,request,capability):
        check_parameters(model,request.inputs,{"prompt","negative_prompt","image","last_image","video","width","height","num_frames","steps","fps","guidance_scale","seed"})
        p=request.inputs
        if not str(p.get("prompt","")).strip(): raise InvalidRequestError("Video prompt required.")
        required={"video.image_to_video":["image"],"video.first_last_frame_to_video":["image","last_image"],"video.video_extension":["video"]}.get(capability,[])
        for field in required: data_bytes(p.get(field),"video" if field=="video" else "image")
        if any(field in p and field not in required for field in ["image","last_image","video"]): raise InvalidRequestError("Input assets conflict with the explicit video capability.")
        for field in ("width","height"):
            value=p.get(field,model.parameters[field]["default"])
            if type(value) is not int or value%32:
                raise InvalidRequestError("Wan canvas dimensions must be multiples of 32; no silent native rounding.")
        frames=p.get("num_frames",17)
        if type(frames) is not int or frames%4!=1: raise InvalidRequestError("Native frame count must be 4k+1.")
        if capability == "video.first_last_frame_to_video" and frames < 9:
            raise InvalidRequestError("Experimental Wan FL2V needs a free interior latent frame (at least 9 pixel frames).")

    def prepare(self,context,request,capability):
        p=defaults(context.model,request.inputs,["width","height","num_frames","steps","fps"])
        for field in ["image","last_image","video"]:
            if field in p: p[field]=base64.b64encode(data_bytes(p[field],"video" if field=="video" else "image")).decode()
        return PreparedCall(capability,{"model":context.binding.model,"capability":capability,"inputs":p})

    def infer(self,context,prepared): return context.post_json("/_factory/infer",prepared.payload,timeout=context.model.runtime_profiles[context.registry.config.hardware].inference_timeout_seconds)
    def normalize_output(self,context,prepared,output): return native_artifacts(context,output)

    def load_native(self,config,dtype):
        import torch
        from diffusers import WanPipeline, WanImageToVideoPipeline, AutoencoderKLWan
        from .endpoint_conditioning import endpoint_pipeline_class, runtime_identity
        image_pipeline_class = endpoint_pipeline_class(WanImageToVideoPipeline)
        precisions=config.get("component_precisions",{})
        if set(precisions)-{"transformer","text_encoder","vae"} or precisions.get("vae") != "fp32":
            raise RuntimeError("Wan requires an explicit Manifest FP32 VAE contract; no dtype fallback")
        dtypes={"fp16":torch.float16,"bf16":torch.bfloat16,"fp32":torch.float32}
        vae=AutoencoderKLWan.from_pretrained(config["model_path"],subfolder="vae",dtype=dtypes[precisions["vae"]],local_files_only=True)
        tiling=config.get("component_tiling",{})
        if set(tiling)-self.supported_tiling_components:
            raise RuntimeError("Unsupported Wan component tiling policy; no ignored configuration")
        actual_tiling={}
        if "vae" in tiling:
            from factory_contracts.manifests.model import ComponentTilingSpec
            policy=ComponentTilingSpec.model_validate(tiling["vae"]).model_dump()
            if any(value%16 for value in policy.values()):
                raise RuntimeError("Wan VAE tile and stride dimensions must follow its native 16-pixel lattice")
            vae.enable_tiling(tile_sample_min_height=policy["tile_height"],tile_sample_min_width=policy["tile_width"],tile_sample_stride_height=policy["stride_height"],tile_sample_stride_width=policy["stride_width"])
            actual_tiling["vae"]={"tile_height":vae.tile_sample_min_height,"tile_width":vae.tile_sample_min_width,"stride_height":vae.tile_sample_stride_height,"stride_width":vae.tile_sample_stride_width}
            if not vae.use_tiling or actual_tiling["vae"] != policy:
                raise RuntimeError("Native Wan VAE did not honor the exact Manifest tiling policy")
        text=WanPipeline.from_pretrained(config["model_path"],vae=vae,dtype=dtype,local_files_only=True)
        # Construct a shared wrapper directly: from_pipe applies a pipeline-wide dtype cast.
        image=image_pipeline_class(**text.components,image_encoder=None,image_processor=None,boundary_ratio=text.config.boundary_ratio,expand_timesteps=text.config.expand_timesteps)
        parameter_precisions=config.get("component_parameter_precisions",{})
        if set(parameter_precisions)-{"transformer","text_encoder","vae"}:
            raise RuntimeError("Unknown Wan component precision override; no ignored policy")
        components={}
        for name in ("transformer", "text_encoder", "vae"):
            module=getattr(text,name)
            if module is not getattr(image,name):
                raise RuntimeError("Wan wrappers must share the exact same checkpoint components")
            params=dict(module.named_parameters())
            overrides=parameter_precisions.get(name,{})
            if not params or set(overrides)-set(params):
                raise RuntimeError("Wan Manifest contains unknown parameter precision overrides: "+name)
            precision=precisions.get(name,config["precision"])
            dtype_map={"fp16":torch.float16,"bf16":torch.bfloat16,"fp32":torch.float32}
            mismatches={key:{"actual":str(p.dtype),"expected":str(dtype_map[overrides.get(key,precision)])} for key,p in params.items() if p.is_floating_point() and p.dtype != dtype_map[overrides.get(key,precision)]}
            if mismatches or any(not params[key].is_floating_point() for key in overrides):
                raise RuntimeError("Wan parameter precision contract failed: "+name+" "+json.dumps(mismatches,sort_keys=True))
            components[name]={"parameter_count":sum(p.numel() for p in params.values()),"dtypes":sorted({str(p.dtype) for p in params.values() if p.is_floating_point()}),"shared_between_wrappers":True,"default_precision":precision,"parameter_precision_overrides":{key:str(params[key].dtype) for key in overrides}}
        # Both pipeline wrappers share the exact same verified components.
        strategy=config.get("offload_strategy")
        if config["cpu_offload"]:
            if strategy == "sequential":
                text.enable_sequential_cpu_offload(gpu_id=0);image.enable_sequential_cpu_offload(gpu_id=0)
            elif strategy == "model":
                text.enable_model_cpu_offload(gpu_id=0);image.enable_model_cpu_offload(gpu_id=0)
            else:
                raise RuntimeError("Wan CPU offload requires an explicit supported Manifest strategy; no automatic fallback")
        else:
            if strategy is not None:raise RuntimeError("Offload strategy conflicts with the selected runtime profile")
            text.to("cuda:0");image.to("cuda:0")
        loaded=LoadedPipelines(text=text,image=image)
        loaded._factory_runtime_identity={"precision":config["precision"],"component_precisions":precisions,"component_parameter_precisions":parameter_precisions,"components":components,"component_tiling":actual_tiling,"cpu_offload":config["cpu_offload"],"offload_strategy":strategy,"cuda_device":"cuda:0"}
        loaded._factory_runtime_identity["endpoint_conditioning"] = runtime_identity()
        logging.info("FACTORY_WAN_RUNTIME " + json.dumps(loaded._factory_runtime_identity,sort_keys=True))
        return loaded

    def native_infer(self,loaded,capability,inputs,config):
        if capability == "video.first_last_frame_to_video":
            from .endpoint_conditioning import POLICY
            # Pinned TI2V-5B's native prepare_latents encodes only the first image.
            # Accepting last_image in its signature does not consume that input.
            if getattr(loaded["image"], "_factory_endpoint_conditioning", None) != POLICY:
                raise RuntimeError("Selected Wan TI2V native conditioning ignores last_image; verified experimental FL2V preparation is required. No first-frame-only substitute.")
        from PIL import Image
        import torch
        from diffusers.utils import export_to_video
        root=Path(config["output_dir"]);root.mkdir(exist_ok=True)
        work=Path(config["input_dir"]);work.mkdir(exist_ok=True)
        p={"prompt":inputs["prompt"],"negative_prompt":inputs.get("negative_prompt",""),"width":inputs.get("width",512),"height":inputs.get("height",512),"num_frames":inputs.get("num_frames",17),
           "num_inference_steps":inputs.get("steps",30),"guidance_scale":inputs.get("guidance_scale",5.0)}
        if "seed" in inputs:p["generator"]=torch.Generator(device="cuda:0").manual_seed(inputs["seed"])
        source=None
        if capability=="video.video_extension":
            source=work/(uuid.uuid4().hex+".mp4");source.write_bytes(base64.b64decode(inputs["video"],validate=True))
            anchor=work/(uuid.uuid4().hex+".png")
            extract_last_frame(source,anchor)
            p["image"]=Image.open(anchor).convert("RGB")
        elif "image" in inputs:p["image"]=Image.open(io.BytesIO(base64.b64decode(inputs["image"],validate=True))).convert("RGB")
        if "last_image" in inputs:p["last_image"]=Image.open(io.BytesIO(base64.b64decode(inputs["last_image"],validate=True))).convert("RGB")
        pipeline=loaded["image"] if "image" in p else loaded["text"]
        try:
            frames=pipeline(**p).frames[0]
        except Exception:
            logging.exception("Selected exact Wan native inference failed")
            raise
        output=root/(uuid.uuid4().hex+".mp4");export_to_video(frames,str(output),fps=inputs.get("fps",16))
        if capability == "video.first_last_frame_to_video":
            evidence = dict(pipeline._factory_endpoint_call_evidence)
            evidence.update(
                source_first_image_sha256=hashlib.sha256(base64.b64decode(inputs["image"],validate=True)).hexdigest(),
                source_last_image_sha256=hashlib.sha256(base64.b64decode(inputs["last_image"],validate=True)).hexdigest(),
                seed=inputs.get("seed"),
                output_mp4_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            emit_endpoint_evidence(evidence)
        if source:
            combined=root/(uuid.uuid4().hex+".mp4")
            fps=inputs.get("fps",16)
            filt=f"[0:v]scale={p['width']}:{p['height']},fps={fps},setsar=1[v0];[1:v]fps={fps},setsar=1[v1];[v0][v1]concat=n=2:v=1:a=0[v]"
            subprocess.run(["ffmpeg","-y","-i",str(source),"-i",str(output),"-filter_complex",filt,"-map","[v]","-map","0:a?","-c:v","libx264","-c:a","aac",str(combined)],check=True,capture_output=True,timeout=120)
            output=combined  # Failure never substitutes the unstitched segment.
        return {"model":config["model"],"artifacts":[{"filename":output.name,"type":"video","mime_type":"video/mp4"}]}
