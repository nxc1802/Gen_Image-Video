import secrets
from pathlib import PurePosixPath
from factory.adapters.comfy import ComfyAdapter, checkpoint
from factory.core.model_adapter import PreparedCall
from factory.core.media import defaults
from factory_contracts.errors import InvalidRequestError, ModelUnavailableError


class Adapter(ComfyAdapter):
    allowed_inputs={"prompt","image","last_image","reference_images","width","height","num_frames","steps","seed","fps"}
    capabilities=frozenset({"video.text_to_video","video.image_to_video","video.first_last_frame_to_video","video.video_with_audio","video.reference_conditioning"})

    def validate_request(self,model,request,capability):
        super().validate_request(model,request,capability)
        p=request.inputs
        if not str(p.get("prompt","")).strip(): raise InvalidRequestError("Video prompt required.")
        allowed_keyframes={"video.image_to_video":{"image"},"video.first_last_frame_to_video":{"image","last_image"},"video.video_with_audio":{"image","last_image"} if not model.capabilities.get("video.reference_conditioning") else set()}.get(capability,set())
        if any(field in p and field not in allowed_keyframes for field in ("image","last_image")):
            raise InvalidRequestError("Keyframe inputs conflict with the explicit video capability.")
        if capability=="video.image_to_video" and not p.get("image"): raise InvalidRequestError("Initial image required.")
        if capability=="video.first_last_frame_to_video" and not (p.get("image") and p.get("last_image")): raise InvalidRequestError("Both keyframes required.")
        if model.capabilities.get("video.reference_conditioning") and not p.get("reference_images"): raise InvalidRequestError("Selected reference checkpoint requires reference images.")
        if p.get("reference_images") and not model.capabilities.get("video.reference_conditioning"): raise InvalidRequestError("Reference conditioning requires the explicitly selected reference checkpoint.")
        if len(p.get("reference_images",[]))>9: raise InvalidRequestError("At most nine reference images are supported.")
        for field in ["width","height"]:
            value=p.get(field,model.parameters[field]["default"])
            if type(value) is not int or value%32: raise InvalidRequestError("Native canvas dimensions must be multiples of 32.")
        frames=p.get("num_frames",model.parameters["num_frames"]["default"])
        if type(frames) is not int or frames<5 or frames%17!=5: raise InvalidRequestError("Native frame grid is 17k+5; no silent frame-count rounding.")

    def install_nodes(self, base, *, backend, source_dir):
        from .native_nodes import install
        install(base, backend, source_dir)

    def prepare(self,context,request,capability):
        m=context.model;p=defaults(m,request.inputs,["width","height","num_frames","steps","fps"])
        ref=bool(m.capabilities.get("video.reference_conditioning"))
        nodes={"model":{"class_type":"FactoryMiniMaxManifestLoader","inputs":{"unet_name":checkpoint(m,"primary")}},
            "clip":{"class_type":"CLIPLoader","inputs":{"clip_name":checkpoint(m,"clip"),"type":"minimax","device":"default"}},
            "vae":{"class_type":"VAELoader","inputs":{"vae_name":checkpoint(m,"video_vae")}},
            "condition":{"class_type":"MiniMaxH3ReferenceToVideo" if ref else "MiniMaxH3ImageToVideo","inputs":{"clip":["clip",0],"vae":["vae",0],"prompt":p["prompt"],"width":p.get("width",768),"height":p.get("height",448),"length":p.get("num_frames",107)}},
            "negative":{"class_type":"ConditioningZeroOut","inputs":{"conditioning":["condition",0]}},
            "shift":{"class_type":"MiniMaxH3SigmaShift","inputs":{"model":["model",0],"shift_video":12.0,"shift_audio":3.0}},
            "sample":{"class_type":"KSampler","inputs":{"model":["shift",0],"positive":["condition",0],"negative":["negative",0],"latent_image":["condition",1],
                "seed":p.get("seed",secrets.randbits(32)),"steps":p.get("steps",8 if "turbo_lora" in m.weights.files else 30),"cfg":1.0,"sampler_name":"euler","scheduler":"simple","denoise":1.0}},
            "decode":{"class_type":"VAEDecode","inputs":{"samples":["sample",0],"vae":["vae",0]}},
            "video":{"class_type":"CreateVideo","inputs":{"images":["decode",0],"fps":24.0}},
            "save":{"class_type":"SaveVideo","inputs":{"video":["video",0],"filename_prefix":"factory","format":"mp4","codec":"h264"}}}
        if ref:
            refs={}
            for index,value in enumerate(p["reference_images"],1):
                key=f"ref{index}"; nodes[key]={"class_type":"LoadImage","inputs":{"image":self.stage(context,value)}};refs[f"ref_images.ref_image_{index}"]=[key,0]
            nodes["condition"]["inputs"].update(ref_image_size="match",**refs)
        else:
            for field,native in [("image","first_frame"),("last_image","last_frame")]:
                if p.get(field):
                    nodes[field]={"class_type":"LoadImage","inputs":{"image":self.stage(context,p[field])}}
                    nodes["condition"]["inputs"][native]=[field,0]
        if "turbo_lora" in m.weights.files:
            nodes["lora"]={"class_type":"LoraLoaderModelOnly","inputs":{"model":["model",0],"lora_name":checkpoint(m,"turbo_lora"),"strength_model":1.0}}
            nodes["shift"]["inputs"]["model"]=["lora",0]
        if capability=="video.video_with_audio":
            nodes["audio_vae"]={"class_type":"VAELoader","inputs":{"vae_name":checkpoint(m,"audio_vae")}}
            nodes["audio"]={"class_type":"VAEDecodeAudio","inputs":{"samples":["sample",0],"vae":["audio_vae",0]}}
            nodes["video"]["inputs"]["audio"]=["audio",0]
        return PreparedCall(capability,{"graph":nodes})

    def normalize_output(self, context, prepared, output):
        normalized={**output,"outputs":{}}
        for node_id,entry in output.get("outputs",{}).items():
            node=prepared.payload["graph"].get(node_id)
            if node is None or node["class_type"] != "SaveVideo":
                raise ModelUnavailableError("Unexpected MiniMax native output node.")
            records=entry.get("images",[])
            if not records or any(PurePosixPath(r["filename"]).suffix.lower() != ".mp4" for r in records):
                raise ModelUnavailableError("MiniMax SaveVideo did not return the selected MP4 format.")
            normalized["outputs"][node_id]={"videos":records}
        return super().normalize_output(context,prepared,normalized)
