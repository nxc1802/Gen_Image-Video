import base64
from pathlib import Path
import uuid
import wave
from factory.core.model_adapter import ModelAdapter, PreparedCall
from factory.core.media import check_parameters, data_bytes, native_artifacts, defaults
from factory_contracts.errors import InvalidRequestError


class Adapter(ModelAdapter):
    supported_backends=frozenset({"acestep_native"})
    operations={"music.text_to_music":"text2music","music.lyrics_to_music":"text2music","music.reference_audio":"text2music",
        "music.cover":"cover","music.repaint":"repaint","music.extract":"extract","music.lego":"lego","music.complete":"complete","music.extend":"repaint"}
    capabilities=frozenset(operations)

    def validate_request(self,model,request,capability):
        check_parameters(model,request.inputs,{"prompt","lyrics","duration_seconds","seed","bpm","key","stem_name","repainting_start","repainting_end","add_instrument","steps","source_audio","reference_audio","thinking","prompt_expansion"})
        p=request.inputs
        if not str(p.get("prompt","")).strip(): raise InvalidRequestError("Music prompt required.")
        if capability=="music.lyrics_to_music" and not p.get("lyrics"): raise InvalidRequestError("Lyrics required.")
        if capability=="music.reference_audio":data_bytes(p.get("reference_audio"),"audio")
        elif capability not in {"music.text_to_music","music.lyrics_to_music"}:data_bytes(p.get("source_audio"),"audio")
        if p.get("thinking") or p.get("prompt_expansion"): raise InvalidRequestError("This adapter uses the explicitly loaded DiT; optional LM inference is not silently enabled.")

    def prepare(self,context,request,capability):
        p={key:request.inputs[key] for key in ["prompt","lyrics","duration_seconds","seed","bpm","key","stem_name","repainting_start","repainting_end","add_instrument"] if key in request.inputs}
        p["steps"]=request.inputs.get("steps",context.model.parameters["steps"]["default"])
        p=defaults(context.model,p,["duration_seconds"])
        p["task_type"]=self.operations[capability]
        for field in ["source_audio","reference_audio"]:
            if request.inputs.get(field):p[field]=base64.b64encode(data_bytes(request.inputs[field],"audio")).decode()
        return PreparedCall(capability,{"model":context.binding.model,"capability":capability,"inputs":p})

    def infer(self,context,prepared): return context.post_json("/_factory/infer",prepared.payload,timeout=600)
    def normalize_output(self,context,prepared,output): return native_artifacts(context,output)

    def load_native(self,config,dtype):
        from acestep.handler import AceStepHandler
        handler=AceStepHandler()
        root=Path(config["bundle_path"]).parent/"ace_project";root.mkdir(exist_ok=True)
        if not (root/"checkpoints").exists():(root/"checkpoints").symlink_to(Path(config["bundle_path"]))
        status,ok=handler.initialize_service(project_root=str(root),config_path=config["primary_directory"],device="cuda",use_flash_attention=False,
            compile_model=False,use_mlx_dit=False,offload_to_cpu=config["cpu_offload"],offload_dit_to_cpu=config["cpu_offload"])
        if not ok or handler.device!="cuda" or handler.dtype!=dtype: raise RuntimeError("Selected native model/device/precision did not load: "+str(status))
        return handler

    def native_infer(self,loaded,capability,inputs,config):
        from acestep.inference import GenerationParams, GenerationConfig, generate_music
        from acestep.constants import TASK_INSTRUCTIONS
        root=Path(config["output_dir"]);root.mkdir(exist_ok=True)
        work=Path(config["input_dir"]);work.mkdir(exist_ok=True)
        paths={}
        for field in ["source_audio","reference_audio"]:
            if field in inputs:
                target=work/(uuid.uuid4().hex+".wav");target.write_bytes(base64.b64decode(inputs[field],validate=True));paths[field]=str(target)
        kwargs={"task_type":self.operations[capability],"caption":inputs["prompt"],"lyrics":inputs.get("lyrics",""),"duration":inputs.get("duration_seconds",10),
            "inference_steps":inputs["steps"],"seed":inputs.get("seed",0),"thinking":False,"use_cot_metas":False,"use_cot_caption":False,"use_cot_language":False}
        if "source_audio" in paths:kwargs["src_audio"]=paths["source_audio"]
        if "reference_audio" in paths:kwargs["reference_audio"]=paths["reference_audio"]
        for field in ["bpm","repainting_start","repainting_end"]:
            if field in inputs:kwargs[field]=inputs[field]
        task=kwargs["task_type"]
        instruction=TASK_INSTRUCTIONS[task+"_default"] if task in {"extract","lego","complete"} else TASK_INSTRUCTIONS[task]
        if task in {"extract","lego"} and inputs.get("stem_name"):
            instruction=TASK_INSTRUCTIONS[task].format(TRACK_NAME=inputs["stem_name"])
        if task=="complete" and inputs.get("add_instrument"):
            instruction=TASK_INSTRUCTIONS[task].format(TRACK_CLASSES=inputs["add_instrument"])
        kwargs["instruction"]=instruction
        if "key" in inputs:kwargs["keyscale"]=inputs["key"]
        if capability=="music.extend":
            with wave.open(paths["source_audio"],"rb") as audio: duration=audio.getnframes()/audio.getframerate()
            if kwargs["duration"]<=duration: raise ValueError("Extension duration must exceed source duration")
            kwargs.update(repainting_start=duration,repainting_end=kwargs["duration"])
        params=GenerationParams(**kwargs)
        result=generate_music(loaded,None,params,GenerationConfig(batch_size=1,use_random_seed=False,seeds=[inputs.get("seed",0)],audio_format="wav"),save_dir=str(root))
        if not result.success or not result.audios: raise RuntimeError(result.error or "Native generation returned no audio")
        records=[]
        for item in result.audios:
            path=Path(item["path"]).resolve()
            if not path.is_relative_to(root.resolve()): raise RuntimeError("SDK audio escaped output directory")
            records.append({"filename":str(path.relative_to(root.resolve())),"type":"audio","mime_type":"audio/wav"})
        return {"model":config["model"],"artifacts":records}
