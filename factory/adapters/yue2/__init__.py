from pathlib import Path
import uuid
from factory.core.model_adapter import ModelAdapter, PreparedCall
from factory.core.media import check_parameters, native_artifacts
from factory_contracts.errors import InvalidRequestError


class Adapter(ModelAdapter):
    supported_backends=frozenset({"yue2_native"})
    capabilities=frozenset({"music.text_to_music","music.lyrics_to_music","music.symbolic_score","music.custom_score","music.score_edit","music.song_variation"})

    def validate_request(self,model,request,capability):
        check_parameters(model,request.inputs,{"prompt","lyrics","score","seed"})
        p=request.inputs
        if not str(p.get("prompt","")).strip(): raise InvalidRequestError("Music style prompt required.")
        if capability in {"music.symbolic_score","music.custom_score","music.score_edit","music.song_variation"}:
            from services.music.yue2.score import validate_abc_score
            if not p.get("score"): raise InvalidRequestError("This operation requires the explicit ABC plan; no unconfigured audio-to-score model fallback.")
            validate_abc_score(p["score"])

    def prepare(self,context,request,capability):
        p={key:request.inputs[key] for key in ["prompt","lyrics","score","seed"] if key in request.inputs}
        return PreparedCall(capability,{"model":context.binding.model,"capability":capability,"inputs":p})

    def infer(self,context,prepared): return context.post_json("/_factory/infer",prepared.payload,timeout=600)
    def normalize_output(self,context,prepared,output): return native_artifacts(context,output)

    def load_native(self,config,dtype):
        from yue2 import YuE2Pipeline
        return YuE2Pipeline.from_pretrained(config["model_path"],vae=str(Path(config["bundle_path"])/"vae"),device="cuda")

    def native_infer(self,loaded,capability,inputs,config):
        p={"style":inputs["prompt"],"lyrics":inputs.get("lyrics",""),"cot":"full","seed":inputs.get("seed",0)}
        if "score" in inputs:p["abc"]=inputs["score"]
        song=loaded(**p)
        root=Path(config["output_dir"]);root.mkdir(exist_ok=True)
        target=root/(uuid.uuid4().hex+".wav");song.save(str(target))
        return {"model":config["model"],"artifacts":[{"filename":target.name,"type":"audio","mime_type":"audio/wav"}]}
