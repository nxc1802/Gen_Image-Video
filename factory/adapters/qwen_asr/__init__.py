import base64
from factory.core.model_adapter import ModelAdapter, PreparedCall
from factory.core.media import data_bytes, check_parameters
from factory_contracts.errors import InvalidRequestError, ModelUnavailableError


class Adapter(ModelAdapter):
    capabilities=frozenset({"speech.asr"})
    supported_backends=frozenset({"qwen_asr"})

    def validate_request(self,model,request,capability):
        check_parameters(model,request.inputs,{"audio","language","prompt","word_timestamps"})
        data_bytes(request.inputs.get("audio"),"audio")
        if request.inputs.get("word_timestamps"): raise InvalidRequestError("This manifest has no forced-aligner weights.")

    def prepare(self,context,request,capability):
        inputs={k:v for k,v in request.inputs.items() if k in {"language","prompt"}}
        inputs["audio"]=base64.b64encode(data_bytes(request.inputs["audio"],"audio")).decode()
        return PreparedCall(capability,{"model":context.binding.model,"capability":capability,"inputs":inputs})

    def infer(self,context,prepared): return context.post_json("/_factory/infer",prepared.payload)

    def normalize_output(self,context,prepared,output):
        if output.get("model")!=context.binding.model or not isinstance(output.get("text"),str) or not output["text"].strip():
            raise ModelUnavailableError("Native ASR returned no valid transcript.")
        return output

    def load_native(self,config,dtype):
        from qwen_asr import Qwen3ASRModel
        loaded=Qwen3ASRModel.from_pretrained(config["model_path"],dtype=dtype,device_map="cuda:0",max_inference_batch_size=1,local_files_only=True)
        parameters=list(loaded.model.parameters())
        if loaded.backend!="transformers" or not parameters or any(p.device.type!="cuda" or p.device.index!=0 for p in parameters):
            raise RuntimeError("Selected ASR model parameters are not resident on the assigned CUDA device")
        floating={str(p.dtype) for p in parameters if p.is_floating_point()}
        if floating!={str(dtype)}: raise RuntimeError("ASR parameter precision differs from selected manifest precision")
        loaded._factory_runtime_identity={"implementation":loaded.backend,"parameter_count":sum(p.numel() for p in parameters),
            "parameter_devices":sorted({str(p.device) for p in parameters}),"parameter_dtypes":sorted(floating)}
        return loaded

    def native_infer(self,loaded,capability,inputs,config):
        import io, wave
        import numpy as np
        raw=base64.b64decode(inputs["audio"],validate=True)
        with wave.open(io.BytesIO(raw),"rb") as audio:
            if audio.getsampwidth()!=2: raise ValueError("Native ASR input must be PCM16 WAV")
            samples=np.frombuffer(audio.readframes(audio.getnframes()),dtype=np.int16).astype(np.float32)/32768
            if audio.getnchannels()>1: samples=samples.reshape(-1,audio.getnchannels()).mean(axis=1)
            rate=audio.getframerate()
        results=loaded.transcribe(audio=(samples,rate),language=inputs.get("language"),context=inputs.get("prompt",""),return_time_stamps=False)
        if len(results)!=1: raise ValueError("Native ASR returned wrong batch count")
        return {"model":config["model"],"text":results[0].text,"language":results[0].language}
