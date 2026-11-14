from factory.adapters.comfy import ComfyAdapter, checkpoint
from factory.core.model_adapter import PreparedCall
from factory_contracts.errors import InvalidRequestError


class Adapter(ComfyAdapter):
    allowed_inputs={"image","scale"}
    capabilities=frozenset({"enhancement.image_super_resolution"})

    def validate_request(self,model,request,capability):
        super().validate_request(model,request,capability)
        if not request.inputs.get("image"): raise InvalidRequestError("Super-resolution requires an input image.")
        if request.inputs.get("scale",4)!=4: raise InvalidRequestError("Selected checkpoint is the x4 model; no alternative upscaler is selected.")

    def prepare(self,context,request,capability):
        graph={"input":{"class_type":"LoadImage","inputs":{"image":self.stage(context,request.inputs["image"])}},
            "model":{"class_type":"UpscaleModelLoader","inputs":{"model_name":checkpoint(context.model,"primary")}},
            "upscale":{"class_type":"ImageUpscaleWithModel","inputs":{"upscale_model":["model",0],"image":["input",0]}},
            "save":{"class_type":"SaveImage","inputs":{"images":["upscale",0],"filename_prefix":"factory_upscale"}}}
        return PreparedCall(capability,{"graph":graph})
