"""Private MiniMax loader: explicit Manifest FP16 compute, no weight conversion/fallback."""

import json
import logging

import comfy.sd
import folder_paths
import torch


class MiniMaxManifestLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unet_name": (folder_paths.get_filename_list("diffusion_models"),)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "factory/minimax"

    def load(self, unet_name):
        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(path, model_options={"dtype": torch.float16})
        if model is None:
            raise RuntimeError("Selected MiniMax checkpoint could not be loaded")
        compute = model.model.get_dtype()
        manual = model.model.manual_cast_dtype
        if compute != torch.float16 or manual not in (None, torch.float16):
            raise RuntimeError(
                "Selected MiniMax Manifest requires explicit FP16 diffusion; no dtype fallback"
            )
        if model.load_device.type != "cuda":
            raise RuntimeError(
                "Selected MiniMax diffusion must execute on its assigned CUDA device"
            )
        logging.info(
            "FACTORY_MINIMAX_RUNTIME "
            + json.dumps(
                {
                    "precision": "fp16",
                    "compute_dtype": str(compute),
                    "manual_cast_dtype": str(manual),
                    "load_device": str(model.load_device),
                    "offload_device": str(model.offload_device),
                    "checkpoint": unet_name,
                },
                sort_keys=True,
            )
        )
        return (model,)


NODE_CLASS_MAPPINGS = {"FactoryMiniMaxManifestLoader": MiniMaxManifestLoader}
