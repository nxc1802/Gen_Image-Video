import secrets
from factory.adapters.comfy import ComfyAdapter, checkpoint
from factory.core.model_adapter import PreparedCall
from factory.core.media import defaults
from factory_contracts.errors import InvalidRequestError


class Adapter(ComfyAdapter):
    allowed_inputs = {
        "prompt",
        "instruction",
        "image",
        "reference_images",
        "mask",
        "width",
        "height",
        "steps",
        "cfg",
        "seed",
        "strength",
    }
    capabilities = frozenset(
        {
            "image.text_to_image",
            "image.image_to_image",
            "image.inpaint",
            "image.outpaint",
            "image.instruction_edit",
            "image.reference_image",
            "image.multi_reference",
        }
    )

    def validate_request(self, model, request, capability):
        super().validate_request(model, request, capability)
        if not str(request.inputs.get("prompt") or request.inputs.get("instruction") or "").strip():
            raise InvalidRequestError("Image prompt/instruction required.")
        if (
            capability != "image.text_to_image"
            and not request.inputs.get("image")
            and not request.inputs.get("reference_images")
        ):
            raise InvalidRequestError("Selected image operation requires source/reference images.")
        if capability in {"image.inpaint", "image.outpaint"} and not request.inputs.get("mask"):
            raise InvalidRequestError(
                "Mask required; outpaint input must be a pre-expanded canvas and matching mask."
            )
        if capability == "image.text_to_image" and any(
            request.inputs.get(key) for key in ["image", "reference_images", "mask"]
        ):
            raise InvalidRequestError(
                "Source assets conflict with explicit text-to-image capability."
            )
        for field in ("width", "height"):
            value = request.inputs.get(field, 1024)
            if type(value) is not int or value % 32:
                raise InvalidRequestError("Image dimensions must be multiples of 32.")

    def install_nodes(self, base, *, backend, source_dir):
        from .native_nodes import install

        install(base, backend, source_dir)

    def prepare(self, context, request, capability):
        p = defaults(context.model, request.inputs, ["width", "height", "steps"])
        nodes = {
            "model": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {"unet_name": checkpoint(context.model, "primary")},
            },
            "clip": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": checkpoint(context.model, "clip"),
                    "type": "qwen_image",
                    "device": "default",
                },
            },
            "prefix_cache": {
                "class_type": "QwenImage21Cache",
                "inputs": {"model": ["model", 0], "device": "off", "dtype": "default"},
            },
            "vae": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": checkpoint(context.model, "vae")},
            },
            "encode": {
                "class_type": "TextEncodeQwenImage21",
                "inputs": {
                    "clip": ["clip", 0],
                    "vae": ["vae", 0],
                    "prompt": p.get("instruction", p.get("prompt", "")),
                    "negative_prompt": p.get("negative_prompt", ""),
                    "resolution": p.get("width", 1024),
                },
            },
            "latent": {
                "class_type": "FactoryQwen21Latent",
                "inputs": {"width": p.get("width", 1024), "height": p.get("height", 1024)},
            },
            "sample": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["prefix_cache", 0],
                    "positive": ["encode", 0],
                    "negative": ["encode", 1],
                    "latent_image": ["latent", 0],
                    "seed": p.get("seed", secrets.randbits(32)),
                    "steps": p.get("steps", context.model.parameters["steps"]["default"]),
                    "cfg": p.get("cfg", 1.0),
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
            "decode": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]},
            },
            "save": {
                "class_type": "SaveImage",
                "inputs": {"images": ["decode", 0], "filename_prefix": "factory"},
            },
        }
        refs = p.get("reference_images") or ([p["image"]] if p.get("image") else [])
        if len(refs) > 16:
            raise InvalidRequestError("At most sixteen reference images are supported.")
        images = {}
        for index, value in enumerate(refs, 1):
            key = f"ref{index}"
            nodes[key] = {
                "class_type": "LoadImage",
                "inputs": {"image": self.stage(context, value)},
            }
            images[f"image_{index}"] = [key, 0]
        if refs:
            nodes["source_scale"] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["ref1", 0],
                    "upscale_method": "lanczos",
                    "width": p["width"],
                    "height": p["height"],
                    "crop": "disabled",
                },
            }
            images["image_1"] = ["source_scale", 0]
        if images:
            nodes["encode"]["inputs"].update(
                {f"images.{key}": value for key, value in images.items()}
            )
        if images:
            nodes["encode"]["inputs"]["resolution"] = 0
            nodes["sample"]["inputs"]["latent_image"] = ["encode", 2]
        if capability == "image.image_to_image":
            nodes["source_latent"] = {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["source_scale", 0], "vae": ["vae", 0]},
            }
            nodes["sample"]["inputs"].update(
                latent_image=["source_latent", 0], denoise=p.get("strength", 0.6)
            )
        if capability in {"image.inpaint", "image.outpaint"}:
            nodes["mask"] = {
                "class_type": "LoadImage",
                "inputs": {"image": self.stage(context, p["mask"])},
            }
            nodes["mask_image"] = {"class_type": "MaskToImage", "inputs": {"mask": ["mask", 1]}}
            nodes["mask_scale"] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["mask_image", 0],
                    "upscale_method": "nearest-exact",
                    "width": p["width"],
                    "height": p["height"],
                    "crop": "disabled",
                },
            }
            nodes["scaled_mask"] = {
                "class_type": "ImageToMask",
                "inputs": {"image": ["mask_scale", 0], "channel": "red"},
            }
            nodes["mask_latent"] = {
                "class_type": "VAEEncodeForInpaint",
                "inputs": {
                    "pixels": ["source_scale", 0],
                    "mask": ["scaled_mask", 0],
                    "vae": ["vae", 0],
                    "grow_mask_by": 6,
                },
            }
            nodes["sample"]["inputs"]["latent_image"] = ["mask_latent", 0]
        return PreparedCall(capability, {"graph": nodes})
