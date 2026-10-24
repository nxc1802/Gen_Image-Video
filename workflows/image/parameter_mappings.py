"""
workflows/image/parameter_mappings.py
Translates FactoryJob input parameters into ComfyUI node inputs for Qwen-Image 2.1 workflows.
"""
import random
from typing import Any, Dict, List, Optional

CONTROLNET_MODELS = {
    "openpose": "qwen_image_control_pose.safetensors",
    "pose": "qwen_image_control_pose.safetensors",
    "depth": "qwen_image_control_depth.safetensors",
    "canny": "qwen_image_control_canny.safetensors",
    "edge": "qwen_image_control_canny.safetensors",
}


class ImageParameterMapper:
    """Binds user/job parameters into a ComfyUI prompt graph."""

    @classmethod
    def apply(
        cls,
        workflow: Dict[str, Any],
        task: str,
        inputs: Dict[str, Any],
        staged_files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Applies input parameters to the workflow in-place and returns it.
        staged_files: maps input field/URL to local staged filename in ComfyUI input/ directory.
        """
        staged = staged_files or {}

        prompt = inputs.get("prompt", "")
        negative_prompt = inputs.get("negative_prompt", "")
        seed = inputs.get("seed")
        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)

        steps = inputs.get("steps", 20)
        cfg = inputs.get("cfg", 1.0)
        sampler_name = inputs.get("sampler_name", "euler")
        scheduler = inputs.get("scheduler", "simple")
        width = inputs.get("width", 1024)
        height = inputs.get("height", 1024)

        # Standard KSampler update if node "476" exists
        if "476" in workflow:
            ksampler = workflow["476"]["inputs"]
            ksampler["seed"] = seed
            ksampler["steps"] = steps
            ksampler["cfg"] = cfg
            ksampler["sampler_name"] = sampler_name
            ksampler["scheduler"] = scheduler

        # Standard TextEncode update if node "471" exists
        if "471" in workflow:
            text_node = workflow["471"]["inputs"]
            if task == "instruction_edit":
                text_node["prompt"] = inputs.get("instruction", prompt)
            else:
                text_node["prompt"] = prompt
            text_node["negative_prompt"] = negative_prompt
            text_node["resolution"] = width

        # Standard EmptyLatentImage update if node "474" exists as EmptyLatentImage
        if "474" in workflow and workflow["474"]["class_type"] == "EmptyLatentImage":
            workflow["474"]["inputs"]["width"] = width
            workflow["474"]["inputs"]["height"] = height

        # Task-specific bindings
        if task == "text_to_image":
            if "476" in workflow:
                workflow["476"]["inputs"]["denoise"] = 1.0

        elif task == "image_to_image":
            denoise = float(inputs.get("denoise_strength", 0.65))
            if "476" in workflow:
                workflow["476"]["inputs"]["denoise"] = denoise
            img_val = staged.get("image_url") or staged.get("image") or inputs.get("image_filename") or inputs.get("image_url", "input.png")
            if "474_img" in workflow:
                workflow["474_img"]["inputs"]["image"] = img_val

        elif task == "inpaint":
            if "476" in workflow:
                workflow["476"]["inputs"]["denoise"] = 1.0
            img_val = staged.get("image_url") or staged.get("image") or inputs.get("image_filename") or inputs.get("image_url", "input.png")
            mask_val = staged.get("mask_url") or staged.get("mask") or inputs.get("mask_filename") or inputs.get("mask_url", "mask.png")
            if "474_img" in workflow:
                workflow["474_img"]["inputs"]["image"] = img_val
            if "474_mask" in workflow:
                workflow["474_mask"]["inputs"]["image"] = mask_val
            if "474" in workflow and "grow_mask_by" in workflow["474"]["inputs"]:
                workflow["474"]["inputs"]["grow_mask_by"] = int(inputs.get("mask_blur", 6))

        elif task == "outpaint":
            denoise = float(inputs.get("denoise_strength", 0.85))
            if "476" in workflow:
                workflow["476"]["inputs"]["denoise"] = denoise
            img_val = staged.get("image_url") or staged.get("image") or inputs.get("image_filename") or inputs.get("image_url", "input.png")
            if "474_img" in workflow:
                workflow["474_img"]["inputs"]["image"] = img_val
            expand = inputs.get("expand", {})
            if "474_pad" in workflow:
                workflow["474_pad"]["inputs"]["left"] = int(expand.get("left", 0))
                workflow["474_pad"]["inputs"]["top"] = int(expand.get("top", 0))
                workflow["474_pad"]["inputs"]["right"] = int(expand.get("right", 0))
                workflow["474_pad"]["inputs"]["bottom"] = int(expand.get("bottom", 0))

        elif task == "instruction_edit":
            if "476" in workflow:
                workflow["476"]["inputs"]["denoise"] = 1.0
            img_val = staged.get("image_url") or staged.get("image") or inputs.get("image_filename") or inputs.get("image_url", "input.png")
            if "ref_img" in workflow:
                workflow["ref_img"]["inputs"]["image"] = img_val

        elif task == "reference_conditioning":
            if "476" in workflow:
                workflow["476"]["inputs"]["denoise"] = 1.0
            ref_urls = inputs.get("reference_image_urls", [])
            for idx, url in enumerate(ref_urls[:4]):
                ref_val = staged.get(url) or staged.get(f"ref_{idx}") or inputs.get(f"ref_{idx}_filename") or url
                node_id = f"ref_img_{idx}"
                workflow[node_id] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": ref_val},
                }
                if "471" in workflow:
                    workflow["471"]["inputs"][f"image_{idx+1}"] = [node_id, 0]

        elif task == "controlnet":
            if "476" in workflow:
                workflow["476"]["inputs"]["denoise"] = 1.0
            ctrl_type = str(inputs.get("control_type", "openpose")).lower()
            ctrl_model = CONTROLNET_MODELS.get(ctrl_type, "qwen_image_control_pose.safetensors")
            if "ctrl_loader" in workflow:
                workflow["ctrl_loader"]["inputs"]["control_net_name"] = ctrl_model
            ctrl_img_val = (
                staged.get("control_image_url")
                or staged.get("control_image")
                or inputs.get("control_image_filename")
                or inputs.get("control_image_url", "control.png")
            )
            if "ctrl_img" in workflow:
                workflow["ctrl_img"]["inputs"]["image"] = ctrl_img_val
            if "ctrl_apply" in workflow:
                workflow["ctrl_apply"]["inputs"]["strength"] = float(inputs.get("strength", 0.85))

        return workflow
