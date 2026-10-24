"""
workflows/video/parameter_mappings.py
Translates FactoryJob input parameters into ComfyUI node inputs for Wan2.2 and MiniMax H3 workflows.
"""
import random
from typing import Any, Dict, Optional


class VideoParameterMapper:
    """Binds video job input parameters into Wan2.2 and MiniMax H3 ComfyUI prompt graphs."""

    @classmethod
    def apply(
        cls,
        workflow: Dict[str, Any],
        model: str,
        task: str,
        inputs: Dict[str, Any],
        staged_files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        staged = staged_files or {}

        prompt = inputs.get("prompt", "")
        negative_prompt = inputs.get("negative_prompt", "")
        seed = inputs.get("seed")
        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)

        fps = inputs.get("fps", 24)
        num_frames = inputs.get("num_frames", 49)
        width = inputs.get("width")
        height = inputs.get("height")

        model_lower = (model or "").lower()
        is_minimax = "minimax" in model_lower or "h3" in model_lower

        if is_minimax:
            width = width or 768
            height = height or 448
            # MiniMax requires multiples of 32
            width = max(32, round(width / 32) * 32)
            height = max(32, round(height / 32) * 32)
        else:
            width = width or 832
            height = height or 480

        # Update text encodes
        for node_id, node in workflow.items():
            c_type = node.get("class_type")
            if c_type == "CLIPTextEncode":
                # Pos prompt if not specified as negative
                if node_id in ("5", "6") and "positive" not in node.get("_role", ""):
                    # Wan text encode
                    if node_id in ("5", "6"):
                        if node_id == "5" or (node_id == "6" and "7" in workflow):
                            node["inputs"]["text"] = prompt
                        elif node_id == "6" or node_id == "7":
                            node["inputs"]["text"] = negative_prompt
                elif node_id == "6":
                    node["inputs"]["text"] = prompt
                elif node_id == "7":
                    node["inputs"]["text"] = negative_prompt

            elif c_type == "VHS_VideoCombine":
                node["inputs"]["frame_rate"] = fps

            elif c_type in ("EmptyHunyuanLatentVideo", "Wan22ImageToVideoLatent"):
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
                node["inputs"]["length"] = num_frames

            elif c_type in ("MiniMaxH3EmptyLatent", "MiniMaxH3ImageToVideo", "MiniMaxH3FL2VLatent", "MiniMaxH3ReferenceToVideo"):
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
                node["inputs"]["length"] = num_frames

            elif c_type == "WanFirstLastFrameToVideo":
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
                node["inputs"]["length"] = num_frames

            elif c_type == "KSampler":
                node["inputs"]["seed"] = seed

            elif c_type == "KSamplerAdvanced":
                node["inputs"]["noise_seed"] = seed

        # Model and task-specific image bindings
        if is_minimax:
            enable_audio = inputs.get("enable_audio", True)
            if not enable_audio:
                # Remove audio decode if disabled
                if "13" in workflow:
                    del workflow["13"]
                if "4" in workflow:
                    del workflow["4"]
                if "14" in workflow and "audio" in workflow["14"]["inputs"]:
                    del workflow["14"]["inputs"]["audio"]

            if task in ("image_to_video", "i2v", "video_with_audio"):
                img_val = staged.get("image_url") or staged.get("image") or inputs.get("image_filename") or inputs.get("image_url")
                if img_val and "8" in workflow:
                    workflow["8"]["inputs"]["image"] = img_val

            if task in ("first_last_frame_to_video", "fl2v"):
                start_val = staged.get("first_frame_url") or inputs.get("first_frame_filename") or inputs.get("first_frame_url", "start.png")
                end_val = staged.get("last_frame_url") or inputs.get("last_frame_filename") or inputs.get("last_frame_url", "end.png")
                if "8_start" in workflow:
                    workflow["8_start"]["inputs"]["image"] = start_val
                if "8_end" in workflow:
                    workflow["8_end"]["inputs"]["image"] = end_val

        else:
            # Wan2.2
            if task in ("image_to_video", "i2v"):
                img_val = staged.get("image_url") or staged.get("image") or inputs.get("image_filename") or inputs.get("image_url", "start.png")
                if "9" in workflow and workflow["9"]["class_type"] == "LoadImage":
                    workflow["9"]["inputs"]["image"] = img_val

            elif task in ("first_last_frame_to_video", "fl2v"):
                start_val = staged.get("first_frame_url") or inputs.get("first_frame_filename") or inputs.get("first_frame_url", "start.png")
                end_val = staged.get("last_frame_url") or inputs.get("last_frame_filename") or inputs.get("last_frame_url", "end.png")
                if "9_start" in workflow:
                    workflow["9_start"]["inputs"]["image"] = start_val
                if "9_end" in workflow:
                    workflow["9_end"]["inputs"]["image"] = end_val

            elif task in ("video_extension", "extension"):
                anchor_val = staged.get("anchor_image") or staged.get("video_url") or inputs.get("anchor_filename") or "anchor.png"
                if "anchor_img" in workflow:
                    workflow["anchor_img"]["inputs"]["image"] = anchor_val

        return workflow
