"""
Visual generation package: ComfyUI-Powered Qwen-Image-2.1, Wan2.2 Video, and MiniMax H3 Engines.
Hệ thống kiến trúc Modular: Tách riêng từng Engine & Workflow cho Qwen-Image, Wan2.2 và MiniMax H3.
"""

from typing import List, Tuple

def get_qwen_image_engine(*args, **kwargs):
    from .qwen_image.engine import get_qwen_image_engine as _get
    return _get(*args, **kwargs)

# Backward compatibility alias
def get_flux_engine(*args, **kwargs):
    return get_qwen_image_engine(*args, **kwargs)

def get_wan_engine(*args, **kwargs):
    from .wan_video.engine import get_wan_engine as _get
    return _get(*args, **kwargs)

def get_wan22_engine(*args, **kwargs):
    from .wan_video.engine import get_wan22_engine as _get
    return _get(*args, **kwargs)

def get_minimax_engine(*args, **kwargs):
    from .minimax_video.engine import get_minimax_engine as _get
    return _get(*args, **kwargs)

def get_minimax_h3_engine(*args, **kwargs):
    return get_minimax_engine(*args, **kwargs)

def get_all_visual_targets(unet_dir, diffusion_dir, text_encoders_dir, clip_dir, vae_dir, loras_dir=None) -> List[Tuple[str, List]]:
    """Tổng hợp toàn bộ danh sách target files từ tất cả các manifest để symlink."""
    from .qwen_image.manifest import get_qwen_image_targets
    from .wan_video.manifest import get_wan_video_targets
    from .minimax_video.manifest import get_minimax_video_targets

    targets = []
    targets.extend(get_qwen_image_targets(unet_dir, diffusion_dir, text_encoders_dir, clip_dir, vae_dir))
    targets.extend(get_wan_video_targets(unet_dir, diffusion_dir, text_encoders_dir, clip_dir, vae_dir))
    targets.extend(get_minimax_video_targets(unet_dir, diffusion_dir, text_encoders_dir, clip_dir, vae_dir, loras_dir=loras_dir))
    return targets

__all__ = [
    "get_qwen_image_engine",
    "get_flux_engine",
    "get_wan_engine",
    "get_wan22_engine",
    "get_minimax_engine",
    "get_minimax_h3_engine",
    "get_all_visual_targets",
]
