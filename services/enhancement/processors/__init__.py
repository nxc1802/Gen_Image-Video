"""
services/enhancement/processors
Media enhancement processing engines:
- Image Super-Resolution (Real-ESRGAN / BSRGAN)
- Video Super-Resolution (Real-ESRGAN Video)
- Frame Interpolation (RIFE v4.6+)
- Audio Enhancement (DeepFilterNet / spectral denoise / voice clarity)
"""
from services.enhancement.processors.image_sr import ImageSuperResolutionProcessor
from services.enhancement.processors.video_sr import VideoSuperResolutionProcessor
from services.enhancement.processors.interpolation import FrameInterpolationProcessor
from services.enhancement.processors.audio_enhance import AudioEnhancementProcessor

__all__ = [
    "ImageSuperResolutionProcessor",
    "VideoSuperResolutionProcessor",
    "FrameInterpolationProcessor",
    "AudioEnhancementProcessor",
]
