"""Visual generation package: FLUX.1 Image and Wan2.1 Video."""

def get_flux_engine(*args, **kwargs):
    from .flux_image import get_flux_engine as _get
    return _get(*args, **kwargs)

def get_wan_engine(*args, **kwargs):
    from .wan_video import get_wan_engine as _get
    return _get(*args, **kwargs)

__all__ = ["get_flux_engine", "get_wan_engine"]
