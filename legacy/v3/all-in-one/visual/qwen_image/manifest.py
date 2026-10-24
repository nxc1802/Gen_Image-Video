"""
Manifest defining model files and target ComfyUI directories for Qwen-Image 2.1
"""

def get_qwen_image_targets(unet_dir, diffusion_dir, text_encoders_dir, clip_dir, vae_dir):
    return [
        ("qwen-image-2.1-Q4_K_M.gguf", [unet_dir, diffusion_dir]),
        ("qwen3vl_8b_fp8_scaled.safetensors", [text_encoders_dir, clip_dir]),
        ("qwen_image_2.1_vae_bf16.safetensors", [vae_dir]),
    ]
