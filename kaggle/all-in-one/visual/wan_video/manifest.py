"""
Manifest defining model files and target ComfyUI directories for Wan2.2 Video
"""

def get_wan_video_targets(unet_dir, diffusion_dir, text_encoders_dir, clip_dir, vae_dir):
    return [
        # Wan2.2 TI2V-5B
        ("Wan2.2-TI2V-5B-Q4_K_M.gguf", [unet_dir, diffusion_dir]),
        ("Wan2.2_VAE.safetensors", [vae_dir]),
        # Wan2.2 T2V-A14B & I2V-A14B
        ("Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf", [unet_dir, diffusion_dir]),
        ("Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf", [unet_dir, diffusion_dir]),
        ("Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf", [unet_dir, diffusion_dir]),
        ("Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf", [unet_dir, diffusion_dir]),
        ("Wan2.1_VAE.safetensors", [vae_dir]),
        # Text Encoder UMT5-XXL
        ("umt5-xxl-encoder-Q4_K_M.gguf", [text_encoders_dir, clip_dir]),
    ]
