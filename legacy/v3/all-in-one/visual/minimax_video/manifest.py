"""
Manifest defining model files and target ComfyUI directories for MiniMax H3 (Video & Audio)
"""

def get_minimax_video_targets(unet_dir, diffusion_dir, text_encoders_dir, clip_dir, vae_dir, loras_dir=None):
    targets = [
        # MiniMax H3 FL2VA DiT Backbone (SafeTensors INT8 ConvRot & GGUF)
        ("MiniMax-H3-FL2VA-Q4_K_M.gguf", [unet_dir, diffusion_dir]),
        ("minimax_h3_fl2va_pruned-Q4_K.gguf", [unet_dir, diffusion_dir]),
        ("minimax_h3_fl2va_pruned_fp8_scaled.safetensors", [unet_dir, diffusion_dir]),
        ("minimax_h3_fl2va_pruned_w6a8.safetensors", [unet_dir, diffusion_dir]),
        ("minimax_h3_fl2va_pruned_int8_convrot.safetensors", [unet_dir, diffusion_dir]),
        ("MiniMax-H3_FL2VA-INT8-ConvRot.safetensors", [unet_dir, diffusion_dir]),
        # MiniMax H3 Ref2VA DiT Backbone (Multi-Reference Image/Video/Audio)
        ("minimax_h3_ref2va_pruned_int8_convrot.safetensors", [unet_dir, diffusion_dir]),
        ("MiniMax-H3_Ref2VA-INT8-ConvRot.safetensors", [unet_dir, diffusion_dir]),
        ("minimax_h3_ref2va_pruned-w4a8_convrot_pruned.safetensors", [unet_dir, diffusion_dir]),
        ("minimax_h3_ref2va_pruned-Q6_K.gguf", [unet_dir, diffusion_dir]),
        ("minimax-h3-ref2va-U16G.gguf", [unet_dir, diffusion_dir]),
        # Text Encoder Qwen3-VL 32B (Shared 100% between FL2VA & Ref2VA)
        ("qwen3_vl_32b_q4_k_m.gguf", [text_encoders_dir, clip_dir]),
        ("qwen3vl_32b_minimax_h3-Q4_K_M.gguf", [text_encoders_dir, clip_dir]),
        ("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", [text_encoders_dir, clip_dir]),
        ("qwen3vl-32B-MiniMax-H3-Q2_K.gguf", [text_encoders_dir, clip_dir]),
        # Video VAE (Shared 100% between FL2VA & Ref2VA)
        ("minimax_h3_video_vae_fp16.safetensors", [vae_dir]),
        ("minimax_h3_video_vae_int8_convrot.safetensors", [vae_dir]),
        ("MiniMax-H3-video_vae_fp16.safetensors", [vae_dir]),
        # Audio VAE (Shared 100% between FL2VA & Ref2VA)
        ("minimax_h3_audio_vae_fp32.safetensors", [vae_dir]),
        ("MiniMax-H3-audio_vae_fp32.safetensors", [vae_dir]),
    ]
    if loras_dir is not None:
        targets.extend([
            # MiniMax H3 Turbo LoRAs (4-8 Steps Distilled Acceleration)
            ("minimax_h3_turbo_4step_pruned_comfyui.safetensors", [loras_dir]),
            ("minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors", [loras_dir]),
            ("minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors", [loras_dir]),
            ("minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", [loras_dir]),
            ("minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", [loras_dir]),
            ("MiniMax-H3-FL2VA-Acc-8Step.safetensors", [loras_dir]),
            ("MiniMax-H3-Ref2VA-Acc-8Step.safetensors", [loras_dir]),
        ])
    return targets

