"""
🎮 ComfyUI Process & Lifecycle Manager
================================================================================
Quản lý tập trung toàn bộ vòng đời ComfyUI Backend trên Kaggle / Multi-GPU:
1. Tự động kiểm tra / clone ComfyUI và các custom nodes (ComfyUI-GGUF, VideoHelperSuite).
2. Tự động tìm kiếm & symlink models từ /kaggle/input sang ComfyUI models directory.
3. Chạy ComfyUI daemon trên GPU được chỉ định trong Config (mặc định GPU 1: cuda:1).
   -> GPU 1 độc lập cho Visual Generation, giải phóng hoàn toàn GPU 0 cho Audio/VLM.
4. Giao tiếp qua REST API nội bộ (127.0.0.1:8188):
   - POST /prompt: Gửi workflow
   - GET /history/{prompt_id}: Theo dõi tiến trình
   - GET /view: Tải ảnh / video kết quả
5. ComfyUI xử lý hoàn toàn Dynamic VRAM và Offloading:
   - Zero Multi-GPU splitting cho 1 model -> Loại bỏ 100% rủi ro OOM và bus contention.
"""

import base64
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

logger = logging.getLogger("ComfyUIManager")


class ComfyUIManager:
    """Quản lý tiến trình, models và giao tiếp REST API với ComfyUI."""

    _instances: Dict[int, "ComfyUIManager"] = {}
    _lock = threading.Lock()
    _setup_lock = threading.Lock()
    _setup_done = False

    def __init__(
        self,
        comfy_dir: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 8188,
        cuda_device: int = 0,
    ):
        default_comfy_dir = (
            "/kaggle/working/ComfyUI"
            if os.path.exists("/kaggle/working")
            else "/content/ComfyUI"
        )
        self.comfy_dir = Path(
            comfy_dir or os.environ.get("COMFYUI_DIR", default_comfy_dir)
        )
        self.host = host
        self.port = port
        self.cuda_device = cuda_device
        self.base_url = f"http://{self.host}:{self.port}"

        self.custom_nodes_dir = self.comfy_dir / "custom_nodes"
        self.models_dir = self.comfy_dir / "models"
        self.unet_dir = self.models_dir / "unet"
        self.diffusion_dir = self.models_dir / "diffusion_models"
        self.text_encoders_dir = self.models_dir / "text_encoders"
        self.clip_dir = self.models_dir / "clip"
        self.vae_dir = self.models_dir / "vae"
        self.loras_dir = self.models_dir / "loras"
        self.input_dir = self.comfy_dir / "input"
        self.output_dir = self.comfy_dir / "output"

        self.process: Optional[subprocess.Popen] = None
        self._execution_lock = threading.Lock()
        self.startup_time_seconds: Optional[float] = None
        self.linked_models: Dict[str, str] = {}

        logger.info(
            f"🎮 ComfyUIManager khởi tạo: dir={self.comfy_dir}, device=cuda:{self.cuda_device}, url={self.base_url}"
        )

    # ==========================================================================
    # 1. SETUP & INSTALLATION
    # ==========================================================================
    def ensure_installed(self) -> bool:
        """Đảm bảo ComfyUI và các custom node cần thiết đã sẵn sàng."""
        try:
            gguf_node_dir = self.custom_nodes_dir / "ComfyUI-GGUF"
            vhs_node_dir = self.custom_nodes_dir / "ComfyUI-VideoHelperSuite"
            marker_file = Path("/tmp/.comfyui_deps_ready")

            # Tạo sẵn các thư mục models
            for d in [
                self.unet_dir,
                self.diffusion_dir,
                self.text_encoders_dir,
                self.clip_dir,
                self.vae_dir,
                self.loras_dir,
                self.input_dir,
                self.output_dir,
            ]:
                d.mkdir(parents=True, exist_ok=True)

            if (
                marker_file.exists()
                and self.comfy_dir.exists()
                and gguf_node_dir.exists()
                and vhs_node_dir.exists()
            ):
                gov_file = self.comfy_dir / "app" / "governance.py"
                if not gov_file.exists():
                    logger.info("🔧 Đồng bộ toàn bộ thư mục app/ cho ComfyUI...")
                    (self.comfy_dir / "app").mkdir(parents=True, exist_ok=True)
                    subprocess.run(
                        f"git clone --depth 1 -q https://github.com/Comfy-Org/ComfyUI.git /tmp/comfy_app_sync && "
                        f"cp -r /tmp/comfy_app_sync/app/* {self.comfy_dir}/app/ 2>/dev/null || true; "
                        f"rm -rf /tmp/comfy_app_sync",
                        shell=True,
                        check=False,
                    )
                self.patch_gguf_ops_compat()
                try:
                    import comfy_aimdo
                except ImportError:
                    logger.info("📦 Bổ sung gói comfy-aimdo, comfy-kitchen cho ComfyUI...")
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "comfy-aimdo", "comfy-kitchen"], check=False)
                logger.info("⚡ ComfyUI & Custom Nodes đã được cài đặt sẵn từ bước Setup.")
                return True

            # 1. Clone ComfyUI nếu chưa có
            if not self.comfy_dir.exists():
                logger.info(f"📥 Đang clone ComfyUI về {self.comfy_dir}...")
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "https://github.com/Comfy-Org/ComfyUI.git",
                        str(self.comfy_dir),
                    ],
                    check=True,
                )

            # 2. Cài đặt / cập nhật dependencies bắt buộc của ComfyUI (comfy-aimdo, comfy-kitchen, av...)
            req_file = self.comfy_dir / "requirements.txt"
            if req_file.exists():
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
                    check=False,
                )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "comfy-aimdo",
                    "comfy-kitchen",
                    "comfyui-frontend-package",
                    "comfyui-workflow-templates",
                    "comfyui-embedded-docs",
                    "av",
                ],
                check=False,
            )

            # 3. Custom Node: ComfyUI-GGUF (Bắt buộc cho Qwen-Image & Wan2.2 GGUF)
            self.custom_nodes_dir.mkdir(parents=True, exist_ok=True)
            if not gguf_node_dir.exists():
                logger.info(f"📥 Đang clone ComfyUI-GGUF về {gguf_node_dir}...")
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "https://github.com/leejet/ComfyUI-GGUF.git",
                        str(gguf_node_dir),
                    ],
                    check=True,
                )
            gguf_req = gguf_node_dir / "requirements.txt"
            if gguf_req.exists():
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "-r", str(gguf_req)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            # 4. Custom Node: ComfyUI-VideoHelperSuite (VHS - Dùng cho VideoCombine)
            if not vhs_node_dir.exists():
                logger.info(f"📥 Đang clone ComfyUI-VideoHelperSuite về {vhs_node_dir}...")
                try:
                    subprocess.run(
                        [
                            "git",
                            "clone",
                            "--depth",
                            "1",
                            "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
                            str(vhs_node_dir),
                        ],
                        check=True,
                    )
                except Exception as e:
                    logger.warning(f"Không thể clone VideoHelperSuite ({e}).")
            vhs_req = vhs_node_dir / "requirements.txt"
            if vhs_req.exists():
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "-r", str(vhs_req)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            try:
                marker_file.write_text("ok", encoding="utf-8")
            except Exception:
                pass

            self.patch_gguf_ops_compat()
            logger.info("✅ ComfyUI dependencies & custom nodes đã sẵn sàng.")
            return True
        except Exception as e:
            logger.error(f"❌ Lỗi khi chuẩn bị ComfyUI: {e}", exc_info=True)
            return False

    def patch_gguf_ops_compat(self) -> None:
        """Tự động vá ComfyUI-GGUF và ComfyUI core để tương thích hoàn toàn với MiniMax H3 và các mô hình mới nhất."""
        # 1. Patch ComfyUI-GGUF ops.py
        gguf_ops = self.custom_nodes_dir / "ComfyUI-GGUF" / "ops.py"
        if gguf_ops.exists():
            try:
                ops_code = gguf_ops.read_text(encoding="utf-8")
                modified = False
                if "out = self.forward_ggml_cast_weights(input, *args, **kwargs)" in ops_code:
                    if "try:" not in ops_code:
                        ops_code = ops_code.replace(
                            "out = self.forward_ggml_cast_weights(input, *args, **kwargs)",
                            "try:\n                out = self.forward_ggml_cast_weights(input, *args, **kwargs)\n            except TypeError:\n                out = self.forward_ggml_cast_weights(input)",
                        )
                        modified = True
                import re
                new_ops = re.sub(
                    r"def forward_ggml_cast_weights\s*\(\s*self\s*,\s*input\s*\)\s*:",
                    "def forward_ggml_cast_weights(self, input, *args, **kwargs):",
                    ops_code,
                )
                if new_ops != ops_code:
                    ops_code = new_ops
                    modified = True
                if modified:
                    gguf_ops.write_text(ops_code, encoding="utf-8")
                    logger.info("🔧 Đã tự động vá ComfyUI-GGUF ops.py (hỗ trợ input_act/kwargs từ ComfyUI).")
            except Exception as e:
                logger.warning(f"Không thể vá ComfyUI-GGUF ops.py: {e}")

        # 2. Patch ComfyUI-GGUF loader.py (cho phép Text Encoders nạp an toàn, không bị lỗi architecture/mmproj)
        gguf_loader = self.custom_nodes_dir / "ComfyUI-GGUF" / "loader.py"
        if gguf_loader.exists():
            try:
                loader_code = gguf_loader.read_text(encoding="utf-8")
                mod_loader = False

                # 2a. Bỏ ValueError khi text model architecture là incompatible with llama.cpp
                for l in loader_code.splitlines():
                    if "incompatible with llama.cpp" in l and l.strip().startswith("raise ValueError"):
                        loader_code = loader_code.replace(l, "            arch_str = 'qwen3'")
                        mod_loader = True
                        break

                # 2b. Nếu qwen3vl thiếu mmproj file, fallback sang vsd = {} thay vì raise ValueError
                for l in loader_code.splitlines():
                    if "requires a matching mmproj GGUF" in l and l.strip().startswith("raise ValueError"):
                        loader_code = loader_code.replace(l, "                    vsd = {}")
                        mod_loader = True
                        break

                if mod_loader:
                    compile(loader_code, "loader.py", "exec")
                    gguf_loader.write_text(loader_code, encoding="utf-8")
                    logger.info("🔧 Đã tự động vá ComfyUI-GGUF loader.py (hỗ trợ MiniMax/Qwen3 text encoders GGUF).")
            except Exception as e:
                logger.warning(f"Không thể vá ComfyUI-GGUF loader.py: {e}")

        # 3. Patch ComfyUI core comfy/sd.py (đảm bảo type='minimax' luôn instantiate MiniMaxH3Tokenizer & TEModel)
        comfy_sd = self.comfy_dir / "comfy" / "sd.py"
        if comfy_sd.exists():
            try:
                sd_code = comfy_sd.read_text(encoding="utf-8")
                target_str = 'elif clip_type == CLIPType.MINIMAX and "model.audio_decoder.projection.weight" in clip_data[0]:'
                patch_str = (
                    'elif clip_type == CLIPType.MINIMAX and "model.audio_decoder.projection.weight" in clip_data[0]:\n'
                    '            tokenizer_data["tokenizer_json"] = clip_data[0].pop("tokenizer_json", None)\n'
                    '            quant = comfy.utils.detect_layer_quantization(clip_data[0], "")\n'
                    '            if quant is not None:\n'
                    '                model_options = model_options.copy()\n'
                    '                model_options["quantization_metadata"] = quant\n'
                    '            clip_target.params["projection_config"] = comfy.text_encoders.minimax_music.detect_merged_config(clip_data[0])\n'
                    '            clip_target.clip = comfy.text_encoders.minimax_music.MiniMaxMusic3TEModel\n'
                    '            clip_target.tokenizer = comfy.text_encoders.minimax_music.MiniMaxMusic3Tokenizer\n'
                    '        elif clip_type == CLIPType.MINIMAX:\n'
                    '            detect = comfy.text_encoders.hunyuan_video.llama_detect(clip_data[0])\n'
                    '            clip_target.clip = comfy.text_encoders.minimax.te(**detect)\n'
                    '            clip_target.tokenizer = comfy.text_encoders.minimax.MiniMaxH3Tokenizer'
                )
                if 'elif clip_type == CLIPType.MINIMAX:\n            detect = comfy.text_encoders.hunyuan_video.llama_detect' not in sd_code:
                    if target_str in sd_code:
                        idx = sd_code.find(target_str)
                        next_elif = sd_code.find('elif te_model == TEModel.CLIP_G:', idx)
                        if next_elif != -1:
                            sd_code = sd_code[:idx] + patch_str + "\n        " + sd_code[next_elif:]
                            comfy_sd.write_text(sd_code, encoding="utf-8")
                            logger.info("🔧 Đã tự động vá ComfyUI comfy/sd.py (hỗ trợ CLIPType.MINIMAX H3).")
            except Exception as e:
                logger.warning(f"Không thể vá ComfyUI comfy/sd.py: {e}")

        # 4. Patch ComfyUI comfy/model_base.py (bảo vệ an toàn audio_scale fallback chống AttributeError)
        comfy_mb = self.comfy_dir / "comfy" / "model_base.py"
        if comfy_mb.exists():
            try:
                mb_code = comfy_mb.read_text(encoding="utf-8")
                if "return self.model_sampling.audio_scale" in mb_code:
                    mb_code = mb_code.replace(
                        "return self.model_sampling.audio_scale",
                        "return getattr(self.model_sampling, 'audio_scale', 1.0)",
                    )
                    compile(mb_code, "model_base.py", "exec")
                    comfy_mb.write_text(mb_code, encoding="utf-8")
                    logger.info("🔧 Đã tự động vá ComfyUI comfy/model_base.py (an toàn audio_scale fallback).")
            except Exception as e:
                logger.warning(f"Không thể vá ComfyUI comfy/model_base.py: {e}")


    # ==========================================================================
    # 2. MODEL DISCOVERY & SYMLINKING
    # ==========================================================================
    def auto_link_models(self, search_roots: Optional[List[str]] = None) -> Dict[str, str]:
        """Tự động tìm kiếm các model GGUF, SafeTensors trong Google Drive (Colab) hoặc /kaggle/input và symlink vào ComfyUI."""
        from config import COLAB_MODELS_DIR
        roots = search_roots or [
            COLAB_MODELS_DIR,
            "/content/models",
            "/kaggle/input",
            "/kaggle/input/datasets",
            "/kaggle/input/datasets/nguynxuncngde180528",
        ]

        linked = {}

        # Lấy danh sách model targets từ tất cả các manifest (Qwen, Wan, MiniMax H3)
        try:
            from visual import get_all_visual_targets
            model_targets = get_all_visual_targets(
                self.unet_dir,
                self.diffusion_dir,
                self.text_encoders_dir,
                self.clip_dir,
                self.vae_dir,
                loras_dir=self.loras_dir,
            )
        except Exception as me:
            logger.warning(f"Không thể nạp dynamic visual targets: {me}. Sử dụng danh sách tĩnh dự phòng.")
            model_targets = [
                # 🖼️ Qwen-Image 2.1
                ("qwen-image-2.1-Q4_K_M.gguf", [self.unet_dir, self.diffusion_dir]),
                ("qwen3vl_8b_fp8_scaled.safetensors", [self.text_encoders_dir, self.clip_dir]),
                ("qwen_image_2.1_vae_bf16.safetensors", [self.vae_dir]),
                # 🎬 Wan2.2 TI2V-5B
                ("Wan2.2-TI2V-5B-Q4_K_M.gguf", [self.unet_dir, self.diffusion_dir]),
                ("Wan2.2_VAE.safetensors", [self.vae_dir]),
                # 🎬 Wan2.2 T2V-A14B & I2V-A14B
                ("Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf", [self.unet_dir, self.diffusion_dir]),
                ("Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf", [self.unet_dir, self.diffusion_dir]),
                ("Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf", [self.unet_dir, self.diffusion_dir]),
                ("Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf", [self.unet_dir, self.diffusion_dir]),
                ("Wan2.1_VAE.safetensors", [self.vae_dir]),
                # 📝 UMT5-XXL
                ("umt5-xxl-encoder-Q4_K_M.gguf", [self.text_encoders_dir, self.clip_dir]),
                # 🚀 MiniMax H3 (FL2VA + Ref2VA + Turbo LoRA)
                ("MiniMax-H3-FL2VA-Q4_K_M.gguf", [self.unet_dir, self.diffusion_dir]),
                ("minimax_h3_fl2va_pruned-Q4_K.gguf", [self.unet_dir, self.diffusion_dir]),
                ("minimax_h3_fl2va_pruned_int8_convrot.safetensors", [self.unet_dir, self.diffusion_dir]),
                ("minimax_h3_ref2va_pruned_int8_convrot.safetensors", [self.unet_dir, self.diffusion_dir]),
                ("qwen3_vl_32b_q4_k_m.gguf", [self.text_encoders_dir, self.clip_dir]),
                ("qwen3vl_32b_minimax_h3-Q4_K_M.gguf", [self.text_encoders_dir, self.clip_dir]),
                ("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", [self.text_encoders_dir, self.clip_dir]),
                ("minimax_h3_video_vae_fp16.safetensors", [self.vae_dir]),
                ("minimax_h3_audio_vae_fp32.safetensors", [self.vae_dir]),
                ("minimax_h3_turbo_4step_pruned_comfyui.safetensors", [self.loras_dir]),
            ]

        for filename, dest_dirs in model_targets:
            src_file = self._find_file(filename, roots)
            if src_file:
                for dest_dir in dest_dirs:
                    dest_file = dest_dir / filename
                    if not dest_file.exists():
                        try:
                            os.symlink(str(src_file), str(dest_file))
                            logger.info(f"🔗 Linked: {filename} -> {dest_file}")
                        except Exception as e:
                            logger.warning(f"Lỗi symlink {src_file} -> {dest_file}: {e}")
                linked[filename] = str(src_file)
            else:
                logger.debug(f"ℹ️ Chưa tìm thấy model file '{filename}' trong {roots}")

        self.linked_models = linked
        return linked

    def _find_file(self, filename: str, roots: List[str]) -> Optional[Path]:
        """Tìm file trong các thư mục gốc."""
        for root_str in roots:
            root_path = Path(root_str)
            if not root_path.exists():
                continue
            # Kiểm tra trực tiếp
            direct = root_path / filename
            if direct.exists():
                return direct
            # Tìm đệ quy (giới hạn độ sâu để nhanh)
            try:
                for match in root_path.glob(f"**/{filename}"):
                    if match.is_file():
                        return match
            except Exception:
                pass
        return None

    # ==========================================================================
    # 3. PROCESS LIFECYCLE
    # ==========================================================================
    def is_running(self) -> bool:
        """Kiểm tra server ComfyUI có đang chạy và phản hồi HTTP không."""
        try:
            req = urllib.request.Request(self.base_url, headers={"User-Agent": "MMO-Factory"})
            with urllib.request.urlopen(req, timeout=1.5) as res:
                return res.status == 200
        except Exception:
            return False

    def start(self, timeout: float = 120.0) -> bool:
        """Khởi động ComfyUI backend trong tiến trình con trên GPU được chỉ định."""
        if self.is_running():
            logger.info("⚡ ComfyUI server đã hoạt động sẵn.")
            return True

        t_total = time.time()
        with self._setup_lock:
            if not ComfyUIManager._setup_done:
                self.ensure_installed()
                self.auto_link_models()
                ComfyUIManager._setup_done = True
            else:
                self.auto_link_models()

        log_path = Path(f"/kaggle/working/comfyui_gpu{self.cuda_device}.log") if Path("/kaggle/working").exists() else self.comfy_dir / f"comfyui_gpu{self.cuda_device}.log"
        log_file = open(log_path, "w", encoding="utf-8")

        env = os.environ.copy()
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            target_gpu = min(self.cuda_device, torch.cuda.device_count() - 1)
            if torch.cuda.device_count() > 1:
                env["CUDA_VISIBLE_DEVICES"] = str(target_gpu)
            else:
                env["CUDA_VISIBLE_DEVICES"] = "0"

        user_dir = self.comfy_dir / f"user_gpu{self.cuda_device}"
        user_dir.mkdir(parents=True, exist_ok=True)
        db_url = f"sqlite:///{user_dir}/comfyui.db"

        # Đảm bảo toàn bộ dependencies bắt buộc của ComfyUI (torchsde, comfy-aimdo, comfy-kitchen, v.v.) luôn đầy đủ
        req_file = self.comfy_dir / "requirements.txt"
        if req_file.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)], check=False)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "torchsde",
                "comfy-aimdo",
                "comfy-kitchen",
                "einops",
                "spandrel",
                "kornia",
                "scipy",
            ],
            check=False,
        )

        cmd = [
            sys.executable,
            str(self.comfy_dir / "main.py"),
            "--listen", self.host,
            "--port", str(self.port),
            "--disable-assets",
            "--database-url", db_url,
            "--user-directory", str(user_dir),
        ]

        logger.info(f"🚀 Đang khởi động ComfyUI trên GPU {self.cuda_device} (port {self.port}): {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            cwd=str(self.comfy_dir),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        # Chờ server sẵn sàng
        start_t = time.time()
        while time.time() - start_t < timeout:
            if self.is_running():
                self.startup_time_seconds = round(time.time() - t_total, 2)
                logger.info(f"🎉 ComfyUI server đã sẵn sàng tại {self.base_url} (sau {time.time() - start_t:.1f}s, tổng {self.startup_time_seconds}s)")
                return True
            if self.process.poll() is not None:
                log_file.flush()
                err_content = log_path.read_text(encoding="utf-8", errors="ignore")[-2000:]
                logger.error(f"❌ ComfyUI tiến trình bị thoát sớm (code {self.process.returncode}):\n{err_content}")
                raise RuntimeError(f"ComfyUI tiến trình bị thoát sớm (code {self.process.returncode}): {err_content}")
            time.sleep(1.0)

        err_content = log_path.read_text(encoding="utf-8", errors="ignore")[-2000:] if log_path.exists() else "No log"
        logger.error(f"❌ Hết thời gian chờ {timeout}s để ComfyUI khởi động.\n{err_content}")
        raise RuntimeError(f"Hết thời gian chờ {timeout}s để ComfyUI khởi động. Log: {err_content}")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Trả về thông tin chẩn đoán ComfyUI server và danh sách models đã symlink."""
        return {
            "is_running": self.is_running(),
            "base_url": self.base_url,
            "cuda_device": self.cuda_device,
            "startup_time_seconds": self.startup_time_seconds,
            "linked_models_count": len(self.linked_models),
            "linked_models": self.linked_models,
        }

    def stop(self):
        """Dừng tiến trình ComfyUI."""
        if self.process and self.process.poll() is None:
            logger.info("🛑 Đang dừng tiến trình ComfyUI...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

    def free_vram(self, unload_models: bool = True, free_memory: bool = True) -> bool:
        """
        Gọi endpoint /free của ComfyUI để giải phóng toàn bộ mô hình khỏi VRAM về CPU RAM.
        Tự động kích hoạt garbage collector và soft empty cache của PyTorch trên GPU 1.
        """
        try:
            url = f"{self.base_url}/free"
            payload = json.dumps({"unload_models": unload_models, "free_memory": free_memory}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as res:
                logger.info("🧹 Đã gọi ComfyUI /free thành công (giải phóng toàn bộ VRAM GPU 1 về Host RAM).")
                return res.status == 200
        except Exception as e:
            logger.warning(f"Lỗi khi gọi ComfyUI /free: {e}")
            return False

    # ==========================================================================
    # 4. REST API COMMUNICATION
    # ==========================================================================
    def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """Gửi workflow JSON lên ComfyUI /prompt và trả về prompt_id."""
        url = f"{self.base_url}/prompt"
        payload = json.dumps({"prompt": workflow, "client_id": str(uuid.uuid4())}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as res:
                data = json.loads(res.read().decode("utf-8"))
                return data["prompt_id"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            logger.error(f"ComfyUI /prompt HTTP Error {e.code}: {err_body}")
            raise RuntimeError(f"ComfyUI rejected prompt: {err_body}")
        except Exception as e:
            logger.error(f"Lỗi khi gửi prompt tới ComfyUI: {e}")
            raise

    def wait_for_completion(
        self,
        prompt_id: str,
        timeout: float = 1200.0,
        poll_interval: float = 1.5,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Poll kết quả từ /history/{prompt_id} cho đến khi hoàn thành.
        Trả về danh sách output items (images/videos metadata).
        """
        start_t = time.time()
        url = f"{self.base_url}/history/{prompt_id}"

        while time.time() - start_t < timeout:
            time.sleep(poll_interval)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "MMO-Factory"})
                with urllib.request.urlopen(req, timeout=10) as res:
                    history = json.loads(res.read().decode("utf-8"))

                if prompt_id not in history:
                    if callback:
                        callback({"type": "progress", "status": "queued", "elapsed": time.time() - start_t})
                    continue

                data = history[prompt_id]
                status = data.get("status", {})

                # Kiểm tra lỗi
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI execution failed: {msgs}")

                outputs = data.get("outputs", {})
                video_results = []
                image_results = []
                for node_id, node_out in outputs.items():
                    if "gifs" in node_out:
                        video_results.extend(node_out["gifs"])
                    if "videos" in node_out:
                        video_results.extend(node_out["videos"])
                    if "images" in node_out:
                        image_results.extend(node_out["images"])

                results = video_results + image_results
                if results:
                    return results

                if callback:
                    callback({"type": "progress", "status": "executing", "elapsed": time.time() - start_t})

            except urllib.error.URLError:
                pass

        raise TimeoutError(f"Hết thời gian chờ {timeout}s thực thi prompt {prompt_id}")

    def get_view_bytes(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> bytes:
        """Tải dữ liệu binary (ảnh/video) từ ComfyUI qua endpoint /view."""
        query = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type,
        })
        url = f"{self.base_url}/view?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": "MMO-Factory"})

        with urllib.request.urlopen(req, timeout=60) as res:
            return res.read()

    def save_input_image(
        self,
        image_data: Union[str, bytes, Image.Image],
        filename: Optional[str] = None,
    ) -> str:
        """Lưu ảnh vào thư mục ComfyUI input và trả về filename để dùng trong node LoadImage."""
        self.input_dir.mkdir(parents=True, exist_ok=True)
        fname = filename or f"input_{uuid.uuid4().hex[:8]}.png"
        target_path = self.input_dir / fname

        if isinstance(image_data, Image.Image):
            image_data.save(str(target_path), format="PNG")
        elif isinstance(image_data, bytes):
            with open(target_path, "wb") as f:
                f.write(image_data)
        elif isinstance(image_data, str):
            # Base64 string hoặc URL hoặc Local path
            if image_data.startswith("data:image"):
                image_data = image_data.split(",", 1)[1]
                img_bytes = base64.b64decode(image_data)
                with open(target_path, "wb") as f:
                    f.write(img_bytes)
            elif os.path.exists(image_data):
                shutil.copyfile(image_data, str(target_path))
            else:
                # Thử decode base64 thuần
                try:
                    img_bytes = base64.b64decode(image_data)
                    with open(target_path, "wb") as f:
                        f.write(img_bytes)
                except Exception:
                    # Nếu là URL
                    if image_data.startswith("http://") or image_data.startswith("https://"):
                        req = urllib.request.Request(image_data, headers={"User-Agent": "MMO-Factory"})
                        with urllib.request.urlopen(req, timeout=30) as res:
                            with open(target_path, "wb") as f:
                                f.write(res.read())
                    else:
                        raise ValueError(f"Không thể xử lý định dạng ảnh đầu vào: {image_data[:50]}...")

        return fname


# ==============================================================================
# 4. MULTI-INSTANCE ACCESSORS & MANAGEMENT
# ==============================================================================
def get_comfyui_manager(
    cuda_device: Optional[int] = None,
    port: Optional[int] = None,
    comfy_dir: Optional[str] = None,
) -> ComfyUIManager:
    """
    Lấy hoặc khởi tạo instance ComfyUIManager cho GPU cụ thể.
    - cuda_device=0 -> ComfyUI trên GPU 0 (port 8188)
    - cuda_device=1 -> ComfyUI trên GPU 1 (port 8189)
    Nếu hệ thống chỉ có 1 GPU, tự động fallback an toàn về GPU 0 (port 8188).
    """
    with ComfyUIManager._lock:
        import torch
        if cuda_device is None:
            dev_id = 0
        else:
            dev_id = int(cuda_device)

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            actual_device = min(dev_id, torch.cuda.device_count() - 1)
        else:
            actual_device = 0

        actual_port = port or (8188 + actual_device)

        if actual_device not in ComfyUIManager._instances:
            mgr = ComfyUIManager(
                comfy_dir=comfy_dir,
                port=actual_port,
                cuda_device=actual_device,
            )
            ComfyUIManager._instances[actual_device] = mgr

        return ComfyUIManager._instances[actual_device]


def get_all_comfyui_diagnostics() -> Dict[str, Any]:
    """Trả về trạng thái chẩn đoán của toàn bộ các ComfyUI instances đang vận hành."""
    with ComfyUIManager._lock:
        diag = {}
        for dev_id, mgr in ComfyUIManager._instances.items():
            diag[f"gpu_{dev_id}"] = mgr.get_diagnostics()
        if not diag:
            # Fallback nếu chưa khởi tạo instance nào
            mgr = get_comfyui_manager(cuda_device=0)
            diag["gpu_0"] = mgr.get_diagnostics()
        return diag


def stop_all_comfyui_managers():
    """Dừng toàn bộ các tiến trình ComfyUI đang chạy."""
    with ComfyUIManager._lock:
        for dev_id, mgr in ComfyUIManager._instances.items():
            try:
                mgr.stop()
            except Exception as e:
                logger.warning(f"Lỗi khi dừng ComfyUI GPU {dev_id}: {e}")


def free_all_comfyui_vram():
    """Gọi giải phóng VRAM trên toàn bộ các ComfyUI instances."""
    with ComfyUIManager._lock:
        for dev_id, mgr in ComfyUIManager._instances.items():
            if mgr.is_running():
                try:
                    mgr.free_vram()
                except Exception as e:
                    logger.warning(f"Lỗi giải phóng VRAM ComfyUI GPU {dev_id}: {e}")
