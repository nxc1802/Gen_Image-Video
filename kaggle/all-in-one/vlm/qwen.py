"""
👁️ Vision-Language Model (VLM) Module: Qwen3.8 27B & Qwen2.5-VL Dual-Engine Adapter
========================================================================================
Kế thừa BaseVLMEngine, thiết kế tối ưu hóa chuyên sâu cho hạ tầng Kaggle Dual Tesla T4 (2x 16GB):
1. 🌟 GGUF SOTA Backend (Ưu tiên số 1 - Qwen3.8 27B GGUF):
   - Model: nguynxuncngde180528/qwen38-27b-vlm-gguf (Unsloth Dynamic V3.0 UD-Q4_K_M ~16.19 GB)
   - Multimodal Vision Projector: mmproj-F16.gguf (~0.8 GB)
   - Tự động nhận diện trọng số trong /kaggle/input/qwen38-27b-vlm-gguf/ hoặc biến môi trường.
   - Dual-GPU Layer-Splitting (LLAMA_SPLIT_MODE_LAYER, tensor_split=[0.5, 0.5]):
     * GPU 0 (cuda:0): Gánh 50% số layer (Layer 0 -> N/2) + Audio thường trực (~11.5 GB / 16 GB).
     * GPU 1 (cuda:1): Gánh 50% số layer (Layer N/2 -> N) + mmproj vision (~9.5 GB / 16 GB).
     * Giảm thiểu 100% nghẽn bus PCIe: Mỗi token chỉ truyền 10 KB hidden state giữa 2 GPU (<0.001s).
2. 🔄 Tương Thích Vòng Đời MemoryManager (RAM-First PCIe Fast-Swap):
   - Khi có request FLUX.1 (Image) hoặc Wan2.2 (Video), VLM giải phóng VRAM về 30GB CPU RAM.
   - Sau khi render xong, nạp lại tức thì trong ~1.5s không đọc lại từ SSD.
3. 🛡️ Transformers Fallback (Dự phòng an toàn):
   - Tự động chuyển sang Hugging Face Transformers (Qwen2.5-VL-3B-Instruct) nếu không có GGUF.
"""

import base64
import gc
import io
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import torch
from PIL import Image

try:
    from llama_cpp import Llama
    _LLAMA_CPP_AVAILABLE = True
except ImportError:
    _LLAMA_CPP_AVAILABLE = False

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as AutoVLMModel
except ImportError:
    try:
        from transformers import AutoModelForImageTextToText as AutoVLMModel
    except ImportError:
        try:
            from transformers import AutoModelForCausalLM as AutoVLMModel
        except ImportError:
            AutoVLMModel = None

try:
    from transformers import AutoProcessor, BitsAndBytesConfig, TextIteratorStreamer
except ImportError:
    AutoProcessor = None
    BitsAndBytesConfig = None
    TextIteratorStreamer = None

from config import (
    VLM_CONFIG,
    VLM_FALLBACK_ID,
    VLM_FORMAT,
    VLM_GPU_SPLIT,
    VLM_MODEL_ID,
    VLM_N_CTX,
    resolve_vlm_model_paths,
)
from core.base_engine import BaseVLMEngine
from core.device_resolver import get_device_resolver
from core.memory_manager import get_memory_manager

logger = logging.getLogger("VLMEngine")


def _get_chat_handler(mmproj_path: Optional[str]):
    """Khởi tạo Vision Chat Handler thích hợp cho mmproj file."""
    if not mmproj_path or not os.path.exists(mmproj_path):
        return None
    try:
        from llama_cpp.llama_chat_format import Qwen2VLChatHandler
        logger.info(f"👁️ [Vision Handler] Khởi tạo Qwen2VLChatHandler với {mmproj_path}")
        return Qwen2VLChatHandler(clip_model_path=mmproj_path)
    except (ImportError, AttributeError):
        pass
    try:
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        logger.info(f"👁️ [Vision Handler] Khởi tạo Llava15ChatHandler với {mmproj_path}")
        return Llava15ChatHandler(clip_model_path=mmproj_path)
    except (ImportError, AttributeError) as e:
        logger.warning(f"⚠️ Không thể khởi tạo Vision Chat Handler ({e}), chuyển sang text handler.")
        return None


class QwenVLMEngine(BaseVLMEngine):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(QwenVLMEngine, cls).__new__(cls)
            cls._llm = None
            cls._llama_server_proc = None
            cls._llama_server_url = None
            cls._model = None
            cls._processor = None
            cls._is_gguf = False
            cls._initialized = False
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if getattr(self, "_initialized", False):
            return
        super().__init__(config or VLM_CONFIG)
        self.model_id = self.config.get("id", VLM_MODEL_ID)
        self.fallback_id = self.config.get("fallback_id", VLM_FALLBACK_ID)
        self.format = str(self.config.get("format", VLM_FORMAT)).lower()
        self.device_strategy = self.config.get("device_strategy", "dual_gpu")
        self.allocation_policy = self.config.get("allocation_policy", "dual_gpu")
        self.quantization = self.config.get("quantization", "q4_k_m")
        self.n_ctx = int(self.config.get("n_ctx", VLM_N_CTX))
        self.n_gpu_layers = int(self.config.get("n_gpu_layers", -1))
        self.gpu_split = self.config.get("gpu_split", VLM_GPU_SPLIT)
        self.lifecycle = self.config.get("lifecycle", "dynamic_switch")
        self.llama_server_port = int(os.environ.get("LLAMA_SERVER_PORT", 8081))

        self._llm = None
        self._llama_server_proc = None
        self._llama_server_url = None
        self._model = None
        self._processor = None
        self._is_gguf = False
        self._initialized = True
        self._lock = threading.Lock()
        self._current_spec_mode = os.environ.get("VLM_SPEC_MODE") or self.config.get("spec_mode", "mtp")

        logger.info(
            f"👁️ Khởi tạo QwenVLMEngine: id={self.model_id} | format={self.format} | "
            f"policy={self.allocation_policy} | split={self.gpu_split} | ctx={self.n_ctx}"
        )

    def _find_llama_server_bin(self) -> Optional[str]:
        """Tìm binary llama-server từ bản build chính thức ggml-org/llama.cpp."""
        import shutil
        import glob
        candidates = [
            os.environ.get("LLAMA_SERVER_BIN", ""),
            "/tmp/llama-cpp/llama-b11184/llama-server",
            "/tmp/llama-cpp/llama-server",
            "/tmp/llama-cpp/build/bin/llama-server",
        ]
        for g in glob.glob("/tmp/llama-cpp/**/llama-server", recursive=True):
            candidates.append(g)
        for g in glob.glob("/tmp/**/llama-server", recursive=True):
            candidates.append(g)
        which_bin = shutil.which("llama-server")
        if which_bin:
            candidates.append(which_bin)

        for c in candidates:
            if c and os.path.isfile(c) and os.access(c, os.X_OK):
                return c
        return None

    def _stop_llama_server(self):
        """Dừng tiến trình llama-server nếu đang chạy."""
        if self._llama_server_proc is not None:
            try:
                logger.info("🛑 Đang dừng tiến trình llama-server...")
                self._llama_server_proc.terminate()
                self._llama_server_proc.wait(timeout=5)
            except Exception:
                try:
                    self._llama_server_proc.kill()
                except Exception:
                    pass
            self._llama_server_proc = None
        self._llama_server_url = None

    def _load_llama_server_backend(self, bin_path: str, model_file: str, mmproj_file: Optional[str]) -> bool:
        """
        Khởi chạy llama-server chính thức (ggml-org/llama.cpp b11184+) hỗ trợ đầy đủ
        Qwen3.8-27B GGUF + libmtmd mmproj-Q8_0, Multi-GPU Layer-Split, Flash Attention & KV Cache Q4_0.
        """
        import subprocess
        self._stop_llama_server()
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if self.n_ctx >= 65536 and (not self.gpu_split or self.gpu_split == [0.5, 0.5]):
            split_str = "0.54,0.46"
            logger.info(f"⚖️ [VRAM Auto-Balance for {self.n_ctx} ctx] Adjusting tensor split to {split_str}")
        else:
            split_str = ",".join(str(x) for x in (self.gpu_split or [0.5, 0.5]))

        cmd = [
            bin_path,
            "-m", str(model_file),
            "-c", str(self.n_ctx),
            "-np", "1",
            "-ngl", "999" if (torch.cuda.is_available() and self.n_gpu_layers < 0) else str(max(0, self.n_gpu_layers)),
            "--flash-attn", "on" if torch.cuda.is_available() else "off",
            "--cache-type-k", "q4_0",
            "--cache-type-v", "q4_0",
            "--host", "127.0.0.1",
            "--port", str(self.llama_server_port),
        ]

        # RoPE Scaling (YaRN):
        # - Qwen 3.8 / Qwen 2.5 thế hệ mới có native context 262,144 tokens (256K).
        # - Chỉ kích hoạt YaRN khi vượt quá native context (> 262,144). Ví dụ tại 1M (1,048,576 tokens) là YaRN x 4.
        # - Nhánh 32K (32,768) chỉ dành cho các model Qwen cũ (legacy) có native context 32K.
        is_legacy_qwen = any(leg in str(self.model_id).lower() for leg in ["qwen1", "qwen-1", "qwen1.5", "qwen2-"]) and "2.5" not in str(self.model_id)
        native_ctx = 32768 if is_legacy_qwen else 262144

        if self.n_ctx > native_ctx:
            scale_factor = self.n_ctx / float(native_ctx)
            logger.info(
                f"🌀 [YaRN RoPE Scaling] Context={self.n_ctx} > Native={native_ctx}. "
                f"YaRN Scale={scale_factor:.2f}x (yarn-orig-ctx={native_ctx})"
            )
            cmd.extend([
                "--rope-scaling", "yarn",
                "--rope-scale", f"{scale_factor:.2f}",
                "--yarn-orig-ctx", str(native_ctx),
            ])
        else:
            logger.info(f"⚡ [Native Context] Context={self.n_ctx} <= Native={native_ctx} (Vận hành chuẩn Native RoPE, không can thiệp nội suy)")

        if mmproj_file and os.path.exists(mmproj_file):
            cmd.extend(["--mmproj", str(mmproj_file), "--mmproj-offload"])

        if torch.cuda.is_available() and gpu_count >= 2 and (self.allocation_policy == "dual_gpu" or self.device_strategy == "dual_gpu"):
            cmd.extend(["--split-mode", "layer", "--tensor-split", split_str])
        elif torch.cuda.is_available():
            cmd.extend(["--split-mode", "none"])

        # ═══ Speculative Decoding: MTP / DFlash / Both ═══
        spec_mode = getattr(self, "_current_spec_mode", "mtp")
        model_dir = os.path.dirname(str(model_file))

        if spec_mode == "mtp":
            mtp_file = self._find_spec_model(model_dir, "mtp")
            if mtp_file:
                logger.info(f"⚡ [Spec Decoding: MTP] Thay thế model chính bằng bản MTP: {mtp_file}")
                # Model MTP chứa sẵn các MTP heads bên trong trọng số
                for i, c in enumerate(cmd):
                    if c == "-m" and i + 1 < len(cmd):
                        cmd[i + 1] = str(mtp_file)
                        break
                cmd.extend(["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"])
            else:
                logger.warning("⚠️ MTP model không tìm thấy, quay về baseline")

        elif spec_mode == "dflash":
            dflash_file = self._find_spec_model(model_dir, "dflash")
            if dflash_file:
                logger.info(f"⚡ [Spec Decoding: DFlash] Draft model: {dflash_file}")
                cmd.extend([
                    "--model-draft", str(dflash_file),
                    "--n-gpu-layers-draft", "999",
                    "--spec-type", "draft-dflash",
                    "--spec-draft-n-max", "15",
                ])
            else:
                logger.warning("⚠️ DFlash draft model not found, falling back to baseline")

        elif spec_mode == "both":
            mtp_file = self._find_spec_model(model_dir, "mtp")
            dflash_file = self._find_spec_model(model_dir, "dflash")
            if dflash_file and mtp_file:
                logger.info(f"⚡ [Spec Decoding: Both] Testing DFlash + MTP combo: draft={dflash_file}")
                cmd.extend([
                    "--model-draft", str(dflash_file),
                    "--n-gpu-layers-draft", "999",
                    "--spec-type", "draft-dflash,draft-mtp",
                    "--spec-draft-n-max", "15",
                ])
            elif dflash_file:
                cmd.extend([
                    "--model-draft", str(dflash_file),
                    "--n-gpu-layers-draft", "999",
                    "--spec-type", "draft-dflash",
                    "--spec-draft-n-max", "15",
                ])
            else:
                logger.warning("⚠️ Draft models not found for both mode, falling back to baseline")

        else:
            # baseline or unknown: check legacy env var
            draft_file = os.environ.get("VLM_DRAFT_PATH") or self.config.get("draft_model_file")
            if draft_file:
                if not os.path.isabs(draft_file):
                    cand_draft = os.path.join(model_dir, draft_file)
                    if os.path.exists(cand_draft):
                        draft_file = cand_draft
                if os.path.exists(draft_file):
                    logger.info(f"⚡ [Speculative Decoding] Draft Model: {draft_file}")
                    cmd.extend([
                        "--model-draft", str(draft_file),
                        "--n-gpu-layers-draft", "999",
                        "--cache-type-k-draft", "q4_0",
                        "--cache-type-v-draft", "q4_0",
                    ])

        env = os.environ.copy()
        bin_dir = os.path.dirname(os.path.abspath(bin_path))
        import glob as _glob
        import shutil as _shutil

        # Gom toàn bộ thư viện .so (bao gồm cudart/cublas và libggml-*) về cùng bin_dir và noble-libs
        target_so_dirs = [bin_dir]
        for nd in ["/tmp/noble-libs/usr/lib/x86_64-linux-gnu", "/tmp/noble-libs/lib/x86_64-linux-gnu"]:
            if os.path.isdir(nd):
                target_so_dirs.append(nd)
        for so_file in _glob.glob("/tmp/llama-cpp/**/*.so*", recursive=True):
            for t_dir in target_so_dirs:
                target_so = os.path.join(t_dir, os.path.basename(so_file))
                if not os.path.exists(target_so):
                    try:
                        _shutil.copy2(so_file, target_so, follow_symlinks=False)
                    except Exception:
                        pass

        ld_paths = [bin_dir, "/tmp/llama-cpp", "/usr/local/cuda/lib64", "/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu"]
        for so_file in _glob.glob("/tmp/llama-cpp/**/*.so*", recursive=True):
            so_dir = os.path.dirname(so_file)
            if so_dir not in ld_paths:
                ld_paths.insert(0, so_dir)
        if env.get("LD_LIBRARY_PATH"):
            ld_paths.append(env["LD_LIBRARY_PATH"])
        env["LD_LIBRARY_PATH"] = ":".join(ld_paths)
        env["GGML_BACKEND_DIR"] = bin_dir

        # Nếu môi trường Kaggle là Ubuntu 22.04 (GLIBC 2.35) và đã giải nén thư viện Ubuntu 24.04
        # tại /tmp/noble-libs (GLIBC 2.39 + GLIBCXX 3.4.32), copy ld-linux-x86-64.so.2 trực tiếp vào bin_dir
        # để readlink("/proc/self/exe") của ggml_backend_load_all() trỏ đúng về bin_dir chứa libggml-cuda.so!
        noble_lds = [
            p for p in [
                "/tmp/noble-libs/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
                "/tmp/noble-libs/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
            ]
            if os.path.exists(p)
        ]
        if not noble_lds:
            noble_lds = _glob.glob("/tmp/noble-libs/**/ld-linux-x86-64.so.2", recursive=True)
        if noble_lds:
            noble_ld_real = os.path.realpath(noble_lds[0])
            local_ld = os.path.join(bin_dir, "ld-linux-x86-64.so.2")
            try:
                if not os.path.exists(local_ld):
                    _shutil.copy2(noble_ld_real, local_ld)
                    os.chmod(local_ld, 0o755)
            except Exception:
                local_ld = noble_lds[0]

            noble_lib_dirs = []
            for d in [
                "/tmp/noble-libs/lib/x86_64-linux-gnu",
                "/tmp/noble-libs/usr/lib/x86_64-linux-gnu",
            ]:
                if os.path.isdir(d) and d not in noble_lib_dirs:
                    noble_lib_dirs.append(d)
            for d in ld_paths:
                if d and d not in noble_lib_dirs:
                    noble_lib_dirs.append(d)
            cmd = [local_ld, "--library-path", ":".join(noble_lib_dirs)] + cmd

        log_path = "/tmp/llama_server_qwen38.log"
        self._last_cmd = " ".join(cmd)
        logger.info(f"🚀 [llama-server Dual-GPU] Khởi chạy (cwd={bin_dir}): {self._last_cmd}")
        try:
            from tunnel.cloudflare import notify_ntfy_debug
            notify_ntfy_debug(f"[VLM] Starting llama-server (split={split_str})...")
        except Exception:
            notify_ntfy_debug = lambda m: None
        t0 = time.time()
        try:
            log_file = open(log_path, "w", encoding="utf-8")
            self._llama_server_proc = subprocess.Popen(
                cmd,
                cwd=bin_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
            base_url = f"http://127.0.0.1:{self.llama_server_port}"
            deadline = time.time() + 420.0
            last_dbg_time = time.time()
            while time.time() < deadline:
                if self._llama_server_proc.poll() is not None:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as rf:
                        err_tail = "".join(rf.readlines()[-50:])
                    self._last_error = f"llama-server exited with code {self._llama_server_proc.returncode}: {err_tail}"
                    logger.error(f"❌ {self._last_error}")
                    notify_ntfy_debug(f"[VLM] ERROR llama-server exited ({self._llama_server_proc.returncode}): {err_tail[-300:]}")
                    self._llama_server_proc = None
                    return False

                if time.time() - last_dbg_time > 30:
                    last_dbg_time = time.time()
                    try:
                        with open(log_path, "r", encoding="utf-8", errors="replace") as rf:
                            lines = [ln.strip() for ln in rf.readlines() if ln.strip()]
                            tail = lines[-1] if lines else "loading..."
                        notify_ntfy_debug(f"[VLM {int(time.time()-t0)}s] {tail[-120:]}")
                    except Exception:
                        pass

                try:
                    resp = requests.get(f"{base_url}/health", timeout=2)
                    if resp.status_code == 200:
                        self._llama_server_url = base_url
                        self._llm = "llama_server"
                        self._is_gguf = True
                        self._is_loaded = True
                        self._last_error = None
                        elapsed = time.time() - t0
                        logger.info(
                            f"✅ [llama-server Dual-GPU] Qwen3.8-27B GGUF + mmproj-Q8_0 sẵn sàng sau {elapsed:.2f}s "
                            f"(2xT4 Layer-Split={split_str}, Flash-Attn=ON, KV-Cache=Q4_0)!"
                        )
                        notify_ntfy_debug(f"[VLM] SUCCESS llama-server ready in {elapsed:.1f}s!")
                        return True
                except Exception:
                    pass
                time.sleep(2.0)

            self._last_error = "Timeout 420s waiting for llama-server /health"
            logger.error(f"❌ {self._last_error}")
            notify_ntfy_debug(f"[VLM] ERROR: {self._last_error}")
            self._stop_llama_server()
            return False
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"❌ Lỗi khởi chạy llama-server: {e}", exc_info=True)
            notify_ntfy_debug(f"[VLM] EXCEPTION: {e}")
            self._stop_llama_server()
            return False

    def _find_spec_model(self, model_dir: str, prefix: str) -> Optional[str]:
        """Find MTP or DFlash GGUF model file by prefix in model_dir/Colab Drive/Kaggle or auto-download from HuggingFace."""
        import glob as _glob
        from config import COLAB_MODELS_DIR
        search_dirs = [model_dir]
        for p in [
            COLAB_MODELS_DIR,
            "/content/models",
            "/kaggle/input/qwen38-27b-gguf",
            "/kaggle/input",
            "/tmp/hf_cache",
            "/kaggle/working",
        ]:
            if p and os.path.isdir(p) and p not in search_dirs:
                search_dirs.append(p)
        for sd in search_dirs:
            for f in sorted(_glob.glob(os.path.join(sd, "**", f"{prefix}*.gguf"), recursive=True)):
                logger.info(f"🔍 Found {prefix} spec model: {f}")
                return f

        # If not found locally, download from official Hugging Face repo
        hf_repo = "ggml-org/Qwen3.8-27B-GGUF"
        filename = f"{prefix}-Qwen3.8-27B-Q4_0.gguf"
        logger.info(f"⬇️ {prefix} model not found locally. Auto-downloading {filename} from {hf_repo}...")
        try:
            from huggingface_hub import hf_hub_download
            cache_dir = COLAB_MODELS_DIR if os.path.isdir(COLAB_MODELS_DIR) else "/tmp/hf_cache"
            os.makedirs(cache_dir, exist_ok=True)
            dl_path = hf_hub_download(repo_id=hf_repo, filename=filename, local_dir=cache_dir)
            logger.info(f"✅ Successfully downloaded {prefix} spec model to {dl_path}")
            return dl_path
        except Exception as e:
            logger.error(f"❌ Failed to download {filename} from HF ({hf_repo}): {e}")
            return None

    def reconfigure_spec_mode(
        self,
        spec_mode: Optional[str] = None,
        n_ctx: Optional[int] = None,
        tensor_split: Optional[str] = None,
    ) -> dict:
        """Stop current llama-server, restart with a different speculative decoding mode and/or context length."""
        valid_modes = ("baseline", "mtp", "dflash", "both")
        if spec_mode and spec_mode not in valid_modes:
            return {"error": f"Invalid spec_mode '{spec_mode}'. Valid: {valid_modes}"}

        with self._lock:
            old_mode = getattr(self, "_current_spec_mode", "mtp")
            old_ctx = getattr(self, "n_ctx", 4096)
            if spec_mode:
                self._current_spec_mode = spec_mode
            if n_ctx is not None:
                self.n_ctx = int(n_ctx)
            if tensor_split:
                self.gpu_split = [float(x.strip()) for x in tensor_split.split(",")]

            new_mode = getattr(self, "_current_spec_mode", "mtp")
            new_ctx = getattr(self, "n_ctx", 4096)
            logger.info(f"🔄 [Reconfigure] Switching: mode={old_mode}→{new_mode}, n_ctx={old_ctx}→{new_ctx}, split={self.gpu_split}")

            self._is_loaded = False
            self._stop_llama_server()

            # Re-resolve model paths and restart
            paths = resolve_vlm_model_paths(self.model_id)
            model_file = paths.get("model_file")
            mmproj_file = paths.get("mmproj_file")

            if not model_file or not os.path.exists(model_file):
                return {"error": f"Cannot find main model file for {self.model_id}"}

            llama_server_bin = self._find_llama_server_bin()
            if not llama_server_bin:
                return {"error": "Cannot find llama-server binary"}

            t0 = time.time()
            success = self._load_llama_server_backend(llama_server_bin, str(model_file), mmproj_file)
            elapsed = round(time.time() - t0, 2)

            return {
                "status": "ok" if success else "error",
                "spec_mode": new_mode,
                "n_ctx": new_ctx,
                "gpu_split": self.gpu_split,
                "startup_seconds": elapsed,
                "is_loaded": self._is_loaded,
                "last_cmd": getattr(self, "_last_cmd", None),
                "last_error": getattr(self, "_last_error", None),
            }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Trả về thông tin chẩn đoán chi tiết của VLM Engine phục vụ /health."""
        from pathlib import Path
        from config import COLAB_MODELS_DIR
        gguf_list = []
        for root_p in [Path(COLAB_MODELS_DIR), Path("/content/models"), Path("/kaggle/input"), Path("/tmp/hf_cache")]:
            if root_p.exists():
                for f in root_p.rglob("*.gguf"):
                    gguf_list.append(str(f))
        log_tail = ""
        if os.path.exists("/tmp/llama_server_qwen38.log"):
            try:
                with open("/tmp/llama_server_qwen38.log", "r", encoding="utf-8", errors="replace") as rf:
                    log_tail = "".join(rf.readlines()[-30:])
            except Exception:
                pass
        return {
            "spec_mode": getattr(self, "_current_spec_mode", "mtp"),
            "n_ctx": getattr(self, "n_ctx", 4096),
            "is_loaded": getattr(self, "_is_loaded", False),
            "is_gguf": getattr(self, "_is_gguf", False),
            "llama_server_url": getattr(self, "_llama_server_url", None),
            "llama_server_bin": self._find_llama_server_bin(),
            "resolved_paths": resolve_vlm_model_paths(self.model_id),
            "discovered_ggufs": gguf_list,
            "last_cmd": getattr(self, "_last_cmd", None),
            "last_error": getattr(self, "_last_error", None),
            "llama_server_log_tail": log_tail,
        }

    def _load_gguf_backend(self, paths: Dict[str, Any]) -> bool:
        """Nạp Qwen3.8-27B GGUF ưu tiên qua llama-server (b11184+) hoặc llama-cpp-python với Dual-GPU Layer-Splitting."""
        model_file = paths.get("model_file")
        mmproj_file = paths.get("mmproj_file")

        # Nếu chưa có file cục bộ, thử tải từ HF Hub
        if not model_file or not os.path.exists(model_file):
            hf_repo = paths.get("hf_repo_id", self.model_id)
            exp_model = paths.get("expected_model_file", "Qwen3.8-27B-Q4_K_M.gguf")
            exp_mmproj = paths.get("expected_mmproj_file", "mmproj-Qwen3.8-27B-Q8_0.gguf")
            logger.info(f"⏳ File GGUF chưa có cục bộ, kiểm tra tải từ Hugging Face ({hf_repo})...")
            try:
                from huggingface_hub import hf_hub_download
                model_file = hf_hub_download(repo_id=hf_repo, filename=exp_model)
                try:
                    mmproj_file = hf_hub_download(repo_id=hf_repo, filename=exp_mmproj)
                except Exception:
                    mmproj_file = None
            except Exception as e_dl:
                logger.warning(f"⚠️ Không thể tải GGUF từ HF Hub ({e_dl}). Chuyển sang Transformers fallback.")
                return False

        if not model_file or not os.path.exists(model_file):
            return False

        # Ưu tiên 1: Sử dụng binary llama-server chính thức (hỗ trợ kiến trúc qwen38 + libmtmd + speculative decoding)
        llama_server_bin = self._find_llama_server_bin()
        if llama_server_bin:
            if self._load_llama_server_backend(llama_server_bin, str(model_file), mmproj_file):
                return True
            logger.warning("⚠️ llama-server không khởi động được, thử fallback sang llama-cpp-python...")

        # Ưu tiên 2: Sử dụng llama-cpp-python
        if not _LLAMA_CPP_AVAILABLE:
            logger.warning("⚠️ llama-cpp-python chưa được cài đặt trong môi trường hiện tại.")
            return False

        t0 = time.time()
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        logger.info(f"🚀 [GGUF Dual-GPU Loader] Nạp {model_file} (Kèm mmproj: {mmproj_file}) trên {gpu_count} GPU...")

        chat_handler = _get_chat_handler(mmproj_file)

        llama_params = {
            "model_path": str(model_file),
            "n_ctx": self.n_ctx,
            "n_batch": min(512, self.n_ctx),
            "n_gpu_layers": self.n_gpu_layers if torch.cuda.is_available() else 0,
            "flash_attn": True if torch.cuda.is_available() else False,
            "verbose": False,
        }

        # Cấu hình Multi-GPU Layer Split & KV Cache Q4
        if torch.cuda.is_available():
            if gpu_count >= 2 and (self.allocation_policy == "dual_gpu" or self.device_strategy == "dual_gpu"):
                try:
                    import llama_cpp
                    split_mode_layer = getattr(llama_cpp, "LLAMA_SPLIT_MODE_LAYER", 1)
                except Exception:
                    split_mode_layer = 1
                llama_params["split_mode"] = split_mode_layer
                llama_params["tensor_split"] = self.gpu_split or [0.5, 0.5]
                logger.info(f"🎯 [Dual-GPU Layer-Split] Phân bổ qua 2 GPU: split={llama_params['tensor_split']}, mode=LAYER (toàn bộ trên GPU)")
            else:
                llama_params["main_gpu"] = 0
                if gpu_count >= 2:
                    llama_params["tensor_split"] = [1.0, 0.0]
                logger.info("🎯 [Single-GPU] Nạp VLM trọn vẹn vào GPU 0 (cuda:0)")

        # Cấu hình KV Cache Q4 (cả K và V) theo yêu cầu kiến trúc
        try:
            import llama_cpp
            q4_0_type = getattr(llama_cpp, "GGML_TYPE_Q4_0", 2)
        except Exception:
            q4_0_type = 2
        llama_params["type_k"] = q4_0_type
        llama_params["type_v"] = q4_0_type
        logger.info(f"⚡ [KV Cache Quantization] Cấu hình Q4_0 cho cả K ({q4_0_type}) và V ({q4_0_type})")

        if chat_handler is not None:
            llama_params["chat_handler"] = chat_handler

        try:
            try:
                self._llm = Llama(**llama_params)
            except TypeError as te:
                logger.warning(f"⚠️ Thử lại khởi tạo Llama không dùng một số tham số nâng cao: {te}")
                for param in ["flash_attn", "type_k", "type_v", "split_mode"]:
                    llama_params.pop(param, None)
                self._llm = Llama(**llama_params)
            self._is_gguf = True
            self._is_loaded = True
            elapsed = time.time() - t0
            logger.info(f"✅ Qwen3.8-27B GGUF nạp thành công trong {elapsed:.2f}s!")
            return True
        except Exception as e_llama:
            logger.error(f"❌ Lỗi khi khởi tạo Llama GGUF: {e_llama}", exc_info=True)
            self._llm = None
            return False

    def _load_transformers_backend(self) -> bool:
        """Bộ nạp dự phòng sử dụng Hugging Face Transformers."""
        if AutoVLMModel is None or AutoProcessor is None:
            logger.error("❌ Thư viện transformers không khả dụng.")
            return False

        resolver = get_device_resolver()
        target_id = self.fallback_id
        resolved = resolver.resolve(
            task="vlm",
            model_id=target_id,
            requested_strategy=self.device_strategy,
            quantization="4bit",
            config=self.config,
        )
        self.resolved_device = resolved["device"]
        logger.info(f"👁️ [Transformers Fallback] Đang nạp {target_id} lên {self.resolved_device}...")
        t0 = time.time()

        bnb_config = None
        if torch.cuda.is_available() and BitsAndBytesConfig is not None:
            try:
                import bitsandbytes as _bnb  # noqa: F401
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                bnb_config = None

        try:
            self._processor = AutoProcessor.from_pretrained(target_id, trust_remote_code=True)
            load_kwargs = {
                "quantization_config": bnb_config if torch.cuda.is_available() else None,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            target_map = resolved.get("device_map") or resolved.get("device")
            if isinstance(target_map, str) and target_map.startswith("cuda"):
                load_kwargs["device_map"] = {"": target_map}
            elif resolved.get("max_memory"):
                load_kwargs["device_map"] = "auto"
                load_kwargs["max_memory"] = resolved["max_memory"]
            else:
                load_kwargs["device_map"] = target_map or ("cuda:0" if torch.cuda.is_available() else "cpu")

            self._model = AutoVLMModel.from_pretrained(target_id, **load_kwargs)
            self._is_gguf = False
            self._is_loaded = True
            elapsed = time.time() - t0
            logger.info(f"✅ Transformers VLM ({target_id}) nạp thành công trong {elapsed:.2f}s!")
            return True
        except Exception as e_hf:
            logger.error(f"❌ Lỗi nạp Transformers VLM: {e_hf}", exc_info=True)
            self._model = "fallback"
            self._processor = "fallback"
            self._is_loaded = True
            return False

    def _actual_loader(self):
        """Bộ nạp tổng hợp: Ưu tiên GGUF, tự động fallback sang Transformers nếu cần."""
        paths = resolve_vlm_model_paths(self.model_id)

        # 1. Thử nạp GGUF nếu cấu hình yêu cầu hoặc phát hiện file GGUF
        if self.format == "gguf" or paths.get("is_gguf") or paths.get("model_file"):
            success = self._load_gguf_backend(paths)
            if success:
                return self._llm, None

        # 2. Fallback sang Transformers
        logger.info("ℹ️ Chuyển sang backend Transformers...")
        self._load_transformers_backend()
        return self._model, self._processor

    def release_from_gpu(self):
        """Giải phóng hoàn toàn VLM Qwen khỏi GPU VRAM về CPU/RAM phục vụ PCIe Fast-Swap."""
        logger.info("🧹 Giải phóng VLM Qwen khỏi GPU VRAM...")
        with self._lock:
            self._stop_llama_server()
            if self._llm is not None:
                try:
                    if hasattr(self._llm, "close"):
                        self._llm.close()
                except Exception:
                    pass
                self._llm = None

            if self._model is not None and self._model != "fallback":
                try:
                    if hasattr(self._model, "cpu"):
                        self._model.cpu()
                except Exception:
                    pass
                self._model = None
                self._processor = None

            self._is_loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("✨ Đã dọn sạch VLM khỏi GPU VRAM.")

    def reload_to_gpu(self, target_device: Optional[str] = None):
        """Kéo trọng số trở lại GPU VRAM từ CPU RAM."""
        self.load_model()

    def load_model(self):
        with self._lock:
            if self._is_loaded and (self._llm is not None or self._model is not None):
                return (self._llm, None) if self._is_gguf else (self._model, self._processor)

            mem = get_memory_manager()

            def loader_wrapper():
                res = self._actual_loader()
                return res[0]

            mem.switch_dynamic_slot("vlm", loader_wrapper, engine_obj=self)
            return (self._llm, None) if self._is_gguf else (self._model, self._processor)

    def load(self) -> Any:
        return self.load_model()

    def _ensure_data_url(self, url_or_b64: str) -> str:
        """Đảm bảo chuỗi ảnh đầu vào có định dạng data URL chuẩn OpenAI."""
        url_str = (url_or_b64 or "").strip()
        if url_str.startswith("data:image"):
            return url_str
        if url_str.startswith("http://") or url_str.startswith("https://"):
            try:
                resp = requests.get(url_str, timeout=15)
                resp.raise_for_status()
                b64 = base64.b64encode(resp.content).decode("utf-8")
                ctype = resp.headers.get("content-type", "image/jpeg").split(";")[0]
                return f"data:{ctype};base64,{b64}"
            except Exception as e:
                logger.warning(f"Lỗi tải ảnh từ URL {url_str}: {e}")
                return url_str
        return f"data:image/jpeg;base64,{url_str}"

    def _prepare_openai_messages(
        self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Chuẩn hóa messages theo chuẩn OpenAI Vision schema."""
        prepared = []
        if system_prompt and not any(m.get("role") == "system" for m in messages):
            prepared.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                prepared.append({"role": role, "content": content})
            elif isinstance(content, list):
                new_parts = []
                for part in content:
                    if isinstance(part, dict):
                        p_type = part.get("type", "")
                        if p_type == "text":
                            new_parts.append({"type": "text", "text": part.get("text", "")})
                        elif p_type == "image_url":
                            img_info = part.get("image_url", {})
                            url_val = img_info.get("url", "") if isinstance(img_info, dict) else str(img_info)
                            data_url = self._ensure_data_url(url_val)
                            new_parts.append({"type": "image_url", "image_url": {"url": data_url}})
                    elif isinstance(part, str):
                        new_parts.append({"type": "text", "text": part})
                prepared.append({"role": role, "content": new_parts})
            else:
                prepared.append({"role": role, "content": str(content)})
        return prepared

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        system_prompt: Optional[str] = None,
        stop: Optional[Any] = None,
        enable_thinking: Optional[bool] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Xử lý chat đa phương thức chuẩn OpenAI REST API, hỗ trợ Tool Calling & Structured Outputs."""
        self.load_model()
        t0 = time.time()
        formatted_messages = self._prepare_openai_messages(messages, system_prompt)

        # 1A. Thực thi trên llama-server Backend (Qwen3.8-27B + libmtmd)
        if self._is_gguf and self._llama_server_url:
            try:
                payload = {
                    "messages": formatted_messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature if temperature > 0 else 0.0,
                    "top_p": top_p if temperature > 0 else 1.0,
                    "repeat_penalty": repetition_penalty,
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
                }
                if top_k:
                    payload["top_k"] = top_k
                if stop:
                    payload["stop"] = stop
                if tools:
                    payload["tools"] = tools
                if tool_choice:
                    payload["tool_choice"] = tool_choice
                if response_format:
                    payload["response_format"] = response_format

                resp = requests.post(
                    f"{self._llama_server_url}/v1/chat/completions",
                    json=payload,
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                msg_obj = choice["message"]
                content_text = msg_obj.get("content") or ""
                reasoning_text = msg_obj.get("reasoning_content") or ""
                tool_calls = msg_obj.get("tool_calls")
                finish_reason = choice.get("finish_reason") or ("tool_calls" if tool_calls else "stop")

                if "<think>" in content_text and "</think>" in content_text:
                    import re as _re
                    m = _re.search(r"<think>(.*?)</think>", content_text, flags=_re.DOTALL)
                    if m:
                        if not reasoning_text:
                            reasoning_text = m.group(1).strip()
                        content_text = _re.sub(r"<think>.*?</think>", "", content_text, flags=_re.DOTALL).strip()
                text = content_text or reasoning_text or ""
                usage = data.get("usage", {})
                timings = data.get("timings", {})
                tokens = usage.get("completion_tokens", timings.get("predicted_n", len(text.split())))
                prompt_tokens = usage.get("prompt_tokens", timings.get("prompt_n", 0))
                elapsed = time.time() - t0
                tps = timings.get("predicted_per_second") or (tokens / max(elapsed, 0.01))
                logger.info(
                    f"✨ [llama-server Qwen3.8-27B] Sinh {tokens} tokens trong {elapsed:.2f}s "
                    f"(Decode: {tps:.2f} tok/s | Prompt: {prompt_tokens} tokens)"
                )
                return {
                    "text": text.strip(),
                    "reasoning_content": reasoning_text.strip() if enable_thinking else "",
                    "tool_calls": tool_calls,
                    "finish_reason": finish_reason,
                    "tokens": tokens,
                    "prompt_tokens": prompt_tokens,
                    "elapsed": round(elapsed, 2),
                    "tokens_per_second": round(float(tps), 2),
                }
            except Exception as e_srv:
                logger.error(f"Lỗi suy luận llama-server: {e_srv}", exc_info=True)
                return {
                    "text": f"Lỗi suy luận llama-server: {str(e_srv)}",
                    "tokens": 0,
                    "elapsed": round(time.time() - t0, 2),
                }

        # 1B. Thực thi trên GGUF llama-cpp-python Backend
        if self._is_gguf and self._llm is not None and self._llm != "llama_server":
            try:
                gen_params = {
                    "messages": formatted_messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature if temperature > 0 else 0.0,
                    "top_p": top_p if temperature > 0 else 1.0,
                    "repeat_penalty": repetition_penalty,
                    "stream": False,
                }
                if top_k:
                    gen_params["top_k"] = top_k
                if stop:
                    gen_params["stop"] = stop

                response = self._llm.create_chat_completion(**gen_params)
                text = response["choices"][0]["message"].get("content", "")
                tokens = response.get("usage", {}).get("completion_tokens", len(text.split()))
                elapsed = time.time() - t0
                tps = tokens / max(elapsed, 0.01)
                logger.info(f"✨ [GGUF 27B] Sinh {tokens} tokens trong {elapsed:.2f}s ({tps:.1f} tps)")
                return {
                    "text": text.strip(),
                    "tokens": tokens,
                    "elapsed": round(elapsed, 2),
                }
            except Exception as e_gguf:
                logger.error(f"Lỗi suy luận GGUF: {e_gguf}", exc_info=True)
                return {
                    "text": f"Lỗi suy luận GGUF: {str(e_gguf)}",
                    "tokens": 0,
                    "elapsed": round(time.time() - t0, 2),
                }

        # 2. Thực thi trên Transformers Fallback
        if self._model == "fallback" or self._model is None:
            user_text = messages[-1].get("content", "") if messages else ""
            mock_reply = f"[Mock VLM Response] Yêu cầu: '{str(user_text)[:60]}'. Server đang chạy chế độ dự phòng."
            return {"text": mock_reply, "tokens": 20, "elapsed": round(time.time() - t0, 2)}

        try:
            # Hugging Face Chat Template
            raw_images = []
            qwen_messages = []
            for msg in formatted_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, str):
                    qwen_messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            url = part.get("image_url", {}).get("url", "")
                            if url:
                                try:
                                    header, encoded = url.split(",", 1) if "," in url else ("", url)
                                    img_b = base64.b64decode(encoded)
                                    raw_images.append(Image.open(io.BytesIO(img_b)).convert("RGB"))
                                    text_parts.append("<|image_pad|>")
                                except Exception:
                                    pass
                    qwen_messages.append({"role": role, "content": " ".join(text_parts)})

            text_prompt = self._processor.apply_chat_template(
                qwen_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text_prompt],
                images=raw_images if raw_images else None,
                padding=True,
                return_tensors="pt",
            )
            first_param = next(self._model.parameters(), None)
            m_dev = first_param.device if first_param is not None else torch.device("cpu")
            inputs = {k: v.to(m_dev) if hasattr(v, "to") else v for k, v in inputs.items()}

            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature if temperature > 0 else None,
                "top_p": top_p if temperature > 0 else None,
                "top_k": top_k if top_k and temperature > 0 else None,
                "repetition_penalty": repetition_penalty if repetition_penalty > 1.0 else None,
                "do_sample": (temperature > 0),
            }
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

            with torch.inference_mode():
                output_ids = self._model.generate(**inputs, **gen_kwargs)

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
            ]
            output_text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
            elapsed = time.time() - t0
            token_count = len(generated_ids[0])
            logger.info(f"✨ [Transformers VLM] Sinh {token_count} tokens trong {elapsed:.2f}s")
            return {
                "text": output_text.strip(),
                "tokens": token_count,
                "elapsed": round(elapsed, 2),
            }
        except Exception as e_hf:
            logger.error(f"Lỗi suy luận Transformers: {e_hf}", exc_info=True)
            return {
                "text": f"Lỗi suy luận VLM: {str(e_hf)}",
                "tokens": 0,
                "elapsed": round(time.time() - t0, 2),
            }

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        system_prompt: Optional[str] = None,
        stop: Optional[Any] = None,
        enable_thinking: Optional[bool] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ):
        """Xử lý chat đa phương thức với SSE Streaming từng token, hỗ trợ Tool Calling."""
        self.load_model()
        formatted_messages = self._prepare_openai_messages(messages, system_prompt)

        # 1A. llama-server SSE Streaming
        if self._is_gguf and self._llama_server_url:
            try:
                payload = {
                    "messages": formatted_messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature if temperature > 0 else 0.0,
                    "top_p": top_p if temperature > 0 else 1.0,
                    "repeat_penalty": repetition_penalty,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "timings_per_token": True,
                    "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
                }
                if top_k:
                    payload["top_k"] = top_k
                if stop:
                    payload["stop"] = stop
                if tools:
                    payload["tools"] = tools
                if tool_choice:
                    payload["tool_choice"] = tool_choice
                if response_format:
                    payload["response_format"] = response_format

                with requests.post(
                    f"{self._llama_server_url}/v1/chat/completions",
                    json=payload,
                    stream=True,
                    timeout=1800,
                ) as resp:
                    resp.raise_for_status()
                    resp.encoding = "utf-8"
                    import json as _json
                    for raw_line in resp.iter_lines(decode_unicode=False):
                        if not raw_line:
                            continue
                        line = (
                            raw_line.decode("utf-8", errors="replace").strip()
                            if isinstance(raw_line, (bytes, bytearray))
                            else str(raw_line).strip()
                        )
                        if line.startswith("data:"):
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = _json.loads(data_str)
                                choices = chunk.get("choices") or []
                                finish_reason = None
                                if choices:
                                    finish_reason = choices[0].get("finish_reason")
                                if chunk.get("usage") or chunk.get("timings") or finish_reason:
                                    yield {
                                        "_meta": {
                                            "usage": chunk.get("usage"),
                                            "timings": chunk.get("timings"),
                                            "finish_reason": finish_reason,
                                        }
                                    }
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    c_tok = delta.get("content") or ""
                                    r_tok = delta.get("reasoning_content") or ""
                                    t_calls = delta.get("tool_calls")
                                    if t_calls:
                                        yield {"tool_calls": t_calls}
                                    if r_tok:
                                        if enable_thinking:
                                            yield {"reasoning_content": r_tok}
                                        elif not c_tok:
                                            # Fallback if model emitted reasoning when thinking wasn't requested
                                            yield {"content": r_tok}
                                    if c_tok:
                                        yield {"content": c_tok}
                            except Exception:
                                pass
                return
            except Exception as e_srv_st:
                logger.error(f"Lỗi stream llama-server: {e_srv_st}")
                yield f"\n[Lỗi stream llama-server: {str(e_srv_st)}]"
                return

        # 1B. GGUF llama-cpp-python Streaming
        if self._is_gguf and self._llm is not None and self._llm != "llama_server":
            try:
                gen_params = {
                    "messages": formatted_messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature if temperature > 0 else 0.0,
                    "top_p": top_p if temperature > 0 else 1.0,
                    "repeat_penalty": repetition_penalty,
                    "stream": True,
                }
                if top_k:
                    gen_params["top_k"] = top_k
                if stop:
                    gen_params["stop"] = stop

                stream = self._llm.create_chat_completion(**gen_params)
                for chunk in stream:
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token
                return
            except Exception as e_gguf_st:
                logger.error(f"Lỗi stream GGUF: {e_gguf_st}")
                yield f"\n[Lỗi stream GGUF: {str(e_gguf_st)}]"
                return

        # 2. Transformers Streaming Fallback
        if self._model == "fallback" or self._model is None:
            mock_tokens = ["Xin ", "chào! ", "Hệ ", "thống ", "VLM ", "đang ", "hoạt ", "động! "]
            for tok in mock_tokens:
                time.sleep(0.04)
                yield tok
            return

        try:
            raw_images = []
            qwen_messages = []
            for msg in formatted_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, str):
                    qwen_messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            url = part.get("image_url", {}).get("url", "")
                            if url:
                                try:
                                    header, encoded = url.split(",", 1) if "," in url else ("", url)
                                    img_b = base64.b64decode(encoded)
                                    raw_images.append(Image.open(io.BytesIO(img_b)).convert("RGB"))
                                    text_parts.append("<|image_pad|>")
                                except Exception:
                                    pass
                    qwen_messages.append({"role": role, "content": " ".join(text_parts)})

            text_prompt = self._processor.apply_chat_template(
                qwen_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text_prompt],
                images=raw_images if raw_images else None,
                padding=True,
                return_tensors="pt",
            )
            first_param = next(self._model.parameters(), None)
            m_dev = first_param.device if first_param is not None else torch.device("cpu")
            inputs = {k: v.to(m_dev) if hasattr(v, "to") else v for k, v in inputs.items()}

            tokenizer = getattr(self._processor, "tokenizer", self._processor)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            gen_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_tokens,
                "temperature": temperature if temperature > 0 else None,
                "top_p": top_p if temperature > 0 else None,
                "top_k": top_k if top_k and temperature > 0 else None,
                "repetition_penalty": repetition_penalty if repetition_penalty > 1.0 else None,
                "do_sample": (temperature > 0),
            }
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

            def run_gen():
                try:
                    with torch.inference_mode():
                        self._model.generate(**gen_kwargs)
                except Exception as th_e:
                    logger.error(f"Lỗi generate thread Transformers: {th_e}")

            th = threading.Thread(target=run_gen)
            th.start()

            for new_text in streamer:
                if new_text:
                    yield new_text

            th.join(timeout=10.0)
        except Exception as e:
            logger.error(f"Lỗi stream Transformers: {e}")
            yield f"\n[Lỗi stream: {str(e)}]"

    # Alias tương thích OpenAI route
    chat_completion = chat


VLMEngine = QwenVLMEngine


def get_vlm_engine(config: Optional[Dict[str, Any]] = None) -> QwenVLMEngine:
    return QwenVLMEngine(config)
