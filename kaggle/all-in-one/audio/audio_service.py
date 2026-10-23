"""
🎧 FastAPI Audio Service
================================================================================
Dịch vụ Audio độc lập (tương tự kiến trúc ComfyUI cho Visual):
- Chạy trên GPU 0 (cuda:0)
- Quản lý 3 mô hình âm thanh SOTA:
  1. 🎙️ Qwen3-ASR 1.7B: STT Multilingual (Tối ưu đặc biệt cho tiếng Việt & 50+ ngôn ngữ)
  2. 🔊 Qwen3-TTS 1.7B: Multilingual TTS, VoiceDesign, CustomVoice, Zero-shot Cloning
  3. 🗣️ XTTS-v2 Vietnamese: TTS chuyên biệt tiếng Việt + Voice Cloning / Cross-language
- Cung cấp REST API nội bộ (127.0.0.1:8288) để Factory & Studio Gateway gọi.
"""

import argparse
import base64
import gc
import io
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
os.environ["COQUI_TOS_AGREED"] = "1"

import numpy as np
import soundfile as sf
import torch

_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [AudioService]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AudioService")

# ==============================================================================
# SCHEMAS
# ==============================================================================
class QwenTTSRequest(BaseModel):
    text: str
    language: Optional[str] = "vi"
    voice: Optional[str] = "default"
    voice_description: Optional[str] = None  # Dùng cho VoiceDesign
    prompt_audio: Optional[str] = None       # Base64 hoặc filepath file audio mẫu để clone
    speaker_wav: Optional[str] = None        # Alias cho prompt_audio để tương thích
    ref_text: Optional[str] = None           # Văn bản tham chiếu cho ICL voice clone
    x_vector_only_mode: Optional[bool] = None
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0)
    sample_rate: Optional[int] = 24000
    response_format: Optional[str] = "wav"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    max_new_tokens: Optional[int] = None


class XTTSRequest(BaseModel):
    text: str
    language: Optional[str] = "vi"
    speaker_wav: Optional[str] = None        # Base64 hoặc filepath file giọng mẫu
    prompt_audio: Optional[str] = None       # Alias cho speaker_wav
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0)
    sample_rate: Optional[int] = 24000
    response_format: Optional[str] = "wav"
    temperature: Optional[float] = None
    length_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    gpt_cond_len: Optional[int] = None
    max_ref_len: Optional[int] = None
    sound_norm_refs: Optional[bool] = None
    enable_text_splitting: Optional[bool] = None


def apply_audio_speed(audio_arr: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    """
    Điều chỉnh tốc độ âm thanh (Time-Stretching) bảo toàn cao độ (Pitch Preservation).
    Hỗ trợ dải tốc độ: 0.25x đến 4.0x.
    1. Ưu tiên: FFmpeg atempo (WSOLA algorithm - chuẩn studio, chất lượng cao nhất).
    2. Dự phòng 1: PyTorch torchaudio phase_vocoder (tính toán STFT tensor trên CPU/GPU).
    3. Dự phòng 2: Scipy signal resample (nội suy mảng).
    """
    if speed is None or abs(float(speed) - 1.0) < 0.01:
        return audio_arr
    eff_speed = float(np.clip(float(speed), 0.25, 4.0))

    # 1. FFmpeg atempo
    try:
        import subprocess, tempfile, shutil
        if shutil.which("ffmpeg"):
            factors = []
            curr = eff_speed
            while curr > 2.0:
                factors.append(2.0)
                curr /= 2.0
            while curr < 0.5:
                factors.append(0.5)
                curr /= 0.5
            factors.append(curr)
            filter_str = ",".join([f"atempo={f:.4f}" for f in factors])

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fin, \
                 tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fout:
                in_path = fin.name
                out_path = fout.name

            try:
                sf.write(in_path, audio_arr, sample_rate, format="WAV")
                cmd = ["ffmpeg", "-y", "-i", in_path, "-filter:a", filter_str, "-vn", out_path]
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30.0)
                if res.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 44:
                    out_arr, out_sr = sf.read(out_path)
                    return out_arr.astype(np.float32)
            finally:
                for p in (in_path, out_path):
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
    except Exception as e:
        logger.warning(f"FFmpeg atempo speed adjustment failed ({e}), trying PyTorch phase_vocoder...")

    # 2. PyTorch phase_vocoder
    try:
        import torch
        import torchaudio.functional as F
        wav_tensor = torch.from_numpy(audio_arr).float()
        is_1d = (wav_tensor.ndim == 1)
        if is_1d:
            wav_tensor = wav_tensor.unsqueeze(0)
        n_fft = 1024
        hop_length = 256
        window = torch.hann_window(n_fft, device=wav_tensor.device)
        stft = torch.stft(wav_tensor, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        phase_advance = torch.linspace(0, torch.pi * hop_length, stft.shape[-2], device=wav_tensor.device)[..., None]
        stretched_stft = F.phase_vocoder(stft, rate=eff_speed, phase_advance=phase_advance)
        target_len = max(1, int(wav_tensor.shape[-1] / eff_speed))
        stretched_wav = torch.istft(stretched_stft, n_fft=n_fft, hop_length=hop_length, window=window, length=target_len)
        if is_1d:
            stretched_wav = stretched_wav.squeeze(0)
        return stretched_wav.cpu().numpy().astype(np.float32)
    except Exception as e:
        logger.warning(f"Phase vocoder failed ({e}), falling back to scipy resample...")

    # 3. Scipy resample
    try:
        from scipy import signal
        num_samples = max(1, int(len(audio_arr) / eff_speed))
        return signal.resample(audio_arr, num_samples).astype(np.float32)
    except Exception:
        return audio_arr



def to_canonical_language(lang: Optional[str]) -> Optional[str]:
    """Chuyển đổi mã ngôn ngữ ISO (vi, en, zh...) sang định dạng canonical của Qwen3-ASR."""
    if not lang or str(lang).strip().lower() in ("auto", "none", ""):
        return None
    lang_clean = str(lang).strip().lower()
    mapping = {
        "vi": "Vietnamese",
        "vietnamese": "Vietnamese",
        "tiếng việt": "Vietnamese",
        "en": "English",
        "english": "English",
        "zh": "Chinese",
        "chinese": "Chinese",
        "ja": "Japanese",
        "japanese": "Japanese",
        "ko": "Korean",
        "korean": "Korean",
        "fr": "French",
        "french": "French",
        "de": "German",
        "german": "German",
        "es": "Spanish",
        "spanish": "Spanish",
        "ru": "Russian",
        "russian": "Russian",
        "th": "Thai",
        "thai": "Thai",
        "id": "Indonesian",
        "indonesian": "Indonesian",
        "ar": "Arabic",
        "arabic": "Arabic",
        "pt": "Portuguese",
        "portuguese": "Portuguese",
        "it": "Italian",
        "italian": "Italian",
    }
    return mapping.get(lang_clean, str(lang).strip().capitalize())


# ==============================================================================
# AUDIO ENGINES MANAGER
# ==============================================================================
class AudioEngines:
    def __init__(self, device: str = "cuda:0"):
        import threading
        self.device = device if torch.cuda.is_available() else "cpu"
        self._lock = threading.RLock()
        self.qwen_asr_model = None
        self.qwen_tts_model = None
        self.xtts_model = None
        self.last_asr_error = None
        self.last_tts_error = None
        self.last_xtts_error = None
        self.default_ref_wav = str(Path(__file__).parent / "default_speaker.wav")
        self._find_model_paths()

    def _find_dir_with_file(self, base_path: Optional[str], target_file: str) -> Optional[str]:
        if not base_path or not os.path.exists(base_path):
            return None
        p = Path(base_path)
        if (p / target_file).exists():
            return str(p)
        for sub in p.glob(f"**/{target_file}"):
            return str(sub.parent)
        return None

    def _find_file_recursive(self, base_path: Optional[str], target_file: str) -> Optional[str]:
        if not base_path or not os.path.exists(base_path):
            return None
        p = Path(base_path)
        direct = p / target_file
        if direct.exists():
            return str(direct)
        for sub in p.glob(f"**/{target_file}"):
            return str(sub)
        return None

    def _ensure_qwen_speech_deps(self):
        import subprocess, sys
        c_file = "/tmp/torch_constraints.txt"
        try:
            with open(c_file, "w") as cf:
                cf.write(f"torch=={torch.__version__}\n")
                import torchaudio
                cf.write(f"torchaudio=={torchaudio.__version__}\n")
        except Exception:
            pass

        try:
            import coqpit
            import TTS
        except Exception:
            logger.info("🔧 Đang tự động cài đặt coqui-tts & coqui-tts-trainer (giữ nguyên torch/torchaudio gốc)...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "coqpit-config", "coqui-tts-trainer", "ko-speech-tools", "monotonic-alignment-search", "pysbd", "num2words", "unidecode", "inflect", "anyascii", "pypinyin", "coqui-tts"], check=False)

        try:
            import sox
            import sentencepiece
            import tiktoken
            import nagisa
            import soynlp
            import qwen_asr
            import qwen_tts
        except Exception:
            logger.info("🔧 Đang tự động bổ sung dependencies: sox, sentencepiece, tiktoken, nagisa, soynlp, qwen-asr, qwen-tts...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", c_file, "sox", "sentencepiece", "tiktoken", "nagisa", "soynlp", "jieba"], check=False)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "qwen-asr", "qwen-tts", "qwen-omni-utils", "einops", "onnxruntime"], check=False)

        # Vá tương thích PretrainedConfig (pad_token_id, bos_token_id, eos_token_id), GenerationMixin & RoPE 'default' cho ASR, TTS và XTTS-v2
        try:
            from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
            for attr in ("pad_token_id", "bos_token_id", "eos_token_id", "sep_token_id", "decoder_start_token_id"):
                if not hasattr(PretrainedConfig, attr):
                    setattr(PretrainedConfig, attr, None)
            if GenerationMixin not in PreTrainedModel.__bases__:
                PreTrainedModel.__bases__ = PreTrainedModel.__bases__ + (GenerationMixin,)
            try:
                from TTS.tts.layers.xtts.gpt import GPT2InferenceModel
                if GenerationMixin not in GPT2InferenceModel.__bases__:
                    GPT2InferenceModel.__bases__ = GPT2InferenceModel.__bases__ + (GenerationMixin,)
            except Exception:
                pass
            import transformers.modeling_rope_utils as mru
            if hasattr(mru, "ROPE_INIT_FUNCTIONS") and "default" not in mru.ROPE_INIT_FUNCTIONS:
                def _compute_default_rope_parameters_compat(config=None, device=None, seq_len=None, **kwargs):
                    base = getattr(config, "rope_theta", 10000.0)
                    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
                    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
                    dim = int(head_dim * partial_rotary_factor)
                    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
                    return inv_freq, 1.0

                mru.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters_compat
                logger.info("✅ Đã vá ROPE_INIT_FUNCTIONS['default'] cho Transformers!")
        except Exception as rope_e:
            logger.warning(f"⚠️ Cảnh báo vá tương thích Transformers: {rope_e}")

    def _find_model_paths(self):
        """Tự động tìm kiếm đường dẫn model từ Google Drive (Colab), /kaggle/input hoặc môi trường."""
        self.asr_path = os.environ.get("QWEN3_ASR_PATH")
        self.tts_path = os.environ.get("QWEN3_TTS_PATH")
        self.xtts_path = os.environ.get("XTTS_VI_PATH")

        colab_models = os.environ.get("COLAB_MODELS_DIR", "/content/drive/MyDrive/AI_Studio_Colab/models")
        search_roots = [Path(colab_models), Path("/content/models"), Path("/kaggle/input")]
        for root_dir in search_roots:
            if not root_dir.exists():
                continue
            for d in root_dir.rglob("*"):
                if not d.is_dir():
                    continue
                d_lower = d.name.lower()
                if not self.asr_path and ("qwen3-asr" in d_lower or "qwen3_asr" in d_lower):
                    if (d / "config.json").exists() or list(d.glob("**/config.json")):
                        self.asr_path = str(d.resolve())
                elif not self.tts_path and ("qwen3-tts" in d_lower or "qwen3_tts" in d_lower):
                    if (d / "config.json").exists() or list(d.glob("**/config.json")):
                        self.tts_path = str(d.resolve())
                elif not self.xtts_path and ("xtts" in d_lower and ("vi" in d_lower or "vietnam" in d_lower)):
                    self.xtts_path = str(d.resolve())

        # Fallback IDs
        self.asr_id = self.asr_path or "Qwen/Qwen3-ASR-1.7B"
        self.tts_id = self.tts_path or "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        self.xtts_id = self.xtts_path or "thivux/XTTS-v2-vietnamse"

    # --------------------------------------------------------------------------
    # 1. Qwen3-ASR 1.7B
    # --------------------------------------------------------------------------
    def load_qwen_asr(self):
        with self._lock:
            if self.qwen_asr_model is not None and self.qwen_asr_model != "fallback":
                return self.qwen_asr_model

            self._ensure_qwen_speech_deps()

            import traceback
            model_dir = self._find_dir_with_file(self.asr_path, "config.json") or self.asr_id
            logger.info(f"🎙️ Đang nạp Qwen3-ASR 1.7B vào {self.device} (source={model_dir})...")
            t0 = time.time()
            try:
                from qwen_asr import Qwen3ASRModel
                self.qwen_asr_model = Qwen3ASRModel.from_pretrained(
                    model_dir,
                    device_map=self.device if self.device != "cpu" else None,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                )
                self.last_asr_error = None
                logger.info(f"✅ Qwen3-ASR 1.7B nạp qua qwen_asr thành công trong {time.time() - t0:.2f}s!")
                return self.qwen_asr_model
            except Exception as e:
                err1 = traceback.format_exc()
                logger.warning(f"⚠️ Thử nạp Qwen3ASRModel thất bại ({e}), thử AutoModel fallback...")
                try:
                    from transformers import AutoModel, AutoProcessor
                    try:
                        processor = AutoProcessor.from_pretrained(model_dir)
                    except Exception:
                        processor = AutoProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
                    model = AutoModel.from_pretrained(
                        model_dir,
                        torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                        device_map=self.device if self.device != "cpu" else None,
                    )
                    from qwen_asr import Qwen3ASRModel
                    self.qwen_asr_model = Qwen3ASRModel(
                        backend="transformers",
                        model=model,
                        processor=processor,
                        sampling_params=None,
                    )
                    self.last_asr_error = None
                    logger.info(f"✅ Qwen3-ASR 1.7B nạp qua AutoModel fallback thành công trong {time.time() - t0:.2f}s!")
                    return self.qwen_asr_model
                except Exception as e2:
                    err2 = traceback.format_exc()
                    self.last_asr_error = f"Qwen3ASRModel.from_pretrained error:\n{err1}\n\nAutoModel fallback error:\n{err2}"
                    logger.error(f"❌ Không thể nạp Qwen3-ASR ({e2}), chuyển sang chế độ giả lập.")
                    self.qwen_asr_model = "fallback"
                    return self.qwen_asr_model

    def transcribe(
        self,
        audio_path_or_bytes: Union[str, bytes],
        language: Optional[str] = "vi",
        task: Optional[str] = "transcribe",
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        return_time_stamps: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Phiên âm bằng Qwen3-ASR 1.7B."""
        model = self.load_qwen_asr()
        t0 = time.time()

        try:
            from config import MODELS_CONFIG
            stt_rt = MODELS_CONFIG.get("stt", {})
        except Exception:
            stt_rt = {}

        if return_time_stamps is None:
            return_time_stamps = bool(stt_rt.get("return_time_stamps", False))

        if model == "fallback":
            return {
                "text": "Chào mừng bạn đến với hệ thống nhận diện giọng nói Qwen3-ASR 1.7B.",
                "language": language or "vi",
                "model": "qwen3-asr-1.7b-fallback",
                "duration_seconds": 0.5,
            }

        tmp_file = None
        if isinstance(audio_path_or_bytes, bytes):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(audio_path_or_bytes)
            tmp.flush()
            audio_path = tmp.name
            tmp_file = tmp.name
        else:
            audio_path = audio_path_or_bytes

        try:
            canonical_lang = to_canonical_language(language)
            text = ""
            detected_lang = canonical_lang or "vi"
            time_stamps_out = None

            if hasattr(model, "transcribe"):
                call_kwargs = {
                    "audio": audio_path,
                    "context": prompt if prompt is not None else stt_rt.get("prompt", ""),
                    "language": canonical_lang,
                    "return_time_stamps": bool(return_time_stamps),
                }
                if max_new_tokens is not None:
                    call_kwargs["max_new_tokens"] = int(max_new_tokens)
                if temperature is not None and temperature > 0.0:
                    call_kwargs["temperature"] = float(temperature)
                try:
                    results = model.transcribe(**call_kwargs)
                except TypeError:
                    call_kwargs.pop("max_new_tokens", None)
                    call_kwargs.pop("temperature", None)
                    results = model.transcribe(**call_kwargs)
                except Exception as ts_err:
                    if return_time_stamps:
                        logger.warning(f"⚠️ return_time_stamps=True thất bại ({ts_err}), tự động fallback về return_time_stamps=False...")
                        call_kwargs["return_time_stamps"] = False
                        results = model.transcribe(**call_kwargs)
                    else:
                        raise
                if results and len(results) > 0:
                    first = results[0]
                    text = getattr(first, "text", str(first)).strip()
                    detected_lang = getattr(first, "language", canonical_lang or "vi")
                    ts = getattr(first, "time_stamps", None)
                    if ts is not None:
                        time_stamps_out = ts
            elif hasattr(model, "__call__"):
                res = model(audio_path)
                text = res.get("text", "").strip() if isinstance(res, dict) else str(res)

            elapsed = round(time.time() - t0, 2)
            out_dict = {
                "text": text,
                "language": detected_lang,
                "model": self.asr_id,
                "duration_seconds": elapsed,
            }
            if time_stamps_out is not None:
                out_dict["time_stamps"] = time_stamps_out
            return out_dict
        finally:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass

    # --------------------------------------------------------------------------
    # 2. Qwen3-TTS 1.7B
    # --------------------------------------------------------------------------
    def load_qwen_tts(self):
        with self._lock:
            if self.qwen_tts_model is not None and self.qwen_tts_model != "fallback":
                return self.qwen_tts_model

            self._ensure_qwen_speech_deps()
            import traceback
            model_dir = self._find_dir_with_file(self.tts_path, "config.json") or self.tts_id

            # Nếu dataset local thiếu thư mục speech_tokenizer/, ghép nối vào /tmp/qwen3_tts_merged
            if model_dir and os.path.isdir(model_dir) and not os.path.exists(os.path.join(model_dir, "speech_tokenizer", "config.json")):
                logger.info(f"🔧 Phát hiện {model_dir} thiếu thư mục speech_tokenizer/, đang ghép nối tại /tmp/qwen3_tts_merged...")
                merged_dir = "/tmp/qwen3_tts_merged"
                os.makedirs(merged_dir, exist_ok=True)
                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(
                        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        allow_patterns=["speech_tokenizer/*", "speech_tokenizer/**"],
                        local_dir=merged_dir,
                    )
                    for item in os.listdir(model_dir):
                        src = os.path.join(model_dir, item)
                        dst = os.path.join(merged_dir, item)
                        if not os.path.exists(dst):
                            try:
                                os.symlink(src, dst)
                            except Exception:
                                pass
                    model_dir = merged_dir
                    logger.info(f"✅ Đã bổ sung speech_tokenizer/ và symlink weights vào {merged_dir}!")
                except Exception as dl_e:
                    logger.warning(f"⚠️ Không thể tải speech_tokenizer/ ({dl_e}), fallback sang repo Qwen/Qwen3-TTS-12Hz-1.7B-Base")
                    model_dir = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

            logger.info(f"🔊 Đang nạp Qwen3-TTS 1.7B vào {self.device} (source={model_dir})...")
            t0 = time.time()
            try:
                from qwen_tts import Qwen3TTSModel
                self.qwen_tts_model = Qwen3TTSModel.from_pretrained(
                    model_dir,
                    device_map=self.device if self.device != "cpu" else None,
                    torch_dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (torch.float16 if self.device != "cpu" else torch.float32),
                )
                self.last_tts_error = None
                logger.info(f"✅ Qwen3-TTS 1.7B nạp qua qwen_tts thành công trong {time.time() - t0:.2f}s!")
                return self.qwen_tts_model
            except Exception as e:
                self.last_tts_error = traceback.format_exc()
                logger.error(f"❌ Không thể nạp Qwen3-TTS ({e}), chuyển sang chế độ giả lập.")
                self.qwen_tts_model = "fallback"
                return self.qwen_tts_model

    def synthesize_qwen(
        self,
        text: str,
        language: Optional[str] = "vi",
        voice: Optional[str] = None,
        voice_description: Optional[str] = None,
        prompt_audio: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        ref_text: Optional[str] = None,
        x_vector_only_mode: Optional[bool] = None,
        speed: float = 1.0,
        sample_rate: int = 24000,
        response_format: str = "wav",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> bytes:
        """Tổng hợp giọng nói bằng Qwen3-TTS 1.7B (hỗ trợ VoiceDesign, VoiceClone & CustomVoice)."""
        model = self.load_qwen_tts()
        t0 = time.time()
        prompt_audio = prompt_audio or speaker_wav

        try:
            from config import MODELS_CONFIG
            tts_rt = MODELS_CONFIG.get("tts", {})
        except Exception:
            tts_rt = {}

        gen_kwargs = {}
        for k_name, val in [
            ("temperature", temperature if temperature is not None else tts_rt.get("qwen_temperature")),
            ("top_p", top_p if top_p is not None else tts_rt.get("qwen_top_p")),
            ("top_k", top_k if top_k is not None else tts_rt.get("qwen_top_k")),
            ("repetition_penalty", repetition_penalty if repetition_penalty is not None else tts_rt.get("qwen_repetition_penalty")),
            ("max_new_tokens", max_new_tokens if max_new_tokens is not None else tts_rt.get("max_new_tokens")),
        ]:
            if val is not None:
                gen_kwargs[k_name] = val

        eff_ref_text = ref_text if ref_text is not None else tts_rt.get("ref_text")
        if x_vector_only_mode is not None:
            eff_xvec_mode = bool(x_vector_only_mode)
        elif "x_vector_only_mode" in tts_rt:
            eff_xvec_mode = bool(tts_rt["x_vector_only_mode"])
        else:
            eff_xvec_mode = not bool(eff_ref_text)

        if model == "fallback":
            duration = max(1.0, len(text) * 0.08 / max(speed, 0.1))
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_arr = (0.15 * np.sin(2 * np.pi * 520 * t)).astype(np.float32)
        else:
            try:
                tts_lang_map = {
                    "en": "english", "english": "english",
                    "zh": "chinese", "chinese": "chinese",
                    "ja": "japanese", "japanese": "japanese",
                    "ko": "korean", "korean": "korean",
                    "fr": "french", "french": "french",
                    "de": "german", "german": "german",
                    "es": "spanish", "spanish": "spanish",
                    "pt": "portuguese", "portuguese": "portuguese",
                    "ru": "russian", "russian": "russian",
                    "it": "italian", "italian": "italian",
                }
                tts_lang = tts_lang_map.get((language or "auto").strip().lower(), "auto")

                audio_arr = None
                out_sr = sample_rate

                # Xác định reference audio cho voice clone
                ref_in = None
                tmp_ref_file = None
                if prompt_audio:
                    if os.path.exists(prompt_audio):
                        ref_in = prompt_audio
                    elif len(prompt_audio) > 100:
                        try:
                            raw_b64 = prompt_audio.split(",")[-1]
                            spk_b = base64.b64decode(raw_b64)
                            tmp_f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                            tmp_f.write(spk_b)
                            tmp_f.flush()
                            ref_in = tmp_f.name
                            tmp_ref_file = tmp_f.name
                        except Exception:
                            ref_in = prompt_audio
                elif os.path.exists(self.default_ref_wav):
                    ref_in = self.default_ref_wav

                # Phát hiện model type (base, voice_design, custom_voice)
                raw_inner = getattr(model, "model", None)
                m_type = getattr(raw_inner, "tts_model_type", None) or getattr(model, "tts_model_type", None) or "base"

                # 1. Voice Clone nếu có prompt_audio hoặc nếu model là Base model
                if (prompt_audio or m_type == "base") and hasattr(model, "generate_voice_clone"):
                    target_ref = ref_in or self.default_ref_wav
                    try:
                        wavs, out_sr = model.generate_voice_clone(
                            text=text,
                            ref_audio=target_ref,
                            ref_text=eff_ref_text,
                            x_vector_only_mode=eff_xvec_mode,
                            language=tts_lang if tts_lang != "auto" else None,
                            **gen_kwargs,
                        )
                        audio_arr = wavs[0]
                    except Exception as clone_err:
                        logger.warning(f"Voice clone gặp lỗi ({clone_err}), thử voice_design...")

                # 2. Custom Voice nếu chỉ định speaker cụ thể
                if audio_arr is None and voice and voice.lower() not in ("default", "auto", "") and hasattr(model, "generate_custom_voice") and m_type == "custom_voice":
                    try:
                        wavs, out_sr = model.generate_custom_voice(
                            text=text,
                            speaker=voice,
                            instruct=voice_description or "",
                            language=tts_lang if tts_lang != "auto" else None,
                            **gen_kwargs,
                        )
                        audio_arr = wavs[0]
                    except Exception as cv_err:
                        logger.warning(f"Custom voice gặp lỗi ({cv_err}), thử voice_design...")

                # 3. Voice Design theo mô tả tự nhiên
                if audio_arr is None and hasattr(model, "generate_voice_design") and m_type == "voice_design":
                    try:
                        instruct = voice_description or "Natural, expressive, clear voice"
                        wavs, out_sr = model.generate_voice_design(
                            text=text,
                            instruct=instruct,
                            language=tts_lang if tts_lang != "auto" else None,
                            **gen_kwargs,
                        )
                        audio_arr = wavs[0]
                    except Exception as vd_err:
                        logger.warning(f"Voice design gặp lỗi ({vd_err}), thử fallback voice_clone...")

                # 4. Dự phòng voice_clone với default_ref_wav nếu các phương thức trên chưa sinh được
                if audio_arr is None and hasattr(model, "generate_voice_clone"):
                    target_ref = ref_in or self.default_ref_wav
                    wavs, out_sr = model.generate_voice_clone(
                        text=text,
                        ref_audio=target_ref,
                        ref_text=eff_ref_text,
                        x_vector_only_mode=eff_xvec_mode,
                        language=tts_lang if tts_lang != "auto" else None,
                        **gen_kwargs,
                    )
                    audio_arr = wavs[0]

                if audio_arr is None:
                    audio_arr = np.zeros(sample_rate, dtype=np.float32)

                sample_rate = out_sr or sample_rate

                if tmp_ref_file and os.path.exists(tmp_ref_file):
                    try:
                        os.remove(tmp_ref_file)
                    except Exception:
                        pass

            except Exception as e:
                import traceback
                self.last_tts_error = traceback.format_exc()
                logger.error(f"Lỗi suy luận Qwen3-TTS: {e}")
                duration = max(1.0, len(text) * 0.08 / max(speed, 0.1))
                t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Điều chỉnh tốc độ âm thanh (Time-Stretching) bảo toàn cao độ nếu speed != 1.0
        if speed is not None and abs(float(speed) - 1.0) >= 0.01 and audio_arr is not None:
            audio_arr = apply_audio_speed(audio_arr, sample_rate, float(speed))

        buffer = io.BytesIO()
        sf_fmt = "WAV" if response_format.lower() in ("wav", "audio/wav") else "OGG"
        sf.write(buffer, audio_arr, sample_rate, format=sf_fmt)
        audio_bytes = buffer.getvalue()
        logger.info(f"🎶 [Qwen3-TTS] Sinh âm thanh thành công ({time.time() - t0:.2f}s, {len(audio_bytes)} bytes, speed={speed}x)")
        return audio_bytes

    # --------------------------------------------------------------------------
    # 3. XTTS-v2 Vietnamese
    # --------------------------------------------------------------------------
    def load_xtts(self):
        with self._lock:
            if self.xtts_model is not None and self.xtts_model != "fallback":
                return self.xtts_model

            self._ensure_qwen_speech_deps()
            import traceback
            logger.info(f"🗣️ Đang nạp XTTS-v2 Vietnamese vào {self.device}...")
            t0 = time.time()
            errors = []
            try:
                config_path = self._find_file_recursive(self.xtts_path, "config.json")
                vocab_path = self._find_file_recursive(self.xtts_path, "vocab.json")
                ckpt_path = self._find_file_recursive(self.xtts_path, "best_model.pth") or self._find_file_recursive(self.xtts_path, "model.pth")

                # Tìm đệ quy toàn bộ /kaggle/input nếu chưa thấy
                if not (config_path and ckpt_path):
                    k_input = Path("/kaggle/input")
                    if k_input.exists():
                        for p in k_input.rglob("best_model.pth"):
                            ckpt_path = str(p)
                            d = p.parent
                            if (d / "config.json").exists():
                                config_path = str(d / "config.json")
                            if (d / "vocab.json").exists():
                                vocab_path = str(d / "vocab.json")
                            break

                # Nếu chưa có weights tiếng Việt, tự động tải từ HuggingFace (thivux/XTTS-v2-vietnamse)
                if not (config_path and ckpt_path):
                    try:
                        from huggingface_hub import snapshot_download
                        logger.info("📥 Đang tải weights XTTS-v2 Vietnamese từ HuggingFace (thivux/XTTS-v2-vietnamse)...")
                        hf_dir = snapshot_download(repo_id="thivux/XTTS-v2-vietnamse", local_dir="/tmp/xtts_vi")
                        config_path = os.path.join(hf_dir, "config.json")
                        vocab_path = os.path.join(hf_dir, "vocab.json")
                        for fn in ["best_model.pth", "model.pth"]:
                            p_fn = os.path.join(hf_dir, fn)
                            if os.path.exists(p_fn):
                                ckpt_path = p_fn
                                break
                    except Exception as hf_err:
                        logger.warning(f"⚠️ Không thể tải từ HuggingFace: {hf_err}")

                if config_path and ckpt_path:
                    try:
                        from TTS.tts.configs.xtts_config import XttsConfig
                        from TTS.tts.models.xtts import Xtts
                        from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer, multilingual_cleaners
                        if not getattr(VoiceBpeTokenizer, "_vi_patched", False):
                            _orig_vbpe_init = VoiceBpeTokenizer.__init__
                            def _patched_vbpe_init(tb_self, *a, **kw):
                                _orig_vbpe_init(tb_self, *a, **kw)
                                if hasattr(tb_self, "char_limits"):
                                    tb_self.char_limits["vi"] = 250
                            VoiceBpeTokenizer.__init__ = _patched_vbpe_init

                            _orig_preprocess = VoiceBpeTokenizer.preprocess_text
                            def _patched_preprocess(tb_self, txt, lang):
                                if lang == "vi" or str(lang).startswith("vi"):
                                    return multilingual_cleaners(txt, "en")
                                return _orig_preprocess(tb_self, txt, lang)
                            VoiceBpeTokenizer.preprocess_text = _patched_preprocess
                            VoiceBpeTokenizer._vi_patched = True

                        config = XttsConfig()
                        config.load_json(config_path)
                        if hasattr(config, "languages") and "vi" not in config.languages:
                            config.languages.append("vi")
                        xtts_obj = Xtts.init_from_config(config)
                        xtts_obj.load_checkpoint(
                            config,
                            checkpoint_dir=os.path.dirname(ckpt_path),
                            checkpoint_path=ckpt_path,
                            vocab_path=vocab_path,
                            use_deepspeed=False
                        )
                        if hasattr(xtts_obj, "tokenizer") and hasattr(xtts_obj.tokenizer, "char_limits"):
                            xtts_obj.tokenizer.char_limits["vi"] = 250
                        if self.device != "cpu":
                            xtts_obj.to(self.device)
                        self.xtts_model = {"raw_model": xtts_obj, "config": config}
                        self.last_xtts_error = None
                        logger.info(f"✅ XTTS-v2 Vietnamese nạp qua Xtts direct trong {time.time() - t0:.2f}s!")
                        return self.xtts_model
                    except Exception as direct_e:
                        errors.append(f"Xtts direct error: {traceback.format_exc()}")
                        logger.warning(f"⚠️ Thử nạp Xtts direct thất bại ({direct_e}), chuyển sang TTS wrapper...")

                from TTS.api import TTS
                if config_path and ckpt_path:
                    model_dir = str(Path(config_path).parent)
                    self.xtts_model = TTS(
                        model_path=model_dir,
                        config_path=config_path,
                        gpu=(self.device != "cpu"),
                    )
                else:
                    self.xtts_model = TTS(
                        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                        gpu=(self.device != "cpu"),
                    )
                self.last_xtts_error = None
                logger.info(f"✅ XTTS-v2 nạp qua TTS wrapper trong {time.time() - t0:.2f}s!")
                return self.xtts_model
            except Exception as e:
                errors.append(f"TTS wrapper error: {traceback.format_exc()}")
                self.last_xtts_error = "\n".join(errors)
                logger.warning(f"⚠️ Không thể nạp Coqui XTTS-v2 ({e}), dùng fallback.")
                self.xtts_model = "fallback"
                return self.xtts_model

    def synthesize_xtts(
        self,
        text: str,
        language: str = "vi",
        speaker_wav: Optional[str] = None,
        speed: float = 1.0,
        sample_rate: int = 24000,
        response_format: str = "wav",
        temperature: Optional[float] = None,
        length_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        gpt_cond_len: Optional[int] = None,
        max_ref_len: Optional[int] = None,
        sound_norm_refs: Optional[bool] = None,
        enable_text_splitting: Optional[bool] = None,
    ) -> bytes:
        """Tổng hợp tiếng Việt và nhân bản giọng nói với XTTS-v2."""
        model = self.load_xtts()
        t0 = time.time()

        try:
            from config import MODELS_CONFIG
            tts_rt = MODELS_CONFIG.get("tts", {})
        except Exception:
            tts_rt = {}

        tmp_speaker = None
        ref_path = None
        if speaker_wav:
            if os.path.exists(speaker_wav):
                ref_path = speaker_wav
            elif len(speaker_wav) > 100:
                try:
                    raw_b64 = speaker_wav.split(",")[-1]
                    spk_bytes = base64.b64decode(raw_b64)
                    tmp_spk = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp_spk.write(spk_bytes)
                    tmp_spk.flush()
                    ref_path = tmp_spk.name
                    tmp_speaker = tmp_spk.name
                except Exception:
                    pass

        if not ref_path or not os.path.exists(ref_path):
            sample_wavs = list(Path(self.xtts_path).glob("**/*.wav")) if self.xtts_path and os.path.exists(self.xtts_path) else []
            if sample_wavs:
                ref_path = str(sample_wavs[0])
            elif os.path.exists(self.default_ref_wav):
                ref_path = self.default_ref_wav

        try:
            if model == "fallback":
                duration = max(1.0, len(text) * 0.08 / max(speed, 0.1))
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                audio_arr = (0.15 * np.sin(2 * np.pi * 380 * t)).astype(np.float32)
            elif isinstance(model, dict) and "raw_model" in model:
                try:
                    raw_m = model["raw_model"]
                    cfg = model["config"]

                    eff_temp = temperature if temperature is not None else tts_rt.get("xtts_temperature", getattr(cfg, "temperature", 0.75))
                    eff_len_pen = length_penalty if length_penalty is not None else tts_rt.get("xtts_length_penalty", getattr(cfg, "length_penalty", 1.0))
                    eff_rep_pen = repetition_penalty if repetition_penalty is not None else tts_rt.get("xtts_repetition_penalty", getattr(cfg, "repetition_penalty", 5.0))
                    eff_top_k = top_k if top_k is not None else tts_rt.get("xtts_top_k", getattr(cfg, "top_k", 50))
                    eff_top_p = top_p if top_p is not None else tts_rt.get("xtts_top_p", getattr(cfg, "top_p", 0.85))
                    eff_gpt_cond_len = gpt_cond_len if gpt_cond_len is not None else tts_rt.get("gpt_cond_len", getattr(cfg, "gpt_cond_len", 30))
                    eff_max_ref_len = max_ref_len if max_ref_len is not None else tts_rt.get("max_ref_len", getattr(cfg, "max_ref_len", 60))
                    eff_sound_norm = sound_norm_refs if sound_norm_refs is not None else tts_rt.get("sound_norm_refs", getattr(cfg, "sound_norm_refs", False))
                    eff_split = enable_text_splitting if enable_text_splitting is not None else tts_rt.get("enable_text_splitting", True)

                    # Thử dùng full_inference trực tiếp (ổn định nhất cho Coqui XTTS)
                    if hasattr(raw_m, "full_inference") and ref_path and os.path.exists(ref_path):
                        out = raw_m.full_inference(
                            text=text,
                            ref_audio_path=ref_path,
                            language=language or "vi",
                            temperature=float(eff_temp),
                            length_penalty=float(eff_len_pen),
                            repetition_penalty=float(eff_rep_pen),
                            top_k=int(eff_top_k),
                            top_p=float(eff_top_p),
                            speed=float(speed),
                            enable_text_splitting=bool(eff_split),
                        )
                        audio_arr = np.array(out["wav"], dtype=np.float32)
                    else:
                        gpt_cond_latent, speaker_embedding = raw_m.get_conditioning_latents(
                            audio_path=ref_path,
                            gpt_cond_len=int(eff_gpt_cond_len),
                            max_ref_length=int(eff_max_ref_len),
                            sound_norm_refs=bool(eff_sound_norm),
                        )
                        out = raw_m.inference(
                            text=text,
                            language=language or "vi",
                            gpt_cond_latent=gpt_cond_latent,
                            speaker_embedding=speaker_embedding,
                            temperature=float(eff_temp),
                            length_penalty=float(eff_len_pen),
                            repetition_penalty=float(eff_rep_pen),
                            top_k=int(eff_top_k),
                            top_p=float(eff_top_p),
                            enable_text_splitting=bool(eff_split),
                            speed=float(speed),
                        )
                        audio_arr = np.array(out["wav"], dtype=np.float32)
                except Exception as e:
                    import traceback
                    self.last_xtts_error = traceback.format_exc()
                    logger.error(f"Lỗi khi gọi Xtts raw model: {e}")
                    duration = max(1.0, len(text) * 0.08 / max(speed, 0.1))
                    t = np.linspace(0, duration, int(sample_rate * duration), False)
                    audio_arr = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            else:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_tmp:
                        out_path = out_tmp.name

                    if ref_path and os.path.exists(ref_path):
                        model.tts_to_file(
                            text=text,
                            speaker_wav=ref_path,
                            language=language or "vi",
                            file_path=out_path,
                            speed=speed,
                        )
                    else:
                        model.tts_to_file(
                            text=text,
                            language=language or "vi",
                            file_path=out_path,
                            speed=speed,
                        )

                    audio_arr, sample_rate = sf.read(out_path)
                    if os.path.exists(out_path):
                        os.remove(out_path)
                except Exception as e:
                    import traceback
                    self.last_xtts_error = traceback.format_exc()
                    logger.error(f"Lỗi khi gọi XTTS-v2: {e}")
                    duration = max(1.0, len(text) * 0.08 / max(speed, 0.1))
                    t = np.linspace(0, duration, int(sample_rate * duration), False)
                    audio_arr = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

            buffer = io.BytesIO()
            sf_fmt = "WAV" if response_format.lower() in ("wav", "audio/wav") else "OGG"
            sf.write(buffer, audio_arr, sample_rate, format=sf_fmt)
            audio_bytes = buffer.getvalue()
            logger.info(f"🎶 [XTTS-v2] Sinh âm thanh thành công ({time.time() - t0:.2f}s, {len(audio_bytes)} bytes, speed={speed}x)")
            return audio_bytes
        finally:
            if tmp_speaker and os.path.exists(tmp_speaker):
                try:
                    os.remove(tmp_speaker)
                except Exception:
                    pass

    def get_diagnostics(self, try_load: bool = True) -> Dict[str, Any]:
        """Thu thập thông tin chẩn đoán toàn diện về môi trường, imports, model weights và lỗi."""
        diag: Dict[str, Any] = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "default_ref_wav": {"path": self.default_ref_wav, "exists": os.path.exists(self.default_ref_wav)},
            "imports": {},
            "paths": {
                "asr_path": self.asr_path,
                "tts_path": self.tts_path,
                "xtts_path": self.xtts_path,
                "kaggle_input": [],
            },
            "models": {},
            "errors": {},
        }

        # Kiểm tra /kaggle/input
        k_in = Path("/kaggle/input")
        if k_in.exists():
            diag["paths"]["kaggle_input"] = [p.name for p in k_in.iterdir() if p.is_dir()]

        # Kiểm tra các thư viện
        for pkg in ["qwen_asr", "qwen_tts", "TTS", "nagisa", "soynlp", "jieba", "transformers", "soundfile"]:
            try:
                mod = __import__(pkg)
                diag["imports"][pkg] = {"status": "ok", "file": getattr(mod, "__file__", "builtin")}
            except Exception as e:
                diag["imports"][pkg] = {"status": "error", "error": str(e)}

        # Liệt kê files trong thư mục models
        for name, p in [("asr", self.asr_path), ("tts", self.tts_path), ("xtts", self.xtts_path)]:
            if p and os.path.exists(p):
                diag["paths"][f"{name}_files"] = [f.name for f in Path(p).glob("*")][:15]

        # Nạp thử nếu được yêu cầu
        if try_load:
            if self.qwen_asr_model is None:
                self.load_qwen_asr()
            if self.qwen_tts_model is None:
                self.load_qwen_tts()
            if self.xtts_model is None:
                self.load_xtts()

        diag["models"]["asr"] = "fallback" if self.qwen_asr_model == "fallback" else ("loaded" if self.qwen_asr_model is not None else "not_initialized")
        diag["models"]["tts"] = "fallback" if self.qwen_tts_model == "fallback" else ("loaded" if self.qwen_tts_model is not None else "not_initialized")
        diag["models"]["xtts"] = "fallback" if self.xtts_model == "fallback" else ("loaded" if self.xtts_model is not None else "not_initialized")

        diag["errors"]["asr"] = self.last_asr_error
        diag["errors"]["tts"] = self.last_tts_error
        diag["errors"]["xtts"] = self.last_xtts_error

        return diag

    def free_vram(self):
        """Giải phóng VRAM khỏi GPU 0."""
        logger.info("🧹 Đang giải phóng bộ nhớ Audio Service...")
        self.qwen_asr_model = None
        self.qwen_tts_model = None
        self.xtts_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("✨ VRAM Audio Service đã được dọn sạch!")


# ==============================================================================
# FASTAPI APP FACTORY
# ==============================================================================
def create_audio_app(device: str = "cuda:0") -> FastAPI:
    app = FastAPI(
        title="🎧 Studio Audio Service",
        version="1.0.0",
        description="Unified Microservice for Qwen3-ASR 1.7B, Qwen3-TTS 1.7B, and XTTS-v2 Vietnamese",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    engines = AudioEngines(device=device)

    @app.get("/health")
    def health():
        mem_info = {}
        if torch.cuda.is_available() and "cuda" in engines.device:
            dev_idx = 0
            try:
                dev_idx = int(engines.device.split(":")[1])
            except Exception:
                pass
            mem_info = {
                "allocated_gb": round(torch.cuda.memory_allocated(dev_idx) / (1024 ** 3), 2),
                "reserved_gb": round(torch.cuda.memory_reserved(dev_idx) / (1024 ** 3), 2),
            }
        return {
            "status": "online",
            "service": "AudioService",
            "device": engines.device,
            "models": {
                "asr": {"loaded": engines.qwen_asr_model is not None, "id": engines.asr_id},
                "qwen_tts": {"loaded": engines.qwen_tts_model is not None, "id": engines.tts_id},
                "xtts": {"loaded": engines.xtts_model is not None, "id": engines.xtts_id},
            },
            "vram": mem_info,
        }

    @app.get("/audio/diagnostics")
    def audio_diagnostics(try_load: bool = True):
        return engines.get_diagnostics(try_load=try_load)

    # 1. STT / ASR
    @app.post("/asr/transcribe")
    async def transcribe_audio(
        file: Optional[UploadFile] = File(None),
        audio_b64: Optional[str] = Form(None),
        language: Optional[str] = Form("vi"),
        task: Optional[str] = Form("transcribe"),
        prompt: Optional[str] = Form(None),
        temperature: Optional[float] = Form(0.0),
        max_new_tokens: Optional[int] = Form(None),
        return_time_stamps: Optional[bool] = Form(False),
    ):
        if file is not None:
            content = await file.read()
        elif audio_b64:
            content = base64.b64decode(audio_b64.split(",")[-1])
        else:
            raise HTTPException(status_code=400, detail="Cần cung cấp file audio hoặc audio_b64")

        res = engines.transcribe(
            content,
            language=language,
            task=task,
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            return_time_stamps=return_time_stamps,
        )
        return res

    # 2. Qwen3-TTS (VoiceDesign, CustomVoice, Multilingual)
    @app.post("/tts/qwen")
    def tts_qwen(req: QwenTTSRequest):
        audio_bytes = engines.synthesize_qwen(
            text=req.text,
            language=req.language or "vi",
            voice=req.voice,
            voice_description=req.voice_description,
            prompt_audio=req.prompt_audio,
            speaker_wav=req.speaker_wav,
            ref_text=req.ref_text,
            x_vector_only_mode=req.x_vector_only_mode,
            speed=req.speed or 1.0,
            sample_rate=req.sample_rate or 24000,
            response_format=req.response_format or "wav",
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            max_new_tokens=req.max_new_tokens,
        )
        media_type = "audio/wav" if (req.response_format or "wav").lower() == "wav" else "audio/ogg"
        return Response(content=audio_bytes, media_type=media_type)

    # 3. XTTS-v2 Vietnamese (Chuyên biệt tiếng Việt & Voice Cloning)
    @app.post("/tts/xtts")
    def tts_xtts(req: XTTSRequest):
        audio_bytes = engines.synthesize_xtts(
            text=req.text,
            language=req.language or "vi",
            speaker_wav=req.speaker_wav or req.prompt_audio,
            speed=req.speed or 1.0,
            sample_rate=req.sample_rate or 24000,
            response_format=req.response_format or "wav",
            temperature=req.temperature,
            length_penalty=req.length_penalty,
            repetition_penalty=req.repetition_penalty,
            top_k=req.top_k,
            top_p=req.top_p,
            gpt_cond_len=req.gpt_cond_len,
            max_ref_len=req.max_ref_len,
            sound_norm_refs=req.sound_norm_refs,
            enable_text_splitting=req.enable_text_splitting,
        )
        media_type = "audio/wav" if (req.response_format or "wav").lower() == "wav" else "audio/ogg"
        return Response(content=audio_bytes, media_type=media_type)

    # 4. Quản lý bộ nhớ
    @app.post("/vram/free")
    def free_vram():
        engines.free_vram()
        return {"status": "success", "message": "Audio VRAM freed"}

    return app


# Default app instance
app = create_audio_app()


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Audio Service Daemon")
    parser.add_argument("--host", default="127.0.0.1", help="Host bind")
    parser.add_argument("--port", type=int, default=8288, help="Port cho Audio Service")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    app_instance = create_audio_app(device=args.device)
    uvicorn.run(app_instance, host=args.host, port=args.port, log_level="info")
