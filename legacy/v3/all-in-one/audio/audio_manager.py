"""
🎮 Audio Service Process & Lifecycle Manager
================================================================================
Quản lý tiến trình Audio Service (FastAPI Microservice trên GPU 0: cuda:0):
1. Tương tự ComfyUIManager ở Image/Video:
   - Chạy nền daemon trên 127.0.0.1:8288.
   - Quản lý 3 mô hình âm thanh: Qwen3-ASR 1.7B, Qwen3-TTS 1.7B, XTTS-v2 Vietnamese.
2. Cung cấp REST Client đồng bộ và bất đồng bộ để Gateway và Factory gọi.
3. Tự động chuyển tiếp in-process fallback nếu không khởi động được process daemon.
"""

import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger("AudioServiceManager")


class AudioServiceManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AudioServiceManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8288,
        cuda_device: int = 0,
    ):
        if self._initialized:
            return
        self._initialized = True

        self.host = host
        self.port = int(os.environ.get("AUDIO_SERVICE_PORT", port))
        self.cuda_device = cuda_device
        self.base_url = f"http://{self.host}:{self.port}"
        self.process: Optional[subprocess.Popen] = None
        self._in_process_engines = None
        self._preload_lock = threading.Lock()
        self._preloaded = False
        # Mặc định chạy trực tiếp in-process để tránh tạo 2 tiến trình chiếm gấp đôi VRAM trên GPU 0
        self.use_in_process = os.environ.get("AUDIO_SERVICE_SPAWN", "0") != "1"

    def is_running(self) -> bool:
        """Kiểm tra server Audio Service đã sẵn sàng hay chưa."""
        if self.use_in_process:
            return self._in_process_engines is not None
        try:
            req = urllib.request.Request(f"{self.base_url}/health", headers={"User-Agent": "StudioGateway"})
            with urllib.request.urlopen(req, timeout=1.0) as res:
                return res.status == 200
        except Exception:
            return False

    def preload_all_models(self):
        """Nạp thẳng cả 3 model Speech (Qwen3-ASR 1.7B, Qwen3-TTS 1.7B, XTTS-v2) vào VRAM."""
        with self._preload_lock:
            eng = self._get_in_process_engines()
            if self._preloaded:
                return eng
            logger.info("🔥 [Speech Auto-Load] Bắt đầu nạp thẳng 3 model Speech vào GPU VRAM...")
            if eng.qwen_asr_model is None:
                eng.load_qwen_asr()
            if eng.qwen_tts_model is None:
                eng.load_qwen_tts()
            if eng.xtts_model is None:
                eng.load_xtts()
            self._preloaded = True
            logger.info("✅ [Speech Auto-Load] Cả 3 model Speech (ASR, Qwen-TTS, XTTS-v2) đã sẵn sàng trong VRAM!")
            return eng

    def start_service(self) -> bool:
        """Khởi động Audio Service và nạp sẵn 3 model Speech vào GPU 0."""
        if self.use_in_process:
            self.preload_all_models()
            return True

        if self.is_running():
            logger.info(f"🎧 Audio Service đã chạy sẵn sàng tại {self.base_url}")
            return True

        script_path = Path(__file__).parent / "audio_service.py"
        if not script_path.exists():
            logger.warning(f"Không tìm thấy {script_path}, sẽ chạy in-process.")
            self.preload_all_models()
            return True

        env = os.environ.copy()
        import torch
        if torch.cuda.is_available():
            env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_device)

        cmd = [
            sys.executable,
            str(script_path),
            "--host", self.host,
            "--port", str(self.port),
            "--device", f"cuda:{self.cuda_device}" if torch.cuda.is_available() else "cpu",
        ]

        logger.info(f"🚀 Khởi động Audio Service daemon: {' '.join(cmd)}")
        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return self.wait_ready(timeout=30.0)
        except Exception as e:
            logger.warning(f"⚠️ Lỗi khởi chạy Audio Service daemon: {e}. Chuyển sang in-process fallback.")
            self.preload_all_models()
            return True

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """Chờ service khởi động hoàn tất."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.is_running():
                logger.info(f"✅ Audio Service đã sẵn sàng sau {time.time() - t0:.1f}s!")
                return True
            time.sleep(0.5)
        logger.warning("⚠️ Timeout khi chờ Audio Service khởi động.")
        return False

    def stop_service(self):
        """Dừng daemon."""
        if self.process and self.process.poll() is None:
            logger.info("🛑 Đang dừng Audio Service daemon...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

    def _get_in_process_engines(self):
        """Khởi tạo in-process engine dự phòng khi process daemon chưa chạy."""
        if self._in_process_engines is None:
            from .audio_service import AudioEngines
            self._in_process_engines = AudioEngines(
                device=f"cuda:{self.cuda_device}" if sys.platform != "darwin" and __import__("torch").cuda.is_available() else "cpu"
            )
        return self._in_process_engines

    # ==========================================================================
    # API METHODS
    # ==========================================================================
    def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = "vi",
        task: Optional[str] = "transcribe",
        prompt: Optional[str] = None,
        model: Optional[str] = "qwen3-asr-1.7b",
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        return_time_stamps: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Gửi yêu cầu nhận diện giọng nói tới Qwen3-ASR."""
        # Thử gọi qua HTTP REST API nếu chế độ subprocess được bật
        if not self.use_in_process and self.is_running():
            try:
                boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
                body = bytearray()

                # field: file
                body.extend(f"--{boundary}\r\n".encode("utf-8"))
                body.extend(b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n')
                body.extend(b"Content-Type: audio/wav\r\n\r\n")
                body.extend(audio_bytes)
                body.extend(b"\r\n")

                # field: language
                if language:
                    body.extend(f"--{boundary}\r\n".encode("utf-8"))
                    body.extend(b'Content-Disposition: form-data; name="language"\r\n\r\n')
                    body.extend(f"{language}\r\n".encode("utf-8"))

                # field: task
                body.extend(f"--{boundary}\r\n".encode("utf-8"))
                body.extend(b'Content-Disposition: form-data; name="task"\r\n\r\n')
                body.extend(f"{task or 'transcribe'}\r\n".encode("utf-8"))

                if prompt:
                    body.extend(f"--{boundary}\r\n".encode("utf-8"))
                    body.extend(b'Content-Disposition: form-data; name="prompt"\r\n\r\n')
                    body.extend(f"{prompt}\r\n".encode("utf-8"))

                if temperature is not None:
                    body.extend(f"--{boundary}\r\n".encode("utf-8"))
                    body.extend(b'Content-Disposition: form-data; name="temperature"\r\n\r\n')
                    body.extend(f"{temperature}\r\n".encode("utf-8"))

                if max_new_tokens is not None:
                    body.extend(f"--{boundary}\r\n".encode("utf-8"))
                    body.extend(b'Content-Disposition: form-data; name="max_new_tokens"\r\n\r\n')
                    body.extend(f"{max_new_tokens}\r\n".encode("utf-8"))

                if return_time_stamps is not None:
                    body.extend(f"--{boundary}\r\n".encode("utf-8"))
                    body.extend(b'Content-Disposition: form-data; name="return_time_stamps"\r\n\r\n')
                    body.extend(f"{str(return_time_stamps).lower()}\r\n".encode("utf-8"))

                body.extend(f"--{boundary}--\r\n".encode("utf-8"))

                req = urllib.request.Request(
                    f"{self.base_url}/asr/transcribe",
                    data=bytes(body),
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                )
                with urllib.request.urlopen(req, timeout=180.0) as res:
                    return json.loads(res.read().decode("utf-8"))
            except Exception as e:
                logger.warning(f"⚠️ Lỗi gọi REST Audio Service ({e}), chuyển sang in-process fallback.")

        # In-process trực tiếp
        eng = self._get_in_process_engines()
        return eng.transcribe(
            audio_bytes,
            language=language,
            task=task,
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            return_time_stamps=return_time_stamps,
        )

    def synthesize_qwen(
        self,
        text: str,
        language: Optional[str] = "vi",
        voice: Optional[str] = "default",
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
        """Gửi yêu cầu tổng hợp giọng nói tới Qwen3-TTS (VoiceDesign / CustomVoice)."""
        prompt_audio = prompt_audio or speaker_wav
        if not self.use_in_process and self.is_running():
            try:
                payload = json.dumps({
                    "text": text,
                    "language": language or "vi",
                    "voice": voice or "default",
                    "voice_description": voice_description,
                    "prompt_audio": prompt_audio,
                    "speaker_wav": speaker_wav,
                    "ref_text": ref_text,
                    "x_vector_only_mode": x_vector_only_mode,
                    "speed": speed,
                    "sample_rate": sample_rate,
                    "response_format": response_format,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "max_new_tokens": max_new_tokens,
                }).encode("utf-8")

                req = urllib.request.Request(
                    f"{self.base_url}/tts/qwen",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=180.0) as res:
                    return res.read()
            except Exception as e:
                logger.warning(f"⚠️ Lỗi gọi REST Audio Service ({e}), chuyển sang in-process.")

        eng = self._get_in_process_engines()
        return eng.synthesize_qwen(
            text=text,
            language=language,
            voice=voice,
            voice_description=voice_description,
            prompt_audio=prompt_audio,
            speaker_wav=speaker_wav,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
            speed=speed,
            sample_rate=sample_rate,
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )

    def synthesize_xtts(
        self,
        text: str,
        language: Optional[str] = "vi",
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
        """Gửi yêu cầu tổng hợp giọng nói tới XTTS-v2 Vietnamese (Voice Cloning)."""
        if not self.use_in_process and self.is_running():
            try:
                payload = json.dumps({
                    "text": text,
                    "language": language or "vi",
                    "speaker_wav": speaker_wav,
                    "speed": speed,
                    "sample_rate": sample_rate,
                    "response_format": response_format,
                    "temperature": temperature,
                    "length_penalty": length_penalty,
                    "repetition_penalty": repetition_penalty,
                    "top_k": top_k,
                    "top_p": top_p,
                    "gpt_cond_len": gpt_cond_len,
                    "max_ref_len": max_ref_len,
                    "sound_norm_refs": sound_norm_refs,
                    "enable_text_splitting": enable_text_splitting,
                }).encode("utf-8")

                req = urllib.request.Request(
                    f"{self.base_url}/tts/xtts",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=180.0) as res:
                    return res.read()
            except Exception as e:
                logger.warning(f"⚠️ Lỗi gọi REST Audio Service ({e}), chuyển sang in-process.")

        eng = self._get_in_process_engines()
        return eng.synthesize_xtts(
            text=text,
            language=language or "vi",
            speaker_wav=speaker_wav,
            speed=speed,
            sample_rate=sample_rate,
            response_format=response_format,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            gpt_cond_len=gpt_cond_len,
            max_ref_len=max_ref_len,
            sound_norm_refs=sound_norm_refs,
            enable_text_splitting=enable_text_splitting,
        )

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        voice_description: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        prompt_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        x_vector_only_mode: Optional[bool] = None,
        engine: Optional[str] = None,
        language: Optional[str] = "vi",
        speed: float = 1.0,
        sample_rate: int = 24000,
        response_format: str = "wav",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        length_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        gpt_cond_len: Optional[int] = None,
        max_ref_len: Optional[int] = None,
        sound_norm_refs: Optional[bool] = None,
        enable_text_splitting: Optional[bool] = None,
    ) -> bytes:
        """
        Bộ định tuyến TTS thông minh:
        - Nếu chỉ định engine='xtts' hoặc voice có 'xtts': Dùng XTTS-v2 Vietnamese.
        - Nếu chỉ định engine='qwen' hoặc voice có 'qwen': Dùng Qwen3-TTS 1.7B.
        - Mặc định: Dùng XTTS nếu là tiếng Việt có kèm speaker_wav mẫu, ngược lại dùng Qwen3-TTS.
        """
        target = (engine or "").lower()
        if not target and voice:
            if "xtts" in voice.lower():
                target = "xtts"
            elif "qwen" in voice.lower():
                target = "qwen"
            elif "kokoro" in voice.lower():
                target = "kokoro"

        is_xtts = (
            "xtts" in target
            or (not target and bool(speaker_wav) and (not language or language.lower() in ("vi", "vietnamese")))
        )
        if not is_xtts and bool(speaker_wav) and not prompt_audio:
            prompt_audio = speaker_wav

        if is_xtts:
            return self.synthesize_xtts(
                text=text,
                language=language,
                speaker_wav=speaker_wav or prompt_audio,
                speed=speed,
                sample_rate=sample_rate,
                response_format=response_format,
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
                gpt_cond_len=gpt_cond_len,
                max_ref_len=max_ref_len,
                sound_norm_refs=sound_norm_refs,
                enable_text_splitting=enable_text_splitting,
            )
        else:
            return self.synthesize_qwen(
                text=text,
                language=language,
                voice=voice,
                voice_description=voice_description,
                prompt_audio=prompt_audio,
                speaker_wav=speaker_wav,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
                speed=speed,
                sample_rate=sample_rate,
                response_format=response_format,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )

    def get_diagnostics(self, try_load: bool = True) -> Dict[str, Any]:
        """Lấy thông tin chẩn đoán từ daemon hoặc in-process."""
        if not self.use_in_process and self.is_running():
            try:
                url = f"{self.base_url}/audio/diagnostics?try_load={str(try_load).lower()}"
                req = urllib.request.Request(url, headers={"User-Agent": "StudioGateway"})
                with urllib.request.urlopen(req, timeout=60.0) as res:
                    return json.loads(res.read().decode("utf-8"))
            except Exception as e:
                logger.warning(f"Lỗi lấy diagnostics từ REST service: {e}")
        eng = self._get_in_process_engines()
        return eng.get_diagnostics(try_load=try_load)

    def free_vram(self):
        """Yêu cầu Audio Service dọn sạch VRAM trên GPU 0."""
        if not self.use_in_process and self.is_running():
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/vram/free",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=5.0) as res:
                    pass
            except Exception:
                pass
        if self._in_process_engines:
            self._in_process_engines.free_vram()


def get_audio_service_manager() -> AudioServiceManager:
    return AudioServiceManager()
