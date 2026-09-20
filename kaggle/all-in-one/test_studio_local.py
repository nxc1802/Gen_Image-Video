#!/usr/bin/env python3
"""
🧪 Test Studio Local: Comprehensive End-to-End Suite for Studio AI
Kiểm tra tự động toàn bộ tính năng và endpoint của Studio AI qua Cloudflare Quick Tunnel:
1. Health & Initial System Memory (VRAM / RAM)
2. VLM Chat Completions (SSE Streaming & TTFT)
3. Audio Text-to-Speech (Kokoro-82M WAV)
4. Audio Speech-to-Text (Whisper Turbo Transcription)
5. FLUX.2 Text-to-Image (SSE Diffusion Step Progress)
6. FLUX.2 Image Inpainting (Masked Editing with SSE Progress)
7. Wan2.1 Text-to-Video (SSE Video Step Progress)
8. Wan2.1 Image-to-Video (ITV with SSE Step Progress)
9. Internal Memory Transition & RAM-VRAM Swapping Verification
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from typing import Dict, Any, Optional
from PIL import Image, ImageDraw
import requests

# Fix unicode output for terminal
if sys.platform != "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def read_base_url_from_file(file_path: str = "public_url.txt") -> str:
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            line = f.read().strip()
            if line:
                return line
    return "http://127.0.0.1:8000/v1"


def parse_sse_events(response: requests.Response):
    """Phân tích cú pháp chuỗi Server-Sent Events (SSE) theo chuẩn RFC."""
    event_type = "message"
    data_lines = []

    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.strip()
        if line.startswith(":"):
            continue
        if not line:
            if data_lines:
                data_str = "\n".join(data_lines)
                if data_str == "[DONE]":
                    yield {"event": "done", "data": "[DONE]"}
                else:
                    try:
                        parsed_json = json.loads(data_str)
                        yield {"event": event_type, "data": parsed_json}
                    except json.JSONDecodeError:
                        yield {"event": event_type, "data": data_str}
                event_type = "message"
                data_lines = []
            continue

        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())

    if data_lines:
        data_str = "\n".join(data_lines)
        try:
            parsed_json = json.loads(data_str)
            yield {"event": event_type, "data": parsed_json}
        except Exception:
            yield {"event": event_type, "data": data_str}


class StudioTester:
    def __init__(self, base_url: str, output_dir: str = "test_outputs", token: str = "dummy"):
        self.base_url = base_url.rstrip("/")
        # URL gốc không có /v1 cho /health
        self.root_url = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        self.output_dir = output_dir
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []

    def record_result(self, name: str, success: bool, duration: float, note: str = ""):
        self.results.append({
            "Test": name,
            "Success": "✅ PASS" if success else "❌ FAIL",
            "Duration (s)": f"{duration:.2f}s",
            "Note": note,
        })

    # =========================================================================
    # CHECK 0: Health & Initial Memory
    # =========================================================================
    def check_health_and_memory(self):
        print("\n" + "=" * 78)
        print("🔍 [TEST 0] KIỂM TRA HEALTH CHECK & BỘ NHỚ HỆ THỐNG BAN ĐẦU")
        print("=" * 78)
        t0 = time.time()
        resp = None
        for attempt in range(12):
            try:
                resp = requests.get(f"{self.root_url}/health", timeout=10)
                if resp.status_code == 200:
                    dur = time.time() - t0
                    print(f"✅ Health check OK sau {dur:.2f}s (Lần thử {attempt+1}): {resp.json()}")
                    break
            except Exception as ex:
                if attempt == 0:
                    print(f"⏳ Đang kết nối tới server ({self.root_url})...")
                time.sleep(3.0)

        if resp is None or resp.status_code != 200:
            print(f"❌ Không thể kết nối tới {self.root_url}/health sau {time.time() - t0:.2f}s")

        # Check /v1/memory
        try:
            resp_mem = requests.get(f"{self.base_url}/memory", headers=self.headers, timeout=15)
            if resp_mem.status_code == 200:
                mem_data = resp_mem.json()
                print("📊 Trạng thái bộ nhớ VRAM & RAM ban đầu:")
                print(json.dumps(mem_data, indent=2, ensure_ascii=False))
                self.record_result("Health & Memory Check", True, time.time() - t0, "Server & GPU sẵn sàng")
                return mem_data
        except Exception as e:
            print(f"❌ Lỗi đọc /v1/memory: {e}")

        self.record_result("Health & Memory Check", False, time.time() - t0, "Lỗi kết nối")
        return None

    # =========================================================================
    # CHECK 1: VLM Chat SSE Streaming
    # =========================================================================
    def test_vlm_chat_stream(self):
        print("\n" + "=" * 78)
        print("💬 [TEST 1] VLM CHAT COMPLETION (SSE STREAMING & TIME-TO-FIRST-TOKEN)")
        print("=" * 78)
        prompt = "Xin chào! Bạn là trợ lý Studio AI. Hãy tóm tắt khả năng của bạn trong đúng 2 câu ngắn."
        payload = {
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "messages": [
                {"role": "system", "content": "Bạn là trợ lý AI thông minh, ngắn gọn, súc tích."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 128,
            "temperature": 0.7,
            "stream": True,
        }

        print(f"📤 Gửi prompt: '{prompt}'")
        print("📥 Streaming tokens: ", end="", flush=True)

        t0 = time.time()
        ttft = None
        collected_text = ""
        token_count = 0

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()

            for ev in parse_sse_events(resp):
                ev_type = ev.get("event")
                data = ev.get("data")
                if ev_type == "done" or data == "[DONE]":
                    break
                if isinstance(data, dict):
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if ttft is None:
                                ttft = time.time() - t0
                            print(content, end="", flush=True)
                            collected_text += content
                            token_count += 1

            total_time = time.time() - t0
            print(f"\n\n⏱️ Thời gian phản hồi: {total_time:.2f}s | TTFT: {ttft or 0:.2f}s | Tokens: {token_count}")
            print(f"⚡ Tốc độ stream: {token_count / max(total_time - (ttft or 0), 0.01):.1f} tokens/s")
            self.record_result(
                "VLM Chat Stream (Qwen)",
                True,
                total_time,
                f"TTFT: {ttft or 0:.2f}s, {token_count} tokens",
            )
            return collected_text
        except Exception as e:
            total_time = time.time() - t0
            print(f"\n❌ Lỗi Chat Stream: {e}")
            self.record_result("VLM Chat Stream (Qwen)", False, total_time, str(e))
            return None

    # =========================================================================
    # CHECK 2: Audio Text-to-Speech (Kokoro-82M)
    # =========================================================================
    def test_audio_tts(self) -> Optional[str]:
        print("\n" + "=" * 78)
        print("🔊 [TEST 2] TEXT-TO-SPEECH (KOKORO-82M FULL FP16)")
        print("=" * 78)
        input_text = "Studio AI đã được khởi chạy thành công trên nền tảng Kaggle."
        payload = {
            "model": "hexgrad/Kokoro-82M",
            "input": input_text,
            "voice": "af_heart",
            "response_format": "wav",
        }

        print(f"📤 Tổng hợp câu: '{input_text}'")
        t0 = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/audio/speech",
                headers=self.headers,
                json=payload,
                timeout=90,
            )
            resp.raise_for_status()
            audio_bytes = resp.content
            total_time = time.time() - t0

            wav_path = os.path.join(self.output_dir, "tts_output.wav")
            with open(wav_path, "wb") as f:
                f.write(audio_bytes)

            print(f"✅ Đã tạo audio WAV ({len(audio_bytes)} bytes) sau {total_time:.2f}s!")
            print(f"💾 File lưu tại: {wav_path}")
            self.record_result("TTS Kokoro-82M", True, total_time, f"Size: {len(audio_bytes)} bytes")
            return wav_path
        except Exception as e:
            total_time = time.time() - t0
            print(f"❌ Lỗi TTS: {e}")
            self.record_result("TTS Kokoro-82M", False, total_time, str(e))
            return None

    # =========================================================================
    # CHECK 3: Audio Speech-to-Text (Whisper Turbo)
    # =========================================================================
    def test_audio_stt(self, wav_path: Optional[str] = None):
        print("\n" + "=" * 78)
        print("🎙️ [TEST 3] SPEECH-TO-TEXT (WHISPER LARGE-V3-TURBO FP16)")
        print("=" * 78)

        # Nếu chưa có file wav, tạo file sine wave 2 giây giả lập
        if not wav_path or not os.path.exists(wav_path):
            import wave, math, struct
            wav_path = os.path.join(self.output_dir, "synthetic_test.wav")
            with wave.open(wav_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                for i in range(16000 * 2):
                    val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * 440.0 * i / 16000))
                    wf.writeframes(struct.pack("<h", val))

        print(f"📤 Gửi tệp âm thanh để nhận diện: {wav_path}")
        t0 = time.time()
        try:
            with open(wav_path, "rb") as f:
                files = {"file": (os.path.basename(wav_path), f, "audio/wav")}
                data = {"model": "openai/whisper-large-v3-turbo"}
                headers = {"Authorization": self.headers["Authorization"]}
                resp = requests.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=90,
                )
            resp.raise_for_status()
            res_json = resp.json()
            total_time = time.time() - t0
            transcribed_text = res_json.get("text", "")
            print(f"✅ Whisper nhận diện sau {total_time:.2f}s:")
            print(f"   📝 Văn bản: '{transcribed_text}'")
            print(f"   🌐 Ngôn ngữ phát hiện: {res_json.get('language')}")
            self.record_result("STT Whisper Turbo", True, total_time, f"Text: '{transcribed_text[:40]}'")
            return transcribed_text
        except Exception as e:
            total_time = time.time() - t0
            print(f"❌ Lỗi STT: {e}")
            self.record_result("STT Whisper Turbo", False, total_time, str(e))
            return None

    # =========================================================================
    # CHECK 4: FLUX.2 Image Generation with SSE Diffusion Progress
    # =========================================================================
    def test_flux_image_generation(self) -> Optional[str]:
        print("\n" + "=" * 78)
        print("🖼️ [TEST 4] FLUX.2 IMAGE GENERATION (SSE DIFFUSION STEP PROGRESS)")
        print("=" * 78)
        prompt = "A cinematic studio portrait of a futuristic robotic cat with glowing turquoise neon lines, 8k resolution"
        payload = {
            "model": "flux-2-klein-4b",
            "prompt": prompt,
            "size": "512x512",
            "steps": 4,
            "stream": True,
        }

        print(f"📤 Gửi yêu cầu tạo ảnh: '{prompt}' (steps=4, size=512x512)")
        t0 = time.time()
        final_b64 = None

        try:
            resp = requests.post(
                f"{self.base_url}/images/generations",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=180,
            )
            resp.raise_for_status()

            for ev in parse_sse_events(resp):
                ev_type = ev.get("event")
                data = ev.get("data")
                if ev_type == "progress" and isinstance(data, dict):
                    step = data.get("step")
                    total = data.get("total_steps")
                    pct = data.get("progress")
                    if data.get("status") == "preparing" or step == 0:
                        print(f"   ⏳ [FLUX] Đang khởi tạo mô hình và kết nối VRAM...")
                    else:
                        print(f"   ⏳ [FLUX Progress] Bước {step}/{total} ({pct}%)")
                elif ev_type == "complete" and isinstance(data, dict):
                    items = data.get("data", [])
                    if items and "b64_json" in items[0]:
                        final_b64 = items[0]["b64_json"]
                        inference_time = data.get("x_inference_time_seconds", 0)
                        print(f"   ✨ Quá trình khử nhiễu hoàn tất (Inference time: {inference_time:.2f}s)")
                elif ev_type == "done" or data == "[DONE]":
                    break

            total_time = time.time() - t0
            if final_b64:
                img_data = base64.b64decode(final_b64)
                img_path = os.path.join(self.output_dir, "flux_generated.png")
                with open(img_path, "wb") as f:
                    f.write(img_data)
                print(f"✅ Tạo ảnh thành công sau {total_time:.2f}s! Đã lưu: {img_path}")
                self.record_result("Image Gen FLUX.2 (flux-2-klein-4b)", True, total_time, f"Saved: {img_path}")
                return img_path
            else:
                print(f"⚠️ Không nhận được ảnh từ SSE stream sau {total_time:.2f}s.")
                self.record_result("Image Gen FLUX.2 (flux-2-klein-4b)", False, total_time, "No image received")
                return None
        except Exception as e:
            total_time = time.time() - t0
            print(f"❌ Lỗi tạo ảnh FLUX: {e}")
            self.record_result("Image Gen FLUX.2 (flux-2-klein-4b)", False, total_time, str(e))
            return None

    # =========================================================================
    # CHECK 5: FLUX.2 Masked Inpainting with SSE Progress
    # =========================================================================
    def test_flux_inpainting(self, base_img_path: Optional[str] = None):
        print("\n" + "=" * 78)
        print("🎨 [TEST 5] FLUX.2 MASKED INPAINTING (IMAGE EDIT WITH SSE PROGRESS)")
        print("=" * 78)

        # Tạo ảnh gốc và mặt nạ nếu chưa có
        if not base_img_path or not os.path.exists(base_img_path):
            img = Image.new("RGB", (512, 512), color=(70, 70, 120))
            draw = ImageDraw.Draw(img)
            draw.rectangle([100, 100, 412, 412], fill=(200, 100, 100))
            base_img_path = os.path.join(self.output_dir, "inpaint_base.png")
            img.save(base_img_path)

        # Tạo mask: hình chữ nhật trắng ở giữa
        mask = Image.new("L", (512, 512), color=0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([180, 180, 332, 332], fill=255)
        mask_path = os.path.join(self.output_dir, "inpaint_mask.png")
        mask.save(mask_path)

        # Đọc base64
        with open(base_img_path, "rb") as f:
            base_b64 = base64.b64encode(f.read()).decode("utf-8")
        with open(mask_path, "rb") as f:
            mask_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = "A glowing golden cyber medal in the center"
        payload = {
            "model": "flux-2-klein-4b",
            "prompt": prompt,
            "image": f"data:image/png;base64,{base_b64}",
            "mask_image": f"data:image/png;base64,{mask_b64}",
            "size": "512x512",
            "steps": 4,
            "stream": True,
        }

        print(f"📤 Gửi yêu cầu Inpainting: '{prompt}'")
        t0 = time.time()
        final_b64 = None

        try:
            resp = requests.post(
                f"{self.base_url}/images/edits",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=180,
            )
            resp.raise_for_status()

            for ev in parse_sse_events(resp):
                ev_type = ev.get("event")
                data = ev.get("data")
                if ev_type == "progress" and isinstance(data, dict):
                    step = data.get("step")
                    total = data.get("total_steps")
                    pct = data.get("progress")
                    if data.get("status") == "preparing" or step == 0:
                        print(f"   ⏳ [Inpaint] Đang khởi tạo Inpainting pipeline và VRAM...")
                    else:
                        print(f"   ⏳ [Inpaint Progress] Bước {step}/{total} ({pct}%)")
                elif ev_type == "complete" and isinstance(data, dict):
                    items = data.get("data", [])
                    if items and "b64_json" in items[0]:
                        final_b64 = items[0]["b64_json"]
                        print(f"   ✨ Inpainting hoàn tất (Inference time: {data.get('x_inference_time_seconds', 0):.2f}s)")
                elif ev_type == "done" or data == "[DONE]":
                    break

            total_time = time.time() - t0
            if final_b64:
                img_data = base64.b64decode(final_b64)
                out_path = os.path.join(self.output_dir, "flux_inpaint_result.png")
                with open(out_path, "wb") as f:
                    f.write(img_data)
                print(f"✅ Inpainting thành công sau {total_time:.2f}s! Đã lưu: {out_path}")
                self.record_result("Inpainting FLUX.2", True, total_time, f"Saved: {out_path}")
                return out_path
            else:
                print(f"⚠️ Không nhận được ảnh inpainting sau {total_time:.2f}s.")
                self.record_result("Inpainting FLUX.2", False, total_time, "No image received")
                return None
        except Exception as e:
            total_time = time.time() - t0
            print(f"❌ Lỗi Inpainting: {e}")
            self.record_result("Inpainting FLUX.2", False, total_time, str(e))
            return None

    # =========================================================================
    # CHECK 6: Wan2.1 Text-to-Video with SSE Progress
    # =========================================================================
    def test_wan_t2v(self):
        print("\n" + "=" * 78)
        print("🎬 [TEST 6] WAN2.1 TEXT-TO-VIDEO (SSE VIDEO STEP PROGRESS)")
        print("=" * 78)
        prompt = "A cute small robot waving its hand, smooth cinematic motion"
        payload = {
            "model": "1.3b",
            "prompt": prompt,
            "num_frames": 17,
            "width": 832,
            "height": 480,
            "steps": 15,
            "guidance": 5.0,
            "stream": True,
        }

        print(f"📤 Gửi yêu cầu sinh Video T2V: '{prompt}' (model=1.3b, frames=17, 832x480, 15 steps)")
        t0 = time.time()
        final_b64 = None

        try:
            resp = requests.post(
                f"{self.base_url}/videos/generations",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=600,
            )
            resp.raise_for_status()

            for ev in parse_sse_events(resp):
                ev_type = ev.get("event")
                data = ev.get("data")
                if ev_type == "progress" and isinstance(data, dict):
                    step = data.get("step")
                    total = data.get("total_steps")
                    pct = data.get("progress")
                    if data.get("status") == "preparing" or step == 0:
                        print(f"   ⏳ [Video] Đang khởi tạo Wan2.1 Video pipeline và bộ đệm...")
                    else:
                        print(f"   ⏳ [Video Progress] Khử nhiễu bước {step}/{total} ({pct}%)")
                elif ev_type == "complete" and isinstance(data, dict):
                    items = data.get("data", [])
                    if items and "b64_json" in items[0]:
                        final_b64 = items[0]["b64_json"]
                        print(f"   🎉 Sinh video hoàn tất! (Inference time: {data.get('x_inference_time_seconds', 0):.2f}s)")
                elif ev_type == "error":
                    err_msg = data.get("error", "Unknown error") if isinstance(data, dict) else str(data)
                    print(f"   ❌ Server báo lỗi Video SSE: {err_msg}")
                elif ev_type == "done" or data == "[DONE]":
                    break

            total_time = time.time() - t0
            if final_b64:
                vid_data = base64.b64decode(final_b64)
                vid_path = os.path.join(self.output_dir, "wan_t2v.mp4")
                with open(vid_path, "wb") as f:
                    f.write(vid_data)
                print(f"✅ Tạo video T2V thành công sau {total_time:.2f}s! ({len(vid_data)} bytes)")
                print(f"💾 File lưu tại: {vid_path}")
                self.record_result("Video T2V Wan2.1 (1.3B)", True, total_time, f"Size: {len(vid_data)} bytes")
                return vid_path
            else:
                print(f"⚠️ Không nhận được video T2V sau {total_time:.2f}s.")
                self.record_result("Video T2V Wan2.1 (1.3B)", False, total_time, "No video received")
                return None
        except Exception as e:
            total_time = time.time() - t0
            print(f"❌ Lỗi sinh video T2V: {e}")
            self.record_result("Video T2V Wan2.1 (1.3B)", False, total_time, str(e))
            return None

    # =========================================================================
    # CHECK 7: Wan2.1 Image-to-Video (ITV) with SSE Progress
    # =========================================================================
    def test_wan_itv(self, ref_image_path: Optional[str] = None):
        print("\n" + "=" * 78)
        print("🎞️ [TEST 7] WAN2.1 IMAGE-TO-VIDEO (ITV WITH SSE PROGRESS)")
        print("=" * 78)

        # Chuẩn bị ảnh tham chiếu
        if not ref_image_path or not os.path.exists(ref_image_path):
            img = Image.new("RGB", (512, 512), color=(40, 90, 140))
            draw = ImageDraw.Draw(img)
            draw.ellipse([150, 150, 362, 362], fill=(240, 200, 80))
            ref_image_path = os.path.join(self.output_dir, "itv_ref.png")
            img.save(ref_image_path)

        with open(ref_image_path, "rb") as f:
            ref_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = "The glowing golden orb slowly rotates and radiates brilliant light beams, cinematic 4k"
        payload = {
            "model": "1.3b",
            "prompt": prompt,
            "image": f"data:image/png;base64,{ref_b64}",
            "num_frames": 17,
            "width": 512,
            "height": 512,
            "stream": True,
        }

        print(f"📤 Gửi yêu cầu ITV: '{prompt}'")
        t0 = time.time()
        final_b64 = None

        try:
            resp = requests.post(
                f"{self.base_url}/videos/generations",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=300,
            )
            resp.raise_for_status()

            for ev in parse_sse_events(resp):
                ev_type = ev.get("event")
                data = ev.get("data")
                if ev_type == "progress" and isinstance(data, dict):
                    step = data.get("step")
                    total = data.get("total_steps")
                    pct = data.get("progress")
                    if data.get("status") == "preparing" or step == 0:
                        print(f"   ⏳ [ITV] Đang khởi tạo Wan2.1 ITV pipeline và bộ đệm...")
                    else:
                        print(f"   ⏳ [ITV Progress] Bước {step}/{total} ({pct}%)")
                elif ev_type == "complete" and isinstance(data, dict):
                    items = data.get("data", [])
                    if items and "b64_json" in items[0]:
                        final_b64 = items[0]["b64_json"]
                        print(f"   🎉 Sinh ITV video hoàn tất! (Inference time: {data.get('x_inference_time_seconds', 0):.2f}s)")
                elif ev_type == "error":
                    err_msg = data.get("error", "Unknown error") if isinstance(data, dict) else str(data)
                    print(f"   ❌ Server báo lỗi ITV SSE: {err_msg}")
                elif ev_type == "done" or data == "[DONE]":
                    break

            total_time = time.time() - t0
            if final_b64:
                vid_data = base64.b64decode(final_b64)
                vid_path = os.path.join(self.output_dir, "wan_itv.mp4")
                with open(vid_path, "wb") as f:
                    f.write(vid_data)
                print(f"✅ Tạo video ITV thành công sau {total_time:.2f}s! ({len(vid_data)} bytes)")
                print(f"💾 File lưu tại: {vid_path}")
                self.record_result("Video ITV Wan2.1", True, total_time, f"Size: {len(vid_data)} bytes")
                return vid_path
            else:
                print(f"⚠️ Không nhận được video ITV sau {total_time:.2f}s.")
                self.record_result("Video ITV Wan2.1", False, total_time, "No video received")
                return None
        except Exception as e:
            total_time = time.time() - t0
            print(f"❌ Lỗi sinh video ITV: {e}")
            self.record_result("Video ITV Wan2.1", False, total_time, str(e))
            return None

    # =========================================================================
    # CHECK 8: Memory & RAM/VRAM Swap Inspection
    # =========================================================================
    def check_memory_swap(self):
        print("\n" + "=" * 78)
        print("🔄 [TEST 8] KIỂM TRA ĐIỀU PHỐI HOÁN ĐỔI NỘI BỘ RAM - VRAM (PCIe FAST-SWAP)")
        print("=" * 78)
        t0 = time.time()
        try:
            resp = requests.get(f"{self.base_url}/memory", headers=self.headers, timeout=15)
            resp.raise_for_status()
            mem = resp.json()
            print("📊 Báo cáo phân bổ VRAM & RAM thực tế:")
            print(json.dumps(mem, indent=2, ensure_ascii=False))

            active_dynamic = mem.get("active_dynamic_slot")
            pinned_slots = mem.get("always_active_slots", [])
            cached_slots = mem.get("cached_dynamic_slots", [])
            ram = mem.get("ram", {})

            print("\n📈 [ĐÁNH GIÁ CƠ CHẾ FAST-SWAP]:")
            print(f"   • Active Dynamic Slot hiện tại : {active_dynamic}")
            print(f"   • Always Active (Ghim VRAM)   : {pinned_slots}")
            print(f"   • Dynamic Slot đỗ trong RAM   : {cached_slots}")
            print(f"   • RAM sử dụng / khả dụng      : {ram.get('used_gb', 'N/A')}GB / {ram.get('total_gb', 'N/A')}GB ({ram.get('percent', 'N/A')}%)")

            is_valid_swap = (active_dynamic in ["flux", "image", "video", "wan_video"]) and (len(cached_slots) >= 1)
            self.record_result(
                "RAM-VRAM PCIe Fast-Swap",
                is_valid_swap,
                time.time() - t0,
                f"Active: {active_dynamic}, Cached: {cached_slots}",
            )
            return mem
        except Exception as e:
            print(f"❌ Lỗi kiểm tra /v1/memory: {e}")
            self.record_result("RAM-VRAM PCIe Fast-Swap", False, time.time() - t0, str(e))
            return None

    # =========================================================================
    # Summary Report
    # =========================================================================
    def print_summary(self):
        print("\n" + "=" * 78)
        print("📋 TỔNG HỢP KẾT QUẢ KIỂM THỬ TOÀN DIỆN (STUDIO AI BENCHMARK)")
        print("=" * 78)
        print(f"{'Endpoint / Feature':<30} | {'Status':<8} | {'Latency':<12} | {'Note'}")
        print("-" * 78)
        for r in self.results:
            print(f"{r['Test']:<30} | {r['Success']:<8} | {r['Duration (s)']:<12} | {r['Note']}")
        print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="🧪 Studio AI Local Comprehensive Tester")
    parser.add_argument("--base-url", default=None, help="Base API URL (ví dụ: https://xxx.trycloudflare.com/v1)")
    parser.add_argument("--output-dir", default="test_outputs", help="Thư mục lưu outputs")
    parser.add_argument("--skip-video", action="store_true", help="Bỏ qua test video nếu muốn chạy nhanh")
    args = parser.parse_args()

    base_url = args.base_url
    if not base_url:
        base_url = read_base_url_from_file("public_url.txt")

    print(f"🎯 Target API Base URL: {base_url}")
    tester = StudioTester(base_url=base_url, output_dir=args.output_dir)

    # 1. Health & Initial Memory
    tester.check_health_and_memory()

    # 2. VLM Chat Stream
    tester.test_vlm_chat_stream()

    # 3. Kokoro TTS
    wav_path = tester.test_audio_tts()

    # 4. Whisper STT
    tester.test_audio_stt(wav_path)

    # 5. FLUX.2 Image Generation (SSE)
    img_path = tester.test_flux_image_generation()

    # 6. FLUX.2 Inpainting (SSE)
    tester.test_flux_inpainting(img_path)

    # 7. Wan2.1 T2V (SSE)
    if not args.skip_video:
        tester.test_wan_t2v()

    # 8. Wan2.1 ITV (SSE)
    if not args.skip_video:
        tester.test_wan_itv(img_path)

    # 9. RAM-VRAM Swapping Check
    tester.check_memory_swap()

    # In bảng tổng kết
    tester.print_summary()


if __name__ == "__main__":
    main()
