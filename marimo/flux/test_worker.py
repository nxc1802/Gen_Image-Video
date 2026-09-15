#!/usr/bin/env python3
import base64
import json
import time
import urllib.request
import urllib.error
import uuid

SUPABASE_URL = "https://fxepzlszglckfsscport.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ4ZXB6bHN6Z2xja2Zzc2Nwb3J0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODk0NDEzMzksImV4cCI6MjEwNTAxNzMzOX0.28rS1waBYB8xvGgHR7utoek9PqBc3ev6HPOG9yo9RdQ"

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}

job_id = str(uuid.uuid4())
prompt = "A majestic mechanical tiger with glowing neon circuitry, cyberpunk Tokyo rooftop, rainy night reflections, photorealistic 8k, hyper-detailed render"

payload = {
    "id": job_id,
    "prompt": prompt,
    "model": "flux-1-schnell",
    "size": "1024x1024",
    "steps": 4,
    "guidance": 0.0,
    "status": "pending",
}

print("=" * 70)
print(f"🚀 [1/3] Gửi Job #{job_id} lên Supabase Queue qua HTTPS...")
print(f"📝 Prompt: {prompt}")
print("=" * 70)

req = urllib.request.Request(
    f"{SUPABASE_URL}/rest/v1/image_jobs",
    data=json.dumps(payload).encode("utf-8"),
    headers=headers,
    method="POST",
)

with urllib.request.urlopen(req) as resp:
    res_data = json.loads(resp.read().decode("utf-8"))
    print("✅ Job đã được tạo thành công trên Supabase!")

print("⏳ [2/3] Đang đợi GPU Blackwell (NVIDIA RTX PRO 6000) trên Molab nhận việc...")
t0 = time.time()
completed = False

for i in range(120):
    time.sleep(1.0)
    query_url = f"{SUPABASE_URL}/rest/v1/image_jobs?id=eq.{job_id}&select=*"
    q_req = urllib.request.Request(query_url, headers=headers)
    with urllib.request.urlopen(q_req) as q_resp:
        rows = json.loads(q_resp.read().decode("utf-8"))
        if rows:
            job = rows[0]
            status = job.get("status")
            if status == "processing":
                print(f"⏳ [{int(time.time() - t0)}s] GPU đang sinh ảnh...")
            elif status == "completed":
                dt = job.get("inference_time_sec") or (time.time() - t0)
                dev = job.get("device_name", "GPU")
                b64 = job.get("result_b64")
                print("=" * 70)
                print(f"🎉 [3/3] HOÀN TẤT XUẤT SẮC!")
                print(f"⚡ Thời gian sinh ảnh: {dt}s")
                print(f"🖥️ Thiết bị thực thi: {dev}")
                if b64:
                    out_file = "/Volumes/WorkSpace/Project/Gen_Image:Video/marimo/flux/test_blackwell_result.png"
                    with open(out_file, "wb") as f:
                        f.write(base64.b64decode(b64))
                    print(f"💾 File ảnh đã được tải về máy: {out_file}")
                print("=" * 70)
                completed = True
                break
            elif status == "failed":
                err = job.get("error_message")
                print(f"❌ GPU báo lỗi: {err}")
                break
    if i % 5 == 0 and not completed and i > 0:
        print(f"   ...đang chờ GPU worker xử lý ({i}s)...")

if not completed:
    print("⚠️ Timeout: Worker chưa nhận job. Kiểm tra lại xem công tắc Worker trên Marimo đã BẬT chưa.")
