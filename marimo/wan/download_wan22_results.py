#!/usr/bin/env python3
"""
📥 Script tải video, ảnh khung hình và metrics JSON của Wan2.2 từ remote Marimo server
"""
import os
import subprocess
import base64
import json
import shutil

URL = "https://sb-fb3c85dabdd3ea8a.sb.molab.run/"
TOKEN = "7e04d65ad03c2c678d8191a03cb22c48dcad9428d0b59fe8ecbb33902df5d33c"
SCRIPT = "/Users/nxc/.agents/skills/marimo-pair/scripts/execute-code.sh"
OUT_DIR = "/Volumes/WorkSpace/Project/Gen_Image:Video/marimo/wan"
ARTIFACT_DIR = "/Users/nxc/.gemini/antigravity-ide/brain/5993ad67-f83f-439c-8e01-7ce59bae024e"

REMOTE_FILES = [
    ("/root/wan22_benchmark/outputs/wan22_5b_q4_opt_p1_cafe.mp4", "wan22_5b_q4_opt_p1_cafe.mp4"),
    ("/root/wan22_benchmark/outputs/wan22_5b_q4_opt_p1_cafe_frame16.png", "wan22_5b_q4_opt_p1_cafe_frame16.png"),
    ("/root/wan22_benchmark/outputs/wan22_5b_q4_opt_p2_puppy.mp4", "wan22_5b_q4_opt_p2_puppy.mp4"),
    ("/root/wan22_benchmark/outputs/wan22_5b_q4_opt_p2_puppy_frame16.png", "wan22_5b_q4_opt_p2_puppy_frame16.png"),
    ("/root/wan22_benchmark/outputs/wan22_5b_q4_opt_p3_dragon.mp4", "wan22_5b_q4_opt_p3_dragon.mp4"),
    ("/root/wan22_benchmark/outputs/wan22_5b_q4_opt_p3_dragon_frame16.png", "wan22_5b_q4_opt_p3_dragon_frame16.png"),
    ("/root/wan22_benchmark/outputs/wan22_a14b_q4_opt_p1_cafe.mp4", "wan22_a14b_q4_opt_p1_cafe.mp4"),
    ("/root/wan22_benchmark/outputs/wan22_a14b_q4_opt_p1_cafe_frame16.png", "wan22_a14b_q4_opt_p1_cafe_frame16.png"),
    ("/root/wan22_benchmark/outputs/wan22_a14b_q4_opt_p2_puppy.mp4", "wan22_a14b_q4_opt_p2_puppy.mp4"),
    ("/root/wan22_benchmark/outputs/wan22_a14b_q4_opt_p2_puppy_frame16.png", "wan22_a14b_q4_opt_p2_puppy_frame16.png"),
    ("/root/wan22_benchmark/outputs/wan22_a14b_q4_opt_p3_dragon.mp4", "wan22_a14b_q4_opt_p3_dragon.mp4"),
    ("/root/wan22_benchmark/outputs/wan22_a14b_q4_opt_p3_dragon_frame16.png", "wan22_a14b_q4_opt_p3_dragon_frame16.png"),
    ("/root/wan22_benchmark/wan22_benchmark_metrics.json", "wan22_benchmark_metrics.json"),
]

def fetch_file(remote_path: str, local_name: str):
    print(f"\n[*] Bắt đầu tải {local_name}...")
    cmd_sz = [SCRIPT, "--url", URL, "--token", TOKEN, "-c", f"import os; print('SIZE:', os.path.getsize('{remote_path}'))"]
    res = subprocess.run(cmd_sz, capture_output=True, text=True)
    if "SIZE:" not in res.stdout:
        print(f"❌ Không tìm thấy file {remote_path} trên server! Output: {res.stdout} {res.stderr}")
        return False

    total_sz = int(res.stdout.split("SIZE:")[1].strip())
    print(f"    Tổng dung lượng: {total_sz:,} bytes ({round(total_sz/1024, 1)} KB)")

    chunk_size = 400 * 1024
    num_chunks = (total_sz + chunk_size - 1) // chunk_size
    all_bytes = bytearray()

    for idx in range(num_chunks):
        offset = idx * chunk_size
        code = f"""
import base64
with open('{remote_path}', 'rb') as fp:
    fp.seek({offset})
    chunk = fp.read({chunk_size})
    print('CHUNK:' + base64.b64encode(chunk).decode('ascii') + ':END')
"""
        cmd_c = [SCRIPT, "--url", URL, "--token", TOKEN, "-c", code]
        res_c = subprocess.run(cmd_c, capture_output=True, text=True)
        if "CHUNK:" not in res_c.stdout or ":END" not in res_c.stdout:
            print(f"❌ Lỗi chunk {idx+1}/{num_chunks}: Output: {res_c.stdout[:200]}")
            return False
        raw_b64 = res_c.stdout.split("CHUNK:")[1].split(":END")[0].strip()
        all_bytes.extend(base64.b64decode(raw_b64))
        print(f"    -> Đã tải chunk {idx+1}/{num_chunks} ({len(all_bytes):,}/{total_sz:,} bytes)")

    out_path = os.path.join(OUT_DIR, local_name)
    with open(out_path, "wb") as fp:
        fp.write(all_bytes)
    print(f"✅ Đã lưu thành công {out_path} ({len(all_bytes):,} bytes)")

    # Copy to artifact dir for report viewing
    if os.path.exists(ARTIFACT_DIR):
        artifact_path = os.path.join(ARTIFACT_DIR, local_name)
        shutil.copy2(out_path, artifact_path)
        print(f"    -> Đã đồng bộ sang artifact dir: {artifact_path}")
    return True

def main():
    print(f"{'='*80}")
    print(f"🚀 TẢI TOÀN BỘ TỆP KẾT QUẢ WAN2.2 (6 VIDEOS + 6 KEYFRAMES + JSON)")
    print(f"{'='*80}")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    success_count = 0
    for rem_path, loc_name in REMOTE_FILES:
        if fetch_file(rem_path, loc_name):
            success_count += 1

    print(f"\n{'='*80}")
    print(f"🎉 Hoàn tất: {success_count}/{len(REMOTE_FILES)} tệp đã được tải về thành công!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
