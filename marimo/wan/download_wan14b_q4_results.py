#!/usr/bin/env python3
"""
📥 Script tải video & benchmark json của Wan2.1-14B 4-bit từ remote Marimo server
"""
import os
import subprocess
import base64
import json

URL = "https://sb-fb3c85dabdd3ea8a.sb.molab.run/"
TOKEN = "7e04d65ad03c2c678d8191a03cb22c48dcad9428d0b59fe8ecbb33902df5d33c"
SCRIPT = "/Users/nxc/.agents/skills/marimo-pair/scripts/execute-code.sh"
OUT_DIR = "/Volumes/WorkSpace/Project/Gen_Image:Video/marimo/wan"

FILES = [
    "wan21_14b_q4_opt_p1_cafe.mp4",
    "wan21_14b_q4_opt_p2_puppy.mp4",
    "wan21_14b_q4_opt_p3_dragon.mp4",
    "wan21_14b_q4_benchmark.json",
]

def fetch_file(filename: str):
    print(f"\n[*] Bắt đầu tải {filename}...")
    # 1. Lấy kích thước file
    cmd_sz = [SCRIPT, "--url", URL, "--token", TOKEN, "-c", f"import os; print('SIZE:', os.path.getsize('{filename}'))"]
    res = subprocess.run(cmd_sz, capture_output=True, text=True)
    if "SIZE:" not in res.stdout:
        print(f"❌ Không tìm thấy file {filename} trên server! Output: {res.stdout} {res.stderr}")
        return False
    
    total_sz = int(res.stdout.split("SIZE:")[1].strip())
    print(f"    Tổng dung lượng: {total_sz:,} bytes ({round(total_sz/1024, 1)} KB)")

    # 2. Tải theo chunk 400KB raw (khoảng 533KB base64, an toàn dưới giới hạn 1MB SSE)
    chunk_size = 400 * 1024
    num_chunks = (total_sz + chunk_size - 1) // chunk_size
    all_bytes = bytearray()

    for idx in range(num_chunks):
        offset = idx * chunk_size
        code = f"""
import base64
with open('{filename}', 'rb') as fp:
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

    out_path = os.path.join(OUT_DIR, filename)
    with open(out_path, "wb") as fp:
        fp.write(all_bytes)
    print(f"✅ Đã lưu thành công {out_path} ({len(all_bytes):,} bytes)")
    return True

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in FILES:
        fetch_file(f)
    print("\n🎉 Toàn bộ file đã được tải về máy thành công!")

if __name__ == "__main__":
    main()
