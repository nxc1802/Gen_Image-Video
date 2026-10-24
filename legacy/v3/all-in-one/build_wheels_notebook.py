#!/usr/bin/env python3
"""
🛠️ Generator: Tạo run_wheels_builder.ipynb và wheels-metadata.json
Xây dựng và xuất bản cả 3 bộ Dataset Runtime/Wheels:
- `kaggle-speech-wheels`
- `kaggle-vlm-wheels`
- `kaggle-image-video-wheels`
"""

import json
import os
from pathlib import Path


def build_notebook():
    curr_dir = Path(__file__).parent
    py_script_path = curr_dir / "build_wheels_dataset.py"

    with open(py_script_path, "r", encoding="utf-8") as f:
        py_code = f.read()

    kaggle_user = "cuongnguyen1802"
    kaggle_key = ""
    env_file = curr_dir / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as ef:
            for line in ef:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k == "KAGGLE_USERNAME" and v:
                        kaggle_user = v
                    elif k == "KAGGLE_KEY" and v:
                        kaggle_key = v

    local_kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_key and local_kaggle_json.exists():
        try:
            with open(local_kaggle_json, "r", encoding="utf-8") as kf:
                k_data = json.load(kf)
                kaggle_user = k_data.get("username", kaggle_user)
                kaggle_key = k_data.get("key", kaggle_key)
        except Exception:
            pass

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ⚙️ Kaggle 3-in-1 Studio Runtime & Wheels Builder\n",
                "Tự động đóng gói và xuất bản 3 bộ Dataset Runtime/Wheels độc lập:\n",
                "1. `kaggle-speech-wheels` (Speech AI Studio)\n",
                "2. `kaggle-vlm-wheels` (VLM AI Studio)\n",
                "3. `kaggle-image-video-wheels` (Image & Video AI Studio)"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 📦 [Cell 1] Cài đặt công cụ build\n",
                "!pip install -q wheel virtualenv kaggle"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ⚙️ [Cell 2] Tạo file mã nguồn build_wheels_dataset.py\n",
                f'py_content = """{py_code.replace(chr(92), chr(92)+chr(92)).replace(chr(34)+chr(34)+chr(34), chr(92)+chr(34)+chr(92)+chr(34)+chr(92)+chr(34))}"""\n',
                'with open("build_wheels_dataset.py", "w", encoding="utf-8") as f:\n',
                "    f.write(py_content)\n",
                'print("✅ Đã ghi thành công build_wheels_dataset.py!")',
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 🔑 [Cell 3] Cấu hình credentials từ Kaggle User Secrets hoặc biến môi trường\n",
                "import os\n",
                "try:\n",
                "    from kaggle_secrets import UserSecretsClient\n",
                "    user_secrets = UserSecretsClient()\n",
                "    for secret_key in ['KAGGLE_USERNAME', 'KAGGLE_KEY', 'NTFY_TOPIC']:\n",
                "        try:\n",
                "            val = user_secrets.get_secret(secret_key)\n",
                "            if val:\n",
                "                os.environ[secret_key] = val\n",
                "        except Exception:\n",
                "            pass\n",
                "except Exception:\n",
                "    pass\n",
                f"os.environ.setdefault('KAGGLE_USERNAME', '{kaggle_user}')\n",
                "print(f\"✅ Credentials đã sẵn sàng cho user: {os.environ.get('KAGGLE_USERNAME')}!\")",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 🚀 [Cell 4] Thực thi biên dịch và xuất bản cả 3 dataset wheels\n",
                "!python3 -u build_wheels_dataset.py"
            ],
        },
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    out_nb_path = curr_dir / "run_wheels_builder.ipynb"
    with open(out_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"✅ Đã tạo notebook: {out_nb_path}")

    meta = {
        "id": f"{kaggle_user}/kaggle-wheels-builder",
        "title": "Kaggle Wheels Builder",
        "code_file": "run_wheels_builder.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }

    meta_path = curr_dir / "wheels-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"✅ Đã tạo metadata: {meta_path}")


if __name__ == "__main__":
    build_notebook()
