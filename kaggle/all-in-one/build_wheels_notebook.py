#!/usr/bin/env python3
"""
🛠️ Generator: Tạo run_wheels_builder.ipynb và wheels-metadata.json
"""

import json
from pathlib import Path

def build_notebook():
    curr_dir = Path(__file__).parent
    py_script_path = curr_dir / "build_wheels_dataset.py"
    
    with open(py_script_path, "r", encoding="utf-8") as f:
        py_code = f.read()

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ⚙️ Kaggle Studio Wheels Builder\n",
                "Notebook tự động biên dịch Diffusers (hỗ trợ Wan2.2) và gom các gói .whl thành Kaggle Dataset `kaggle-studio-wheels`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 📦 [Cell 1] Cài đặt các công cụ build\n",
                "!pip install -q --upgrade wheel kaggle"
            ]
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
                '    f.write(py_content)\n',
                'print("✅ Đã ghi thành công build_wheels_dataset.py!")'
            ]
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
                "if os.path.exists('.env'):\n",
                "    try:\n",
                "        with open('.env', 'r') as f:\n",
                "            for line in f:\n",
                "                if '=' in line and not line.strip().startswith('#'):\n",
                "                    k, v = line.strip().split('=', 1)\n",
                "                    os.environ.setdefault(k.strip(), v.strip().strip('\"\\''))\n",
                "    except Exception:\n",
                "        pass\n",
                "print('✅ Credentials đã sẵn sàng!')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 🚀 [Cell 4] Thực thi biên dịch và xuất bản dataset wheels\n",
                "!python3 build_wheels_dataset.py"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_nb_path = curr_dir / "run_wheels_builder.ipynb"
    with open(out_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"✅ Đã tạo notebook: {out_nb_path}")

    meta = {
        "id": "nguynxuncngde180528/kaggle-wheels-builder",
        "title": "Kaggle Wheels Builder",
        "code_file": "run_wheels_builder.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }

    meta_path = curr_dir / "wheels-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"✅ Đã tạo metadata: {meta_path}")

if __name__ == "__main__":
    build_notebook()
