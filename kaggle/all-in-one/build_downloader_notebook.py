#!/usr/bin/env python3
"""
🛠️ Generator: Tạo run_downloader.ipynb và downloader-metadata.json
"""

import json
from pathlib import Path

def build_notebook(models: list = None):
    curr_dir = Path(__file__).parent
    py_script_path = curr_dir / "download_models_to_dataset.py"
    
    if models is None:
        models = ["flux2_klein_4b", "qwen3_4b"]

    with open(py_script_path, "r", encoding="utf-8") as f:
        py_code = f.read()

    run_cmds = "\n".join([f"!python3 download_models_to_dataset.py --model={m}" for m in models])

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 📥 Kaggle Model Downloader & Dataset Publisher\n",
                f"Notebook tự động tải mô hình GGUF ({', '.join(models)}) từ Hugging Face và xuất bản thành các Kaggle Dataset riêng biệt."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 📦 [Cell 1] Cài đặt các thư viện cần thiết\n",
                "!pip install -q --upgrade huggingface_hub kaggle"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ⚙️ [Cell 2] Tạo file mã nguồn thực thi download_models_to_dataset.py\n",
                f'py_content = """{py_code.replace(chr(92), chr(92)+chr(92)).replace(chr(34)+chr(34)+chr(34), chr(92)+chr(34)+chr(92)+chr(34)+chr(92)+chr(34))}"""\n',
                'with open("download_models_to_dataset.py", "w", encoding="utf-8") as f:\n',
                '    f.write(py_content)\n',
                'print("✅ Đã ghi thành công download_models_to_dataset.py!")'
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
                "    for secret_key in ['KAGGLE_USERNAME', 'KAGGLE_KEY', 'HF_TOKEN', 'NTFY_TOPIC']:\n",
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
                "os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'\n",
                "print('✅ Credentials và môi trường đã sẵn sàng!')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# 🚀 [Cell 4] Thực thi tải mô hình ({', '.join(models)}) và xuất bản Dataset lên Kaggle\n",
                f"{run_cmds}\n"
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

    out_nb_path = curr_dir / "run_downloader.ipynb"
    with open(out_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"✅ Đã tạo notebook: {out_nb_path}")

    meta = {
        "id": "nguynxuncngde180528/kaggle-model-downloader",
        "title": "Kaggle Model Downloader",
        "code_file": "run_downloader.ipynb",
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

    meta_path = curr_dir / "downloader-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"✅ Đã tạo metadata: {meta_path}")

if __name__ == "__main__":
    build_notebook()
