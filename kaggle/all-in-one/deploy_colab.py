#!/usr/bin/env python3
"""
🚀 Google Colab CLI Orchestrator (`deploy_colab.py`) — Multi-Account & Auto-Rotation
====================================================================================
Tự động hóa toàn bộ quy trình triển khai AI Studio lên **Google Colab** qua
`google-colab-cli` (`colab`), kết hợp **Google Drive** làm bộ nhớ đệm bền vững
và **Hỗ trợ Đa Tài Khoản (Multi-Account Pool & Auto-Failover Rotation)**:

1. 👥 **Quản lý Đa Tài Khoản (`~/.colab_accounts/<account_name>/`)**:
   - Thêm nhiều tài khoản Google Colab độc lập (không cần `gcloud`, chỉ cần mở link OAuth2
     trên trình duyệt ẩn danh / profile tương ứng và dán mã xác nhận):
     `python3 deploy_colab.py --add-account acc1`
     `python3 deploy_colab.py --add-account acc2`
     `python3 deploy_colab.py --add-account acc3`
   - Kiểm tra đồng loạt hạn mức Compute Units & phiên đang chạy của mọi tài khoản:
     `python3 deploy_colab.py --list-accounts`
   - Tự động luân phiên (`--account auto` - mặc định):
     * Tự động chọn tài khoản đang rảnh (`Active assignments: 0`) hoặc còn nhiều Compute Units nhất.
     * Nếu tài khoản đang chọn bị hết hạn mức GPU / cooldown khi gọi `colab new --gpu ...`,
       tự động **xoay vòng (failover)** sang tài khoản tiếp theo trong danh sách!
     * Có thể chạy **song song nhiều tài khoản cùng lúc** (ví dụ `acc1` chạy `vlm`, `acc2` chạy `image_video`).

2. 🏗️ **Chế độ `setup-drive` (CPU-Only Session - Không tốn GPU Compute Units)**:
   - Khởi tạo phiên Colab **CPU-only** (`colab new -s studio-builder --high-mem`).
   - Mount Google Drive (`colab drivemount`).
   - Đóng gói sẵn toàn bộ thư viện `.whl` + `llama-server` + `ComfyUI` (`*_runtime_bundle.tar.bin`)
     và tải trọn bộ Models lưu vào `/content/drive/MyDrive/AI_Studio_Colab/`.
   - 💡 **Mẹo dùng chung Google Drive cho nhiều tài khoản**: Chỉ cần chạy `setup-drive` trên `acc1`,
     sau đó Share thư mục `AI_Studio_Colab` cho `acc2`, `acc3` và chọn **"Add shortcut to Drive" (Thêm lối tắt vào Drive)**
     ngay tại `My Drive` của `acc2`, `acc3`. Mọi tài khoản sẽ đọc chung 100% Models & Wheels mà không tốn dung lượng hay tải lại!

3. 👁️ **Chế độ `vlm` (Mặc định: `--gpu A100 --high-mem`)**:
   - Khởi tạo phiên **NVIDIA A100** cho `Qwen 3.8 27B GGUF` + `mmproj` + `MTP`/`DFlash`.

4. 🎬 **Chế độ `image_video` (Mặc định: `--gpu G4 --high-mem`)**:
   - Khởi tạo phiên **NVIDIA G4** cho `Qwen-Image 2.1` + `MiniMax H3` (FL2VA / Ref2VA / Turbo) + `Wan 2.2`.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
REAL_USER_HOME = Path.home()
ACCOUNTS_ROOT = REAL_USER_HOME / ".colab_accounts"

# Cấu hình phần cứng mặc định tối ưu theo từng chế độ
MODE_DEFAULTS: Dict[str, Dict[str, Optional[str]]] = {
    "setup-drive": {
        "session": "studio-builder",
        "gpu": None,  # CPU-Only!
        "high_mem": "false",
    },
    "vlm": {
        "session": "studio-vlm",
        "gpu": "A100",
        "high_mem": "true",
    },
    "image_video": {
        "session": "studio-visual",
        "gpu": "G4",
        "high_mem": "true",
    },
    "speech": {
        "session": "studio-speech",
        "gpu": "T4",
        "high_mem": "false",
    },
    "embedding": {
        "session": "studio-embedding",
        "gpu": "T4",
        "high_mem": "false",
    },
    "full": {
        "session": "studio-full",
        "gpu": "A100",
        "high_mem": "true",
    },
}

# Các thư mục/file cần đóng gói theo từng chế độ (Selective Packaging)
MODE_FILES: Dict[str, List[str]] = {
    "vlm": [
        "config.py",
        "main.py",
        "factory.py",
        "core",
        "vlm",
        "tunnel",
        "configs/models_vlm.yaml",
        "configs/models.yaml",
    ],
    "image_video": [
        "config.py",
        "main.py",
        "factory.py",
        "core",
        "visual",
        "tunnel",
        "configs/models_image_video.yaml",
        "configs/models.yaml",
    ],
    "speech": [
        "config.py",
        "main.py",
        "factory.py",
        "core",
        "audio",
        "tunnel",
        "configs/models_speech.yaml",
        "configs/models.yaml",
    ],
    "embedding": [
        "config.py",
        "main.py",
        "factory.py",
        "core",
        "embedding",
        "tunnel",
        "configs/models_embedding.yaml",
        "configs/models.yaml",
    ],
    "full": [
        "config.py",
        "main.py",
        "factory.py",
        "core",
        "vlm",
        "visual",
        "audio",
        "embedding",
        "tunnel",
        "configs",
    ],
}


def load_dotenv_vars() -> Dict[str, str]:
    """Đọc biến môi trường từ .env (nếu có)."""
    env_vars: Dict[str, str] = {}
    for env_path in [ROOT_DIR / ".env", BASE_DIR / ".env"]:
        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and v:
                    env_vars[k] = v
    for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "HF_TOKEN", "NTFY_TOPIC", "COLAB_DRIVE_ROOT"):
        if os.environ.get(k):
            env_vars[k] = os.environ[k]
    return env_vars


def find_colab_bin() -> str:
    """Tìm đường dẫn binary `colab` của google-colab-cli."""
    candidates = [
        shutil.which("colab"),
        str(REAL_USER_HOME / ".local" / "bin" / "colab"),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    print("❌ Không tìm thấy `colab` CLI! Hãy cài đặt bằng: `uv tool install google-colab-cli`")
    sys.exit(1)


# ==============================================================================
# MULTI-ACCOUNT POOL MANAGEMENT (~/.colab_accounts/<account>/)
# ==============================================================================
def get_account_home(account: Optional[str]) -> Path:
    """
    Trả về thư mục HOME cô lập cho tài khoản Colab được chỉ định.
    Mỗi tài khoản lưu trữ riêng `.config/colab-cli/token.json` và `.config/colab-cli/sessions.json`.
    """
    if not account or account in ("default", "auto"):
        return REAL_USER_HOME

    acc_clean = re.sub(r"[^a-zA-Z0-9_.-]", "_", account.strip())
    acc_dir = ACCOUNTS_ROOT / acc_clean
    (acc_dir / ".config" / "colab-cli").mkdir(parents=True, exist_ok=True)

    # Symlink ~/.ssh để `colab ssh` hoạt động bình thường nếu người dùng cần
    real_ssh = REAL_USER_HOME / ".ssh"
    acc_ssh = acc_dir / ".ssh"
    if real_ssh.exists() and not acc_ssh.exists():
        try:
            os.symlink(str(real_ssh), str(acc_ssh))
        except Exception:
            pass
    return acc_dir


def list_configured_accounts() -> List[str]:
    """Liệt kê danh sách các tài khoản đã được cấu hình trong `~/.colab_accounts/`."""
    accounts: List[str] = []
    if ACCOUNTS_ROOT.exists():
        for d in sorted(ACCOUNTS_ROOT.iterdir()):
            if d.is_dir():
                token_file = d / ".config" / "colab-cli" / "token.json"
                adc_file = d / "adc.json"
                if token_file.exists() or adc_file.exists():
                    accounts.append(d.name)
    default_token = REAL_USER_HOME / ".config" / "colab-cli" / "token.json"
    if default_token.exists() and "default" not in accounts:
        accounts.append("default")
    return accounts


def build_colab_env(account: Optional[str]) -> Dict[str, str]:
    """Tạo môi trường thực thi cô lập `HOME` theo từng tài khoản."""
    env = os.environ.copy()
    acc_home = get_account_home(account)
    env["HOME"] = str(acc_home)
    adc_path = acc_home / "adc.json"
    if adc_path.exists():
        env["GOOGLE_APPLICATION_CREDENTIALS"] = str(adc_path)
    return env


def run_colab(
    colab_bin: str,
    args: List[str],
    account: Optional[str] = None,
    auth_mode: Optional[str] = None,
    check: bool = True,
    capture: bool = False,
    quiet: bool = False,
) -> subprocess.CompletedProcess:
    """Thực thi lệnh `colab ...` dưới ngữ cảnh tài khoản `account`."""
    cmd = [colab_bin]
    if auth_mode:
        cmd.extend(["--auth", auth_mode])
    cmd.extend(args)
    acc_label = f"[{account}] " if account and account != "default" else ""
    if not quiet:
        print(f"💻 {acc_label}$ {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        env=build_colab_env(account),
        check=check,
        text=True,
        capture_output=capture,
    )


def add_account_interactive(colab_bin: str, account_name: str, auth_mode: str = "oauth2") -> None:
    """
    Thêm hoặc đăng nhập lại một tài khoản Google Colab vào pool `~/.colab_accounts/<account_name>/`.
    Sử dụng luồng OAuth2 copy-paste (mở link trên bất kỳ trình duyệt / tab ẩn danh nào).
    """
    acc_home = get_account_home(account_name)
    token_file = acc_home / ".config" / "colab-cli" / "token.json"
    if token_file.exists():
        token_file.unlink()

    print("=" * 72)
    print(f"🔐 ĐĂNG NHẬP TÀI KHOẢN COLAB MỚI: '{account_name}'")
    print(f"📁 Hồ sơ lưu tại: {acc_home / '.config' / 'colab-cli'}")
    print("💡 Hãy mở đường link bên dưới trong trình duyệt (hoặc cửa sổ Ẩn danh / Profile Chrome")
    print("   của tài khoản Google bạn muốn thêm), cấp quyền và dán mã Authorization Code vào đây:")
    print("=" * 72)

    res = run_colab(colab_bin, ["whoami"], account=account_name, auth_mode=auth_mode, check=False)
    if res.returncode == 0:
        print(f"\n✅ Đã lưu xác thực cho tài khoản '{account_name}' thành công!")
        run_colab(colab_bin, ["usage"], account=account_name, auth_mode=auth_mode, check=False)
    else:
        print(f"\n❌ Đăng nhập tài khoản '{account_name}' chưa hoàn tất.")


def remove_account(account_name: str) -> None:
    """Xóa hồ sơ tài khoản khỏi `~/.colab_accounts/`."""
    acc_dir = ACCOUNTS_ROOT / account_name
    if acc_dir.exists():
        shutil.rmtree(acc_dir)
        print(f"🗑️ Đã xóa hồ sơ tài khoản '{account_name}' ({acc_dir}).")
    else:
        print(f"⚠️ Không tìm thấy hồ sơ tài khoản '{account_name}' trong {ACCOUNTS_ROOT}.")


def probe_account_info(colab_bin: str, account: str, auth_mode: Optional[str] = None) -> Dict[str, Any]:
    """Truy vấn thông tin email, số dư Compute Units và các phiên đang chạy của 1 tài khoản."""
    info: Dict[str, Any] = {
        "account": account,
        "email": "unknown",
        "balance": 0.0,
        "rate": 0.0,
        "assignments": 0,
        "sessions_raw": "",
        "valid": False,
    }
    who_res = run_colab(
        colab_bin, ["whoami"], account=account, auth_mode=auth_mode, check=False, capture=True, quiet=True
    )
    if who_res.returncode == 0 and who_res.stdout:
        info["valid"] = True
        m_email = re.search(r"email\s*:\s*(\S+)", who_res.stdout, re.IGNORECASE)
        if m_email:
            info["email"] = m_email.group(1)

    usage_res = run_colab(
        colab_bin, ["usage"], account=account, auth_mode=auth_mode, check=False, capture=True, quiet=True
    )
    if usage_res.returncode == 0 and usage_res.stdout:
        info["valid"] = True
        m_bal = re.search(r"Current balance:\s*([0-9.]+)", usage_res.stdout)
        m_rate = re.search(r"Usage rate:\s*([0-9.]+)", usage_res.stdout)
        m_cnt = re.search(r"Active assignments:\s*([0-9.]+)", usage_res.stdout)
        if m_bal:
            info["balance"] = float(m_bal.group(1))
        if m_rate:
            info["rate"] = float(m_rate.group(1))
        if m_cnt:
            info["assignments"] = int(float(m_cnt.group(1)))

    sess_res = run_colab(
        colab_bin, ["sessions"], account=account, auth_mode=auth_mode, check=False, capture=True, quiet=True
    )
    if sess_res.returncode == 0 and sess_res.stdout:
        info["sessions_raw"] = sess_res.stdout.strip()

    return info


def show_all_accounts_summary(colab_bin: str, auth_mode: Optional[str] = None) -> None:
    """In bảng tổng hợp trạng thái toàn bộ các tài khoản Colab trong pool."""
    accounts = list_configured_accounts()
    print("=" * 88)
    print(f"👥 DANH SÁCH TÀI KHOẢN GOOGLE COLAB ĐÃ CẤU HÌNH ({len(accounts)} tài khoản)")
    print("=" * 88)
    if not accounts:
        print("⚠️ Chưa có tài khoản nào được lưu. Hãy thêm tài khoản bằng lệnh:")
        print("   python3 deploy_colab.py --add-account acc1")
        print("   python3 deploy_colab.py --add-account acc2")
        print("=" * 88)
        return

    print(f"{'TÀI KHOẢN':<14} | {'EMAIL':<30} | {'COMPUTE UNITS':<15} | {'ACTIVE VMs':<10} | {'TRẠNG THÁI'}")
    print("-" * 88)
    for acc in accounts:
        info = probe_account_info(colab_bin, acc, auth_mode=auth_mode)
        status_str = "✅ Sẵn sàng" if info["valid"] else "❌ Hết hạn / Lỗi"
        bal_str = f"{info['balance']:.2f} ({info['rate']:.1f}/h)"
        print(f"{acc:<14} | {info['email']:<30} | {bal_str:<15} | {info['assignments']:<10} | {status_str}")
    print("=" * 88)


def select_ordered_accounts(
    colab_bin: str,
    requested_account: Optional[str],
    session_name: str,
    mode: str,
    auth_mode: Optional[str] = None,
) -> List[str]:
    """
    Trả về danh sách tài khoản theo thứ tự ưu tiên để thực thi hoặc xoay vòng (auto-rotate):
    - Nếu người dùng chỉ định cụ thể `--account acc2`, chỉ trả về `['acc2']`.
    - Nếu `--account auto` (hoặc không truyền):
      1. Ưu tiên tài khoản đang giữ phiên `session_name` hoạt động.
      2. Ưu tiên tài khoản có `assignments == 0` (chưa chạy VM nào) và còn nhiều Compute Units nhất.
    """
    if requested_account and requested_account != "auto":
        return [requested_account]

    accounts = list_configured_accounts()
    if not accounts:
        return ["default"]
    if len(accounts) == 1:
        return accounts

    # Kiểm tra file ghi nhớ tài khoản đang chạy của mode này
    active_marker = BASE_DIR / f".active_colab_account_{mode}.txt"
    remembered = active_marker.read_text(encoding="utf-8").strip() if active_marker.exists() else None

    scored: List[Tuple[int, int, float, str]] = []
    for acc in accounts:
        info = probe_account_info(colab_bin, acc, auth_mode=auth_mode)
        if not info["valid"]:
            continue
        has_target_session = 1 if (session_name in info["sessions_raw"]) else 0
        is_remembered = 1 if (acc == remembered and has_target_session) else 0
        # Sắp xếp giảm dần theo: (is_remembered, has_target_session, -assignments, balance)
        scored.append((is_remembered, has_target_session, -info["assignments"], info["balance"], acc))

    if not scored:
        return accounts

    scored.sort(reverse=True)
    ordered = [item[-1] for item in scored]
    print(f"🔄 [Multi-Account Pool] Thứ tự ưu tiên tài khoản: {' -> '.join(ordered)}")
    return ordered


def save_active_account(mode: str, account: str) -> None:
    """Lưu lại tên tài khoản đang vận hành mode hiện tại để `--status` và `--stop` tự nhận diện."""
    try:
        (BASE_DIR / f".active_colab_account_{mode}.txt").write_text(f"{account}\n", encoding="utf-8")
    except Exception:
        pass


def load_active_account(mode: str) -> Optional[str]:
    """Đọc tên tài khoản đang vận hành mode hiện tại (nếu có)."""
    marker = BASE_DIR / f".active_colab_account_{mode}.txt"
    if marker.exists():
        val = marker.read_text(encoding="utf-8").strip()
        if val:
            return val
    return None


def ensure_session_and_drive_with_rotation(
    colab_bin: str,
    candidate_accounts: List[str],
    session_name: str,
    gpu: Optional[str],
    high_mem: bool,
    mount_drive: bool = True,
    auth_mode: Optional[str] = None,
) -> str:
    """
    Khởi tạo phiên Colab (CPU hoặc GPU) và mount Google Drive.
    Nếu có nhiều tài khoản trong `candidate_accounts`, tự động xoay vòng (failover)
    sang tài khoản tiếp theo nếu tài khoản hiện tại bị hết quota GPU hoặc lỗi cấp phát VM!
    Trả về tên tài khoản đã khởi tạo thành công.
    """
    last_err = None
    for idx, acc in enumerate(candidate_accounts, 1):
        print(f"\n👤 [Tài khoản {idx}/{len(candidate_accounts)}: '{acc}'] Đang kiểm tra phiên '{session_name}'...")
        sess_res = run_colab(
            colab_bin, ["sessions"], account=acc, auth_mode=auth_mode, check=False, capture=True, quiet=True
        )
        session_exists = session_name in (sess_res.stdout or "")

        if not session_exists:
            new_args = ["new", "-s", session_name]
            if gpu and gpu.upper() not in ("CPU", "NONE"):
                new_args.extend(["--gpu", gpu.upper()])
            if high_mem:
                new_args.append("--high-mem")
            hw_label = f"GPU {gpu.upper()}" if (gpu and gpu.upper() not in ("CPU", "NONE")) else "CPU-Only"
            print(f"🚀 [{acc}] Đang cấp phát phiên Colab '{session_name}' ({hw_label}, high_mem={high_mem})...")
            res = run_colab(colab_bin, new_args, account=acc, auth_mode=auth_mode, check=False)
            if res.returncode != 0:
                last_err = f"Account '{acc}' failed to provision {hw_label} (exit {res.returncode})"
                print(f"⚠️ [{acc}] Không thể cấp phát {hw_label} trên tài khoản này (có thể hết hạn mức GPU / Compute Units).")
                if idx < len(candidate_accounts):
                    print(f"🔄 Đang tự động chuyển đổi (rotate) sang tài khoản tiếp theo: '{candidate_accounts[idx]}'...")
                continue
        else:
            print(f"♻️ [{acc}] Tái sử dụng phiên Colab đang chạy: '{session_name}'")

        if mount_drive:
            chk_code = (
                "import os, shutil; "
                "mounted = os.path.isdir('/content/drive/MyDrive') or os.path.isdir('/content/drive/My Drive'); "
                "print('DRIVE_ALREADY_MOUNTED=1' if mounted else 'DRIVE_ALREADY_MOUNTED=0'); "
                "not mounted and os.path.exists('/content/drive') and shutil.rmtree('/content/drive', ignore_errors=True)"
            )
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
                tf.write(chk_code)
                chk_script = tf.name
            try:
                chk_res = run_colab(
                    colab_bin,
                    ["exec", "-s", session_name, "-f", chk_script],
                    account=acc,
                    auth_mode=auth_mode,
                    check=False,
                    capture=True,
                    quiet=True,
                )
            finally:
                if os.path.exists(chk_script):
                    os.unlink(chk_script)

            if "DRIVE_ALREADY_MOUNTED=1" in (chk_res.stdout or ""):
                print(f"✅ [{acc}] Google Drive đã được mount sẵn tại '/content/drive' trên phiên '{session_name}'.")
            else:
                print(f"📂 [{acc}] Đang mount Google Drive vào '/content/drive' trên phiên '{session_name}'...")
                dm_res = run_colab(
                    colab_bin, ["drivemount", "-s", session_name], account=acc, auth_mode=auth_mode, check=False
                )
                if dm_res.returncode != 0:
                    print(f"⚠️ [{acc}] Cảnh báo: mount Google Drive trả về mã {dm_res.returncode}.")

        return acc

    raise RuntimeError(
        f"❌ Đã thử toàn bộ {len(candidate_accounts)} tài khoản ({candidate_accounts}) nhưng không thể cấp phát phiên Colab. "
        f"Lỗi cuối: {last_err}"
    )


def create_code_archive(mode: str) -> Path:
    """Đóng gói mã nguồn gọn nhẹ cho chế độ được chọn."""
    rel_paths = MODE_FILES.get(mode, MODE_FILES["full"])
    tmp_tar = Path(tempfile.gettempdir()) / f"studio_colab_{mode}.tar.gz"
    if tmp_tar.exists():
        tmp_tar.unlink()

    with tarfile.open(tmp_tar, "w:gz") as tar:
        for rel in rel_paths:
            src = BASE_DIR / rel
            if not src.exists():
                continue
            tar.add(
                str(src),
                arcname=rel,
                filter=lambda ti: None
                if ("__pycache__" in ti.name or ti.name.endswith(".pyc") or ".git" in ti.name)
                else ti,
            )
    return tmp_tar


def run_setup_drive_mode(
    colab_bin: str,
    candidate_accounts: List[str],
    session_name: str,
    high_mem: bool,
    drive_root: str,
    targets: str,
    skip_wheels: bool,
    skip_models: bool,
    include_wan22: bool,
    force: bool,
    keep_alive: bool,
    auth_mode: Optional[str],
    env_vars: Dict[str, str],
) -> None:
    """Thực thi chế độ `setup-drive` trên phiên Colab CPU-Only."""
    active_acc = ensure_session_and_drive_with_rotation(
        colab_bin=colab_bin,
        candidate_accounts=candidate_accounts,
        session_name=session_name,
        gpu=None,  # Luôn dùng CPU-Only để tiết kiệm Compute Units!
        high_mem=high_mem,
        mount_drive=True,
        auth_mode=auth_mode,
    )
    save_active_account("setup-drive", active_acc)

    builder_local = BASE_DIR / "colab_drive_builder.py"
    run_colab(
        colab_bin,
        ["upload", "-s", session_name, str(builder_local), "/content/colab_drive_builder.py"],
        account=active_acc,
        auth_mode=auth_mode,
        check=True,
    )

    cmd_args = [
        "/content/colab_drive_builder.py",
        "--drive-root",
        drive_root,
        "--targets",
        targets,
    ]
    if skip_wheels:
        cmd_args.append("--skip-wheels")
    if skip_models:
        cmd_args.append("--skip-models")
    if include_wan22:
        cmd_args.append("--include-wan22")
    if force:
        cmd_args.append("--force")

    runner_code = f"""
import os, subprocess, sys
env = os.environ.copy()
env.update({repr(env_vars)})
env["COLAB_DRIVE_ROOT"] = {repr(drive_root)}
cmd = [sys.executable] + {repr(cmd_args)}
proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()
if proc.returncode != 0:
    raise RuntimeError(f"colab_drive_builder.py exited with code {{proc.returncode}}")
print("COLAB_DRIVE_BUILD_SUCCESS=1", flush=True)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(runner_code)
        runner_script = tf.name

    try:
        print(f"🏗️ [{active_acc}] Đang chạy `colab_drive_builder.py` trên phiên CPU-Only (tải Models & đóng gói Wheels vào Drive)...")
        res = run_colab(
            colab_bin,
            ["exec", "-s", session_name, "--timeout", "7200", "-f", runner_script],
            account=active_acc,
            auth_mode=auth_mode,
            check=True,
        )
    finally:
        if os.path.exists(runner_script):
            os.remove(runner_script)

    if not keep_alive:
        print(f"🛑 [{active_acc}] Tự động dừng phiên CPU-Only '{session_name}' sau khi hoàn tất chuẩn bị Google Drive...")
        run_colab(colab_bin, ["stop", "-s", session_name], account=active_acc, auth_mode=auth_mode, check=False)


def build_remote_boot_script(
    mode: str,
    drive_root: str,
    env_vars: Dict[str, str],
) -> str:
    """
    Sinh đoạn script Python chạy trên Colab GPU VM để:
    1. Giải nén mã nguồn `/content/studio_code.tar.gz` vào `/content/studio`.
    2. Tự động nhận diện đường dẫn Google Drive (kể cả khi thư mục `AI_Studio_Colab` được share từ tài khoản khác qua Shortcut).
    3. Giải nén offline runtime bundle từ `wheels/<mode>_runtime_bundle.tar.bin` giúp khởi động trong vài giây.
    4. Khởi chạy `main.py --mode <mode>` dưới nền và đợi Cloudflare Quick Tunnel URL xuất hiện.
    """
    bundle_map = {
        "vlm": "vlm_runtime_bundle.tar.bin",
        "image_video": "image_video_runtime_bundle.tar.bin",
        "speech": "speech_runtime_bundle.tar.bin",
        "embedding": "embedding_runtime_bundle.tar.bin",
        "full": "vlm_runtime_bundle.tar.bin",
    }
    bundle_name = bundle_map.get(mode, "vlm_runtime_bundle.tar.bin")

    return f"""
import os, sys, time, glob, shutil, subprocess

drive_root = {repr(drive_root)}
# Hỗ trợ tự động dò tìm nếu thư mục được Share từ tài khoản chính sang tài khoản phụ
if not os.path.exists(drive_root):
    for alt_root in [
        "/content/drive/MyDrive/AI_Studio_Colab",
        "/content/drive/Shareddrives/AI_Studio_Colab",
    ] + glob.glob("/content/drive/.shortcut-targets-by-id/*/AI_Studio_Colab"):
        if os.path.exists(alt_root):
            drive_root = alt_root
            print(f"🔗 Phát hiện Shared Google Drive folder tại: {{drive_root}}", flush=True)
            break

mode = {repr(mode)}
extra_env = {repr(env_vars)}
bundle_path = os.path.join(drive_root, "wheels", {repr(bundle_name)})
models_dir = os.path.join(drive_root, "models")

os.makedirs("/content/studio", exist_ok=True)
subprocess.run("tar -xzf /content/studio_code.tar.gz -C /content/studio", shell=True, check=True)

# Ghi constraint giữ nguyên torch mặc định của Colab GPU VM
c_file = "/tmp/colab_torch_constraints.txt"
try:
    import torch, torchaudio, torchvision
    with open(c_file, "w") as cf:
        cf.write(f"torch=={{torch.__version__.split('+')[0]}}\\n")
        cf.write(f"torchaudio=={{torchaudio.__version__.split('+')[0]}}\\n")
        cf.write(f"torchvision=={{torchvision.__version__.split('+')[0]}}\\n")
except Exception:
    pass

# 1. Khôi phục siêu tốc từ Offline Bundle trên Google Drive (nếu có)
if os.path.exists(bundle_path) and os.path.getsize(bundle_path) > 10_000_000:
    print(f"⚡ [Fast-Boot] Phát hiện Offline Bundle trên Google Drive: {{bundle_path}}", flush=True)
    stage_dir = "/tmp/studio_bundle_extract"
    if not os.path.exists("/tmp/.studio_bundle_installed_" + mode):
        shutil.rmtree(stage_dir, ignore_errors=True)
        os.makedirs(stage_dir, exist_ok=True)
        subprocess.run(f"tar -xf '{{bundle_path}}' -C {{stage_dir}}", shell=True, check=True)

        # Binaries (cloudflared)
        if os.path.exists(f"{{stage_dir}}/bin/cloudflared"):
            shutil.copy2(f"{{stage_dir}}/bin/cloudflared", "/usr/local/bin/cloudflared")
            os.chmod("/usr/local/bin/cloudflared", 0o755)
            shutil.copy2(f"{{stage_dir}}/bin/cloudflared", "/tmp/cloudflared")
            os.chmod("/tmp/cloudflared", 0o755)

        # llama-server & noble-libs (cho VLM & Embedding)
        if os.path.isdir(f"{{stage_dir}}/llama-cpp"):
            shutil.rmtree("/tmp/llama-cpp", ignore_errors=True)
            shutil.move(f"{{stage_dir}}/llama-cpp", "/tmp/llama-cpp")
            subprocess.run("chmod -R +x /tmp/llama-cpp", shell=True, check=False)
        if os.path.isdir(f"{{stage_dir}}/noble-libs"):
            shutil.rmtree("/tmp/noble-libs", ignore_errors=True)
            shutil.move(f"{{stage_dir}}/noble-libs", "/tmp/noble-libs")
            subprocess.run("chmod -R +x /tmp/noble-libs", shell=True, check=False)

        # ComfyUI & Custom Nodes (cho Image/Video)
        if os.path.isdir(f"{{stage_dir}}/ComfyUI") and not os.path.exists("/content/ComfyUI"):
            shutil.move(f"{{stage_dir}}/ComfyUI", "/content/ComfyUI")
            with open("/tmp/.comfyui_deps_ready", "w") as mf:
                mf.write("ok")

        # Cài đặt offline wheels siêu tốc
        wheels_dir = f"{{stage_dir}}/wheels"
        if os.path.isdir(wheels_dir) and glob.glob(f"{{wheels_dir}}/*.whl"):
            print("📦 Đang cài đặt Offline Wheels từ Google Drive...", flush=True)
            subprocess.run(
                f"{{sys.executable}} -m pip install -q --no-index --find-links={{wheels_dir}} -c {{c_file}} {{wheels_dir}}/*.whl || "
                f"{{sys.executable}} -m pip install -q --no-deps --find-links={{wheels_dir}} {{wheels_dir}}/*.whl",
                shell=True,
                check=False,
            )

        with open("/tmp/.studio_bundle_installed_" + mode, "w") as f:
            f.write("ok")
        shutil.rmtree(stage_dir, ignore_errors=True)
        print("✅ Hoàn tất khôi phục Offline Bundle từ Google Drive!", flush=True)
else:
    print("ℹ️ Chưa có Offline Bundle trên Drive, cài đặt nhanh các gói cơ bản...", flush=True)
    subprocess.run(
        f"{{sys.executable}} -m pip install -q -c {{c_file}} fastapi uvicorn python-multipart requests httpx sse-starlette pydantic pyyaml huggingface_hub accelerate",
        shell=True,
        check=False,
    )
    if mode in ("vlm", "embedding", "full") and not glob.glob("/tmp/llama-cpp/**/llama-server", recursive=True):
        print("📥 Đang tải llama-server b8763 CUDA 12.4...", flush=True)
        os.makedirs("/tmp/llama-cpp", exist_ok=True)
        subprocess.run(
            "curl -sL --retry 3 'https://github.com/ggml-org/llama.cpp/releases/download/b8763/llama-b8763-bin-ubuntu-x64.tar.gz' -o /tmp/llama.tar.gz && "
            "curl -sL --retry 3 'https://github.com/ggml-org/llama.cpp/releases/download/b8763/cudart-llama-bin-linux-cuda-12.4-x64.tar.gz' -o /tmp/cudart.tar.gz && "
            "tar -xzf /tmp/llama.tar.gz -C /tmp/llama-cpp && tar -xzf /tmp/cudart.tar.gz -C /tmp/llama-cpp && chmod -R +x /tmp/llama-cpp",
            shell=True,
            check=False,
        )

# 2. Dừng tiến trình main.py cũ nếu đang chạy
subprocess.run("pkill -f '/content/studio/main.py' || true", shell=True, check=False)
for p_url in ["/content/public_url.txt", "/content/studio/public_url.txt"]:
    if os.path.exists(p_url):
        os.remove(p_url)

# 3. Khởi chạy AI Studio Server trong nền
env = os.environ.copy()
env.update(extra_env)
env["COLAB_DRIVE_ROOT"] = drive_root
env["COLAB_MODELS_DIR"] = models_dir
env["COMFYUI_DIR"] = "/content/ComfyUI"
env["STUDIO_DEPLOY_MODE"] = mode

llama_bins = glob.glob("/tmp/llama-cpp/**/llama-server", recursive=True)
if llama_bins:
    env["LLAMA_SERVER_BIN"] = llama_bins[0]

log_file = open("/content/studio_server.log", "w", encoding="utf-8")
proc = subprocess.Popen(
    [sys.executable, "/content/studio/main.py", "--mode", mode, "--port", "8000"],
    cwd="/content/studio",
    env=env,
    stdout=log_file,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print(f"🚀 AI Studio Server (mode={{mode}}, PID={{proc.pid}}) đang khởi động...", flush=True)

# 4. Chờ Cloudflare Quick Tunnel URL xuất hiện
deadline = time.time() + 540.0
found_url = None
while time.time() < deadline:
    if proc.poll() is not None:
        log_file.flush()
        tail = open("/content/studio_server.log", "r", encoding="utf-8", errors="ignore").read()[-3000:]
        raise RuntimeError(f"AI Studio Server exited early (code {{proc.returncode}}):\\n{{tail}}")

    for cand in ["/content/public_url.txt", "/content/studio/public_url.txt", os.path.join(drive_root, "public_url.txt")]:
        if os.path.exists(cand):
            content = open(cand, "r", encoding="utf-8").read().strip()
            if content.startswith("https://"):
                found_url = content
                break
    if found_url:
        break
    time.sleep(2.0)

if found_url:
    print(f"\\n🎯 COLAB_PUBLIC_URL={{found_url}}\\n", flush=True)
else:
    tail = open("/content/studio_server.log", "r", encoding="utf-8", errors="ignore").read()[-2500:]
    raise TimeoutError(f"Chưa nhận được URL sau 540s. Log cuối:\\n{{tail}}")
"""


def show_status(colab_bin: str, account: str, session_name: str, auth_mode: Optional[str]) -> None:
    """Kiểm tra trạng thái phiên Colab, hạn mức Compute Units và sức khỏe AI Studio Server."""
    print("=" * 72)
    print(f"📡 DANH SÁCH PHIÊN COLAB ĐANG HOẠT ĐỘNG (Tài khoản: '{account}')")
    print("=" * 72)
    run_colab(colab_bin, ["sessions"], account=account, auth_mode=auth_mode, check=False)

    print("\n" + "=" * 72)
    print(f"💳 HẠN MỨC COMPUTE UNITS & GPUs KHẢ DỤNG (Tài khoản: '{account}')")
    print("=" * 72)
    run_colab(colab_bin, ["usage"], account=account, auth_mode=auth_mode, check=False)

    probe_code = """
import os, urllib.request
for p in ["/content/public_url.txt", "/content/studio/public_url.txt"]:
    if os.path.exists(p):
        print("🌐 Public Tunnel URL:", open(p).read().strip())
        break
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as r:
        print("💚 Local /health:", r.read().decode("utf-8")[:600])
except Exception as e:
    print("⚠️ Local /health chưa phản hồi:", e)
if os.path.exists("/content/studio_server.log"):
    lines = open("/content/studio_server.log", errors="ignore").readlines()[-20:]
    print("\\n📜 20 dòng log mới nhất (/content/studio_server.log):")
    print("".join(lines))
"""
    print("\n" + "=" * 72)
    print(f"🔍 TRẠNG THÁI SERVER TRÊN PHIÊN '{session_name}' (Tài khoản: '{account}')")
    print("=" * 72)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(probe_code)
        probe_path = tf.name
    try:
        run_colab(colab_bin, ["exec", "-s", session_name, "-f", probe_path], account=account, auth_mode=auth_mode, check=False)
    finally:
        if os.path.exists(probe_path):
            os.remove(probe_path)


def main():
    parser = argparse.ArgumentParser(
        description="Triển khai AI Studio lên Google Colab (Hỗ trợ Đa Tài Khoản Luân Phiên, A100 cho VLM, G4 cho Image/Video)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["setup-drive", "vlm", "image_video", "speech", "embedding", "full"],
        default="vlm",
        help="Chế độ triển khai: setup-drive (CPU-only tải model & wheel vào Drive), vlm (A100), image_video (G4), speech (T4), embedding (T4)",
    )
    parser.add_argument(
        "--account",
        "-a",
        type=str,
        default="auto",
        help="Tên tài khoản Colab sử dụng (ví dụ: acc1, acc2) hoặc 'auto' để tự động chọn & xoay vòng khi hết quota (mặc định: auto)",
    )
    parser.add_argument(
        "--add-account",
        type=str,
        default=None,
        metavar="ACCOUNT_NAME",
        help="Thêm/đăng nhập một tài khoản Google Colab mới vào pool (ví dụ: --add-account acc1)",
    )
    parser.add_argument(
        "--remove-account",
        type=str,
        default=None,
        metavar="ACCOUNT_NAME",
        help="Xóa một tài khoản khỏi pool (ví dụ: --remove-account acc1)",
    )
    parser.add_argument(
        "--list-accounts",
        action="store_true",
        help="Liệt kê toàn bộ các tài khoản Colab trong pool kèm số dư Compute Units và số phiên đang chạy",
    )
    parser.add_argument(
        "--session",
        "-s",
        type=str,
        default=None,
        help="Tên phiên Colab (mặc định tự chọn theo mode: studio-builder, studio-vlm, studio-visual...)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Loại GPU trên Colab: A100, G4, L4, T4, H100 hoặc CPU (mặc định: A100 cho vlm, G4 cho image_video, CPU cho setup-drive)",
    )
    parser.add_argument(
        "--high-mem",
        dest="high_mem",
        action="store_true",
        default=None,
        help="Bật chế độ High-RAM trên Colab (mặc định bật cho A100, G4 và setup-drive)",
    )
    parser.add_argument(
        "--no-high-mem",
        dest="high_mem",
        action="store_false",
        help="Tắt chế độ High-RAM (dùng Standard RAM)",
    )
    parser.add_argument(
        "--drive-root",
        type=str,
        default="/content/drive/MyDrive/AI_Studio_Colab",
        help="Thư mục gốc lưu trữ Models & Wheels trên Google Drive (mặc định: /content/drive/MyDrive/AI_Studio_Colab)",
    )
    parser.add_argument(
        "--no-drive",
        action="store_true",
        help="Bỏ qua bước mount Google Drive (chỉ dùng ổ cứng tạm của Colab VM)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="vlm,image_video",
        help="[Dành cho --mode setup-drive] Các phân hệ cần tải vào Drive: vlm,image_video,speech,embedding,all",
    )
    parser.add_argument(
        "--skip-wheels",
        action="store_true",
        help="[Dành cho --mode setup-drive] Bỏ qua đóng gói wheels bundle, chỉ tải models",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="[Dành cho --mode setup-drive] Bỏ qua tải models, chỉ đóng gói wheels bundle",
    )
    parser.add_argument(
        "--include-wan22",
        action="store_true",
        help="[Dành cho --mode setup-drive] Tải thêm trọng số Wan 2.2 bên cạnh Qwen-Image 2.1 & MiniMax H3",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="[Dành cho --mode setup-drive] Buộc đóng gói lại wheels bundle trên Drive",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="[Dành cho --mode setup-drive] Giữ nguyên phiên CPU sau khi tải xong thay vì tự động dừng",
    )
    parser.add_argument(
        "--auth",
        type=str,
        choices=["adc", "oauth2"],
        default=None,
        help="Phương thức xác thực truyền cho `colab` CLI (mặc định: oauth2 cho đa tài khoản)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Kiểm tra trạng thái phiên Colab, hạn mức Compute Units và log server",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Dừng và giải phóng phiên Colab tương ứng ngay lập tức",
    )
    args = parser.parse_args()

    colab_bin = find_colab_bin()

    # Quản lý Pool Tài Khoản
    if args.add_account:
        add_account_interactive(colab_bin, args.add_account, auth_mode=args.auth or "oauth2")
        return

    if args.remove_account:
        remove_account(args.remove_account)
        return

    if args.list_accounts:
        show_all_accounts_summary(colab_bin, auth_mode=args.auth)
        return

    defaults = MODE_DEFAULTS[args.mode]
    session_name = args.session or defaults["session"] or f"studio-{args.mode}"
    gpu_type = args.gpu if args.gpu is not None else defaults["gpu"]
    high_mem = args.high_mem if args.high_mem is not None else (defaults["high_mem"] == "true")

    if args.stop:
        target_acc = (
            args.account
            if (args.account and args.account != "auto")
            else (load_active_account(args.mode) or "default")
        )
        print(f"🛑 [{target_acc}] Đang dừng phiên Colab '{session_name}'...")
        run_colab(colab_bin, ["stop", "-s", session_name], account=target_acc, auth_mode=args.auth, check=False)
        return

    if args.status:
        target_acc = (
            args.account
            if (args.account and args.account != "auto")
            else (load_active_account(args.mode) or "default")
        )
        show_status(colab_bin, target_acc, session_name, auth_mode=args.auth)
        return

    env_vars = load_dotenv_vars()
    candidate_accounts = select_ordered_accounts(
        colab_bin=colab_bin,
        requested_account=args.account,
        session_name=session_name,
        mode=args.mode,
        auth_mode=args.auth,
    )

    # Nhánh 1: Chuẩn bị Models & Offline Wheels trên phiên CPU-Only
    if args.mode == "setup-drive":
        run_setup_drive_mode(
            colab_bin=colab_bin,
            candidate_accounts=candidate_accounts,
            session_name=session_name,
            high_mem=high_mem,
            drive_root=args.drive_root,
            targets=args.targets,
            skip_wheels=args.skip_wheels,
            skip_models=args.skip_models,
            include_wan22=args.include_wan22,
            force=args.force_rebuild,
            keep_alive=args.keep_alive,
            auth_mode=args.auth,
            env_vars=env_vars,
        )
        return

    # Nhánh 2: Khởi chạy AI Studio Server trên GPU (có Auto-Rotate qua danh sách tài khoản)
    active_acc = ensure_session_and_drive_with_rotation(
        colab_bin=colab_bin,
        candidate_accounts=candidate_accounts,
        session_name=session_name,
        gpu=gpu_type,
        high_mem=high_mem,
        mount_drive=not args.no_drive,
        auth_mode=args.auth,
    )
    save_active_account(args.mode, active_acc)

    archive_path = create_code_archive(args.mode)
    try:
        print(f"📦 [{active_acc}] Đang tải gói mã nguồn ({archive_path.stat().st_size / 1024:.1f} KB) lên phiên '{session_name}'...")
        run_colab(
            colab_bin,
            ["upload", "-s", session_name, str(archive_path), "/content/studio_code.tar.gz"],
            account=active_acc,
            auth_mode=args.auth,
            check=True,
        )
    finally:
        if archive_path.exists():
            archive_path.unlink()

    boot_code = build_remote_boot_script(
        mode=args.mode,
        drive_root=args.drive_root,
        env_vars=env_vars,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(boot_code)
        boot_script = tf.name

    try:
        print(f"⚡ [{active_acc}] Đang khởi chạy AI Studio (mode={args.mode}, gpu={gpu_type}) trên Colab...")
        res = run_colab(
            colab_bin,
            ["exec", "-s", session_name, "--timeout", "900", "-f", boot_script],
            account=active_acc,
            auth_mode=args.auth,
            check=True,
            capture=True,
        )
        output_text = (res.stdout or "") + "\n" + (res.stderr or "")
        print(output_text)

        match = re.search(r"COLAB_PUBLIC_URL=(https://[a-zA-Z0-9.-]+\.trycloudflare\.com/v1)", output_text)
        if match:
            live_url = match.group(1)
            for out_file in [BASE_DIR / "current_url.txt", BASE_DIR / f"current_url_{args.mode}.txt"]:
                out_file.write_text(f"{live_url}\n", encoding="utf-8")
            print("\n" + "=" * 72)
            print(f"🎉 GOOGLE COLAB AI STUDIO ({args.mode.upper()} - GPU {gpu_type} - Account '{active_acc}') ĐÃ SẴN SÀNG!")
            print(f"🔗 Live OpenAI Base URL : {live_url}")
            print(f"💾 Đã lưu URL tại       : {BASE_DIR / 'current_url.txt'}")
            print(f"🛑 Để tắt phiên GPU     : python3 deploy_colab.py --mode {args.mode} --stop")
            print("=" * 72)
    finally:
        if os.path.exists(boot_script):
            os.remove(boot_script)


if __name__ == "__main__":
    main()
