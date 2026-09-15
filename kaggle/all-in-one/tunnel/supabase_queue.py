"""
⚡ Supabase Queue Worker Module
Lắng nghe hàng đợi từ Cloud Database (Supabase) chạy song song với FastAPI.
Đảm bảo kết nối không bao giờ bị đứt gãy dù tunnel Cloudflare có bị reset.
"""

import logging
import threading
import time
from typing import Optional
from config import SUPABASE_URL, SUPABASE_KEY, ENABLE_SUPABASE
from visual.flux_image import get_flux_engine
from visual.wan_video import get_wan_engine

logger = logging.getLogger("SupabaseQueue")


def run_supabase_polling_loop():
    if not ENABLE_SUPABASE or not SUPABASE_URL or not SUPABASE_KEY:
        logger.info("Supabase Queue không được bật, bỏ qua.")
        return

    try:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info(f"⚡ Đã kết nối Supabase Broker: {SUPABASE_URL}")
    except Exception as e:
        logger.warning(f"Không thể khởi tạo Supabase client: {e}")
        return

    flux = get_flux_engine()

    while True:
        try:
            # Lấy các job đang pending
            res = (
                sb.table("image_jobs")
                .select("*")
                .eq("status", "pending")
                .order("created_at")
                .limit(1)
                .execute()
            )

            if res.data and len(res.data) > 0:
                job = res.data[0]
                job_id = job["id"]
                prompt = job.get("prompt", "")
                size = job.get("size", "1024x1024")
                steps = job.get("steps", 4)
                guidance = job.get("guidance", 0.0)
                seed = job.get("seed")

                logger.info(f"[SUPABASE QUEUE] 📥 Bắt đầu xử lý Job #{job_id}: '{prompt[:40]}...'")
                sb.table("image_jobs").update({"status": "processing"}).eq("id", job_id).execute()

                t0 = time.time()
                try:
                    b64_str, elapsed = flux.generate(
                        prompt=prompt,
                        size=size,
                        steps=steps,
                        guidance=guidance,
                        seed=seed,
                    )
                    sb.table("image_jobs").update(
                        {
                            "status": "completed",
                            "result_b64": b64_str,
                            "inference_time_sec": elapsed,
                            "device_name": "Kaggle 2x Tesla T4",
                        }
                    ).eq("id", job_id).execute()
                    logger.info(f"[SUPABASE QUEUE] ✅ Hoàn tất Job #{job_id} trong {elapsed:.2f}s!")
                except Exception as gen_err:
                    logger.error(f"[SUPABASE QUEUE] ❌ Lỗi xử lý Job #{job_id}: {gen_err}")
                    sb.table("image_jobs").update(
                        {
                            "status": "failed",
                            "error_message": str(gen_err),
                        }
                    ).eq("id", job_id).execute()

            time.sleep(1.0)
        except Exception as loop_err:
            time.sleep(3.0)


def start_supabase_worker():
    """Khởi chạy worker Supabase trong luồng ngầm (daemon thread)."""
    t = threading.Thread(target=run_supabase_polling_loop, daemon=True)
    t.start()
    logger.info("⚡ Supabase Queue Worker daemon đã khởi động.")
    return t
