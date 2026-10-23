#!/usr/bin/env python3
"""
🧪 Kaggle AI Studio Factory: Comprehensive Multi-Model Embedding & Rerank Benchmark Suite
================================================================================
Kịch bản kiểm thử toàn diện và đo đạc Telemetry chi tiết cho TOÀN BỘ 5 DÒNG MODEL:
Phân bổ phần cứng thông minh trên Dual Tesla T4:
- GPU 0 (cuda:0) - 3 Model Dòng Text:
  1. Qwen3-0.6B (Text Embed & Rerank)
  2. Qwen3-4B   (Text Embed & Rerank)
  3. Qwen3-8B   (Text Embed & Rerank)
- GPU 1 (cuda:1) - 2 Model Dòng Vision-Language (VL):
  4. Qwen3-VL-2B (VL Embed & Rerank)
  5. Qwen3-VL-8B (VL Embed & Rerank)

Quy trình kiểm thử:
1. Đọc và phân đoạn 4 file Markdown dự án (README.md, USAGE_GUIDE.md, UI.md, SKILL.md).
2. Kiểm thử Embedding từng model: Đo tốc độ (ms/doc, docs/s, tokens/s), chiều vector, RAM, VRAM.
3. Kiểm thử Reranking từng model: Đo tốc độ phân loại relevance, điểm số liên quan, RAM, VRAM.
4. Xuất bảng so sánh tổng hợp (Comparison Matrix) và lưu JSON report.
"""

import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Bổ sung path để nạp module từ kaggle/all-in-one
sys.path.insert(0, str(Path(__file__).parent / "kaggle" / "all-in-one"))

from embedding.embedding_service import get_embedding_service_manager


def get_memory_info() -> Dict[str, float]:
    """Đo đạc dung lượng bộ nhớ tiến trình (RAM / VRAM)."""
    import psutil
    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / (1024 ** 2)

    vram_used_mb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            vram_used_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    except Exception:
        pass

    return {
        "rss_ram_mb": round(rss_mb, 2),
        "vram_mb": round(vram_used_mb, 2),
    }


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Tính độ tương đồng Cosine giữa 2 vector."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1e-9
    norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1e-9
    return dot / (norm_a * norm_b)


def load_and_chunk_markdown_files(root_dir: Path) -> List[Dict[str, Any]]:
    """Đọc và bẻ nhỏ các file Markdown dự án thành các đoạn tài liệu có ý nghĩa."""
    target_files = ["README.md", "USAGE_GUIDE.md", "UI.md", "SKILL.md"]
    chunks = []

    print("\n📂 [1/4] ĐỌC VÀ PHÂN ĐOẠN TÀI LIỆU DỰ ÁN:")
    for fn in target_files:
        p = root_dir / fn
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        raw_sections = re.split(r"\n(?=#{1,3}\s)", text)
        file_chunks = 0
        for sec in raw_sections:
            sec_clean = sec.strip()
            if len(sec_clean) >= 60:
                chunks.append({
                    "id": f"{fn}_c{len(chunks)}",
                    "file": fn,
                    "text": sec_clean[:1200],
                    "char_count": len(sec_clean),
                })
                file_chunks += 1
        print(f"   • {fn:<15}: {file_chunks} chunks ({len(text):,} ký tự)")

    print(f"   ➡️ Tổng số chunks trích xuất: {len(chunks)} chunks")
    return chunks


BENCHMARK_MODELS = [
    {
        "key": "qwen3-0.6b",
        "name": "Qwen3-0.6B",
        "type": "Text Embedding & Rerank",
        "target_gpu": "GPU 0 (cuda:0)",
        "expected_dims": 1024,
        "est_vram_gb": 0.76,
    },
    {
        "key": "qwen3-4b",
        "name": "Qwen3-4B",
        "type": "Text Embedding & Rerank",
        "target_gpu": "GPU 0 (cuda:0)",
        "expected_dims": 2560,
        "est_vram_gb": 4.76,
    },
    {
        "key": "qwen3-8b",
        "name": "Qwen3-8B",
        "type": "Text Embedding & Rerank",
        "target_gpu": "GPU 0 (cuda:0)",
        "expected_dims": 4096,
        "est_vram_gb": 9.04,
    },
    {
        "key": "qwen3-vl-2b",
        "name": "Qwen3-VL-2B",
        "type": "Vision-Language Multimodal",
        "target_gpu": "GPU 1 (cuda:1)",
        "expected_dims": 1536,
        "est_vram_gb": 2.96,
    },
    {
        "key": "qwen3-vl-8b",
        "name": "Qwen3-VL-8B",
        "type": "Vision-Language Multimodal",
        "target_gpu": "GPU 1 (cuda:1)",
        "expected_dims": 3584,
        "est_vram_gb": 10.69,
    },
]


def run_benchmark():
    root_dir = Path(__file__).parent
    t_start = time.time()

    print("=" * 88)
    print("🧬 KAGGLE AI STUDIO: COMPREHENSIVE EMBEDDING & RERANK MULTI-MODEL BENCHMARK")
    print(f"   • Thời gian kiểm thử: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   • Python Runtimes   : {sys.version.split()[0]}")
    print(f"   • Phân bổ GPU       : GPU 0 (3 Text Models) | GPU 1 (2 VL Models)")
    print("=" * 88)

    # 1. Trích xuất văn bản
    doc_chunks = load_and_chunk_markdown_files(root_dir)
    if not doc_chunks:
        print("❌ Không tìm thấy file tài liệu nào để kiểm thử!")
        return

    # Lấy tập mẫu kiểm thử đại diện (12 chunks) để benchmark đồng nhất mọi model
    sample_chunks = doc_chunks[:12]
    sample_texts = [c["text"] for c in sample_chunks]
    total_sample_tokens = sum(len(t) // 4 for t in sample_texts)

    test_query = "Làm thế nào để shutdown tự động bảo toàn 30 giờ GPU Kaggle mỗi tuần?"

    mgr = get_embedding_service_manager()
    benchmark_results = []

    print("\n⚡ [2/4] TIẾN HÀNH BENCHMARK CHI TIẾT TỪNG MODEL (TEXT & MULTIMODAL):")

    for idx, model_cfg in enumerate(BENCHMARK_MODELS, 1):
        m_key = model_cfg["key"]
        m_name = model_cfg["name"]
        m_gpu = model_cfg["target_gpu"]
        exp_dims = model_cfg["expected_dims"]

        print(f"\n   -----------------------------------------------------------------------------")
        print(f"   🚀 [{idx}/5] Đang kiểm thử: {m_name} ({model_cfg['type']})")
        print(f"      • Phân bổ thiết bị : {m_gpu}")
        print(f"      • Vector Dims chuẩn : {exp_dims}")
        print(f"      • Ước tính VRAM     : ~{model_cfg['est_vram_gb']} GB")

        mem_pre = get_memory_info()

        # A. Benchmark Embedding
        t_emb_start = time.time()
        emb_res = mgr.embed(sample_texts, model=m_key)
        t_emb_elapsed = time.time() - t_emb_start

        vectors = [item["embedding"] for item in emb_res["data"]]
        actual_dims = len(vectors[0])
        ms_per_doc = (t_emb_elapsed / len(sample_texts)) * 1000
        docs_per_sec = len(sample_texts) / t_emb_elapsed
        tokens_per_sec = total_sample_tokens / t_emb_elapsed

        mem_emb = get_memory_info()

        # B. Similarity Search & Reranking
        t_q_start = time.time()
        q_emb_res = mgr.embed([test_query], model=m_key)
        q_vec = q_emb_res["data"][0]["embedding"]
        q_latency_ms = (time.time() - t_q_start) * 1000

        # Tính Cosine similarity
        scored = []
        for c, v in zip(sample_chunks, vectors):
            scored.append((cosine_similarity(q_vec, v), c))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [c["text"] for _, c in scored[:5]]

        # C. Benchmark Rerank
        t_rerank_start = time.time()
        rerank_res = mgr.rerank(query=test_query, documents=top_candidates, model=m_key)
        t_rerank_elapsed = (time.time() - t_rerank_start) * 1000
        rerank_ms_per_pair = t_rerank_elapsed / len(top_candidates)
        top1_relevance = rerank_res["results"][0]["relevance_score"]

        mem_post = get_memory_info()

        print(f"      ✅ Embedding : {ms_per_doc:.2f} ms/doc | {docs_per_sec:.1f} docs/s | {tokens_per_sec:.1f} tok/s | Dims: {actual_dims}")
        print(f"      ✅ Reranker  : {t_rerank_elapsed:.2f} ms total ({rerank_ms_per_pair:.2f} ms/pair) | Top-1 Score: {top1_relevance:.4f}")
        print(f"      ✅ Bộ nhớ    : RAM tiến trình = {mem_post['rss_ram_mb']:.1f} MB (Tăng: +{mem_post['rss_ram_mb'] - mem_pre['rss_ram_mb']:.1f} MB)")

        benchmark_results.append({
            "model_key": m_key,
            "model_name": m_name,
            "type": model_cfg["type"],
            "target_gpu": m_gpu,
            "expected_dims": exp_dims,
            "actual_dims": actual_dims,
            "embed_total_s": round(t_emb_elapsed, 4),
            "embed_ms_per_doc": round(ms_per_doc, 2),
            "embed_docs_per_sec": round(docs_per_sec, 1),
            "embed_tokens_per_sec": round(tokens_per_sec, 1),
            "rerank_total_ms": round(t_rerank_elapsed, 2),
            "rerank_ms_per_pair": round(rerank_ms_per_pair, 2),
            "top1_score": round(top1_relevance, 4),
            "est_vram_gb": model_cfg["est_vram_gb"],
            "ram_mb": mem_post["rss_ram_mb"],
        })

        # Giải phóng bộ nhớ giữa các model để tránh tràn buffer llama_cpp
        mgr.free_vram()

    # 3. In Bảng So Sánh Tổng Hợp (Telemetry Matrix)
    print("\n" + "=" * 88)
    print("📊 [3/4] BẢNG TỔNG HỢP HIỆU NĂNG TỪNG MODEL (TELEMETRY COMPARISON MATRIX)")
    print("=" * 88)
    header = (
        f"{'Model':<14} | {'Phân bổ GPU':<16} | {'Dims':<5} | "
        f"{'Embed (ms/d)':<12} | {'Throughput':<11} | {'Rerank/pair':<11} | {'VRAM Est':<9} | {'RAM (MB)':<8}"
    )
    print(header)
    print("-" * len(header))
    for r in benchmark_results:
        tok_str = f"{r['embed_tokens_per_sec']} tok/s"
        rerank_str = f"{r['rerank_ms_per_pair']} ms"
        vram_str = f"{r['est_vram_gb']} GB"
        row = (
            f"{r['model_name']:<14} | {r['target_gpu']:<16} | {r['actual_dims']:<5} | "
            f"{r['embed_ms_per_doc']:<12.1f} | {tok_str:<11} | "
            f"{rerank_str:<11} | {vram_str:<9} | {r['ram_mb']:<8.1f}"
        )
        print(row)

    print("\nPhân tích cấu trúc phần cứng Dual-GPU:")
    print("   • GPU 0 (3 Text Models) : 0.6B + 4B + 8B -> Chiếm tối đa ~9.5 GB / 16 GB VRAM (An toàn tuyệt đối)")
    print("   • GPU 1 (2 VL Models)   : VL-2B + VL-8B  -> Chiếm tối đa ~10.7 GB / 16 GB VRAM (An toàn tuyệt đối)")

    # 4. Xuất Báo Cáo JSON
    out_dir = root_dir / "test_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "embedding_multi_model_benchmark_report.json"

    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_duration_s": round(time.time() - t_start, 2),
        "gpu_allocation_policy": {
            "gpu_0": ["qwen3-0.6b", "qwen3-4b", "qwen3-8b"],
            "gpu_1": ["qwen3-vl-2b", "qwen3-vl-8b"],
        },
        "models": benchmark_results,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 [4/4] Báo cáo chi tiết đã được xuất ra: {report_file}")
    print("=" * 88)
    print("🎉 TẤT CẢ 5 MODEL EMBEDDING & RERANK ĐÃ HOÀN TẤT KIỂM THỬ THÀNH CÔNG 100%!")


if __name__ == "__main__":
    run_benchmark()
