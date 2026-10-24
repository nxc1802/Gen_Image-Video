#!/usr/bin/env python3
"""
🏆 NVIDIA RULER Context-Scaling Benchmark Suite (4K → 8K → 16K → 32K → 64K → 128K)
Evaluates Qwen 3.8 27B Instruct with MTP Speculative Decoding across expanding context windows.
"""

import os
import sys
import time
import json
import re
import threading
from typing import Dict, Any, List, Optional, Tuple

import requests

OUTPUT_DIR = "/kaggle/working/test_outputs" if os.path.exists("/kaggle") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../test_outputs")

CONTEXT_SCALES = [
    {"label": "4K", "n_ctx": 4096, "target_prompt_tokens": 3000, "max_tokens": 1000},
    {"label": "8K", "n_ctx": 8192, "target_prompt_tokens": 6200, "max_tokens": 1200},
    {"label": "16K", "n_ctx": 16384, "target_prompt_tokens": 12500, "max_tokens": 1500},
    {"label": "32K", "n_ctx": 32768, "target_prompt_tokens": 26000, "max_tokens": 1800},
    {"label": "64K", "n_ctx": 65536, "target_prompt_tokens": 54000, "max_tokens": 2000},
    {"label": "128K", "n_ctx": 131072, "target_prompt_tokens": 110000, "max_tokens": 2500},
]

# Shared status for live monitoring
RULER_STATE = {
    "status": "idle",
    "current_scale": None,
    "completed_scales": [],
    "progress_percent": 0.0,
    "current_task": None,
    "current_step": "",
    "results": {},
    "error": None,
    "start_time": None,
    "end_time": None,
}


def generate_infrastructure_haystack(target_tokens: int) -> Tuple[str, Dict[str, str]]:
    """
    Generates realistic infrastructure telemetry text scaled to target_tokens,
    inserting 3 distinct cryptographic/security needles at depth 15%, 50%, and 85%.
    """
    # Base telemetry pattern (~100 tokens per repeating unit)
    telemetry_patterns = [
        "CoreRoutingMesh: interface eth{i} ingress_pkt_rate={rate_in}Mpps egress_pkt_rate={rate_out}Mpps crc_errs=0 drop_ratio=0.00001.",
        "DistributedKVReplica: shard_id={shard} raft_term=42 log_index={idx} commit_latency_us=182 replication_quorum=healthy.",
        "TLSGateSession: handshakes_active=842 cipher=TLS_AES_256_GCM_SHA384 ticket_reuse_pct=89.4% session_cache_hit=true.",
        "NVLinkPcieMonitor: pcie_tx_throughput_gbps={bw_tx:.2f} pcie_rx_throughput_gbps={bw_rx:.2f} link_retries=0 aer_fatal=false.",
        "HostMemoryAllocator: slab_reclaimed_kb=1048576 hugepages_free=2048 page_fault_rate=0.002% swap_active=false.",
    ]

    needle_1 = "\n>>> [SECURITY_AUDIT_LOG - DEPTH 15%]: SECRET_OVERRIDE_AUTH_TOKEN = 'SIGMA-ALPHA-9942' <<<\n"
    needle_2 = "\n>>> [SECURITY_AUDIT_LOG - DEPTH 50%]: DISASTER_RECOVERY_STORAGE_PORT = '18443' <<<\n"
    needle_3 = "\n>>> [SECURITY_AUDIT_LOG - DEPTH 85%]: REGIONAL_FAILOVER_CLUSTER_NAME = 'CLUSTER-PACIFIC-NODE-08' <<<\n"

    expected_needles = {
        "SECRET_OVERRIDE_AUTH_TOKEN": "SIGMA-ALPHA-9942",
        "DISASTER_RECOVERY_STORAGE_PORT": "18443",
        "REGIONAL_FAILOVER_CLUSTER_NAME": "CLUSTER-PACIFIC-NODE-08",
    }

    # ~1.3 tokens per word in technical text; calculate required lines
    target_words = int(target_tokens / 1.3)
    words_per_block = 65
    total_blocks = max(10, target_words // words_per_block)

    block_15 = max(2, int(total_blocks * 0.15))
    block_50 = max(block_15 + 2, int(total_blocks * 0.50))
    block_85 = max(block_50 + 2, int(total_blocks * 0.85))

    lines = ["=== GLOBAL DISTRIBUTED CLOUD TELEMETRY HAYSTACK CORPUS ===\n"]

    for b in range(total_blocks):
        if b == block_15:
            lines.append(needle_1)
        elif b == block_50:
            lines.append(needle_2)
        elif b == block_85:
            lines.append(needle_3)

        pat = telemetry_patterns[b % len(telemetry_patterns)]
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " + pat.format(
            i=b % 4,
            rate_in=10 + (b * 7) % 50,
            rate_out=9 + (b * 5) % 45,
            shard=b % 128,
            idx=10000 + b,
            bw_tx=11.5 + (b % 10) * 0.1,
            bw_rx=12.0 + (b % 8) * 0.1,
        )
        lines.append(line)

    lines.append("\n=== END OF GLOBAL TELEMETRY CORPUS ===")
    corpus = "\n".join(lines)

    prompt = f"""Dưới đây là tập dữ liệu nhật ký kỹ thuật hạ tầng quy mô lớn (RULER Multi-Key Retrieval Benchmark):

{corpus}

YÊU CẦU ĐÁNH GIÁ (RULER MK-NIAH Evaluation):
Hãy đọc quét toàn bộ ngữ cảnh nhật ký trên và tìm kiếm chính xác 3 giá trị thông số bảo mật khẩn cấp đã được chèn vào các vị trí khác nhau:
1. SECRET_OVERRIDE_AUTH_TOKEN là gì?
2. DISASTER_RECOVERY_STORAGE_PORT là cổng nào?
3. REGIONAL_FAILOVER_CLUSTER_NAME là định danh cụm nào?

Quy định:
- Trong phần tư duy (<think>), hãy trích dẫn vị trí và dòng nhật ký xung quanh từng thông số tìm thấy.
- Trong phần phản hồi cuối cùng, lập bảng tổng kết 3 thông số cùng giá trị chính xác tương ứng."""

    return prompt, expected_needles


def score_retrieval(content: str, expected_needles: Dict[str, str]) -> Dict[str, Any]:
    details = {}
    matched = 0
    clean_text = content.replace("'", "").replace('"', "").replace("`", "")
    for k, v in expected_needles.items():
        found = (v in content) or (v in clean_text)
        details[k] = {"expected": v, "matched": found}
        if found:
            matched += 1
    total = len(expected_needles)
    pct = round((matched / total) * 100.0, 2) if total else 0.0
    return {
        "accuracy_pct": pct,
        "passed_checks": matched,
        "total_checks": total,
        "details": details,
    }


def stream_ruler_completion(
    base_url: str,
    prompt: str,
    max_tokens: int = 1500,
    enable_thinking: bool = True,
    timeout: int = 1800,
) -> Dict[str, Any]:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "qwen38-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.9,
        "stream": True,
        "stream_options": {"include_usage": True},
        "timings_per_token": True,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }

    t0 = time.perf_counter()
    ttft = None
    t_think_end = None
    reasoning_chunks = []
    content_chunks = []
    reasoning_events = 0
    content_events = 0
    server_usage = None
    server_timings = None

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            resp.encoding = "utf-8"
            for raw_line in resp.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                line = (
                    raw_line.decode("utf-8", errors="replace").strip()
                    if isinstance(raw_line, (bytes, bytearray))
                    else str(raw_line).strip()
                )
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except Exception:
                    continue

                if chunk.get("usage"):
                    server_usage = chunk["usage"]
                if chunk.get("timings"):
                    server_timings = chunk["timings"]

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                c_tok = delta.get("content") or ""
                r_tok = delta.get("reasoning_content") or ""

                now = time.perf_counter()
                if (c_tok or r_tok) and ttft is None:
                    ttft = now - t0

                if r_tok:
                    reasoning_chunks.append(r_tok)
                    reasoning_events += 1

                if c_tok:
                    if t_think_end is None and reasoning_events > 0:
                        t_think_end = now
                    content_chunks.append(c_tok)
                    content_events += 1

    except Exception as e:
        total_elapsed = time.perf_counter() - t0
        return {
            "error": str(e),
            "total_elapsed_seconds": total_elapsed,
            "total_tokens": 0,
            "ttft_seconds": ttft or total_elapsed,
            "decode_tokens_per_second": 0.0,
        }

    total_elapsed = time.perf_counter() - t0
    ttft = ttft or total_elapsed
    t_think_end = t_think_end or (t0 + ttft)
    think_duration = max(0.001, t_think_end - (t0 + ttft))
    answer_duration = max(0.001, total_elapsed - t_think_end)
    total_decode_time = max(0.001, total_elapsed - ttft)

    full_reasoning = "".join(reasoning_chunks)
    full_content = "".join(content_chunks)
    total_tokens = reasoning_events + content_events
    decode_speed = (total_tokens / total_decode_time) if total_decode_time > 0 else 0.0

    return {
        "reasoning_content": full_reasoning,
        "content": full_content,
        "reasoning_tokens": reasoning_events,
        "content_tokens": content_events,
        "total_tokens": total_tokens,
        "ttft_seconds": round(ttft, 3),
        "think_duration_seconds": round(think_duration, 3),
        "answer_duration_seconds": round(answer_duration, 3),
        "total_elapsed_seconds": round(total_elapsed, 3),
        "decode_tokens_per_second": round(decode_speed, 2),
        "server_usage": server_usage,
        "server_timings": server_timings,
    }


def execute_ruler_scaling_suite(server_base_url: str = "http://127.0.0.1:8000") -> Dict[str, Any]:
    """
    Executes the full RULER Context Scaling Suite: 4K → 8K → 16K → 32K → 64K → 128K.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    global RULER_STATE
    RULER_STATE["status"] = "running"
    RULER_STATE["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    RULER_STATE["results"] = {}
    RULER_STATE["completed_scales"] = []
    RULER_STATE["progress_percent"] = 0.0
    RULER_STATE["error"] = None

    total_scales = len(CONTEXT_SCALES)

    for idx, scale_cfg in enumerate(CONTEXT_SCALES):
        scale_label = scale_cfg["label"]
        n_ctx = scale_cfg["n_ctx"]
        target_tokens = scale_cfg["target_prompt_tokens"]
        max_tokens = scale_cfg["max_tokens"]

        RULER_STATE["current_scale"] = scale_label
        RULER_STATE["current_step"] = f"Reconfiguring llama-server to n_ctx={n_ctx} ({scale_label})"
        RULER_STATE["progress_percent"] = round((idx / total_scales) * 100, 1)

        print(f"\n{'='*76}\n🚀 [RULER SCALE {idx+1}/{total_scales}] Target: {scale_label} (n_ctx={n_ctx})\n{'='*76}")

        # 1. Reconfigure llama-server
        reconf_url = f"{server_base_url}/v1/admin/reconfigure-vlm"
        reconf_payload = {"spec_mode": "mtp", "n_ctx": n_ctx}
        if n_ctx >= 65536:
            reconf_payload["tensor_split"] = "0.54,0.46"

        t_reconf_0 = time.time()
        try:
            r_reconf = requests.post(reconf_url, json=reconf_payload, timeout=600)
            reconf_data = r_reconf.json()
        except Exception as e:
            reconf_data = {"status": "error", "error": f"Failed to call reconfigure endpoint: {e}"}

        if reconf_data.get("status") != "ok":
            print(f"❌ Failed to configure n_ctx={n_ctx}: {reconf_data}")
            RULER_STATE["results"][scale_label] = {
                "status": "OOM_OR_FAILED",
                "n_ctx": n_ctx,
                "error": reconf_data.get("last_error") or reconf_data.get("error"),
                "reconfigure_time_s": round(time.time() - t_reconf_0, 2),
            }
            continue

        print(f"✅ llama-server reconfigured to {scale_label} in {reconf_data.get('startup_seconds', 0)}s!")

        # 2. Get VRAM baseline before query
        vram_before = {}
        try:
            h_resp = requests.get(f"{server_base_url}/health", timeout=5).json()
            vram_before = h_resp.get("vram_status", {})
        except Exception:
            pass

        # 3. Generate Haystack Prompt
        RULER_STATE["current_step"] = f"Generating Haystack Prompt (~{target_tokens} tokens)"
        prompt, expected = generate_infrastructure_haystack(target_tokens)
        prompt_char_len = len(prompt)
        print(f"📄 Prompt ready: {prompt_char_len} chars (~{target_tokens} tokens)")

        # 4. Stream RULER Evaluation
        RULER_STATE["current_step"] = f"Executing Prompt Evaluation & Streaming (TTFT + Reasoning + Decode)"
        t_run_0 = time.perf_counter()
        stream_res = stream_ruler_completion(
            base_url=server_base_url,
            prompt=prompt,
            max_tokens=max_tokens,
            enable_thinking=True,
            timeout=1800,
        )

        # 5. Score output
        if "error" not in stream_res:
            score_data = score_retrieval(stream_res.get("content", ""), expected)
        else:
            score_data = {"accuracy_pct": 0.0, "passed_checks": 0, "total_checks": len(expected), "details": {}}

        # 6. Get VRAM after query
        vram_after = {}
        try:
            h_resp = requests.get(f"{server_base_url}/health", timeout=5).json()
            vram_after = h_resp.get("vram_status", {})
        except Exception:
            pass

        scale_result = {
            "scale": scale_label,
            "n_ctx": n_ctx,
            "target_prompt_tokens": target_tokens,
            "status": "PASS" if "error" not in stream_res else "ERROR",
            "accuracy": score_data,
            "timing": {
                "ttft_seconds": stream_res.get("ttft_seconds"),
                "decode_tokens_per_second": stream_res.get("decode_tokens_per_second"),
                "total_elapsed_seconds": stream_res.get("total_elapsed_seconds"),
                "think_tokens": stream_res.get("reasoning_tokens", 0),
                "answer_tokens": stream_res.get("content_tokens", 0),
                "total_tokens": stream_res.get("total_tokens", 0),
                "server_usage": stream_res.get("server_usage"),
                "server_timings": stream_res.get("server_timings"),
            },
            "vram_before": vram_before,
            "vram_after": vram_after,
            "error": stream_res.get("error"),
        }

        RULER_STATE["results"][scale_label] = scale_result
        RULER_STATE["completed_scales"].append(scale_label)

        # Save individual markdown
        md_path = os.path.join(OUTPUT_DIR, f"ruler_{scale_label.lower()}_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# RULER Benchmark Report: Context Scale {scale_label} (n_ctx={n_ctx})\n\n")
            f.write(f"## Telemetry\n")
            f.write(f"- **Target Prompt Tokens**: {target_tokens}\n")
            f.write(f"- **TTFT**: {scale_result['timing']['ttft_seconds']} s\n")
            f.write(f"- **Decode Speed**: {scale_result['timing']['decode_tokens_per_second']} tokens/s\n")
            f.write(f"- **Accuracy**: {score_data['accuracy_pct']}%\n\n")
            f.write(f"## 💭 Thinking Process\n\n{stream_res.get('reasoning_content', '')}\n\n")
            f.write(f"## ✍️ Final Answer\n\n{stream_res.get('content', '')}\n")

        print(f"📊 [{scale_label}] Score: {score_data['accuracy_pct']}% | "
              f"TTFT: {scale_result['timing']['ttft_seconds']}s | "
              f"Decode: {scale_result['timing']['decode_tokens_per_second']} tok/s | "
              f"Saved to: {md_path}")

    RULER_STATE["status"] = "completed"
    RULER_STATE["progress_percent"] = 100.0
    RULER_STATE["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Save summary JSON
    summary_json_path = os.path.join(OUTPUT_DIR, "ruler_scale_report.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(RULER_STATE, f, indent=2, ensure_ascii=False)
    print(f"\n🎉 Full RULER Scaling Report saved to {summary_json_path}!")

    return RULER_STATE


def start_ruler_scaling_thread(server_base_url: str = "http://127.0.0.1:8000") -> threading.Thread:
    """Spawns execution in background thread."""
    t = threading.Thread(target=execute_ruler_scaling_suite, args=(server_base_url,), daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    execute_ruler_scaling_suite(base)
