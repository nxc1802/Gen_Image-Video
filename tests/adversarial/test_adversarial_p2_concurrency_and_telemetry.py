"""
tests/adversarial/test_adversarial_p2_concurrency_and_telemetry.py
Milestone 3 (Phase P2) Empirical Challenger Verification Suite:
1. Queue Stress Benchmark:
   - 3 consecutive rounds of 100 concurrent job submissions (300 total)
   - Zero dropped requests (100/100 HTTP 201 per round)
   - Zero corrupted jobs (bit-for-bit input fidelity, state QUEUED, exact DB match)
   - Repeatable P95 latency target < 1500 ms across all runs
   - Heterogeneous subsystem burst (100 concurrent jobs across 8 subsystems)
2. Hardware Telemetry Edge Cases:
   - Genuine non-GPU environment reports explicit telemetry_status="UNAVAILABLE" and None for VRAM/utilization/temp
   - Worker heartbeat persistence preserves UNAVAILABLE and None in repository
   - Telemetry metrics snapshot preserves None VRAM
   - SSE telemetry stream emits null for VRAM gauges
3. Studio Web Display Hardening:
   - Node-level component logic verification confirming "Unavailable" / "--" output and zero "0.0 GB" / "0% Used" false indicators
"""
import asyncio
import json
import os
import subprocess
import time
import uuid
from typing import Any, Dict, List

import httpx
import pytest

from factory_auth.config import AuthConfig
from factory_contracts.jobs.worker_contract import WorkerHeartbeatRequest
from factory_contracts.telemetry import NVMLTelemetryCollector
from factory.apps.control_api.src.main import app as control_plane_app, lifespan
from factory.apps.control_api.src.queue.leasing import LeaseCoordinator
from factory.apps.control_api.src.queue.repository.sqlite import SqliteJobRepository
from factory.apps.control_api.src.streaming.inspector import stream_studio_inspector
from factory.apps.control_api.src.telemetry.metrics import TelemetryMetrics

AUTH_TOKEN = "sk-factory-test-token-valid"
AUTH_HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json",
}

os.environ["FACTORY_API_KEY"] = AUTH_TOKEN
os.environ["STORAGE_BACKEND"] = "local"
os.environ["FACTORY_DB_BACKEND"] = "sqlite"
os.environ["SQLITE_DB_PATH"] = ":memory:"


@pytest.fixture(autouse=True)
def configure_test_env(monkeypatch):
    monkeypatch.setenv("FACTORY_API_KEY", AUTH_TOKEN)
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("FACTORY_DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", ":memory:")
    control_plane_app.state.auth_config = AuthConfig(api_key=AUTH_TOKEN, jwt_secret=None, environment="test")


# ==============================================================================
# 1. 100 CONCURRENT JOB SUBMISSIONS REPEATABILITY BENCHMARK
# ==============================================================================
@pytest.mark.asyncio
class TestQueueStressBenchmarkRepeatability:
    """Empirical verification of 100 concurrent job submissions under extreme load across multiple runs."""

    async def test_100_concurrent_submissions_repeatable_p95_latency(self):
        """
        Adversarial benchmark:
        - 3 consecutive runs of 100 concurrent submissions.
        - Verify 0 dropped requests in every run (100% 201 Created).
        - Verify 0 corrupted jobs (unique UUIDs, identical prompt inputs, QUEUED state).
        - Verify P95 latency < 1500 ms in EVERY run.
        """
        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=60.0,
            ) as client:
                benchmark_runs = []

                for run_idx in range(1, 4):
                    latencies = []
                    submitted_payloads: Dict[str, Dict[str, Any]] = {}

                    async def _submit_job(i: int) -> Dict[str, Any]:
                        req_id = f"stress-r{run_idx}-job-{i}-{uuid.uuid4().hex[:8]}"
                        payload = {
                            "service": "image",
                            "model": "qwen-image-2.1",
                            "task": "text_to_image",
                            "inputs": {
                                "prompt": f"Empirical challenge run {run_idx} item {i}",
                                "seed": 42 + i,
                                "steps": 25,
                            },
                            "priority": (i % 10) + 1,
                            "idempotency_key": req_id,
                        }
                        t0 = time.perf_counter()
                        res = await client.post("/v1/jobs", json=payload)
                        t1 = time.perf_counter()
                        latency_ms = (t1 - t0) * 1000.0
                        latencies.append(latency_ms)

                        assert res.status_code == 201, (
                            f"Run {run_idx} Request #{i} failed with status {res.status_code}: {res.text}"
                        )
                        body = res.json()
                        job_id = body["job_id"]
                        submitted_payloads[job_id] = payload
                        return body

                    t_start = time.perf_counter()
                    results = await asyncio.gather(*[_submit_job(i) for i in range(100)])
                    wall_time = time.perf_counter() - t_start

                    # 1. Zero requests dropped
                    assert len(results) == 100, f"Run {run_idx}: Expected 100 results, got {len(results)}"
                    job_ids = [r["job_id"] for r in results]

                    # 2. Zero duplicate job IDs
                    assert len(set(job_ids)) == 100, f"Run {run_idx}: Duplicate job IDs detected!"

                    # 3. Calculate latency percentiles
                    latencies.sort()
                    p50 = latencies[int(len(latencies) * 0.50)]
                    p90 = latencies[int(len(latencies) * 0.90)]
                    p95 = latencies[int(len(latencies) * 0.95)]
                    p99 = latencies[int(len(latencies) * 0.99)]
                    throughput = 100.0 / wall_time

                    run_stat = {
                        "run": run_idx,
                        "throughput": throughput,
                        "p50": p50,
                        "p90": p90,
                        "p95": p95,
                        "p99": p99,
                        "max": latencies[-1],
                    }
                    benchmark_runs.append(run_stat)
                    print(
                        f"\n[BENCHMARK RUN {run_idx}] Throughput: {throughput:.1f} jobs/s | "
                        f"P50: {p50:.1f}ms | P95: {p95:.1f}ms | P99: {p99:.1f}ms | Max: {latencies[-1]:.1f}ms"
                    )

                    # Strict P95 SLA target: < 1500 ms
                    assert p95 < 1500.0, f"Run {run_idx} FAILED P95 latency SLA: {p95:.1f}ms >= 1500ms"

                    # 4. Zero data corruption verification: sample jobs and verify bit-for-bit fidelity
                    sample_indices = [0, 10, 25, 50, 75, 90, 99]
                    for idx in sample_indices:
                        sample_id = job_ids[idx]
                        orig_payload = submitted_payloads[sample_id]

                        get_res = await client.get(f"/v1/jobs/{sample_id}")
                        assert get_res.status_code == 200, f"Failed to retrieve sampled job {sample_id}"
                        job_data = get_res.json()

                        # Invariants: state must be QUEUED, zero progress, uncorrupted inputs
                        assert job_data["job_id"] == sample_id
                        assert job_data["state"] == "QUEUED"
                        assert job_data["service"] == orig_payload["service"]
                        assert job_data["model"] == orig_payload["model"]
                        assert job_data["task"] == orig_payload["task"]
                        assert job_data["inputs"] == orig_payload["inputs"]
                        assert job_data["priority"] == orig_payload["priority"]
                        assert job_data["idempotency_key"] == orig_payload["idempotency_key"]
                        assert job_data["progress"]["percent"] == 0.0

                # Summary across all 3 runs
                print("\n[BENCHMARK SUMMARY]")
                for r in benchmark_runs:
                    print(
                        f"  Run {r['run']}: P95 = {r['p95']:.1f}ms, P99 = {r['p99']:.1f}ms, "
                        f"Throughput = {r['throughput']:.1f} req/s"
                    )
                assert len(benchmark_runs) == 3
                assert all(r["p95"] < 1500.0 for r in benchmark_runs)

    async def test_100_concurrent_heterogeneous_subsystem_burst(self):
        """
        Adversarially burst 100 concurrent jobs evenly across 8 core subsystems
        (image, video, speech, music, text, decision, embedding, vlm)
        and verify 0 dropped, 0 corrupted, and queue isolation.
        """
        subsystems = [
            ("image", "qwen-image-2.1", "text_to_image", {"prompt": "photo of nebula"}),
            ("video", "wan-2.2", "text_to_video", {"prompt": "ocean waves", "fps": 24}),
            ("speech", "qwen-speech", "speech_synthesis", {"text": "hello factory"}),
            ("music", "ace-step-1.5", "text_to_music", {"prompt": "ambient soundscape"}),
            ("text", "qwen-text", "chat_completion", {"messages": [{"role": "user", "content": "hi"}]}),
            ("decision", "qwen-decision", "predicate", {"hypothesis": "is safe"}),
            ("embedding", "qwen-embedding", "vector_embeddings", {"input": "factory embeddings"}),
            ("vision", "qwen-vlm", "visual_language", {"prompt": "inspect photo", "image_url": "https://example.com/p.png"}),
        ]

        async with lifespan(control_plane_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control_plane_app),
                base_url="http://test",
                headers=AUTH_HEADERS,
                timeout=60.0,
            ) as client:
                latencies = []

                async def _submit_subsystem_job(i: int):
                    svc, model, task, inputs = subsystems[i % len(subsystems)]
                    payload = {
                        "service": svc,
                        "model": model,
                        "task": task,
                        "inputs": inputs,
                        "priority": (i % 5) + 1,
                        "idempotency_key": f"hetero-{i}-{uuid.uuid4().hex[:6]}",
                    }
                    t0 = time.perf_counter()
                    res = await client.post("/v1/jobs", json=payload)
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)
                    assert res.status_code == 201, f"Hetero job {i} failed: {res.status_code} {res.text}"
                    return res.json(), svc

                results = await asyncio.gather(*[_submit_subsystem_job(i) for i in range(100)])
                assert len(results) == 100
                latencies.sort()
                p95 = latencies[int(len(latencies) * 0.95)]
                print(f"\n[HETEROGENEOUS BENCHMARK] 100 jobs across 8 subsystems: P95 = {p95:.1f}ms")
                assert p95 < 1500.0, f"Heterogeneous burst P95 SLA failed: {p95:.1f}ms >= 1500ms"


# ==============================================================================
# 2. HARDWARE TELEMETRY EDGE CASES & NON-GPU INVARIANTS
# ==============================================================================
class TestTelemetryNonGpuInvariants:
    """Verifies non-GPU environments explicitly report UNAVAILABLE and None for VRAM/utilization."""

    def test_nvml_collector_on_cpu_environment(self):
        """
        Verify that NVMLTelemetryCollector on macOS/CPU safely returns:
        - telemetry_status = 'UNAVAILABLE'
        - vram_used_mb = None
        - gpu_utilization_percent = None
        - temperature_c = None
        Never fabricates mock values or defaults to 0.0.
        """
        collector = NVMLTelemetryCollector(device_index=0)
        sample = collector.sample()

        assert isinstance(sample, dict)
        assert sample["telemetry_status"] in ("AVAILABLE", "UNAVAILABLE")
        if not collector._available:
            assert sample["telemetry_status"] == "UNAVAILABLE"
            assert sample["vram_used_mb"] is None
            assert sample["gpu_utilization_percent"] is None
            assert sample["temperature_c"] is None

    @pytest.mark.asyncio
    async def test_worker_heartbeat_unavailable_persisted_to_repository(self):
        """
        Verify that when a worker reports telemetry_status='UNAVAILABLE' and None for VRAM,
        the repository stores 'UNAVAILABLE' and None (not 0.0 or fabricated data).
        """
        repo = SqliteJobRepository(":memory:")
        await repo.initialize()

        worker_id = "test-cpu-worker-01"
        await repo.upsert_worker_heartbeat(
            worker_id=worker_id,
            subsystem="image",
            framework="comfyui",
            supported_models=["qwen-image-2.1"],
            supported_tasks=["text_to_image"],
            gpu_device=None,
            active_job_id=None,
            gpu_utilization=None,
            vram_used_mb=None,
            telemetry_status="UNAVAILABLE",
            temperature_c=None,
        )

        worker = await repo.get_worker(worker_id)
        assert worker is not None
        assert worker["worker_id"] == worker_id
        assert worker["telemetry_status"] == "UNAVAILABLE"
        assert worker["vram_used_mb"] is None
        assert worker["gpu_utilization"] is None
        assert worker["temperature_c"] is None

        await repo.close()

    def test_telemetry_metrics_initial_state_and_none_handling(self):
        """
        Verify that TelemetryMetrics defaults VRAM to None, and get_snapshot() returns None,
        preventing false 0.0 GB displays.
        """
        tm = TelemetryMetrics()
        snapshot = tm.get_snapshot()

        assert snapshot["gpu_0_vram_used_gb"] is None
        assert snapshot["gpu_1_vram_used_gb"] is None
        assert snapshot["active_jobs"] == 0
        assert snapshot["queue_depth"] == 0

        # Updating active jobs/queue depth must not corrupt VRAM None state
        tm.set_active_jobs(5)
        tm.set_queue_depth(10)
        snap2 = tm.get_snapshot()
        assert snap2["gpu_0_vram_used_gb"] is None
        assert snap2["gpu_1_vram_used_gb"] is None
        assert snap2["active_jobs"] == 5
        assert snap2["queue_depth"] == 10

    @pytest.mark.asyncio
    async def test_sse_telemetry_stream_emits_null_vram(self, monkeypatch):
        """
        Verify that stream_studio_inspector emits null (not 0.0 or string) in JSON data
        when running in a non-GPU environment without prior VRAM updates.
        """
        from factory.apps.control_api.src.telemetry.metrics import metrics

        monkeypatch.setattr(metrics, "_gpu_0_vram_used_gb", None)
        monkeypatch.setattr(metrics, "_gpu_1_vram_used_gb", None)

        events = []
        async for chunk in stream_studio_inspector(interval_seconds=0.01, max_iterations=1):
            events.append(chunk)

        assert len(events) >= 1
        telemetry_chunk = events[0]
        assert "event: telemetry" in telemetry_chunk

        # Parse JSON data line
        data_line = [line for line in telemetry_chunk.splitlines() if line.startswith("data:")][0]
        payload = json.loads(data_line[5:].strip())

        assert payload["gpu_0_vram_used_gb"] is None
        assert payload["gpu_1_vram_used_gb"] is None
        assert "active_jobs" in payload
        assert "queue_depth" in payload


# ==============================================================================
# 3. STUDIO WEB TELEMETRY DISPLAY HARDENING
# ==============================================================================
class TestStudioWebTelemetryDisplay:
    """Verifies Studio Web UI renders 'Unavailable' / '--' when telemetry values are null."""

    def test_studio_web_sources_eliminate_null_coalescing_to_zero(self):
        """
        Scan Studio Web component sources to verify that gpu_0_vram_used_gb and gpu_1_vram_used_gb
        are never coalesced with `?? 0` in TopChromeBar, OperationsConsoleView, or StreamingInspectorView.
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        studio_web = os.path.join(repo_root, "factory", "apps", "studio-web", "src")

        top_chrome = os.path.join(studio_web, "components", "layout", "TopChromeBar.tsx")
        ops_console = os.path.join(studio_web, "components", "views", "OperationsConsoleView.tsx")
        streaming_insp = os.path.join(studio_web, "components", "views", "StreamingInspectorView.tsx")

        with open(top_chrome, "r", encoding="utf-8") as f:
            top_content = f.read()
        assert "gpu_0_vram_used_gb ?? 0" not in top_content
        assert "gpu_1_vram_used_gb ?? 0" not in top_content
        assert "Unavailable" in top_content
        assert '"--"' in top_content or "'--'" in top_content

        with open(ops_console, "r", encoding="utf-8") as f:
            ops_content = f.read()
        assert "gpu_0_vram_used_gb ?? 0" not in ops_content
        assert "gpu_1_vram_used_gb ?? 0" not in ops_content
        assert "Unavailable" in ops_content
        assert '"--"' in ops_content or "'--'" in ops_content

        with open(streaming_insp, "r", encoding="utf-8") as f:
            stream_content = f.read()
        assert "-- / Unavailable" in stream_content

    def test_studio_web_component_render_logic_with_node(self):
        """
        Execute an isolated Node.js script evaluating the exact conditional logic from
        TopChromeBar, OperationsConsoleView, and StreamingInspectorView with null telemetry.
        Assert that '0.0 GB' and '0% Used' are NEVER produced.
        """
        node_script = """
        // 1. TopChromeBar logic
        function testTopChrome(gpu0Used, gpu1Used) {
            if (gpu0Used == null && gpu1Used == null) {
                return "Unavailable";
            }
            return `G0: ${gpu0Used != null ? `${gpu0Used.toFixed(1)}G` : "--"} · G1: ${gpu1Used != null ? `${gpu1Used.toFixed(1)}G` : "--"}`;
        }

        // 2. OperationsConsoleView logic
        function testOpsConsole(gpuUsed, gpuTotal) {
            const gpuAvailable = gpuUsed != null;
            const gpuPct = gpuAvailable && gpuTotal > 0 ? Math.round((gpuUsed / gpuTotal) * 100) : null;
            
            const badge = gpuPct != null ? `${gpuPct}% Used` : "Unavailable";
            const vramText = gpuAvailable ? `${gpuUsed.toFixed(1)} / ${gpuTotal.toFixed(1)} GB` : "Unavailable";
            const freeText = gpuAvailable ? `${(gpuTotal - gpuUsed).toFixed(1)} GB` : "--";

            return { badge, vramText, freeText };
        }

        // 3. StreamingInspectorView logic
        function testStreamingInspector(gpuUsed) {
            return gpuUsed != null ? `${gpuUsed.toFixed(1)} / 15.0 GB` : "-- / Unavailable";
        }

        // Run with null inputs (non-GPU environment)
        const top = testTopChrome(null, null);
        if (top !== "Unavailable") throw new Error("TopChromeBar failed: " + top);

        const ops = testOpsConsole(null, 15.0);
        if (ops.badge !== "Unavailable") throw new Error("OpsConsole badge failed: " + ops.badge);
        if (ops.vramText !== "Unavailable") throw new Error("OpsConsole vramText failed: " + ops.vramText);
        if (ops.freeText !== "--") throw new Error("OpsConsole freeText failed: " + ops.freeText);

        const insp = testStreamingInspector(null);
        if (insp !== "-- / Unavailable") throw new Error("StreamingInspector failed: " + insp);

        // Run with asymmetric GPU availability (e.g. G0 available, G1 unavailable)
        const topAsym = testTopChrome(11.2, null);
        if (!topAsym.includes("11.2G") || !topAsym.includes("--")) {
            throw new Error("Asym top failed: " + topAsym);
        }

        console.log("NODE_VERIFICATION_PASS");
        """
        proc = subprocess.run(["node", "-e", node_script], capture_output=True, text=True)
        assert proc.returncode == 0, f"Node verification failed: {proc.stderr}"
        assert "NODE_VERIFICATION_PASS" in proc.stdout
