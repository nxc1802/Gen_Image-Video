"use client";

import React, { useMemo } from "react";
import {
  Cpu,
  Stack,
  Lightning,
  Cube,
  ArrowsClockwise,
  CheckCircle,
  WarningCircle,
  Clock,
  Terminal,
  ArrowRight,
} from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";

export function OperationsConsoleView() {
  const {
    telemetry,
    jobs,
    models,
    services,
    connectionStatus,
    refreshManifests,
    refreshJobs,
    setActiveView,
  } = useStudio();

  // Metrics
  const gpu0Used = telemetry?.gpu_0_vram_used_gb ?? 0;
  const gpu0Total = telemetry?.gpu_0_vram_total_gb ?? 15.0;
  const gpu0Pct = Math.min(100, Math.round((gpu0Used / gpu0Total) * 100));

  const gpu1Used = telemetry?.gpu_1_vram_used_gb ?? 0;
  const gpu1Total = telemetry?.gpu_1_vram_total_gb ?? 15.0;
  const gpu1Pct = Math.min(100, Math.round((gpu1Used / gpu1Total) * 100));

  const queueDepth = telemetry?.queue_depth ?? jobs.filter((j) => j.state === "QUEUED").length;
  const activeJobsCount =
    telemetry?.active_jobs ??
    jobs.filter((j) => ["ADMITTED", "RUNNING", "STREAMING"].includes(j.state)).length;

  const getVramColor = (pct: number) => {
    if (pct >= 90) return "text-spot-redText bg-spot-redBg border-[#E6C5C5]";
    if (pct >= 70) return "text-amber-700 bg-amber-50 border-amber-200";
    return "text-spot-greenText bg-spot-greenBg border-[#C2DEC4]";
  };

  const getVramBarColor = (pct: number) => {
    if (pct >= 90) return "bg-red-500";
    if (pct >= 70) return "bg-amber-500";
    return "bg-emerald-500";
  };

  const recentJobs = useMemo(() => {
    return [...jobs]
      .sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""))
      .slice(0, 5);
  }, [jobs]);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 rounded-[10px] bg-surface border border-hairline">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-serif text-2xl text-ink">
              Operations Console
            </h1>
            <span
              className={`px-2 py-0.5 rounded-full text-[10px] font-mono uppercase tracking-wider ${
                connectionStatus === "online"
                  ? "bg-spot-greenBg text-spot-greenText"
                  : "bg-spot-redBg text-spot-redText"
              }`}
            >
              {connectionStatus}
            </span>
          </div>
          <p className="text-xs text-ink-muted mt-1">
            Real-time telemetry, NVIDIA Tesla T4 dual-VRAM monitors, Two-Tier Queue depth, and active worker leases.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => {
              refreshManifests(false);
              refreshJobs();
            }}
            className="px-3 py-1.5 rounded-[6px] border border-hairline bg-surface hover:bg-subtle text-ink text-xs font-mono flex items-center gap-1.5 transition-transform active:scale-[0.98]"
          >
            <ArrowsClockwise weight="bold" className="w-3.5 h-3.5" />
            <span>Sync Telemetry</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveView("explorer")}
            className="px-3 py-1.5 rounded-[6px] bg-ink text-surface hover:bg-[#2a2a2a] text-xs font-medium flex items-center gap-1.5 transition-transform active:scale-[0.98]"
          >
            <span>Open API Explorer</span>
            <ArrowRight weight="bold" className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Primary KPI Grid: Dual T4 Gauges + Queue + Active Jobs */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* GPU 0 VRAM Gauge */}
        <div className="p-4 rounded-[10px] bg-surface border border-hairline space-y-3">
          <div className="flex items-center justify-between text-xs font-mono text-ink-muted">
            <span className="flex items-center gap-1.5 text-ink font-semibold">
              <Cpu weight="bold" className="w-4 h-4" />
              <span>GPU 0 (Tesla T4)</span>
            </span>
            <span className={`px-2 py-0.5 rounded-[4px] border text-[10px] ${getVramColor(gpu0Pct)}`}>
              {gpu0Pct}% Used
            </span>
          </div>
          <div>
            <div className="text-2xl font-mono font-bold text-ink">
              {gpu0Used.toFixed(1)} <span className="text-xs font-normal text-ink-muted">/ {gpu0Total.toFixed(1)} GB</span>
            </div>
            <div className="w-full h-2 rounded-full bg-subtle overflow-hidden mt-2 border border-hairline">
              <div
                className={`h-full transition-all duration-500 ${getVramBarColor(gpu0Pct)}`}
                style={{ width: `${gpu0Pct}%` }}
              />
            </div>
          </div>
          <div className="text-[11px] font-mono text-ink-muted flex justify-between">
            <span>Free: {(gpu0Total - gpu0Used).toFixed(1)} GB</span>
            <span>Target: Image &amp; Vision</span>
          </div>
        </div>

        {/* GPU 1 VRAM Gauge */}
        <div className="p-4 rounded-[10px] bg-surface border border-hairline space-y-3">
          <div className="flex items-center justify-between text-xs font-mono text-ink-muted">
            <span className="flex items-center gap-1.5 text-ink font-semibold">
              <Cpu weight="bold" className="w-4 h-4" />
              <span>GPU 1 (Tesla T4)</span>
            </span>
            <span className={`px-2 py-0.5 rounded-[4px] border text-[10px] ${getVramColor(gpu1Pct)}`}>
              {gpu1Pct}% Used
            </span>
          </div>
          <div>
            <div className="text-2xl font-mono font-bold text-ink">
              {gpu1Used.toFixed(1)} <span className="text-xs font-normal text-ink-muted">/ {gpu1Total.toFixed(1)} GB</span>
            </div>
            <div className="w-full h-2 rounded-full bg-subtle overflow-hidden mt-2 border border-hairline">
              <div
                className={`h-full transition-all duration-500 ${getVramBarColor(gpu1Pct)}`}
                style={{ width: `${gpu1Pct}%` }}
              />
            </div>
          </div>
          <div className="text-[11px] font-mono text-ink-muted flex justify-between">
            <span>Free: {(gpu1Total - gpu1Used).toFixed(1)} GB</span>
            <span>Target: Video &amp; Text</span>
          </div>
        </div>

        {/* Two-Tier Queue Depth */}
        <div className="p-4 rounded-[10px] bg-surface border border-hairline space-y-3">
          <div className="flex items-center justify-between text-xs font-mono text-ink-muted">
            <span className="flex items-center gap-1.5 text-ink font-semibold">
              <Stack weight="bold" className="w-4 h-4" />
              <span>Queue Depth</span>
            </span>
            <span className="text-[10px] text-ink-muted font-mono">Tier-2 Queue</span>
          </div>
          <div>
            <div className="text-2xl font-mono font-bold text-ink">
              {queueDepth} <span className="text-xs font-normal text-ink-muted">jobs waiting</span>
            </div>
            <div className="text-xs text-ink-secondary mt-1">
              Admission status: <span className="text-spot-greenText font-semibold">HEALTHY</span>
            </div>
          </div>
          <div className="text-[11px] font-mono text-ink-muted flex justify-between">
            <span>Engine: Redis + Postgres</span>
            <button
              onClick={() => setActiveView("jobs")}
              className="text-ink hover:underline font-medium"
            >
              Inspect Queue →
            </button>
          </div>
        </div>

        {/* Active Worker Leases */}
        <div className="p-4 rounded-[10px] bg-surface border border-hairline space-y-3">
          <div className="flex items-center justify-between text-xs font-mono text-ink-muted">
            <span className="flex items-center gap-1.5 text-ink font-semibold">
              <Lightning weight="bold" className="w-4 h-4 text-amber-500" />
              <span>Active Leases</span>
            </span>
            <span className="text-[10px] text-ink-muted font-mono">Workers</span>
          </div>
          <div>
            <div className="text-2xl font-mono font-bold text-ink">
              {activeJobsCount} <span className="text-xs font-normal text-ink-muted">jobs running</span>
            </div>
            <div className="text-xs text-ink-secondary mt-1">
              Leases assigned: <span className="font-semibold text-ink">{telemetry?.leases?.length ?? activeJobsCount}</span>
            </div>
          </div>
          <div className="text-[11px] font-mono text-ink-muted flex justify-between">
            <span>Lease TTL: 300s</span>
            <button
              onClick={() => setActiveView("inspector")}
              className="text-ink hover:underline font-medium"
            >
              Live Inspector →
            </button>
          </div>
        </div>
      </div>

      {/* Middle Row: Subsystem Health Grid & Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Registered Subsystems Inventory */}
        <div className="lg:col-span-2 p-5 rounded-[10px] bg-surface border border-hairline space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-hairline">
            <div className="flex items-center gap-2">
              <Cube weight="bold" className="w-4 h-4 text-ink" />
              <h2 className="text-sm font-mono font-semibold text-ink">
                Registered Factory Subsystems ({services.length})
              </h2>
            </div>
            <button
              onClick={() => setActiveView("registry")}
              className="text-xs font-mono text-ink hover:underline"
            >
              View Full Registry →
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {services.map((svc) => (
              <div
                key={svc.id}
                className="p-3 rounded-[8px] bg-canvas border border-hairline flex items-center justify-between"
              >
                <div>
                  <div className="text-xs font-mono font-semibold text-ink">
                    {svc.name}
                  </div>
                  <div className="text-[11px] font-mono text-ink-muted">
                    {svc.network?.static_domain || "internal"}
                  </div>
                </div>
                <div className="text-right">
                  <span className="inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-full bg-spot-greenBg text-spot-greenText">
                    <span className="w-1.5 h-1.5 rounded-full bg-spot-greenText" />
                    ONLINE
                  </span>
                  <div className="text-[10px] font-mono text-ink-muted mt-0.5">
                    {svc.framework || "vLLM / ComfyUI"}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Recent Jobs Feed */}
        <div className="p-5 rounded-[10px] bg-surface border border-hairline space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-hairline">
            <div className="flex items-center gap-2">
              <Clock weight="bold" className="w-4 h-4 text-ink" />
              <h2 className="text-sm font-mono font-semibold text-ink">
                Recent Queue Jobs
              </h2>
            </div>
            <button
              onClick={() => setActiveView("jobs")}
              className="text-xs font-mono text-ink hover:underline"
            >
              All Jobs →
            </button>
          </div>

          {recentJobs.length === 0 ? (
            <div className="p-8 text-center text-xs font-mono text-ink-muted">
              No recent jobs. Launch a task in the API Explorer.
            </div>
          ) : (
            <div className="space-y-2.5">
              {recentJobs.map((job) => (
                <div
                  key={job.job_id}
                  onClick={() => setActiveView("jobs")}
                  className="p-2.5 rounded-[6px] bg-canvas border border-hairline hover:border-ink/20 cursor-pointer transition-colors flex items-center justify-between"
                >
                  <div className="space-y-0.5">
                    <div className="text-xs font-mono font-medium text-ink flex items-center gap-1.5">
                      <span>{job.job_id.slice(0, 12)}</span>
                      <span className="text-[10px] text-ink-muted">
                        ({job.service}/{job.task})
                      </span>
                    </div>
                    <div className="text-[10px] font-mono text-ink-muted">
                      {job.created_at ? new Date(job.created_at).toLocaleTimeString() : "--:--:--"}
                    </div>
                  </div>
                  <div>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                        job.state === "SUCCEEDED"
                          ? "bg-spot-greenBg text-spot-greenText"
                          : job.state === "FAILED"
                          ? "bg-spot-redBg text-spot-redText"
                          : job.state === "RUNNING" || job.state === "STREAMING"
                          ? "bg-blue-100 text-blue-800"
                          : "bg-subtle text-ink-secondary"
                      }`}
                    >
                      {job.state}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Raw SSE Telemetry Log Stream */}
      <div className="p-4 rounded-[10px] bg-[#141517] border border-[#2B2D31] text-[#E0E2E6] space-y-2">
        <div className="flex items-center justify-between text-xs font-mono text-[#909296]">
          <span className="flex items-center gap-1.5 text-white font-medium">
            <Terminal weight="bold" className="w-4 h-4 text-emerald-400" />
            <span>Telemetry Stream (/api/studio/inspect/stream)</span>
          </span>
          <span className="text-[10px]">
            Keepalive: {telemetry?.timestamp ? new Date(telemetry.timestamp * 1000).toLocaleTimeString() : "Active"}
          </span>
        </div>
        <div className="p-2.5 rounded-[6px] bg-[#111214] font-mono text-xs text-[#A6A7AD] overflow-x-auto">
          <code>
            {telemetry?.raw_message ||
              JSON.stringify(
                {
                  gpu_0_vram_used_gb: gpu0Used,
                  gpu_1_vram_used_gb: gpu1Used,
                  active_jobs: activeJobsCount,
                  queue_depth: queueDepth,
                  status: "connected",
                },
                null,
                2
              )}
          </code>
        </div>
      </div>
    </div>
  );
}
