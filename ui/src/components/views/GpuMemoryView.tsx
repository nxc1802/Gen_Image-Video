"use client";

import React, { useState } from "react";
import { Trash, ArrowsClockwise, Cpu } from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";

export function GpuMemoryView() {
  const {
    lastMemory,
    loadedModels,
    refreshHealthAndMemory,
    handleFreeAllVram,
  } = useStudio();

  const [busy, setBusy] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const onFreeVramClick = async () => {
    setBusy(true);
    setStatusMessage("Unloading active models and running torch.cuda.empty_cache()...");
    try {
      const msg = await handleFreeAllVram();
      setStatusMessage(msg);
    } catch (err) {
      setStatusMessage(
        err instanceof Error ? err.message : "Failed to free VRAM"
      );
    } finally {
      setBusy(false);
    }
  };

  const gpus = lastMemory?.gpus || [];
  const ram = lastMemory?.ram;

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="bg-surface border border-hairline rounded-[10px] p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <span className="text-[10px] font-mono uppercase tracking-widest text-ink-muted">
            GET /v1/memory &middot; POST /v1/memory/free
          </span>
          <h2 className="font-serif text-2xl text-ink mt-0.5">
            Dual NVIDIA T4 VRAM &amp; System RAM Telemetry
          </h2>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => refreshHealthAndMemory(false)}
            className="px-3 py-2 rounded-[6px] border border-hairline bg-surface hover:bg-subtle text-xs font-mono text-ink flex items-center gap-1.5"
          >
            <ArrowsClockwise weight="bold" className="w-3.5 h-3.5" />
            <span>Poll Telemetry</span>
          </button>

          <button
            type="button"
            disabled={busy}
            onClick={onFreeVramClick}
            className="px-4 py-2 rounded-[6px] bg-ink text-surface hover:bg-[#2a2a2a] disabled:opacity-50 text-xs font-mono flex items-center gap-1.5 transition-transform active:scale-[0.98]"
          >
            <Trash weight="bold" className="w-3.5 h-3.5" />
            <span>{busy ? "Reclaiming VRAM..." : "Free All VRAM"}</span>
          </button>
        </div>
      </div>

      {statusMessage && (
        <div className="p-3.5 rounded-[8px] border border-hairline bg-subtle text-xs font-mono text-ink-secondary">
          {statusMessage}
        </div>
      )}

      {/* GPU Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {gpus.length > 0 ? (
          gpus.map((gpu, idx) => {
            const id = gpu.gpu_id ?? gpu.index ?? idx;
            const alloc = gpu.allocated_gb ?? 0;
            const reserved = gpu.reserved_gb ?? 0;
            const total = gpu.total_gb ?? 15.0;
            const free = gpu.free_gb ?? Math.max(0, total - alloc);
            const pct = Math.min(100, Math.round((alloc / total) * 100));

            return (
              <div
                key={id}
                className="bg-surface border border-hairline rounded-[10px] p-5 space-y-4"
              >
                <div className="flex items-center justify-between border-b border-hairline pb-3">
                  <div className="flex items-center gap-2">
                    <Cpu weight="bold" className="w-4 h-4 text-ink" />
                    <span className="font-serif text-xl text-ink">
                      GPU {id}: {gpu.name || "NVIDIA Tesla T4"}
                    </span>
                  </div>
                  <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-spot-blueBg text-spot-blueText">
                    {pct}% Allocated
                  </span>
                </div>

                {/* Progress Bar */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-ink font-medium">
                      {alloc.toFixed(2)} GB / {total.toFixed(2)} GB
                    </span>
                    <span className="text-ink-muted">
                      {free.toFixed(2)} GB Free
                    </span>
                  </div>
                  <div className="w-full h-2.5 bg-muted rounded-[3px] overflow-hidden">
                    <div
                      className="h-full bg-ink transition-all duration-300"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-2 pt-2 text-xs font-mono">
                  <div className="p-2.5 rounded-[6px] bg-canvas border border-hairline">
                    <div className="text-[10px] text-ink-muted uppercase">
                      Allocated
                    </div>
                    <div className="text-ink font-semibold mt-0.5">
                      {alloc.toFixed(2)} GB
                    </div>
                  </div>
                  <div className="p-2.5 rounded-[6px] bg-canvas border border-hairline">
                    <div className="text-[10px] text-ink-muted uppercase">
                      Reserved
                    </div>
                    <div className="text-ink font-semibold mt-0.5">
                      {reserved.toFixed(2)} GB
                    </div>
                  </div>
                  <div className="p-2.5 rounded-[6px] bg-canvas border border-hairline">
                    <div className="text-[10px] text-ink-muted uppercase">
                      Total Capacity
                    </div>
                    <div className="text-ink font-semibold mt-0.5">
                      {total.toFixed(2)} GB
                    </div>
                  </div>
                </div>
              </div>
            );
          })
        ) : (
          <div className="md:col-span-2 bg-surface border border-hairline rounded-[10px] p-8 text-center text-xs font-mono text-ink-muted">
            No live GPU telemetry received yet. Connect to the Kaggle gateway to
            inspect dual T4 VRAM allocation.
          </div>
        )}
      </div>

      {/* System RAM & Active Loaded Models */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="bg-surface border border-hairline rounded-[10px] p-5 space-y-3">
          <h3 className="font-serif text-xl text-ink border-b border-hairline pb-2">
            Host System RAM
          </h3>
          {ram ? (
            <div className="space-y-2 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-ink-muted">Used / Total:</span>
                <span className="text-ink font-semibold">
                  {(ram.used_gb ?? 0).toFixed(2)} GB /{" "}
                  {(ram.total_gb ?? 30).toFixed(2)} GB ({ram.percent ?? 0}%)
                </span>
              </div>
              <div className="w-full h-2 bg-muted rounded-[3px] overflow-hidden">
                <div
                  className="h-full bg-ink"
                  style={{ width: `${Math.min(100, ram.percent ?? 0)}%` }}
                />
              </div>
            </div>
          ) : (
            <div className="text-xs font-mono text-ink-muted">
              Host RAM stats unavailable.
            </div>
          )}
        </div>

        <div className="bg-surface border border-hairline rounded-[10px] p-5 space-y-3">
          <h3 className="font-serif text-xl text-ink border-b border-hairline pb-2">
            Currently Resident Models in VRAM
          </h3>
          {loadedModels.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {loadedModels.map((modelName) => (
                <span
                  key={modelName}
                  className="px-2.5 py-1 rounded-full text-xs font-mono bg-spot-greenBg text-spot-greenText"
                >
                  {modelName}
                </span>
              ))}
            </div>
          ) : (
            <div className="text-xs font-mono text-ink-muted">
              No models currently resident in VRAM (Idle).
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
