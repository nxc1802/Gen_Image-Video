"use client";

import React from "react";
import { ArrowsClockwise, PlugsConnected, ShieldCheck } from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";

export function TopChromeBar() {
  const {
    gatewayUrl,
    connectionStatus,
    telemetry,
    refreshManifests,
    refreshJobs,
    setIsConnectionModalOpen,
  } = useStudio();

  const gpu0Used = telemetry?.gpu_0_vram_used_gb ?? 0;
  const gpu1Used = telemetry?.gpu_1_vram_used_gb ?? 0;
  const queueDepth = telemetry?.queue_depth ?? 0;
  const activeJobs = telemetry?.active_jobs ?? 0;

  const statusBadge = () => {
    if (connectionStatus === "online") {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-spot-greenBg text-spot-greenText border border-[#C2DEC4]">
          <span className="w-1.5 h-1.5 rounded-full bg-spot-greenText" />
          ONLINE
        </span>
      );
    }
    if (connectionStatus === "connecting") {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-amber-50 text-amber-800 border border-amber-300">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
          CONNECTING
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-spot-redBg text-spot-redText border border-[#E6C5C5]">
        <span className="w-1.5 h-1.5 rounded-full bg-spot-redText" />
        OFFLINE
      </span>
    );
  };

  return (
    <header className="h-12 bg-surface border-b border-hairline px-5 flex items-center justify-between shrink-0 z-20">
      {/* Left: Window Dots & Brand */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1.5" aria-hidden="true">
          <span className="w-2.5 h-2.5 rounded-full bg-[#E5E5E0] border border-[#D8D8D2]" />
          <span className="w-2.5 h-2.5 rounded-full bg-[#E5E5E0] border border-[#D8D8D2]" />
          <span className="w-2.5 h-2.5 rounded-full bg-[#E5E5E0] border border-[#D8D8D2]" />
        </div>
        <div className="h-4 w-[1px] bg-hairline" />
        <div className="flex items-baseline gap-2.5">
          <span className="font-serif text-lg tracking-tight text-ink">
            Factory v4
          </span>
          <span className="font-mono text-[11px] text-ink-muted uppercase tracking-wider">
            Operations Console
          </span>
        </div>
      </div>

      {/* Center: Live Telemetry Indicators */}
      <div className="hidden md:flex items-center gap-3 font-mono text-xs">
        {statusBadge()}

        <div className="px-2.5 py-1 rounded-[6px] bg-subtle border border-hairline text-ink-secondary flex items-center gap-2">
          <span className="text-ink-muted">VRAM:</span>
          <span>
            G0: {gpu0Used.toFixed(1)}G · G1: {gpu1Used.toFixed(1)}G
          </span>
        </div>

        <div className="px-2.5 py-1 rounded-[6px] bg-subtle border border-hairline text-ink-secondary flex items-center gap-2">
          <span className="text-ink-muted">QUEUE:</span>
          <span>{queueDepth} pending</span>
        </div>

        <div className="px-2.5 py-1 rounded-[6px] bg-subtle border border-hairline text-ink-secondary flex items-center gap-2">
          <span className="text-ink-muted">RUNNING:</span>
          <span>{activeJobs}</span>
        </div>

        <div className="hidden lg:flex items-center gap-1 text-[11px] text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
          <ShieldCheck weight="bold" className="w-3.5 h-3.5" />
          <span>Bearer Auth</span>
        </div>
      </div>

      {/* Right: Controls */}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => {
            refreshManifests(false);
            refreshJobs();
          }}
          title="Sync manifests and queue state"
          className="px-2.5 py-1.5 rounded-[6px] border border-hairline bg-surface hover:bg-subtle text-ink text-xs font-mono flex items-center gap-1.5 transition-transform active:scale-[0.98]"
        >
          <ArrowsClockwise weight="bold" className="w-3.5 h-3.5" />
          <span>Sync</span>
        </button>

        <button
          type="button"
          onClick={() => setIsConnectionModalOpen(true)}
          className="px-3 py-1.5 rounded-[6px] bg-ink text-surface hover:bg-[#2a2a2a] text-xs font-medium flex items-center gap-1.5 transition-transform active:scale-[0.98]"
        >
          <PlugsConnected weight="bold" className="w-3.5 h-3.5" />
          <span className="truncate max-w-[140px]">
            {gatewayUrl ? "Connected" : "Configure"}
          </span>
          <kbd className="ml-1 px-1 py-0.2 text-[10px] font-mono bg-[#2F3437] text-[#EAEAEA] rounded-[3px] border border-[#444]">
            ⌘K
          </kbd>
        </button>
      </div>
    </header>
  );
}
