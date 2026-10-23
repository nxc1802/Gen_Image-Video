"use client";

import React from "react";
import { ArrowsClockwise, PlugsConnected } from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";

export function TopChromeBar() {
  const {
    gatewayUrl,
    connectionStatus,
    activeModalities,
    loadedModels,
    vramSummary,
    refreshHealthAndMemory,
    setIsConnectionModalOpen,
  } = useStudio();

  const statusBadge = () => {
    if (connectionStatus === "online") {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-spot-greenBg text-spot-greenText">
          <span className="w-1.5 h-1.5 rounded-full bg-spot-greenText" />
          ONLINE ({activeModalities.join(",") || "ALL"})
        </span>
      );
    }
    if (connectionStatus === "offline") {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-spot-redBg text-spot-redText">
          <span className="w-1.5 h-1.5 rounded-full bg-spot-redText" />
          OFFLINE
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-spot-yellowBg text-spot-yellowText">
        <span className="w-1.5 h-1.5 rounded-full bg-spot-yellowText" />
        UNCONFIGURED
      </span>
    );
  };

  return (
    <header className="h-12 bg-surface border-b border-hairline px-5 flex items-center justify-between shrink-0 z-20">
      {/* Left: Faux-OS Window Dots & Brand */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1.5" aria-hidden="true">
          <span className="w-2.5 h-2.5 rounded-full bg-[#E5E5E0] border border-[#D8D8D2]" />
          <span className="w-2.5 h-2.5 rounded-full bg-[#E5E5E0] border border-[#D8D8D2]" />
          <span className="w-2.5 h-2.5 rounded-full bg-[#E5E5E0] border border-[#D8D8D2]" />
        </div>
        <div className="h-4 w-[1px] bg-hairline" />
        <div className="flex items-baseline gap-2.5">
          <span className="font-serif text-xl tracking-tight text-ink">
            Multimodal AI Studio
          </span>
          <span className="font-mono text-[11px] text-ink-muted uppercase tracking-wider">
            2x T4 Factory v2.0
          </span>
        </div>
      </div>

      {/* Center: Telemetry Pill Strip */}
      <div className="hidden md:flex items-center gap-3 font-mono text-xs">
        {statusBadge()}
        <div className="px-2.5 py-1 rounded-[6px] bg-subtle border border-hairline text-ink-secondary flex items-center gap-2">
          <span className="text-ink-muted">VRAM:</span>
          <span>{vramSummary}</span>
        </div>
        <div className="px-2.5 py-1 rounded-[6px] bg-subtle border border-hairline text-ink-secondary flex items-center gap-2">
          <span className="text-ink-muted">LOADED:</span>
          <span className="truncate max-w-[180px]">
            {loadedModels.length > 0 ? loadedModels.join(", ") : "idle"}
          </span>
        </div>
      </div>

      {/* Right: Connection Controls */}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => refreshHealthAndMemory(false)}
          title="Poll /health & /v1/memory"
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
          <span>{gatewayUrl ? "Tunnel Connected" : "Connect Tunnel"}</span>
          <kbd className="ml-1 px-1 py-0.2 text-[10px] font-mono bg-[#2F3437] text-[#EAEAEA] rounded-[3px] border border-[#444]">
            K
          </kbd>
        </button>
      </div>
    </header>
  );
}
