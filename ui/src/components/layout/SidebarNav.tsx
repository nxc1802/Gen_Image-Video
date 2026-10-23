"use client";

import React from "react";
import {
  ChatTeardropText,
  Eye,
  PaintBrushBroad,
  FilmStrip,
  Waveform,
  Cube,
  Cpu,
  SlidersHorizontal,
  Trash,
  Power,
} from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";
import type { StudioViewId } from "@/types/studio";

interface NavItem {
  id: StudioViewId;
  label: string;
  shortcut: string;
  icon: React.ComponentType<{ weight?: "bold" | "regular" | "fill"; className?: string }>;
}

const PLAYGROUND_NAV: NavItem[] = [
  { id: "chat", label: "Chat Playground", shortcut: "1", icon: ChatTeardropText },
  { id: "vision", label: "Vision Studio", shortcut: "2", icon: Eye },
  { id: "image", label: "Image Studio", shortcut: "3", icon: PaintBrushBroad },
  { id: "video", label: "Video Studio", shortcut: "4", icon: FilmStrip },
  { id: "audio", label: "Audio Studio", shortcut: "5", icon: Waveform },
];

const SYSTEM_NAV: NavItem[] = [
  { id: "models", label: "Models & Health", shortcut: "6", icon: Cube },
  { id: "memory", label: "GPU Memory", shortcut: "7", icon: Cpu },
  { id: "config", label: "Config & Admin", shortcut: "8", icon: SlidersHorizontal },
];

export function SidebarNav() {
  const { activeView, setActiveView, handleFreeAllVram, setIsShutdownModalOpen } =
    useStudio();

  const renderNavButton = (item: NavItem) => {
    const Icon = item.icon;
    const isActive = activeView === item.id;
    return (
      <button
        key={item.id}
        type="button"
        onClick={() => setActiveView(item.id)}
        className={`w-full px-2.5 py-2 rounded-[6px] text-left text-xs flex items-center justify-between transition-colors ${
          isActive
            ? "bg-muted text-ink font-semibold"
            : "text-ink-secondary hover:bg-subtle hover:text-ink"
        }`}
      >
        <span className="flex items-center gap-2.5">
          <Icon weight="bold" className="w-4 h-4 shrink-0" />
          <span>{item.label}</span>
        </span>
        <kbd className="text-[10px] font-mono px-1.5 py-0.5 rounded-[4px] border border-hairline bg-canvas text-ink-muted">
          {item.shortcut}
        </kbd>
      </button>
    );
  };

  return (
    <aside className="w-[228px] bg-surface border-r border-hairline flex flex-col justify-between shrink-0 select-none">
      <div className="p-3 space-y-5 overflow-y-auto">
        <div>
          <div className="px-2.5 pb-1.5 text-[10px] font-mono uppercase tracking-[0.08em] text-ink-muted">
            Playgrounds
          </div>
          <nav className="space-y-0.5">{PLAYGROUND_NAV.map(renderNavButton)}</nav>
        </div>

        <div>
          <div className="px-2.5 pb-1.5 text-[10px] font-mono uppercase tracking-[0.08em] text-ink-muted">
            System &amp; Telemetry
          </div>
          <nav className="space-y-0.5">{SYSTEM_NAV.map(renderNavButton)}</nav>
        </div>
      </div>

      {/* Bottom Quick Actions */}
      <div className="p-3 border-t border-hairline space-y-2 bg-subtle">
        <div className="flex items-center justify-between text-[11px] font-mono text-ink-muted px-1">
          <span>Quick Hardware Ops</span>
          <span>2x T4</span>
        </div>
        <button
          type="button"
          onClick={() => handleFreeAllVram()}
          className="w-full px-2.5 py-1.5 rounded-[6px] border border-hairline bg-surface hover:bg-canvas text-ink text-xs font-mono flex items-center justify-between transition-transform active:scale-[0.98]"
        >
          <span className="flex items-center gap-1.5">
            <Trash weight="bold" className="w-3.5 h-3.5" />
            <span>Free All VRAM</span>
          </span>
          <span className="text-[10px] text-ink-muted">GC</span>
        </button>
        <button
          type="button"
          onClick={() => setIsShutdownModalOpen(true)}
          className="w-full px-2.5 py-1.5 rounded-[6px] border border-[#E6C5C5] bg-spot-redBg text-spot-redText text-xs font-mono flex items-center justify-between transition-transform active:scale-[0.98]"
        >
          <span className="flex items-center gap-1.5">
            <Power weight="bold" className="w-3.5 h-3.5" />
            <span>Shutdown Kernel</span>
          </span>
          <span className="text-[10px]">SIGKILL</span>
        </button>
      </div>
    </aside>
  );
}
