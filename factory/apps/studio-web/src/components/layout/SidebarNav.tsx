"use client";

import React from "react";
import {
  Gauge,
  Compass,
  Stack,
  Broadcast,
  Cube,
  PlugsConnected,
  Terminal,
} from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";
import type { StudioViewId } from "@/types/studio";

interface NavItem {
  id: StudioViewId;
  label: string;
  shortcut: string;
  icon: React.ComponentType<{ weight?: "bold" | "regular" | "fill"; className?: string }>;
}

const CORE_NAV_ITEMS: NavItem[] = [
  { id: "console", label: "Operations Console", shortcut: "1", icon: Gauge },
  { id: "explorer", label: "API Explorer", shortcut: "2", icon: Compass },
  { id: "jobs", label: "Job Dashboard", shortcut: "3", icon: Stack },
  { id: "inspector", label: "Streaming Inspector", shortcut: "4", icon: Broadcast },
  { id: "registry", label: "Model Registry", shortcut: "5", icon: Cube },
];

export function SidebarNav() {
  const { activeView, setActiveView, setIsConnectionModalOpen, jobs, connectionStatus } =
    useStudio();

  const queuedOrRunningCount = jobs.filter((j) =>
    ["QUEUED", "ADMITTED", "RUNNING", "STREAMING"].includes(j.state)
  ).length;

  return (
    <aside className="w-[240px] bg-surface border-r border-hairline flex flex-col justify-between shrink-0 select-none">
      <div className="p-3 space-y-5 overflow-y-auto">
        <div>
          <div className="px-2.5 pb-2 text-[10px] font-mono uppercase tracking-[0.08em] text-ink-muted">
            Operations &amp; Control
          </div>
          <nav className="space-y-1">
            {CORE_NAV_ITEMS.map((item) => {
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
                  <div className="flex items-center gap-1.5">
                    {item.id === "jobs" && queuedOrRunningCount > 0 && (
                      <span className="px-1.5 py-0.2 rounded-full text-[10px] font-mono font-bold bg-ink text-surface">
                        {queuedOrRunningCount}
                      </span>
                    )}
                    <kbd className="text-[10px] font-mono px-1.5 py-0.5 rounded-[4px] border border-hairline bg-canvas text-ink-muted">
                      {item.shortcut}
                    </kbd>
                  </div>
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Bottom Status & Settings */}
      <div className="p-3 border-t border-hairline space-y-2 bg-subtle">
        <div className="flex items-center justify-between text-[11px] font-mono text-ink-muted px-1">
          <span className="flex items-center gap-1">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                connectionStatus === "online"
                  ? "bg-spot-greenText"
                  : "bg-spot-redText"
              }`}
            />
            <span className="capitalize">{connectionStatus}</span>
          </span>
          <span>Dual T4</span>
        </div>

        <button
          type="button"
          onClick={() => setIsConnectionModalOpen(true)}
          className="w-full px-2.5 py-1.5 rounded-[6px] border border-hairline bg-surface hover:bg-canvas text-ink text-xs font-mono flex items-center justify-between transition-transform active:scale-[0.98]"
        >
          <span className="flex items-center gap-1.5">
            <PlugsConnected weight="bold" className="w-3.5 h-3.5" />
            <span>Connection</span>
          </span>
          <kbd className="text-[10px] font-mono text-ink-muted">⌘K</kbd>
        </button>
      </div>
    </aside>
  );
}
