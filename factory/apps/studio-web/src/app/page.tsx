"use client";

import React, { useEffect } from "react";
import { StudioProvider, useStudio } from "@/context/StudioContext";
import { TopChromeBar } from "@/components/layout/TopChromeBar";
import { SidebarNav } from "@/components/layout/SidebarNav";
import { ConnectionModal } from "@/components/layout/ConnectionModal";
import { OperationsConsoleView } from "@/components/views/OperationsConsoleView";
import { ApiExplorerView } from "@/components/views/ApiExplorerView";
import { JobDashboardView } from "@/components/views/JobDashboardView";
import { StreamingInspectorView } from "@/components/views/StreamingInspectorView";
import { ModelRegistryView } from "@/components/views/ModelRegistryView";
import type { StudioViewId } from "@/types/studio";

const SHORTCUT_MAP: Record<string, StudioViewId> = {
  "1": "console",
  "2": "explorer",
  "3": "jobs",
  "4": "inspector",
  "5": "registry",
};

function StudioWorkspace() {
  const { activeView, setActiveView, setIsConnectionModalOpen } = useStudio();

  // Global keyboard shortcuts (1-5 to switch views, Cmd/Ctrl+K for connection settings)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setIsConnectionModalOpen(true);
        return;
      }

      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      const mapped = SHORTCUT_MAP[e.key];
      if (mapped && !e.metaKey && !e.ctrlKey && !e.altKey) {
        setActiveView(mapped);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [setActiveView, setIsConnectionModalOpen]);

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-canvas">
      <TopChromeBar />

      <div className="flex-1 flex overflow-hidden">
        <SidebarNav />

        <main className="flex-1 overflow-y-auto p-5">
          {activeView === "console" && <OperationsConsoleView />}
          {activeView === "explorer" && <ApiExplorerView />}
          {activeView === "jobs" && <JobDashboardView />}
          {activeView === "inspector" && <StreamingInspectorView />}
          {activeView === "registry" && <ModelRegistryView />}
        </main>
      </div>

      <ConnectionModal />
    </div>
  );
}

export default function HomePage() {
  return (
    <StudioProvider>
      <StudioWorkspace />
    </StudioProvider>
  );
}
