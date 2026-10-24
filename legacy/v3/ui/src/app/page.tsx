"use client";

import React, { useEffect } from "react";
import { StudioProvider, useStudio } from "@/context/StudioContext";
import { TopChromeBar } from "@/components/layout/TopChromeBar";
import { SidebarNav } from "@/components/layout/SidebarNav";
import { ConnectionModal } from "@/components/layout/ConnectionModal";
import { ShutdownModal } from "@/components/layout/ShutdownModal";
import { ChatView } from "@/components/views/ChatView";
import { VisionView } from "@/components/views/VisionView";
import { ImageView } from "@/components/views/ImageView";
import { VideoView } from "@/components/views/VideoView";
import { AudioView } from "@/components/views/AudioView";
import { ModelsView } from "@/components/views/ModelsView";
import { GpuMemoryView } from "@/components/views/GpuMemoryView";
import { ConfigAdminView } from "@/components/views/ConfigAdminView";
import type { StudioViewId } from "@/types/studio";

const SHORTCUT_MAP: Record<string, StudioViewId> = {
  "1": "chat",
  "2": "vision",
  "3": "image",
  "4": "video",
  "5": "audio",
  "6": "models",
  "7": "memory",
  "8": "config",
};

function StudioWorkspace() {
  const {
    activeView,
    setActiveView,
    setIsConnectionModalOpen,
  } = useStudio();

  // Global keyboard shortcuts (1-8 to switch views when not typing in an input, Cmd/Ctrl+K for Connection modal)
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
          {activeView === "chat" && <ChatView />}
          {activeView === "vision" && <VisionView />}
          {activeView === "image" && <ImageView />}
          {activeView === "video" && <VideoView />}
          {activeView === "audio" && <AudioView />}
          {activeView === "models" && <ModelsView />}
          {activeView === "memory" && <GpuMemoryView />}
          {activeView === "config" && <ConfigAdminView />}
        </main>
      </div>

      <ConnectionModal />
      <ShutdownModal />
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
