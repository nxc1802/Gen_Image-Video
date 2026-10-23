"use client";

import React from "react";
import { SpinnerGap } from "@phosphor-icons/react";
import type { SseProgressState } from "@/types/studio";

interface ProgressStripProps {
  progress: SseProgressState;
}

export function ProgressStrip({ progress }: ProgressStripProps) {
  if (!progress.active) return null;

  return (
    <div className="border border-hairline rounded-[6px] bg-subtle p-3">
      <div className="flex items-center justify-between text-[11px] font-mono mb-1.5">
        <span className="text-ink font-medium flex items-center gap-1.5">
          <SpinnerGap weight="bold" className="w-3.5 h-3.5 animate-spin" />
          <span>{progress.stage}</span>
        </span>
        <span className="text-ink-muted">
          {progress.total > 0
            ? `step ${progress.step}/${progress.total} (${progress.percent}%)`
            : `${progress.percent}%`}{" "}
          &middot; {progress.elapsedSec}s
        </span>
      </div>
      <div className="w-full h-1.5 bg-muted rounded-[2px] overflow-hidden">
        <div
          className="h-full bg-ink transition-all duration-200"
          style={{ width: `${Math.max(4, progress.percent)}%` }}
        />
      </div>
    </div>
  );
}
