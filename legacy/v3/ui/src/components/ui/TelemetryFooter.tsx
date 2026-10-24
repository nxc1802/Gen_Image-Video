"use client";

import React from "react";
import type { TelemetryMeta } from "@/types/studio";

interface TelemetryFooterProps {
  telemetry: TelemetryMeta | null;
}

export function TelemetryFooter({ telemetry }: TelemetryFooterProps) {
  if (!telemetry) return null;

  return (
    <div className="border-t border-hairline px-5 py-2.5 bg-subtle flex flex-wrap items-center justify-between text-[11px] font-mono text-ink-muted gap-2">
      <div>
        <span className="text-ink font-medium">{telemetry.endpoint}</span>
      </div>
      <div className="flex items-center gap-3">
        {telemetry.model && <span>model: {telemetry.model}</span>}
        {telemetry.extra && <span>{telemetry.extra}</span>}
        {telemetry.elapsedSec !== undefined && (
          <span>elapsed: {telemetry.elapsedSec}s</span>
        )}
      </div>
    </div>
  );
}
