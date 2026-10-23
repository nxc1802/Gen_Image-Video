"use client";

import React, { useState } from "react";
import { WarningOctagon } from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";

export function ShutdownModal() {
  const {
    apiClient,
    isShutdownModalOpen,
    setIsShutdownModalOpen,
    refreshHealthAndMemory,
  } = useStudio();

  const [confirmText, setConfirmText] = useState("");
  const [feedback, setFeedback] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (!isShutdownModalOpen) return null;

  const handleConfirmShutdown = async () => {
    if (confirmText.trim().toUpperCase() !== "SHUTDOWN") {
      setFeedback("Please type SHUTDOWN in uppercase to confirm.");
      return;
    }

    setBusy(true);
    setFeedback("Sending POST /v1/admin/shutdown?confirm=1 ...");
    try {
      const res = await apiClient.shutdownKernel();
      setFeedback(res.message || "Kernel termination sequence initiated.");
      setTimeout(() => {
        setIsShutdownModalOpen(false);
        setConfirmText("");
        setFeedback(null);
        refreshHealthAndMemory(true);
      }, 1400);
    } catch (err) {
      // Often the connection drops immediately as SIGKILL runs
      setFeedback(
        `Shutdown signal dispatched (${
          err instanceof Error ? err.message : "connection closed"
        }).`
      );
      setTimeout(() => {
        setIsShutdownModalOpen(false);
        setConfirmText("");
        setFeedback(null);
        refreshHealthAndMemory(true);
      }, 1500);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-ink/30 z-50 flex items-center justify-center p-4">
      <div className="bg-surface border border-hairline rounded-[10px] w-full max-w-md p-6 space-y-4">
        <div className="flex items-start justify-between border-b border-hairline pb-3">
          <div>
            <span className="inline-block px-2 py-0.5 rounded-full text-[10px] font-mono bg-spot-redBg text-spot-redText uppercase mb-1">
              Destructive Action
            </span>
            <h3 className="font-serif text-2xl text-ink flex items-center gap-2">
              <WarningOctagon weight="bold" className="w-5 h-5 text-spot-redText" />
              <span>Terminate Kaggle Kernel</span>
            </h3>
          </div>
        </div>

        <p className="text-xs text-ink-secondary leading-relaxed">
          This triggers{" "}
          <code className="font-mono bg-subtle px-1 py-0.5 border border-hairline rounded-[3px]">
            POST /v1/admin/shutdown?confirm=1
          </code>{" "}
          which executes{" "}
          <code className="font-mono bg-subtle px-1 py-0.5 border border-hairline rounded-[3px]">
            os.kill(1, SIGKILL)
          </code>{" "}
          inside the Kaggle VM to immediately stop GPU billing and save weekly
          T4 quota.
        </p>

        <div>
          <label className="block text-[11px] font-mono uppercase tracking-wider text-ink-muted mb-1">
            Type SHUTDOWN to confirm
          </label>
          <input
            type="text"
            value={confirmText}
            onChange={(e) => setConfirmText(e.target.value)}
            placeholder="SHUTDOWN"
            className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
          />
        </div>

        {feedback && (
          <div className="px-3 py-2 rounded-[6px] border border-hairline bg-subtle text-xs font-mono text-ink-secondary">
            {feedback}
          </div>
        )}

        <div className="flex items-center justify-end gap-2 pt-2 border-t border-hairline">
          <button
            type="button"
            onClick={() => {
              setIsShutdownModalOpen(false);
              setConfirmText("");
              setFeedback(null);
            }}
            className="px-3.5 py-2 rounded-[6px] border border-hairline bg-surface hover:bg-subtle text-xs font-mono text-ink"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={handleConfirmShutdown}
            className="px-4 py-2 rounded-[6px] bg-spot-redText text-surface hover:opacity-90 text-xs font-mono font-medium transition-transform active:scale-[0.98]"
          >
            Confirm SIGKILL
          </button>
        </div>
      </div>
    </div>
  );
}
