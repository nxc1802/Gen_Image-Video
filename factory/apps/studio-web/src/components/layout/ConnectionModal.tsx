"use client";

import React, { useState, useEffect } from "react";
import { X, CheckCircle, WarningCircle, ShieldCheck, Key } from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";

export function ConnectionModal() {
  const {
    gatewayUrl,
    apiKey,
    isConnectionModalOpen,
    setIsConnectionModalOpen,
    saveConnectionSettings,
  } = useStudio();

  const [urlInput, setUrlInput] = useState(gatewayUrl);
  const [keyInput, setKeyInput] = useState(apiKey);
  const [feedback, setFeedback] = useState<{
    type: "info" | "success" | "error";
    message: string;
  } | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (isConnectionModalOpen) {
      setUrlInput(gatewayUrl);
      setKeyInput(apiKey);
      setFeedback(null);
    }
  }, [isConnectionModalOpen, gatewayUrl, apiKey]);

  if (!isConnectionModalOpen) return null;

  const handleSaveAndTest = async () => {
    setBusy(true);
    setFeedback({
      type: "info",
      message: "Saving settings and testing Bearer token connection...",
    });
    try {
      await saveConnectionSettings(urlInput, keyInput);
      setFeedback({
        type: "success",
        message: "Connected to Factory v4 Control Plane successfully.",
      });
      setTimeout(() => {
        setIsConnectionModalOpen(false);
      }, 1200);
    } catch (err: any) {
      setFeedback({
        type: "error",
        message: err?.message || "Failed to connect to Control Plane",
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-ink/40 z-50 flex items-center justify-center p-4 backdrop-blur-sm animate-fadeIn">
      <div className="bg-surface border border-hairline rounded-[10px] w-full max-w-lg p-6 space-y-5 shadow-2xl">
        <div className="flex items-center justify-between border-b border-hairline pb-3">
          <div>
            <h3 className="font-serif text-2xl text-ink">
              Control Plane Connection
            </h3>
            <p className="text-xs text-ink-muted mt-0.5">
              Configure your Factory v4 Control API endpoint and Bearer authentication.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setIsConnectionModalOpen(false)}
            className="p-1.5 rounded-[6px] border border-hairline hover:bg-subtle text-ink"
          >
            <X weight="bold" className="w-4 h-4" />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-[11px] font-mono uppercase tracking-wider text-ink-muted mb-1">
              Control Plane Base URL
            </label>
            <input
              type="text"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              placeholder="https://studio.ai.example.com or http://localhost:8000"
              className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono text-ink focus:outline-none focus:ring-1 focus:ring-ink"
            />
            <span className="text-[10px] font-mono text-ink-muted mt-1 block">
              Default: http://localhost:8000 (routes to /api/studio and /v1/jobs)
            </span>
          </div>

          <div>
            <label className="block text-[11px] font-mono uppercase tracking-wider text-ink-muted mb-1 flex items-center gap-1.5">
              <Key weight="bold" className="w-3.5 h-3.5 text-amber-500" />
              <span>Bearer API Key</span>
              <span className="text-spot-redText text-[10px] font-bold">*</span>
            </label>
            <input
              type="password"
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              placeholder="sk-factory-studio"
              className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono text-ink focus:outline-none focus:ring-1 focus:ring-ink"
            />
            <span className="text-[10px] font-mono text-ink-muted mt-1 flex items-center gap-1">
              <ShieldCheck weight="bold" className="w-3.5 h-3.5 text-emerald-600" />
              <span>Mandatory Bearer token authentication enforced on 100% of routes.</span>
            </span>
          </div>

          {feedback && (
            <div
              className={`px-3 py-2.5 rounded-[6px] border text-xs font-mono flex items-center gap-2 ${
                feedback.type === "success"
                  ? "bg-spot-greenBg border-[#C2DEC4] text-spot-greenText"
                  : feedback.type === "error"
                  ? "bg-spot-redBg border-[#E6C5C5] text-spot-redText"
                  : "bg-subtle border-hairline text-ink-secondary"
              }`}
            >
              {feedback.type === "success" ? (
                <CheckCircle weight="bold" className="w-4 h-4 shrink-0" />
              ) : feedback.type === "error" ? (
                <WarningCircle weight="bold" className="w-4 h-4 shrink-0" />
              ) : null}
              <span>{feedback.message}</span>
            </div>
          )}
        </div>

        <div className="flex items-center justify-end gap-2 pt-3 border-t border-hairline">
          <button
            type="button"
            onClick={() => setIsConnectionModalOpen(false)}
            className="px-3.5 py-2 rounded-[6px] border border-hairline bg-surface hover:bg-subtle text-xs font-mono text-ink"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={handleSaveAndTest}
            className="px-4 py-2 rounded-[6px] bg-ink text-surface hover:bg-[#2a2a2a] text-xs font-medium transition-transform active:scale-[0.98] disabled:opacity-50"
          >
            {busy ? "Connecting..." : "Save & Connect"}
          </button>
        </div>
      </div>
    </div>
  );
}
