"use client";

import React, { useState, useEffect } from "react";
import { X, Broadcast, CheckCircle, WarningCircle } from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";

export function ConnectionModal() {
  const {
    gatewayUrl,
    apiKey,
    ntfyTopic,
    isConnectionModalOpen,
    setIsConnectionModalOpen,
    saveConnectionSettings,
    discoverFromNtfy,
  } = useStudio();

  const [urlInput, setUrlInput] = useState(gatewayUrl);
  const [keyInput, setKeyInput] = useState(apiKey);
  const [topicInput, setTopicInput] = useState(ntfyTopic);
  const [feedback, setFeedback] = useState<{
    type: "info" | "success" | "error";
    message: string;
  } | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (isConnectionModalOpen) {
      setUrlInput(gatewayUrl);
      setKeyInput(apiKey);
      setTopicInput(ntfyTopic);
      setFeedback(null);
    }
  }, [isConnectionModalOpen, gatewayUrl, apiKey, ntfyTopic]);

  if (!isConnectionModalOpen) return null;

  const handleSaveAndTest = async () => {
    setBusy(true);
    setFeedback({ type: "info", message: "Saving settings and probing /health..." });
    try {
      await saveConnectionSettings(urlInput, keyInput, topicInput);
      setFeedback({
        type: "success",
        message: "Configuration saved and gateway probed.",
      });
    } catch (err) {
      setFeedback({
        type: "error",
        message: err instanceof Error ? err.message : "Failed to connect",
      });
    } finally {
      setBusy(false);
    }
  };

  const handleDiscoverNtfy = async () => {
    setBusy(true);
    setFeedback({
      type: "info",
      message: `Querying https://ntfy.sh/${topicInput.trim()}/json?poll=1 ...`,
    });
    try {
      const discoveredUrl = await discoverFromNtfy(topicInput);
      setUrlInput(discoveredUrl);
      setFeedback({
        type: "success",
        message: `Discovered active tunnel: ${discoveredUrl}`,
      });
    } catch (err) {
      setFeedback({
        type: "error",
        message: err instanceof Error ? err.message : "ntfy discovery failed",
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-ink/30 z-50 flex items-center justify-center p-4">
      <div className="bg-surface border border-hairline rounded-[10px] w-full max-w-lg p-6 space-y-5">
        <div className="flex items-center justify-between border-b border-hairline pb-3">
          <div>
            <h3 className="font-serif text-2xl text-ink">
              Gateway &amp; Tunnel Configuration
            </h3>
            <p className="text-xs text-ink-muted">
              Connect to your Kaggle Cloudflare Tunnel or local FastAPI instance.
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
              Gateway Base URL
            </label>
            <input
              type="text"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              placeholder="https://xxxx-xxxx.trycloudflare.com"
              className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
            />
          </div>

          <div>
            <label className="block text-[11px] font-mono uppercase tracking-wider text-ink-muted mb-1">
              Bearer API Key (Optional)
            </label>
            <input
              type="password"
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              placeholder="sk-..."
              className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
            />
          </div>

          <div className="border-t border-hairline pt-3">
            <label className="block text-[11px] font-mono uppercase tracking-wider text-ink-muted mb-1">
              Auto-Fetch from ntfy.sh Topic
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={topicInput}
                onChange={(e) => setTopicInput(e.target.value)}
                placeholder="my-secret-kaggle-tunnel"
                className="flex-1 px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
              />
              <button
                type="button"
                disabled={busy}
                onClick={handleDiscoverNtfy}
                className="px-3 py-2 rounded-[6px] border border-hairline bg-subtle hover:bg-muted text-xs font-mono text-ink flex items-center gap-1.5 transition-transform active:scale-[0.98]"
              >
                <Broadcast weight="bold" className="w-3.5 h-3.5" />
                <span>Fetch URL</span>
              </button>
            </div>
          </div>

          {feedback && (
            <div
              className={`px-3 py-2 rounded-[6px] border text-xs font-mono flex items-center gap-2 ${
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

        <div className="flex items-center justify-end gap-2 pt-2 border-t border-hairline">
          <button
            type="button"
            onClick={() => setIsConnectionModalOpen(false)}
            className="px-3.5 py-2 rounded-[6px] border border-hairline bg-surface hover:bg-subtle text-xs font-mono text-ink"
          >
            Close
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={handleSaveAndTest}
            className="px-4 py-2 rounded-[6px] bg-ink text-surface hover:bg-[#2a2a2a] text-xs font-medium transition-transform active:scale-[0.98]"
          >
            Save &amp; Test Connection
          </button>
        </div>
      </div>
    </div>
  );
}
