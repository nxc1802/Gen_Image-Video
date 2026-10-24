"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  SlidersHorizontal,
  Power,
  ArrowsClockwise,
  CheckCircle,
} from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";

export function ConfigAdminView() {
  const { apiClient, setIsShutdownModalOpen, setIsConnectionModalOpen } =
    useStudio();

  const [llmModel, setLlmModel] = useState("Qwen/Qwen3-VL-8B-Instruct");
  const [imageModel, setImageModel] = useState("Qwen/Qwen-Image");
  const [editModel, setEditModel] = useState("Qwen/Qwen-Image-Edit");
  const [videoModel, setVideoModel] = useState(
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
  );
  const [ttsModel, setTtsModel] = useState(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
  );
  const [sttModel, setSttModel] = useState("Qwen/Qwen3-ASR-1.7B");
  const [enableThinking, setEnableThinking] = useState(false);

  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);

  const fetchConfig = useCallback(async () => {
    setBusy(true);
    setFeedback(null);
    try {
      const cfg = await apiClient.getRuntimeConfig();
      if (cfg.llm_model) setLlmModel(cfg.llm_model);
      if (cfg.image_model) setImageModel(cfg.image_model);
      if (cfg.image_edit_model) setEditModel(cfg.image_edit_model);
      if (cfg.video_model) setVideoModel(cfg.video_model);
      if (cfg.tts_model) setTtsModel(cfg.tts_model);
      if (cfg.stt_model) setSttModel(cfg.stt_model);
      if (typeof cfg.enable_thinking === "boolean") {
        setEnableThinking(cfg.enable_thinking);
      }
      setFeedback("Synced runtime configuration from GET /v1/config.");
    } catch (err) {
      setFeedback(
        err instanceof Error
          ? `GET /v1/config notice: ${err.message}`
          : "Could not fetch /v1/config"
      );
    } finally {
      setBusy(false);
    }
  }, [apiClient]);

  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

  const handleApplyConfig = async () => {
    setBusy(true);
    setFeedback("Applying hot-reload configuration via POST /v1/config ...");
    try {
      await apiClient.updateRuntimeConfig({
        llm_model: llmModel,
        image_model: imageModel,
        image_edit_model: editModel,
        video_model: videoModel,
        tts_model: ttsModel,
        stt_model: sttModel,
        enable_thinking: enableThinking,
      });
      setFeedback("Runtime configuration hot-reloaded successfully.");
    } catch (err) {
      setFeedback(
        err instanceof Error ? err.message : "POST /v1/config failed"
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
      {/* Left: Hot-Reload Runtime Config */}
      <div className="lg:col-span-7 bg-surface border border-hairline rounded-[10px] p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-hairline pb-3">
          <div>
            <span className="text-[10px] font-mono uppercase tracking-widest text-ink-muted">
              GET &amp; POST /v1/config
            </span>
            <h2 className="font-serif text-2xl text-ink mt-0.5">
              Hot-Reload Runtime Model Configuration
            </h2>
          </div>
          <button
            type="button"
            onClick={fetchConfig}
            className="px-3 py-1.5 rounded-[6px] border border-hairline bg-subtle hover:bg-muted text-xs font-mono flex items-center gap-1.5"
          >
            <ArrowsClockwise weight="bold" className="w-3.5 h-3.5" />
            <span>Reload</span>
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="block text-[10px] font-mono uppercase text-ink-muted mb-1">
              LLM / VLM Checkpoint (llm_model)
            </label>
            <input
              type="text"
              value={llmModel}
              onChange={(e) => setLlmModel(e.target.value)}
              className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
            />
          </div>

          <div>
            <label className="block text-[10px] font-mono uppercase text-ink-muted mb-1">
              Primary Image Model (image_model)
            </label>
            <input
              type="text"
              value={imageModel}
              onChange={(e) => setImageModel(e.target.value)}
              className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
            />
          </div>

          <div>
            <label className="block text-[10px] font-mono uppercase text-ink-muted mb-1">
              Image Edit / Inpaint (image_edit_model)
            </label>
            <input
              type="text"
              value={editModel}
              onChange={(e) => setEditModel(e.target.value)}
              className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
            />
          </div>

          <div>
            <label className="block text-[10px] font-mono uppercase text-ink-muted mb-1">
              Video Backbone (video_model)
            </label>
            <input
              type="text"
              value={videoModel}
              onChange={(e) => setVideoModel(e.target.value)}
              className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
            />
          </div>

          <div>
            <label className="block text-[10px] font-mono uppercase text-ink-muted mb-1">
              TTS Checkpoint (tts_model)
            </label>
            <input
              type="text"
              value={ttsModel}
              onChange={(e) => setTtsModel(e.target.value)}
              className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
            />
          </div>

          <div>
            <label className="block text-[10px] font-mono uppercase text-ink-muted mb-1">
              ASR Checkpoint (stt_model)
            </label>
            <input
              type="text"
              value={sttModel}
              onChange={(e) => setSttModel(e.target.value)}
              className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
            />
          </div>
        </div>

        <label className="flex items-center justify-between py-2 px-3 rounded-[6px] border border-hairline bg-subtle cursor-pointer">
          <span className="text-xs font-medium text-ink">
            Default Global Chain-of-Thought (enable_thinking)
          </span>
          <input
            type="checkbox"
            checked={enableThinking}
            onChange={(e) => setEnableThinking(e.target.checked)}
            className="accent-ink"
          />
        </label>

        {feedback && (
          <div className="px-3 py-2 rounded-[6px] border border-hairline bg-subtle text-xs font-mono text-ink-secondary flex items-center gap-2">
            <CheckCircle weight="bold" className="w-4 h-4 shrink-0" />
            <span>{feedback}</span>
          </div>
        )}

        <button
          type="button"
          disabled={busy}
          onClick={handleApplyConfig}
          className="w-full py-2.5 rounded-[6px] bg-ink text-surface hover:bg-[#2a2a2a] disabled:opacity-50 text-xs font-medium flex items-center justify-center gap-2 transition-transform active:scale-[0.98]"
        >
          <SlidersHorizontal weight="bold" className="w-4 h-4" />
          <span>Apply Runtime Config (POST /v1/config)</span>
        </button>
      </div>

      {/* Right: Tunnel & Emergency Kernel Shutdown */}
      <div className="lg:col-span-5 space-y-5">
        <div className="bg-surface border border-hairline rounded-[10px] p-6 space-y-3">
          <span className="text-[10px] font-mono uppercase tracking-widest text-ink-muted">
            Tunnel &amp; Auth
          </span>
          <h3 className="font-serif text-xl text-ink">
            Cloudflare Tunnel &amp; ntfy.sh Discovery
          </h3>
          <p className="text-xs text-ink-secondary leading-relaxed">
            Configure your active Kaggle Cloudflare Tunnel URL, Bearer API key,
            or automatically poll your{" "}
            <code className="font-mono bg-subtle px-1 py-0.5 border border-hairline rounded-[3px]">
              ntfy.sh
            </code>{" "}
            topic.
          </p>
          <button
            type="button"
            onClick={() => setIsConnectionModalOpen(true)}
            className="w-full py-2 rounded-[6px] border border-hairline bg-subtle hover:bg-muted text-xs font-mono text-ink"
          >
            Open Connection &amp; ntfy Modal
          </button>
        </div>

        <div className="bg-surface border border-[#E6C5C5] rounded-[10px] p-6 space-y-3">
          <span className="px-2 py-0.5 rounded-full text-[10px] font-mono uppercase bg-spot-redBg text-spot-redText">
            POST /v1/admin/shutdown?confirm=1
          </span>
          <h3 className="font-serif text-xl text-ink">
            Kaggle Kernel Emergency Shutdown
          </h3>
          <p className="text-xs text-ink-secondary leading-relaxed">
            Immediately terminates the Kaggle notebook VM via{" "}
            <code className="font-mono bg-subtle px-1 py-0.5 border border-hairline rounded-[3px]">
              os.kill(1, SIGKILL)
            </code>{" "}
            to stop GPU billing and preserve weekly dual-T4 quota.
          </p>
          <button
            type="button"
            onClick={() => setIsShutdownModalOpen(true)}
            className="w-full py-2.5 rounded-[6px] bg-spot-redText text-surface hover:opacity-90 text-xs font-mono font-medium flex items-center justify-center gap-2 transition-transform active:scale-[0.98]"
          >
            <Power weight="bold" className="w-4 h-4" />
            <span>Open Kernel Shutdown Confirmation</span>
          </button>
        </div>
      </div>
    </div>
  );
}
