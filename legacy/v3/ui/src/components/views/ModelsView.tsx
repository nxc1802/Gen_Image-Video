"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  ArrowsClockwise,
  CheckCircle,
  WarningCircle,
  Lightning,
} from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";
import type { ModelEntry, AudioDiagnosticsResponse } from "@/types/studio";

export function ModelsView() {
  const { apiClient, loadedModels, activeModalities } = useStudio();

  const [models, setModels] = useState<ModelEntry[]>([]);
  const [diag, setDiag] = useState<AudioDiagnosticsResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadCatalogAndDiagnostics = useCallback(
    async (tryLoadAudio = false) => {
      setBusy(true);
      setError(null);
      try {
        const [modelsRes, diagRes] = await Promise.all([
          apiClient.getModels(),
          apiClient.getAudioDiagnostics(tryLoadAudio).catch(() => null),
        ]);
        setModels(modelsRes.data || []);
        setDiag(diagRes);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to load /v1/models"
        );
      } finally {
        setBusy(false);
      }
    },
    [apiClient]
  );

  useEffect(() => {
    loadCatalogAndDiagnostics(false);
  }, [loadCatalogAndDiagnostics]);

  const renderEngineStateBadge = (state?: string) => {
    if (state === "loaded") {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono bg-spot-greenBg text-spot-greenText">
          <CheckCircle weight="bold" className="w-3 h-3" />
          LOADED IN GPU
        </span>
      );
    }
    if (state === "fallback") {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono bg-spot-yellowBg text-spot-yellowText">
          <WarningCircle weight="bold" className="w-3 h-3" />
          FALLBACK
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono bg-subtle text-ink-muted border border-hairline">
        {state || "STANDBY"}
      </span>
    );
  };

  return (
    <div className="space-y-5">
      {/* Header Banner */}
      <div className="bg-surface border border-hairline rounded-[10px] p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <span className="text-[10px] font-mono uppercase tracking-widest text-ink-muted">
            GET /v1/models &middot; GET /v1/audio/diagnostics
          </span>
          <h2 className="font-serif text-2xl text-ink mt-0.5">
            Model Registry &amp; Audio Engine Diagnostics
          </h2>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            disabled={busy}
            onClick={() => loadCatalogAndDiagnostics(true)}
            className="px-3 py-2 rounded-[6px] border border-hairline bg-subtle hover:bg-muted text-xs font-mono text-ink flex items-center gap-1.5 transition-transform active:scale-[0.98]"
          >
            <Lightning weight="bold" className="w-3.5 h-3.5" />
            <span>Force Load Audio Models (try_load=true)</span>
          </button>

          <button
            type="button"
            disabled={busy}
            onClick={() => loadCatalogAndDiagnostics(false)}
            className="px-3.5 py-2 rounded-[6px] bg-ink text-surface hover:bg-[#2a2a2a] text-xs font-mono flex items-center gap-1.5 transition-transform active:scale-[0.98]"
          >
            <ArrowsClockwise
              weight="bold"
              className={`w-3.5 h-3.5 ${busy ? "animate-spin" : ""}`}
            />
            <span>Refresh Registry</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-[8px] border border-[#E6C5C5] bg-spot-redBg text-spot-redText text-xs font-mono">
          {error}
        </div>
      )}

      {/* Audio Engine Diagnostics */}
      <div className="bg-surface border border-hairline rounded-[10px] p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between border-b border-hairline pb-2.5 gap-2">
          <h3 className="font-serif text-xl text-ink">
            Audio Runtime Diagnostics (/v1/audio/diagnostics)
          </h3>
          {diag && (
            <span className="text-xs font-mono text-ink-muted">
              Device: {diag.device || "cuda:0"} ({diag.cuda_device_name || "GPU"})
            </span>
          )}
        </div>

        {diag ? (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
              <div className="border border-hairline rounded-[8px] p-4 bg-canvas space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-ink-muted uppercase text-[10px]">
                    1. Qwen3-ASR 1.7B
                  </span>
                  {renderEngineStateBadge(diag.models?.asr)}
                </div>
                <div className="text-ink font-semibold truncate">
                  {diag.paths?.asr_path || "Qwen/Qwen3-ASR-1.7B"}
                </div>
                {diag.errors?.asr && (
                  <div className="text-[11px] text-spot-redText break-all">
                    {diag.errors.asr}
                  </div>
                )}
              </div>

              <div className="border border-hairline rounded-[8px] p-4 bg-canvas space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-ink-muted uppercase text-[10px]">
                    2. Qwen3-TTS 1.7B
                  </span>
                  {renderEngineStateBadge(diag.models?.tts)}
                </div>
                <div className="text-ink font-semibold truncate">
                  {diag.paths?.tts_path || "Qwen/Qwen3-TTS-12Hz-1.7B-Base"}
                </div>
                {diag.errors?.tts && (
                  <div className="text-[11px] text-spot-redText break-all">
                    {diag.errors.tts}
                  </div>
                )}
              </div>

              <div className="border border-hairline rounded-[8px] p-4 bg-canvas space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-ink-muted uppercase text-[10px]">
                    3. XTTS-v2 Vietnamese
                  </span>
                  {renderEngineStateBadge(diag.models?.xtts)}
                </div>
                <div className="text-ink font-semibold truncate">
                  {diag.paths?.xtts_path || "thivux/XTTS-v2-vietnamse"}
                </div>
                {diag.errors?.xtts && (
                  <div className="text-[11px] text-spot-redText break-all">
                    {diag.errors.xtts}
                  </div>
                )}
              </div>
            </div>

            {diag.imports && (
              <div className="border border-hairline rounded-[8px] p-4 bg-canvas space-y-2">
                <div className="text-[10px] font-mono uppercase text-ink-muted">
                  Python Audio Package Imports &amp; Reference Speaker
                </div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(diag.imports).map(([pkg, info]) => (
                    <span
                      key={pkg}
                      className={`px-2.5 py-1 rounded-[5px] text-xs font-mono border ${
                        info.status === "ok"
                          ? "bg-spot-greenBg text-spot-greenText border-[#C2DEC4]"
                          : "bg-spot-redBg text-spot-redText border-[#E6C5C5]"
                      }`}
                    >
                      {pkg}: {info.status}
                    </span>
                  ))}
                  {diag.default_ref_wav && (
                    <span className="px-2.5 py-1 rounded-[5px] text-xs font-mono bg-subtle border border-hairline text-ink-secondary">
                      default_speaker.wav:{" "}
                      {diag.default_ref_wav.exists ? "present" : "missing"}
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="py-8 text-center text-xs font-mono text-ink-muted">
            Audio diagnostics will populate when connected to a gateway with
            audio enabled.
          </div>
        )}
      </div>

      {/* Bento Grid: Model Registry */}
      <div className="bg-surface border border-hairline rounded-[10px] p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-hairline pb-2.5">
          <h3 className="font-serif text-xl text-ink">
            Registered Models ({models.length})
          </h3>
          <span className="text-xs font-mono text-ink-muted">
            Active Modalities: {activeModalities.join(", ") || "none"}
          </span>
        </div>

        {models.length === 0 ? (
          <div className="py-12 text-center text-xs font-mono text-ink-muted">
            Connect to the gateway and click Refresh Registry to inspect{" "}
            <code>/v1/models</code>.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {models.map((m) => {
              const isLoaded =
                m.loaded ||
                loadedModels.some(
                  (lm) =>
                    m.id.toLowerCase().includes(lm.toLowerCase()) ||
                    lm.toLowerCase().includes(m.id.toLowerCase())
                ) ||
                (diag?.models?.asr === "loaded" &&
                  m.id.toLowerCase().includes("asr")) ||
                (diag?.models?.tts === "loaded" &&
                  m.id.toLowerCase().includes("qwen3-tts")) ||
                (diag?.models?.xtts === "loaded" &&
                  m.id.toLowerCase().includes("xtts"));

              return (
                <div
                  key={m.id}
                  className="border border-hairline rounded-[8px] p-4 bg-canvas flex flex-col justify-between space-y-3"
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-mono uppercase bg-spot-blueBg text-spot-blueText">
                      {m.type || m.owned_by || "model"}
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded-full text-[10px] font-mono ${
                        isLoaded
                          ? "bg-spot-greenBg text-spot-greenText"
                          : "bg-subtle text-ink-muted border border-hairline"
                      }`}
                    >
                      {isLoaded ? "LOADED IN VRAM" : "CATALOG"}
                    </span>
                  </div>
                  <div className="font-mono text-xs font-semibold text-ink break-all">
                    {m.id}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
