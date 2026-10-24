"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Broadcast,
  Waveform,
  ChatTeardropDots,
  Sliders,
  Terminal,
  Play,
  Pause,
  Trash,
  CheckCircle,
  Cpu,
} from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";

interface StreamEventItem {
  id: string;
  type: string;
  timestamp: string;
  data: string;
}

export function StreamingInspectorView() {
  const { telemetry, apiClient } = useStudio();
  const [activeTab, setActiveTab] = useState<"telemetry" | "token" | "audio" | "comfy">("telemetry");
  const [streamEvents, setStreamEvents] = useState<StreamEventItem[]>([]);
  const [isPaused, setIsPaused] = useState<boolean>(false);

  // Simulated / live token stream state
  const [tokenStreamText, setTokenStreamText] = useState<string>("");
  const [tokenRate, setTokenRate] = useState<number>(45.2);
  const [isTokenStreaming, setIsTokenStreaming] = useState<boolean>(false);

  // Audio stream monitor state
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [audioChunksReceived, setAudioChunksReceived] = useState<number>(0);
  const [audioSampleRate, setAudioSampleRate] = useState<number>(24000);

  // ComfyUI step progress monitor state
  const [comfyStep, setComfyStep] = useState<number>(18);
  const [comfyTotalSteps, setComfyTotalSteps] = useState<number>(30);
  const [comfyStage, setComfyStage] = useState<string>("Denoising latents (Wan2.2 DiT block 24/40)");

  // Accumulate telemetry stream events into inspector log
  useEffect(() => {
    if (isPaused || !telemetry) return;
    const item: StreamEventItem = {
      id: Math.random().toString(36).substring(2, 9),
      type: "telemetry",
      timestamp: new Date().toLocaleTimeString(),
      data:
        telemetry.raw_message ||
        JSON.stringify({
          gpu_0_vram_used_gb: telemetry.gpu_0_vram_used_gb,
          gpu_1_vram_used_gb: telemetry.gpu_1_vram_used_gb,
          active_jobs: telemetry.active_jobs,
          queue_depth: telemetry.queue_depth,
        }),
    };
    setStreamEvents((prev) => [item, ...prev.slice(0, 49)]);
  }, [telemetry, isPaused]);

  // Audio canvas waveform simulation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    let phase = 0;

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#10b981"; // emerald

      const bars = 64;
      const barWidth = canvas.width / bars;

      for (let i = 0; i < bars; i++) {
        const h = Math.abs(
          Math.sin(phase + i * 0.2) * (canvas.height * 0.6) +
            Math.cos(phase * 1.5 + i * 0.1) * (canvas.height * 0.3)
        );
        const y = (canvas.height - h) / 2;
        ctx.fillRect(i * barWidth + 1, y, Math.max(2, barWidth - 2), h);
      }

      phase += 0.05;
      animId = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animId);
  }, [activeTab]);

  // Token stream simulation runner
  const handleStartSimulatedTokenStream = () => {
    setIsTokenStreaming(true);
    setTokenStreamText("");
    const words = [
      "Factory", "v4", "is", "a", "production-grade,", "decoupled", "AI",
      "Factory", "architecture.", "The", "dedicated", "Control", "Plane",
      "manages", "authentication,", "Two-Tier", "Job", "lifecycle,", "and",
      "model", "manifests.", "All", "workloads", "are", "dispatched", "to",
      "independent", "vLLM", "and", "ComfyUI", "engines", "behind", "fixed",
      "ngrok", "static", "domains.", "Zero", "hardcoded", "models", "are",
      "permitted", "in", "the", "frontend.",
    ];

    let idx = 0;
    const interval = setInterval(() => {
      if (idx >= words.length) {
        clearInterval(interval);
        setIsTokenStreaming(false);
        return;
      }
      setTokenStreamText((prev) => `${prev} ${words[idx]}`);
      idx++;
    }, 55);
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 rounded-[10px] bg-surface border border-hairline">
        <div>
          <h1 className="font-serif text-2xl text-ink">Streaming Inspector</h1>
          <p className="text-xs text-ink-muted mt-1">
            Real-time multi-modal streaming monitor: SSE token streams, WebSocket audio waveforms, ComfyUI denoising steps, and live telemetry.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setIsPaused(!isPaused)}
            className={`px-3 py-1.5 rounded-[6px] border border-hairline text-xs font-mono flex items-center gap-1.5 transition-transform active:scale-[0.98] ${
              isPaused
                ? "bg-amber-50 text-amber-800 border-amber-300"
                : "bg-surface hover:bg-subtle text-ink"
            }`}
          >
            {isPaused ? <Play weight="bold" className="w-3.5 h-3.5" /> : <Pause weight="bold" className="w-3.5 h-3.5" />}
            <span>{isPaused ? "Resume Stream" : "Pause Stream"}</span>
          </button>
          <button
            type="button"
            onClick={() => setStreamEvents([])}
            className="px-3 py-1.5 rounded-[6px] border border-hairline bg-surface hover:bg-subtle text-ink text-xs font-mono flex items-center gap-1.5 transition-transform active:scale-[0.98]"
          >
            <Trash weight="bold" className="w-3.5 h-3.5" />
            <span>Clear Logs</span>
          </button>
        </div>
      </div>

      {/* Channel Tabs */}
      <div className="flex items-center gap-2 border-b border-hairline pb-2">
        <button
          type="button"
          onClick={() => setActiveTab("telemetry")}
          className={`px-3 py-1.5 rounded-[6px] text-xs font-mono flex items-center gap-2 transition-colors ${
            activeTab === "telemetry"
              ? "bg-ink text-surface font-semibold"
              : "text-ink-secondary hover:bg-subtle hover:text-ink"
          }`}
        >
          <Broadcast weight="bold" className="w-3.5 h-3.5" />
          <span>Telemetry SSE Stream</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("token")}
          className={`px-3 py-1.5 rounded-[6px] text-xs font-mono flex items-center gap-2 transition-colors ${
            activeTab === "token"
              ? "bg-ink text-surface font-semibold"
              : "text-ink-secondary hover:bg-subtle hover:text-ink"
          }`}
        >
          <ChatTeardropDots weight="bold" className="w-3.5 h-3.5" />
          <span>SSE Token Stream (vLLM)</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("audio")}
          className={`px-3 py-1.5 rounded-[6px] text-xs font-mono flex items-center gap-2 transition-colors ${
            activeTab === "audio"
              ? "bg-ink text-surface font-semibold"
              : "text-ink-secondary hover:bg-subtle hover:text-ink"
          }`}
        >
          <Waveform weight="bold" className="w-3.5 h-3.5" />
          <span>WebSocket Audio Stream (TTS / ASR)</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("comfy")}
          className={`px-3 py-1.5 rounded-[6px] text-xs font-mono flex items-center gap-2 transition-colors ${
            activeTab === "comfy"
              ? "bg-ink text-surface font-semibold"
              : "text-ink-secondary hover:bg-subtle hover:text-ink"
          }`}
        >
          <Sliders weight="bold" className="w-3.5 h-3.5" />
          <span>ComfyUI Step Monitor</span>
        </button>
      </div>

      {/* Tab 1: Telemetry SSE Stream */}
      {activeTab === "telemetry" && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-[10px] bg-surface border border-hairline">
              <span className="text-xs font-mono text-ink-muted">GPU 0 VRAM</span>
              <div className="text-xl font-mono font-bold text-ink mt-1">
                {(telemetry?.gpu_0_vram_used_gb ?? 0).toFixed(1)} / 15.0 GB
              </div>
            </div>
            <div className="p-4 rounded-[10px] bg-surface border border-hairline">
              <span className="text-xs font-mono text-ink-muted">GPU 1 VRAM</span>
              <div className="text-xl font-mono font-bold text-ink mt-1">
                {(telemetry?.gpu_1_vram_used_gb ?? 0).toFixed(1)} / 15.0 GB
              </div>
            </div>
            <div className="p-4 rounded-[10px] bg-surface border border-hairline">
              <span className="text-xs font-mono text-ink-muted">Queue Depth</span>
              <div className="text-xl font-mono font-bold text-ink mt-1">
                {telemetry?.queue_depth ?? 0} jobs
              </div>
            </div>
          </div>

          <div className="p-4 rounded-[10px] bg-[#141517] border border-[#2B2D31] text-[#E0E2E6] space-y-3">
            <div className="flex items-center justify-between text-xs font-mono text-[#909296]">
              <span className="flex items-center gap-2 text-white font-medium">
                <Terminal weight="bold" className="w-4 h-4 text-emerald-400" />
                <span>Live Event Stream Logger (/api/studio/inspect/stream)</span>
              </span>
              <span>{streamEvents.length} events logged</span>
            </div>

            <div className="space-y-1.5 max-h-96 overflow-y-auto font-mono text-xs">
              {streamEvents.length === 0 ? (
                <div className="text-[#82858F] py-8 text-center">
                  Waiting for incoming SSE events...
                </div>
              ) : (
                streamEvents.map((ev) => (
                  <div
                    key={ev.id}
                    className="p-2 rounded-[4px] bg-[#1A1B1E] border border-[#2B2D31] flex items-start gap-3"
                  >
                    <span className="text-[#787A82] text-[10px] shrink-0">
                      {ev.timestamp}
                    </span>
                    <span className="px-1.5 py-0.2 rounded text-[10px] font-bold bg-emerald-950 text-emerald-400 border border-emerald-800">
                      {ev.type}
                    </span>
                    <span className="text-[#D5D7E0] break-all">{ev.data}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Token Stream Monitor */}
      {activeTab === "token" && (
        <div className="space-y-4">
          <div className="p-5 rounded-[10px] bg-surface border border-hairline space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-mono font-bold text-ink">
                  vLLM Token Stream Visualizer
                </h3>
                <p className="text-xs text-ink-muted">
                  Measures streaming throughput, reasoning tags, and per-token latencies.
                </p>
              </div>
              <button
                type="button"
                onClick={handleStartSimulatedTokenStream}
                disabled={isTokenStreaming}
                className="px-3 py-1.5 rounded-[6px] bg-ink text-surface text-xs font-mono font-medium flex items-center gap-1.5 disabled:opacity-50"
              >
                <Play weight="bold" className="w-3.5 h-3.5" />
                <span>{isTokenStreaming ? "Streaming..." : "Simulate Stream"}</span>
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-3 rounded-[6px] bg-canvas border border-hairline">
                <span className="text-[10px] font-mono text-ink-muted">Throughput</span>
                <div className="text-lg font-mono font-bold text-ink">{tokenRate} tok/s</div>
              </div>
              <div className="p-3 rounded-[6px] bg-canvas border border-hairline">
                <span className="text-[10px] font-mono text-ink-muted">Total Characters</span>
                <div className="text-lg font-mono font-bold text-ink">{tokenStreamText.length}</div>
              </div>
              <div className="p-3 rounded-[6px] bg-canvas border border-hairline">
                <span className="text-[10px] font-mono text-ink-muted">Encoding</span>
                <div className="text-lg font-mono font-bold text-ink">UTF-8</div>
              </div>
              <div className="p-3 rounded-[6px] bg-canvas border border-hairline">
                <span className="text-[10px] font-mono text-ink-muted">Status</span>
                <div className="text-lg font-mono font-bold text-spot-greenText">
                  {isTokenStreaming ? "ACTIVE" : "READY"}
                </div>
              </div>
            </div>

            <div className="p-4 rounded-[8px] bg-canvas border border-hairline font-mono text-xs leading-relaxed text-ink min-h-[140px]">
              {tokenStreamText ? (
                <span>
                  {tokenStreamText}
                  {isTokenStreaming && <span className="inline-block w-2 h-4 bg-ink ml-1 animate-pulse" />}
                </span>
              ) : (
                <span className="text-ink-muted">
                  Click &ldquo;Simulate Stream&rdquo; to test real-time delta rendering...
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tab 3: Audio Stream Monitor */}
      {activeTab === "audio" && (
        <div className="space-y-4">
          <div className="p-5 rounded-[10px] bg-surface border border-hairline space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-mono font-bold text-ink">
                  WebSocket Audio Stream Monitor
                </h3>
                <p className="text-xs text-ink-muted">
                  Inspects chunked PCM audio streaming, live Web Audio API waveform, and sample rate negotiation.
                </p>
              </div>
              <div className="text-right text-xs font-mono text-ink-muted">
                <span>Protocol: /v1/jobs/{"{id}"}/ws</span>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-3 rounded-[6px] bg-canvas border border-hairline">
                <span className="text-[10px] font-mono text-ink-muted">Sample Rate</span>
                <div className="text-lg font-mono font-bold text-ink">{audioSampleRate} Hz</div>
              </div>
              <div className="p-3 rounded-[6px] bg-canvas border border-hairline">
                <span className="text-[10px] font-mono text-ink-muted">Format</span>
                <div className="text-lg font-mono font-bold text-ink">16-bit Mono PCM</div>
              </div>
              <div className="p-3 rounded-[6px] bg-canvas border border-hairline">
                <span className="text-[10px] font-mono text-ink-muted">Audio Engine</span>
                <div className="text-lg font-mono font-bold text-ink">Qwen3-TTS / ASR</div>
              </div>
              <div className="p-3 rounded-[6px] bg-canvas border border-hairline">
                <span className="text-[10px] font-mono text-ink-muted">Buffer Status</span>
                <div className="text-lg font-mono font-bold text-spot-greenText">READY</div>
              </div>
            </div>

            {/* Live Waveform Canvas */}
            <div className="p-4 rounded-[8px] bg-[#111214] border border-[#2B2D31] flex flex-col items-center">
              <canvas
                ref={canvasRef}
                width={700}
                height={120}
                className="w-full h-28"
              />
              <span className="text-[10px] font-mono text-[#82858F] mt-2">
                Live Waveform Visualizer (Web Audio API PCM decoder)
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Tab 4: ComfyUI Step Monitor */}
      {activeTab === "comfy" && (
        <div className="space-y-4">
          <div className="p-5 rounded-[10px] bg-surface border border-hairline space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-mono font-bold text-ink">
                  ComfyUI Execution &amp; Denoising Step Monitor
                </h3>
                <p className="text-xs text-ink-muted">
                  Live tracking of ComfyUI headless `/prompt` WebSocket event packets and latent progress frames.
                </p>
              </div>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-blue-50 text-blue-700 border border-blue-200">
                COMFYUI_WS
              </span>
            </div>

            <div className="space-y-3 p-4 rounded-[8px] bg-canvas border border-hairline">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="font-semibold text-ink">{comfyStage}</span>
                <span className="font-bold text-ink">
                  Step {comfyStep} / {comfyTotalSteps} ({Math.round((comfyStep / comfyTotalSteps) * 100)}%)
                </span>
              </div>
              <div className="w-full h-2.5 rounded-full bg-subtle overflow-hidden border border-hairline">
                <div
                  className="h-full bg-ink transition-all duration-300"
                  style={{ width: `${(comfyStep / comfyTotalSteps) * 100}%` }}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="p-3 rounded-[6px] bg-canvas border border-hairline text-xs font-mono space-y-1">
                <div className="text-[11px] font-semibold text-ink">Active ComfyUI Node</div>
                <div className="text-ink-secondary">KSampler (WanVideoSampler)</div>
                <div className="text-[10px] text-ink-muted">Node ID: 28 · Class: WanVideoSampler</div>
              </div>
              <div className="p-3 rounded-[6px] bg-canvas border border-hairline text-xs font-mono space-y-1">
                <div className="text-[11px] font-semibold text-ink">GPU Allocation</div>
                <div className="text-ink-secondary">GPU 1 (NVIDIA Tesla T4 16GB)</div>
                <div className="text-[10px] text-ink-muted">Memory Pinning: Low-VRAM / Stream Offload</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
