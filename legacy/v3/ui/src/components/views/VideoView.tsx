"use client";

import React, { useState } from "react";
import { FilmStrip, DownloadSimple, UploadSimple, X } from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";
import { ParameterAccordion } from "@/components/ui/ParameterAccordion";
import { ProgressStrip } from "@/components/ui/ProgressStrip";
import { TelemetryFooter } from "@/components/ui/TelemetryFooter";
import { fileToDataUri } from "@/lib/utils";
import type { SseProgressState, TelemetryMeta } from "@/types/studio";

export function VideoView() {
  const { apiClient, refreshHealthAndMemory } = useStudio();

  const [model, setModel] = useState("Wan-AI/Wan2.2-TI2V-5B-Diffusers");
  const [prompt, setPrompt] = useState(
    "Slow cinematic dolly shot across a tranquil black-sand volcanic beach at dusk, waves gently rolling in, 24fps film grain."
  );
  const [negativePrompt, setNegativePrompt] = useState(
    "flickering, distorted limbs, watermark, low resolution"
  );
  const [size, setSize] = useState("416x240");
  // Wan2.2 requires 4k+1 frames
  const [numFrames, setNumFrames] = useState<number>(9);
  const [fps, setFps] = useState<number>(16);
  const [steps, setSteps] = useState<number>(4);
  const [guidanceScale, setGuidanceScale] = useState<number>(1.0);
  const [seed, setSeed] = useState<string>("-1");
  const [streamSse, setStreamSse] = useState<boolean>(true);
  const [conditioningImageUri, setConditioningImageUri] = useState<string | null>(
    null
  );

  const [videoB64, setVideoB64] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [progress, setProgress] = useState<SseProgressState>({
    active: false,
    stage: "",
    step: 0,
    total: 0,
    percent: 0,
    elapsedSec: 0,
  });
  const [telemetry, setTelemetry] = useState<TelemetryMeta | null>(null);

  const handleModelSelect = (newModel: string) => {
    setModel(newModel);
    if (newModel.includes("A14B")) {
      setSize("416x240");
      setNumFrames(9);
      setSteps(4);
      setGuidanceScale(1.0);
    } else if (newModel.includes("MiniMax") || newModel.includes("H3")) {
      setSize("416x240");
      setNumFrames(9);
      setSteps(4);
      setGuidanceScale(1.0);
    }
  };

  const handleUploadFirstFrame = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const uri = await fileToDataUri(file);
    setConditioningImageUri(uri);
  };

  const handleSynthesizeVideo = async () => {
    if (!prompt.trim() || isGenerating) return;
    setIsGenerating(true);
    const seedVal = parseInt(seed, 10);
    const cleanSeed = !Number.isNaN(seedVal) && seedVal >= 0 ? seedVal : null;
    const t0 = performance.now();

    try {
      const res = await apiClient.generateVideo(
        {
          model,
          prompt,
          negative_prompt: negativePrompt || null,
          image: conditioningImageUri,
          size,
          num_frames: numFrames,
          fps,
          num_inference_steps: steps,
          guidance_scale: guidanceScale,
          seed: cleanSeed,
          stream: streamSse,
        },
        setProgress
      );

      const b64 = res.data?.[0]?.b64_json || null;
      setVideoB64(b64);
      setTelemetry({
        endpoint: "POST /v1/videos/generations",
        model,
        extra: `${size} | ${numFrames}f @ ${fps}fps (${
          conditioningImageUri ? "I2V" : "T2V"
        })`,
        elapsedSec: ((performance.now() - t0) / 1000).toFixed(2),
      });
    } catch (err) {
      alert(err instanceof Error ? err.message : "Video synthesis failed");
    } finally {
      setProgress((p) => ({ ...p, active: false }));
      setIsGenerating(false);
      refreshHealthAndMemory(true);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 h-[calc(100vh-5.5rem)]">
      {/* Left Parameters */}
      <div className="lg:col-span-4 bg-surface border border-hairline rounded-[10px] p-5 flex flex-col justify-between overflow-y-auto">
        <div className="space-y-4">
          <div className="border-b border-hairline pb-3">
            <span className="text-[10px] font-mono uppercase tracking-widest text-ink-muted">
              POST /v1/videos/generations
            </span>
            <h2 className="font-serif text-2xl text-ink mt-0.5">
              Video Studio
            </h2>
          </div>

          <div>
            <label className="block text-[11px] font-mono uppercase tracking-wider text-ink-muted mb-1">
              Video Backbone
            </label>
            <select
              value={model}
              onChange={(e) => handleModelSelect(e.target.value)}
              className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
            >
              <option value="Wan-AI/Wan2.2-TI2V-5B-Diffusers">
                Wan-AI/Wan2.2-TI2V-5B-Diffusers (T2V &amp; I2V)
              </option>
              <option value="Wan-AI/Wan2.2-T2V-A14B-Diffusers">
                Wan-AI/Wan2.2-T2V-A14B-Diffusers (MoE 14B T2V)
              </option>
              <option value="Wan-AI/Wan2.2-I2V-A14B-Diffusers">
                Wan-AI/Wan2.2-I2V-A14B-Diffusers (MoE 14B I2V)
              </option>
              <option value="Comfy-Org/MiniMax-H3">
                Comfy-Org/MiniMax-H3 (T2VA &amp; I2VA + Native Audio)
              </option>
            </select>
          </div>

          <div>
            <label className="block text-[11px] font-mono uppercase tracking-wider text-ink-muted mb-1">
              Motion &amp; Scene Prompt
            </label>
            <textarea
              rows={3}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-sans resize-none"
            />
          </div>

          {/* Optional First-Frame Conditioning (I2V) */}
          <div className="border border-hairline rounded-[6px] p-3 bg-subtle space-y-2">
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-ink font-medium">
                First-Frame Image (Optional I2V)
              </span>
              {conditioningImageUri && (
                <button
                  type="button"
                  onClick={() => setConditioningImageUri(null)}
                  className="text-spot-redText flex items-center gap-1"
                >
                  <X weight="bold" className="w-3 h-3" />
                  <span>Clear</span>
                </button>
              )}
            </div>
            {conditioningImageUri ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={conditioningImageUri}
                alt="I2V conditioning frame"
                className="max-h-28 rounded-[4px] border border-hairline mx-auto object-contain bg-surface"
              />
            ) : (
              <label className="w-full py-2 rounded-[5px] border border-dashed border-[#CCCCCC] bg-surface hover:bg-canvas text-xs font-mono text-ink flex items-center justify-center gap-1.5 cursor-pointer">
                <UploadSimple weight="bold" className="w-3.5 h-3.5" />
                <span>Attach Starting Frame</span>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleUploadFirstFrame}
                  className="hidden"
                />
              </label>
            )}
          </div>

          <ParameterAccordion title="Frame Geometry (4k+1 Rule) & Sampling" defaultOpen>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10px] font-mono text-ink-muted mb-1">
                  Resolution
                </label>
                <select
                  value={size}
                  onChange={(e) => setSize(e.target.value)}
                  className="w-full px-2 py-1.5 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
                >
                  <option value="416x240">416x240 (16:9 Fast Preview)</option>
                  <option value="480x320">480x320 (3:2 Fast Compact)</option>
                  <option value="512x288">512x288 (16:9 Low-Res)</option>
                  <option value="640x480">640x480 (4:3 Standard)</option>
                  <option value="832x480">832x480 (16:9 Landscape)</option>
                  <option value="480x832">480x832 (9:16 Portrait)</option>
                </select>
              </div>
              <div>
                <label className="block text-[10px] font-mono text-ink-muted mb-1">
                  Frames (4k+1)
                </label>
                <select
                  value={numFrames}
                  onChange={(e) => setNumFrames(parseInt(e.target.value, 10))}
                  className="w-full px-2 py-1.5 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
                >
                  <option value={9}>9 frames (~0.5s Fast)</option>
                  <option value={17}>17 frames (~1.0s)</option>
                  <option value={33}>33 frames (~2.0s)</option>
                  <option value={49}>49 frames (~3.0s)</option>
                  <option value={81}>81 frames (~5.0s)</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2">
              <div>
                <label className="block text-[10px] font-mono text-ink-muted mb-1">
                  FPS
                </label>
                <input
                  type="number"
                  min={8}
                  max={30}
                  value={fps}
                  onChange={(e) => setFps(parseInt(e.target.value, 10) || 16)}
                  className="w-full px-2 py-1.5 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
                />
              </div>
              <div>
                <label className="block text-[10px] font-mono text-ink-muted mb-1">
                  Steps
                </label>
                <input
                  type="number"
                  min={2}
                  max={50}
                  value={steps}
                  onChange={(e) => setSteps(parseInt(e.target.value, 10) || 4)}
                  className="w-full px-2 py-1.5 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
                />
              </div>
              <div>
                <label className="block text-[10px] font-mono text-ink-muted mb-1">
                  CFG Scale
                </label>
                <input
                  type="number"
                  step={0.5}
                  min={1}
                  max={15}
                  value={guidanceScale}
                  onChange={(e) =>
                    setGuidanceScale(parseFloat(e.target.value) || 5.0)
                  }
                  className="w-full px-2 py-1.5 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10px] font-mono text-ink-muted mb-1">
                  Seed (-1 = random)
                </label>
                <input
                  type="text"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                  className="w-full px-2 py-1.5 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
                />
              </div>
              <div className="flex items-end">
                <label className="w-full flex items-center justify-between py-1.5 px-2.5 rounded-[6px] border border-hairline bg-subtle cursor-pointer text-xs font-mono">
                  <span>SSE Stream</span>
                  <input
                    type="checkbox"
                    checked={streamSse}
                    onChange={(e) => setStreamSse(e.target.checked)}
                    className="accent-ink"
                  />
                </label>
              </div>
            </div>
          </ParameterAccordion>
        </div>

        <button
          type="button"
          disabled={isGenerating}
          onClick={handleSynthesizeVideo}
          className="mt-4 w-full py-2.5 rounded-[6px] bg-ink text-surface hover:bg-[#2a2a2a] disabled:opacity-50 text-xs font-medium flex items-center justify-center gap-2 transition-transform active:scale-[0.98]"
        >
          <FilmStrip weight="bold" className="w-4 h-4" />
          <span>{isGenerating ? "Rendering Video..." : "Synthesize MP4 Video"}</span>
        </button>
      </div>

      {/* Right Video Viewport */}
      <div className="lg:col-span-8 bg-surface border border-hairline rounded-[10px] flex flex-col justify-between overflow-hidden">
        <div className="p-6 overflow-y-auto flex-1 space-y-4">
          <ProgressStrip progress={progress} />

          {videoB64 ? (
            <div className="space-y-4 text-center">
              <video
                src={`data:video/mp4;base64,${videoB64}`}
                controls
                autoPlay
                loop
                className="max-h-[460px] mx-auto rounded-[8px] border border-hairline bg-ink"
              />
              <div className="flex justify-center">
                <a
                  href={`data:video/mp4;base64,${videoB64}`}
                  download={`wan22-video-${Date.now()}.mp4`}
                  className="px-3.5 py-1.5 rounded-[6px] border border-hairline bg-subtle hover:bg-muted text-xs font-mono text-ink inline-flex items-center gap-1.5"
                >
                  <DownloadSimple weight="bold" className="w-3.5 h-3.5" />
                  <span>Download MP4</span>
                </a>
              </div>
            </div>
          ) : (
            <div className="h-72 flex flex-col items-center justify-center text-center">
              <p className="font-serif text-2xl text-ink mb-1">
                Wan2.2 Temporal Synthesis Theatre
              </p>
              <p className="text-xs text-ink-muted font-mono max-w-md">
                Supports Text-to-Video and First-Frame Image-to-Video with
                real-time SSE denoising step telemetry.
              </p>
            </div>
          )}
        </div>

        <TelemetryFooter telemetry={telemetry} />
      </div>
    </div>
  );
}
