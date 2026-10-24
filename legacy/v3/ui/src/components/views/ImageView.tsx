"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  PaintBrushBroad,
  DownloadSimple,
  Eraser,
  Columns,
  UploadSimple,
} from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";
import { ParameterAccordion } from "@/components/ui/ParameterAccordion";
import { ProgressStrip } from "@/components/ui/ProgressStrip";
import { TelemetryFooter } from "@/components/ui/TelemetryFooter";
import { fileToDataUri } from "@/lib/utils";
import type { SseProgressState, TelemetryMeta } from "@/types/studio";

type ImageSubTab = "t2i" | "i2i";

export function ImageView() {
  const { apiClient, refreshHealthAndMemory } = useStudio();

  const [subTab, setSubTab] = useState<ImageSubTab>("t2i");
  const [model, setModel] = useState("abenzerps/Qwen-Image-2.1-GGUF");
  const [prompt, setPrompt] = useState(
    "Architectural photograph of a brutalist concrete studio in Kyoto surrounded by moss, soft morning fog, 35mm medium format film."
  );
  const [negativePrompt, setNegativePrompt] = useState("");
  const [size, setSize] = useState("512x512");
  const [steps, setSteps] = useState(6);
  const [guidanceScale, setGuidanceScale] = useState(1.0);
  const [strength, setStrength] = useState(0.75);
  const [seed, setSeed] = useState<string>("-1");
  const [streamSse, setStreamSse] = useState(true);
  const [compareMode, setCompareMode] = useState(false);

  // I2I / Inpaint Canvas state
  const [sourceImageUri, setSourceImageUri] = useState<string | null>(null);
  const [enableMaskBrush, setEnableMaskBrush] = useState(false);
  const [brushSize, setBrushSize] = useState(28);
  const [hasDrawnMask, setHasDrawnMask] = useState(false);

  const bgCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const isDrawingRef = useRef(false);

  // Output state
  const [outputB64, setOutputB64] = useState<string | null>(null);
  const [compareOutput, setCompareOutput] = useState<{
    fluxB64: string | null;
    fluxTime: string;
    qwenB64: string | null;
    qwenTime: string;
    status: string;
  } | null>(null);
  const [progress, setProgress] = useState<SseProgressState>({
    active: false,
    stage: "",
    step: 0,
    total: 0,
    percent: 0,
    elapsedSec: 0,
  });
  const [isGenerating, setIsGenerating] = useState(false);
  const [telemetry, setTelemetry] = useState<TelemetryMeta | null>(null);

  // Adjust default steps when switching between fast and high-detail presets
  const handleModelChange = (newModel: string) => {
    setModel(newModel);
    if (newModel.includes("schnell") || newModel.includes("GGUF")) {
      setSteps(6);
      setGuidanceScale(1.0);
    } else if (steps < 10) {
      setSteps(12);
    }
  };

  // Draw uploaded source image onto background canvas
  useEffect(() => {
    if (!sourceImageUri || !bgCanvasRef.current || !maskCanvasRef.current) return;
    const img = new Image();
    img.onload = () => {
      const bgCanvas = bgCanvasRef.current;
      const maskCanvas = maskCanvasRef.current;
      if (!bgCanvas || !maskCanvas) return;
      bgCanvas.width = 512;
      bgCanvas.height = 512;
      maskCanvas.width = 512;
      maskCanvas.height = 512;
      const ctx = bgCanvas.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, 512, 512);
        ctx.drawImage(img, 0, 0, 512, 512);
      }
      clearMaskCanvas();
    };
    img.src = sourceImageUri;
  }, [sourceImageUri]);

  const clearMaskCanvas = () => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) return;
    const mctx = maskCanvas.getContext("2d");
    if (mctx) {
      mctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    }
    setHasDrawnMask(false);
  };

  const paintAtEvent = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!enableMaskBrush || !isDrawingRef.current || !maskCanvasRef.current) return;
    const canvas = maskCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
    ctx.beginPath();
    ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    ctx.fill();
    setHasDrawnMask(true);
  };

  const exportBinaryMaskDataUri = (): string | null => {
    if (!enableMaskBrush || !hasDrawnMask || !maskCanvasRef.current) return null;
    const srcCanvas = maskCanvasRef.current;
    const outCanvas = document.createElement("canvas");
    outCanvas.width = srcCanvas.width;
    outCanvas.height = srcCanvas.height;
    const ctx = outCanvas.getContext("2d");
    if (!ctx) return null;

    // Fill black (keep area), then draw white strokes (inpaint area)
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, outCanvas.width, outCanvas.height);
    ctx.drawImage(srcCanvas, 0, 0);
    return outCanvas.toDataURL("image/png");
  };

  const handleUploadSource = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const uri = await fileToDataUri(file);
    setSourceImageUri(uri);
  };

  const handleGenerate = async () => {
    if (!prompt.trim() || isGenerating) return;
    const isEdit = subTab === "i2i";
    if (isEdit && !sourceImageUri) {
      alert("Please upload a source image for Image Edit / Inpaint.");
      return;
    }

    setIsGenerating(true);
    const seedVal = parseInt(seed, 10);
    const cleanSeed = !Number.isNaN(seedVal) && seedVal >= 0 ? seedVal : null;

    if (compareMode && !isEdit) {
      setCompareOutput({
        fluxB64: null,
        fluxTime: "0.0",
        qwenB64: null,
        qwenTime: "0.0",
        status: "Pass #1: Running Fast 4-Step Pass...",
      });

      const t0 = performance.now();
      try {
        const startFlux = performance.now();
        const resFlux = await apiClient.generateOrEditImage(
          false,
          {
            model: "abenzerps/Qwen-Image-2.1-GGUF",
            prompt,
            size,
            num_inference_steps: 4,
            guidance_scale: 1.0,
            seed: cleanSeed,
            stream: streamSse,
          },
          setProgress
        );
        const fluxTime = ((performance.now() - startFlux) / 1000).toFixed(2);
        const fluxB64 = resFlux.data?.[0]?.b64_json || null;

        setCompareOutput({
          fluxB64,
          fluxTime,
          qwenB64: null,
          qwenTime: "0.0",
          status: "Pass #2: Running High-Detail 8-Step Pass...",
        });

        const startQwen = performance.now();
        const resQwen = await apiClient.generateOrEditImage(
          false,
          {
            model: "Qwen/Qwen-Image",
            prompt,
            negative_prompt: negativePrompt || null,
            size,
            num_inference_steps: 8,
            guidance_scale: guidanceScale,
            seed: cleanSeed,
            stream: streamSse,
          },
          setProgress
        );
        const qwenTime = ((performance.now() - startQwen) / 1000).toFixed(2);
        const qwenB64 = resQwen.data?.[0]?.b64_json || null;

        setCompareOutput({
          fluxB64,
          fluxTime,
          qwenB64,
          qwenTime,
          status: "Completed Sequential Comparison",
        });

        setTelemetry({
          endpoint: "POST /v1/images/generations (Sequential Compare)",
          model: "Qwen-Image-2.1 (4-Step vs 8-Step)",
          extra: `size=${size}`,
          elapsedSec: ((performance.now() - t0) / 1000).toFixed(2),
        });
      } catch (err) {
        setCompareOutput((prev) =>
          prev
            ? {
                ...prev,
                status: `Error: ${
                  err instanceof Error ? err.message : "Failed"
                }`,
              }
            : null
        );
      } finally {
        setProgress((p) => ({ ...p, active: false }));
        setIsGenerating(false);
        refreshHealthAndMemory(true);
      }
      return;
    }

    // Single Image Generation or Edit
    setCompareOutput(null);
    const t0 = performance.now();
    try {
      const maskUri = isEdit ? exportBinaryMaskDataUri() : null;
      const res = await apiClient.generateOrEditImage(
        isEdit,
        {
          model: isEdit ? "Qwen/Qwen-Image-Edit" : model,
          prompt,
          negative_prompt: negativePrompt || null,
          size,
          num_inference_steps: steps,
          guidance_scale: guidanceScale,
          seed: cleanSeed,
          stream: streamSse,
          ...(isEdit
            ? {
                image: sourceImageUri || "",
                mask_image: maskUri,
                strength,
              }
            : {}),
        },
        setProgress
      );

      const b64 = res.data?.[0]?.b64_json || null;
      setOutputB64(b64);
      setTelemetry({
        endpoint: isEdit
          ? "POST /v1/images/edits"
          : "POST /v1/images/generations",
        model: isEdit ? "Qwen/Qwen-Image-Edit" : model,
        extra: `${size} | steps=${steps}`,
        elapsedSec: ((performance.now() - t0) / 1000).toFixed(2),
      });
    } catch (err) {
      alert(err instanceof Error ? err.message : "Image synthesis failed");
    } finally {
      setProgress((p) => ({ ...p, active: false }));
      setIsGenerating(false);
      refreshHealthAndMemory(true);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 h-[calc(100vh-5.5rem)]">
      {/* Left Control Column */}
      <div className="lg:col-span-4 bg-surface border border-hairline rounded-[10px] p-5 flex flex-col justify-between overflow-y-auto">
        <div className="space-y-4">
          <div className="border-b border-hairline pb-3">
            <span className="text-[10px] font-mono uppercase tracking-widest text-ink-muted">
              POST /v1/images/generations &amp; /edits
            </span>
            <h2 className="font-serif text-2xl text-ink mt-0.5">
              Image Studio
            </h2>
          </div>

          {/* Mode Sub-tabs */}
          <div className="grid grid-cols-2 gap-1.5 p-1 rounded-[6px] bg-subtle border border-hairline">
            <button
              type="button"
              onClick={() => setSubTab("t2i")}
              className={`py-1.5 rounded-[5px] text-xs font-mono ${
                subTab === "t2i"
                  ? "bg-surface text-ink font-semibold border border-hairline"
                  : "text-ink-muted"
              }`}
            >
              Text-to-Image
            </button>
            <button
              type="button"
              onClick={() => {
                setSubTab("i2i");
                setModel("Qwen/Qwen-Image-Edit");
              }}
              className={`py-1.5 rounded-[5px] text-xs font-mono ${
                subTab === "i2i"
                  ? "bg-surface text-ink font-semibold border border-hairline"
                  : "text-ink-muted"
              }`}
            >
              Edit &amp; Inpaint
            </button>
          </div>

          {subTab === "t2i" && (
            <div>
              <label className="block text-[11px] font-mono uppercase tracking-wider text-ink-muted mb-1">
                Diffusion Pipeline
              </label>
              <select
                value={model}
                onChange={(e) => handleModelChange(e.target.value)}
                className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
              >
                <option value="abenzerps/Qwen-Image-2.1-GGUF">
                  abenzerps/Qwen-Image-2.1-GGUF (Q4_K_M ComfyUI)
                </option>
                <option value="Qwen/Qwen-Image">
                  Qwen/Qwen-Image (High Detail)
                </option>
                <option value="black-forest-labs/FLUX.1-schnell">
                  black-forest-labs/FLUX.1-schnell (Fast Alias)
                </option>
              </select>
            </div>
          )}

          <div>
            <label className="block text-[11px] font-mono uppercase tracking-wider text-ink-muted mb-1">
              Prompt
            </label>
            <textarea
              rows={3}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-sans resize-none"
            />
          </div>

          <ParameterAccordion title="Resolution & Diffusion Steps" defaultOpen>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10px] font-mono text-ink-muted mb-1">
                  Size
                </label>
                <select
                  value={size}
                  onChange={(e) => setSize(e.target.value)}
                  className="w-full px-2 py-1.5 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
                >
                  <option value="512x512">512x512 (1:1 Fast)</option>
                  <option value="768x768">768x768 (1:1 Medium)</option>
                  <option value="1024x1024">1024x1024 (1:1 High)</option>
                  <option value="1024x576">1024x576 (16:9)</option>
                  <option value="576x1024">576x1024 (9:16)</option>
                </select>
              </div>
              <div>
                <label className="block text-[10px] font-mono text-ink-muted mb-1">
                  Steps ({steps})
                </label>
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={steps}
                  onChange={(e) => setSteps(parseInt(e.target.value, 10) || 4)}
                  className="w-full px-2.5 py-1.5 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10px] font-mono text-ink-muted mb-1">
                  Guidance ({guidanceScale})
                </label>
                <input
                  type="number"
                  step={0.5}
                  min={0}
                  max={20}
                  value={guidanceScale}
                  onChange={(e) =>
                    setGuidanceScale(parseFloat(e.target.value) || 0)
                  }
                  className="w-full px-2.5 py-1.5 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
                />
              </div>
              <div>
                <label className="block text-[10px] font-mono text-ink-muted mb-1">
                  Seed (-1 = random)
                </label>
                <input
                  type="text"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
                />
              </div>
            </div>

            <div>
              <label className="block text-[10px] font-mono text-ink-muted mb-1">
                Negative Prompt
              </label>
              <input
                type="text"
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                placeholder="blurry, low quality, distorted..."
                className="w-full px-2.5 py-1.5 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
              />
            </div>
          </ParameterAccordion>

          <div className="space-y-2">
            <label className="flex items-center justify-between py-1.5 px-2.5 rounded-[6px] border border-hairline bg-subtle cursor-pointer">
              <span className="text-xs font-medium text-ink">
                SSE Step Progress Stream
              </span>
              <input
                type="checkbox"
                checked={streamSse}
                onChange={(e) => setStreamSse(e.target.checked)}
                className="accent-ink"
              />
            </label>

            {subTab === "t2i" && (
              <label className="flex items-center justify-between py-1.5 px-2.5 rounded-[6px] border border-hairline bg-subtle cursor-pointer">
                <span className="text-xs font-medium text-ink flex items-center gap-1.5">
                  <Columns weight="bold" className="w-3.5 h-3.5" />
                  <span>Compare FLUX vs Qwen-Image</span>
                </span>
                <input
                  type="checkbox"
                  checked={compareMode}
                  onChange={(e) => setCompareMode(e.target.checked)}
                  className="accent-ink"
                />
              </label>
            )}
          </div>
        </div>

        <button
          type="button"
          disabled={isGenerating}
          onClick={handleGenerate}
          className="mt-4 w-full py-2.5 rounded-[6px] bg-ink text-surface hover:bg-[#2a2a2a] disabled:opacity-50 text-xs font-medium flex items-center justify-center gap-2 transition-transform active:scale-[0.98]"
        >
          <PaintBrushBroad weight="bold" className="w-4 h-4" />
          <span>
            {isGenerating
              ? "Synthesizing..."
              : subTab === "i2i"
              ? "Run Image Edit / Inpaint"
              : "Synthesize Image"}
          </span>
        </button>
      </div>

      {/* Right Viewport Column */}
      <div className="lg:col-span-8 bg-surface border border-hairline rounded-[10px] flex flex-col justify-between overflow-hidden">
        <div className="p-5 overflow-y-auto flex-1 space-y-4">
          <ProgressStrip progress={progress} />

          {/* I2I & Inpainting Canvas Editor */}
          {subTab === "i2i" && (
            <div className="border border-hairline rounded-[8px] p-4 bg-subtle space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <label className="px-3 py-1.5 rounded-[6px] border border-hairline bg-surface hover:bg-canvas text-xs font-mono cursor-pointer flex items-center gap-1.5">
                    <UploadSimple weight="bold" className="w-3.5 h-3.5" />
                    <span>Upload Source Image</span>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleUploadSource}
                      className="hidden"
                    />
                  </label>

                  <label className="flex items-center gap-1.5 text-xs font-mono px-2.5 py-1.5 rounded-[6px] border border-hairline bg-surface cursor-pointer">
                    <input
                      type="checkbox"
                      checked={enableMaskBrush}
                      onChange={(e) => setEnableMaskBrush(e.target.checked)}
                      className="accent-ink"
                    />
                    <span>Inpaint Mask Brush</span>
                  </label>
                </div>

                <div className="flex items-center gap-3 text-xs font-mono">
                  <label className="flex items-center gap-1.5">
                    <span className="text-ink-muted">Brush:</span>
                    <input
                      type="range"
                      min={8}
                      max={80}
                      value={brushSize}
                      onChange={(e) => setBrushSize(parseInt(e.target.value, 10))}
                      className="w-20"
                    />
                  </label>
                  <label className="flex items-center gap-1.5">
                    <span className="text-ink-muted">Strength ({strength}):</span>
                    <input
                      type="range"
                      min={0.1}
                      max={1.0}
                      step={0.05}
                      value={strength}
                      onChange={(e) => setStrength(parseFloat(e.target.value))}
                      className="w-20"
                    />
                  </label>
                  <button
                    type="button"
                    onClick={clearMaskCanvas}
                    className="px-2.5 py-1 rounded-[5px] border border-hairline bg-surface hover:bg-canvas flex items-center gap-1"
                  >
                    <Eraser weight="bold" className="w-3.5 h-3.5" />
                    <span>Clear Mask</span>
                  </button>
                </div>
              </div>

              {sourceImageUri ? (
                <div className="relative w-[320px] h-[320px] mx-auto border border-hairline rounded-[6px] overflow-hidden bg-canvas">
                  <canvas
                    ref={bgCanvasRef}
                    className="absolute inset-0 w-full h-full"
                  />
                  <canvas
                    ref={maskCanvasRef}
                    onMouseDown={(e) => {
                      isDrawingRef.current = true;
                      paintAtEvent(e);
                    }}
                    onMouseMove={paintAtEvent}
                    onMouseUp={() => {
                      isDrawingRef.current = false;
                    }}
                    onMouseLeave={() => {
                      isDrawingRef.current = false;
                    }}
                    className={`absolute inset-0 w-full h-full ${
                      enableMaskBrush ? "cursor-crosshair" : "pointer-events-none"
                    }`}
                  />
                </div>
              ) : (
                <div className="py-8 text-center text-xs font-mono text-ink-muted">
                  Upload a source image above to edit or paint an inpainting mask.
                </div>
              )}
            </div>
          )}

          {/* Output Canvas Display */}
          {compareOutput ? (
            <div className="space-y-3">
              <div className="text-xs font-mono text-ink-secondary px-1">
                {compareOutput.status}
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="border border-hairline rounded-[8px] p-3 bg-canvas space-y-2">
                  <div className="flex justify-between text-[11px] font-mono">
                    <span className="px-2 py-0.5 rounded-full bg-spot-blueBg text-spot-blueText">
                      FLUX.1-schnell
                    </span>
                    <span className="text-ink-muted">{compareOutput.fluxTime}s</span>
                  </div>
                  {compareOutput.fluxB64 ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={`data:image/png;base64,${compareOutput.fluxB64}`}
                      alt="FLUX output"
                      className="w-full rounded-[6px] border border-hairline"
                    />
                  ) : (
                    <div className="h-56 flex items-center justify-center text-xs font-mono text-ink-muted">
                      Waiting...
                    </div>
                  )}
                </div>

                <div className="border border-hairline rounded-[8px] p-3 bg-canvas space-y-2">
                  <div className="flex justify-between text-[11px] font-mono">
                    <span className="px-2 py-0.5 rounded-full bg-spot-yellowBg text-spot-yellowText">
                      Qwen/Qwen-Image
                    </span>
                    <span className="text-ink-muted">{compareOutput.qwenTime}s</span>
                  </div>
                  {compareOutput.qwenB64 ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={`data:image/png;base64,${compareOutput.qwenB64}`}
                      alt="Qwen-Image output"
                      className="w-full rounded-[6px] border border-hairline"
                    />
                  ) : (
                    <div className="h-56 flex items-center justify-center text-xs font-mono text-ink-muted">
                      Queued...
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : outputB64 ? (
            <div className="space-y-3 text-center">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`data:image/png;base64,${outputB64}`}
                alt="Synthesized result"
                className="max-h-[460px] mx-auto rounded-[8px] border border-hairline object-contain bg-canvas"
              />
              <div className="flex justify-center">
                <a
                  href={`data:image/png;base64,${outputB64}`}
                  download={`studio-image-${Date.now()}.png`}
                  className="px-3.5 py-1.5 rounded-[6px] border border-hairline bg-subtle hover:bg-muted text-xs font-mono text-ink inline-flex items-center gap-1.5"
                >
                  <DownloadSimple weight="bold" className="w-3.5 h-3.5" />
                  <span>Download PNG</span>
                </a>
              </div>
            </div>
          ) : (
            <div className="h-72 flex flex-col items-center justify-center text-center">
              <p className="font-serif text-2xl text-ink mb-1">
                Diffusion Image Canvas
              </p>
              <p className="text-xs text-ink-muted font-mono">
                FLUX.1-schnell (GPU 0) &middot; Qwen-Image &amp; Qwen-Image-Edit
              </p>
            </div>
          )}
        </div>

        <TelemetryFooter telemetry={telemetry} />
      </div>
    </div>
  );
}
