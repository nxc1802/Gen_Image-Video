"use client";

import React, { useState, useRef } from "react";
import { UploadSimple, Eye, LinkSimple } from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";
import { TelemetryFooter } from "@/components/ui/TelemetryFooter";
import { fileToDataUri } from "@/lib/utils";
import type { TelemetryMeta } from "@/types/studio";

const VISION_PRESETS = [
  {
    label: "OCR & Structured Markdown",
    prompt:
      "Extract all visible text in this image with exact layout hierarchy formatted in clean Markdown.",
  },
  {
    label: "UI / Architecture Audit",
    prompt:
      "Analyze this interface or diagram. Identify visual hierarchy, component structure, and layout details.",
  },
  {
    label: "Dense Scene Caption",
    prompt:
      "Provide a comprehensive description of the subject, lighting, camera composition, and materials.",
  },
];

export function VisionView() {
  const { apiClient, refreshHealthAndMemory } = useStudio();

  const [imageUri, setImageUri] = useState<string>("");
  const [imageUrlInput, setImageUrlInput] = useState<string>("");
  const [prompt, setPrompt] = useState<string>(VISION_PRESETS[0].prompt);
  const [enableThinking, setEnableThinking] = useState<boolean>(false);
  const [responseText, setResponseText] = useState<string>("");
  const [reasoningText, setReasoningText] = useState<string>("");
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [telemetry, setTelemetry] = useState<TelemetryMeta | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const uri = await fileToDataUri(file);
    setImageUri(uri);
  };

  const handleApplyUrl = () => {
    if (imageUrlInput.trim()) {
      setImageUri(imageUrlInput.trim());
    }
  };

  const handleAnalyze = async () => {
    if (!imageUri) {
      setResponseText("[Validation] Please upload an image or provide an image URL first.");
      return;
    }
    if (isRunning) return;

    setIsRunning(true);
    setResponseText("");
    setReasoningText("");
    const t0 = performance.now();

    try {
      let accText = "";
      let accReasoning = "";
      const res = await apiClient.streamChatCompletion(
        {
          model: "qwen3.8-27b-vlm",
          messages: [
            {
              role: "user",
              content: [
                { type: "image_url", image_url: { url: imageUri } },
                { type: "text", text: prompt },
              ],
            },
          ],
          enable_thinking: enableThinking,
          stream: true,
        },
        (deltaText, deltaReasoning) => {
          accText += deltaText;
          if (deltaReasoning) accReasoning += deltaReasoning;
          setResponseText(accText);
          setReasoningText(accReasoning);
        }
      );

      const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
      setResponseText(res.content || accText);
      setReasoningText(res.reasoning || accReasoning);
      setTelemetry({
        endpoint: "POST /v1/chat/completions (Multimodal Vision)",
        model: "qwen3.8-27b-vlm",
        extra: `thinking=${enableThinking}`,
        elapsedSec: elapsed,
      });
    } catch (err) {
      setResponseText(
        `[Error] ${err instanceof Error ? err.message : "Vision request failed"}`
      );
    } finally {
      setIsRunning(false);
      refreshHealthAndMemory(true);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 h-[calc(100vh-5.5rem)]">
      {/* Left Column: Image Dropzone & Prompt Controls */}
      <div className="lg:col-span-5 bg-surface border border-hairline rounded-[10px] p-5 flex flex-col justify-between overflow-y-auto">
        <div className="space-y-4">
          <div className="border-b border-hairline pb-3">
            <span className="text-[10px] font-mono uppercase tracking-widest text-ink-muted">
              POST /v1/chat/completions (image_url)
            </span>
            <h2 className="font-serif text-2xl text-ink mt-0.5">
              Vision Studio
            </h2>
          </div>

          {/* Upload Box */}
          <div className="border border-dashed border-[#CCCCCC] rounded-[8px] p-4 bg-subtle text-center space-y-2">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
            {imageUri ? (
              <div className="space-y-2">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={imageUri}
                  alt="Vision target"
                  className="max-h-48 mx-auto rounded-[6px] border border-hairline object-contain bg-surface"
                />
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="px-3 py-1 rounded-[6px] border border-hairline bg-surface text-xs font-mono"
                >
                  Replace Image
                </button>
              </div>
            ) : (
              <div className="py-6 space-y-2">
                <UploadSimple weight="bold" className="w-6 h-6 mx-auto text-ink-muted" />
                <p className="text-xs text-ink-secondary">
                  Upload a local image (converted to Base64 Data URI)
                </p>
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="px-3 py-1.5 rounded-[6px] border border-hairline bg-surface hover:bg-canvas text-xs font-mono text-ink"
                >
                  Select Image File
                </button>
              </div>
            )}
          </div>

          {/* Remote URL Input */}
          <div>
            <label className="block text-[11px] font-mono uppercase tracking-wider text-ink-muted mb-1">
              Or Remote Image URL
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={imageUrlInput}
                onChange={(e) => setImageUrlInput(e.target.value)}
                placeholder="https://example.com/diagram.png"
                className="flex-1 px-3 py-1.5 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
              />
              <button
                type="button"
                onClick={handleApplyUrl}
                className="px-2.5 py-1.5 rounded-[6px] border border-hairline bg-subtle hover:bg-muted text-xs font-mono flex items-center gap-1"
              >
                <LinkSimple weight="bold" className="w-3.5 h-3.5" />
                <span>Load</span>
              </button>
            </div>
          </div>

          {/* Preset Prompt Chips */}
          <div>
            <span className="block text-[11px] font-mono uppercase tracking-wider text-ink-muted mb-1.5">
              Inspection Presets
            </span>
            <div className="flex flex-wrap gap-1.5">
              {VISION_PRESETS.map((preset) => (
                <button
                  key={preset.label}
                  type="button"
                  onClick={() => setPrompt(preset.prompt)}
                  className="px-2.5 py-1 rounded-[5px] border border-hairline bg-subtle hover:bg-muted text-[11px] font-mono text-ink"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-[11px] font-mono uppercase tracking-wider text-ink-muted mb-1">
              Vision Prompt
            </label>
            <textarea
              rows={4}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-sans resize-none"
            />
          </div>

          <label className="flex items-center justify-between py-1.5 px-2.5 rounded-[6px] border border-hairline bg-subtle cursor-pointer">
            <span className="text-xs font-medium text-ink">
              Enable Multimodal Thinking Trace
            </span>
            <input
              type="checkbox"
              checked={enableThinking}
              onChange={(e) => setEnableThinking(e.target.checked)}
              className="accent-ink"
            />
          </label>
        </div>

        <button
          type="button"
          disabled={isRunning}
          onClick={handleAnalyze}
          className="mt-4 w-full py-2.5 rounded-[6px] bg-ink text-surface hover:bg-[#2a2a2a] disabled:opacity-50 text-xs font-medium flex items-center justify-center gap-2 transition-transform active:scale-[0.98]"
        >
          <Eye weight="bold" className="w-4 h-4" />
          <span>{isRunning ? "Inspecting Image..." : "Run Vision Inspection"}</span>
        </button>
      </div>

      {/* Right Column: Structured Vision Output */}
      <div className="lg:col-span-7 bg-surface border border-hairline rounded-[10px] flex flex-col justify-between overflow-hidden">
        <div className="p-6 overflow-y-auto flex-1 space-y-4">
          <div className="border-b border-hairline pb-2 flex items-center justify-between">
            <span className="text-[11px] font-mono uppercase tracking-wider text-ink-muted">
              Vision Analysis Output
            </span>
            <span className="text-[11px] font-mono text-ink-muted">
              qwen3.8-27b-vlm
            </span>
          </div>

          {reasoningText && (
            <div className="border border-hairline rounded-[6px] bg-canvas p-3">
              <div className="text-[10px] font-mono uppercase text-ink-muted mb-1">
                Thinking Trace
              </div>
              <pre className="text-[11px] font-mono text-ink-secondary whitespace-pre-wrap">
                {reasoningText}
              </pre>
            </div>
          )}

          {responseText ? (
            <div className="text-xs text-ink leading-relaxed whitespace-pre-wrap font-sans">
              {responseText}
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-xs text-ink-muted font-mono">
              Upload an image and run inspection to view structured output.
            </div>
          )}
        </div>

        <TelemetryFooter telemetry={telemetry} />
      </div>
    </div>
  );
}
