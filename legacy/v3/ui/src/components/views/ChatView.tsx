"use client";

import React, { useState, useRef } from "react";
import {
  Trash,
  Image as ImageIcon,
  X,
  PaperPlaneRight,
  Columns,
  CaretDown,
  CaretRight,
} from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";
import { ParameterAccordion } from "@/components/ui/ParameterAccordion";
import { TelemetryFooter } from "@/components/ui/TelemetryFooter";
import { fileToDataUri } from "@/lib/utils";
import type { ChatMessageContentPart, TelemetryMeta } from "@/types/studio";

interface ChatTurn {
  id: string;
  role: "user" | "assistant";
  text: string;
  imageUri?: string | null;
  reasoning?: string;
  elapsedSec?: string;
  model?: string;
}

export function ChatView() {
  const { apiClient, refreshHealthAndMemory } = useStudio();

  // Parameters
  const [model, setModel] = useState("qwen3.8-27b-vlm");
  const [enableThinking, setEnableThinking] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const [streamSse, setStreamSse] = useState(true);
  const [systemPrompt, setSystemPrompt] = useState(
    "You are a precise, helpful multimodal engineering assistant."
  );
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.95);
  const [maxTokens, setMaxTokens] = useState(2048);

  // Conversation state
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [promptInput, setPromptInput] = useState("");
  const [attachedImageUri, setAttachedImageUri] = useState<string | null>(null);
  const [attachedImageName, setAttachedImageName] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [telemetry, setTelemetry] = useState<TelemetryMeta | null>(null);

  // Compare mode results
  const [compareResult, setCompareResult] = useState<{
    prompt: string;
    runA: { text: string; elapsed: string; status: string };
    runB: { text: string; elapsed: string; status: string };
  } | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleAttachFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const uri = await fileToDataUri(file);
    setAttachedImageUri(uri);
    setAttachedImageName(file.name);
  };

  const clearAttachment = () => {
    setAttachedImageUri(null);
    setAttachedImageName("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const clearHistory = () => {
    setTurns([]);
    setCompareResult(null);
    setTelemetry(null);
  };

  const handleSend = async () => {
    const trimmed = promptInput.trim();
    if (!trimmed && !attachedImageUri) return;
    if (isGenerating) return;

    const currentImage = attachedImageUri;
    setPromptInput("");
    clearAttachment();
    setIsGenerating(true);

    const buildMessages = () => {
      const msgs: Array<{
        role: "system" | "user" | "assistant";
        content: string | ChatMessageContentPart[];
      }> = [];

      if (systemPrompt.trim()) {
        msgs.push({ role: "system", content: systemPrompt.trim() });
      }

      for (const t of turns) {
        if (t.role === "user" && t.imageUri) {
          msgs.push({
            role: "user",
            content: [
              { type: "image_url", image_url: { url: t.imageUri } },
              { type: "text", text: t.text },
            ],
          });
        } else {
          msgs.push({ role: t.role, content: t.text });
        }
      }

      if (currentImage) {
        msgs.push({
          role: "user",
          content: [
            { type: "image_url", image_url: { url: currentImage } },
            { type: "text", text: trimmed || "Describe this image." },
          ],
        });
      } else {
        msgs.push({ role: "user", content: trimmed });
      }

      return msgs;
    };

    const messagesPayload = buildMessages();

    if (compareMode) {
      setCompareResult({
        prompt: trimmed,
        runA: { text: "", elapsed: "0.0", status: "Running Pass #1 (temp=0.2)..." },
        runB: { text: "", elapsed: "0.0", status: "Queued (Sequential VRAM execution)" },
      });

      const t0 = performance.now();
      try {
        // Sequential Run A
        let textA = "";
        const startA = performance.now();
        await apiClient.streamChatCompletion(
          {
            model,
            messages: messagesPayload,
            temperature: 0.2,
            top_p: topP,
            max_tokens: maxTokens,
            enable_thinking: enableThinking,
            stream: streamSse,
          },
          (delta) => {
            textA += delta;
            setCompareResult((prev) =>
              prev
                ? {
                    ...prev,
                    runA: {
                      text: textA,
                      elapsed: ((performance.now() - startA) / 1000).toFixed(2),
                      status: "Streaming...",
                    },
                  }
                : null
            );
          }
        );
        const elapsedA = ((performance.now() - startA) / 1000).toFixed(2);

        // Sequential Run B
        let textB = "";
        const startB = performance.now();
        setCompareResult((prev) =>
          prev
            ? {
                ...prev,
                runA: { text: textA, elapsed: elapsedA, status: "Done" },
                runB: {
                  text: "",
                  elapsed: "0.0",
                  status: "Running Pass #2 (temp=0.9)...",
                },
              }
            : null
        );

        await apiClient.streamChatCompletion(
          {
            model,
            messages: messagesPayload,
            temperature: 0.9,
            top_p: topP,
            max_tokens: maxTokens,
            enable_thinking: enableThinking,
            stream: streamSse,
          },
          (delta) => {
            textB += delta;
            setCompareResult((prev) =>
              prev
                ? {
                    ...prev,
                    runB: {
                      text: textB,
                      elapsed: ((performance.now() - startB) / 1000).toFixed(2),
                      status: "Streaming...",
                    },
                  }
                : null
            );
          }
        );
        const elapsedB = ((performance.now() - startB) / 1000).toFixed(2);
        setCompareResult((prev) =>
          prev
            ? {
                ...prev,
                runB: { text: textB, elapsed: elapsedB, status: "Done" },
              }
            : null
        );

        setTelemetry({
          endpoint: "POST /v1/chat/completions (Sequential Compare)",
          model,
          extra: `Pass A: ${elapsedA}s | Pass B: ${elapsedB}s`,
          elapsedSec: ((performance.now() - t0) / 1000).toFixed(2),
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Compare error";
        setCompareResult((prev) =>
          prev
            ? {
                ...prev,
                runB: { ...prev.runB, status: `Error: ${msg}` },
              }
            : null
        );
      } finally {
        setIsGenerating(false);
        refreshHealthAndMemory(true);
      }
      return;
    }

    // Standard Single-Turn / Multi-Turn execution
    setCompareResult(null);
    const userTurnId = `user-${Date.now()}`;
    const assistantTurnId = `assistant-${Date.now() + 1}`;

    setTurns((prev) => [
      ...prev,
      {
        id: userTurnId,
        role: "user",
        text: trimmed || "(Image attached)",
        imageUri: currentImage,
      },
      {
        id: assistantTurnId,
        role: "assistant",
        text: "",
        reasoning: "",
        model,
      },
    ]);

    const t0 = performance.now();
    try {
      let accText = "";
      let accReasoning = "";
      const result = await apiClient.streamChatCompletion(
        {
          model,
          messages: messagesPayload,
          temperature,
          top_p: topP,
          max_tokens: maxTokens,
          enable_thinking: enableThinking,
          stream: streamSse,
        },
        (deltaText, deltaReasoning) => {
          accText += deltaText;
          if (deltaReasoning) accReasoning += deltaReasoning;
          setTurns((prev) =>
            prev.map((turn) =>
              turn.id === assistantTurnId
                ? { ...turn, text: accText, reasoning: accReasoning }
                : turn
            )
          );
        }
      );

      const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
      setTurns((prev) =>
        prev.map((turn) =>
          turn.id === assistantTurnId
            ? {
                ...turn,
                text: result.content || accText,
                reasoning: result.reasoning || accReasoning,
                elapsedSec: elapsed,
              }
            : turn
        )
      );

      setTelemetry({
        endpoint: "POST /v1/chat/completions",
        model,
        extra: `thinking=${enableThinking} | stream=${streamSse}`,
        elapsedSec: elapsed,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Request failed";
      setTurns((prev) =>
        prev.map((turn) =>
          turn.id === assistantTurnId
            ? { ...turn, text: `[Error] ${msg}` }
            : turn
        )
      );
    } finally {
      setIsGenerating(false);
      refreshHealthAndMemory(true);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 h-[calc(100vh-5.5rem)]">
      {/* Left Column: Parameters */}
      <div className="lg:col-span-4 bg-surface border border-hairline rounded-[10px] p-5 flex flex-col justify-between overflow-y-auto">
        <div className="space-y-4">
          <div className="border-b border-hairline pb-3">
            <span className="text-[10px] font-mono uppercase tracking-widest text-ink-muted">
              POST /v1/chat/completions
            </span>
            <h2 className="font-serif text-2xl text-ink mt-0.5">
              Chat &amp; Thinking
            </h2>
          </div>

          <div>
            <label className="block text-[11px] font-mono uppercase tracking-wider text-ink-muted mb-1">
              Model Checkpoint
            </label>
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
            />
          </div>

          {/* Toggles */}
          <div className="space-y-2 pt-1">
            <label className="flex items-center justify-between py-1.5 px-2.5 rounded-[6px] border border-hairline bg-subtle cursor-pointer">
              <span className="text-xs font-medium text-ink">
                Enable Thinking Trace
              </span>
              <input
                type="checkbox"
                checked={enableThinking}
                onChange={(e) => setEnableThinking(e.target.checked)}
                className="accent-ink"
              />
            </label>

            <label className="flex items-center justify-between py-1.5 px-2.5 rounded-[6px] border border-hairline bg-subtle cursor-pointer">
              <span className="text-xs font-medium text-ink flex items-center gap-1.5">
                <Columns weight="bold" className="w-3.5 h-3.5" />
                <span>Sequential Compare (0.2 vs 0.9)</span>
              </span>
              <input
                type="checkbox"
                checked={compareMode}
                onChange={(e) => setCompareMode(e.target.checked)}
                className="accent-ink"
              />
            </label>

            <label className="flex items-center justify-between py-1.5 px-2.5 rounded-[6px] border border-hairline bg-subtle cursor-pointer">
              <span className="text-xs font-medium text-ink">
                SSE Token Streaming
              </span>
              <input
                type="checkbox"
                checked={streamSse}
                onChange={(e) => setStreamSse(e.target.checked)}
                className="accent-ink"
              />
            </label>
          </div>

          <ParameterAccordion title="System Prompt" defaultOpen>
            <textarea
              rows={3}
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              className="w-full px-2.5 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-sans resize-none"
            />
          </ParameterAccordion>

          <ParameterAccordion title="Sampling Parameters" defaultOpen>
            <div>
              <div className="flex justify-between text-[11px] font-mono mb-1">
                <span className="text-ink-muted">temperature</span>
                <span className="text-ink font-medium">{temperature}</span>
              </div>
              <input
                type="range"
                min={0}
                max={1.5}
                step={0.05}
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <div className="flex justify-between text-[11px] font-mono mb-1">
                <span className="text-ink-muted">top_p</span>
                <span className="text-ink font-medium">{topP}</span>
              </div>
              <input
                type="range"
                min={0.1}
                max={1.0}
                step={0.05}
                value={topP}
                onChange={(e) => setTopP(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <div className="flex justify-between text-[11px] font-mono mb-1">
                <span className="text-ink-muted">max_tokens</span>
                <span className="text-ink font-medium">{maxTokens}</span>
              </div>
              <input
                type="number"
                min={64}
                max={8192}
                step={64}
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value, 10) || 1024)}
                className="w-full px-2.5 py-1.5 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
              />
            </div>
          </ParameterAccordion>
        </div>

        <button
          type="button"
          onClick={clearHistory}
          className="mt-4 w-full px-3 py-2 rounded-[6px] border border-hairline bg-subtle hover:bg-muted text-xs font-mono text-ink flex items-center justify-center gap-1.5"
        >
          <Trash weight="bold" className="w-3.5 h-3.5" />
          <span>Clear Conversation History</span>
        </button>
      </div>

      {/* Right Column: Transcript Canvas & Input */}
      <div className="lg:col-span-8 bg-surface border border-hairline rounded-[10px] flex flex-col justify-between overflow-hidden">
        <div className="p-5 overflow-y-auto flex-1 space-y-4">
          {compareResult ? (
            <div className="space-y-3">
              <div className="border border-hairline rounded-[8px] p-3 bg-subtle text-xs">
                <span className="font-mono text-[10px] uppercase text-ink-muted block mb-1">
                  Comparison Prompt
                </span>
                <p className="text-ink">{compareResult.prompt}</p>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="border border-hairline rounded-[8px] p-4 bg-canvas space-y-2">
                  <div className="flex items-center justify-between border-b border-hairline pb-2 text-[11px] font-mono">
                    <span className="px-2 py-0.5 rounded-full bg-spot-blueBg text-spot-blueText">
                      RUN #1 &middot; temp=0.2
                    </span>
                    <span className="text-ink-muted">
                      {compareResult.runA.status} ({compareResult.runA.elapsed}s)
                    </span>
                  </div>
                  <div className="text-xs leading-relaxed whitespace-pre-wrap text-ink">
                    {compareResult.runA.text || "..."}
                  </div>
                </div>

                <div className="border border-hairline rounded-[8px] p-4 bg-canvas space-y-2">
                  <div className="flex items-center justify-between border-b border-hairline pb-2 text-[11px] font-mono">
                    <span className="px-2 py-0.5 rounded-full bg-spot-yellowBg text-spot-yellowText">
                      RUN #2 &middot; temp=0.9
                    </span>
                    <span className="text-ink-muted">
                      {compareResult.runB.status} ({compareResult.runB.elapsed}s)
                    </span>
                  </div>
                  <div className="text-xs leading-relaxed whitespace-pre-wrap text-ink">
                    {compareResult.runB.text || "..."}
                  </div>
                </div>
              </div>
            </div>
          ) : turns.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center py-16">
              <p className="font-serif text-2xl text-ink mb-1">
                Qwen3-VL Multimodal Reasoning Workspace
              </p>
              <p className="text-xs text-ink-muted max-w-md">
                Send text or attach an inline image. Toggle{" "}
                <code className="font-mono bg-subtle px-1 py-0.5 border border-hairline rounded-[3px]">
                  enable_thinking
                </code>{" "}
                to inspect chain-of-thought reasoning blocks.
              </p>
            </div>
          ) : (
            turns.map((turn) => (
              <ChatTurnBubble key={turn.id} turn={turn} />
            ))
          )}
        </div>

        {/* Input Composer */}
        <div className="p-4 border-t border-hairline bg-subtle space-y-2">
          {attachedImageUri && (
            <div className="flex items-center justify-between px-3 py-1.5 rounded-[6px] border border-hairline bg-surface text-xs font-mono">
              <span className="truncate max-w-xs">
                Attached: {attachedImageName}
              </span>
              <button
                type="button"
                onClick={clearAttachment}
                className="text-spot-redText hover:underline flex items-center gap-1"
              >
                <X weight="bold" className="w-3.5 h-3.5" />
                <span>Remove</span>
              </button>
            </div>
          )}

          <div className="flex items-end gap-2">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleAttachFile}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              title="Attach image for multimodal chat"
              className="p-2.5 rounded-[6px] border border-hairline bg-surface hover:bg-canvas text-ink transition-transform active:scale-[0.98]"
            >
              <ImageIcon weight="bold" className="w-4 h-4" />
            </button>

            <textarea
              rows={2}
              value={promptInput}
              onChange={(e) => setPromptInput(e.target.value)}
              onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Enter prompt... (Press Cmd/Ctrl + Enter to send)"
              className="flex-1 px-3 py-2 rounded-[6px] border border-hairline bg-surface text-xs font-sans resize-none"
            />

            <button
              type="button"
              disabled={isGenerating}
              onClick={handleSend}
              className="px-4 py-2.5 rounded-[6px] bg-ink text-surface hover:bg-[#2a2a2a] disabled:opacity-50 text-xs font-medium flex items-center gap-2 transition-transform active:scale-[0.98]"
            >
              <PaperPlaneRight weight="bold" className="w-3.5 h-3.5" />
              <span>{isGenerating ? "Running..." : "Send"}</span>
              <kbd className="px-1 py-0.2 text-[10px] font-mono bg-[#2F3437] text-[#EAEAEA] rounded-[3px] border border-[#444]">
                &crarr;
              </kbd>
            </button>
          </div>
        </div>

        <TelemetryFooter telemetry={telemetry} />
      </div>
    </div>
  );
}

function ChatTurnBubble({ turn }: { turn: ChatTurn }) {
  const [reasoningOpen, setReasoningOpen] = useState(true);

  return (
    <div
      className={`border border-hairline rounded-[8px] p-4 ${
        turn.role === "user" ? "bg-subtle" : "bg-surface"
      }`}
    >
      <div className="flex items-center justify-between text-[11px] font-mono text-ink-muted mb-2 border-b border-hairline pb-1.5">
        <span className="uppercase font-semibold text-ink">
          {turn.role === "user" ? "User" : `Assistant (${turn.model || "LLM"})`}
        </span>
        {turn.elapsedSec && <span>{turn.elapsedSec}s</span>}
      </div>

      {turn.imageUri && (
        <div className="mb-3">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={turn.imageUri}
            alt="User attachment"
            className="max-h-48 rounded-[6px] border border-hairline object-contain"
          />
        </div>
      )}

      {turn.reasoning && (
        <div className="mb-3 border border-hairline rounded-[6px] bg-canvas p-2.5">
          <button
            type="button"
            onClick={() => setReasoningOpen((o) => !o)}
            className="w-full flex items-center justify-between text-[11px] font-mono text-ink-muted"
          >
            <span>Chain-of-Thought Reasoning Trace</span>
            {reasoningOpen ? (
              <CaretDown weight="bold" className="w-3.5 h-3.5" />
            ) : (
              <CaretRight weight="bold" className="w-3.5 h-3.5" />
            )}
          </button>
          {reasoningOpen && (
            <pre className="mt-2 text-[11px] font-mono text-ink-secondary whitespace-pre-wrap leading-relaxed">
              {turn.reasoning}
            </pre>
          )}
        </div>
      )}

      <div className="text-xs text-ink leading-relaxed whitespace-pre-wrap">
        {turn.text || "..."}
      </div>
    </div>
  );
}
