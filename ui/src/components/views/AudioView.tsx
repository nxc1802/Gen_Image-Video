"use client";

import React, { useState, useRef } from "react";
import {
  Microphone,
  Stop,
  Waveform,
  SpeakerHigh,
  UploadSimple,
  DownloadSimple,
  Columns,
} from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";
import { TelemetryFooter } from "@/components/ui/TelemetryFooter";
import {
  fileToDataUri,
  normalizeAudioFileToWav,
  renderWaveformToCanvas,
} from "@/lib/utils";
import type { TranscriptionResponse, TelemetryMeta } from "@/types/studio";

type TtsModeTab = "preset" | "clone" | "design";

export function AudioView() {
  const { apiClient, refreshHealthAndMemory } = useStudio();

  // --- STT State ---
  const [sttFile, setSttFile] = useState<File | null>(null);
  const [sttPreviewUrl, setSttPreviewUrl] = useState<string | null>(null);
  const [sttModel, setSttModel] = useState("Qwen/Qwen3-ASR-1.7B");
  const [sttLang, setSttLang] = useState("vi");
  const [sttFormat, setSttFormat] = useState<"json" | "verbose_json">(
    "verbose_json"
  );
  // Default false because Qwen3-ASR requires ForcedAligner-0.6B when return_time_stamps=true
  const [sttTimestamps, setSttTimestamps] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [sttBusy, setSttBusy] = useState(false);
  const [sttError, setSttError] = useState<string | null>(null);
  const [sttResult, setSttResult] = useState<TranscriptionResponse | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);

  // --- TTS State ---
  const [ttsMode, setTtsMode] = useState<TtsModeTab>("preset");
  const [ttsText, setTtsText] = useState(
    "Xin chào, hệ thống âm thanh trí tuệ nhân tạo đang hoạt động trực tiếp trên Kaggle Dual T4."
  );
  const [ttsModel, setTtsModel] = useState("xtts-v2-vietnamese");
  const [ttsLang, setTtsLang] = useState("vi");
  const [ttsVoice, setTtsVoice] = useState("Vivian");
  const [ttsSpeed, setTtsSpeed] = useState(1.0);
  const [ttsInstruct, setTtsInstruct] = useState(
    "Speak with a calm, articulate, studio-grade broadcast tone."
  );
  const [refAudioUri, setRefAudioUri] = useState<string | null>(null);
  const [refAudioName, setRefAudioName] = useState<string>("");
  const [refText, setRefText] = useState<string>("");
  const [compareTts, setCompareTts] = useState(false);
  const [ttsBusy, setTtsBusy] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [compareAudio, setCompareAudio] = useState<{
    xttsUrl: string | null;
    xttsTime: string;
    qwenUrl: string | null;
    qwenTime: string;
    status: string;
  } | null>(null);

  const [telemetry, setTelemetry] = useState<TelemetryMeta | null>(null);
  const waveformCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const selectSttFile = (file: File | null) => {
    setSttFile(file);
    setSttError(null);
    if (file) {
      setSttPreviewUrl(URL.createObjectURL(file));
    } else {
      setSttPreviewUrl(null);
    }
  };

  const toggleMicRecording = async () => {
    if (isRecording) {
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordedChunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, {
          type: "audio/webm",
        });
        const file = new File([blob], `mic-recording-${Date.now()}.webm`, {
          type: "audio/webm",
        });
        selectSttFile(file);
        stream.getTracks().forEach((t) => t.stop());
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
    } catch (err) {
      setSttError(
        `Microphone access failed: ${
          err instanceof Error ? err.message : "Permission denied"
        }`
      );
    }
  };

  const handleRunTranscription = async () => {
    if (!sttFile || sttBusy) return;
    setSttBusy(true);
    setSttError(null);
    const t0 = performance.now();
    try {
      const res = await apiClient.transcribeAudio({
        file: sttFile,
        model: sttModel,
        language: sttLang,
        responseFormat: sttFormat,
        timestampGranularities: sttTimestamps,
      });
      setSttResult(res);
      setTelemetry({
        endpoint: "POST /v1/audio/transcriptions",
        model: res.model || sttModel,
        extra: `lang=${res.language || sttLang}`,
        elapsedSec: ((performance.now() - t0) / 1000).toFixed(2),
      });
    } catch (err) {
      setSttError(err instanceof Error ? err.message : "Transcription failed");
    } finally {
      setSttBusy(false);
      refreshHealthAndMemory(true);
    }
  };

  const handleUploadRefAudio = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const normalized = await normalizeAudioFileToWav(file, 24000);
    const uri = await fileToDataUri(normalized);
    setRefAudioUri(uri);
    setRefAudioName(file.name);
  };

  const handleSynthesizeSpeech = async () => {
    if (!ttsText.trim() || ttsBusy) return;
    setTtsBusy(true);
    const t0 = performance.now();

    if (compareTts) {
      setCompareAudio({
        xttsUrl: null,
        xttsTime: "0.0",
        qwenUrl: null,
        qwenTime: "0.0",
        status: "Pass #1: Synthesizing with XTTS-v2 Vietnamese...",
      });

      try {
        const startX = performance.now();
        const blobX = await apiClient.synthesizeSpeech({
          model: "xtts-v2-vietnamese",
          input: ttsText,
          voice: ttsVoice,
          language: ttsLang,
          speed: ttsSpeed,
          ref_audio: refAudioUri,
          ref_text: refText || null,
        });
        const xttsUrl = URL.createObjectURL(blobX);
        const xttsTime = ((performance.now() - startX) / 1000).toFixed(2);

        setCompareAudio({
          xttsUrl,
          xttsTime,
          qwenUrl: null,
          qwenTime: "0.0",
          status: "Pass #2: Synthesizing with Qwen3-TTS 1.7B...",
        });

        const startQ = performance.now();
        const blobQ = await apiClient.synthesizeSpeech({
          model: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
          input: ttsText,
          voice: ttsVoice,
          language: ttsLang,
          speed: ttsSpeed,
          instruct: ttsInstruct || null,
          ref_audio: ttsMode === "clone" ? refAudioUri : null,
          ref_text: ttsMode === "clone" ? refText || null : null,
        });
        const qwenUrl = URL.createObjectURL(blobQ);
        const qwenTime = ((performance.now() - startQ) / 1000).toFixed(2);
        await renderWaveformToCanvas(blobQ, waveformCanvasRef.current);

        setCompareAudio({
          xttsUrl,
          xttsTime,
          qwenUrl,
          qwenTime,
          status: "Completed Sequential TTS Comparison",
        });

        setTelemetry({
          endpoint: "POST /v1/audio/speech (Sequential Compare)",
          model: "XTTS-v2 vs Qwen3-TTS",
          elapsedSec: ((performance.now() - t0) / 1000).toFixed(2),
        });
      } catch (err) {
        setCompareAudio((prev) =>
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
        setTtsBusy(false);
        refreshHealthAndMemory(true);
      }
      return;
    }

    // Standard Single TTS Run
    setCompareAudio(null);
    try {
      const effectiveModel =
        ttsMode === "design"
          ? "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
          : ttsModel;

      const blob = await apiClient.synthesizeSpeech({
        model: effectiveModel,
        input: ttsText,
        voice: ttsVoice,
        language: ttsLang,
        speed: ttsSpeed,
        instruct: ttsMode !== "clone" ? ttsInstruct : null,
        ref_audio: ttsMode === "clone" ? refAudioUri : null,
        ref_text: ttsMode === "clone" ? refText || null : null,
      });

      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
      await renderWaveformToCanvas(blob, waveformCanvasRef.current);

      setTelemetry({
        endpoint: "POST /v1/audio/speech",
        model: effectiveModel,
        extra: `mode=${ttsMode} | lang=${ttsLang}`,
        elapsedSec: ((performance.now() - t0) / 1000).toFixed(2),
      });
    } catch (err) {
      alert(err instanceof Error ? err.message : "Speech synthesis failed");
    } finally {
      setTtsBusy(false);
      refreshHealthAndMemory(true);
    }
  };

  // Helper to send synthesized TTS audio directly into the STT panel for instant testing
  const useSynthesizedAudioForStt = async () => {
    if (!audioUrl) return;
    const r = await fetch(audioUrl);
    const blob = await r.blob();
    const file = new File([blob], "synthesized-speech.wav", {
      type: "audio/wav",
    });
    selectSttFile(file);
  };

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left: Speech-to-Text (STT) */}
        <div className="lg:col-span-6 bg-surface border border-hairline rounded-[10px] p-5 flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            <div className="border-b border-hairline pb-3">
              <span className="text-[10px] font-mono uppercase tracking-widest text-ink-muted">
                POST /v1/audio/transcriptions
              </span>
              <h2 className="font-serif text-2xl text-ink mt-0.5">
                Speech-to-Text (ASR)
              </h2>
            </div>

            {/* File or Mic Picker */}
            <div className="grid grid-cols-2 gap-2">
              <label className="py-2.5 px-3 rounded-[6px] border border-hairline bg-subtle hover:bg-muted text-xs font-mono text-ink flex items-center justify-center gap-1.5 cursor-pointer">
                <UploadSimple weight="bold" className="w-4 h-4" />
                <span className="truncate">
                  {sttFile ? sttFile.name : "Upload Audio File"}
                </span>
                <input
                  type="file"
                  accept="audio/*,.mp3,.wav,.m4a,.ogg,.flac,.webm"
                  onChange={(e) => selectSttFile(e.target.files?.[0] || null)}
                  className="hidden"
                />
              </label>

              <button
                type="button"
                onClick={toggleMicRecording}
                className={`py-2.5 px-3 rounded-[6px] border text-xs font-mono flex items-center justify-center gap-1.5 ${
                  isRecording
                    ? "bg-spot-redBg border-[#E6C5C5] text-spot-redText"
                    : "bg-subtle border-hairline text-ink hover:bg-muted"
                }`}
              >
                {isRecording ? (
                  <>
                    <Stop weight="bold" className="w-4 h-4" />
                    <span>Stop Recording</span>
                  </>
                ) : (
                  <>
                    <Microphone weight="bold" className="w-4 h-4" />
                    <span>Record Microphone</span>
                  </>
                )}
              </button>
            </div>

            {sttPreviewUrl && (
              <div className="p-2 rounded-[6px] border border-hairline bg-canvas space-y-1">
                <div className="flex items-center justify-between text-[10px] font-mono text-ink-muted">
                  <span className="truncate">{sttFile?.name}</span>
                  <span>Auto-normalized to 16kHz PCM WAV</span>
                </div>
                <audio src={sttPreviewUrl} controls className="w-full h-8" />
              </div>
            )}

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10px] font-mono uppercase text-ink-muted mb-1">
                  ASR Engine
                </label>
                <select
                  value={sttModel}
                  onChange={(e) => setSttModel(e.target.value)}
                  className="w-full px-2.5 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
                >
                  <option value="Qwen/Qwen3-ASR-1.7B">
                    Qwen/Qwen3-ASR-1.7B
                  </option>
                  <option value="whisper-large-v3-turbo">
                    whisper-large-v3-turbo
                  </option>
                </select>
              </div>
              <div>
                <label className="block text-[10px] font-mono uppercase text-ink-muted mb-1">
                  Language Hint
                </label>
                <select
                  value={sttLang}
                  onChange={(e) => setSttLang(e.target.value)}
                  className="w-full px-2.5 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
                >
                  <option value="vi">vi (Vietnamese)</option>
                  <option value="en">en (English)</option>
                  <option value="auto">auto-detect</option>
                  <option value="zh">zh (Chinese)</option>
                  <option value="ja">ja (Japanese)</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10px] font-mono uppercase text-ink-muted mb-1">
                  Response Format
                </label>
                <select
                  value={sttFormat}
                  onChange={(e) =>
                    setSttFormat(e.target.value as "json" | "verbose_json")
                  }
                  className="w-full px-2.5 py-1.5 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
                >
                  <option value="verbose_json">
                    verbose_json (with metadata)
                  </option>
                  <option value="json">json</option>
                </select>
              </div>
              <div className="flex items-end">
                <label className="w-full flex items-center justify-between py-1.5 px-2.5 rounded-[6px] border border-hairline bg-subtle text-xs font-mono cursor-pointer">
                  <span>Timestamps (Aligner)</span>
                  <input
                    type="checkbox"
                    checked={sttTimestamps}
                    onChange={(e) => setSttTimestamps(e.target.checked)}
                    className="accent-ink"
                  />
                </label>
              </div>
            </div>

            <button
              type="button"
              disabled={!sttFile || sttBusy}
              onClick={handleRunTranscription}
              className="w-full py-2.5 rounded-[6px] bg-ink text-surface hover:bg-[#2a2a2a] disabled:opacity-50 text-xs font-medium flex items-center justify-center gap-2 transition-transform active:scale-[0.98]"
            >
              <Waveform weight="bold" className="w-4 h-4" />
              <span>{sttBusy ? "Transcribing..." : "Transcribe Audio"}</span>
            </button>

            {/* Transcript Display */}
            <div className="border border-hairline rounded-[8px] p-3.5 bg-canvas min-h-[180px] space-y-2">
              <div className="flex justify-between text-[10px] font-mono uppercase text-ink-muted border-b border-hairline pb-1.5">
                <span>Transcript Output</span>
                {sttResult?.language && <span>lang: {sttResult.language}</span>}
              </div>
              {sttError && (
                <div className="text-[11px] font-mono text-spot-redText bg-spot-redBg border border-[#E6C5C5] px-2.5 py-2 rounded-[5px]">
                  {sttError}
                </div>
              )}
              {sttResult ? (
                <div className="space-y-3">
                  <p className="text-xs text-ink leading-relaxed whitespace-pre-wrap">
                    {sttResult.text}
                  </p>
                  {sttResult.notice && (
                    <div className="text-[11px] font-mono text-spot-yellowText bg-spot-yellowBg px-2.5 py-1.5 rounded-[5px]">
                      {sttResult.notice}
                    </div>
                  )}
                  {sttResult.segments && sttResult.segments.length > 0 && (
                    <div className="space-y-1 pt-2 border-t border-hairline max-h-40 overflow-y-auto">
                      {sttResult.segments.map((seg, idx) => (
                        <div
                          key={idx}
                          className="text-[11px] font-mono text-ink-secondary flex gap-2"
                        >
                          <span className="text-ink-muted shrink-0">
                            [{seg.start.toFixed(1)}s - {seg.end.toFixed(1)}s]
                          </span>
                          <span>{seg.text}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="py-10 text-center text-xs font-mono text-ink-muted">
                  Upload or record audio (or synthesize TTS on the right and
                  click &ldquo;Send to ASR&rdquo;) to transcribe.
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right: Text-to-Speech (TTS) */}
        <div className="lg:col-span-6 bg-surface border border-hairline rounded-[10px] p-5 flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            <div className="border-b border-hairline pb-3">
              <span className="text-[10px] font-mono uppercase tracking-widest text-ink-muted">
                POST /v1/audio/speech
              </span>
              <h2 className="font-serif text-2xl text-ink mt-0.5">
                Speech Synthesis &amp; Voice Cloning
              </h2>
            </div>

            {/* 3 Mode Sub-tabs */}
            <div className="grid grid-cols-3 gap-1.5 p-1 rounded-[6px] bg-subtle border border-hairline">
              {(["preset", "clone", "design"] as TtsModeTab[]).map((tab) => (
                <button
                  key={tab}
                  type="button"
                  onClick={() => setTtsMode(tab)}
                  className={`py-1.5 rounded-[5px] text-xs font-mono uppercase ${
                    ttsMode === tab
                      ? "bg-surface text-ink font-semibold border border-hairline"
                      : "text-ink-muted"
                  }`}
                >
                  {tab === "preset"
                    ? "Preset Voice"
                    : tab === "clone"
                    ? "Voice Clone"
                    : "Voice Design"}
                </button>
              ))}
            </div>

            <div>
              <label className="block text-[10px] font-mono uppercase text-ink-muted mb-1">
                Input Script
              </label>
              <textarea
                rows={3}
                value={ttsText}
                onChange={(e) => setTtsText(e.target.value)}
                className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-sans resize-none"
              />
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              <div>
                <label className="block text-[10px] font-mono uppercase text-ink-muted mb-1">
                  TTS Engine
                </label>
                <select
                  value={ttsModel}
                  onChange={(e) => setTtsModel(e.target.value)}
                  className="w-full px-2 py-1.5 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
                >
                  <option value="xtts-v2-vietnamese">
                    XTTS-v2 Vietnamese
                  </option>
                  <option value="Qwen/Qwen3-TTS-12Hz-1.7B-Base">
                    Qwen3-TTS 1.7B
                  </option>
                </select>
              </div>

              <div>
                <label className="block text-[10px] font-mono uppercase text-ink-muted mb-1">
                  Language
                </label>
                <select
                  value={ttsLang}
                  onChange={(e) => setTtsLang(e.target.value)}
                  className="w-full px-2 py-1.5 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
                >
                  <option value="vi">vi (Vietnamese)</option>
                  <option value="en">en (English)</option>
                  <option value="zh">zh (Chinese)</option>
                  <option value="ja">ja (Japanese)</option>
                </select>
              </div>

              <div>
                <label className="block text-[10px] font-mono uppercase text-ink-muted mb-1">
                  Speaker Voice
                </label>
                <select
                  value={ttsVoice}
                  onChange={(e) => setTtsVoice(e.target.value)}
                  className="w-full px-2 py-1.5 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
                >
                  <option value="Vivian">Vivian</option>
                  <option value="Serena">Serena</option>
                  <option value="Ryan">Ryan</option>
                  <option value="Aiden">Aiden</option>
                  <option value="Uncle_Fu">Uncle_Fu</option>
                </select>
              </div>

              <div>
                <label className="block text-[10px] font-mono uppercase text-ink-muted mb-1">
                  Speed ({ttsSpeed}x)
                </label>
                <input
                  type="range"
                  min={0.5}
                  max={1.8}
                  step={0.1}
                  value={ttsSpeed}
                  onChange={(e) => setTtsSpeed(parseFloat(e.target.value))}
                  className="w-full mt-2"
                />
              </div>
            </div>

            {ttsMode === "clone" ? (
              <div className="border border-hairline rounded-[6px] p-3 bg-subtle space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono text-ink">
                    Reference Audio Sample (3-10s WAV/MP3)
                  </span>
                  <label className="px-2.5 py-1 rounded-[5px] border border-hairline bg-surface hover:bg-canvas text-xs font-mono cursor-pointer">
                    <span>{refAudioName || "Choose File"}</span>
                    <input
                      type="file"
                      accept="audio/*"
                      onChange={handleUploadRefAudio}
                      className="hidden"
                    />
                  </label>
                </div>
                <input
                  type="text"
                  value={refText}
                  onChange={(e) => setRefText(e.target.value)}
                  placeholder="Reference transcript spoken in sample (ref_text for Qwen3-TTS)..."
                  className="w-full px-2.5 py-1.5 rounded-[6px] border border-hairline bg-surface text-xs font-mono"
                />
              </div>
            ) : (
              <div>
                <label className="block text-[10px] font-mono uppercase text-ink-muted mb-1">
                  Persona / Prosody Instruction (voice_description)
                </label>
                <input
                  type="text"
                  value={ttsInstruct}
                  onChange={(e) => setTtsInstruct(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-[6px] border border-hairline bg-canvas text-xs font-mono"
                />
              </div>
            )}

            <label className="flex items-center justify-between py-1.5 px-2.5 rounded-[6px] border border-hairline bg-subtle cursor-pointer">
              <span className="text-xs font-medium text-ink flex items-center gap-1.5">
                <Columns weight="bold" className="w-3.5 h-3.5" />
                <span>Sequential Compare (XTTS-v2 vs Qwen3-TTS)</span>
              </span>
              <input
                type="checkbox"
                checked={compareTts}
                onChange={(e) => setCompareTts(e.target.checked)}
                className="accent-ink"
              />
            </label>

            <button
              type="button"
              disabled={ttsBusy}
              onClick={handleSynthesizeSpeech}
              className="w-full py-2.5 rounded-[6px] bg-ink text-surface hover:bg-[#2a2a2a] disabled:opacity-50 text-xs font-medium flex items-center justify-center gap-2 transition-transform active:scale-[0.98]"
            >
              <SpeakerHigh weight="bold" className="w-4 h-4" />
              <span>{ttsBusy ? "Synthesizing..." : "Synthesize WAV Audio"}</span>
            </button>

            {/* Waveform & Audio Player */}
            <div className="border border-hairline rounded-[8px] p-3.5 bg-canvas space-y-3">
              <div className="flex items-center justify-between text-[10px] font-mono uppercase text-ink-muted">
                <span>Waveform Visualizer</span>
                <div className="flex items-center gap-3">
                  {audioUrl && (
                    <button
                      type="button"
                      onClick={useSynthesizedAudioForStt}
                      className="text-ink hover:underline"
                    >
                      Send to ASR &rarr;
                    </button>
                  )}
                  {audioUrl && (
                    <a
                      href={audioUrl}
                      download={`speech-${Date.now()}.wav`}
                      className="text-ink hover:underline flex items-center gap-1"
                    >
                      <DownloadSimple weight="bold" className="w-3 h-3" />
                      <span>Download WAV</span>
                    </a>
                  )}
                </div>
              </div>

              <canvas
                ref={waveformCanvasRef}
                width={520}
                height={68}
                className="w-full h-[68px] rounded-[4px] border border-hairline bg-canvas"
              />

              {compareAudio ? (
                <div className="space-y-2 pt-1">
                  <div className="text-[11px] font-mono text-ink-secondary">
                    {compareAudio.status}
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    <div className="p-2 border border-hairline rounded-[6px] bg-surface space-y-1">
                      <div className="text-[10px] font-mono text-ink-muted">
                        XTTS-v2 ({compareAudio.xttsTime}s)
                      </div>
                      {compareAudio.xttsUrl && (
                        <audio
                          src={compareAudio.xttsUrl}
                          controls
                          className="w-full h-8"
                        />
                      )}
                    </div>
                    <div className="p-2 border border-hairline rounded-[6px] bg-surface space-y-1">
                      <div className="text-[10px] font-mono text-ink-muted">
                        Qwen3-TTS ({compareAudio.qwenTime}s)
                      </div>
                      {compareAudio.qwenUrl && (
                        <audio
                          src={compareAudio.qwenUrl}
                          controls
                          className="w-full h-8"
                        />
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                audioUrl && (
                  <audio
                    src={audioUrl}
                    controls
                    autoPlay
                    className="w-full h-9"
                  />
                )
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="bg-surface border border-hairline rounded-[10px] overflow-hidden">
        <TelemetryFooter telemetry={telemetry} />
      </div>
    </div>
  );
}
