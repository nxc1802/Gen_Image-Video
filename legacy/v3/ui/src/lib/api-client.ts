import type {
  AudioDiagnosticsResponse,
  ChatCompletionRequestPayload,
  GpuDeviceItem,
  HealthResponse,
  ImageGenerationRequestPayload,
  MemoryResponse,
  ModelEntry,
  RuntimeConfigPayload,
  SpeechSynthesisRequestPayload,
  SseProgressState,
  TranscriptionResponse,
  VideoGenerationRequestPayload,
} from "@/types/studio";
import { normalizeAudioFileToWav } from "@/lib/utils";

export class StudioApiClient {
  constructor(
    private readonly baseUrl: string,
    private readonly apiKey: string = ""
  ) {}

  private endpoint(path: string): string {
    const cleanPath = path.startsWith("/") ? path : `/${path}`;
    if (typeof window !== "undefined") {
      return `/api/proxy${cleanPath}`;
    }
    return `${this.baseUrl}${cleanPath}`;
  }

  private headers(isJson = true): HeadersInit {
    const h: Record<string, string> = {};
    if (isJson) {
      h["Content-Type"] = "application/json";
    }
    if (this.apiKey) {
      h["Authorization"] = `Bearer ${this.apiKey}`;
    }
    if (this.baseUrl) {
      h["x-target-base-url"] = this.baseUrl;
    }
    return h;
  }

  async getHealth(): Promise<HealthResponse> {
    const res = await fetch(this.endpoint("/health"), {
      headers: this.headers(false),
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async getModels(): Promise<{ data: ModelEntry[] }> {
    const res = await fetch(this.endpoint("/v1/models"), {
      headers: this.headers(false),
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    return { data: Array.isArray(json.data) ? json.data : [] };
  }

  async getMemory(): Promise<MemoryResponse> {
    const res = await fetch(this.endpoint("/v1/memory"), {
      headers: this.headers(false),
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const raw = await res.json();

    const gpus: GpuDeviceItem[] = [];
    if (Array.isArray(raw.gpus)) {
      gpus.push(...raw.gpus);
    } else if (raw.gpus && typeof raw.gpus === "object") {
      Object.entries(raw.gpus).forEach(([k, val], idx) => {
        const item = val as Record<string, unknown>;
        const alloc = Math.max(
          Number(item.allocated_gb ?? 0),
          Number(item.hardware_used_gb ?? 0)
        );
        const resv = Math.max(
          Number(item.reserved_gb ?? 0),
          Number(item.hardware_used_gb ?? 0)
        );
        gpus.push({
          gpu_id: Number(k.replace(/\D/g, "")) || idx,
          name: String(item.name || "NVIDIA Tesla T4"),
          allocated_gb: alloc,
          reserved_gb: resv,
          total_gb: Number(item.total_gb ?? 14.56),
          free_gb: Number(item.free_gb ?? item.hardware_free_gb ?? 14.56),
        });
      });
    } else {
      if (raw.gpu_0) gpus.push({ gpu_id: 0, ...raw.gpu_0 });
      if (raw.gpu_1) gpus.push({ gpu_id: 1, ...raw.gpu_1 });
    }

    return {
      gpus,
      ram: raw.ram,
      loaded_models:
        raw.loaded_models ||
        raw.always_active_slots ||
        (raw.active_dynamic_slot ? [raw.active_dynamic_slot] : []),
    };
  }

  async freeAllVram(): Promise<{ status: string; message?: string }> {
    const res = await fetch(this.endpoint("/v1/memory/free"), {
      method: "POST",
      headers: this.headers(true),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async getAudioDiagnostics(tryLoad = false): Promise<AudioDiagnosticsResponse> {
    const res = await fetch(
      this.endpoint(
        `/v1/audio/diagnostics?try_load=${tryLoad ? "true" : "false"}`
      ),
      {
        headers: this.headers(false),
        cache: "no-store",
      }
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async getRuntimeConfig(): Promise<RuntimeConfigPayload> {
    const res = await fetch(this.endpoint("/v1/config"), {
      headers: this.headers(false),
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const raw = await res.json();
    const mc = raw.models_config || raw;
    return {
      llm_model: mc.vlm?.id || mc.vlm?.model_id || raw.llm_model,
      image_model: mc.image?.id || mc.image?.model_id || raw.image_model,
      image_edit_model:
        mc.image_edit?.id || mc.image?.id || raw.image_edit_model,
      video_model: mc.video?.id || mc.video?.model_id || raw.video_model,
      tts_model: mc.tts?.id || mc.tts?.model_id || raw.tts_model,
      stt_model:
        mc.stt?.id || mc.asr?.id || mc.stt?.model_id || raw.stt_model,
      enable_thinking: Boolean(raw.enable_thinking),
    };
  }

  async updateRuntimeConfig(
    payload: RuntimeConfigPayload
  ): Promise<Record<string, unknown>> {
    const body = {
      vlm: payload.llm_model ? { id: payload.llm_model } : undefined,
      image: payload.image_model ? { id: payload.image_model } : undefined,
      video: payload.video_model ? { id: payload.video_model } : undefined,
      tts: payload.tts_model ? { id: payload.tts_model } : undefined,
      stt: payload.stt_model ? { id: payload.stt_model } : undefined,
    };

    let res = await fetch(this.endpoint("/v1/config"), {
      method: "PATCH",
      headers: this.headers(true),
      body: JSON.stringify(body),
    });
    if (res.status === 405) {
      res = await fetch(this.endpoint("/v1/config"), {
        method: "POST",
        headers: this.headers(true),
        body: JSON.stringify(body),
      });
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async shutdownKernel(): Promise<{ status: string; message: string }> {
    const res = await fetch(this.endpoint("/v1/admin/shutdown?confirm=1"), {
      method: "POST",
      headers: this.headers(true),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async streamChatCompletion(
    payload: ChatCompletionRequestPayload,
    onDelta: (deltaText: string, deltaReasoning?: string) => void
  ): Promise<{ content: string; reasoning: string }> {
    const res = await fetch(this.endpoint("/v1/chat/completions"), {
      method: "POST",
      headers: this.headers(true),
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    }

    if (!payload.stream) {
      const data = await res.json();
      const msg = data.choices?.[0]?.message || {};
      const content = msg.content || "";
      const reasoning = msg.reasoning_content || "";
      onDelta(content, reasoning);
      return { content, reasoning };
    }

    const reader = res.body?.getReader();
    if (!reader) throw new Error("ReadableStream unavailable");

    const decoder = new TextDecoder("utf-8");
    const contentMojibakeDecoder = new TextDecoder("utf-8");
    const reasoningMojibakeDecoder = new TextDecoder("utf-8");

    const normalizeToken = (tok: string, td: TextDecoder): string => {
      if (!tok) return "";
      let hasHighLatin1 = false;
      for (let i = 0; i < tok.length; i++) {
        const c = tok.charCodeAt(i);
        if (c > 0xff) {
          return tok;
        }
        if (c >= 0x80) {
          hasHighLatin1 = true;
        }
      }
      if (!hasHighLatin1) {
        return td.decode() + tok;
      }
      const bytes = new Uint8Array(tok.length);
      for (let i = 0; i < tok.length; i++) {
        bytes[i] = tok.charCodeAt(i);
      }
      return td.decode(bytes, { stream: true });
    };

    let buffer = "";
    let fullContent = "";
    let fullReasoning = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const raw = trimmed.slice(5).trim();
        if (raw === "[DONE]") break;
        try {
          const parsed = JSON.parse(raw);
          const delta = parsed.choices?.[0]?.delta || {};
          const dText = normalizeToken(
            delta.content || "",
            contentMojibakeDecoder
          );
          const dReason = normalizeToken(
            delta.reasoning_content || "",
            reasoningMojibakeDecoder
          );
          if (dText || dReason) {
            fullContent += dText;
            fullReasoning += dReason;
            onDelta(dText, dReason);
          }
        } catch {
          // Ignore keepalive comments
        }
      }
    }

    const tailContent = contentMojibakeDecoder.decode();
    const tailReasoning = reasoningMojibakeDecoder.decode();
    if (tailContent || tailReasoning) {
      fullContent += tailContent;
      fullReasoning += tailReasoning;
      onDelta(tailContent, tailReasoning);
    }

    if (fullContent.includes("<think>") && fullContent.includes("</think>")) {
      const m = fullContent.match(/<think>([\s\S]*?)<\/think>/);
      if (m) {
        if (!fullReasoning) {
          fullReasoning = m[1].trim();
        }
        fullContent = fullContent.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
      }
    }

    return { content: fullContent, reasoning: fullReasoning };
  }

  async generateOrEditImage(
    isEdit: boolean,
    payload: ImageGenerationRequestPayload,
    onProgress?: (state: SseProgressState) => void
  ): Promise<{ data: Array<{ b64_json: string }> }> {
    const endpointPath = isEdit ? "/v1/images/edits" : "/v1/images/generations";
    const t0 = performance.now();
    const totalSteps = payload.num_inference_steps || payload.steps || 4;

    if (onProgress) {
      onProgress({
        active: true,
        stage: isEdit
          ? "Running Image Edit / Inpaint..."
          : "Synthesizing Image...",
        step: 0,
        total: totalSteps,
        percent: 5,
        elapsedSec: "0.0",
      });
    }

    const res = await fetch(this.endpoint(endpointPath), {
      method: "POST",
      headers: this.headers(true),
      body: JSON.stringify({
        ...payload,
        steps: totalSteps,
        guidance: payload.guidance_scale ?? payload.guidance ?? 3.5,
      }),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    }

    const contentType = res.headers.get("content-type") || "";
    if (!payload.stream || !contentType.includes("text/event-stream")) {
      return res.json();
    }

    const reader = res.body?.getReader();
    if (!reader) throw new Error("Stream reader unavailable");

    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let finalB64 = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const raw = trimmed.slice(5).trim();
        if (raw === "[DONE]") break;
        const ev = JSON.parse(raw);
        if (ev.error) throw new Error(ev.error);

        const step = ev.step ?? 0;
        const total = ev.total_steps ?? ev.total ?? totalSteps;
        if (step && total && onProgress) {
          onProgress({
            active: true,
            stage: ev.stage || "Denoising latents...",
            step,
            total,
            percent: Math.round((step / total) * 100),
            elapsedSec: ((performance.now() - t0) / 1000).toFixed(1),
          });
        }
        if (ev.data?.[0]?.b64_json) {
          finalB64 = ev.data[0].b64_json;
        } else if (ev.b64_json) {
          finalB64 = ev.b64_json;
        }
      }
    }

    return { data: finalB64 ? [{ b64_json: finalB64 }] : [] };
  }

  async generateVideo(
    payload: VideoGenerationRequestPayload,
    onProgress?: (state: SseProgressState) => void
  ): Promise<{ data: Array<{ b64_json: string }> }> {
    const t0 = performance.now();
    const totalSteps = payload.num_inference_steps || payload.steps || 30;
    const [w, h] = (payload.size || "832x480")
      .split("x")
      .map((n) => parseInt(n, 10));

    if (onProgress) {
      onProgress({
        active: true,
        stage: "Initializing Wan2.2 Video Pipeline...",
        step: 0,
        total: totalSteps,
        percent: 4,
        elapsedSec: "0.0",
      });
    }

    const res = await fetch(this.endpoint("/v1/videos/generations"), {
      method: "POST",
      headers: this.headers(true),
      body: JSON.stringify({
        ...payload,
        width: w || 832,
        height: h || 480,
        steps: totalSteps,
        guidance: payload.guidance_scale ?? payload.guidance ?? 5.0,
      }),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    }

    const contentType = res.headers.get("content-type") || "";
    if (!payload.stream || !contentType.includes("text/event-stream")) {
      return res.json();
    }

    const reader = res.body?.getReader();
    if (!reader) throw new Error("Stream reader unavailable");

    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let finalB64 = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const raw = trimmed.slice(5).trim();
        if (raw === "[DONE]") break;
        const ev = JSON.parse(raw);
        if (ev.error) throw new Error(ev.error);

        const step = ev.step ?? 0;
        const total = ev.total_steps ?? ev.total ?? totalSteps;
        if (step && total && onProgress) {
          onProgress({
            active: true,
            stage: ev.stage || "Denoising video frames...",
            step,
            total,
            percent: Math.round((step / total) * 100),
            elapsedSec: ((performance.now() - t0) / 1000).toFixed(1),
          });
        }
        if (ev.data?.[0]?.b64_json) {
          finalB64 = ev.data[0].b64_json;
        } else if (ev.b64_json) {
          finalB64 = ev.b64_json;
        }
      }
    }

    return { data: finalB64 ? [{ b64_json: finalB64 }] : [] };
  }

  async transcribeAudio(params: {
    file: File;
    model: string;
    language: string;
    responseFormat: string;
    timestampGranularities: boolean;
  }): Promise<TranscriptionResponse> {
    // Normalize any MP3 / M4A / WebM / OGG to 16kHz Mono 16-bit PCM WAV in browser
    const normalizedFile = await normalizeAudioFileToWav(params.file, 16000);

    const sendRequest = async (audioFile: File, withTimestamps: boolean) => {
      const fd = new FormData();
      fd.append("file", audioFile);
      fd.append("model", params.model);
      if (params.language && params.language !== "auto") {
        fd.append("language", params.language);
      }
      fd.append("response_format", params.responseFormat);
      fd.append("return_time_stamps", withTimestamps ? "true" : "false");

      return fetch(this.endpoint("/v1/audio/transcriptions"), {
        method: "POST",
        headers: this.headers(false),
        body: fd,
      });
    };

    let res: Response | null = null;
    let fallbackNotice: string | undefined;

    if (params.timestampGranularities) {
      fallbackNotice =
        "Note: ForcedAligner is not loaded on Kaggle; transcribed with return_time_stamps=false.";
    }

    res = await sendRequest(normalizedFile, false);

    if (!res.ok && normalizedFile !== params.file) {
      res = await sendRequest(params.file, false);
    }

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    }

    const raw = await res.json();
    const segments =
      raw.segments ||
      (Array.isArray(raw.time_stamps)
        ? raw.time_stamps.map(
            (ts: {
              start_time?: number;
              end_time?: number;
              start?: number;
              end?: number;
              text?: string;
            }) => ({
              start: ts.start_time ?? ts.start ?? 0,
              end: ts.end_time ?? ts.end ?? 0,
              text: ts.text ?? "",
            })
          )
        : []);

    return {
      text: raw.text || "",
      language: raw.language,
      model: raw.model,
      x_duration_seconds: raw.x_duration_seconds,
      segments,
      notice: fallbackNotice,
    };
  }

  async synthesizeSpeech(payload: SpeechSynthesisRequestPayload): Promise<Blob> {
    const isXtts = (payload.model || "").toLowerCase().includes("xtts");
    const bodyPayload: Record<string, unknown> = {
      model: payload.model,
      input: payload.input,
      voice: payload.voice,
      language: payload.language || "vi",
      speed: payload.speed ?? 1.0,
      response_format: payload.response_format || "wav",
    };

    if (payload.instruct) {
      bodyPayload.voice_description = payload.instruct;
    }
    if (payload.ref_text) {
      bodyPayload.ref_text = payload.ref_text;
    }
    if (payload.ref_audio) {
      if (isXtts) {
        bodyPayload.speaker_wav = payload.ref_audio;
      } else {
        bodyPayload.prompt_audio = payload.ref_audio;
      }
    }

    const res = await fetch(this.endpoint("/v1/audio/speech"), {
      method: "POST",
      headers: this.headers(true),
      body: JSON.stringify(bodyPayload),
    });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    }
    return res.blob();
  }

  static async discoverNtfyTunnelUrl(topic: string): Promise<string> {
    const cleanedTopic = topic.trim();
    if (!cleanedTopic) throw new Error("Please enter an ntfy.sh topic name");
    const res = await fetch(
      `https://ntfy.sh/${encodeURIComponent(cleanedTopic)}/raw?poll=1&since=12h`,
      { cache: "no-store" }
    );
    if (!res.ok) throw new Error(`ntfy HTTP ${res.status}`);
    const text = await res.text();
    const matches = text.match(/https:\/\/[a-zA-Z0-9.-]+\.trycloudflare\.com/g);
    if (!matches || matches.length === 0) {
      throw new Error(
        `No active trycloudflare.com URL found in topic "${cleanedTopic}"`
      );
    }
    return matches[matches.length - 1];
  }
}
