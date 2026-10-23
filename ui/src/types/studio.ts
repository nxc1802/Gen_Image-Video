export type StudioViewId =
  | "chat"
  | "vision"
  | "image"
  | "video"
  | "audio"
  | "models"
  | "memory"
  | "config";

export interface SseProgressState {
  active: boolean;
  stage: string;
  step: number;
  total: number;
  percent: number;
  elapsedSec: number | string;
}

export interface TelemetryMeta {
  endpoint: string;
  model?: string;
  extra?: string;
  elapsedSec?: string | number;
}

export interface GpuDeviceItem {
  gpu_id?: number;
  index?: number;
  name?: string;
  allocated_gb?: number;
  reserved_gb?: number;
  hardware_used_gb?: number;
  hardware_free_gb?: number;
  total_gb?: number;
  free_gb?: number;
}

export interface SystemRamTelemetry {
  total_gb?: number;
  available_gb?: number;
  used_gb?: number;
  percent?: number;
  status?: string;
}

export interface MemoryResponse {
  gpus?: GpuDeviceItem[];
  ram?: SystemRamTelemetry;
  loaded_models?: string[];
}

export interface CatalogEntry {
  model_id: string;
  loaded?: boolean;
  preload?: boolean;
  lifecycle?: string;
  device_strategy?: string;
  quantization?: string;
}

export interface HealthResponse {
  status: string;
  service?: string;
  studio_mode?: string;
  active_modalities?: string[];
  loaded_models?: string[];
  idle_shutdown_minutes?: number;
  seconds_since_last_request?: number;
  catalog?: Record<string, CatalogEntry>;
  vram_status?: Record<string, unknown>;
}

export interface ModelEntry {
  id: string;
  object?: string;
  owned_by?: string;
  type?: string;
  loaded?: boolean;
  device_strategy?: string;
  quantization?: string;
}

export interface AudioDiagnosticsResponse {
  device?: string;
  cuda_available?: boolean;
  cuda_device_name?: string;
  default_ref_wav?: {
    path?: string;
    exists?: boolean;
  };
  imports?: Record<
    string,
    {
      status?: string;
      file?: string;
      error?: string;
    }
  >;
  paths?: {
    asr_path?: string;
    tts_path?: string;
    xtts_path?: string;
    kaggle_input?: string[];
    [key: string]: unknown;
  };
  models?: {
    asr?: string;
    tts?: string;
    xtts?: string;
  };
  errors?: {
    asr?: string | null;
    tts?: string | null;
    xtts?: string | null;
  };
  [key: string]: unknown;
}

export interface ChatMessageContentPart {
  type: "text" | "image_url";
  text?: string;
  image_url?: { url: string };
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string | ChatMessageContentPart[];
}

export interface ChatCompletionRequestPayload {
  model: string;
  messages: ChatMessage[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  repetition_penalty?: number;
  enable_thinking?: boolean;
  system_prompt?: string;
  stop?: string[];
  stream?: boolean;
}

export interface ImageGenerationRequestPayload {
  model?: string;
  prompt: string;
  negative_prompt?: string | null;
  size?: string;
  num_inference_steps?: number;
  steps?: number;
  guidance_scale?: number;
  guidance?: number;
  seed?: number | null;
  strength?: number;
  image?: string;
  mask_image?: string | null;
  stream?: boolean;
}

export interface VideoGenerationRequestPayload {
  model?: string;
  prompt: string;
  negative_prompt?: string | null;
  image?: string | null;
  size?: string;
  num_frames?: number;
  fps?: number;
  num_inference_steps?: number;
  steps?: number;
  guidance_scale?: number;
  guidance?: number;
  seed?: number | null;
  stream?: boolean;
}

export interface TranscriptionSegment {
  start: number;
  end: number;
  text: string;
}

export interface TranscriptionResponse {
  text: string;
  language?: string;
  model?: string;
  x_duration_seconds?: number;
  segments?: TranscriptionSegment[];
  notice?: string;
}

export interface SpeechSynthesisRequestPayload {
  model?: string;
  input: string;
  voice?: string;
  language?: string;
  speed?: number;
  instruct?: string | null;
  ref_audio?: string | null;
  ref_text?: string | null;
  response_format?: string;
}

export interface RuntimeConfigPayload {
  llm_model?: string;
  image_model?: string;
  image_edit_model?: string;
  video_model?: string;
  tts_model?: string;
  stt_model?: string;
  enable_thinking?: boolean;
  [key: string]: unknown;
}
