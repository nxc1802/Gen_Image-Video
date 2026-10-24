// Factory Studio v4 Operational Types
// Strict declarative TypeScript definitions for manifests, two-tier queue jobs, telemetry, and forms.

export type StudioViewId = "console" | "explorer" | "jobs" | "inspector" | "registry";

export type SubsystemId =
  | "text"
  | "vlm"
  | "decision"
  | "speech"
  | "embedding"
  | "image"
  | "video"
  | "music"
  | "enhancement"
  | "jobs";

// ==========================================
// Manifest Schemas
// ==========================================

export interface ParameterConstraint {
  min_value?: number;
  max_value?: number;
  step?: number;
  choices?: (string | number | boolean)[];
  regex_pattern?: string;
  max_length?: number;
}

export type ParameterType =
  | "string"
  | "integer"
  | "float"
  | "boolean"
  | "image_url"
  | "audio_url"
  | "video_url"
  | "mask_url"
  | "choices"
  | "json"
  | "array";

export interface ParameterSpec {
  name: string;
  type: ParameterType;
  description: string;
  required: boolean;
  default?: any;
  constraints?: ParameterConstraint;
}

export interface CapabilityOutputSpec {
  name: string;
  type: string;
  format?: string;
}

export interface CapabilityManifest {
  apiVersion?: string;
  kind?: "CapabilityManifest";
  id: string;
  subsystem: SubsystemId | string;
  display_name: string;
  description: string;
  inputs: ParameterSpec[];
  outputs: CapabilityOutputSpec[];
  supported_model_families?: string[];
  validation?: {
    test_fixture?: string;
    verification_method?: string;
  };
}

export interface ModelHardware {
  min_vram_gb: number;
  preferred_gpu?: string;
  allocation_policy?: string;
}

export interface ModelManifest {
  id: string;
  family: string;
  service_id: string;
  display_name?: string;
  version?: string;
  status: "full" | "partial" | "planned";
  coverage_percent?: number;
  validation_status?: "pending_live" | "contract_verified" | "live_verified";
  validation_notes?: string | null;
  hardware: ModelHardware;
  capabilities: Record<string, boolean>;
  limits?: {
    max_width?: number;
    max_height?: number;
    max_context_tokens?: number;
    max_audio_length_seconds?: number;
    [key: string]: any;
  };
}

export interface ServiceNetwork {
  static_domain: string;
  port: number;
}

export interface ServiceManifest {
  id: string;
  name: string;
  subsystem: SubsystemId | string;
  version: string;
  network: ServiceNetwork;
  framework?: string;
  hardware?: {
    gpu?: string;
    min_vram_gb?: number;
  };
}

export interface ManifestSchemasResponse {
  schemas: {
    services: ServiceManifest[];
    models: ModelManifest[];
  };
}

// ==========================================
// Factory Two-Tier Queue Job Contracts
// ==========================================

export type JobState =
  | "CREATED"
  | "QUEUED"
  | "ADMITTED"
  | "RUNNING"
  | "STREAMING"
  | "SUCCEEDED"
  | "FAILED"
  | "CANCELLED"
  | "RETRYING";

export interface JobProgress {
  stage: string;
  step: number;
  total_steps: number;
  percent: number;
  message?: string;
}

export interface JobArtifact {
  name: string;
  type: string;
  storage_path: string;
  download_url?: string;
  sha256?: string;
  size_bytes?: number;
}

export interface JobError {
  code: string;
  message: string;
  request_id: string;
  details?: Record<string, any>;
}

export interface FactoryJob {
  job_id: string;
  request_id: string;
  service: string;
  model: string;
  task: string;
  state: JobState;
  inputs: Record<string, any>;
  priority: number;
  created_at: string;
  admitted_at?: string;
  started_at?: string;
  completed_at?: string;
  worker_id?: string;
  progress?: JobProgress;
  artifacts?: JobArtifact[];
  error?: JobError;
}

export interface JobSubmissionPayload {
  service: string;
  model: string;
  task: string;
  inputs: Record<string, any>;
  priority?: number;
}

export interface JobLifecycleEvent {
  event: "state_change" | "progress" | "artifact" | "error" | "ping";
  job_id: string;
  state?: JobState;
  progress?: JobProgress;
  artifact?: JobArtifact;
  error?: JobError;
  timestamp: number;
}

// ==========================================
// Asset Service
// ==========================================

export interface AssetUploadResponse {
  asset_id: string;
  url: string;
  s3_path: string;
  sha256: string;
  size_bytes: number;
  mime_type: string;
  created_at: string;
}

// ==========================================
// Code Generation
// ==========================================

export interface CodegenResult {
  curl: string;
  python: string;
  typescript: string;
  openai?: string;
}

// ==========================================
// Real-time Telemetry & Inspector
// ==========================================

export interface WorkerLeaseInfo {
  worker_id: string;
  model_id: string;
  state: string;
  lease_expiry?: string;
  heartbeat_at?: string;
}

export interface TelemetrySnapshot {
  gpu_0_vram_used_gb: number;
  gpu_1_vram_used_gb: number;
  gpu_0_vram_total_gb?: number;
  gpu_1_vram_total_gb?: number;
  active_jobs: number;
  queue_depth: number;
  timestamp?: number;
  worker_count?: number;
  leases?: WorkerLeaseInfo[];
  raw_message?: string;
}

// ==========================================
// Health & System
// ==========================================

export interface HealthResponse {
  status: string;
  service?: string;
  version?: string;
  active_modalities?: string[];
  loaded_models?: string[];
  idle_shutdown_minutes?: number;
  seconds_since_last_request?: number;
  vram_status?: Record<string, unknown>;
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

export interface MemoryResponse {
  gpus?: GpuDeviceItem[];
  loaded_models?: string[];
}

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
