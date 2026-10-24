/**
 * packages/contracts/ts/index.ts
 * Factory v4 Authoritative TypeScript Contracts & Interfaces
 */

export type SubsystemType =
  | 'text'
  | 'vlm'
  | 'decision'
  | 'embedding'
  | 'speech'
  | 'image'
  | 'video'
  | 'music'
  | 'enhancement'
  | 'jobs';

export type ErrorCode =
  | 'UNAUTHORIZED'
  | 'FORBIDDEN'
  | 'INVALID_REQUEST'
  | 'BAD_REQUEST'
  | 'CAPABILITY_NOT_SUPPORTED'
  | 'SSRF_BLOCKED'
  | 'INVALID_MEDIA_TYPE'
  | 'NOT_FOUND'
  | 'METHOD_NOT_ALLOWED'
  | 'JOB_CANCELLED'
  | 'CONFLICT'
  | 'RESOURCE_CONFLICT'
  | 'PAYLOAD_TOO_LARGE'
  | 'UNSUPPORTED_MEDIA_TYPE'
  | 'VALIDATION_ERROR'
  | 'RATE_LIMIT_EXCEEDED'
  | 'RATE_LIMITED'
  | 'QUEUE_FULL'
  | 'BAD_GATEWAY'
  | 'MODEL_UNAVAILABLE'
  | 'SERVICE_UNAVAILABLE'
  | 'TIMEOUT'
  | 'GATEWAY_TIMEOUT'
  | 'INTERNAL_ERROR'
  | 'INTERNAL_SERVER_ERROR';

export interface ErrorDetail {
  code: ErrorCode;
  message: string;
  request_id: string;
  timestamp: string;
  details?: Record<string, unknown>;
}

export interface UnifiedErrorResponse {
  error: ErrorDetail;
}

export interface ServiceManifest {
  apiVersion: string;
  kind: 'ServiceManifest';
  metadata: {
    id: string;
    name: string;
    subsystem: SubsystemType;
    version: string;
  };
  network: {
    static_domain: string;
    internal_port: number;
    protocol: 'http' | 'https' | 'ws' | 'wss';
    auth_required: boolean;
    cors_origins: string[];
  };
  health_check: {
    endpoint: string;
    interval_seconds: number;
    timeout_seconds: number;
    unhealthy_threshold: number;
  };
  framework: {
    type: 'vllm' | 'comfyui' | 'acestep' | 'fastapi' | 'custom';
    version: string;
    entrypoint: string;
  };
  hardware: {
    gpu_required: boolean;
    gpu_device_index?: number | null;
    min_vram_gb: number;
    min_ram_gb: number;
    cuda_version_min?: string;
  };
  supported_modalities: string[];
  auth_policy: {
    scheme: string;
    jwt_audience?: string;
    rate_limit_rpm: number;
    allow_unauthenticated: false;
  };
}

export type ModelStatus = 'full' | 'partial';

export interface ModelManifest {
  apiVersion: string;
  kind: 'ModelManifest';
  metadata: {
    id: string;
    family: string;
    display_name: string;
    service_id: string;
    version: string;
  };
  weights: {
    primary_checkpoint: string;
    components?: Record<string, string>;
    format: 'safetensors' | 'gguf' | 'diffusers' | 'torch' | 'onnx';
    quantization: 'none' | 'fp8' | 'q4_k_m' | 'q8_0' | 'awq' | 'gptq' | '4bit';
    default_precision: 'bf16' | 'fp16' | 'fp32';
  };
  hardware: {
    min_vram_gb: number;
    context_window?: number | null;
    preferred_gpu: string;
    allocation_policy: 'single_gpu' | 'dual_gpu_split' | 'offload_ram';
  };
  capabilities: Record<string, boolean>;
  status: ModelStatus;
  coverage_percent: number;
  missing_capabilities: string[];
  limits: {
    max_context_tokens?: number | null;
    max_width?: number | null;
    max_height?: number | null;
    max_frames?: number | null;
    max_duration_seconds?: number | null;
  };
}

export interface CapabilityManifest {
  apiVersion: string;
  kind: 'CapabilityManifest';
  id: string;
  subsystem: string;
  display_name: string;
  description: string;
  inputs: Array<{
    name: string;
    type: string;
    description: string;
    required: boolean;
    default?: unknown;
    constraints?: {
      min_value?: number;
      max_value?: number;
      choices?: unknown[];
      regex_pattern?: string;
      max_length?: number;
      step?: number;
    };
  }>;
  outputs: Array<{
    name: string;
    type: string;
    format: string;
  }>;
  validation: {
    test_fixture: string;
    golden_output_hash?: string;
    verification_method: string;
  };
  supported_model_families: string[];
}

export interface HardwareTopologyManifest {
  apiVersion: string;
  kind: 'HardwareTopologyManifest';
  id: string;
  name: string;
  environment: string;
  host: {
    cpu_cores: number;
    system_ram_gb: number;
    storage_temp_gb: number;
  };
  gpus: Array<{
    index: number;
    name: string;
    total_vram_gb: number;
    max_allocatable_vram_gb: number;
    assigned_services: string[];
  }>;
}

export type JobState =
  | 'CREATED'
  | 'QUEUED'
  | 'ADMITTED'
  | 'RUNNING'
  | 'STREAMING'
  | 'SUCCEEDED'
  | 'FAILED'
  | 'CANCELLED'
  | 'RETRYING';

export interface JobProgress {
  stage: string;
  step: number;
  total: number;
  percent: number;
  message?: string | null;
  preview_url?: string | null;
}

export interface ArtifactReference {
  artifact_id: string;
  mime_type: string;
  size_bytes: number;
  storage_path: string;
  download_url: string;
  sha256: string;
  created_at: string;
}

export interface FactoryJob {
  job_id: string;
  request_id: string;
  idempotency_key?: string | null;
  service: string;
  model: string;
  task: string;
  inputs: Record<string, unknown>;
  priority: number;
  state: JobState;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  progress: JobProgress;
  worker?: {
    worker_id: string;
    gpu_device?: number | null;
    lease_expires_at: string;
  } | null;
  retry_count: number;
  max_retries: number;
  error?: ErrorDetail | null;
  artifacts: ArtifactReference[];
}

export interface JobSubmissionRequest {
  service: string;
  model: string;
  task: string;
  inputs: Record<string, unknown>;
  priority?: number;
  idempotency_key?: string;
}
