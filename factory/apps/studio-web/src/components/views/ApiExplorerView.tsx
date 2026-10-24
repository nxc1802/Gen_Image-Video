"use client";

import React, { useState, useMemo, useEffect } from "react";
import {
  Code,
  Cube,
  FileText,
  Sliders,
  CheckCircle,
  WarningCircle,
  ArrowRight,
  Sparkle,
} from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";
import type {
  CapabilityManifest,
  JobSubmissionPayload,
  ModelManifest,
  SubsystemId,
} from "@/types/studio";
import { DynamicSchemaForm } from "@/components/forms/DynamicSchemaForm";
import { ApiCodeGenerator } from "@/components/codegen/ApiCodeGenerator";

const ALL_SUBSYSTEMS: Array<{ id: SubsystemId; label: string }> = [
  { id: "image", label: "Image (ComfyUI)" },
  { id: "video", label: "Video (Wan / MiniMax)" },
  { id: "text", label: "Text (vLLM)" },
  { id: "vlm", label: "Vision-Language (vLLM)" },
  { id: "speech", label: "Speech (ASR / TTS)" },
  { id: "music", label: "Music (ACE-Step)" },
  { id: "enhancement", label: "Enhancement" },
  { id: "decision", label: "Decision Engine" },
  { id: "embedding", label: "Embedding & Rerank" },
];

export function ApiExplorerView() {
  const {
    services,
    models,
    capabilities,
    apiClient,
    refreshJobs,
    setActiveView,
  } = useStudio();

  const [selectedSubsystem, setSelectedSubsystem] = useState<SubsystemId>("image");
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [selectedCapabilityId, setSelectedCapabilityId] = useState<string>("");
  const [formInputs, setFormInputs] = useState<Record<string, any>>({});
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [submitSuccess, setSubmitSuccess] = useState<{
    jobId: string;
    message: string;
  } | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Models matching current subsystem
  const filteredModels = useMemo(() => {
    return models.filter((m) => {
      const svc = services.find((s) => s.id === m.service_id);
      return (
        svc?.subsystem === selectedSubsystem ||
        m.family.toLowerCase().includes(selectedSubsystem) ||
        m.id.toLowerCase().includes(selectedSubsystem)
      );
    });
  }, [models, services, selectedSubsystem]);

  // Update selected model when filtered models change
  useEffect(() => {
    if (filteredModels.length > 0) {
      if (!filteredModels.some((m) => m.id === selectedModelId)) {
        setSelectedModelId(filteredModels[0].id);
      }
    } else {
      setSelectedModelId("");
    }
  }, [filteredModels, selectedModelId]);

  const activeModel: ModelManifest | undefined = useMemo(() => {
    return models.find((m) => m.id === selectedModelId) || filteredModels[0];
  }, [models, selectedModelId, filteredModels]);

  // Capabilities matching selected subsystem or active model
  const matchingCapabilities = useMemo(() => {
    const list = capabilities.filter(
      (c) =>
        c.subsystem === selectedSubsystem ||
        c.id.startsWith(`${selectedSubsystem}.`) ||
        (activeModel?.family &&
          c.supported_model_families?.includes(activeModel.family))
    );

    if (list.length > 0) return list;

    // Fallback: build synthetic capability from active model capabilities dictionary
    if (activeModel?.capabilities) {
      return Object.keys(activeModel.capabilities).map((capKey) => ({
        id: capKey,
        subsystem: selectedSubsystem,
        display_name: capKey.replace(/[._]/g, " ").toUpperCase(),
        description: `Operational task ${capKey} backed by ${activeModel.id}`,
        inputs: [
          {
            name: "prompt",
            type: "string" as const,
            description: "Main generation or instruction prompt",
            required: true,
          },
          ...(selectedSubsystem === "image" || selectedSubsystem === "video"
            ? [
                {
                  name: "width",
                  type: "integer" as const,
                  description: "Output width in pixels",
                  required: false,
                  default: 1024,
                  constraints: { min_value: 512, max_value: 2048, step: 64 },
                },
                {
                  name: "height",
                  type: "integer" as const,
                  description: "Output height in pixels",
                  required: false,
                  default: 1024,
                  constraints: { min_value: 512, max_value: 2048, step: 64 },
                },
              ]
            : []),
        ],
        outputs: [{ name: "artifact", type: "url" }],
      }));
    }

    return [];
  }, [capabilities, selectedSubsystem, activeModel]);

  // Select active capability
  useEffect(() => {
    if (matchingCapabilities.length > 0) {
      if (!matchingCapabilities.some((c) => c.id === selectedCapabilityId)) {
        setSelectedCapabilityId(matchingCapabilities[0].id);
      }
    } else {
      setSelectedCapabilityId("");
    }
  }, [matchingCapabilities, selectedCapabilityId]);

  const activeCapability = useMemo(() => {
    return (
      matchingCapabilities.find((c) => c.id === selectedCapabilityId) ||
      matchingCapabilities[0]
    );
  }, [matchingCapabilities, selectedCapabilityId]);

  // Job Submission Payload
  const currentPayload: JobSubmissionPayload = useMemo(() => {
    return {
      service: selectedSubsystem,
      model: activeModel?.id || selectedModelId || "unknown-model",
      task: activeCapability?.id || "default_task",
      inputs: formInputs,
    };
  }, [selectedSubsystem, activeModel, selectedModelId, activeCapability, formInputs]);

  // Handle form submission
  const handleSubmitJob = async (inputs: Record<string, any>) => {
    setIsSubmitting(true);
    setSubmitSuccess(null);
    setSubmitError(null);

    const payload: JobSubmissionPayload = {
      service: selectedSubsystem,
      model: activeModel?.id || selectedModelId,
      task: activeCapability?.id || "default_task",
      inputs,
    };

    try {
      const job = await apiClient.submitJob(payload);
      setSubmitSuccess({
        jobId: job.job_id,
        message: `Job ${job.job_id} admitted into Two-Tier Queue (State: ${job.state})`,
      });
      await refreshJobs();
    } catch (err: any) {
      setSubmitError(err?.message || "Job submission failed");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="p-5 rounded-[10px] bg-surface border border-hairline">
        <h1 className="font-serif text-2xl text-ink">API Explorer</h1>
        <p className="text-xs text-ink-muted mt-1">
          Dynamic manifest-driven parameter execution engine. Zero hardcoded models — all schemas, constraints, and hardware bounds are loaded live from the Control Plane.
        </p>

        {/* Subsystem Selection Pills */}
        <div className="flex flex-wrap gap-2 mt-4 pt-4 border-t border-hairline">
          {ALL_SUBSYSTEMS.map((sub) => {
            const isActive = selectedSubsystem === sub.id;
            return (
              <button
                key={sub.id}
                type="button"
                onClick={() => setSelectedSubsystem(sub.id)}
                className={`px-3 py-1.5 rounded-full text-xs font-mono transition-colors ${
                  isActive
                    ? "bg-ink text-surface font-semibold shadow-sm"
                    : "bg-subtle text-ink-secondary hover:bg-muted hover:text-ink"
                }`}
              >
                {sub.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Model & Task Configuration Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Model Card Selector */}
        <div className="p-4 rounded-[10px] bg-surface border border-hairline space-y-3">
          <div className="flex items-center justify-between text-xs font-mono text-ink-muted">
            <span className="flex items-center gap-1.5 text-ink font-semibold">
              <Cube weight="bold" className="w-4 h-4" />
              <span>Target Model</span>
            </span>
            <span>{filteredModels.length} models available</span>
          </div>

          {filteredModels.length === 0 ? (
            <div className="p-4 rounded-[6px] bg-canvas border border-hairline text-xs font-mono text-ink-muted text-center">
              No registered model for {selectedSubsystem}. Check Model Registry.
            </div>
          ) : (
            <div className="space-y-2">
              <select
                value={selectedModelId}
                onChange={(e) => setSelectedModelId(e.target.value)}
                className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono font-medium text-ink"
              >
                {filteredModels.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.id} ({m.family}) — {m.status.toUpperCase()} ({m.coverage_percent ?? 100}%)
                  </option>
                ))}
              </select>

              {activeModel && (
                <div className="p-3 rounded-[6px] bg-canvas border border-hairline text-xs font-mono text-ink-secondary space-y-1">
                  <div className="flex justify-between text-[11px]">
                    <span>Hardware Requirement:</span>
                    <span className="text-ink font-semibold">
                      ≥{activeModel.hardware.min_vram_gb} GB VRAM ({activeModel.hardware.preferred_gpu || "Tesla T4"})
                    </span>
                  </div>
                  <div className="flex justify-between text-[11px]">
                    <span>Coverage Status:</span>
                    <span className="text-spot-greenText font-semibold">
                      {activeModel.status.toUpperCase()} ({activeModel.coverage_percent ?? 100}%)
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Task / Capability Selector */}
        <div className="p-4 rounded-[10px] bg-surface border border-hairline space-y-3">
          <div className="flex items-center justify-between text-xs font-mono text-ink-muted">
            <span className="flex items-center gap-1.5 text-ink font-semibold">
              <Sparkle weight="bold" className="w-4 h-4 text-amber-500" />
              <span>Task / Capability</span>
            </span>
            <span>{matchingCapabilities.length} capabilities</span>
          </div>

          {matchingCapabilities.length === 0 ? (
            <div className="p-4 rounded-[6px] bg-canvas border border-hairline text-xs font-mono text-ink-muted text-center">
              No specific capability declared. Default execution enabled.
            </div>
          ) : (
            <div className="space-y-2">
              <select
                value={selectedCapabilityId}
                onChange={(e) => setSelectedCapabilityId(e.target.value)}
                className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-canvas text-xs font-mono font-medium text-ink"
              >
                {matchingCapabilities.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.display_name} ({c.id})
                  </option>
                ))}
              </select>

              {activeCapability && (
                <div className="p-3 rounded-[6px] bg-canvas border border-hairline text-xs font-mono text-ink-secondary">
                  <p className="text-[11px] leading-relaxed">
                    {activeCapability.description}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Submission Feedback Banners */}
      {submitSuccess && (
        <div className="p-4 rounded-[8px] bg-spot-greenBg border border-[#C2DEC4] text-spot-greenText text-xs font-mono flex items-center justify-between animate-fadeIn">
          <div className="flex items-center gap-2">
            <CheckCircle weight="bold" className="w-4 h-4 shrink-0" />
            <span>{submitSuccess.message}</span>
          </div>
          <button
            type="button"
            onClick={() => setActiveView("jobs")}
            className="px-3 py-1 rounded-[4px] bg-emerald-800 text-white font-medium hover:bg-emerald-900 transition-colors flex items-center gap-1"
          >
            <span>Track in Job Dashboard</span>
            <ArrowRight weight="bold" className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {submitError && (
        <div className="p-4 rounded-[8px] bg-spot-redBg border border-[#E6C5C5] text-spot-redText text-xs font-mono flex items-center gap-2 animate-fadeIn">
          <WarningCircle weight="bold" className="w-4 h-4 shrink-0" />
          <span>{submitError}</span>
        </div>
      )}

      {/* Main Execution Split: Dynamic Form vs Live Code Generator */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: Dynamic Form Engine */}
        <div className="lg:col-span-7 p-5 rounded-[10px] bg-surface border border-hairline">
          {activeCapability ? (
            <DynamicSchemaForm
              capability={activeCapability}
              initialValues={formInputs}
              onChange={(vals) => setFormInputs(vals)}
              onSubmit={handleSubmitJob}
              isSubmitting={isSubmitting}
            />
          ) : (
            <div className="p-12 text-center text-xs font-mono text-ink-muted">
              Select a subsystem and capability to render the dynamic parameter form.
            </div>
          )}
        </div>

        {/* Right: Real-Time API Code Generator */}
        <div className="lg:col-span-5 space-y-4">
          <div className="p-4 rounded-[10px] bg-surface border border-hairline">
            <div className="flex items-center justify-between mb-3">
              <span className="flex items-center gap-1.5 text-xs font-mono font-semibold text-ink">
                <Code weight="bold" className="w-4 h-4" />
                <span>Live API Code Generator</span>
              </span>
              <span className="text-[10px] font-mono text-ink-muted">
                Reactive to form inputs
              </span>
            </div>
            <ApiCodeGenerator payload={currentPayload} />
          </div>
        </div>
      </div>
    </div>
  );
}
