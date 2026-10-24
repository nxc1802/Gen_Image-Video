"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  UploadSimple,
  Sparkle,
  ArrowClockwise,
  WarningCircle,
  CheckCircle,
  FileText,
  Sliders,
  ToggleLeft,
  ToggleRight,
} from "@phosphor-icons/react";
import type { CapabilityManifest, ParameterSpec } from "@/types/studio";
import { useStudio } from "@/context/StudioContext";

interface DynamicSchemaFormProps {
  capability: CapabilityManifest;
  initialValues?: Record<string, any>;
  onChange?: (values: Record<string, any>) => void;
  onSubmit?: (values: Record<string, any>) => void;
  isSubmitting?: boolean;
}

export function DynamicSchemaForm({
  capability,
  initialValues = {},
  onChange,
  onSubmit,
  isSubmitting = false,
}: DynamicSchemaFormProps) {
  const { apiClient } = useStudio();
  const [formValues, setFormValues] = useState<Record<string, any>>(initialValues);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [uploadingField, setUploadingField] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<Record<string, string>>({});

  // Sync when capability changes or initialValues change
  useEffect(() => {
    const defaults: Record<string, any> = {};
    for (const spec of capability.inputs || []) {
      if (initialValues[spec.name] !== undefined) {
        defaults[spec.name] = initialValues[spec.name];
      } else if (spec.default !== undefined) {
        defaults[spec.name] = spec.default;
      } else if (spec.type === "boolean") {
        defaults[spec.name] = false;
      } else if (spec.constraints?.choices && spec.constraints.choices.length > 0) {
        defaults[spec.name] = spec.constraints.choices[0];
      } else if (spec.type === "integer" || spec.type === "float") {
        defaults[spec.name] = spec.constraints?.min_value ?? 0;
      } else {
        defaults[spec.name] = "";
      }
    }
    setFormValues(defaults);
    setErrors({});
    if (onChange) onChange(defaults);
  }, [capability.id]);

  const updateField = useCallback(
    (name: string, value: any) => {
      setFormValues((prev) => {
        const next = { ...prev, [name]: value };
        if (onChange) onChange(next);
        return next;
      });
      // Clear error on edit
      setErrors((prev) => {
        if (!prev[name]) return prev;
        const copy = { ...prev };
        delete copy[name];
        return copy;
      });
    },
    [onChange]
  );

  // Load Example Payload
  const handleLoadExample = () => {
    const example: Record<string, any> = {};
    for (const spec of capability.inputs || []) {
      if (spec.default !== undefined) {
        example[spec.name] = spec.default;
      } else if (spec.name === "prompt") {
        example[spec.name] =
          "Cinematic shot of a high-tech robotic factory assembling AI chips, volumetric lighting, photorealistic, 8k resolution";
      } else if (spec.name === "negative_prompt") {
        example[spec.name] = "blurry, distorted, low quality, artifacts";
      } else if (spec.type === "image_url" || spec.type === "mask_url") {
        example[spec.name] = "https://assets.ai.example.com/samples/factory_robot.png";
      } else if (spec.type === "audio_url") {
        example[spec.name] = "https://assets.ai.example.com/samples/speech_sample.wav";
      } else if (spec.type === "video_url") {
        example[spec.name] = "https://assets.ai.example.com/samples/factory_pan.mp4";
      } else if (spec.constraints?.choices && spec.constraints.choices.length > 0) {
        example[spec.name] = spec.constraints.choices[0];
      } else if (spec.type === "integer" || spec.type === "float") {
        example[spec.name] = spec.constraints?.min_value ?? 1;
      } else if (spec.type === "boolean") {
        example[spec.name] = false;
      } else {
        example[spec.name] = "";
      }
    }
    setFormValues(example);
    setErrors({});
    if (onChange) onChange(example);
  };

  // Reset form
  const handleReset = () => {
    const empty: Record<string, any> = {};
    for (const spec of capability.inputs || []) {
      empty[spec.name] = spec.type === "boolean" ? false : "";
    }
    setFormValues(empty);
    setErrors({});
    if (onChange) onChange(empty);
  };

  // Asset upload handler
  const handleFileUpload = async (fieldName: string, file: File) => {
    setUploadingField(fieldName);
    try {
      const resp = await apiClient.uploadAsset(file);
      updateField(fieldName, resp.url);
      setUploadSuccess((prev) => ({
        ...prev,
        [fieldName]: `Uploaded (${(resp.size_bytes / 1024).toFixed(1)} KB)`,
      }));
      setTimeout(() => {
        setUploadSuccess((prev) => {
          const c = { ...prev };
          delete c[fieldName];
          return c;
        });
      }, 4000);
    } catch (err: any) {
      setErrors((prev) => ({
        ...prev,
        [fieldName]: err?.message || "Asset upload failed",
      }));
    } finally {
      setUploadingField(null);
    }
  };

  // Validate and submit
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const newErrors: Record<string, string> = {};

    for (const spec of capability.inputs || []) {
      const val = formValues[spec.name];

      if (spec.required && (val === undefined || val === null || val === "")) {
        newErrors[spec.name] = `${spec.name} is required`;
        continue;
      }

      if (spec.type === "integer" || spec.type === "float") {
        const numVal = Number(val);
        if (isNaN(numVal)) {
          if (spec.required) newErrors[spec.name] = "Must be a valid number";
        } else {
          if (
            spec.constraints?.min_value !== undefined &&
            numVal < spec.constraints.min_value
          ) {
            newErrors[
              spec.name
            ] = `Minimum allowed value is ${spec.constraints.min_value}`;
          }
          if (
            spec.constraints?.max_value !== undefined &&
            numVal > spec.constraints.max_value
          ) {
            newErrors[
              spec.name
            ] = `Maximum allowed value is ${spec.constraints.max_value}`;
          }
        }
      }

      if (
        (spec.type === "image_url" ||
          spec.type === "audio_url" ||
          spec.type === "video_url") &&
        typeof val === "string" &&
        val.startsWith("data:")
      ) {
        newErrors[
          spec.name
        ] = "Direct base64 is forbidden; upload file or provide S3 storage URL.";
      }
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    if (onSubmit) {
      onSubmit(formValues);
    }
  };

  // Render individual field dispatcher
  const renderField = (spec: ParameterSpec) => {
    const val = formValues[spec.name];
    const error = errors[spec.name];
    const isPrompt =
      spec.name === "prompt" ||
      spec.name === "negative_prompt" ||
      (spec.constraints?.max_length && spec.constraints.max_length > 200);

    const hasChoices =
      spec.type === "choices" ||
      (spec.constraints?.choices && spec.constraints.choices.length > 0);

    return (
      <div
        key={spec.name}
        className="space-y-1.5 p-3 rounded-[8px] bg-canvas border border-hairline transition-all hover:border-ink/20"
      >
        <div className="flex items-center justify-between">
          <label className="text-xs font-mono font-medium text-ink flex items-center gap-1.5">
            <span>{spec.name}</span>
            {spec.required && (
              <span className="text-spot-redText text-[10px] font-bold" title="Required">
                *
              </span>
            )}
            <span className="text-[10px] text-ink-muted font-normal">
              ({spec.type})
            </span>
          </label>
          {spec.constraints && (spec.constraints.min_value !== undefined || spec.constraints.max_value !== undefined) && (
            <span className="text-[10px] font-mono text-ink-muted">
              [{spec.constraints.min_value ?? "-∞"} .. {spec.constraints.max_value ?? "+∞"}]
              {spec.constraints.step ? ` step ${spec.constraints.step}` : ""}
            </span>
          )}
        </div>

        {spec.description && (
          <p className="text-[11px] text-ink-secondary leading-snug">
            {spec.description}
          </p>
        )}

        {/* Dropdown choices */}
        {hasChoices ? (
          <select
            value={val ?? ""}
            onChange={(e) => updateField(spec.name, e.target.value)}
            className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-surface text-xs font-mono text-ink focus:outline-none focus:ring-1 focus:ring-ink"
          >
            {(spec.constraints?.choices || []).map((choice) => (
              <option key={String(choice)} value={String(choice)}>
                {String(choice)}
              </option>
            ))}
          </select>
        ) : spec.type === "boolean" ? (
          /* Boolean Toggle */
          <div className="flex items-center gap-2 pt-1">
            <button
              type="button"
              onClick={() => updateField(spec.name, !val)}
              className="flex items-center gap-2 text-xs font-mono text-ink focus:outline-none"
            >
              {val ? (
                <ToggleRight weight="fill" className="w-6 h-6 text-ink" />
              ) : (
                <ToggleLeft weight="regular" className="w-6 h-6 text-ink-muted" />
              )}
              <span>{val ? "Enabled (true)" : "Disabled (false)"}</span>
            </button>
          </div>
        ) : spec.type === "integer" || spec.type === "float" ? (
          /* Number with Slider */
          <div className="space-y-2 pt-1">
            <div className="flex items-center gap-3">
              <input
                type="range"
                min={spec.constraints?.min_value ?? 0}
                max={spec.constraints?.max_value ?? 100}
                step={spec.constraints?.step ?? (spec.type === "float" ? 0.05 : 1)}
                value={Number(val ?? spec.constraints?.min_value ?? 0)}
                onChange={(e) =>
                  updateField(
                    spec.name,
                    spec.type === "integer"
                      ? parseInt(e.target.value, 10)
                      : parseFloat(e.target.value)
                  )
                }
                className="flex-1 accent-ink cursor-pointer"
              />
              <input
                type="number"
                min={spec.constraints?.min_value}
                max={spec.constraints?.max_value}
                step={spec.constraints?.step ?? (spec.type === "float" ? 0.05 : 1)}
                value={val ?? ""}
                onChange={(e) =>
                  updateField(
                    spec.name,
                    spec.type === "integer"
                      ? parseInt(e.target.value, 10)
                      : parseFloat(e.target.value)
                  )
                }
                className="w-24 px-2 py-1 text-right rounded-[6px] border border-hairline bg-surface text-xs font-mono text-ink"
              />
            </div>
          </div>
        ) : spec.type === "image_url" ||
          spec.type === "mask_url" ||
          spec.type === "audio_url" ||
          spec.type === "video_url" ? (
          /* Media Asset Uploader & Preview */
          <div className="space-y-2 pt-1">
            <div className="flex gap-2">
              <input
                type="text"
                placeholder={`https://... or upload local file below`}
                value={val ?? ""}
                onChange={(e) => updateField(spec.name, e.target.value)}
                className="flex-1 px-3 py-1.5 rounded-[6px] border border-hairline bg-surface text-xs font-mono text-ink"
              />
              <label
                className={`px-3 py-1.5 rounded-[6px] border border-hairline bg-subtle hover:bg-muted text-xs font-mono text-ink flex items-center gap-1.5 cursor-pointer shrink-0 ${
                  uploadingField === spec.name ? "opacity-50 pointer-events-none" : ""
                }`}
              >
                <UploadSimple weight="bold" className="w-3.5 h-3.5" />
                <span>{uploadingField === spec.name ? "Uploading..." : "Upload File"}</span>
                <input
                  type="file"
                  className="hidden"
                  accept={
                    spec.type === "audio_url"
                      ? "audio/*"
                      : spec.type === "video_url"
                      ? "video/*"
                      : "image/*"
                  }
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleFileUpload(spec.name, f);
                  }}
                />
              </label>
            </div>

            {uploadSuccess[spec.name] && (
              <div className="text-[11px] font-mono text-spot-greenText flex items-center gap-1">
                <CheckCircle weight="bold" className="w-3.5 h-3.5" />
                <span>{uploadSuccess[spec.name]}</span>
              </div>
            )}

            {/* Media Preview */}
            {val && typeof val === "string" && val.startsWith("http") && (
              <div className="p-2 rounded-[6px] bg-subtle border border-hairline inline-block max-w-full">
                {(spec.type === "image_url" || spec.type === "mask_url") && (
                  <img
                    src={val}
                    alt={spec.name}
                    className="max-h-36 rounded-[4px] object-contain border border-hairline"
                  />
                )}
                {spec.type === "audio_url" && (
                  <audio controls src={val} className="h-8 max-w-xs" />
                )}
                {spec.type === "video_url" && (
                  <video
                    controls
                    src={val}
                    className="max-h-40 rounded-[4px] border border-hairline"
                  />
                )}
              </div>
            )}
          </div>
        ) : isPrompt ? (
          /* Multiline Auto-resizing Textarea */
          <div className="space-y-1">
            <textarea
              rows={3}
              value={val ?? ""}
              placeholder={`Enter ${spec.name}...`}
              onChange={(e) => updateField(spec.name, e.target.value)}
              className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-surface text-xs font-sans text-ink focus:outline-none focus:ring-1 focus:ring-ink"
            />
            <div className="text-right text-[10px] font-mono text-ink-muted">
              {String(val || "").length} chars
            </div>
          </div>
        ) : (
          /* Standard String or Fallback Input */
          <input
            type="text"
            value={val ?? ""}
            placeholder={`Enter ${spec.name}...`}
            onChange={(e) => updateField(spec.name, e.target.value)}
            className="w-full px-3 py-2 rounded-[6px] border border-hairline bg-surface text-xs font-mono text-ink focus:outline-none focus:ring-1 focus:ring-ink"
          />
        )}

        {error && (
          <div className="text-[11px] font-mono text-spot-redText flex items-center gap-1">
            <WarningCircle weight="bold" className="w-3.5 h-3.5" />
            <span>{error}</span>
          </div>
        )}
      </div>
    );
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Top action bar: Example preset, Reset */}
      <div className="flex items-center justify-between pb-2 border-b border-hairline">
        <div className="flex items-center gap-2">
          <Sliders weight="bold" className="w-4 h-4 text-ink" />
          <span className="text-xs font-mono font-semibold uppercase tracking-wider text-ink">
            {capability.display_name} Parameters
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleLoadExample}
            className="px-2.5 py-1 rounded-[6px] border border-hairline bg-surface hover:bg-subtle text-xs font-mono text-ink flex items-center gap-1.5 transition-transform active:scale-[0.98]"
          >
            <Sparkle weight="bold" className="w-3.5 h-3.5 text-amber-500" />
            <span>Load Example</span>
          </button>
          <button
            type="button"
            onClick={handleReset}
            className="px-2.5 py-1 rounded-[6px] border border-hairline bg-surface hover:bg-subtle text-xs font-mono text-ink-muted hover:text-ink flex items-center gap-1 transition-transform active:scale-[0.98]"
          >
            <ArrowClockwise weight="bold" className="w-3.5 h-3.5" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Field List */}
      <div className="space-y-3">
        {(capability.inputs || []).map(renderField)}
      </div>

      {/* Submit Button */}
      <div className="pt-2">
        <button
          type="submit"
          disabled={isSubmitting}
          className="w-full py-2.5 rounded-[6px] bg-ink hover:bg-[#2a2a2a] text-surface text-xs font-semibold uppercase tracking-wider flex items-center justify-center gap-2 transition-transform active:scale-[0.99] disabled:opacity-50"
        >
          {isSubmitting ? (
            <>
              <ArrowClockwise weight="bold" className="w-4 h-4 animate-spin" />
              <span>Submitting to Queue...</span>
            </>
          ) : (
            <>
              <FileText weight="bold" className="w-4 h-4" />
              <span>Submit Factory Job</span>
            </>
          )}
        </button>
      </div>
    </form>
  );
}
