"use client";

import React, { useState } from "react";
import {
  Cube,
  Cpu,
  Globe,
  CheckCircle,
  WarningCircle,
  Sparkle,
  HardDrives,
  ListDashes,
} from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";

type RegistryTab = "services" | "models" | "capabilities" | "hardware";

export function ModelRegistryView() {
  const { services, models, capabilities } = useStudio();
  const [activeTab, setActiveTab] = useState<RegistryTab>("services");

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="p-5 rounded-[10px] bg-surface border border-hairline">
        <h1 className="font-serif text-2xl text-ink">Model &amp; Service Registry</h1>
        <p className="text-xs text-ink-muted mt-1">
          Declarative service topologies, model manifests, hardware boundaries, and capability coverage matrices.
        </p>

        {/* Tab Strip */}
        <div className="flex flex-wrap gap-2 mt-4 pt-4 border-t border-hairline">
          <button
            type="button"
            onClick={() => setActiveTab("services")}
            className={`px-3 py-1.5 rounded-full text-xs font-mono transition-colors flex items-center gap-1.5 ${
              activeTab === "services"
                ? "bg-ink text-surface font-semibold"
                : "bg-subtle text-ink-secondary hover:bg-muted hover:text-ink"
            }`}
          >
            <Globe weight="bold" className="w-3.5 h-3.5" />
            <span>Services &amp; Edge Domains ({services.length})</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab("models")}
            className={`px-3 py-1.5 rounded-full text-xs font-mono transition-colors flex items-center gap-1.5 ${
              activeTab === "models"
                ? "bg-ink text-surface font-semibold"
                : "bg-subtle text-ink-secondary hover:bg-muted hover:text-ink"
            }`}
          >
            <Cube weight="bold" className="w-3.5 h-3.5" />
            <span>Models &amp; Weights ({models.length})</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab("capabilities")}
            className={`px-3 py-1.5 rounded-full text-xs font-mono transition-colors flex items-center gap-1.5 ${
              activeTab === "capabilities"
                ? "bg-ink text-surface font-semibold"
                : "bg-subtle text-ink-secondary hover:bg-muted hover:text-ink"
            }`}
          >
            <Sparkle weight="bold" className="w-3.5 h-3.5" />
            <span>Capability Matrix ({capabilities.length})</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab("hardware")}
            className={`px-3 py-1.5 rounded-full text-xs font-mono transition-colors flex items-center gap-1.5 ${
              activeTab === "hardware"
                ? "bg-ink text-surface font-semibold"
                : "bg-subtle text-ink-secondary hover:bg-muted hover:text-ink"
            }`}
          >
            <Cpu weight="bold" className="w-3.5 h-3.5" />
            <span>Hardware Topologies</span>
          </button>
        </div>
      </div>

      {/* Tab 1: Services */}
      {activeTab === "services" && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {services.map((svc) => (
            <div
              key={svc.id}
              className="p-4 rounded-[10px] bg-surface border border-hairline space-y-3"
            >
              <div className="flex items-center justify-between pb-2 border-b border-hairline">
                <span className="text-xs font-mono font-bold text-ink">
                  {svc.name}
                </span>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-spot-greenBg text-spot-greenText">
                  v{svc.version}
                </span>
              </div>

              <div className="space-y-1.5 text-xs font-mono text-ink-secondary">
                <div className="flex justify-between">
                  <span className="text-ink-muted">Subsystem:</span>
                  <span className="text-ink font-semibold">{svc.subsystem}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-ink-muted">Static Domain:</span>
                  <span className="text-ink font-mono text-[11px] truncate max-w-[180px]">
                    {svc.network.static_domain}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-ink-muted">Internal Port:</span>
                  <span className="text-ink">{svc.network.port}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-ink-muted">Serving Framework:</span>
                  <span className="text-ink">{svc.framework || "vLLM / ComfyUI"}</span>
                </div>
                {svc.hardware && (
                  <div className="flex justify-between">
                    <span className="text-ink-muted">Hardware Binding:</span>
                    <span className="text-ink">
                      {svc.hardware.gpu || "GPU"} (≥{svc.hardware.min_vram_gb || 14}GB)
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Tab 2: Models */}
      {activeTab === "models" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {models.map((model) => (
            <div
              key={model.id}
              className="p-5 rounded-[10px] bg-surface border border-hairline space-y-4"
            >
              <div className="flex items-center justify-between pb-2 border-b border-hairline">
                <div>
                  <h3 className="text-sm font-mono font-bold text-ink">
                    {model.display_name || model.id}
                  </h3>
                  <span className="text-[11px] font-mono text-ink-muted">
                    Family: {model.family} · Service: {model.service_id}
                  </span>
                </div>
                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-spot-greenBg text-spot-greenText border border-[#C2DEC4]">
                  {model.status.toUpperCase()} ({model.coverage_percent ?? 100}%)
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="p-2.5 rounded bg-canvas border border-hairline">
                  <span className="text-[10px] text-ink-muted block">Min VRAM</span>
                  <span className="font-semibold text-ink">
                    {model.hardware.min_vram_gb} GB
                  </span>
                </div>
                <div className="p-2.5 rounded bg-canvas border border-hairline">
                  <span className="text-[10px] text-ink-muted block">Preferred GPU</span>
                  <span className="font-semibold text-ink">
                    {model.hardware.preferred_gpu || "NVIDIA Tesla T4"}
                  </span>
                </div>
              </div>

              {/* Declared Capabilities */}
              <div className="space-y-1.5">
                <div className="text-[11px] font-mono font-semibold uppercase tracking-wider text-ink-muted">
                  Declared Capabilities
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(model.capabilities || {}).map(([cap, supported]) => (
                    <span
                      key={cap}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono flex items-center gap-1 ${
                        supported
                          ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                          : "bg-subtle text-ink-muted border border-hairline"
                      }`}
                    >
                      {supported ? (
                        <CheckCircle weight="bold" className="w-3 h-3 text-emerald-600" />
                      ) : (
                        <WarningCircle weight="bold" className="w-3 h-3 text-ink-muted" />
                      )}
                      <span>{cap}</span>
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Tab 3: Capabilities Matrix */}
      {activeTab === "capabilities" && (
        <div className="p-5 rounded-[10px] bg-surface border border-hairline overflow-x-auto space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-mono font-bold text-ink">
              Capability Contract &amp; Schema Matrix
            </h3>
            <span className="text-xs font-mono text-ink-muted">
              {capabilities.length} total capabilities
            </span>
          </div>

          <table className="w-full text-left text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-hairline text-[11px] text-ink-muted uppercase tracking-wider">
                <th className="py-2.5 px-3">Capability ID</th>
                <th className="py-2.5 px-3">Display Name</th>
                <th className="py-2.5 px-3">Subsystem</th>
                <th className="py-2.5 px-3">Inputs</th>
                <th className="py-2.5 px-3">Supported Model Families</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-hairline">
              {capabilities.map((cap) => (
                <tr key={cap.id} className="hover:bg-subtle transition-colors">
                  <td className="py-2.5 px-3 font-semibold text-ink">{cap.id}</td>
                  <td className="py-2.5 px-3 text-ink-secondary">{cap.display_name}</td>
                  <td className="py-2.5 px-3">
                    <span className="px-2 py-0.5 rounded text-[10px] bg-subtle text-ink">
                      {cap.subsystem}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-ink-muted">
                    {(cap.inputs || []).map((i) => i.name).join(", ") || "none"}
                  </td>
                  <td className="py-2.5 px-3 text-ink font-medium">
                    {(cap.supported_model_families || []).join(", ") || "*"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Tab 4: Hardware Topologies */}
      {activeTab === "hardware" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-5 rounded-[10px] bg-surface border border-hairline space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-hairline">
              <span className="text-sm font-mono font-bold text-ink">
                Kaggle Dual-T4 Topology
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-spot-greenBg text-spot-greenText font-bold">
                PRIMARY TARGET
              </span>
            </div>
            <div className="space-y-2 text-xs font-mono text-ink-secondary">
              <div className="p-3 rounded bg-canvas border border-hairline space-y-1">
                <div className="font-semibold text-ink">GPU 0 (16GB PCIe)</div>
                <div>Role: Headless ComfyUI (Image Pipeline) &amp; Fast TTS/ASR</div>
                <div className="text-[10px] text-ink-muted">Allocated: Qwen-Image-2.1, Qwen3-Speech</div>
              </div>
              <div className="p-3 rounded bg-canvas border border-hairline space-y-1">
                <div className="font-semibold text-ink">GPU 1 (16GB PCIe)</div>
                <div>Role: Headless ComfyUI (Wan2.2 Video Pipeline) &amp; vLLM Text</div>
                <div className="text-[10px] text-ink-muted">Allocated: Wan2.2 DiT, Qwen-2.5-72B (AWQ)</div>
              </div>
            </div>
          </div>

          <div className="p-5 rounded-[10px] bg-surface border border-hairline space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-hairline">
              <span className="text-sm font-mono font-bold text-ink">
                Colab Single-T4 Topology
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-amber-50 text-amber-800 font-bold">
                DEV FALLBACK
              </span>
            </div>
            <div className="space-y-2 text-xs font-mono text-ink-secondary">
              <div className="p-3 rounded bg-canvas border border-hairline space-y-1">
                <div className="font-semibold text-ink">GPU 0 (16GB PCIe)</div>
                <div>Role: Unified sequential worker with aggressive VRAM offloading</div>
                <div className="text-[10px] text-ink-muted">Memory Pinning: Low-VRAM offload enabled</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
