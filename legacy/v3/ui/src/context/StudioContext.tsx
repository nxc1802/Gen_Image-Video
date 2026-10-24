"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { MemoryResponse, StudioViewId } from "@/types/studio";
import { StudioApiClient } from "@/lib/api-client";
import { cleanGatewayUrl } from "@/lib/utils";

const DEFAULT_TUNNEL_URL =
  "https://palace-dealing-optional-layer.trycloudflare.com";
const DEFAULT_NTFY_TOPIC = "studio-ai-url-cuongnguyen1802";

interface StudioContextValue {
  activeView: StudioViewId;
  setActiveView: (v: StudioViewId) => void;
  gatewayUrl: string;
  apiKey: string;
  ntfyTopic: string;
  connectionStatus: "online" | "offline" | "unconfigured";
  activeModalities: string[];
  loadedModels: string[];
  vramSummary: string;
  lastMemory: MemoryResponse | null;
  apiClient: StudioApiClient;
  refreshHealthAndMemory: (silent?: boolean) => Promise<void>;
  handleFreeAllVram: () => Promise<string>;
  saveConnectionSettings: (
    newUrl: string,
    newKey: string,
    newTopic: string
  ) => Promise<void>;
  discoverFromNtfy: (topicOverride?: string) => Promise<string>;
  isConnectionModalOpen: boolean;
  setIsConnectionModalOpen: (open: boolean) => void;
  isShutdownModalOpen: boolean;
  setIsShutdownModalOpen: (open: boolean) => void;
}

const StudioContext = createContext<StudioContextValue | null>(null);

export function StudioProvider({ children }: { children: React.ReactNode }) {
  const [activeView, setActiveView] = useState<StudioViewId>("audio");
  const [gatewayUrl, setGatewayUrl] = useState<string>(DEFAULT_TUNNEL_URL);
  const [apiKey, setApiKey] = useState<string>("");
  const [ntfyTopic, setNtfyTopic] = useState<string>(DEFAULT_NTFY_TOPIC);
  const [connectionStatus, setConnectionStatus] = useState<
    "online" | "offline" | "unconfigured"
  >("unconfigured");
  const [activeModalities, setActiveModalities] = useState<string[]>([]);
  const [loadedModels, setLoadedModels] = useState<string[]>([]);
  const [vramSummary, setVramSummary] = useState<string>("GPU0: -- | GPU1: --");
  const [lastMemory, setLastMemory] = useState<MemoryResponse | null>(null);

  const [isConnectionModalOpen, setIsConnectionModalOpen] =
    useState<boolean>(false);
  const [isShutdownModalOpen, setIsShutdownModalOpen] =
    useState<boolean>(false);

  useEffect(() => {
    const savedKey = localStorage.getItem("studio_api_key");
    const savedTopic = localStorage.getItem("studio_ntfy_topic");
    setGatewayUrl(DEFAULT_TUNNEL_URL);
    localStorage.setItem("studio_base_url", DEFAULT_TUNNEL_URL);
    if (savedKey) setApiKey(savedKey);
    if (savedTopic) setNtfyTopic(savedTopic);
  }, []);

  const apiClient = useMemo(
    () => new StudioApiClient(gatewayUrl || DEFAULT_TUNNEL_URL, apiKey),
    [gatewayUrl, apiKey]
  );

  const refreshHealthAndMemory = useCallback(
    async (silent = false) => {
      if (!gatewayUrl && silent) return;
      try {
        const [health, mem] = await Promise.all([
          apiClient.getHealth(),
          apiClient.getMemory().catch(() => null),
        ]);

        setConnectionStatus("online");

        // Extract modalities
        if (Array.isArray(health.active_modalities)) {
          setActiveModalities(health.active_modalities);
        } else if (health.catalog) {
          setActiveModalities(Object.keys(health.catalog));
        } else if (health.studio_mode) {
          setActiveModalities([health.studio_mode]);
        }

        // Extract loaded models
        const loaded: string[] = [];
        if (Array.isArray(health.loaded_models)) {
          loaded.push(...health.loaded_models);
        }
        if (health.catalog) {
          Object.values(health.catalog).forEach((entry) => {
            if ((entry.loaded || entry.preload) && entry.model_id) {
              loaded.push(entry.model_id);
            }
          });
        }
        if (health.studio_mode === "speech") {
          try {
            const diag = await apiClient.getAudioDiagnostics(false);
            const activeAudio: string[] = [];
            if (diag.models?.asr === "loaded") activeAudio.push("Qwen3-ASR-1.7B");
            if (diag.models?.tts === "loaded") activeAudio.push("Qwen3-TTS-1.7B");
            if (diag.models?.xtts === "loaded") activeAudio.push("XTTS-v2-VI");
            if (activeAudio.length > 0) {
              setLoadedModels(activeAudio);
            } else {
              setLoadedModels(loaded);
            }
          } catch {
            setLoadedModels(loaded);
          }
        } else {
          setLoadedModels(loaded);
        }

        if (mem) {
          setLastMemory(mem);
          if (mem.gpus && mem.gpus.length > 0) {
            const parts = mem.gpus.map((g, i) => {
              const id = g.gpu_id ?? i;
              const alloc = (g.allocated_gb ?? 0).toFixed(1);
              const total = (g.total_gb ?? 15).toFixed(1);
              return `G${id}: ${alloc}/${total}G`;
            });
            setVramSummary(parts.join(" · "));
          }
        }
      } catch {
        // If current URL is offline, try auto-discovering latest tunnel from ntfy once
        try {
          const latestUrl = await StudioApiClient.discoverNtfyTunnelUrl(
            ntfyTopic
          );
          const cleaned = cleanGatewayUrl(latestUrl);
          if (cleaned && cleaned !== gatewayUrl) {
            setGatewayUrl(cleaned);
            localStorage.setItem("studio_base_url", cleaned);
            return;
          }
        } catch {
          // ignore
        }
        setConnectionStatus(gatewayUrl ? "offline" : "unconfigured");
      }
    },
    [apiClient, gatewayUrl, ntfyTopic]
  );

  useEffect(() => {
    refreshHealthAndMemory(true);
    const timer = setInterval(() => refreshHealthAndMemory(true), 15000);
    return () => clearInterval(timer);
  }, [refreshHealthAndMemory]);

  const saveConnectionSettings = useCallback(
    async (newUrl: string, newKey: string, newTopic: string) => {
      const cleaned = cleanGatewayUrl(newUrl);
      setGatewayUrl(cleaned);
      setApiKey(newKey.trim());
      setNtfyTopic(newTopic.trim());
      localStorage.setItem("studio_base_url", cleaned);
      localStorage.setItem("studio_api_key", newKey.trim());
      localStorage.setItem("studio_ntfy_topic", newTopic.trim());

      const probeClient = new StudioApiClient(cleaned, newKey.trim());
      await probeClient.getHealth();
      setConnectionStatus("online");
      await refreshHealthAndMemory(false);
    },
    [refreshHealthAndMemory]
  );

  const discoverFromNtfy = useCallback(
    async (topicOverride?: string): Promise<string> => {
      const targetTopic = (topicOverride ?? ntfyTopic).trim();
      const discovered = await StudioApiClient.discoverNtfyTunnelUrl(
        targetTopic
      );
      const cleaned = cleanGatewayUrl(discovered);
      setGatewayUrl(cleaned);
      setNtfyTopic(targetTopic);
      localStorage.setItem("studio_base_url", cleaned);
      localStorage.setItem("studio_ntfy_topic", targetTopic);
      return cleaned;
    },
    [ntfyTopic]
  );

  const handleFreeAllVram = useCallback(async (): Promise<string> => {
    const res = await apiClient.freeAllVram();
    await refreshHealthAndMemory(true);
    return res.message || `VRAM reclaimed (${res.status})`;
  }, [apiClient, refreshHealthAndMemory]);

  return (
    <StudioContext.Provider
      value={{
        activeView,
        setActiveView,
        gatewayUrl,
        apiKey,
        ntfyTopic,
        connectionStatus,
        activeModalities,
        loadedModels,
        vramSummary,
        lastMemory,
        apiClient,
        refreshHealthAndMemory,
        handleFreeAllVram,
        saveConnectionSettings,
        discoverFromNtfy,
        isConnectionModalOpen,
        setIsConnectionModalOpen,
        isShutdownModalOpen,
        setIsShutdownModalOpen,
      }}
    >
      {children}
    </StudioContext.Provider>
  );
}

export function useStudio() {
  const ctx = useContext(StudioContext);
  if (!ctx) throw new Error("useStudio must be used within a StudioProvider");
  return ctx;
}
