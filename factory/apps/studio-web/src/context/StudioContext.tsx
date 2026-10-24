"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type {
  CapabilityManifest,
  FactoryJob,
  ModelManifest,
  ServiceManifest,
  StudioViewId,
  TelemetrySnapshot,
} from "@/types/studio";
import { StudioApiClient } from "@/lib/api-client";
import { cleanGatewayUrl } from "@/lib/utils";

const DEFAULT_CONTROL_URL =
  process.env.NEXT_PUBLIC_CONTROL_API_URL || "http://localhost:8000";
const DEFAULT_API_KEY =
  process.env.NEXT_PUBLIC_API_KEY || "sk-factory-studio";

interface StudioContextValue {
  activeView: StudioViewId;
  setActiveView: (v: StudioViewId) => void;
  gatewayUrl: string;
  apiKey: string;
  connectionStatus: "online" | "offline" | "connecting";
  services: ServiceManifest[];
  models: ModelManifest[];
  capabilities: CapabilityManifest[];
  jobs: FactoryJob[];
  telemetry: TelemetrySnapshot | null;
  globalError: string | null;
  setGlobalError: (err: string | null) => void;
  apiClient: StudioApiClient;
  refreshManifests: (silent?: boolean) => Promise<void>;
  refreshJobs: () => Promise<void>;
  saveConnectionSettings: (newUrl: string, newKey: string) => Promise<void>;
  isConnectionModalOpen: boolean;
  setIsConnectionModalOpen: (open: boolean) => void;
}

const StudioContext = createContext<StudioContextValue | null>(null);

export function StudioProvider({ children }: { children: React.ReactNode }) {
  const [activeView, setActiveView] = useState<StudioViewId>("console");
  const [gatewayUrl, setGatewayUrl] = useState<string>(DEFAULT_CONTROL_URL);
  const [apiKey, setApiKey] = useState<string>(DEFAULT_API_KEY);
  const [connectionStatus, setConnectionStatus] = useState<
    "online" | "offline" | "connecting"
  >("connecting");

  const [services, setServices] = useState<ServiceManifest[]>([]);
  const [models, setModels] = useState<ModelManifest[]>([]);
  const [capabilities, setCapabilities] = useState<CapabilityManifest[]>([]);
  const [jobs, setJobs] = useState<FactoryJob[]>([]);
  const [telemetry, setTelemetry] = useState<TelemetrySnapshot | null>(null);
  const [globalError, setGlobalError] = useState<string | null>(null);

  const [isConnectionModalOpen, setIsConnectionModalOpen] =
    useState<boolean>(false);

  // Restore saved connection settings from localStorage
  useEffect(() => {
    if (typeof window === "undefined") return;
    const savedUrl = localStorage.getItem("studio_control_url");
    const savedKey = localStorage.getItem("studio_api_key");

    if (savedUrl) {
      setGatewayUrl(savedUrl);
    }
    if (savedKey) {
      setApiKey(savedKey);
    }
  }, []);

  const apiClient = useMemo(
    () => new StudioApiClient(gatewayUrl || DEFAULT_CONTROL_URL, apiKey),
    [gatewayUrl, apiKey]
  );

  // Load and refresh manifests from backend
  const refreshManifests = useCallback(
    async (silent = false) => {
      try {
        if (!silent) setConnectionStatus("connecting");

        // Try /api/studio/schemas first (Feature 40), then fallback to individual endpoints
        let loadedServices: ServiceManifest[] = [];
        let loadedModels: ModelManifest[] = [];
        let loadedCapabilities: CapabilityManifest[] = [];

        try {
          const schemasResp = await apiClient.getSchemas();
          if (schemasResp?.schemas) {
            loadedServices = schemasResp.schemas.services || [];
            loadedModels = schemasResp.schemas.models || [];
          }
        } catch {
          // Fallback to separate endpoints
          const [sList, mList] = await Promise.all([
            apiClient.getServices().catch(() => []),
            apiClient.getModels().catch(() => []),
          ]);
          loadedServices = sList;
          loadedModels = mList;
        }

        try {
          loadedCapabilities = await apiClient.getCapabilities();
        } catch {
          loadedCapabilities = [];
        }

        setServices(loadedServices);
        setModels(loadedModels);
        setCapabilities(loadedCapabilities);
        setConnectionStatus("online");
        setGlobalError(null);
      } catch (err: any) {
        if (!silent) {
          setConnectionStatus("offline");
          setGlobalError(err?.message || "Failed to connect to Control Plane");
        }
      }
    },
    [apiClient]
  );

  // Refresh job list
  const refreshJobs = useCallback(async () => {
    try {
      const jobList = await apiClient.listJobs(50);
      setJobs(jobList);
    } catch {
      // ignore transient job list poll error
    }
  }, [apiClient]);

  // Initial load
  useEffect(() => {
    refreshManifests(false);
    refreshJobs();
  }, [refreshManifests, refreshJobs]);

  // Periodic poll of jobs (every 4 seconds)
  useEffect(() => {
    const timer = setInterval(() => {
      refreshJobs();
    }, 4000);
    return () => clearInterval(timer);
  }, [refreshJobs]);

  // Real-time telemetry inspector stream subscription
  useEffect(() => {
    const unsubscribe = apiClient.subscribeInspectorStream(
      (snapshot) => {
        setTelemetry(snapshot);
        setConnectionStatus("online");
      },
      () => {
        // stream dropped or reconnecting
      }
    );

    return () => unsubscribe();
  }, [apiClient]);

  // Save connection settings
  const saveConnectionSettings = useCallback(
    async (newUrl: string, newKey: string) => {
      const cleanedUrl = cleanGatewayUrl(newUrl, DEFAULT_CONTROL_URL);
      const cleanedKey = newKey.trim() || DEFAULT_API_KEY;

      setGatewayUrl(cleanedUrl);
      setApiKey(cleanedKey);

      if (typeof window !== "undefined") {
        localStorage.setItem("studio_control_url", cleanedUrl);
        localStorage.setItem("studio_api_key", cleanedKey);
      }

      setConnectionStatus("connecting");
      const probeClient = new StudioApiClient(cleanedUrl, cleanedKey);
      try {
        await probeClient.getHealth();
        setConnectionStatus("online");
        setGlobalError(null);
      } catch {
        // Health check failed
      }
      await refreshManifests(false);
      await refreshJobs();
    },
    [refreshManifests, refreshJobs]
  );

  return (
    <StudioContext.Provider
      value={{
        activeView,
        setActiveView,
        gatewayUrl,
        apiKey,
        connectionStatus,
        services,
        models,
        capabilities,
        jobs,
        telemetry,
        globalError,
        setGlobalError,
        apiClient,
        refreshManifests,
        refreshJobs,
        saveConnectionSettings,
        isConnectionModalOpen,
        setIsConnectionModalOpen,
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
