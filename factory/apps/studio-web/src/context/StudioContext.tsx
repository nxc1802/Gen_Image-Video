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
// Never embed a production credential in a NEXT_PUBLIC_* build variable.
const DEFAULT_API_KEY = "";

// Storage keys for browser storage
const STORAGE_KEYS = {
  CONTROL_URL: "studio_control_url",
  API_KEY: "studio_api_key",
  REMEMBER_KEY: "studio_remember_api_key",
} as const;

function safeGetStorage(storage: Storage | null, key: string): string | null {
  if (!storage) return null;
  try {
    return storage.getItem(key);
  } catch {
    return null;
  }
}

function safeSetStorage(
  storage: Storage | null,
  key: string,
  value: string
): void {
  if (!storage) return;
  try {
    storage.setItem(key, value);
  } catch {
    // Graceful fallback to in-memory state on quota or permission error
  }
}

function safeRemoveStorage(storage: Storage | null, key: string): void {
  if (!storage) return;
  try {
    storage.removeItem(key);
  } catch {
    // Ignore removal errors
  }
}

interface StudioContextValue {
  activeView: StudioViewId;
  setActiveView: (v: StudioViewId) => void;
  gatewayUrl: string;
  apiKey: string;
  persistApiKey: boolean;
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
  saveConnectionSettings: (
    newUrl: string,
    newKey: string,
    persistKey?: boolean
  ) => Promise<void>;
  isConnectionModalOpen: boolean;
  setIsConnectionModalOpen: (open: boolean) => void;
}

const StudioContext = createContext<StudioContextValue | null>(null);

export function StudioProvider({ children }: { children: React.ReactNode }) {
  const [activeView, setActiveView] = useState<StudioViewId>("console");
  const [gatewayUrl, setGatewayUrl] = useState<string>(DEFAULT_CONTROL_URL);
  const [apiKey, setApiKey] = useState<string>(DEFAULT_API_KEY);
  const [persistApiKey, setPersistApiKey] = useState<boolean>(false);
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

  // Restore saved connection settings from browser storage
  // Prioritizes session-scoped storage (sessionStorage) or explicit opt-in persistent storage (localStorage)
  useEffect(() => {
    if (typeof window === "undefined") return;

    // 1. Base URL is non-sensitive configuration, can be restored from localStorage
    const savedUrl = safeGetStorage(
      window.localStorage,
      STORAGE_KEYS.CONTROL_URL
    );
    if (savedUrl) {
      setGatewayUrl(savedUrl);
    }

    // 2. Determine API Key retrieval with security precedence
    const hasPersistOptIn =
      safeGetStorage(window.localStorage, STORAGE_KEYS.REMEMBER_KEY) === "true";
    const localSavedKey = safeGetStorage(
      window.localStorage,
      STORAGE_KEYS.API_KEY
    );
    const sessionSavedKey = safeGetStorage(
      window.sessionStorage,
      STORAGE_KEYS.API_KEY
    );

    if (sessionSavedKey) {
      // Active tab session has precedence
      setApiKey(sessionSavedKey);
      setPersistApiKey(hasPersistOptIn);
    } else if (hasPersistOptIn && localSavedKey) {
      // Explicit opt-in persistent storage
      setApiKey(localSavedKey);
      setPersistApiKey(true);
      safeSetStorage(
        window.sessionStorage,
        STORAGE_KEYS.API_KEY,
        localSavedKey
      );
    } else if (localSavedKey && !hasPersistOptIn) {
      // MIGRATION / CLEANUP: Legacy key was left in localStorage without explicit opt-in.
      // Migrate to session-scoped storage immediately and purge from permanent localStorage.
      setApiKey(localSavedKey);
      setPersistApiKey(false);
      safeSetStorage(
        window.sessionStorage,
        STORAGE_KEYS.API_KEY,
        localSavedKey
      );
      safeRemoveStorage(window.localStorage, STORAGE_KEYS.API_KEY);
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

        const [mList, sList, cList] = await Promise.all([
          apiClient.getModels().catch(() => []),
          apiClient.getServices().catch(() => []),
          apiClient.getCapabilities().catch(() => []),
        ]);

        setModels(mList);
        setServices(sList);
        setCapabilities(cList);
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
    async (newUrl: string, newKey: string, persistKey = false) => {
      const cleanedUrl = cleanGatewayUrl(newUrl, DEFAULT_CONTROL_URL);
      const cleanedKey = newKey.trim();

      setGatewayUrl(cleanedUrl);
      setApiKey(cleanedKey);
      setPersistApiKey(persistKey);

      if (typeof window !== "undefined") {
        safeSetStorage(
          window.localStorage,
          STORAGE_KEYS.CONTROL_URL,
          cleanedUrl
        );

        if (!cleanedKey) {
          // Clear credentials completely
          safeRemoveStorage(window.sessionStorage, STORAGE_KEYS.API_KEY);
          safeRemoveStorage(window.localStorage, STORAGE_KEYS.API_KEY);
          safeRemoveStorage(window.localStorage, STORAGE_KEYS.REMEMBER_KEY);
        } else if (persistKey) {
          // Explicit opt-in persistent storage
          safeSetStorage(
            window.sessionStorage,
            STORAGE_KEYS.API_KEY,
            cleanedKey
          );
          safeSetStorage(
            window.localStorage,
            STORAGE_KEYS.API_KEY,
            cleanedKey
          );
          safeSetStorage(
            window.localStorage,
            STORAGE_KEYS.REMEMBER_KEY,
            "true"
          );
        } else {
          // Default session-scoped storage: write to sessionStorage, purge from localStorage
          safeSetStorage(
            window.sessionStorage,
            STORAGE_KEYS.API_KEY,
            cleanedKey
          );
          safeRemoveStorage(window.localStorage, STORAGE_KEYS.API_KEY);
          safeRemoveStorage(window.localStorage, STORAGE_KEYS.REMEMBER_KEY);
        }
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
        persistApiKey,
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
