"use client";

import React, { useState, useEffect } from "react";
import {
  Copy,
  Check,
  TerminalWindow,
  Code,
  FileCode,
  Lightning,
} from "@phosphor-icons/react";
import type { CodegenResult, JobSubmissionPayload } from "@/types/studio";
import { useStudio } from "@/context/StudioContext";
import { generateClientSnippets } from "@/lib/codegen";

interface ApiCodeGeneratorProps {
  payload: JobSubmissionPayload;
  className?: string;
}

type TargetTab = "curl" | "python" | "typescript" | "openai";

export function ApiCodeGenerator({ payload, className = "" }: ApiCodeGeneratorProps) {
  const { apiClient, gatewayUrl, apiKey } = useStudio();
  const [activeTab, setActiveTab] = useState<TargetTab>("curl");
  const [snippets, setSnippets] = useState<CodegenResult>(() =>
    generateClientSnippets(payload, gatewayUrl, apiKey)
  );
  const [copied, setCopied] = useState<boolean>(false);

  // Re-generate snippets when payload, gateway, or apiKey changes
  useEffect(() => {
    let isCancelled = false;

    // Immediately generate client-side snippet for zero lag
    const local = generateClientSnippets(payload, gatewayUrl, apiKey);
    setSnippets(local);

    // Also attempt remote codegen endpoint asynchronously
    apiClient
      .generateCode(payload)
      .then((remote) => {
        if (!isCancelled && remote) {
          setSnippets({
            curl: remote.curl || local.curl,
            python: remote.python || local.python,
            typescript: remote.typescript || local.typescript,
            openai: remote.openai || local.openai,
          });
        }
      })
      .catch(() => {
        // Keep local fallback
      });

    return () => {
      isCancelled = true;
    };
  }, [payload, gatewayUrl, apiKey, apiClient]);

  const currentCode = snippets[activeTab] || snippets.curl || "";

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(currentCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
    }
  };

  const tabs: Array<{ id: TargetTab; label: string; icon: React.ComponentType<any> }> = [
    { id: "curl", label: "cURL", icon: TerminalWindow },
    { id: "python", label: "Python (httpx)", icon: Code },
    { id: "typescript", label: "TypeScript (fetch)", icon: FileCode },
    { id: "openai", label: "OpenAI SDK", icon: Lightning },
  ];

  return (
    <div
      className={`rounded-[10px] bg-[#141517] border border-[#2B2D31] text-[#E0E2E6] flex flex-col overflow-hidden ${className}`}
    >
      {/* Top Header & Language Tabs */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-[#2B2D31] bg-[#1A1B1E]">
        <div className="flex items-center gap-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`px-2.5 py-1 rounded-[5px] text-xs font-mono flex items-center gap-1.5 transition-colors ${
                  isActive
                    ? "bg-[#2C2E33] text-white font-medium"
                    : "text-[#909296] hover:text-[#C1C2C5] hover:bg-[#25262B]"
                }`}
              >
                <Icon weight="bold" className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Copy Button with Toast Feedback */}
        <button
          type="button"
          onClick={handleCopy}
          className="px-2.5 py-1 rounded-[5px] bg-[#2C2E33] hover:bg-[#373A40] text-xs font-mono text-white flex items-center gap-1.5 transition-transform active:scale-[0.98]"
        >
          {copied ? (
            <>
              <Check weight="bold" className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-emerald-400">Copied!</span>
            </>
          ) : (
            <>
              <Copy weight="bold" className="w-3.5 h-3.5 text-[#C1C2C5]" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Code Display Area */}
      <div className="relative p-3 overflow-x-auto max-h-[380px] bg-[#111214] font-mono text-xs leading-relaxed text-[#D5D7E0] selection:bg-[#3B4252]">
        <pre className="whitespace-pre">
          <code>{currentCode}</code>
        </pre>
      </div>

      {/* Footer hint */}
      <div className="px-3 py-1.5 border-t border-[#2B2D31] bg-[#16171A] text-[10px] font-mono text-[#82858F] flex items-center justify-between">
        <span>Target: POST /v1/jobs (Two-Tier Factory Queue)</span>
        <span>Auth: Mandatory Bearer Token</span>
      </div>
    </div>
  );
}
