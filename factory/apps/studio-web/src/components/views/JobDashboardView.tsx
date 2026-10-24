"use client";

import React, { useState, useMemo } from "react";
import {
  Stack,
  ArrowsClockwise,
  Prohibit,
  ArrowCounterClockwise,
  DownloadSimple,
  CheckCircle,
  WarningCircle,
  Clock,
  Eye,
  FileCode,
  X,
  Play,
} from "@phosphor-icons/react";
import { useStudio } from "@/context/StudioContext";
import type { FactoryJob, JobState } from "@/types/studio";

const LIFECYCLE_STEPS: JobState[] = [
  "CREATED",
  "QUEUED",
  "ADMITTED",
  "RUNNING",
  "STREAMING",
  "SUCCEEDED",
];

export function JobDashboardView() {
  const { jobs, refreshJobs, apiClient, setActiveView } = useStudio();
  const [filterState, setFilterState] = useState<string>("ALL");
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<boolean>(false);
  const [actionNotice, setActionNotice] = useState<string | null>(null);

  // Filtered jobs
  const filteredJobs = useMemo(() => {
    if (filterState === "ALL") return jobs;
    if (filterState === "RUNNING") {
      return jobs.filter((j) => ["ADMITTED", "RUNNING", "STREAMING"].includes(j.state));
    }
    return jobs.filter((j) => j.state === filterState);
  }, [jobs, filterState]);

  const selectedJob = useMemo(() => {
    return jobs.find((j) => j.job_id === selectedJobId) || null;
  }, [jobs, selectedJobId]);

  // Cancel job
  const handleCancelJob = async (jobId: string) => {
    setActionLoading(true);
    setActionNotice(null);
    try {
      await apiClient.cancelJob(jobId);
      setActionNotice(`Job ${jobId} cancellation requested.`);
      await refreshJobs();
    } catch (err: any) {
      setActionNotice(`Failed to cancel job: ${err?.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  // Retry job
  const handleRetryJob = async (jobId: string) => {
    setActionLoading(true);
    setActionNotice(null);
    try {
      const resp = await apiClient.retryJob(jobId);
      setActionNotice(`Job ${jobId} scheduled for retry. New state: ${resp.status}`);
      await refreshJobs();
    } catch (err: any) {
      setActionNotice(`Failed to retry job: ${err?.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  // State badge styling
  const renderStateBadge = (state: JobState) => {
    switch (state) {
      case "SUCCEEDED":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-spot-greenBg text-spot-greenText border border-[#C2DEC4]">
            SUCCEEDED
          </span>
        );
      case "FAILED":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-spot-redBg text-spot-redText border border-[#E6C5C5]">
            FAILED
          </span>
        );
      case "CANCELLED":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-gray-100 text-gray-700 border border-gray-300">
            CANCELLED
          </span>
        );
      case "RUNNING":
      case "STREAMING":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-blue-50 text-blue-700 border border-blue-200 animate-pulse">
            {state}
          </span>
        );
      case "QUEUED":
      case "ADMITTED":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-50 text-amber-700 border border-amber-200">
            {state}
          </span>
        );
      default:
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-subtle text-ink-secondary border border-hairline">
            {state}
          </span>
        );
    }
  };

  // Render State Machine Visualizer
  const renderLifecycleTimeline = (job: FactoryJob) => {
    const isTerminalFailed = job.state === "FAILED" || job.state === "CANCELLED";
    const currentIndex = LIFECYCLE_STEPS.indexOf(job.state);

    return (
      <div className="p-4 rounded-[8px] bg-canvas border border-hairline space-y-2">
        <div className="text-[11px] font-mono font-semibold uppercase tracking-wider text-ink-muted">
          State Machine Progression
        </div>
        <div className="flex items-center gap-1 sm:gap-2 overflow-x-auto py-2">
          {LIFECYCLE_STEPS.map((step, idx) => {
            const isCompleted = currentIndex > idx || job.state === "SUCCEEDED";
            const isCurrent = job.state === step;
            return (
              <React.Fragment key={step}>
                <div
                  className={`px-2.5 py-1 rounded-[5px] text-[10px] font-mono font-semibold whitespace-nowrap transition-colors ${
                    isCurrent
                      ? isTerminalFailed
                        ? "bg-spot-redBg text-spot-redText border border-[#E6C5C5]"
                        : "bg-ink text-surface"
                      : isCompleted
                      ? "bg-spot-greenBg text-spot-greenText border border-[#C2DEC4]"
                      : "bg-subtle text-ink-muted border border-hairline"
                  }`}
                >
                  {step}
                </div>
                {idx < LIFECYCLE_STEPS.length - 1 && (
                  <span className="text-ink-muted font-mono text-xs">→</span>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 rounded-[10px] bg-surface border border-hairline">
        <div>
          <h1 className="font-serif text-2xl text-ink">
            Two-Tier Queue Job Dashboard
          </h1>
          <p className="text-xs text-ink-muted mt-1">
            Real-time tracking of admitted jobs, worker leases, state transitions, and presigned S3 artifact downloads.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => refreshJobs()}
            className="px-3 py-1.5 rounded-[6px] border border-hairline bg-surface hover:bg-subtle text-ink text-xs font-mono flex items-center gap-1.5 transition-transform active:scale-[0.98]"
          >
            <ArrowsClockwise weight="bold" className="w-3.5 h-3.5" />
            <span>Refresh Queue</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveView("explorer")}
            className="px-3 py-1.5 rounded-[6px] bg-ink text-surface hover:bg-[#2a2a2a] text-xs font-medium flex items-center gap-1.5 transition-transform active:scale-[0.98]"
          >
            <span>+ Submit New Job</span>
          </button>
        </div>
      </div>

      {actionNotice && (
        <div className="p-3 rounded-[6px] bg-subtle border border-hairline text-xs font-mono text-ink flex items-center justify-between">
          <span>{actionNotice}</span>
          <button onClick={() => setActionNotice(null)} className="text-ink-muted hover:text-ink">
            <X weight="bold" className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Filter Tabs */}
      <div className="flex items-center gap-2 border-b border-hairline pb-2">
        {["ALL", "RUNNING", "QUEUED", "SUCCEEDED", "FAILED", "CANCELLED"].map((f) => {
          const isActive = filterState === f;
          return (
            <button
              key={f}
              type="button"
              onClick={() => setFilterState(f)}
              className={`px-3 py-1 rounded-[6px] text-xs font-mono transition-colors ${
                isActive
                  ? "bg-ink text-surface font-semibold"
                  : "text-ink-secondary hover:bg-subtle hover:text-ink"
              }`}
            >
              {f} ({f === "ALL" ? jobs.length : jobs.filter((j) => (f === "RUNNING" ? ["ADMITTED", "RUNNING", "STREAMING"].includes(j.state) : j.state === f)).length})
            </button>
          );
        })}
      </div>

      {/* Job Table & Details Split */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Job Table */}
        <div className={`p-4 rounded-[10px] bg-surface border border-hairline overflow-x-auto ${selectedJob ? "lg:col-span-7" : "lg:col-span-12"}`}>
          {filteredJobs.length === 0 ? (
            <div className="p-12 text-center text-xs font-mono text-ink-muted">
              No jobs matching filter &ldquo;{filterState}&rdquo;.
            </div>
          ) : (
            <table className="w-full text-left text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-hairline text-[11px] text-ink-muted uppercase tracking-wider">
                  <th className="py-2.5 px-3">Job ID</th>
                  <th className="py-2.5 px-3">Service / Task</th>
                  <th className="py-2.5 px-3">State</th>
                  <th className="py-2.5 px-3">Progress</th>
                  <th className="py-2.5 px-3">Created</th>
                  <th className="py-2.5 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-hairline">
                {filteredJobs.map((job) => {
                  const isSelected = job.job_id === selectedJobId;
                  const progressPct = job.progress?.percent ?? (job.state === "SUCCEEDED" ? 100 : 0);

                  return (
                    <tr
                      key={job.job_id}
                      onClick={() => setSelectedJobId(job.job_id)}
                      className={`hover:bg-subtle cursor-pointer transition-colors ${
                        isSelected ? "bg-muted font-medium" : ""
                      }`}
                    >
                      <td className="py-2.5 px-3 font-semibold text-ink">
                        {job.job_id.slice(0, 10)}...
                      </td>
                      <td className="py-2.5 px-3 text-ink-secondary">
                        <span className="text-ink font-medium">{job.service}</span>
                        <span className="text-ink-muted text-[10px] ml-1">/ {job.task}</span>
                      </td>
                      <td className="py-2.5 px-3">
                        {renderStateBadge(job.state)}
                      </td>
                      <td className="py-2.5 px-3 min-w-[120px]">
                        <div className="space-y-1">
                          <div className="w-full h-1.5 rounded-full bg-subtle overflow-hidden border border-hairline">
                            <div
                              className="h-full bg-ink transition-all duration-300"
                              style={{ width: `${progressPct}%` }}
                            />
                          </div>
                          <div className="text-[10px] text-ink-muted flex justify-between">
                            <span>{job.progress?.stage || job.state}</span>
                            <span>{progressPct}%</span>
                          </div>
                        </div>
                      </td>
                      <td className="py-2.5 px-3 text-ink-muted text-[11px] whitespace-nowrap">
                        {job.created_at ? new Date(job.created_at).toLocaleTimeString() : "--:--"}
                      </td>
                      <td className="py-2.5 px-3 text-right" onClick={(e) => e.stopPropagation()}>
                        <div className="flex items-center justify-end gap-1.5">
                          {["QUEUED", "ADMITTED", "RUNNING", "STREAMING"].includes(job.state) && (
                            <button
                              type="button"
                              onClick={() => handleCancelJob(job.job_id)}
                              disabled={actionLoading}
                              title="Cancel Job"
                              className="p-1 rounded hover:bg-spot-redBg text-spot-redText"
                            >
                              <Prohibit weight="bold" className="w-3.5 h-3.5" />
                            </button>
                          )}
                          {["FAILED", "CANCELLED"].includes(job.state) && (
                            <button
                              type="button"
                              onClick={() => handleRetryJob(job.job_id)}
                              disabled={actionLoading}
                              title="Retry Job"
                              className="p-1 rounded hover:bg-subtle text-ink"
                            >
                              <ArrowCounterClockwise weight="bold" className="w-3.5 h-3.5" />
                            </button>
                          )}
                          <button
                            type="button"
                            onClick={() => setSelectedJobId(job.job_id)}
                            title="Inspect Details"
                            className="p-1 rounded hover:bg-subtle text-ink-muted hover:text-ink"
                          >
                            <Eye weight="bold" className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* Selected Job Inspection Drawer */}
        {selectedJob && (
          <div className="lg:col-span-5 p-5 rounded-[10px] bg-surface border border-hairline space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-hairline">
              <div>
                <h3 className="text-sm font-mono font-bold text-ink flex items-center gap-2">
                  <span>Job: {selectedJob.job_id}</span>
                </h3>
                <span className="text-[11px] font-mono text-ink-muted">
                  Request ID: {selectedJob.request_id || "req_internal"}
                </span>
              </div>
              <button
                type="button"
                onClick={() => setSelectedJobId(null)}
                className="p-1 rounded hover:bg-subtle text-ink-muted hover:text-ink"
              >
                <X weight="bold" className="w-4 h-4" />
              </button>
            </div>

            {/* State Machine Flow */}
            {renderLifecycleTimeline(selectedJob)}

            {/* Worker Lease Details */}
            <div className="p-3 rounded-[6px] bg-canvas border border-hairline text-xs font-mono space-y-1.5">
              <div className="text-[11px] font-semibold text-ink uppercase tracking-wider">
                Worker Lease Information
              </div>
              <div className="flex justify-between text-ink-secondary">
                <span>Worker ID:</span>
                <span className="font-semibold text-ink">{selectedJob.worker_id || "Unassigned / Auto"}</span>
              </div>
              <div className="flex justify-between text-ink-secondary">
                <span>Priority:</span>
                <span className="text-ink">{selectedJob.priority ?? 0}</span>
              </div>
              <div className="flex justify-between text-ink-secondary">
                <span>Created At:</span>
                <span>{selectedJob.created_at || "--"}</span>
              </div>
              {selectedJob.completed_at && (
                <div className="flex justify-between text-ink-secondary">
                  <span>Completed At:</span>
                  <span>{selectedJob.completed_at}</span>
                </div>
              )}
            </div>

            {/* Artifacts Viewer */}
            {selectedJob.artifacts && selectedJob.artifacts.length > 0 && (
              <div className="p-3 rounded-[6px] bg-canvas border border-hairline text-xs font-mono space-y-2">
                <div className="text-[11px] font-semibold text-ink uppercase tracking-wider flex items-center gap-1.5">
                  <DownloadSimple weight="bold" className="w-4 h-4 text-emerald-600" />
                  <span>Presigned S3 Artifacts ({selectedJob.artifacts.length})</span>
                </div>
                <div className="space-y-2">
                  {selectedJob.artifacts.map((art, idx) => (
                    <div
                      key={idx}
                      className="p-2.5 rounded-[4px] bg-surface border border-hairline space-y-1"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-ink">{art.name}</span>
                        {art.download_url && (
                          <a
                            href={art.download_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-2 py-0.5 rounded bg-ink text-surface text-[10px] font-medium flex items-center gap-1"
                          >
                            <DownloadSimple weight="bold" className="w-3 h-3" />
                            <span>Download</span>
                          </a>
                        )}
                      </div>
                      <div className="text-[10px] text-ink-muted truncate">
                        S3: {art.storage_path}
                      </div>
                      {art.download_url && art.type.startsWith("image/") && (
                        <img
                          src={art.download_url}
                          alt={art.name}
                          className="mt-2 max-h-48 rounded border border-hairline object-contain"
                        />
                      )}
                      {art.download_url && art.type.startsWith("video/") && (
                        <video
                          controls
                          src={art.download_url}
                          className="mt-2 max-h-48 rounded border border-hairline"
                        />
                      )}
                      {art.download_url && art.type.startsWith("audio/") && (
                        <audio controls src={art.download_url} className="mt-2 w-full h-8" />
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Error inspector */}
            {selectedJob.error && (
              <div className="p-3 rounded-[6px] bg-spot-redBg border border-[#E6C5C5] text-spot-redText text-xs font-mono space-y-1">
                <div className="font-bold flex items-center gap-1.5">
                  <WarningCircle weight="bold" className="w-4 h-4" />
                  <span>Error Code: {selectedJob.error.code}</span>
                </div>
                <p className="text-[11px] leading-relaxed">
                  {selectedJob.error.message}
                </p>
                {selectedJob.error.request_id && (
                  <div className="text-[10px] text-ink-muted">
                    Request ID: {selectedJob.error.request_id}
                  </div>
                )}
              </div>
            )}

            {/* Raw Input Parameters */}
            <div className="space-y-1">
              <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-ink-muted">
                Inputs Payload
              </span>
              <pre className="p-3 rounded-[6px] bg-[#141517] text-[#D5D7E0] text-[11px] font-mono overflow-x-auto max-h-44">
                <code>{JSON.stringify(selectedJob.inputs, null, 2)}</code>
              </pre>
            </div>

            {/* Drawer Actions */}
            <div className="pt-2 flex gap-2">
              {["QUEUED", "ADMITTED", "RUNNING", "STREAMING"].includes(selectedJob.state) && (
                <button
                  type="button"
                  onClick={() => handleCancelJob(selectedJob.job_id)}
                  disabled={actionLoading}
                  className="flex-1 py-2 rounded-[6px] bg-spot-redBg text-spot-redText border border-[#E6C5C5] text-xs font-mono font-semibold flex items-center justify-center gap-1.5"
                >
                  <Prohibit weight="bold" className="w-3.5 h-3.5" />
                  <span>Cancel Job</span>
                </button>
              )}
              {["FAILED", "CANCELLED"].includes(selectedJob.state) && (
                <button
                  type="button"
                  onClick={() => handleRetryJob(selectedJob.job_id)}
                  disabled={actionLoading}
                  className="flex-1 py-2 rounded-[6px] bg-ink text-surface text-xs font-mono font-semibold flex items-center justify-center gap-1.5"
                >
                  <ArrowCounterClockwise weight="bold" className="w-3.5 h-3.5" />
                  <span>Retry Job</span>
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
