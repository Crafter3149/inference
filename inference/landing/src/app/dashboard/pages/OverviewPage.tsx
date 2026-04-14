"use client";

import React from "react";
import { Card } from "../components/Card";
import { StatusBadge } from "../components/StatusBadge";
import { useServerData } from "../hooks/useServerData";
import { useModelsData } from "../hooks/useModelsData";
import { useMetrics } from "../hooks/useMetrics";
import { useAutoRefresh } from "../hooks/useAutoRefresh";

function formatBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = bytes;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(1)} ${units[i]}`;
}

function formatDuration(seconds: number): string {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const parts: string[] = [];
  if (d > 0) parts.push(`${d}d`);
  if (h > 0) parts.push(`${h}h`);
  if (m > 0) parts.push(`${m}m`);
  parts.push(`${s}s`);
  return parts.join(" ");
}

export function OverviewPage() {
  const { serverInfo, healthStatus, error, refetch: refreshServer } = useServerData();
  const { models, refetch: refreshModels } = useModelsData();
  const { data: metrics, refetch: refreshMetrics } = useMetrics();

  useAutoRefresh(() => { refreshServer(); refreshModels(); refreshMetrics(); }, { interval: 5000 });

  // Map sanitized Prometheus model IDs back to real model IDs from registry
  const sanitize = (s: string) => s.replace(/[^a-zA-Z0-9_]/g, "_");
  const resolveModelId = (sanitizedId: string): string => {
    const match = models.find((m) => sanitize(m.model_id) === sanitizedId);
    return match ? match.model_id : sanitizedId;
  };

  return (
    <div>
      <h2 className="text-2xl font-bold text-white mb-6">Overview</h2>

      {error && (
        <div className="mb-4 p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Row 1: Server + Loaded Models */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        <Card title="Server" className="col-span-1 md:col-span-2 lg:col-span-2">
          {serverInfo ? (
            <div className="space-y-3">
              <StatusBadge status={healthStatus} />
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Name</span>
                  <p className="text-white">{serverInfo.name}</p>
                </div>
                <div>
                  <span className="text-gray-500">Version</span>
                  <p className="text-white font-mono">{serverInfo.version}</p>
                </div>
                {metrics.memoryBytes !== null && (
                  <div>
                    <span className="text-gray-500">Memory</span>
                    <p className="text-white font-mono">{formatBytes(metrics.memoryBytes)}</p>
                  </div>
                )}
                {metrics.uptimeSeconds !== null && (
                  <div>
                    <span className="text-gray-500">Uptime</span>
                    <p className="text-white font-mono">{formatDuration(metrics.uptimeSeconds)}</p>
                  </div>
                )}
                <div>
                  <span className="text-gray-500">Endpoint</span>
                  <p className="text-white font-mono">{typeof window !== "undefined" ? window.location.host : ""}</p>
                </div>
                <div>
                  <span className="text-gray-500">UUID</span>
                  <p className="text-white font-mono text-xs">{serverInfo.uuid}</p>
                </div>
              </div>
            </div>
          ) : (
            <p className="text-gray-500">Connecting...</p>
          )}
        </Card>

        <Card title="Loaded Models" badge={models.length}>
          {models.length === 0 ? (
            <p className="text-sm text-gray-500">No models loaded</p>
          ) : (
            <div className="space-y-3">
              {models.map((m) => (
                <div key={m.model_id} className="text-sm">
                  <div className="text-white font-mono truncate">{m.model_id}</div>
                  <div className="text-gray-500 text-xs mt-0.5">
                    Task: {m.task_type}
                    {m.batch_size != null && ` | Batch Size: ${m.batch_size}`}
                    {m.input_width != null && m.input_height != null && ` | Input: ${m.input_width}×${m.input_height}`}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* Row 2: HTTP Request Stats + Model Inference Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        {/* A-class: HTTP Request Stats */}
        <Card title="API Requests">
          <div className="space-y-4">
            <div className="flex items-baseline gap-4">
              <span className="text-3xl font-bold text-white">{metrics.httpTotal}</span>
              <span className="text-sm text-gray-500">total requests</span>
              <span className="ml-auto text-sm font-mono text-green-400">
                {metrics.httpSuccessRate.toFixed(1)}% success
              </span>
            </div>
            {metrics.endpoints.length > 0 ? (
              <div className="space-y-2">
                {metrics.endpoints.slice(0, 8).map((ep) => (
                  <div key={ep.endpoint} className="flex items-center justify-between text-sm">
                    <span className="text-gray-300 font-mono truncate mr-3">{ep.endpoint}</span>
                    <div className="flex items-center gap-3 shrink-0">
                      <span className="text-white font-mono">{ep.total}</span>
                      {ep.errors > 0 && (
                        <span className="text-red-400 text-xs">{ep.errors} err</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-gray-500">No inference requests yet</p>
            )}
          </div>
        </Card>

        {/* B-class: Per-Model Inference Stats (10s window) */}
        <Card title="Model Inference (10s window)">
          <div className="space-y-4">
            <div className="flex items-baseline gap-4">
              <span className="text-3xl font-bold text-white">{metrics.inferencesTotal}</span>
              <span className="text-sm text-gray-500">inferences</span>
              {metrics.errorsTotal > 0 && (
                <span className="text-sm text-red-400">{metrics.errorsTotal} errors</span>
              )}
              {metrics.avgInferenceTimeTotal > 0 && (
                <span className="ml-auto text-sm font-mono text-gray-400">
                  avg {(metrics.avgInferenceTimeTotal * 1000).toFixed(0)}ms
                </span>
              )}
            </div>
            {metrics.models.length > 0 ? (
              <div className="space-y-2">
                {metrics.models.map((m) => {
                  const realId = resolveModelId(m.modelId);
                  return (
                    <div key={m.modelId} className="text-sm">
                      <div className="text-gray-300 font-mono truncate">{realId}</div>
                      <div className="flex items-center gap-3 text-xs mt-0.5">
                        <span className="text-white">{m.numInferences} req</span>
                        <span className="text-gray-500 font-mono">
                          {(m.avgInferenceTime * 1000).toFixed(0)}ms avg
                        </span>
                        {m.numErrors > 0 && (
                          <span className="text-red-400">{m.numErrors} err</span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-sm text-gray-500">No model activity in last 10s</p>
            )}
          </div>
        </Card>
      </div>

      {/* Row 3: Quick Links */}
      <Card title="Quick Links">
        <div className="flex flex-wrap gap-3">
          <a
            href="/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="px-4 py-2 bg-surface hover:bg-surface-hover border border-border rounded-lg text-sm text-gray-300 hover:text-white transition-colors"
          >
            API Documentation (Swagger)
          </a>
          <a
            href="/build"
            target="_blank"
            rel="noopener noreferrer"
            className="px-4 py-2 bg-surface hover:bg-surface-hover border border-border rounded-lg text-sm text-gray-300 hover:text-white transition-colors"
          >
            Workflow Builder
            <span className="ml-1 text-xs text-gray-500">(requires internet)</span>
          </a>
        </div>
      </Card>
    </div>
  );
}
