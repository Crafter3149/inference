"use client";

import React from "react";
import { Card } from "../components/Card";
import { StatusBadge } from "../components/StatusBadge";
import { useServerData } from "../hooks/useServerData";
import { useModelsData } from "../hooks/useModelsData";
import { useAutoRefresh } from "../hooks/useAutoRefresh";

export function OverviewPage() {
  const { serverInfo, healthStatus, error, refetch: refreshServer } = useServerData();
  const { models, refetch: refreshModels } = useModelsData();

  useAutoRefresh(() => { refreshServer(); refreshModels(); }, { interval: 5000 });

  return (
    <div>
      <h2 className="text-2xl font-bold text-white mb-6">Overview</h2>

      {error && (
        <div className="mb-4 p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-300 text-sm">
          {error}
        </div>
      )}

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
                <div className="col-span-2">
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
          <div className="text-4xl font-bold text-white">{models.length}</div>
          <p className="text-sm text-gray-500 mt-1">
            {models.length === 0 ? "No models loaded" : `${models.length} model${models.length > 1 ? "s" : ""} active`}
          </p>
        </Card>
      </div>

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
