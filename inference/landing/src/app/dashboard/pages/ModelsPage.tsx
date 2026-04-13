"use client";

import React, { useState } from "react";
import { Card } from "../components/Card";
import { useModelsData } from "../hooks/useModelsData";
import { useAutoRefresh } from "../hooks/useAutoRefresh";

export function ModelsPage() {
  const { models, loading, error, refetch, addModel, removeModel, clearModels } = useModelsData();
  const [modelId, setModelId] = useState("");
  const [modelType, setModelType] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);

  useAutoRefresh(refetch, { interval: 5000 });

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!modelId.trim()) return;
    setActionLoading(true);
    setActionError(null);
    try {
      await addModel(modelId.trim(), modelType.trim() || undefined, apiKey.trim() || undefined);
      setModelId("");
      setModelType("");
      setApiKey("");
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to add model");
    } finally {
      setActionLoading(false);
    }
  };

  const handleRemove = async (id: string) => {
    setActionLoading(true);
    setActionError(null);
    try {
      await removeModel(id);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to remove model");
    } finally {
      setActionLoading(false);
    }
  };

  const handleClear = async () => {
    if (models.length === 0) return;
    setActionLoading(true);
    setActionError(null);
    try {
      await clearModels();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to clear models");
    } finally {
      setActionLoading(false);
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-white">Models</h2>
        <button
          onClick={handleClear}
          disabled={actionLoading || models.length === 0}
          className="px-3 py-1.5 text-sm bg-red-900/30 hover:bg-red-900/50 text-red-300 border border-red-700/50 rounded-lg disabled:opacity-50 transition-colors"
        >
          Clear All
        </button>
      </div>

      {(error || actionError) && (
        <div className="mb-4 p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-300 text-sm">
          {error || actionError}
        </div>
      )}

      <Card title="Load Model" className="mb-6">
        <form onSubmit={handleAdd} className="flex flex-wrap gap-3 items-end">
          <div className="flex-1 min-w-[200px]">
            <label className="block text-xs text-gray-500 mb-1">Model ID *</label>
            <input
              type="text"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              placeholder="project/version"
              className="w-full px-3 py-2 bg-surface border border-border rounded-lg text-sm text-white placeholder-gray-600 focus:outline-none focus:border-accent"
            />
          </div>
          <div className="w-40">
            <label className="block text-xs text-gray-500 mb-1">Model Type</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="w-full px-3 py-2 bg-surface border border-border rounded-lg text-sm text-white focus:outline-none focus:border-accent"
            >
              <option value="">Auto-detect</option>
              <option value="object-detection">Object Detection</option>
              <option value="classification">Classification</option>
              <option value="instance-segmentation">Instance Segmentation</option>
            </select>
          </div>
          <div className="w-48">
            <label className="block text-xs text-gray-500 mb-1">API Key</label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Optional"
              className="w-full px-3 py-2 bg-surface border border-border rounded-lg text-sm text-white placeholder-gray-600 focus:outline-none focus:border-accent"
            />
          </div>
          <button
            type="submit"
            disabled={actionLoading || !modelId.trim()}
            className="px-4 py-2 bg-accent hover:bg-accent-hover text-white text-sm rounded-lg disabled:opacity-50 transition-colors"
          >
            {actionLoading ? "Loading..." : "Load Model"}
          </button>
        </form>
      </Card>

      <Card title="Loaded Models" badge={models.length}>
        {loading ? (
          <p className="text-gray-500">Loading...</p>
        ) : models.length === 0 ? (
          <p className="text-gray-500 text-sm">No models loaded.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-gray-500 text-left">
                  <th className="pb-2 font-medium">Model ID</th>
                  <th className="pb-2 font-medium">Task Type</th>
                  <th className="pb-2 font-medium">Input Size</th>
                  <th className="pb-2 font-medium">Batch</th>
                  <th className="pb-2 font-medium w-20"></th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.model_id} className="border-b border-border/50 hover:bg-surface-hover">
                    <td className="py-3 text-white font-mono text-xs">{m.model_id}</td>
                    <td className="py-3">
                      <span className="px-2 py-0.5 bg-accent/10 text-accent-hover rounded text-xs">
                        {m.task_type}
                      </span>
                    </td>
                    <td className="py-3 text-gray-400">
                      {m.input_width && m.input_height ? `${m.input_width} x ${m.input_height}` : "-"}
                    </td>
                    <td className="py-3 text-gray-400">{m.batch_size ?? "-"}</td>
                    <td className="py-3">
                      <button
                        onClick={() => handleRemove(m.model_id)}
                        disabled={actionLoading}
                        className="text-xs text-red-400 hover:text-red-300 disabled:opacity-50"
                      >
                        Unload
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
