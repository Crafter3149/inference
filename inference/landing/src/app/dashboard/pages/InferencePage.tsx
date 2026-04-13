"use client";

import React, { useState, useCallback } from "react";
import { Card } from "../components/Card";
import { JsonViewer } from "../components/JsonViewer";
import { ImageUploader } from "../components/ImageUploader";
import { BBoxCanvas } from "../components/BBoxCanvas";
import { ClassificationChart } from "../components/ClassificationChart";
import { useModelsData } from "../hooks/useModelsData";
import { useAutoRefresh } from "../hooks/useAutoRefresh";
import { api } from "../api";
import type { ObjectDetectionResponse, ClassificationResponse } from "../types";

type TaskType = "object-detection" | "classification" | "instance-segmentation";

const TASK_LABELS: Record<TaskType, string> = {
  "object-detection": "Object Detection",
  "classification": "Classification",
  "instance-segmentation": "Instance Segmentation",
};

export function InferencePage() {
  const { models, refetch: refreshModels } = useModelsData();
  useAutoRefresh(refreshModels, { interval: 5000 });

  const [selectedModel, setSelectedModel] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("object-detection");
  const [confidence, setConfidence] = useState(0.4);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [result, setResult] = useState<ObjectDetectionResponse | ClassificationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inferTime, setInferTime] = useState<number | null>(null);

  const handleModelChange = useCallback(
    (modelId: string) => {
      setSelectedModel(modelId);
      const model = models.find((m) => m.model_id === modelId);
      if (model) {
        const t = model.task_type as TaskType;
        if (t in TASK_LABELS) setTaskType(t);
      }
    },
    [models],
  );

  const handleImageSelect = useCallback((base64: string) => {
    setImageBase64(base64);
    setImageDataUrl(`data:image/png;base64,${base64}`);
    setResult(null);
    setError(null);
    setInferTime(null);
  }, []);

  const runInference = async () => {
    if (!imageBase64 || !selectedModel) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setInferTime(null);

    try {
      const image = { type: "base64", value: imageBase64 };
      let res: ObjectDetectionResponse | ClassificationResponse;

      if (taskType === "classification") {
        res = await api.inferClassification({ model_id: selectedModel, image, confidence });
      } else if (taskType === "instance-segmentation") {
        res = await api.inferInstanceSegmentation({ model_id: selectedModel, image, confidence });
      } else {
        res = await api.inferObjectDetection({ model_id: selectedModel, image, confidence });
      }

      setInferTime(res.time ? res.time * 1000 : 0);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inference failed");
    } finally {
      setLoading(false);
    }
  };

  const renderResult = () => {
    if (!result || !imageDataUrl) return null;

    if (taskType === "object-detection" || taskType === "instance-segmentation") {
      const r = result as ObjectDetectionResponse;
      return (
        <BBoxCanvas
          imageSrc={imageDataUrl}
          predictions={r.predictions}
          imageWidth={r.image.width}
          imageHeight={r.image.height}
        />
      );
    }

    if (taskType === "classification") {
      const r = result as ClassificationResponse;
      return <ClassificationChart predictions={r.predictions} top={r.top} confidence={r.confidence} />;
    }

    return null;
  };

  return (
    <div>
      <h2 className="text-2xl font-bold text-white mb-6">Inference</h2>

      {error && (
        <div className="mb-4 p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-300 text-sm">{error}</div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <Card title="Settings">
            <div className="space-y-4">
              <div>
                <label className="block text-xs text-gray-500 mb-1">Model</label>
                <select
                  value={selectedModel}
                  onChange={(e) => handleModelChange(e.target.value)}
                  className="w-full px-3 py-2 bg-surface border border-border rounded-lg text-sm text-white focus:outline-none focus:border-accent"
                >
                  <option value="">Select a model</option>
                  {models.map((m) => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.model_id} ({m.task_type})
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">Task Type</label>
                <select
                  value={taskType}
                  onChange={(e) => setTaskType(e.target.value as TaskType)}
                  className="w-full px-3 py-2 bg-surface border border-border rounded-lg text-sm text-white focus:outline-none focus:border-accent"
                >
                  {Object.entries(TASK_LABELS).map(([k, v]) => (
                    <option key={k} value={k}>{v}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">Confidence: {confidence.toFixed(2)}</label>
                <input
                  type="range" min="0.01" max="1" step="0.01"
                  value={confidence}
                  onChange={(e) => setConfidence(+e.target.value)}
                  className="w-full accent-accent"
                />
              </div>
              <button
                onClick={runInference}
                disabled={loading || !imageBase64 || !selectedModel}
                className="w-full px-4 py-2.5 bg-accent hover:bg-accent-hover text-white text-sm font-medium rounded-lg disabled:opacity-50 transition-colors"
              >
                {loading ? "Running..." : "Run Inference"}
              </button>
            </div>
          </Card>
          <Card title="Image">
            <ImageUploader onImageSelect={handleImageSelect} disabled={loading} />
          </Card>
        </div>

        <div className="space-y-4">
          {inferTime !== null && (
            <Card>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Inference Time</span>
                <span className="text-white font-mono">{inferTime.toFixed(0)} ms</span>
              </div>
            </Card>
          )}
          {result !== null && (
            <>
              <Card title="Visualization">{renderResult()}</Card>
              <Card title="Raw Output">
                <JsonViewer data={result} />
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
