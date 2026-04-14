"use client";

import React, { useState, useCallback, useEffect, useRef } from "react";
import { Card } from "../components/Card";
import { JsonViewer } from "../components/JsonViewer";
import { ImageUploader } from "../components/ImageUploader";
import { BBoxCanvas } from "../components/BBoxCanvas";
import { ClassificationChart } from "../components/ClassificationChart";
import { HeatmapCanvas } from "../components/HeatmapCanvas";
import { useModelsData } from "../hooks/useModelsData";
import { useAutoRefresh } from "../hooks/useAutoRefresh";
import { api } from "../api";
import type { ObjectDetectionResponse, ClassificationResponse, AnomalyDetectionResponse } from "../types";

const inputClass =
  "w-full px-3 py-2 bg-surface border border-border rounded-lg text-sm text-white focus:outline-none focus:border-accent";
const labelClass = "block text-xs text-gray-500 mb-1";

export function InferencePage() {
  const { models, refetch: refreshModels } = useModelsData();
  useAutoRefresh(refreshModels, { interval: 5000 });

  const [selectedModel, setSelectedModel] = useState("");
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [result, setResult] = useState<ObjectDetectionResponse | ClassificationResponse | AnomalyDetectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const elapsedRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [inferTime, setInferTime] = useState<number | null>(null);

  useEffect(() => {
    if (loading) {
      setElapsed(0);
      const t0 = Date.now();
      elapsedRef.current = setInterval(() => setElapsed(Date.now() - t0), 200);
    } else if (elapsedRef.current) {
      clearInterval(elapsedRef.current);
      elapsedRef.current = null;
    }
    return () => { if (elapsedRef.current) clearInterval(elapsedRef.current); };
  }, [loading]);

  // OD / IS params
  const [confidence, setConfidence] = useState(0.4);
  const [iouThreshold, setIouThreshold] = useState(0.3);
  const [maxDetections, setMaxDetections] = useState(300);
  const [classFilter, setClassFilter] = useState("");
  const [visualizePredictions, setVisualizePredictions] = useState(false);

  // IS-only params
  const [maskDecodeMode, setMaskDecodeMode] = useState("accurate");

  const selectedTaskType = models.find((m) => m.model_id === selectedModel)?.task_type ?? "";

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

    const SUPPORTED_INFER_TYPES = ["object-detection", "classification", "instance-segmentation", "anomaly-detection"];
    if (!SUPPORTED_INFER_TYPES.includes(selectedTaskType)) {
      setError(
        `Task type "${selectedTaskType}" does not have a dedicated inference endpoint. ` +
        `Use the Workflow page to run this model.`
      );
      setLoading(false);
      return;
    }

    try {
      const image = { type: "base64", value: imageBase64 };
      let res: ObjectDetectionResponse | ClassificationResponse | AnomalyDetectionResponse;

      const odParams: Record<string, unknown> = {
        model_id: selectedModel,
        image,
        confidence,
        iou_threshold: iouThreshold,
        max_detections: maxDetections,
        visualize_predictions: visualizePredictions,
      };
      if (classFilter.trim()) {
        odParams.class_filter = classFilter.split(",").map((s) => s.trim()).filter(Boolean);
      }

      if (selectedTaskType === "anomaly-detection") {
        res = await api.inferAnomalyDetection({
          model_id: selectedModel,
          image,
        });
      } else if (selectedTaskType === "classification") {
        res = await api.inferClassification({
          model_id: selectedModel,
          image,
          confidence,
          visualize_predictions: visualizePredictions,
        });
      } else if (selectedTaskType === "instance-segmentation") {
        res = await api.inferInstanceSegmentation({
          ...odParams,
          mask_decode_mode: maskDecodeMode,
        });
      } else {
        res = await api.inferObjectDetection(odParams);
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

    if (selectedTaskType === "object-detection" || selectedTaskType === "instance-segmentation") {
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

    if (selectedTaskType === "classification") {
      const r = result as ClassificationResponse;
      return <ClassificationChart predictions={r.predictions} top={r.top} confidence={r.confidence} />;
    }

    if (selectedTaskType === "anomaly-detection") {
      const r = result as AnomalyDetectionResponse;
      return (
        <HeatmapCanvas
          imageSrc={imageDataUrl}
          anomalyMap={r.anomaly_map}
          anomalyScore={r.anomaly_score}
        />
      );
    }

    return null;
  };

  const isOdOrIs = selectedTaskType === "object-detection" || selectedTaskType === "instance-segmentation";
  const isCls = selectedTaskType === "classification";
  const isIs = selectedTaskType === "instance-segmentation";

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
                <label className={labelClass}>Model</label>
                <select
                  value={selectedModel}
                  onChange={(e) => {
                    setSelectedModel(e.target.value);
                    setResult(null);
                    setError(null);
                    setInferTime(null);
                  }}
                  className={inputClass}
                >
                  <option value="">Select a model</option>
                  {models.map((m) => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.model_id} ({m.task_type})
                    </option>
                  ))}
                </select>
              </div>
              {selectedTaskType && (
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">Task Type</span>
                  <span className="px-2 py-0.5 bg-accent/10 text-accent-hover rounded text-xs">{selectedTaskType}</span>
                </div>
              )}

              {/* OD + IS + CLS: confidence */}
              {(isOdOrIs || isCls) && (
                <div>
                  <label className={labelClass}>Confidence: {confidence.toFixed(2)}</label>
                  <input type="range" min="0.01" max="1" step="0.01" value={confidence}
                    onChange={(e) => setConfidence(+e.target.value)} className="w-full accent-accent" />
                </div>
              )}

              {/* OD + IS: iou_threshold */}
              {isOdOrIs && (
                <div>
                  <label className={labelClass}>IoU Threshold: {iouThreshold.toFixed(2)}</label>
                  <input type="range" min="0.01" max="1" step="0.01" value={iouThreshold}
                    onChange={(e) => setIouThreshold(+e.target.value)} className="w-full accent-accent" />
                </div>
              )}

              {/* OD + IS: max_detections */}
              {isOdOrIs && (
                <div>
                  <label className={labelClass}>Max Detections</label>
                  <input type="number" min="1" max="3000" value={maxDetections}
                    onChange={(e) => setMaxDetections(+e.target.value)} className={inputClass} />
                </div>
              )}

              {/* OD + IS: class_filter */}
              {isOdOrIs && (
                <div>
                  <label className={labelClass}>Class Filter (comma-separated)</label>
                  <input type="text" value={classFilter} placeholder="e.g. person, car"
                    onChange={(e) => setClassFilter(e.target.value)} className={inputClass} />
                </div>
              )}

              {/* IS-only: mask_decode_mode */}
              {isIs && (
                <div>
                  <label className={labelClass}>Mask Decode Mode</label>
                  <select value={maskDecodeMode} onChange={(e) => setMaskDecodeMode(e.target.value)} className={inputClass}>
                    <option value="accurate">Accurate</option>
                    <option value="fast">Fast</option>
                    <option value="tradeoff">Tradeoff</option>
                  </select>
                </div>
              )}

              {/* OD + IS + CLS: visualize_predictions */}
              {(isOdOrIs || isCls) && (
                <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
                  <input type="checkbox" checked={visualizePredictions}
                    onChange={(e) => setVisualizePredictions(e.target.checked)} className="accent-accent" />
                  Visualize Predictions (server-side)
                </label>
              )}

              <button
                onClick={runInference}
                disabled={loading || !imageBase64 || !selectedModel}
                className="w-full px-4 py-2.5 bg-accent hover:bg-accent-hover text-white text-sm font-medium rounded-lg disabled:opacity-50 transition-colors"
              >
                {loading ? `Running... ${(elapsed / 1000).toFixed(1)}s` : "Run Inference"}
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
