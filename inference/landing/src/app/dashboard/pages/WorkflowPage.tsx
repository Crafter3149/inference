"use client";

import React, { useState, useCallback, useRef } from "react";
import { Card } from "../components/Card";
import { JsonViewer } from "../components/JsonViewer";
import { ImageUploader } from "../components/ImageUploader";
import { api } from "../api";

const EXAMPLE_WORKFLOW = JSON.stringify(
  {
    version: "1.0",
    inputs: [{ type: "InferenceImage", name: "image" }],
    steps: [
      {
        type: "roboflow_core/roboflow_object_detection_model@v2",
        name: "detection",
        image: "$inputs.image",
        model_id: "YOUR_MODEL_ID",
        confidence: 0.4,
      },
    ],
    outputs: [
      { type: "JsonField", name: "predictions", selector: "$steps.detection.predictions" },
    ],
  },
  null,
  2,
);

export function WorkflowPage() {
  const [specText, setSpecText] = useState(EXAMPLE_WORKFLOW);
  const [images, setImages] = useState<{ base64: string; dataUrl: string }[]>([]);
  const [result, setResult] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageSelect = useCallback((base64: string) => {
    setImages((prev) => [...prev, { base64, dataUrl: `data:image/png;base64,${base64}` }]);
  }, []);

  const loadJsonFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setSpecText(reader.result as string);
    reader.readAsText(file);
    e.target.value = "";
  };

  const run = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setElapsed(null);

    try {
      const spec = JSON.parse(specText);
      const inputs: Record<string, unknown> = {};
      if (images.length === 1) {
        inputs.image = { type: "base64", value: images[0].base64 };
      } else if (images.length > 1) {
        inputs.image = images.map((img) => ({ type: "base64", value: img.base64 }));
      }

      const t0 = performance.now();
      const res = await api.runWorkflow(spec, inputs);
      setElapsed(performance.now() - t0);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Workflow execution failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2 className="text-2xl font-bold text-white mb-6">Workflow</h2>

      {error && (
        <div className="mb-4 p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-300 text-sm">{error}</div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <Card title="Workflow Specification">
            <div className="flex gap-2 mb-3">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-3 py-1 text-xs bg-surface hover:bg-surface-hover border border-border rounded-lg text-gray-300 transition-colors"
              >
                Load JSON file
              </button>
              <button
                onClick={() => setSpecText(EXAMPLE_WORKFLOW)}
                className="px-3 py-1 text-xs bg-surface hover:bg-surface-hover border border-border rounded-lg text-gray-300 transition-colors"
              >
                Reset to example
              </button>
              <input ref={fileInputRef} type="file" accept=".json" className="hidden" onChange={loadJsonFile} />
            </div>
            <textarea
              value={specText}
              onChange={(e) => setSpecText(e.target.value)}
              spellCheck={false}
              className="w-full h-80 px-4 py-3 bg-surface border border-border rounded-lg text-sm text-gray-300 font-mono resize-y focus:outline-none focus:border-accent"
            />
          </Card>

          <Card title="Images">
            <ImageUploader onImageSelect={handleImageSelect} disabled={loading} />
            {images.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2">
                {images.map((img, i) => (
                  <div key={i} className="relative group">
                    <img src={img.dataUrl} alt={`Image ${i + 1}`} className="h-16 rounded" />
                    <button
                      onClick={() => setImages((prev) => prev.filter((_, j) => j !== i))}
                      className="absolute -top-1 -right-1 w-5 h-5 bg-red-600 text-white rounded-full text-xs opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      x
                    </button>
                  </div>
                ))}
                <button onClick={() => setImages([])} className="text-xs text-gray-500 hover:text-red-400 self-center">
                  Clear all
                </button>
              </div>
            )}
          </Card>

          <button
            onClick={run}
            disabled={loading}
            className="w-full px-4 py-2.5 bg-accent hover:bg-accent-hover text-white text-sm font-medium rounded-lg disabled:opacity-50 transition-colors"
          >
            {loading ? "Running..." : "Run Workflow"}
          </button>
        </div>

        <div className="space-y-4">
          {elapsed !== null && (
            <Card>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Execution Time</span>
                <span className="text-white font-mono">{elapsed.toFixed(0)} ms</span>
              </div>
            </Card>
          )}
          {result !== null && (
            <Card title="Result">
              <JsonViewer data={result} maxHeight="40rem" />
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
