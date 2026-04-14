import { useState, useCallback, useEffect } from "react";

/** Per-model inference stats (B-class metrics, 10s window). */
export interface ModelMetrics {
  modelId: string;
  numInferences: number;
  avgInferenceTime: number;
  numErrors: number;
}

/** HTTP request stats per endpoint (A-class metrics). */
export interface EndpointStats {
  endpoint: string;
  total: number;
  success: number;
  errors: number;
}

export interface MetricsData {
  /** Process resident memory in bytes. */
  memoryBytes: number | null;
  /** Process uptime in seconds. */
  uptimeSeconds: number | null;
  /** HTTP request stats grouped by handler (inference-related only). */
  endpoints: EndpointStats[];
  /** Total HTTP requests (inference-related). */
  httpTotal: number;
  /** Overall HTTP success rate (0-100). */
  httpSuccessRate: number;
  /** Per-model inference metrics from the 10s window. */
  models: ModelMetrics[];
  /** Totals across all models (10s window). */
  inferencesTotal: number;
  errorsTotal: number;
  avgInferenceTimeTotal: number;
}

const EMPTY: MetricsData = {
  memoryBytes: null,
  uptimeSeconds: null,
  endpoints: [],
  httpTotal: 0,
  httpSuccessRate: 100,
  models: [],
  inferencesTotal: 0,
  errorsTotal: 0,
  avgInferenceTimeTotal: 0,
};

const INFER_PREFIXES = ["/infer/", "/workflows/run", "/inference_pipelines/"];

function parseMetrics(text: string): MetricsData {
  const data: MetricsData = { ...EMPTY, endpoints: [], models: [] };

  let processStart: number | null = null;
  const epSuccess: Record<string, number> = {};
  const epError: Record<string, number> = {};

  for (const line of text.split("\n")) {
    if (!line || line.startsWith("#")) continue;

    // --- A-class: process metrics ---
    if (line.startsWith("process_resident_memory_bytes")) {
      const v = Number(line.split(/\s+/).pop());
      if (!Number.isNaN(v)) data.memoryBytes = v;
      continue;
    }
    if (line.startsWith("process_start_time_seconds")) {
      const v = Number(line.split(/\s+/).pop());
      if (!Number.isNaN(v)) processStart = v;
      continue;
    }

    // --- A-class: HTTP request counters ---
    if (line.startsWith("http_requests_total")) {
      const m = line.match(/handler="([^"]+)".*status="(\d)xx".*\s+(\d+(?:\.\d+)?)/);
      if (m) {
        const [, handler, statusDigit, countStr] = m;
        const count = parseFloat(countStr);
        const isInfer = INFER_PREFIXES.some((p) => handler.startsWith(p) || handler.includes(p));
        if (isInfer && handler !== "none") {
          if (statusDigit === "2") {
            epSuccess[handler] = (epSuccess[handler] || 0) + count;
          } else if (statusDigit === "4" || statusDigit === "5") {
            epError[handler] = (epError[handler] || 0) + count;
          }
        }
      }
      continue;
    }

    // --- B-class: per-model gauges ---
    if (line.startsWith("num_inferences_") && !line.startsWith("num_inferences_total")) {
      const name = line.split(/\s+/)[0];
      const modelId = name.replace("num_inferences_", "");
      const v = Number(line.split(/\s+/).pop());
      let entry = data.models.find((m) => m.modelId === modelId);
      if (!entry) {
        entry = { modelId, numInferences: 0, avgInferenceTime: 0, numErrors: 0 };
        data.models.push(entry);
      }
      entry.numInferences = v;
      continue;
    }
    if (line.startsWith("avg_inference_time_") && !line.startsWith("avg_inference_time_total")) {
      const name = line.split(/\s+/)[0];
      const modelId = name.replace("avg_inference_time_", "");
      const v = Number(line.split(/\s+/).pop());
      let entry = data.models.find((m) => m.modelId === modelId);
      if (!entry) {
        entry = { modelId, numInferences: 0, avgInferenceTime: 0, numErrors: 0 };
        data.models.push(entry);
      }
      entry.avgInferenceTime = v;
      continue;
    }
    if (line.startsWith("num_errors_") && !line.startsWith("num_errors_total")) {
      const name = line.split(/\s+/)[0];
      const modelId = name.replace("num_errors_", "");
      const v = Number(line.split(/\s+/).pop());
      let entry = data.models.find((m) => m.modelId === modelId);
      if (!entry) {
        entry = { modelId, numInferences: 0, avgInferenceTime: 0, numErrors: 0 };
        data.models.push(entry);
      }
      entry.numErrors = v;
      continue;
    }

    // --- B-class: totals ---
    if (line.startsWith("num_inferences_total ")) {
      data.inferencesTotal = Number(line.split(/\s+/).pop()) || 0;
      continue;
    }
    if (line.startsWith("avg_inference_time_total ")) {
      data.avgInferenceTimeTotal = Number(line.split(/\s+/).pop()) || 0;
      continue;
    }
    if (line.startsWith("num_errors_total ")) {
      data.errorsTotal = Number(line.split(/\s+/).pop()) || 0;
      continue;
    }
  }

  // Uptime
  if (processStart !== null) {
    data.uptimeSeconds = Math.max(0, Date.now() / 1000 - processStart);
  }

  // Aggregate endpoint stats
  const allHandlers = Array.from(new Set([...Object.keys(epSuccess), ...Object.keys(epError)]));
  let totalReqs = 0;
  let totalSuccess = 0;
  for (const ep of allHandlers) {
    const s = epSuccess[ep] || 0;
    const e = epError[ep] || 0;
    data.endpoints.push({ endpoint: ep, total: s + e, success: s, errors: e });
    totalReqs += s + e;
    totalSuccess += s;
  }
  data.endpoints.sort((a, b) => b.total - a.total);
  data.httpTotal = totalReqs;
  data.httpSuccessRate = totalReqs > 0 ? (totalSuccess / totalReqs) * 100 : 100;

  return data;
}

export function useMetrics() {
  const [data, setData] = useState<MetricsData>(EMPTY);
  const [error, setError] = useState<string | null>(null);

  const refetch = useCallback(async () => {
    try {
      const res = await fetch("/metrics");
      if (!res.ok) {
        setError(`Metrics endpoint returned ${res.status}`);
        return;
      }
      setError(null);
      setData(parseMetrics(await res.text()));
    } catch {
      setError("Failed to connect to /metrics");
    }
  }, []);

  useEffect(() => { refetch(); }, [refetch]);

  return { data, error, refetch };
}
