import type {
  ServerInfo,
  HealthStatus,
  ModelsResponse,
  ObjectDetectionResponse,
  ClassificationResponse,
  WorkflowResponse,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

export const api = {
  getInfo: () => request<ServerInfo>("/info"),

  getHealth: (): Promise<HealthStatus> =>
    fetch("/healthz")
      .then((r) => (r.ok ? "healthy" : "error") as HealthStatus)
      .catch(() => "error" as const),

  getModels: () => request<ModelsResponse>("/model/registry"),

  addModel: (model_id: string, model_type?: string, api_key?: string) =>
    request<ModelsResponse>("/model/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id, model_type, api_key }),
    }),

  removeModel: (model_id: string) =>
    request<ModelsResponse>("/model/remove", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id }),
    }),

  clearModels: () =>
    request<ModelsResponse>("/model/clear", { method: "POST" }),

  inferObjectDetection: (body: Record<string, unknown>) =>
    request<ObjectDetectionResponse>("/infer/object_detection", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  inferClassification: (body: Record<string, unknown>) =>
    request<ClassificationResponse>("/infer/classification", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  inferInstanceSegmentation: (body: Record<string, unknown>) =>
    request<ObjectDetectionResponse>("/infer/instance_segmentation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  runWorkflow: (specification: unknown, inputs: Record<string, unknown>) =>
    request<WorkflowResponse>("/workflows/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ specification, inputs }),
    }),
};
