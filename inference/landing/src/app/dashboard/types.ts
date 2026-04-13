// Server
export interface ServerInfo {
  name: string;
  version: string;
  uuid: string;
}

export type HealthStatus = "healthy" | "error" | "loading";

// Models
export interface ModelInfo {
  model_id: string;
  task_type: string;
  batch_size: number | null;
  input_height: number | null;
  input_width: number | null;
}

export interface ModelsResponse {
  models: ModelInfo[];
}

// Inference - Object Detection / Instance Segmentation
export interface ObjectDetectionPrediction {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  class: string;
  class_id: number;
  detection_id: string;
  parent_id: string | null;
}

export interface ObjectDetectionResponse {
  image: { width: number; height: number };
  predictions: ObjectDetectionPrediction[];
  visualization: string | null;
  time: number;
}

// Inference - Classification
export interface ClassificationPrediction {
  class: string;
  class_id: number;
  confidence: number;
}

export interface ClassificationResponse {
  image: { width: number; height: number };
  predictions: ClassificationPrediction[];
  top: string;
  confidence: number;
  time: number;
}

// Workflow
export interface WorkflowResponse {
  outputs: Record<string, unknown>[];
  profiler_trace: unknown[] | null;
}
