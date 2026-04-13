import { useState, useEffect, useCallback } from "react";
import { api } from "../api";
import type { ModelInfo } from "../types";

export function useModelsData() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refetch = useCallback(async () => {
    try {
      const data = await api.getModels();
      setModels(data.models || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch models");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refetch();
  }, [refetch]);

  const addModel = async (model_id: string, model_type?: string, api_key?: string) => {
    const data = await api.addModel(model_id, model_type, api_key);
    setModels(data.models);
    return data;
  };

  const removeModel = async (model_id: string) => {
    const data = await api.removeModel(model_id);
    setModels(data.models);
    return data;
  };

  const clearModels = async () => {
    const data = await api.clearModels();
    setModels(data.models);
    return data;
  };

  return { models, loading, error, refetch, addModel, removeModel, clearModels };
}
