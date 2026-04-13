import { useState, useEffect, useCallback } from "react";
import { api } from "../api";
import type { ServerInfo, HealthStatus } from "../types";

export function useServerData() {
  const [serverInfo, setServerInfo] = useState<ServerInfo | null>(null);
  const [healthStatus, setHealthStatus] = useState<HealthStatus>("loading");
  const [error, setError] = useState<string | null>(null);

  const refetch = useCallback(async () => {
    try {
      const [info, health] = await Promise.all([api.getInfo(), api.getHealth()]);
      setServerInfo(info);
      setHealthStatus(health);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Server connection failed");
      setHealthStatus("error");
    }
  }, []);

  useEffect(() => {
    refetch();
  }, [refetch]);

  return { serverInfo, healthStatus, error, refetch };
}
