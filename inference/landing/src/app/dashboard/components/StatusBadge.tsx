import React from "react";
import type { HealthStatus } from "../types";

const statusConfig: Record<HealthStatus, { color: string; label: string }> = {
  healthy: { color: "bg-green-500", label: "Healthy" },
  error: { color: "bg-red-500", label: "Error" },
  loading: { color: "bg-yellow-500", label: "Loading" },
};

export function StatusBadge({ status }: { status: HealthStatus }) {
  const { color, label } = statusConfig[status];
  return (
    <span className="inline-flex items-center gap-1.5 text-sm">
      <span className={`w-2 h-2 rounded-full ${color}`} />
      {label}
    </span>
  );
}
