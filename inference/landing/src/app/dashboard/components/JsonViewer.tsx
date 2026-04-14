"use client";

import React, { useState } from "react";

interface JsonViewerProps {
  data: unknown;
  maxHeight?: string;
}

const MAX_JSON_LENGTH = 100_000;

export function JsonViewer({ data, maxHeight = "24rem" }: JsonViewerProps) {
  const [collapsed, setCollapsed] = useState(false);
  const raw = JSON.stringify(data, null, 2);
  const truncated = raw.length > MAX_JSON_LENGTH;
  const formatted = truncated ? raw.slice(0, MAX_JSON_LENGTH) + "\n\n... (truncated)" : raw;

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-gray-500">JSON Response</span>
        <div className="flex gap-2">
          <button onClick={() => setCollapsed(!collapsed)} className="text-xs text-gray-400 hover:text-white">
            {collapsed ? "Expand" : "Collapse"}
          </button>
          <button onClick={() => navigator.clipboard.writeText(raw)} className="text-xs text-gray-400 hover:text-white">
            Copy
          </button>
        </div>
      </div>
      {!collapsed && (
        <pre
          className="bg-surface rounded-lg p-4 text-xs text-gray-300 overflow-auto border border-border"
          style={{ maxHeight }}
        >
          {formatted}
        </pre>
      )}
    </div>
  );
}
