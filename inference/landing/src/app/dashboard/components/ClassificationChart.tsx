"use client";

import React from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import type { ClassificationPrediction } from "../types";

const COLORS = [
  "#6366f1", "#818cf8", "#a5b4fc", "#c7d2fe", "#e0e7ff",
  "#8b5cf6", "#a78bfa", "#c4b5fd",
];

interface ClassificationChartProps {
  predictions: ClassificationPrediction[];
  top: string;
  confidence: number;
}

export function ClassificationChart({ predictions, top, confidence }: ClassificationChartProps) {
  const data = predictions
    .slice(0, 10)
    .map((p) => ({ name: p.class, confidence: +(p.confidence * 100).toFixed(1) }));

  return (
    <div>
      <div className="mb-4 text-center">
        <span className="text-sm text-gray-400">Top prediction:</span>
        <p className="text-xl font-bold text-white">
          {top} <span className="text-accent-hover">({(confidence * 100).toFixed(1)}%)</span>
        </p>
      </div>
      <ResponsiveContainer width="100%" height={Math.max(200, data.length * 36)}>
        <BarChart data={data} layout="vertical" margin={{ left: 80, right: 20 }}>
          <XAxis type="number" domain={[0, 100]} tick={{ fill: "#9ca3af", fontSize: 12 }} />
          <YAxis type="category" dataKey="name" tick={{ fill: "#d1d5db", fontSize: 12 }} width={70} />
          <Tooltip
            contentStyle={{ background: "#1a1d27", border: "1px solid #2a2e3a", borderRadius: "8px", color: "#fff" }}
            formatter={(value) => [`${value}%`, "Confidence"]}
          />
          <Bar dataKey="confidence" radius={[0, 4, 4, 0]}>
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
