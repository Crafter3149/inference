"use client";

import React, { useEffect, useRef } from "react";
import type { ObjectDetectionPrediction } from "../types";

const COLORS = [
  "#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6",
  "#ec4899", "#06b6d4", "#f97316", "#14b8a6", "#a855f7",
];

interface BBoxCanvasProps {
  imageSrc: string;
  predictions: ObjectDetectionPrediction[];
  imageWidth: number;
  imageHeight: number;
}

export function BBoxCanvas({ imageSrc, predictions, imageWidth, imageHeight }: BBoxCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      const maxW = canvas.parentElement?.clientWidth ?? 640;
      const scale = Math.min(maxW / img.width, 1);
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      const scaleX = canvas.width / imageWidth;
      const scaleY = canvas.height / imageHeight;
      const classSet = Array.from(new Set(predictions.map((p) => p.class)));
      const colorMap = new Map(classSet.map((c, i) => [c, COLORS[i % COLORS.length]]));

      for (const pred of predictions) {
        const color = colorMap.get(pred.class) ?? COLORS[0];
        const x = (pred.x - pred.width / 2) * scaleX;
        const y = (pred.y - pred.height / 2) * scaleY;
        const w = pred.width * scaleX;
        const h = pred.height * scaleY;

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        const label = `${pred.class} ${(pred.confidence * 100).toFixed(0)}%`;
        ctx.font = "bold 12px sans-serif";
        const textW = ctx.measureText(label).width + 8;
        ctx.fillStyle = color;
        ctx.fillRect(x, y - 18, textW, 18);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, x + 4, y - 5);
      }
    };
    img.src = imageSrc;
  }, [imageSrc, predictions, imageWidth, imageHeight]);

  return <canvas ref={canvasRef} className="max-w-full rounded-lg" />;
}
