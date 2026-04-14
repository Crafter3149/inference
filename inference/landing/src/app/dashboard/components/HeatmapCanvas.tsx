"use client";

import React, { useEffect, useRef } from "react";

interface HeatmapCanvasProps {
  imageSrc: string;
  anomalyMap: number[][] | number[][][];
  anomalyScore: number;
}

/** Map a 0-1 value to a turbo-like RGB color. */
function turboColormap(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.min(1, Math.max(0, 0.13572 + t * (4.6153 + t * (-42.659 + t * (132.13 + t * (-152.95 + t * 56.289)))))));
  const g = Math.round(255 * Math.min(1, Math.max(0, 0.09140 + t * (2.1275 + t * (-14.051 + t * (57.376 + t * (-90.649 + t * 43.713)))))));
  const b = Math.round(255 * Math.min(1, Math.max(0, 0.10667 + t * (12.289 + t * (-60.582 + t * (132.73 + t * (-134.93 + t * 50.109)))))));
  return [r, g, b];
}

export function HeatmapCanvas({ imageSrc, anomalyMap, anomalyScore }: HeatmapCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // anomalyMap may be (1, H, W) or (H, W)
    let map2d: number[][];
    if (anomalyMap.length > 0 && Array.isArray(anomalyMap[0]) && Array.isArray((anomalyMap[0] as number[][])[0])) {
      map2d = anomalyMap[0] as number[][];
    } else {
      map2d = anomalyMap as number[][];
    }

    const mapH = map2d.length;
    const mapW = map2d[0]?.length ?? 0;
    if (mapH === 0 || mapW === 0) return;

    // Find min/max for normalization
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let y = 0; y < mapH; y++) {
      for (let x = 0; x < mapW; x++) {
        const v = map2d[y][x];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
    }
    const range = maxVal - minVal || 1;

    const img = new Image();
    img.onload = () => {
      const maxDisplayW = canvas.parentElement?.clientWidth ?? 640;
      const scale = Math.min(maxDisplayW / img.width, 1);
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;

      // Draw original image
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // Create heatmap overlay on offscreen canvas
      const heatCanvas = document.createElement("canvas");
      heatCanvas.width = mapW;
      heatCanvas.height = mapH;
      const heatCtx = heatCanvas.getContext("2d")!;
      const imageData = heatCtx.createImageData(mapW, mapH);

      for (let y = 0; y < mapH; y++) {
        for (let x = 0; x < mapW; x++) {
          const norm = (map2d[y][x] - minVal) / range;
          const [r, g, b] = turboColormap(norm);
          const idx = (y * mapW + x) * 4;
          imageData.data[idx] = r;
          imageData.data[idx + 1] = g;
          imageData.data[idx + 2] = b;
          imageData.data[idx + 3] = 160; // alpha for overlay blending
        }
      }
      heatCtx.putImageData(imageData, 0, 0);

      // Composite heatmap over the image
      ctx.drawImage(heatCanvas, 0, 0, canvas.width, canvas.height);

      // Draw score label
      const label = `Score: ${anomalyScore.toFixed(4)}`;
      ctx.font = "bold 14px sans-serif";
      const textW = ctx.measureText(label).width + 12;
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      ctx.fillRect(4, 4, textW, 24);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, 10, 21);
    };
    img.src = imageSrc;
  }, [imageSrc, anomalyMap, anomalyScore]);

  return <canvas ref={canvasRef} className="max-w-full rounded-lg" />;
}
