"use client";

import React, { useRef, useState, useCallback } from "react";

interface ImageUploaderProps {
  onImageSelect: (base64: string, file: File) => void;
  disabled?: boolean;
}

export function ImageUploader({ onImageSelect, disabled }: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const processFile = useCallback(
    (file: File) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        setPreview(result);
        const base64 = result.split(",")[1];
        onImageSelect(base64, file);
      };
      reader.readAsDataURL(file);
    },
    [onImageSelect],
  );

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) processFile(file);
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      className={`relative border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${
        dragOver ? "border-accent bg-accent/5" : "border-border hover:border-gray-500"
      } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) processFile(file);
        }}
      />
      {preview ? (
        <img src={preview} alt="Preview" className="max-h-48 mx-auto rounded-lg" />
      ) : (
        <div className="text-gray-500">
          <p className="text-lg mb-1">Drop image here or click to browse</p>
          <p className="text-xs">PNG, JPG, BMP supported</p>
        </div>
      )}
    </div>
  );
}
