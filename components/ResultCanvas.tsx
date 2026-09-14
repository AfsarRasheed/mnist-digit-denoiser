"use client";

import { useEffect, useRef } from "react";
import { renderPixelsToCanvas } from "@/lib/preprocessing/mnistAdapter";

interface ResultCanvasProps {
  pixels: number[] | Float32Array | null;
  stage: string;
  label: string;
  description: string;
  meta: string;
  /** Waiting on data that hasn't arrived yet — shows a skeleton, not the last stale image. */
  pending?: boolean;
  emphasize?: boolean;
  size?: number;
  onExpand?: () => void;
}

export default function ResultCanvas({
  pixels,
  stage,
  label,
  description,
  meta,
  pending,
  emphasize,
  size = 116,
  onExpand,
}: ResultCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (pixels && !pending) {
      renderPixelsToCanvas(canvas, pixels, 28);
    } else {
      const ctx = canvas.getContext("2d");
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [pixels, pending]);

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="flex items-baseline gap-1.5">
        <span className={`font-mono text-[11px] ${emphasize ? "text-accent" : "text-muted-2"}`}>{stage}</span>
        <span className={`text-xs font-medium ${emphasize ? "text-accent" : "text-foreground"}`}>{label}</span>
      </div>

      <button
        type="button"
        onClick={onExpand}
        disabled={!onExpand || pending}
        className={`focus-ring rounded-md border bg-black p-1 transition ${
          emphasize ? "border-accent/40" : "border-border-strong"
        } ${onExpand && !pending ? "cursor-zoom-in hover:border-accent/50" : "cursor-default"}`}
        aria-label={onExpand ? `Expand ${label.toLowerCase()} image` : undefined}
      >
        {pending ? (
          <div
            className="animate-pulse rounded-sm bg-white/5"
            style={{ width: size, height: size }}
          />
        ) : (
          <canvas ref={canvasRef} width={size} height={size} className="pixel-canvas rounded-sm" />
        )}
      </button>

      <p className="text-[11px] text-muted-2">{description}</p>
      <p className="font-mono text-[11px] text-muted-2">{meta}</p>
    </div>
  );
}
