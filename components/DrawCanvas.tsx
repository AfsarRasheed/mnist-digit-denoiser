"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toMnistFrame, renderPixelsToCanvas } from "@/lib/preprocessing/mnistAdapter";
import { previewCorruption } from "@/lib/preprocessing/livePreview";
import { TRAINING_SIGMA } from "@/lib/ml/noise";
import NoiseSlider from "./NoiseSlider";
import FlowArrow from "./FlowArrow";

interface DrawCanvasProps {
  onSubmit: (pixels: number[], noiseSigma: number) => void;
  onClear: () => void;
  isLoading: boolean;
}

const CANVAS_SIZE = 248;
const STROKE_WIDTH = 18;

export default function DrawCanvas({ onSubmit, onClear, isLoading }: DrawCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewRef = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const lastPoint = useRef<{ x: number; y: number } | null>(null);
  const [isEmpty, setIsEmpty] = useState(true);
  const [cleanPixels, setCleanPixels] = useState<number[] | null>(null);
  const [sigma, setSigma] = useState(TRAINING_SIGMA);

  // Live preview of what the model actually receives: the real corrupt()
  // function applied to the current drawing at the current noise level.
  const livePreview = useMemo(
    () => (cleanPixels ? previewCorruption(cleanPixels, sigma) : null),
    [cleanPixels, sigma]
  );

  useEffect(() => {
    const preview = previewRef.current;
    if (!preview) return;
    if (livePreview) renderPixelsToCanvas(preview, livePreview, 28);
    else preview.getContext("2d")?.clearRect(0, 0, preview.width, preview.height);
  }, [livePreview]);

  const paintBackground = useCallback(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  }, []);

  useEffect(() => {
    paintBackground();
  }, [paintBackground]);

  const updatePreview = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { pixels, isEmpty: empty } = toMnistFrame(canvas);
    setCleanPixels(empty ? null : pixels);
  }, []);

  const getPoint = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * CANVAS_SIZE,
      y: ((e.clientY - rect.top) / rect.height) * CANVAS_SIZE,
    };
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    canvasRef.current?.setPointerCapture(e.pointerId);
    drawing.current = true;
    lastPoint.current = getPoint(e);
    setIsEmpty(false);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawing.current) return;
    e.preventDefault();
    const ctx = canvasRef.current?.getContext("2d");
    const point = getPoint(e);
    if (ctx && lastPoint.current) {
      ctx.strokeStyle = "white";
      ctx.lineWidth = STROKE_WIDTH;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(lastPoint.current.x, lastPoint.current.y);
      ctx.lineTo(point.x, point.y);
      ctx.stroke();
    }
    lastPoint.current = point;
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    drawing.current = false;
    lastPoint.current = null;
    canvasRef.current?.releasePointerCapture(e.pointerId);
    updatePreview();
  };

  const handleClear = () => {
    paintBackground();
    setIsEmpty(true);
    setCleanPixels(null);
    onClear();
  };

  const handleProcess = () => {
    if (!cleanPixels) return;
    onSubmit(cleanPixels, sigma);
  };

  return (
    <div className="flex w-full flex-col items-center gap-5">
      <div className="flex flex-col items-center gap-3 sm:flex-row sm:items-center sm:gap-4">
        <div className="flex flex-col items-center gap-2">
          <p className="text-xs text-muted">Your input</p>
          <div className="rounded-lg border border-border bg-surface-2 p-2.5">
            <div className="relative">
              <canvas
                ref={canvasRef}
                width={CANVAS_SIZE}
                height={CANVAS_SIZE}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerCancel={handlePointerUp}
                className="touch-none rounded-md border border-border-strong"
                style={{ width: CANVAS_SIZE, height: CANVAS_SIZE, maxWidth: "100%", cursor: "crosshair" }}
              />
              {isEmpty && (
                <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                  <p className="text-sm text-white/35">Draw a digit, 0–9</p>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="flex flex-col items-center gap-1 text-muted-2">
          <FlowArrow vertical className="sm:hidden" />
          <FlowArrow className="hidden sm:block" />
          <span className="text-[10px] leading-none">preprocess + noise</span>
        </div>

        <div className="flex flex-col items-center gap-2">
          <p className="text-xs text-muted">Model input</p>
          <div className="rounded-md border border-border-strong bg-black p-1">
            <canvas
              ref={previewRef}
              width={28}
              height={28}
              className="pixel-canvas rounded-sm"
              style={{ width: 64, height: 64 }}
            />
          </div>
          <p className="font-mono text-[11px] text-muted-2">σ = {sigma.toFixed(2)}</p>
        </div>
      </div>

      <NoiseSlider value={sigma} onChange={setSigma} disabled={isLoading} />

      <div className="flex w-full max-w-xs gap-3">
        <button
          onClick={handleClear}
          disabled={isLoading}
          className="focus-ring flex-1 rounded-md border border-border-strong bg-surface px-4 py-2 text-sm font-medium text-foreground transition hover:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Clear
        </button>
        <button
          onClick={handleProcess}
          disabled={isEmpty || isLoading}
          className="focus-ring flex flex-1 items-center justify-center gap-1.5 rounded-md bg-accent px-4 py-2 text-sm font-medium text-white transition hover:bg-accent-hover active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40 disabled:active:scale-100"
        >
          {isLoading ? (
            <>
              <span className="h-3.5 w-3.5 animate-spin-slow rounded-full border-2 border-white/30 border-t-white" />
              Processing
            </>
          ) : (
            <>
              Process Drawing
              <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 12h14m0 0-5-5m5 5-5 5" />
              </svg>
            </>
          )}
        </button>
      </div>
    </div>
  );
}
