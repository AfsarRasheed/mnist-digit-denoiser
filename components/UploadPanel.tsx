"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { imageFileToMnistCanvas, toMnistFrame, renderPixelsToCanvas } from "@/lib/preprocessing/mnistAdapter";
import { previewCorruption } from "@/lib/preprocessing/livePreview";
import { TRAINING_SIGMA } from "@/lib/ml/noise";
import NoiseSlider from "./NoiseSlider";
import FlowArrow from "./FlowArrow";

interface UploadPanelProps {
  onSubmit: (pixels: number[], noiseSigma: number) => void;
  onClear: () => void;
  isLoading: boolean;
}

const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp"];
const MAX_SIZE_BYTES = 5 * 1024 * 1024;

export default function UploadPanel({ onSubmit, onClear, isLoading }: UploadPanelProps) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [extractedPixels, setExtractedPixels] = useState<number[] | null>(null);
  const [isDecoding, setIsDecoding] = useState(false);
  const [sigma, setSigma] = useState(TRAINING_SIGMA);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);

  // Live preview of what the model actually receives: the real corrupt()
  // function applied to the extracted digit at the current noise level.
  const livePreview = useMemo(
    () => (extractedPixels ? previewCorruption(extractedPixels, sigma) : null),
    [extractedPixels, sigma]
  );

  useEffect(() => {
    const preview = previewCanvasRef.current;
    if (!preview) return;
    if (livePreview) renderPixelsToCanvas(preview, livePreview, 28);
    else preview.getContext("2d")?.clearRect(0, 0, preview.width, preview.height);
  }, [livePreview]);

  const processFile = useCallback(async (file: File) => {
    setError(null);
    setExtractedPixels(null);

    if (!ACCEPTED_TYPES.includes(file.type)) {
      setError("Unsupported file type. Please upload a PNG, JPG, or WEBP image.");
      return;
    }
    if (file.size > MAX_SIZE_BYTES) {
      setError("That image is too large. Please upload a file under 5 MB.");
      return;
    }

    setIsDecoding(true);
    try {
      const objectUrl = URL.createObjectURL(file);
      setPreviewUrl(objectUrl);

      const mnistCanvas = await imageFileToMnistCanvas(file);
      const { pixels, isEmpty } = toMnistFrame(mnistCanvas);

      if (isEmpty) {
        setError("Couldn't find a digit in that image. Try a clearer photo with more contrast.");
        setExtractedPixels(null);
      } else {
        setExtractedPixels(pixels);
      }
    } catch {
      setError("This image couldn't be read. Please try a different file.");
    } finally {
      setIsDecoding(false);
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  };

  // Ctrl+V / Cmd+V anywhere while this tab is open pastes an image from the clipboard.
  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.kind === "file" && item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) {
            e.preventDefault();
            processFile(file);
          }
          return;
        }
      }
    };
    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, [processFile]);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) processFile(file);
  };

  const handleClear = () => {
    setPreviewUrl(null);
    setExtractedPixels(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    onClear();
  };

  const handleProcess = () => {
    if (extractedPixels) onSubmit(extractedPixels, sigma);
  };

  return (
    <div className="flex w-full flex-col items-center gap-5">
      {!previewUrl ? (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragOver(true);
          }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click();
          }}
          className={`focus-ring flex h-44 w-full max-w-sm cursor-pointer flex-col items-center justify-center gap-1.5 rounded-md border border-dashed transition ${
            isDragOver ? "border-accent bg-accent-soft" : "border-border-strong bg-surface-2 hover:border-accent/50"
          }`}
        >
          <svg className="h-7 w-7 text-muted" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
          </svg>
          <p className="text-sm font-medium text-foreground">Drop a digit image here</p>
          <p className="text-xs text-muted">Click to browse, or paste with Ctrl+V</p>
          <p className="text-[11px] text-muted-2">PNG, JPG, or WEBP · up to 5 MB</p>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/jpg,image/webp"
            onChange={handleFileChange}
            className="hidden"
          />
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3 sm:flex-row sm:items-center sm:gap-4">
          <div className="flex flex-col items-center gap-2">
            <p className="text-xs text-muted">Your input</p>
            <div className="rounded-lg border border-border bg-surface-2 p-2.5">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={previewUrl}
                alt="Uploaded digit preview"
                className="h-28 w-28 rounded-md border border-border-strong object-contain bg-black"
              />
            </div>
          </div>

          <div className="flex flex-col items-center gap-1 text-muted-2">
            <FlowArrow vertical className="sm:hidden" />
            <FlowArrow className="hidden sm:block" />
            <span className="text-[10px] leading-none">preprocess + noise</span>
          </div>

          <div className="flex flex-col items-center gap-2">
            <p className="text-xs text-muted">Model input</p>
            <div className="flex h-16 w-16 items-center justify-center rounded-md border border-border-strong bg-black p-1">
              {isDecoding ? (
                <span className="text-xs text-muted-2">…</span>
              ) : (
                <canvas
                  ref={previewCanvasRef}
                  width={28}
                  height={28}
                  className="pixel-canvas rounded-sm"
                  style={{ width: 64, height: 64 }}
                />
              )}
            </div>
            <p className="font-mono text-[11px] text-muted-2">σ = {sigma.toFixed(2)}</p>
          </div>
        </div>
      )}

      {error && <p className="max-w-sm text-center text-sm text-danger">{error}</p>}

      <NoiseSlider value={sigma} onChange={setSigma} disabled={isLoading} />

      <div className="flex w-full max-w-xs gap-3">
        <button
          onClick={handleClear}
          disabled={isLoading || !previewUrl}
          className="focus-ring flex-1 rounded-md border border-border-strong bg-surface px-4 py-2 text-sm font-medium text-foreground transition hover:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Clear
        </button>
        <button
          onClick={handleProcess}
          disabled={!extractedPixels || isLoading || isDecoding}
          className="focus-ring flex flex-1 items-center justify-center gap-1.5 rounded-md bg-accent px-4 py-2 text-sm font-medium text-white transition hover:bg-accent-hover active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40 disabled:active:scale-100"
        >
          {isLoading ? (
            <>
              <span className="h-3.5 w-3.5 animate-spin-slow rounded-full border-2 border-white/30 border-t-white" />
              Processing
            </>
          ) : (
            <>
              Process Image
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
