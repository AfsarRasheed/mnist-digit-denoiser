"use client";

import { useEffect, useRef, useState } from "react";
import { PRESETS } from "@/lib/ml/presets";
import { renderPixelsToCanvas } from "@/lib/preprocessing/mnistAdapter";
import { TRAINING_SIGMA } from "@/lib/ml/noise";

interface PresetGridProps {
  onSubmit: (digit: number) => void;
  onClear: () => void;
  isLoading: boolean;
}

function PresetThumb({
  digit,
  pixels,
  selected,
  onClick,
}: {
  digit: number;
  pixels: number[];
  selected: boolean;
  onClick: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (canvasRef.current) renderPixelsToCanvas(canvasRef.current, pixels, 28);
  }, [pixels]);

  return (
    <button
      onClick={onClick}
      aria-pressed={selected}
      className={`focus-ring group relative flex flex-col items-center gap-1.5 rounded-md border p-2 transition ${
        selected ? "border-accent bg-accent-soft" : "border-border bg-surface hover:border-border-strong"
      }`}
    >
      <div className="rounded-sm border border-border-strong bg-black p-1">
        <canvas ref={canvasRef} width={28} height={28} className="pixel-canvas rounded-sm" style={{ width: 44, height: 44 }} />
      </div>
      <span className={`text-xs font-medium ${selected ? "text-accent" : "text-muted group-hover:text-foreground"}`}>
        {digit}
      </span>
    </button>
  );
}

export default function PresetGrid({ onSubmit, onClear, isLoading }: PresetGridProps) {
  const [selected, setSelected] = useState<number | null>(null);

  const handleSelect = (digit: number) => {
    if (digit === selected) return;
    setSelected(digit);
    onClear();
  };

  return (
    <div className="flex w-full flex-col items-center gap-4">
      <p className="max-w-xs text-center text-xs leading-relaxed text-muted">
        One real MNIST test sample per digit, corrupted at the fixed training noise level (σ{" "}
        {TRAINING_SIGMA.toFixed(2)}) for a consistent comparison across digits.
      </p>

      <div className="grid grid-cols-5 gap-2 sm:gap-3">
        {PRESETS.map((preset) => (
          <PresetThumb
            key={preset.digit}
            digit={preset.digit}
            pixels={preset.pixels}
            selected={selected === preset.digit}
            onClick={() => handleSelect(preset.digit)}
          />
        ))}
      </div>

      <div className="flex w-full max-w-xs flex-col items-center gap-2.5">
        <p className="text-xs text-muted">
          {selected === null ? (
            "Select a digit to begin"
          ) : (
            <>
              Selected sample <span className="font-mono font-medium text-foreground">{selected}</span>
              <span className="mx-1.5 text-border-strong">·</span>28 × 28 · grayscale
            </>
          )}
        </p>
        <button
          onClick={() => selected !== null && onSubmit(selected)}
          disabled={selected === null || isLoading}
          className="focus-ring flex w-full items-center justify-center gap-1.5 rounded-md bg-accent px-4 py-2 text-sm font-medium text-white transition hover:bg-accent-hover active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40 disabled:active:scale-100"
        >
          {isLoading ? (
            <>
              <span className="h-3.5 w-3.5 animate-spin-slow rounded-full border-2 border-white/30 border-t-white" />
              Processing
            </>
          ) : (
            <>
              Process
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
