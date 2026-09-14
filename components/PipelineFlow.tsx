"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ResultCanvas from "./ResultCanvas";
import ModelNode from "./ModelNode";
import FlowArrow from "./FlowArrow";
import { renderPixelsToCanvas } from "@/lib/preprocessing/mnistAdapter";
import { pixelDifference } from "@/lib/ml/metrics";

interface PipelineFlowProps {
  original: number[] | null;
  corrupted: number[] | null;
  reconstructed: number[] | null;
  noiseSigma: number;
  inferenceTimeMs: number;
  /** 0 = only original is in; 1 = corrupted has arrived; 2 = fully revealed. */
  revealStep: 0 | 1 | 2;
}

type StageId = "original" | "corrupted" | "reconstructed";

export default function PipelineFlow({
  original,
  corrupted,
  reconstructed,
  noiseSigma,
  inferenceTimeMs,
  revealStep,
}: PipelineFlowProps) {
  const [expanded, setExpanded] = useState<StageId | null>(null);
  const stage2Pending = revealStep < 1 || !corrupted;
  const stage3Pending = revealStep < 2 || !reconstructed;

  useEffect(() => {
    if (expanded) {
      const onKey = (e: KeyboardEvent) => e.key === "Escape" && setExpanded(null);
      window.addEventListener("keydown", onKey);
      return () => window.removeEventListener("keydown", onKey);
    }
  }, [expanded]);

  return (
    <div>
      <div className="flex flex-col items-center gap-3 sm:flex-row sm:items-start sm:justify-center sm:gap-2.5">
        <ResultCanvas
          stage="01"
          label="Original"
          pixels={original}
          description="Clean input"
          meta="28 × 28 · grayscale"
          onExpand={original ? () => setExpanded("original") : undefined}
        />

        <div className="pt-8">
          <FlowArrow vertical className="sm:hidden" />
          <FlowArrow className="hidden sm:block" />
        </div>

        <ResultCanvas
          stage="02"
          label="Corrupted"
          pixels={corrupted}
          pending={stage2Pending}
          description={stage2Pending ? "Adding Gaussian noise…" : "Noise applied"}
          meta={stage2Pending ? "—" : `σ = ${noiseSigma.toFixed(2)}`}
          onExpand={corrupted && !stage2Pending ? () => setExpanded("corrupted") : undefined}
        />

        <div className="pt-8">
          <ModelNode active={stage3Pending} className="sm:hidden" vertical />
          <ModelNode active={stage3Pending} className="hidden sm:flex" />
        </div>

        <ResultCanvas
          stage="03"
          label="Reconstructed"
          pixels={reconstructed}
          pending={stage3Pending}
          description={stage3Pending ? "Reconstructing…" : "Autoencoder output"}
          meta={stage3Pending ? "—" : `${inferenceTimeMs.toFixed(2)} ms`}
          emphasize={!stage3Pending}
          onExpand={reconstructed && !stage3Pending ? () => setExpanded("reconstructed") : undefined}
        />
      </div>

      {expanded && original && (
        <ExpandModal
          stage={expanded}
          original={original}
          corrupted={corrupted}
          reconstructed={reconstructed}
          onClose={() => setExpanded(null)}
        />
      )}
    </div>
  );
}

const STAGE_LABEL: Record<StageId, string> = {
  original: "Original",
  corrupted: "Corrupted",
  reconstructed: "Reconstructed",
};

function ExpandModal({
  stage,
  original,
  corrupted,
  reconstructed,
  onClose,
}: {
  stage: StageId;
  original: number[];
  corrupted: number[] | null;
  reconstructed: number[] | null;
  onClose: () => void;
}) {
  const [showDifference, setShowDifference] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const stagePixels: Record<StageId, number[] | null> = { original, corrupted, reconstructed };
  const activePixels = stagePixels[stage];

  const difference = useMemo(
    () => (reconstructed ? pixelDifference(reconstructed, original) : null),
    [reconstructed, original]
  );

  const canShowDifference = stage === "reconstructed" && difference !== null;
  const displayPixels = showDifference && difference ? difference.pixels : activePixels;

  const handleDownload = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const suffix = showDifference ? "difference" : stage;
      a.href = url;
      a.download = `mnist-denoiser-${suffix}.png`;
      a.click();
      URL.revokeObjectURL(url);
    }, "image/png");
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`${STAGE_LABEL[stage]} image, enlarged`}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-6"
      onClick={onClose}
    >
      <div
        className="flex max-w-[calc(100vw-3rem)] flex-col items-center gap-3 rounded-md bg-surface p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex w-full items-center justify-between gap-4">
          <p className="text-sm font-medium text-foreground">
            {showDifference ? "Difference" : STAGE_LABEL[stage]}
          </p>
          <div className="flex items-center gap-1">
            <button
              onClick={handleDownload}
              className="focus-ring rounded-sm p-0.5 text-muted hover:text-foreground"
              aria-label="Download PNG"
              title="Download PNG"
            >
              <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v11m0 0-4-4m4 4 4-4M5 19h14" />
              </svg>
            </button>
            <button
              onClick={onClose}
              className="focus-ring rounded-sm p-0.5 text-muted hover:text-foreground"
              aria-label="Close"
            >
              <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 6l12 12M18 6 6 18" />
              </svg>
            </button>
          </div>
        </div>

        <div className="rounded-md border border-border-strong bg-black p-1.5">
          <canvas
            ref={(el) => {
              canvasRef.current = el;
              if (el && displayPixels) renderPixelsToCanvas(el, displayPixels, 28);
            }}
            width={240}
            height={240}
            className="pixel-canvas rounded-sm"
          />
        </div>

        {canShowDifference ? (
          <div className="flex w-full items-center justify-center gap-1 rounded-md border border-border bg-surface-2 p-0.5 text-xs">
            <button
              onClick={() => setShowDifference(false)}
              className={`focus-ring flex-1 rounded-sm py-1 font-medium transition ${
                !showDifference ? "bg-surface text-foreground shadow-sm" : "text-muted hover:text-foreground"
              }`}
            >
              Image
            </button>
            <button
              onClick={() => setShowDifference(true)}
              className={`focus-ring flex-1 rounded-sm py-1 font-medium transition ${
                showDifference ? "bg-surface text-foreground shadow-sm" : "text-muted hover:text-foreground"
              }`}
            >
              Difference
            </button>
          </div>
        ) : null}

        <p className="text-center text-[11px] leading-relaxed text-muted-2">
          {showDifference && difference
            ? `|Reconstructed − Original| per pixel, contrast-stretched for visibility. Brighter = larger error. Actual max difference: ${difference.maxDelta}/255.`
            : "28 × 28 · grayscale"}
        </p>
      </div>
    </div>
  );
}
