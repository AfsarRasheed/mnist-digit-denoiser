"use client";

import { useEffect, useState, type ReactNode } from "react";
import { usePredict } from "@/hooks/usePredict";
import { MODEL_PARAM_COUNT_DISPLAY } from "@/lib/ml/constants";
import { getPreset } from "@/lib/ml/presets";
import DrawCanvas from "./DrawCanvas";
import UploadPanel from "./UploadPanel";
import PresetGrid from "./PresetGrid";
import PipelineFlow from "./PipelineFlow";
import MetricsPanel from "./MetricsPanel";
import GhostPipeline from "./GhostPipeline";

type Tab = "draw" | "upload" | "preset";
type RevealStep = 0 | 1 | 2;

const TABS: { id: Tab; label: string; icon: ReactNode }[] = [
  {
    id: "draw",
    label: "Draw Digit",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} className="h-4 w-4">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" />
      </svg>
    ),
  },
  {
    id: "upload",
    label: "Upload Image",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} className="h-4 w-4">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
      </svg>
    ),
  },
  {
    id: "preset",
    label: "MNIST Presets",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} className="h-4 w-4">
        <rect x="3" y="3" width="7" height="7" rx="1.5" />
        <rect x="14" y="3" width="7" height="7" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" />
        <rect x="14" y="14" width="7" height="7" rx="1.5" />
      </svg>
    ),
  },
];

const REVEAL_STAGGER_MS = 220;

export default function Studio() {
  const [tab, setTab] = useState<Tab>("draw");
  const { result, isLoading, error, predict, rerun, reset, canRerun } = usePredict();
  const [pendingOriginal, setPendingOriginal] = useState<number[] | null>(null);
  const [revealStep, setRevealStep] = useState<RevealStep>(0);

  // Once a response lands, reveal "corrupted" first, then "reconstructed" +
  // metrics a moment later — the data all arrived together, but a small
  // stagger makes the pipeline read as a sequence rather than a pop-in.
  useEffect(() => {
    if (isLoading || !result) return;
    const stepOne = setTimeout(() => setRevealStep(1), 0);
    const stepTwo = setTimeout(() => setRevealStep(2), REVEAL_STAGGER_MS);
    return () => {
      clearTimeout(stepOne);
      clearTimeout(stepTwo);
    };
  }, [isLoading, result]);

  const startSubmit = (originalPixels: number[], args: Parameters<typeof predict>[0]) => {
    setPendingOriginal(originalPixels);
    setRevealStep(0);
    predict(args);
  };

  const handleDrawSubmit = (pixels: number[], noiseSigma: number) =>
    startSubmit(pixels, { source: "draw", pixels, noiseSigma });
  const handleUploadSubmit = (pixels: number[], noiseSigma: number) =>
    startSubmit(pixels, { source: "upload", pixels, noiseSigma });
  const handlePresetSubmit = (digit: number) =>
    startSubmit(getPreset(digit)?.pixels ?? [], { source: "preset", presetDigit: digit });

  const handleRerun = () => {
    setRevealStep(0);
    rerun();
  };

  const clearResults = () => {
    reset();
    setPendingOriginal(null);
    setRevealStep(0);
  };

  const handleTabChange = (next: Tab) => {
    if (next === tab) return;
    setTab(next);
    clearResults();
  };

  const showResult = Boolean(result) || (isLoading && pendingOriginal !== null);

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-6 px-5 py-8 sm:px-6 sm:py-10 lg:px-8">
      <header className="flex flex-col gap-1.5 sm:flex-row sm:items-start sm:justify-between sm:gap-4">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-foreground">MNIST Digit Denoiser</h1>
          <p className="mt-1 max-w-md text-sm text-muted">
            A convolutional autoencoder that removes Gaussian noise from handwritten digits.
          </p>
        </div>
        <p className="whitespace-nowrap text-sm text-muted sm:pt-0.5">
          <span className="text-accent">●</span> Ready <span className="mx-1.5 text-border-strong">·</span>{" "}
          {MODEL_PARAM_COUNT_DISPLAY} parameters <span className="mx-1.5 text-border-strong">·</span> trained on
          MNIST
        </p>
      </header>

      <div className="workspace flex flex-col lg:flex-row">
        <section className="flex flex-col gap-5 border-b border-border p-5 lg:w-[400px] lg:flex-shrink-0 lg:border-b-0 lg:border-r">
          <div role="tablist" className="no-scrollbar flex gap-2.5 overflow-x-auto border-b border-border sm:gap-5">
            {TABS.map((t) => (
              <button
                key={t.id}
                role="tab"
                aria-selected={tab === t.id}
                onClick={() => handleTabChange(t.id)}
                className={`focus-ring relative flex flex-shrink-0 items-center gap-1.5 whitespace-nowrap pb-2.5 text-[13px] font-medium transition sm:text-sm ${
                  tab === t.id ? "text-accent" : "text-muted hover:text-foreground"
                }`}
              >
                {t.icon}
                <span>{t.label}</span>
                {tab === t.id && <span className="absolute inset-x-0 -bottom-px h-[2px] bg-accent" />}
              </button>
            ))}
          </div>

          <div className="flex flex-1 items-center justify-center">
            {tab === "draw" && (
              <DrawCanvas onSubmit={handleDrawSubmit} onClear={clearResults} isLoading={isLoading} />
            )}
            {tab === "upload" && (
              <UploadPanel onSubmit={handleUploadSubmit} onClear={clearResults} isLoading={isLoading} />
            )}
            {tab === "preset" && (
              <PresetGrid onSubmit={handlePresetSubmit} onClear={clearResults} isLoading={isLoading} />
            )}
          </div>

          {error && (
            <div className="rounded-md border border-danger/25 bg-danger-soft px-3.5 py-2.5 text-sm text-danger">
              {error}
            </div>
          )}
        </section>

        <section className="flex flex-1 flex-col gap-5 p-5">
          <div className="flex items-center justify-between">
            <p className="eyebrow">Results</p>
            {result && !isLoading && canRerun && (
              <button
                onClick={handleRerun}
                className="focus-ring flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium text-muted transition hover:bg-surface-2 hover:text-foreground"
              >
                <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h5M20 20v-5h-5M4.5 9a7.5 7.5 0 0 1 12.8-4.2M19.5 15a7.5 7.5 0 0 1-12.8 4.2" />
                </svg>
                Run again
              </button>
            )}
          </div>

          <div className="flex flex-1 flex-col justify-center">
            {!showResult && (
              <div className="flex flex-col items-center justify-center gap-5 py-10 text-center">
                <GhostPipeline />
                <p className="max-w-xs text-sm text-muted">
                  Draw a digit, upload an image, or select a preset, then run it through the
                  autoencoder to see results here.
                </p>
              </div>
            )}

            {showResult && (
              <div className="animate-fade-in flex flex-col gap-6">
                <PipelineFlow
                  original={result?.original ?? pendingOriginal}
                  corrupted={result?.corrupted ?? null}
                  reconstructed={result?.reconstructed ?? null}
                  noiseSigma={result?.metrics.noiseSigma ?? 0}
                  inferenceTimeMs={result?.metrics.inferenceTimeMs ?? 0}
                  revealStep={isLoading ? 0 : revealStep}
                />
                <div className="h-px bg-border" />
                <MetricsPanel metrics={revealStep === 2 ? result?.metrics ?? null : null} />
              </div>
            )}
          </div>
        </section>
      </div>

      <footer className="text-center text-xs text-muted-2">
        Inference runs against the original trained model — no external ML service is used.
      </footer>
    </div>
  );
}
