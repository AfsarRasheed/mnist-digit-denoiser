"use client";

import { useEffect, useState } from "react";

/**
 * The real pipeline is one synchronous request, so these phases aren't
 * independently timed — they're shown in their real, correct order to make
 * the workflow legible during the (usually sub-second) wait, not to imply
 * granular server-reported progress.
 */
const PHASES = [
  "Preparing input",
  "Adding Gaussian noise",
  "Running the autoencoder",
  "Reconstructing output",
  "Calculating metrics",
] as const;

export function useProcessingPhase(isActive: boolean): string {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (!isActive) return;
    // Deferred (not synchronous in the effect body) so each run's narrative
    // restarts from the first phase.
    const resetTimer = setTimeout(() => setIndex(0), 0);
    const interval = setInterval(() => {
      setIndex((i) => (i + 1) % PHASES.length);
    }, 900);
    return () => {
      clearTimeout(resetTimer);
      clearInterval(interval);
    };
  }, [isActive]);

  return PHASES[index];
}
