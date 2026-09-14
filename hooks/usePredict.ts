"use client";

import { useCallback, useRef, useState } from "react";
import type { InputSource, PredictResponseBody } from "@/lib/ml/types";
import { API_BASE_URL } from "@/lib/config";

interface PredictArgs {
  source: InputSource;
  pixels?: number[];
  presetDigit?: number;
  noiseSigma?: number;
}

interface UsePredictState {
  result: PredictResponseBody | null;
  isLoading: boolean;
  error: string | null;
  canRerun: boolean;
}

const FRIENDLY_NETWORK_ERROR =
  "Couldn't reach the inference service. Check your connection and try again.";

export function usePredict() {
  const [state, setState] = useState<UsePredictState>({
    result: null,
    isLoading: false,
    error: null,
    canRerun: false,
  });
  const inFlight = useRef(false);
  const lastArgs = useRef<PredictArgs | null>(null);

  const predict = useCallback(async (args: PredictArgs) => {
    if (inFlight.current) return; // avoid duplicate concurrent requests
    inFlight.current = true;
    lastArgs.current = args;
    setState((s) => ({ ...s, isLoading: true, error: null, canRerun: true }));

    try {
      const res = await fetch(`${API_BASE_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(args),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.error ?? "The model couldn't process that input. Please try again.");
      }

      const data: PredictResponseBody = await res.json();
      setState((s) => ({ ...s, result: data, isLoading: false, error: null }));
    } catch (err) {
      const message = err instanceof TypeError ? FRIENDLY_NETWORK_ERROR : (err as Error).message;
      setState((s) => ({ ...s, isLoading: false, error: message }));
    } finally {
      inFlight.current = false;
    }
  }, []);

  /** Re-submits the last input. Since corruption uses fresh random noise per
   * request, this draws a new noisy sample from the same clean input. */
  const rerun = useCallback(() => {
    if (lastArgs.current) predict(lastArgs.current);
  }, [predict]);

  const reset = useCallback(() => {
    lastArgs.current = null;
    setState({ result: null, isLoading: false, error: null, canRerun: false });
  }, []);

  return { ...state, predict, rerun, reset };
}
