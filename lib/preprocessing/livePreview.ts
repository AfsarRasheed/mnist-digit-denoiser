import { corrupt } from "@/lib/ml/noise";
import { toFloat01, to255 } from "@/lib/ml/pixels";

/**
 * Client-side preview of what the model will actually receive: the same
 * `corrupt()` used by the API route, applied to the same normalized pixels.
 * This is not a separate "fake" noise algorithm — it's the authoritative
 * corruption function, just run in the browser for instant slider feedback.
 * The real Process action still calls the server, which re-runs this same
 * function (with a fresh random draw) as the actual model input.
 */
export function previewCorruption(cleanPixels255: number[], sigma: number): number[] {
  return to255(corrupt(toFloat01(cleanPixels255), sigma));
}
