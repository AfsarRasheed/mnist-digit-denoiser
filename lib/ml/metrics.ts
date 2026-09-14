/**
 * Reconstruction quality metrics, computed the same way as the notebook's
 * `model.evaluate(X_test_clipped, X_test)`: mean squared error between the
 * model's reconstruction and the ORIGINAL clean image (not the noisy input).
 */
export function meanSquaredError(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum / a.length;
}

/** Peak signal-to-noise ratio in dB, derived from the MSE above (max signal = 1.0 for [0,1] images). */
export function peakSignalToNoiseRatio(mse: number): number {
  if (mse <= 0) return Infinity;
  return 10 * Math.log10(1 / mse);
}

/**
 * Per-pixel |a - b| on 0-255 arrays, contrast-stretched so the largest
 * difference in this image maps to 255 — otherwise real reconstruction
 * error (usually small) would render as an almost-black square. Returns
 * the stretched pixels plus the true (pre-stretch) max delta so the UI can
 * disclose the actual scale rather than implying raw brightness = raw error.
 */
export function pixelDifference(a: number[], b: number[]): { pixels: number[]; maxDelta: number } {
  const raw = a.map((v, i) => Math.abs(v - b[i]));
  const maxDelta = Math.max(...raw);
  const scale = maxDelta > 0 ? 255 / maxDelta : 1;
  return { pixels: raw.map((v) => Math.round(v * scale)), maxDelta };
}
