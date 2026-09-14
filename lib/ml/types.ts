/**
 * Shared types for the denoising autoencoder pipeline.
 * Pixel arrays are always length 784 (28*28, row-major, single channel),
 * values in [0, 255], matching the MNIST convention used by the source
 * Colab notebook: background = 0 (black), stroke = high value (white).
 */

export type InputSource = "draw" | "upload" | "preset";

export interface PredictRequestBody {
  pixels: number[];
  source: InputSource;
  presetDigit?: number;
  /** Only honored for "draw" and "upload" — presets always use the fixed training sigma. */
  noiseSigma?: number;
}

export interface PredictMetrics {
  mse: number;
  psnr: number;
  noiseSigma: number;
  noiseMean: number;
  inputDims: string;
  inferenceTimeMs: number;
  modelStatus: "ready";
  paramCount: number;
}

export interface PredictResponseBody {
  original: number[];
  corrupted: number[];
  reconstructed: number[];
  metrics: PredictMetrics;
}

export interface PredictErrorBody {
  error: string;
}
