import raw from "./data/presets.json";

/**
 * One deterministic real sample per digit (0-9), taken from the official
 * MNIST test set — the exact same dataset source the training Colab loads
 * via `keras.datasets.mnist.load_data()` (downloaded from
 * storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz).
 *
 * For each digit we picked the first test-set example with that label, so
 * the selection is fixed and reproducible across sessions/builds. See
 * /scripts/extract-model-weights.py for how these were extracted.
 */

export interface PresetDigit {
  digit: number;
  /** length-784 grayscale pixels, 0-255, MNIST convention (white stroke on black bg) */
  pixels: number[];
  testSetIndex: number;
}

type RawPresets = {
  sourceDataset: string;
  indices: Record<string, number>;
  digits: Record<string, number[]>;
};

const data = raw as RawPresets;

export const PRESET_SOURCE_DATASET = data.sourceDataset;

export const PRESETS: PresetDigit[] = Array.from({ length: 10 }, (_, digit) => ({
  digit,
  pixels: data.digits[String(digit)],
  testSetIndex: data.indices[String(digit)],
}));

export function getPreset(digit: number): PresetDigit | undefined {
  return PRESETS.find((p) => p.digit === digit);
}
