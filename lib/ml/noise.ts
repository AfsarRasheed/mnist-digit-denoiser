/**
 * Corruption step, reproduced exactly from the training Colab:
 *
 *   X_noise = X + 0.3 * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
 *   X_clipped = np.clip(X_noise, 0.0, 1.0)
 *
 * i.e. additive, zero-mean Gaussian noise applied to the [0, 1]-normalized
 * image, then clipped back into [0, 1]. Sigma = 0.3 is the exact value the
 * autoencoder was trained on; the app lets Draw/Upload override it (within
 * MIN/MAX below) to demonstrate how the same trained model responds to
 * lighter or heavier corruption than it saw during training. MNIST presets
 * always use TRAINING_SIGMA, so they remain a fixed, comparable reference
 * across all ten digits.
 */
export const TRAINING_SIGMA = 0.3;
export const NOISE_MEAN = 0.0;
export const MIN_SIGMA = 0.05;
export const MAX_SIGMA = 0.8;

export function clampSigma(sigma: number): number {
  return Math.min(MAX_SIGMA, Math.max(MIN_SIGMA, sigma));
}

/** Standard normal sample via the Box-Muller transform. */
function randomNormal(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * Applies Gaussian corruption + clip to a [0,1]-normalized image and
 * returns a new Float32Array (input is left untouched).
 */
export function corrupt(clean: Float32Array, sigma: number = TRAINING_SIGMA): Float32Array {
  const out = new Float32Array(clean.length);
  for (let i = 0; i < clean.length; i++) {
    const noisy = clean[i] + NOISE_MEAN + sigma * randomNormal();
    out[i] = Math.min(1, Math.max(0, noisy));
  }
  return out;
}
