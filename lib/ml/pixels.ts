const PIXEL_COUNT = 28 * 28;

/** 0-255 grayscale ints -> [0,1] floats. Shared by the API route and the
 * client-side live noise preview so both normalize identically. */
export function toFloat01(pixels: number[]): Float32Array {
  const out = new Float32Array(PIXEL_COUNT);
  for (let i = 0; i < PIXEL_COUNT; i++) {
    out[i] = Math.min(255, Math.max(0, pixels[i])) / 255;
  }
  return out;
}

/** [0,1] floats -> 0-255 ints, rounded. */
export function to255(pixels: Float32Array): number[] {
  const out = new Array<number>(pixels.length);
  for (let i = 0; i < pixels.length; i++) {
    out[i] = Math.round(Math.min(1, Math.max(0, pixels[i])) * 255);
  }
  return out;
}
