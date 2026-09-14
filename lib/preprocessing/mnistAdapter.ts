"use client";

/**
 * Adapter that turns free-form user input (a drawn stroke or an uploaded
 * photo) into the same 28x28, single-channel, MNIST-convention frame the
 * autoencoder was trained on. The training Colab never needed this step —
 * it consumed the pre-built MNIST arrays directly — so this reproduces the
 * standard MNIST construction recipe (bounding-box crop, scale so the
 * digit's longest side is ~20px, center by center of mass) rather than
 * inventing a new one.
 */

export interface Mnist28Result {
  /** length-784 grayscale pixels, 0-255, white stroke on black background */
  pixels: number[];
  isEmpty: boolean;
}

const FRAME_SIZE = 28;
const DIGIT_CORE_SIZE = 20; // matches the ~20px digit height used to build MNIST
const INK_THRESHOLD = 20;

/**
 * Crops to the ink bounding box, scales it to fit a 20px core, and centers
 * the result on a 28x28 canvas using center-of-mass alignment.
 * `source` must already be single-channel MNIST-convention (bright stroke
 * on dark background) in its red channel.
 */
export function toMnistFrame(source: HTMLCanvasElement): Mnist28Result {
  const ctx = source.getContext("2d");
  if (!ctx) return { pixels: new Array(784).fill(0), isEmpty: true };

  const { width, height } = source;
  const { data } = ctx.getImageData(0, 0, width, height);

  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = data[(y * width + x) * 4];
      if (v > INK_THRESHOLD) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < 0) {
    return { pixels: new Array(784).fill(0), isEmpty: true };
  }

  const boxW = maxX - minX + 1;
  const boxH = maxY - minY + 1;
  const margin = Math.round(Math.max(boxW, boxH) * 0.15);

  const cropX = Math.max(0, minX - margin);
  const cropY = Math.max(0, minY - margin);
  const cropW = Math.min(width - cropX, boxW + margin * 2);
  const cropH = Math.min(height - cropY, boxH + margin * 2);

  const scale = DIGIT_CORE_SIZE / Math.max(cropW, cropH);
  const resizedW = Math.max(1, Math.round(cropW * scale));
  const resizedH = Math.max(1, Math.round(cropH * scale));

  const resizeCanvas = document.createElement("canvas");
  resizeCanvas.width = resizedW;
  resizeCanvas.height = resizedH;
  const rctx = resizeCanvas.getContext("2d")!;
  rctx.imageSmoothingEnabled = true;
  rctx.imageSmoothingQuality = "high";
  rctx.drawImage(source, cropX, cropY, cropW, cropH, 0, 0, resizedW, resizedH);

  const finalCanvas = document.createElement("canvas");
  finalCanvas.width = FRAME_SIZE;
  finalCanvas.height = FRAME_SIZE;
  const fctx = finalCanvas.getContext("2d")!;
  fctx.fillStyle = "black";
  fctx.fillRect(0, 0, FRAME_SIZE, FRAME_SIZE);
  const offsetX = Math.floor((FRAME_SIZE - resizedW) / 2);
  const offsetY = Math.floor((FRAME_SIZE - resizedH) / 2);
  fctx.drawImage(resizeCanvas, offsetX, offsetY);

  const { data: fdata } = fctx.getImageData(0, 0, FRAME_SIZE, FRAME_SIZE);

  let sumX = 0;
  let sumY = 0;
  let sumV = 0;
  for (let y = 0; y < FRAME_SIZE; y++) {
    for (let x = 0; x < FRAME_SIZE; x++) {
      const v = fdata[(y * FRAME_SIZE + x) * 4];
      sumX += x * v;
      sumY += y * v;
      sumV += v;
    }
  }

  const pixels = new Array<number>(FRAME_SIZE * FRAME_SIZE).fill(0);

  if (sumV > 0) {
    const comX = sumX / sumV;
    const comY = sumY / sumV;
    const shiftX = Math.round((FRAME_SIZE - 1) / 2 - comX);
    const shiftY = Math.round((FRAME_SIZE - 1) / 2 - comY);

    for (let y = 0; y < FRAME_SIZE; y++) {
      for (let x = 0; x < FRAME_SIZE; x++) {
        const srcX = x - shiftX;
        const srcY = y - shiftY;
        if (srcX >= 0 && srcX < FRAME_SIZE && srcY >= 0 && srcY < FRAME_SIZE) {
          pixels[y * FRAME_SIZE + x] = fdata[(srcY * FRAME_SIZE + srcX) * 4];
        }
      }
    }
  } else {
    for (let i = 0; i < pixels.length; i++) pixels[i] = fdata[i * 4];
  }

  return { pixels, isEmpty: false };
}

/**
 * Decodes an uploaded image file into a grayscale canvas in MNIST
 * convention. Auto-inverts when the image looks like dark ink on a light
 * background (mean brightness > 127), since photos of handwriting are
 * usually captured that way, while MNIST expects a light stroke on a dark
 * background.
 */
export async function imageFileToMnistCanvas(file: File): Promise<HTMLCanvasElement> {
  const bitmap = await createImageBitmap(file);

  const maxDim = 512;
  const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(bitmap, 0, 0, w, h);

  const imageData = ctx.getImageData(0, 0, w, h);
  const { data } = imageData;

  let sum = 0;
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
    gray[i] = luminance;
    sum += luminance;
  }

  const mean = sum / (w * h);
  const invert = mean > 127;

  for (let i = 0; i < w * h; i++) {
    const v = invert ? 255 - gray[i] : gray[i];
    data[i * 4] = v;
    data[i * 4 + 1] = v;
    data[i * 4 + 2] = v;
    data[i * 4 + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

/** Renders a length-784 (or arbitrary square) pixel array onto a canvas, upscaled with crisp nearest-neighbor edges. */
export function renderPixelsToCanvas(
  canvas: HTMLCanvasElement,
  pixels: number[] | Float32Array,
  gridSize = FRAME_SIZE
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const off = document.createElement("canvas");
  off.width = gridSize;
  off.height = gridSize;
  const octx = off.getContext("2d")!;
  const imageData = octx.createImageData(gridSize, gridSize);
  for (let i = 0; i < gridSize * gridSize; i++) {
    const v = Math.round(Math.min(255, Math.max(0, pixels[i])));
    imageData.data[i * 4] = v;
    imageData.data[i * 4 + 1] = v;
    imageData.data[i * 4 + 2] = v;
    imageData.data[i * 4 + 3] = 255;
  }
  octx.putImageData(imageData, 0, 0);

  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
}
