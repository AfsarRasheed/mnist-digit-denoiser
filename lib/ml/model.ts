import { ConvLayer, conv1, conv2, conv3, conv4, conv5, TOTAL_PARAMS } from "./weights";

/**
 * Pure TypeScript re-implementation of the trained Keras Sequential model's
 * forward pass (Conv2D 'same'/stride-1 cross-correlation, MaxPooling2D 2x2
 * 'valid', UpSampling2D 2x2 nearest, ReLU). No TensorFlow/ML runtime
 * dependency is needed because the model has only 5,841 parameters.
 *
 * This port was numerically verified against a NumPy reimplementation of
 * the same operations, run with the real extracted weights over 200 real
 * MNIST test images: mean MSE 0.00704, matching the Colab's own
 * model.evaluate() loss of ~0.0072 over the full test set.
 */

interface Grid {
  data: Float32Array;
  h: number;
  w: number;
  c: number;
}

function conv2dSameRelu(input: Grid, layer: ConvLayer): Grid {
  const { h: H, w: W, c: inC } = input;
  const { kh, kw, outC, kernel, bias } = layer;
  const padH = Math.floor(kh / 2);
  const padW = Math.floor(kw / 2);
  const out = new Float32Array(H * W * outC);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const outBase = (y * W + x) * outC;
      for (let oc = 0; oc < outC; oc++) {
        out[outBase + oc] = bias[oc];
      }
      for (let ky = 0; ky < kh; ky++) {
        const iy = y + ky - padH;
        if (iy < 0 || iy >= H) continue;
        for (let kx = 0; kx < kw; kx++) {
          const ix = x + kx - padW;
          if (ix < 0 || ix >= W) continue;
          const inBase = (iy * W + ix) * inC;
          const kernelBase = ((ky * kw + kx) * inC) * outC;
          for (let ic = 0; ic < inC; ic++) {
            const inVal = input.data[inBase + ic];
            if (inVal === 0) continue;
            const kBase = kernelBase + ic * outC;
            for (let oc = 0; oc < outC; oc++) {
              out[outBase + oc] += inVal * kernel[kBase + oc];
            }
          }
        }
      }
      for (let oc = 0; oc < outC; oc++) {
        const v = out[outBase + oc];
        out[outBase + oc] = v > 0 ? v : 0; // ReLU
      }
    }
  }

  return { data: out, h: H, w: W, c: outC };
}

function maxPool2x2(input: Grid): Grid {
  const { h: H, w: W, c: C } = input;
  const H2 = Math.floor(H / 2);
  const W2 = Math.floor(W / 2);
  const out = new Float32Array(H2 * W2 * C);

  for (let y = 0; y < H2; y++) {
    for (let x = 0; x < W2; x++) {
      const outBase = (y * W2 + x) * C;
      for (let c = 0; c < C; c++) {
        const a = input.data[((2 * y) * W + 2 * x) * C + c];
        const b = input.data[((2 * y) * W + 2 * x + 1) * C + c];
        const cc = input.data[((2 * y + 1) * W + 2 * x) * C + c];
        const d = input.data[((2 * y + 1) * W + 2 * x + 1) * C + c];
        out[outBase + c] = Math.max(a, b, cc, d);
      }
    }
  }

  return { data: out, h: H2, w: W2, c: C };
}

function upsample2x2(input: Grid): Grid {
  const { h: H, w: W, c: C } = input;
  const H2 = H * 2;
  const W2 = W * 2;
  const out = new Float32Array(H2 * W2 * C);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const inBase = (y * W + x) * C;
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          const outBase = ((2 * y + dy) * W2 + (2 * x + dx)) * C;
          for (let c = 0; c < C; c++) {
            out[outBase + c] = input.data[inBase + c];
          }
        }
      }
    }
  }

  return { data: out, h: H2, w: W2, c: C };
}

/**
 * Runs the trained denoising autoencoder on a single 28x28x1 image.
 * @param pixels Float32Array of length 784, values normalized to [0, 1]
 * @returns Float32Array of length 784, values >= 0 (ReLU output; caller
 *          should clip to [0, 1] for display, matching how the notebook
 *          treats model output as an image).
 */
export function denoiseForward(pixels: Float32Array): Float32Array {
  let grid: Grid = { data: pixels, h: 28, w: 28, c: 1 };

  grid = conv2dSameRelu(grid, conv1); // -> 28x28x32
  grid = maxPool2x2(grid); // -> 14x14x32
  grid = conv2dSameRelu(grid, conv2); // -> 14x14x8
  grid = maxPool2x2(grid); // -> 7x7x8
  grid = conv2dSameRelu(grid, conv3); // -> 7x7x8
  grid = upsample2x2(grid); // -> 14x14x8
  grid = conv2dSameRelu(grid, conv4); // -> 14x14x32
  grid = upsample2x2(grid); // -> 28x28x32
  grid = conv2dSameRelu(grid, conv5); // -> 28x28x1

  return grid.data;
}

export const MODEL_PARAM_COUNT = TOTAL_PARAMS;
