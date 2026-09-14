import raw from "./data/model-weights.json";

/**
 * Weights extracted directly from the trained Keras model `denoising.h5`
 * (see /scripts/extract-model-weights.py). Architecture, layer names and
 * shapes below mirror the model exactly as verified against model_config
 * embedded in the .h5 file:
 *
 *   Conv2D(32, same, relu) -> MaxPool2D(2x2)
 *   Conv2D(8,  same, relu) -> MaxPool2D(2x2)
 *   Conv2D(8,  same, relu) -> UpSampling2D(2x2)
 *   Conv2D(32, same, relu) -> UpSampling2D(2x2)
 *   Conv2D(1,  same, relu)
 */

export interface ConvLayer {
  kernel: Float32Array; // flattened (kh, kw, inC, outC), Keras storage order
  bias: Float32Array;
  kh: number;
  kw: number;
  inC: number;
  outC: number;
}

type RawLayer = { kernel_shape: number[]; kernel: number[]; bias: number[] };
type RawWeights = Record<string, RawLayer>;

function toLayer(raw: RawLayer): ConvLayer {
  const [kh, kw, inC, outC] = raw.kernel_shape;
  return {
    kernel: Float32Array.from(raw.kernel),
    bias: Float32Array.from(raw.bias),
    kh,
    kw,
    inC,
    outC,
  };
}

const data = raw as RawWeights;

export const conv1 = toLayer(data["conv2d_11"]); // 1 -> 32
export const conv2 = toLayer(data["conv2d_12"]); // 32 -> 8
export const conv3 = toLayer(data["conv2d_13"]); // 8 -> 8
export const conv4 = toLayer(data["conv2d_14"]); // 8 -> 32
export const conv5 = toLayer(data["conv2d_15"]); // 32 -> 1

export const TOTAL_PARAMS = [conv1, conv2, conv3, conv4, conv5].reduce(
  (sum, layer) => sum + layer.kernel.length + layer.bias.length,
  0
);
