import { NextResponse } from "next/server";
import { denoiseForward, MODEL_PARAM_COUNT } from "@/lib/ml/model";
import { corrupt, clampSigma, TRAINING_SIGMA, NOISE_MEAN } from "@/lib/ml/noise";
import { toFloat01, to255 } from "@/lib/ml/pixels";
import { meanSquaredError, peakSignalToNoiseRatio } from "@/lib/ml/metrics";
import { getPreset } from "@/lib/ml/presets";
import type { PredictRequestBody, PredictResponseBody, InputSource } from "@/lib/ml/types";

export const runtime = "nodejs";

const PIXEL_COUNT = 28 * 28;
const VALID_SOURCES: InputSource[] = ["draw", "upload", "preset"];

function badRequest(message: string) {
  return NextResponse.json({ error: message }, { status: 400 });
}

export async function POST(request: Request) {
  let body: PredictRequestBody;
  try {
    body = await request.json();
  } catch {
    return badRequest("Request body must be valid JSON.");
  }

  const { source, presetDigit } = body;

  if (!source || !VALID_SOURCES.includes(source)) {
    return badRequest("`source` must be one of: draw, upload, preset.");
  }

  let cleanPixels: number[];
  // Presets always use the fixed training sigma, regardless of what the
  // client sends — they exist as a controlled, comparable reference across
  // all ten digits. Draw/Upload may override it within [MIN_SIGMA, MAX_SIGMA].
  let noiseSigma = TRAINING_SIGMA;

  if (source === "preset") {
    if (typeof presetDigit !== "number" || presetDigit < 0 || presetDigit > 9) {
      return badRequest("`presetDigit` must be an integer between 0 and 9.");
    }
    // Authoritative lookup — never trust client-supplied pixels for presets,
    // so preset inference always runs on the real MNIST sample.
    const preset = getPreset(presetDigit);
    if (!preset) {
      return badRequest("Unknown preset digit.");
    }
    cleanPixels = preset.pixels;
  } else {
    if (typeof body.noiseSigma === "number" && Number.isFinite(body.noiseSigma)) {
      noiseSigma = clampSigma(body.noiseSigma);
    }
    if (!Array.isArray(body.pixels) || body.pixels.length !== PIXEL_COUNT) {
      return badRequest(`\`pixels\` must be an array of ${PIXEL_COUNT} numbers.`);
    }
    if (body.pixels.some((v) => typeof v !== "number" || Number.isNaN(v))) {
      return badRequest("`pixels` must contain only finite numbers.");
    }
    cleanPixels = body.pixels;
  }

  try {
    const clean01 = toFloat01(cleanPixels);
    const corrupted01 = corrupt(clean01, noiseSigma);

    const start = performance.now();
    const reconstructedRaw = denoiseForward(corrupted01);
    const inferenceTimeMs = performance.now() - start;

    const mse = meanSquaredError(reconstructedRaw, clean01);
    const psnr = peakSignalToNoiseRatio(mse);

    const response: PredictResponseBody = {
      original: to255(clean01),
      corrupted: to255(corrupted01),
      reconstructed: to255(reconstructedRaw),
      metrics: {
        mse,
        psnr,
        noiseSigma,
        noiseMean: NOISE_MEAN,
        inputDims: "28x28x1",
        inferenceTimeMs,
        modelStatus: "ready",
        paramCount: MODEL_PARAM_COUNT,
      },
    };

    return NextResponse.json(response);
  } catch (err) {
    console.error("Inference failed:", err);
    return NextResponse.json(
      { error: "Inference failed. Please try again." },
      { status: 500 }
    );
  }
}
