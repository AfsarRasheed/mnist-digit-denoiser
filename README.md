# MNIST Denoising Autoencoder

A production web app around a real, trained convolutional autoencoder: draw a
digit, upload a photo of one, or pick a real MNIST sample, and watch the
model remove Gaussian noise from it live. Every prediction is genuine
inference against the weights trained in the original Colab notebook — there
are no mocked images, no hardcoded metrics, and no placeholder digits.

## Project Overview

- **Draw Digit** — freehand canvas input (mouse + touch).
- **Upload Image** — drag-and-drop, file picker, or clipboard paste (PNG/JPG/WEBP).
- **MNIST Presets** — one real, deterministic sample per digit (0–9) from the
  actual MNIST test set.

Whichever input method is used, the image is normalized into the model's
28×28 grayscale format, corrupted with the same Gaussian noise recipe used
during training, denoised by the real autoencoder, and displayed alongside
real metrics (reconstruction MSE, PSNR, inference time, etc.) computed from
that actual run — not simulated.

## Architecture

```
Frontend (React / Next.js App Router)
   │  POST /api/predict  { source, pixels | presetDigit }
   ▼
Route Handler (app/api/predict/route.ts, Node.js runtime)
   │
   ├─ Input validation (shape, range, known preset)
   ├─ Preset lookup is authoritative server-side (client pixels are never
   │  trusted for presets — this guarantees preset inference always runs on
   │  the real MNIST sample, not on anything a client could tamper with)
   ├─ Noise generation   → lib/ml/noise.ts
   ├─ Autoencoder forward pass → lib/ml/model.ts (+ lib/ml/weights.ts)
   ├─ Metrics (MSE, PSNR, timing) → lib/ml/metrics.ts
   ▼
Structured JSON response { original, corrupted, reconstructed, metrics }
   ▼
Frontend renders all three 28×28 frames on <canvas> and the metrics panel
```

All three input methods funnel into the exact same `/api/predict` pipeline —
noise generation, model inference, and metrics are never duplicated per
input method. The only thing that legitimately differs per source is turning
raw input (a stroke, a photo, a preset id) into a canonical 28×28 grayscale
frame; that adapter lives in `lib/preprocessing/mnistAdapter.ts` and is
shared between Draw and Upload.

**Why no separate ML backend/service:** the trained model has only 5,841
parameters (~23 KB of weights). Rather than bundling TensorFlow/Keras into a
serverless function (large cold starts, native binary issues on
Vercel/Netlify), the model's forward pass — five `Conv2D` layers,
`MaxPooling2D`, `UpSampling2D`, `ReLU` — is re-implemented directly in
TypeScript (`lib/ml/model.ts`) using the real extracted weights. This keeps
the entire app as one deployable Next.js project with zero ML runtime
dependencies, while still running the *actual* trained model, not an
approximation of it. See `scripts/extract_model_assets.py` for how the
weights were extracted and numerically verified against the original.

## Model

Source of truth: `AutoEncoders_Task_Submission.ipynb` (Colab) and its saved
output, `denoising.h5`. Architecture and weights were read directly from the
`.h5` file's embedded Keras config (see `scripts/extract_model_assets.py`),
not re-derived by hand:

| Layer | Output shape | Notes |
|---|---|---|
| Input | 28×28×1 | grayscale, normalized to [0, 1] |
| Conv2D(32, 3×3, same, ReLU) | 28×28×32 | |
| MaxPooling2D(2×2) | 14×14×32 | |
| Conv2D(8, 3×3, same, ReLU) | 14×14×8 | |
| MaxPooling2D(2×2) | 7×7×8 | bottleneck |
| Conv2D(8, 3×3, same, ReLU) | 7×7×8 | |
| UpSampling2D(2×2) | 14×14×8 | |
| Conv2D(32, 3×3, same, ReLU) | 14×14×32 | |
| UpSampling2D(2×2) | 28×28×32 | |
| Conv2D(1, 3×3, same, ReLU) | 28×28×1 | output |

- **Framework (training):** TensorFlow/Keras 3
- **Loss:** mean squared error · **Optimizer:** Adam
- **Params:** 5,841
- **Training data:** `keras.datasets.mnist`, reshaped to `(N, 28, 28, 1)`,
  normalized by `/255.0`
- **Training target:** `model.fit(x=X_train_noisy_clipped, y=X_train_clean, ...)`
  — the model is trained to map a noisy image back to the clean original,
  i.e. it's a **denoising** autoencoder, not a plain reconstruction one.

The TypeScript port (`lib/ml/model.ts`) implements the same five
operations — `Conv2D` as 'same'-padding, stride-1 cross-correlation (Keras
does not flip the kernel), `MaxPooling2D` as 2×2/stride-2/'valid',
`UpSampling2D` as 2×2 nearest-neighbor — using the exact weights extracted
from `denoising.h5`. It was verified against a NumPy re-implementation of
the identical operations, run on 200 real MNIST test images: mean MSE
0.00704, matching the notebook's own `model.evaluate(X_test_clipped, X_test)`
loss of ≈0.0072 over the full 10,000-image test set.

## Preprocessing

The Colab never had to build a "raw image → MNIST tensor" adapter — it only
ever consumed the pre-built MNIST arrays. Free-form input (a drawn stroke, an
uploaded photo) needs one, so `lib/preprocessing/mnistAdapter.ts` reproduces
the standard MNIST construction recipe rather than inventing a new one:

1. **Grayscale** — uploaded images are converted using standard luminance
   weights (`0.299R + 0.587G + 0.114B`).
2. **Auto-invert** — if the image's mean brightness is high (photo of dark
   ink on light paper), it's inverted so the result matches MNIST's
   convention: a light stroke on a dark background. Drawings are already in
   that convention (white pen on a black canvas), so this step is a no-op
   for the Draw tab.
3. **Bounding-box crop** — the "ink" region (pixels above a small threshold)
   is located and cropped with a small margin.
4. **Scale to a 20px core** — the crop is resized so its longest side is
   20px, mirroring the digit size used when MNIST itself was built.
5. **Center by center of mass** — the 20px digit is placed on a 28×28 black
   canvas and shifted so its pixel-intensity center of mass lands at (13.5,
   13.5), the same centering method used to construct MNIST.
6. **Normalize** — pixel values scaled to `[0, 1]` before being handed to the
   model.

MNIST presets skip straight to step 6 — they're already real, correctly
formatted MNIST samples, so there's nothing to adapt.

## Noise / Denoising

Corruption is reproduced exactly from the notebook (`lib/ml/noise.ts`):

```python
X_noise   = X + 0.3 * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
X_clipped = np.clip(X_noise, 0.0, 1.0)
```

i.e. zero-mean Gaussian noise with **σ = 0.3**, added per-pixel to the
`[0, 1]`-normalized image, then clipped back into `[0, 1]`. σ = 0.3 is the
exact corruption level the autoencoder was trained on, and it's always what
MNIST Presets use, so the ten reference digits stay directly comparable to
one another. Draw and Upload additionally expose a **noise level slider**
(`components/NoiseSlider.tsx`, clamped server-side to `[0.05, 0.8]` in
`lib/ml/noise.ts`) so a user can see how the same trained model responds to
lighter or heavier corruption than it saw during training — this is the one
place the app intentionally departs from the fixed training value, since
the point of Draw/Upload is hands-on exploration rather than a fixed
reference. A fresh noise sample is drawn on every request (matching the
stochastic corruption used per-epoch during training), so reprocessing the
same input twice will look slightly different even at the same σ — that's
expected, not a bug.

**Live noise preview.** Dragging the slider updates the "Model input"
preview immediately, in the browser — using the *same* `corrupt()` function
the server calls, not a separate approximation (`lib/preprocessing/livePreview.ts`
imports it directly). This is a preview only: it never touches the network,
and the authoritative corrupted image the model actually sees is still
generated server-side, with a fresh random draw, when Process is clicked.

**Reconstruction MSE** is computed the same way as the notebook's own
`model.evaluate(X_test_clipped, X_test)`: mean squared error between the
model's output and the **original clean image** (not the noisy input),
since that comparison is what the notebook itself uses to judge denoising
quality.

**Inspecting a result.** Clicking any of the three result images opens it
enlarged; for the reconstructed image, this view also offers a **Difference**
mode — the real pixel-wise `|reconstructed − original|`, contrast-stretched
for visibility with the true (pre-stretch) max delta disclosed in the
caption, not a fabricated heatmap. Either view can be downloaded as a PNG.

## MNIST Presets

The ten presets are real samples pulled from the official MNIST **test**
set — the same dataset source the training Colab loads via
`keras.datasets.mnist.load_data()` (downloaded from
`storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz`). For each
digit, the first test-set example with that label is selected, so the set is
fixed and reproducible across builds and sessions (see
`scripts/extract_model_assets.py`). The 10 pixel arrays are baked into
`lib/ml/data/presets.json` (~26 KB) at build time — no dataset download
happens per visitor, and no artificial or generated digit images are used
anywhere.

## Local Development

```bash
npm install
npm run dev
```

Open http://localhost:3000. No environment variables or external services
are required for local development.

```bash
npm run build   # production build (also type-checks)
npm run start   # serve the production build
npm run lint    # ESLint
```

To regenerate `lib/ml/data/model-weights.json` and `presets.json` from
scratch (only needed if `denoising.h5` changes):

```bash
pip install -r scripts/requirements.txt
python scripts/extract_model_assets.py
```

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | No | Only set this if `/api/predict` is deployed separately from the frontend (see "ML Backend" below). Leave unset for the default, single-deployment setup — the frontend calls its own same-origin `/api/predict`. |

See `.env.example`. No secrets, API keys, or database credentials are
needed anywhere in this project.

## Vercel Deployment

This is a standard Next.js App Router project — no custom `vercel.json` is
needed.

**Live**: https://mnist-digit-denoiser.vercel.app

1. Push the repository to GitHub/GitLab/Bitbucket.
2. In Vercel, "Add New Project" → import the repository. The Next.js
   preset is auto-detected.
3. Leave build/output settings at their defaults (`next build`).
4. No environment variables are required for the default architecture.
5. Deploy. `/api/predict` becomes a Vercel serverless function automatically.

This project's Vercel deployment is connected directly to the
`AfsarRasheed/mnist-digit-denoiser` GitHub repository, so every push to
`main` triggers a new production deployment automatically.

## Netlify Deployment

A `netlify.toml` is included, enabling the official Next.js Runtime plugin:

```toml
[build]
  command = "npm run build"
  publish = ".next"

[[plugins]]
  package = "@netlify/plugin-nextjs"
```

1. Push the repository, then "Add new site" → "Import an existing project"
   in Netlify.
2. Netlify detects `netlify.toml` and installs `@netlify/plugin-nextjs`
   automatically; no manual configuration is required.
3. Deploy. `/api/predict` runs as a Netlify Function.

## ML Backend

The trained model is small enough (5,841 params, ~23 KB) to run inline,
in-process, inside the same serverless function that serves the page — so
**no separate ML hosting is required** for this project. There's no
Python/TensorFlow runtime anywhere in the deployed app; `lib/ml/model.ts` is
a dependency-free TypeScript port of the trained model's forward pass, using
its real extracted weights (see "Model" above).

If a future model trained in the same Colab is too large or complex for
this approach (e.g. it needs GPU inference or a heavy Python-only
dependency), the API boundary is already isolated behind
`POST /api/predict` with a stable JSON contract
(`lib/ml/types.ts`). In that case:

- Stand up a dedicated inference service (e.g. a small FastAPI/Flask app
  behind Cloud Run, a GPU box, or a managed endpoint) exposing the same
  request/response shape.
- Set `NEXT_PUBLIC_API_BASE_URL` to that service's URL — the frontend
  already reads this variable (`lib/config.ts`) and needs no code changes.
- Add CORS headers on that external service for the deployed frontend's
  origin (not needed today, since everything is same-origin by default).

## Project Structure

```
app/
  page.tsx                 → renders <Studio />
  api/predict/route.ts     → the inference endpoint
components/                → Studio, DrawCanvas, UploadPanel, PresetGrid,
                              PipelineFlow, ModelNode, GhostPipeline,
                              MetricsPanel, ResultCanvas, ...
hooks/
  usePredict.ts            → fetch wrapper (loading/error/rerun/duplicate-request guard)
  useProcessingPhase.ts     → cycling status text while a request is in flight
lib/
  ml/
    model.ts               → forward pass (Conv2D/MaxPool/UpSample/ReLU)
    weights.ts             → typed weight loader
    noise.ts                → Gaussian corruption (σ = 0.3, shared by server + live preview)
    pixels.ts               → 0-255 <-> [0,1] conversion, shared by server + live preview
    metrics.ts              → MSE / PSNR / pixel-wise difference
    presets.ts              → typed preset loader
    types.ts                → shared request/response types
    data/                   → model-weights.json, presets.json (generated)
  preprocessing/
    mnistAdapter.ts          → draw/upload → 28×28 MNIST frame adapter
    livePreview.ts           → client-side corruption preview (same corrupt() as the server)
  config.ts                → API base URL resolution
scripts/
  extract_model_assets.py  → one-time .h5 → JSON extraction + verification
denoising.h5                → original trained model (source of truth)
AutoEncoders_Task_Submission.ipynb → original training Colab
```

## Known Limitations

- The MNIST-adapter preprocessing for Draw/Upload (bounding-box crop +
  center-of-mass centering) is a well-established approximation of how
  MNIST itself was built, not a component of the original Colab — the
  notebook never needed one, since it only ever consumed ready-made MNIST
  arrays. Unusual strokes or photos (multiple digits, very thick/thin
  pens, heavy backgrounds) can still be adapted imperfectly.
- The upload auto-invert heuristic (mean brightness > 127 ⇒ invert) is a
  simple global heuristic; photos with uneven lighting may occasionally
  need a cleaner, higher-contrast source image.
- Reconstruction quality is inherent to the trained model (5,841 params,
  5 epochs) — it favors a smoothed, mildly blurry reconstruction, which is
  a property of the original training run, not of this web app.
