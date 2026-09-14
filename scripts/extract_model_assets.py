"""
One-time extraction script: converts the trained Keras model (denoising.h5,
produced by AutoEncoders_Task_Submission.ipynb) and the MNIST test set into
the static JSON assets consumed by the Next.js app's TypeScript inference
engine (lib/ml/model.ts, lib/ml/weights.ts, lib/ml/presets.ts).

Why this exists: the web app deliberately has no TensorFlow/Keras runtime
dependency (the trained model has only 5,841 parameters, so a pure
TypeScript port of its forward pass is simpler and far lighter for
serverless deployment than bundling TensorFlow). This script is the one
place that reads the original .h5 file and the original MNIST source, so
the exact trained weights and exact real MNIST samples are re-derived from
the source of truth rather than hand-copied.

Usage (run once, from the project root, whenever denoising.h5 changes):
    pip install -r scripts/requirements.txt
    python scripts/extract_model_assets.py

Outputs:
    lib/ml/data/model-weights.json   (~124 KB) - all 5 Conv2D kernels/biases
    lib/ml/data/presets.json         (~26 KB)  - one real test-set sample per digit 0-9

Verification: a NumPy re-implementation of the same Conv2D('same')/
MaxPooling2D(2x2,'valid')/UpSampling2D(2x2,'nearest')/ReLU forward pass used
by lib/ml/model.ts is run here against 200 real, freshly-noised MNIST test
images. Its mean MSE should land close to the notebook's own
model.evaluate(X_test_clipped, X_test) loss (~0.0072 over the full test
set) as a sanity check that the extracted weights and the ported forward
pass faithfully reproduce the original model.
"""

import json
import os

import h5py
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5_PATH = os.path.join(BASE, "denoising.h5")
OUT_DIR = os.path.join(BASE, "lib", "ml", "data")

# Same public source the training Colab uses via keras.datasets.mnist.load_data()
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

CONV_LAYERS = ["conv2d_11", "conv2d_12", "conv2d_13", "conv2d_14", "conv2d_15"]
NOISE_SIGMA = 0.3  # from the notebook: X + 0.3 * np.random.normal(0, 1, size=X.shape)


def extract_weights():
    with h5py.File(H5_PATH, "r") as f:
        root = f["model_weights"]
        seq_name = next(iter(root[CONV_LAYERS[0]].keys()))  # e.g. "sequential_7"
        weights = {}
        for name in CONV_LAYERS:
            grp = root[name][seq_name][name]
            weights[name] = {
                "kernel": grp["kernel"][:],
                "bias": grp["bias"][:],
            }
    return weights


def conv2d_same(x, kernel, bias):
    H, W, Cin = x.shape
    kh, kw, kCin, Cout = kernel.shape
    assert kCin == Cin
    pad_h, pad_w = kh // 2, kw // 2
    xp = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    out = np.zeros((H, W, Cout), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            patch = xp[i : i + H, j : j + W, :]
            out += np.tensordot(patch, kernel[i, j], axes=([2], [0]))
    return out + bias


def relu(x):
    return np.maximum(x, 0)


def maxpool2x2(x):
    H, W, C = x.shape
    H2, W2 = H // 2, W // 2
    x = x[: H2 * 2, : W2 * 2, :].reshape(H2, 2, W2, 2, C)
    return x.max(axis=(1, 3))


def upsample2x2(x):
    return np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)


def forward(x, weights):
    for name in CONV_LAYERS[:2]:
        x = relu(conv2d_same(x, weights[name]["kernel"], weights[name]["bias"]))
        x = maxpool2x2(x)
    x = relu(conv2d_same(x, weights[CONV_LAYERS[2]]["kernel"], weights[CONV_LAYERS[2]]["bias"]))
    for name in CONV_LAYERS[3:]:
        x = upsample2x2(x)
        x = relu(conv2d_same(x, weights[name]["kernel"], weights[name]["bias"]))
    return x


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    weights = extract_weights()

    weights_export = {
        name: {
            "kernel_shape": list(weights[name]["kernel"].shape),
            "kernel": weights[name]["kernel"].astype(np.float32).flatten().tolist(),
            "bias": weights[name]["bias"].astype(np.float32).flatten().tolist(),
        }
        for name in CONV_LAYERS
    }
    with open(os.path.join(OUT_DIR, "model-weights.json"), "w") as f:
        json.dump(weights_export, f)
    print("Wrote lib/ml/data/model-weights.json")

    # Download the exact same MNIST source the Colab uses, then verify + build presets.
    import urllib.request

    npz_path = os.path.join(BASE, ".mnist_cache.npz")
    if not os.path.exists(npz_path):
        urllib.request.urlretrieve(MNIST_URL, npz_path)
    with np.load(npz_path) as mnist:
        X_test, y_test = mnist["x_test"].copy(), mnist["y_test"].copy()
    X_test_f = X_test.astype(np.float32).reshape(-1, 28, 28, 1) / 255.0

    rng = np.random.RandomState(42)
    N = 200
    clean = X_test_f[:N]
    noisy_clipped = np.clip(clean + NOISE_SIGMA * rng.normal(size=clean.shape), 0.0, 1.0)
    mses = [np.mean((forward(noisy_clipped[i], weights) - clean[i]) ** 2) for i in range(N)]
    print(f"Verification: mean MSE over {N} real test images = {np.mean(mses):.5f}")
    print("(Notebook's own model.evaluate loss over the full test set was ~0.0072)")

    indices = {str(d): int(np.where(y_test == d)[0][0]) for d in range(10)}
    presets = {
        "sourceDataset": "MNIST test set (keras.datasets.mnist, same source as training Colab)",
        "indices": indices,
        "digits": {d: X_test[i].astype(np.uint8).flatten().tolist() for d, i in indices.items()},
    }
    with open(os.path.join(OUT_DIR, "presets.json"), "w") as f:
        json.dump(presets, f)
    print("Wrote lib/ml/data/presets.json")
    os.remove(npz_path)


if __name__ == "__main__":
    main()
