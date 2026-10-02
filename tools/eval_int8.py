"""
Measure the accuracy cost of INT8 quantisation before writing any C kernel.

Evaluates the JAX model three ways on the same fixtures:
  1. float32 weights (ground truth)
  2. per-tensor symmetric INT8 weights, dequantised back to float32
  3. per-channel symmetric INT8 weights (upper bound on what per-tensor gives up)

The network is run in float32 throughout; only the weights are quantised. That
isolates quantisation error from any activation-quantisation error, which is
the number that decides whether per-tensor INT8 is viable at all.

Usage:
    python tools/eval_int8.py
"""

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "models"))

from models import get_model                                   # noqa: E402
from features import AudioFeatureExtractor                     # noqa: E402

CKPT = os.path.join(ROOT, "models", "best_v11_production.flax")
ARCH = "bcconformer_v3"
T, MEL = 49, 40


def load():
    model = get_model(ARCH, num_classes=2)
    v = model.init(jax.random.PRNGKey(0), jnp.ones((1, T, MEL, 1), jnp.float32),
                   train=False)
    tmpl = {"params": v["params"], "arch": ARCH, "param_count": 0,
            "val_acc": 0.0, "tpr": 0.0, "fpr": 0.0, "threshold": 0.85}
    raw = open(CKPT, "rb").read()
    try:
        st = serialization.from_bytes(tmpl, raw)
    except ValueError:
        st = serialization.msgpack_restore(raw)
    return model, jax.tree_util.tree_map(jnp.asarray, st["params"])


def flatten(tree):
    out = {}

    def walk(prefix, node):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(f"{prefix}/{k}" if prefix else k, v)
        else:
            out[prefix] = np.asarray(node)

    walk("", tree)
    return out


def quantise_tensor(arr):
    """Symmetric INT8, one scale for the whole tensor. Returns (q, scale)."""
    amax = float(np.abs(arr).max()) if arr.size else 0.0
    scale = (amax / 127.0) if amax > 0 else 1.0
    q = np.clip(np.rint(arr / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantise_tree(flat, per_tensor=True, per_channel=False):
    """Rebuild a params tree from quantised-then-dequantised tensors."""
    out = {}
    for name, arr in flat.items():
        if per_channel and arr.ndim >= 2:
            # one scale per output channel = last axis for Dense/Conv kernels
            axes = tuple(range(arr.ndim - 1))
            amax = np.max(np.abs(arr), axis=axes, keepdims=True)
            scale = np.where(amax > 0, amax / 127.0, 1.0)
            q = np.clip(np.rint(arr / scale), -127, 127)
            deq = q * scale
        else:
            _, scale = quantise_tensor(arr)
            _, _, deq = None, None, (np.rint(arr / scale) * scale)
        node = out
        parts = name.split("/")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = jnp.asarray(deq.astype(np.float32))
    return out


def prob(model, params, spec):
    x = jnp.asarray(spec)[None, ..., None]
    logits = model.apply({"params": params}, x, train=False)
    return float(jax.nn.softmax(logits, axis=-1)[0, 1])


def main():
    model, params = load()
    flat = flatten(params)
    ex = AudioFeatureExtractor()

    fixtures = []
    rng = np.random.default_rng(0)
    fixtures.append(("noise", (rng.normal(0, 0.05, 16000) * 32768).astype(np.float32)))
    # A 1 kHz tone and a swept tone: structured signals probe the mel filterbank
    # and the convolutions differently from white noise.
    t = np.arange(16000, dtype=np.float32)
    fixtures.append(("tone1k", (8000 * np.sin(2 * np.pi * 1000 * t / 16000)).astype(np.float32)))
    sweep = 8000 * np.sin(2 * np.pi * (300 + 2000 * t / 16000) * t / 16000)
    fixtures.append(("sweep", sweep.astype(np.float32)))

    print(f"{'fixture':<10}{'float32':>12}{'int8/tensor':>13}{'int8/chan':>12}"
          f"{'err tensor':>12}{'err chan':>11}")
    print("-" * 70)

    for name, wav in fixtures:
        spec = ex.compute_spectrogram(wav)
        p_f32 = prob(model, params, spec)
        p_t = prob(model, dequantise_tree(flat, per_tensor=True), spec)
        p_c = prob(model, dequantise_tree(flat, per_channel=True), spec)
        print(f"{name:<10}{p_f32:>12.6f}{p_t:>13.6f}{p_c:>12.6f}"
              f"{abs(p_t - p_f32):>12.6f}{abs(p_c - p_f32):>11.6f}")

    thr = 0.68
    print()
    print(f"detection threshold in the checkpoint: {thr}")
    print("Fixtures above are all negatives, so a correct INT8 port must keep")
    print("them well below threshold. Fixtures near or above it need a real")
    print("keyword sample to judge, which this repo does not contain.")
    print()
    print("weights: float32 %d B  ->  int8 %d B (%.0f%% smaller)"
          % (sum(v.size for v in flat.values()) * 4,
             sum(v.size for v in flat.values()),
             100 * (1 - 0.25)))


if __name__ == "__main__":
    main()