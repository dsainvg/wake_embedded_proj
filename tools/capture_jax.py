"""
Ground-truth stage values, captured from the real Flax model itself.

tools/dump_stages.py re-implements BCConformerV3 in numpy/jax by hand, which is
how this project managed to have a reference that disagreed with the model it
was supposed to describe. This script does not re-implement anything: it wraps
the live flax modules, runs model.apply once, and records what each one actually
computed. Whatever it prints is what the checkpoint does.

Usage:
    python tools/capture_jax.py noise
"""

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "models"))

from models import get_model  # noqa: E402
from features import AudioFeatureExtractor  # noqa: E402

CKPT = os.path.join(ROOT, "models", "best_v11_production.flax")
ARCH = "bcconformer_v3"
T, MEL = 49, 40

RECORDS = []


def _record(kind, name, x, y):
    RECORDS.append((kind, name, np.asarray(x), np.asarray(y)))


def group_stats(a, n_groups=4):
    """Per-group mean/std of a channel-last tensor, the way Flax GroupNorm
    computes them (statistics per group, over every other axis)."""
    a = np.asarray(a, np.float64)
    a = a.reshape(-1, a.shape[-1])
    c = a.shape[-1]
    per = c // n_groups
    out = []
    for g in range(n_groups):
        v = a[:, g * per:(g + 1) * per]
        out.append((float(v.mean()), float(v.std())))
    return out


def _wrap(cls, kind):
    orig = cls.__call__

    def patched(self, *args, **kwargs):
        y = orig(self, *args, **kwargs)
        name = "/".join(str(p) for p in self.path)
        _record(kind, name, args[0], y)
        return y

    cls.__call__ = patched


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "noise"

    for cls, kind in ((nn.GroupNorm, "GroupNorm"),
                      (nn.LayerNorm, "LayerNorm"),
                      (nn.Conv, "Conv"),
                      (nn.Dense, "Dense")):
        _wrap(cls, kind)

    model = get_model(ARCH, num_classes=2)
    dummy = jnp.ones((1, T, MEL, 1), dtype=jnp.float32)
    v = model.init(jax.random.PRNGKey(0), dummy, train=False)
    tmpl = {"params": v["params"], "arch": ARCH, "param_count": 0,
            "val_acc": 0.0, "tpr": 0.0, "fpr": 0.0, "threshold": 0.85}
    raw = open(CKPT, "rb").read()
    try:
        st = serialization.from_bytes(tmpl, raw)
    except ValueError:
        st = serialization.msgpack_restore(raw)
    params = jax.tree_util.tree_map(jnp.asarray, st["params"])

    ex = AudioFeatureExtractor()
    spec = ex.compute_spectrogram(make_wave(tag))
    x = jnp.asarray(spec)[None, ..., None]

    RECORDS.clear()
    logits = model.apply({"params": params}, x, train=False)
    probs = jax.nn.softmax(logits, axis=-1)

    print(f"{'#':>3} {'kind':<10} {'name':<28} {'in shape':<18} "
          f"{'out shape':<18} {'in rms':>10} {'out rms':>10} {'grp stats in':>34}")
    print("-" * 132)
    for i, (kind, name, xi, yo) in enumerate(RECORDS):
        gs = ""
        if kind == "GroupNorm":
            gs = " ".join(f"{m:+.3f}/{sd:.3f}" for m, sd in group_stats(xi))
        print(f"{i:>3} {kind:<10} {name[-28:]:<28} "
              f"{str(xi.shape):<18} {str(yo.shape):<18} "
              f"{rms(xi):>10.5f} {rms(yo):>10.5f} {gs:>34}")
    print("-" * 132)
    out_dir = os.path.join(ROOT, "main", "model", "reference", "stages")
    os.makedirs(out_dir, exist_ok=True)
    want = {0: "conv1", 2: "fconv", 4: "conv2", 5: "gn3", 6: "melgate"}
    for i, name in want.items():
        if i >= len(RECORDS):
            continue
        kind, mname, xi, yo = RECORDS[i]
        p = os.path.join(out_dir, f"jax_{tag}_{name}.npy")
        np.save(p, yo.reshape(yo.shape[1:]))
        print(f"saved {p}  from {kind} {mname} {yo.shape}")
    print(f"P(keyword) = {float(probs[0, 1]):.8f}")


def rms(a):
    a = np.asarray(a, np.float64)
    return float(np.sqrt((a ** 2).mean()))


def make_wave(name):
    t = np.arange(16000, dtype=np.float64)
    if name == "noise":
        rng = np.random.default_rng(0)
        return (rng.normal(0, 0.05, 16000) * 32768).astype(np.float32)
    if name == "tone1k":
        return (8000 * np.sin(2 * np.pi * 1000 * t / 16000)).astype(np.float32)
    if name == "sweep":
        return (8000 * np.sin(2 * np.pi * (300 + 2000 * t / 16000)
                              * t / 16000)).astype(np.float32)
    if name == "silence":
        return np.zeros(16000, np.float32)
    if name == "quiet":
        rng = np.random.default_rng(7)
        return (rng.normal(0, 0.002, 16000) * 32768).astype(np.float32)
    if name == "loud":
        rng = np.random.default_rng(11)
        return np.clip(rng.normal(0, 0.35, 16000) * 32768,
                       -32768, 32767).astype(np.float32)
    raise ValueError(name)


if __name__ == "__main__":
    main()