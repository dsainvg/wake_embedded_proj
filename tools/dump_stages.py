"""
Dump per-layer reference activations for the Amaze model so the C port can be
bisected against them.

Mirrors the capture points in main/kws_model.c (see kws_model.h for the stage
list). Writes one .npy per stage into main/model/reference/stages/.

Usage:
    python tools/dump_stages.py [--wav path.wav] [--tag name]
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "models"))

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization

import flax.linen as nn
from models import get_model, _delta_stack
from features import AudioFeatureExtractor

OUT = os.path.join(ROOT, "main", "model", "reference", "stages")
CKPT = os.path.join(ROOT, "models", "best_v11_production.flax")
ARCH = "bcconformer_v3"

T, MEL, DIM, C1, C2 = 49, 40, 48, 32, 48
STEM_W, C2_W = 20, 10


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
    p = jax.tree_util.tree_map(jnp.asarray, st["params"])
    return model, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=None)
    ap.add_argument("--tag", default="noise")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    model, params = load()
    ex = AudioFeatureExtractor()

    if args.wav:
        import soundfile as sf
        audio, sr = sf.read(args.wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert sr == 16000
        wav = audio[:16000]
        tag = os.path.splitext(os.path.basename(args.wav))[0]
    else:
        rng = np.random.default_rng(0)
        wav = (rng.normal(0, 0.05, 16000) * 32768).astype(np.float32)
        tag = args.tag

    np.save(os.path.join(OUT, f"{tag}_0_spec.npy"),
            ex.compute_spectrogram(wav))

    x = jnp.asarray(ex.compute_spectrogram(wav))[None, ..., None]

    def save(stage, name, arr):
        np.save(os.path.join(OUT, f"{tag}_{stage}_{name}.npy"),
                np.asarray(arr, dtype=np.float32))

    # ---- stem, reimplemented so each intermediate is observable ----------
    h = _delta_stack(x)
    h = _conv(h, params, "Conv_0")
    save(10, "conv1", np.asarray(h).reshape(T, STEM_W, C1))
    h = _gn(h, params["GroupNorm_0"]["scale"], params["GroupNorm_0"]["bias"])
    h = jax.nn.swish(h)
    save(11, "gn1swish", np.asarray(h).reshape(T, STEM_W, C1))

    f = _conv(h, params, "Conv_1")
    f = _gn(f, params["GroupNorm_1"]["scale"], params["GroupNorm_1"]["bias"])
    save(12, "fconv", np.asarray(f).reshape(T, STEM_W, C1))
    f = jax.nn.swish(f)
    save(13, "gn2swish", np.asarray(f).reshape(T, STEM_W, C1))
    h = h + f + jnp.mean(f, axis=2, keepdims=True)
    save(1, "stem", np.asarray(h).reshape(T, STEM_W, C1))

    h = _conv(h, params, "Conv_2")
    save(2, "conv2", np.asarray(h).reshape(T, C2_W, C2))

    h = _gn(h, params["GroupNorm_2"]["scale"], params["GroupNorm_2"]["bias"])
    h = jax.nn.swish(h)
    save(3, "gn3swish", np.asarray(h).reshape(T, C2_W, C2))

    # Conv_3 has kernel (1,48,1): a 1x1 over time that only projects 48 -> 1,
    # leaving (B,49,10,1). It is a channel projection, not a spatial conv, so
    # express it as a dot and restore the trailing axis.
    # Conv_3's kernel is (1,48,1): kernel_size (1,) over time, so it only projects
    # 48 -> 1 and leaves (B,49,10,1). Dropping the size-1 time axis and using
    # matmul reproduces exactly what the C does, which reads mg[c].
    mel_score = jnp.matmul(h, params["Conv_3"]["kernel"][0])     # (1,49,10,1)
    mel_w = jax.nn.softmax(mel_score, axis=2)
    seq = jnp.sum(h * mel_w, axis=2)
    save(4, "seq", np.asarray(seq))

    # ---- conformer stack --------------------------------------------------
    cur = seq
    for b in range(3):
        cur = _block(cur, params, b)
        save(5 + b, f"block{b + 1}", np.asarray(cur))

    feat = _pool(cur, params)
    save(8, "feat", np.asarray(feat))

    logits = _dense(feat, params["Dense_0"]["kernel"], params["Dense_0"]["bias"])
    probs = jax.nn.softmax(logits, axis=-1)
    save(9, "logits", np.asarray(logits))
    print(f"stages -> {OUT}   P(keyword)={float(probs[0, 1]):.8f}")
    print(f"tag={tag}")


# ---- helpers that apply already-initialised Flax modules ---------------
def _conv(x, params, name):
    """Apply a stored Flax Conv kernel.

    Flax picks the spatial rank from the KERNEL rank, not the module call:
      rank 4 -> 2-D (HWIO) over (T, mel), e.g. Conv_0 (3,3,3,32)
      rank 3 -> 1-D (WIO)  over T only,  e.g. Conv_3 (1,48,1)
    Convol_1's kernel (1,3,1,32) is rank 4 and grouped, so it is a depthwise 2-D
    conv over the mel axis.
    """
    k = params[name]["kernel"]
    b = params[name].get("bias")
    xj = jnp.asarray(x)

    if k.ndim == 4:
        strides = (1, 2) if name in ("Conv_0", "Conv_2") else (1, 1)
        groups = xj.shape[-1] if name == "Conv_1" else 1
        y = jax.lax.conv_general_dilated(
            xj, k, window_strides=strides, padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            rhs_dilation=(1, 1), feature_group_count=groups)
    elif k.ndim == 3:
        y = jax.lax.conv_general_dilated(
            xj, k, window_strides=(1,), padding="SAME",
            dimension_numbers=("NWC", "WIO", "NWC"),
            rhs_dilation=(1,), feature_group_count=1)
    else:
        raise ValueError(f"{name}: unsupported kernel rank {k.ndim}")

    return y if b is None else y + b


def _block(x, params, b):
    p = params[f"RelConformerBlock_{b}"]
    a = p["RelPosSelfAttention_0"]
    ln = nn.LayerNorm(name=f"RelConformerBlock_{b}_LayerNorm_0")  # unused
    # attention
    n = p["LayerNorm_0"]
    y = _ln(x, n["scale"], n["bias"])
    qkv = _dense(y, a["Dense_0"]["kernel"], a["Dense_0"]["bias"])
    q, k_, v = jnp.split(qkv, 3, axis=-1)
    bt, t, _ = q.shape
    hd = DIM // 4
    q = jnp.transpose(q.reshape(bt, t, 4, hd), (0, 2, 1, 3))
    k_ = jnp.transpose(k_.reshape(bt, t, 4, hd), (0, 2, 1, 3))
    v = jnp.transpose(v.reshape(bt, t, 4, hd), (0, 2, 1, 3))
    logits = jnp.einsum("bhid,bhjd->bhij", q, k_) / jnp.sqrt(jnp.asarray(hd, jnp.float32))
    rel = jnp.arange(t)[:, None] - jnp.arange(t)[None, :] + (t - 1)
    logits = logits + jnp.take(a["rel_bias"], rel, axis=0).transpose(2, 0, 1)[None]
    w = jax.nn.softmax(logits, axis=-1)
    o = jnp.einsum("bhij,bhjd->bhid", w, v)
    o = jnp.transpose(o, (0, 2, 1, 3)).reshape(bt, t, DIM)
    x = x + _dense(o, a["Dense_1"]["kernel"], a["Dense_1"]["bias"])

    # conv + SE
    n1 = p["LayerNorm_1"]
    z = _ln(x, n1["scale"], n1["bias"])
    y = jax.nn.swish(_dw(z, p["Conv_0"]["kernel"])) + jax.nn.swish(_dw(z, p["Conv_1"]["kernel"]))
    y = _conv1x1(y, p["Conv_2"]["kernel"])
    zz = jnp.mean(y, axis=1, keepdims=True)
    zz = jax.nn.swish(_dense(zz, p["Dense_0"]["kernel"], None))
    zz = _dense(zz, p["Dense_1"]["kernel"], None)
    x = x + y * jax.nn.sigmoid(zz)

    # ffn
    n2 = p["LayerNorm_2"]
    z = _ln(x, n2["scale"], n2["bias"])
    z = jax.nn.swish(_dense(z, p["Dense_2"]["kernel"], p["Dense_2"]["bias"]))
    x = x + _dense(z, p["Dense_3"]["kernel"], p["Dense_3"]["bias"])
    return x


def _gn(x, scale, bias, groups=4, eps=1e-5):
    """Flax GroupNorm: statistics per (sample, group) over every OTHER axis."""
    *lead, c = x.shape
    g = c // groups
    xr = x.reshape(*lead, groups, g)
    m = jnp.mean(xr, axis=(-1, -2), keepdims=True)
    v = jnp.var(xr, axis=(-1, -2), keepdims=True)
    xr = (xr - m) / jnp.sqrt(v + eps)
    xr = xr.reshape(*lead, c)
    return xr * scale + bias


def _ln(x, scale, bias, eps=1e-6):
    m = jnp.mean(x, axis=-1, keepdims=True)
    v = jnp.var(x, axis=-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps) * scale + bias


def _dense(x, k, b):
    y = jnp.dot(x, k)
    return y if b is None else y + b


def _dw(x, k):
    return jax.lax.conv_general_dilated(
        x, k, window_strides=(1,), padding="SAME",
        dimension_numbers=("NWC", "WIO", "NWC"),
        feature_group_count=x.shape[-1])


def _conv1x1(x, k):
    """Pointwise conv, kernel (1, in, out). Drop the size-1 spatial axis;
    jnp.dot would otherwise contract against the wrong one."""
    return jnp.matmul(x, k[0])


def _pool(x, params):
    p = params["SoftORStatsPool_0"]
    mean = jnp.mean(x, axis=1)
    mx = jnp.max(x, axis=1)
    proj = _dense(x, p["Dense_0"]["kernel"], p["Dense_0"]["bias"])
    s = jnp.einsum("bth,h->bt", proj, p["or_w"]) + p["or_b"]
    so = jax.scipy.special.logsumexp(s, axis=1) - jnp.log(s.shape[1])
    return jnp.concatenate([mean, so[:, None], mx], axis=-1)


if __name__ == "__main__":
    main()