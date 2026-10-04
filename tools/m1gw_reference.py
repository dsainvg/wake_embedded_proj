"""
float32 reference for m1_g_wide, and the host parity fixture it produces.

This is the numerical ground truth for main/kws_m1gw.c. It reads the SAME
checkpoint through the SAME torch-free reader the exporter uses, so the two
implementations cannot disagree because of a weight file -- only because of
arithmetic.

It exists because the C kernel is otherwise unverifiable here: no torch on the
build host, so the trained model cannot be executed to compare against, and no
corpus is on disk to recompute the 78.73% TPR. What this can do is catch the
failure that actually bites a hand port -- a transposed weight, a wrong stride,
a padding convention that disagrees with the trainer -- and it does that by
running the reference on real log-mel and comparing the probability, not by
comparing intermediate tensors.

IMPORTANT SCOPE NOTE: a float32 reference checks the C port's *structure*, not
its quantisation. Agreement here means the kernel computes the right function
up to int8 rounding. It does not prove the trained model's TPR survives
quantisation; only the recorded TPR and an on-device sweep can show that.

Usage:
    python tools/m1gw_reference.py --spec main/model/reference/loud_spec.npy \
        --out build/host/m1gw_fixture
    python tools/m1gw_reference.py --self-test
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from export_m1gw_weights import (ARCH_NAME, DIM, STEM_CHANNELS, NUM_BLOCKS,  # noqa: E402
                                NUM_HEADS, FFN_HIDDEN, SE_DIM, MEL_POOL_RANK,
                                TIME_STRIDE, INPUT_T, INPUT_MEL, read_checkpoint,
                                F32_ONLY, SCALE_AXIS, DEFAULT_CKPT)

HEAD_DIM = DIM // NUM_HEADS
M1 = (INPUT_MEL + 2 - 3) // 2 + 1                 # 20
T2 = (INPUT_T + 2 - 3) // TIME_STRIDE + 1          # 13
M2 = (M1 + 2 - 3) // 2 + 1                        # 10


def W(tensors, name):
    return tensors[name].astype(np.float64)


def deq(tensors, name):
    """The float weight tensor, exactly as the checkpoint stores it.

    There is nothing to dequantise here. The checkpoint holds float32 weights;
    the int8 blob is a *separate* artefact produced by tools/export_m1gw_weights.py
    for the firmware. This function used to apply a per-channel scale on the way
    through, which shrank every weight by max/127 -- a factor of ~127 -- and made
    the reference disagree with a correct C kernel by a uniform 120x while
    looking perfectly self-consistent.

    It is kept as a named function because every call site reads better as
    "the weights the network actually uses", and because it documents the trap.
    """
    return W(tensors, name)


def delta_stack(x):
    """(49,40) -> (3,49,40): [log-mel, central difference, second difference].

    Edge replication on both ends, matching models_torch._time_diff. The subtlety
    that killed an earlier port: _time_diff is applied twice, so d2's edges use
    the ALREADY-clamped d1, not a fresh clamp of x. Computing d2 from x with
    independent clamps gives different values at the first and last frame.
    """
    def tdiff(a):
        nxt = np.minimum(np.arange(a.shape[0]) + 1, a.shape[0] - 1)
        prv = np.maximum(np.arange(a.shape[0]) - 1, 0)
        return 0.5 * (a[nxt] - a[prv])

    d1 = tdiff(x)
    d2 = tdiff(d1)
    return np.stack([x, d1, d2], axis=0)


def gn(x, rows, ch, groups, gw, gb):
    """GroupNorm over [rows][ch] then SiLU, channel-innermost layout.

    One mean and one variance per GROUP, over every element in that group -- all
    of its channels and all rows. Normalising per channel instead is
    InstanceNorm, which is a different layer: with groups=1 over the 3 stem
    channels it uses 3 statistics where torch uses 1, and with groups=4 over 32
    channels it uses 32 where torch uses 4. It does not fail loudly either -- it
    just makes the network nearly input-insensitive, which is what happened
    before this was caught.
    """
    per = ch // groups
    y = np.empty_like(x)
    for g in range(groups):
        sl = slice(g * per, (g + 1) * per)
        v = x[:, sl]
        mean = v.mean()                      # scalar over the WHOLE group
        var = v.var()
        inv = 1.0 / np.sqrt(var + 1e-5)
        y[:, sl] = (v - mean) * inv * gw[sl] + gb[sl]
    return y / (1.0 + np.exp(-y))


def ln(x, gw, gb):
    """LayerNorm over the last axis of a [T,D] tensor."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-5) * gw + gb


def depthwise(x, w, stride_t, stride_m, out_t, out_m):
    """Conv2d(C,C,3x3,groups=C) zero-padded. x is [t][m][C]."""
    in_t, in_m, C = x.shape
    # torch stores a depthwise Conv2d as (Cout, in/groups, kh, kw) = (C,1,kh,kw),
    # so the per-channel taps live at w[:, 0].
    assert w.shape == (C, 1, 3, 3), (w.shape, C)
    padded = np.zeros((in_t + 2, in_m + 2, C))
    padded[1:-1, 1:-1] = x
    out = np.zeros((out_t, out_m, C))
    for ot in range(out_t):
        for om in range(out_m):
            acc = np.zeros(C)
            for kt in (-1, 0, 1):
                for km in (-1, 0, 1):
                    it, im = ot * stride_t + kt + 1, om * stride_m + km + 1
                    acc += padded[it, im] * w[:, 0, kt + 1, km + 1]
            out[ot, om] = acc
    return out


def forward(tensors, spec, want_seq=False):
    p = {}

    # --- stems ---
    # delta_stack returns (3,T,M) channel-first; the whole stem path is (T,M,C).
    x = np.transpose(delta_stack(spec), (1, 2, 0))               # (49,40,3)
    x = depthwise(x, deq(tensors, "stem1.dw.weight"), 1, 2, INPUT_T, M1)
    x = gn(x.reshape(-1, 3), INPUT_T * M1, 3, 1,
           W(tensors, "stem1.norm.weight"), W(tensors, "stem1.norm.bias")).reshape(INPUT_T, M1, 3)
    x = x @ deq(tensors, "stem1.pw.weight").reshape(STEM_CHANNELS, 3).T   # (49,20,32)

    x = depthwise(x, deq(tensors, "down.dw.weight"), TIME_STRIDE, 1, T2, M1)
    x = gn(x.reshape(-1, STEM_CHANNELS), T2 * M1, STEM_CHANNELS, 4,
           W(tensors, "down.norm.weight"), W(tensors, "down.norm.bias")).reshape(T2, M1, STEM_CHANNELS)
    x = x @ deq(tensors, "down.pw.weight").reshape(STEM_CHANNELS, STEM_CHANNELS).T

    x = depthwise(x, deq(tensors, "stem2.dw.weight"), 1, 2, T2, M2)
    x = gn(x.reshape(-1, STEM_CHANNELS), T2 * M2, STEM_CHANNELS, 4,
           W(tensors, "stem2.norm.weight"), W(tensors, "stem2.norm.bias")).reshape(T2, M2, STEM_CHANNELS)
    x = x @ deq(tensors, "stem2.pw.weight").reshape(DIM, STEM_CHANNELS).T        # (13,10,96)

    # --- MelAttnPool: scorer, tanh, softmax over mel, weighted sum ---
    s = x @ deq(tensors, "mel_gate.fc1.weight").reshape(MEL_POOL_RANK, DIM).T \
        + W(tensors, "mel_gate.fc1.bias")
    s = np.tanh(s)                                                     # (13,10,8)
    sc = s @ deq(tensors, "mel_gate.fc2.weight").reshape(1, MEL_POOL_RANK).T  # (13,10,1)
    sc = sc - sc.max(axis=1, keepdims=True)
    e = np.exp(sc)
    wmel = e / e.sum(axis=1, keepdims=True)
    seq = (x * wmel).sum(axis=1)                                       # (13,96)

    # --- conformer blocks ---
    for b in range(NUM_BLOCKS):
        q = f"blocks.{b}."
        h = ln(seq, W(tensors, q + "norm1.weight"), W(tensors, q + "norm1.bias"))
        qkv = h @ deq(tensors, q + "attn.qkv.weight").T + W(tensors, q + "attn.qkv.bias")
        qq = qkv[:, :DIM].reshape(T2, NUM_HEADS, HEAD_DIM)
        kk = qkv[:, DIM:2 * DIM].reshape(T2, NUM_HEADS, HEAD_DIM)
        vv = qkv[:, 2 * DIM:].reshape(T2, NUM_HEADS, HEAD_DIM)

        idx = np.arange(T2)
        rel = (idx[:, None] - idx[None, :]) + (T2 - 1)          # signed distance
        rb = W(tensors, q + "attn.rel_bias")[rel]              # (13,13,4)

        att = np.einsum("ihd,jhd->hij", qq, kk) / math.sqrt(HEAD_DIM) + rb.transpose(2, 0, 1)
        att = att - att.max(axis=-1, keepdims=True)
        att = np.exp(att)
        att /= att.sum(axis=-1, keepdims=True)
        o = np.einsum("hij,jhd->ihd", att, vv).reshape(T2, DIM)
        seq = seq + (o @ deq(tensors, q + "attn.proj.weight").T + W(tensors, q + "attn.proj.bias"))

        h = ln(seq, W(tensors, q + "norm2.weight"), W(tensors, q + "norm2.bias"))
        acc = np.zeros_like(h)
        for i, k in enumerate((3, 7)):
            pad = k // 2
            hp = np.pad(h, ((pad, pad), (0, 0)))
            # Conv1d depthwise is stored (Cout, in/groups, k) = (96,1,k).
            wp = W(tensors, q + f"dw.{i}.weight")[:, 0, :]        # (96, k)
            assert wp.shape == (DIM, k), (wp.shape, DIM, k)
            tap = np.zeros_like(h)
            for kt in range(k):
                tap += hp[kt:kt + T2] * wp[:, kt][None, :]
            acc += tap / (1.0 + np.exp(-tap))
        y = acc @ deq(tensors, q + "pw.weight").T

        z = y.mean(axis=0) @ deq(tensors, q + "se_fc1.weight").T
        z = z / (1.0 + np.exp(-z))
        g = (z @ deq(tensors, q + "se_fc2.weight").T)
        seq = seq + y * (1.0 / (1.0 + np.exp(-g)))[None, :]

        h = ln(seq, W(tensors, q + "norm3.weight"), W(tensors, q + "norm3.bias"))
        f = h @ deq(tensors, q + "ff1.weight").T + W(tensors, q + "ff1.bias")
        f = f / (1.0 + np.exp(-f))
        seq = seq + (f @ deq(tensors, q + "ff2.weight").T + W(tensors, q + "ff2.bias"))

    # --- SoftORPool + head ---
    pool = np.concatenate([seq.mean(axis=0),
                           np.log(np.exp(seq - seq.max(axis=0)).sum(axis=0)) + seq.max(axis=0)
                           - math.log(T2),
                           seq.max(axis=0)])
    if want_seq:
        return float("nan"), seq
    logits = pool @ deq(tensors, "head.weight").T + W(tensors, "head.bias")
    e = np.exp(logits - logits.max())
    return e / e.sum()


def self_test():
    """Shapes and the two conventions that are easy to get wrong."""
    assert (M1, T2, M2) == (20, 13, 10), (M1, T2, M2)
    # T=13 must come from the conv, not from 49/4.
    assert T2 == (INPUT_T + 2 - 3) // TIME_STRIDE + 1

    meta, tensors = read_checkpoint(DEFAULT_CKPT)
    assert meta["arch"] == ARCH_NAME
    assert sum(t.size for t in tensors.values()) == meta["params"]

    # The edge convention: _time_diff clamps the INDEX, so the first and last
    # frames get one-sided differences, not zeros. A port that pads with zeros
    # instead produces different values at both edges and the error is invisible
    # in every interior sample.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((INPUT_T, INPUT_MEL))
    d = delta_stack(x)
    assert d.shape == (3, INPUT_T, INPUT_MEL)
    assert np.allclose(d[1, 0], 0.5 * (x[1] - x[0])), "first frame is a forward difference"
    assert np.allclose(d[1, -1], 0.5 * (x[-1] - x[-2])), "last frame is a backward difference"
    # d2 must come from the already-clamped d1, not from a fresh clamp of x.
    assert np.allclose(d[2, 0], 0.5 * (d[1, 1] - d[1, 0])), \
        "second difference must reuse the clamped first difference"
    assert not np.allclose(d[2, 0], 0.5 * (0.5 * (x[2] - x[0]))), \
        "second difference is not a second difference of x"

    # SoftORPool must be permutation-invariant over time, or the block has
    # reintroduced the positional prior the relative bias exists to remove.
    seq = rng.standard_normal((T2, DIM))
    perm = rng.permutation(T2)
    def pool_of(s):
        return np.concatenate([s.mean(0), np.log(np.exp(s - s.max(0)).sum(0)) + s.max(0) - math.log(T2),
                               s.max(0)])
    assert np.allclose(pool_of(seq), pool_of(seq[perm]), atol=1e-9)

    # An all-zero input must not produce NaN anywhere: a silent channel would
    # otherwise divide by zero and the probability would read as noise.
    pr = forward(tensors, np.zeros((INPUT_T, INPUT_MEL)))
    assert np.all(np.isfinite(pr)), pr
    print(f"  shapes           T2={T2} M1={M1} M2={M2}")
    print(f"  params           {sum(t.size for t in tensors.values()):,}")
    print(f"  P(keyword|silence) {pr[1]:.6f}  P(other) {pr[0]:.6f}")
    print("  self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=None, help=".npy (49,40) float32 log-mel")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=None, help="write spec.bin + expected.txt here")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        print("=" * 78)
        print(" m1_g_wide reference self-test")
        print("=" * 78)
        self_test()
        return 0

    meta, tensors = read_checkpoint(args.checkpoint)
    spec = np.load(args.spec).astype(np.float64)
    if spec.shape != (INPUT_T, INPUT_MEL):
        raise SystemExit(f"{args.spec} is {spec.shape}, expected {(INPUT_T, INPUT_MEL)}")

    pr = forward(tensors, spec)
    print(f"  P(keyword) = {pr[1]:.6f}   P(other) = {pr[0]:.6f}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        spec.astype("<f4").tofile(os.path.join(args.out, "spec.bin"))
        with open(os.path.join(args.out, "expected.txt"), "w") as fh:
            fh.write(f"{pr[1]:.9f} {pr[0]:.9f}\n")
        print(f"  wrote {args.out}/spec.bin and expected.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())