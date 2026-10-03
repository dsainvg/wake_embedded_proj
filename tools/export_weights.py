"""
Export the Amaze v11 checkpoint into artefacts the ESP32 firmware can consume.

Three blobs come out of main/model/ :

  kws_model_i8.bin   int8 weights, the only weight data the firmware reads.
                     84 KB instead of 339 KB.
  kws_scales.bin     one float32 per tensor, the dequantisation scale.
  kws_param_f32.bin  the handful of tensors that stay float32 (see
                     FLOAT_TENSORS) -- normalisation gains and biases, the
                     attention position bias and the soft-OR scoring vector.
  kws_model.bin      float32 weights in the same layout. Not built into the
                     firmware; the host harness uses it to measure what the
                     INT8 path costs numerically.

LAYOUT
------
Every tensor with two or more axes is stored with its axes REVERSED, i.e. what
Flax calls a (in, out) Dense kernel becomes (out, in), and a (kT, kF, in, out)
conv kernel becomes (out, in, kF, kT).

The reason is the inner loop of the INT8 kernels. Both the conv2d kernel and
the GEMM accumulate `out[o] += sum_i q[i] * w[o][i]`, which needs the reduction
axis (the input channel) contiguous. Flax's layout puts it strided by `out`.
Transposing once, offline, turns a strided reduction into a contiguous one and
is what lets the Xtensa 8-bit MAC path consume four weights per instruction
from a single 32-bit load.

kws_model_data.h is generated from the real checkpoint tree rather than
hand-transcribed, so a Flax rename or reshape surfaces immediately instead of
silently reordering weights.

Usage (from the repo root):
    python tools/export_weights.py --int8
    python tools/export_weights.py --int8 --dump
    python tools/export_weights.py --int8 --dump --wav clip.wav
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")
OUT_DIR = os.path.join(ROOT, "main", "model")
sys.path.insert(0, MODELS)

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization

from models import get_model, _delta_stack, SEBlock  # noqa: E402,F401
import flax.linen as nn  # noqa: E402,F401

CKPT = os.path.join(MODELS, "best_v11_production.flax")
ARCH = "bcconformer_v3"

# Tensor order is frozen here and mirrored by the enum in main/kws_model.h.
# It is generated from the real checkpoint tree rather than hand-transcribed, so
# a Flax rename surfaces immediately instead of silently reordering weights.
#
# Shapes (from the checkpoint):
#   Conv_0            (3,3,3,32)    stem conv, stride (1,2)
#   Conv_1            (1,3,1,32)    depthwise over mel (feature_group_count=32
#                                    makes the channel axis 1 wide)
#   Conv_2            (3,3,32,48)   stem conv, stride (1,2)
#   Conv_3            (1,48,1)      mel attention gate
#   block Conv_0/1/2  (3,1,48) (7,1,48) (1,48,48)
#   block Dense_0/1  (48,6) (6,48)  SE
#   block Dense_2/3  (48,96) (96,48) FFN
#   attn Dense_0/1   (48,144) (48,48), rel_bias (97,4)
#   pool Dense_0     (48,24), or_w (24,), or_b scalar
#   Dense_0          (97,2) classifier

BLOCK_TENSORS = [
    "LayerNorm_0/scale", "LayerNorm_0/bias",
    "RelPosSelfAttention_0/Dense_0/kernel", "RelPosSelfAttention_0/Dense_0/bias",
    "RelPosSelfAttention_0/rel_bias",
    "RelPosSelfAttention_0/Dense_1/kernel", "RelPosSelfAttention_0/Dense_1/bias",
    "LayerNorm_1/scale", "LayerNorm_1/bias",
    "Conv_0/kernel", "Conv_1/kernel", "Conv_2/kernel",
    "Dense_0/kernel", "Dense_1/kernel",
    "LayerNorm_2/scale", "LayerNorm_2/bias",
    "Dense_2/kernel", "Dense_2/bias",
    "Dense_3/kernel", "Dense_3/bias",
]

TENSOR_ORDER = (
    [
        "Conv_0/kernel",
        "GroupNorm_0/scale", "GroupNorm_0/bias",
        "Conv_1/kernel",
        "GroupNorm_1/scale", "GroupNorm_1/bias",
        "Conv_2/kernel",
        "GroupNorm_2/scale", "GroupNorm_2/bias",
        "Conv_3/kernel",
    ]
    + [f"RelConformerBlock_{b}/{t}" for b in range(3) for t in BLOCK_TENSORS]
    + [
        "SoftORStatsPool_0/Dense_0/kernel",
        "SoftORStatsPool_0/Dense_0/bias",
        "SoftORStatsPool_0/or_w",
        "SoftORStatsPool_0/or_b",
        "Dense_0/kernel",
        "Dense_0/bias",
    ]
)

# Tensors stored tap-major, (out, kF, kT, in), instead of the plain axis
# reversal. conv2 only; see emit_layout for why its kernel needs each tap's
# input channels contiguous.
TAP_MAJOR = {"Conv_2/kernel"}

# Tensors kept in float32.
#
# Two groups, and both are about error reaching the logits rather than about
# saving bytes -- together they are 1.1 KB, which is nothing.
#
#   1. Every 1-D tensor: GroupNorm / LayerNorm scale and bias, the Dense biases,
#      the soft-OR scoring vector. Each one is added to or multiplies EVERY
#      activation, so its quantisation error is not averaged down by a long
#      reduction; it arrives intact at the logits.
#
#   2. The relative position bias. It is added to the attention logits, i.e.
#      straight into a softmax. An INT8 step of even 0.01 in a logit is a 1%
#      error in the attention weight of that frame, and there are only 97x4
#      values to store.
#
#   3. The five convolutions whose arithmetic stays float32: the two stem
#      convolutions and the two depthwise-over-time kernels per block. Together
#      2,400 values, 9.6 KB of flash. They are depthwise or tiny, so they are
#      7.6% of the MACs -- int8 there buys nothing worth the requantisation
#      plumbing, and float keeps conv1 exact so the stem is a clean reference.
FLOAT_TENSORS = (
    "GroupNorm_0/scale", "GroupNorm_0/bias",
    "GroupNorm_1/scale", "GroupNorm_1/bias",
    "GroupNorm_2/scale", "GroupNorm_2/bias",
    "Conv_0/kernel", "Conv_1/kernel",
    "SoftORStatsPool_0/or_w", "SoftORStatsPool_0/or_b",
    "Dense_0/bias",
)
FLOAT_TENSORS += tuple(
    t for b in range(3) for t in (
        f"RelConformerBlock_{b}/LayerNorm_{i}/{p}" for i in range(3)
        for p in ("scale", "bias")
    )
)
FLOAT_TENSORS += tuple(
    f"RelConformerBlock_{b}/RelPosSelfAttention_0/rel_bias" for b in range(3)
)
FLOAT_TENSORS += tuple(
    f"RelConformerBlock_{b}/RelPosSelfAttention_0/Dense_1/bias" for b in range(3)
)
FLOAT_TENSORS += tuple(
    f"RelConformerBlock_{b}/RelPosSelfAttention_0/Dense_0/bias" for b in range(3)
)
FLOAT_TENSORS += tuple(
    f"RelConformerBlock_{b}/Dense_{i}/bias" for b in range(3) for i in (2, 3)
)
FLOAT_TENSORS += tuple(
    f"RelConformerBlock_{b}/Conv_{i}/kernel" for b in range(3) for i in (0, 1)
)
FLOAT_TENSORS += ("SoftORStatsPool_0/Dense_0/bias",)
FLOAT_TENSORS = tuple(sorted(set(FLOAT_TENSORS)))


# Reduction-axis length of every tensor the INT8 kernels reduce over, as
# (number of independent output rows, length of one reduction).
#
# The kernels consume 16 bytes per vector instruction, so every reduction is
# rounded up to a multiple of 16 and the pad bytes are zero. Zeros contribute
# nothing to the dot product, and rounding the stride up to a multiple of 16
# also keeps every row of every weight matrix 16-byte aligned -- which is the
# difference between one instruction per 16 MACs and four.
#
# conv2's 3x3x32 window is already 288 = 18 x 16, so it is listed for
# completeness rather than because it changes. The classifier's 97 and the
# squeeze-and-excite expand's 6 are the two that actually gain padding, 25 bytes
# in total.
T, MEL, DIM, C1, C2, SE = 49, 40, 48, 32, 48, 6

REDUCE = {
    "Conv_2/kernel": (C2, 3 * 3 * C1),                # stem conv, stride (1,2)
    "RelConformerBlock_{b}/RelPosSelfAttention_0/Dense_0/kernel": (None, DIM),
    "RelConformerBlock_{b}/RelPosSelfAttention_0/Dense_1/kernel": (None, DIM),
    "RelConformerBlock_{b}/Conv_2/kernel": (None, DIM),
    "RelConformerBlock_{b}/Dense_0/kernel": (None, DIM),
    "RelConformerBlock_{b}/Dense_1/kernel": (None, SE),
    "RelConformerBlock_{b}/Dense_2/kernel": (None, DIM),
    "RelConformerBlock_{b}/Dense_3/kernel": (None, 2 * DIM),
    "SoftORStatsPool_0/Dense_0/kernel": (None, DIM),
    "Dense_0/kernel": (None, 2 * DIM + 1),
}


def reduce_len(name):
    """Reduction length of a tensor, or None if the kernels never reduce over it.

    The block tensors share one definition, keyed with a {b} placeholder, so a
    concrete name is folded back onto the pattern before looking it up. Not
    doing so silently leaves those tensors unpadded, which costs correctness
    rather than speed: the kernels then stride by a rounded-up length that no
    longer matches the exported one.
    """
    if name in REDUCE:
        return REDUCE[name][1]
    for key, (_, red) in REDUCE.items():
        for b in range(3):
            if key.replace("{b}", str(b)) == name:
                return red
    return None


def pad_rows(arr, n_out, red_len):
    """Pad each of `n_out` rows from `red_len` to a multiple of 16, zero filled.

    Only valid for a tensor whose reduction axis is the trailing flattened one,
    which every tensor in REDUCE satisfies: after the axis reversal a conv
    kernel is (out, in, kF, kT) and a Dense kernel is (out, in).
    """
    pad = (-red_len) % 16
    if pad == 0:
        return arr
    out = np.zeros((n_out, red_len + pad), dtype=arr.dtype)
    out[:, :red_len] = arr.reshape(n_out, red_len)
    return out.reshape(-1)


def load_params():
    model = get_model(ARCH, num_classes=2)
    dummy = jnp.ones((1, 49, 40, 1), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), dummy, train=False)
    template = {
        "params": variables["params"],
        "arch": ARCH,
        "param_count": 0, "val_acc": 0.0, "tpr": 0.0, "fpr": 0.0,
        "threshold": 0.85,
    }
    with open(CKPT, "rb") as f:
        encoded = f.read()
    try:
        st = serialization.from_bytes(template, encoded)
    except ValueError:
        st = serialization.msgpack_restore(encoded)
    params = jax.tree_util.tree_map(jnp.asarray, st["params"])
    return model, params, st


def flatten(tree):
    """Flatten a Flax param tree to {'a/b/c': array}."""
    out = {}

    def walk(prefix, node):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(f"{prefix}/{k}" if prefix else k, v)
        else:
            out[prefix] = np.asarray(node)

    walk("", tree)
    return out


def emit_layout(flat):
    """Apply the axis-reversal rule and return {name: array}.

    conv2 is the one exception, and it is a permutation rather than a
    reversal. Its window carries one activation scale per (frame, mel) tap, so
    the kernel has to reduce each tap separately to apply that tap's scale.
    Under the plain reversal a stored row is (in, kF, kT) -- channel-major --
    which makes each tap strided by kT*kW and therefore impossible to feed to
    EE.VMULAS.S8. Storing (kF, kT, in) instead puts each tap's 32 channels
    contiguous, so nine 32-wide dots are the same eighteen vector instructions
    as one 288-wide dot, and the nine per-tap scales still cost only nine
    multiplies per output channel.
    """
    out = {}
    for n, a in flat.items():
        if n in TAP_MAJOR:
            # Flax (kT, kF, in, out) -> (out, kF, kT, in), spelled out because
            # the two middle axes must swap, which no single reversal or
            # transpose expresses.
            assert a.ndim == 4, (n, a.shape)
            out[n] = a.transpose(3, 1, 0, 2).copy()
            continue
        out[n] = (a.T.copy() if a.ndim >= 2 else a.astype(np.float32).copy())
    return out


def quantise(arr):
    """Symmetric per-tensor INT8. Returns (int8 array, float32 scale).

    Scale is max|x| / 127. Per-tensor, not per-channel: measured on this
    checkpoint, per-channel came out WORSE on the noise fixture (0.0328 vs
    0.0270 against float32), because a per-channel scale lets one near-zero
    channel amplify its own quantisation noise. A zero tensor gets scale 1.0 so
    dequantising it yields exactly zero instead of dividing by zero.
    """
    amax = float(np.abs(arr).max()) if arr.size else 0.0
    scale = (amax / 127.0) if amax > 0 else 1.0
    q = np.clip(np.rint(arr / scale), -127, 127).astype(np.int8)
    return q, np.float32(scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true",
                    help="also write reference tensors for C validation")
    ap.add_argument("--int8", action="store_true",
                    help="emit the int8 weight blob, scales and float params")
    ap.add_argument("--fixtures", default="noise",
                    help="comma-separated fixture names to dump (see FIXTURES)")
    ap.add_argument("--wav", default=None, help="wav file to dump reference outputs for")
    args = ap.parse_args()

    model, params, st = load_params()
    flat = flatten(params)

    print("=" * 70)
    print(f"checkpoint : {os.path.relpath(CKPT, ROOT)}")
    print(f"arch       : {ARCH}")
    n = sum(v.size for v in flat.values())
    print(f"params     : {n:,}")
    print("-" * 70)
    print("operating point frozen in the checkpoint")
    for k in ("threshold", "deploy_agg", "deploy_need", "deploy_confirm_window",
              "deploy_refractory_steps", "deploy_span", "deploy_recall",
              "deploy_fa_per_hour", "tpr_offset_stress", "fpr_budget"):
        if k in st:
            print(f"  {k:26s} {st[k]}")
    print("=" * 70)

    missing = sorted(set(flat) - set(TENSOR_ORDER))
    extra = sorted(set(TENSOR_ORDER) - set(flat))
    if missing:
        print("WARNING: params in checkpoint but NOT in TENSOR_ORDER:")
        for m in missing:
            print(f"   {m}  {flat[m].shape}")
    if extra:
        print("WARNING: names in TENSOR_ORDER but NOT in checkpoint:")
        for e in extra:
            print(f"   {e}")
    if missing or extra:
        sys.exit("TENSOR_ORDER does not match the checkpoint - fix and rerun")

    bad = sorted(set(FLOAT_TENSORS) - set(flat))
    if bad:
        sys.exit("FLOAT_TENSORS names not present in the checkpoint: " + ", ".join(bad))

    os.makedirs(OUT_DIR, exist_ok=True)

    lay = emit_layout(flat)
    shapes = {n: a.shape for n, a in lay.items()}

    # ---- pad every reduction to a multiple of 16 --------------------------
    #      Both blobs get the same padding, so the float32 and INT8 kernels
    #      stride identically and the host A/B comparison stays apples to
    #      apples. Pad bytes are zero, so a reduction never sees them.
    for name in list(lay):
        red = reduce_len(name)
        if red is None:
            continue
        n_out = lay[name].size // red
        pad = (-red) % 16
        if pad == 0:
            continue
        lay[name] = pad_rows(lay[name], n_out, red)
        shapes[name] = (n_out, red + pad)

    # ---- always emit the float32 blob in the transposed layout ----------
    fblob, header, offset = [], [], 0
    for name in TENSOR_ORDER:
        arr = lay[name].astype(np.float32).ravel()
        header.append((name, shapes[name], offset))
        fblob.append(arr)
        offset += arr.size
    fblob = np.concatenate(fblob)
    fpath = os.path.join(OUT_DIR, "kws_model.bin")
    fblob.tofile(fpath)

    # ---- int8 blob + scales + float parameter blob ----------------------
    # Offsets are rounded up to 16 so every tensor starts on a vector
    # boundary; the vector kernel reads 16 bytes at a time and must not read
    # off the front of the flash region.
    i8_parts, pf32_parts, scales = [], [], []
    i8_offset, pf32_offset = 0, 0
    i8_meta = []
    for name, shape, _ in header:
        arr = lay[name].astype(np.float32).ravel()
        if name in FLOAT_TENSORS:
            q, scale = arr.astype(np.int8), np.float32(1.0)
            pf32_parts.append(arr)
            i8_meta.append((name, shape, 0xFFFF, pf32_offset, arr.size))
            pf32_offset += arr.size
        else:
            q, scale = quantise(arr)
            gap = (-i8_offset) % 16
            if gap:
                i8_parts.append(np.zeros(gap, np.int8))
                i8_offset += gap
            i8_parts.append(q)
            i8_meta.append((name, shape, i8_offset, 0, q.size))
            i8_offset += q.size
        scales.append(scale)

    # 128 bytes of zero slack past the last real weight: the vector kernel is
    # allowed to read up to 16 bytes past the end of a reduction, and the
    # classifier's reduction is padded out to 112 bytes. Without the slack the
    # last vector load could touch memory past the end of the blob.
    slack = 128
    i8_len = i8_offset + slack
    i8_parts.append(np.zeros(slack, np.int8))

    pf32 = (np.concatenate(pf32_parts) if pf32_parts
            else np.zeros(0, np.float32)).astype(np.float32)

    with open(os.path.join(OUT_DIR, "kws_model_data.h"), "w", newline="\n") as f:
        w = f.write
        w("/* Generated by tools/export_weights.py - do not edit by hand.\n")
        w(f" * Source: models/{os.path.basename(CKPT)}  arch {ARCH}  "
          f"{n:,} params\n")
        w(" *\n")
        w(" * Layout: every tensor with >= 2 axes has its axes REVERSED, so a\n")
        w(" * Flax (in,out) Dense kernel is stored (out,in) and a HWIO conv kernel\n")
        w(" * (out,in,kF,kT). The INT8 kernels reduce over the input channel and\n")
        w(" * need it contiguous. See the module docstring in export_weights.py.\n")
        w(" *\n")
        w(" * Include AFTER kws_model.h, which owns the tensor enum and the\n")
        w(" * kws_tensor_meta type. Entry i of each table is tensor i of the\n")
        w(" * enum; the static assert below enforces that they have not drifted.\n")
        w(" */\n")
        w("#pragma once\n\n")
        w(f"#define KWS_PARAM_COUNT  {n}u\n")
        w("/* kws_model.h declares a fallback for both of these so it can be\n")
        w(" * included on its own; drop any definition it already provided. */\n")
        w("#ifdef KWS_I8_BYTES\n#undef KWS_I8_BYTES\n#endif\n")
        w(f"#define KWS_I8_BYTES     {i8_len}u\n")
        w("#ifdef KWS_PF32_COUNT\n#undef KWS_PF32_COUNT\n#endif\n")
        w(f"#define KWS_PF32_COUNT   {pf32.size}u\n\n")

        w("/* Shapes are in the REVERSED layout described above. count is the\n")
        w(" * element count; offset is the byte offset into kws_weights_i8, or\n")
        w(" * 0xFFFF when the tensor is kept in float32 and addressed through\n")
        w(" * kws_param_f32 instead. foff is the element offset of the same tensor\n")
        w(" * in the float32 reference blob, which the host harness binds to\n")
        w(" * measure what the INT8 path costs numerically. */\n")
        w("const kws_tensor_meta kws_tensors[] = {\n")
        for (name, shape, off, po, cnt), (_, _, foff) in zip(i8_meta, header):
            d = list(shape) + [1] * (4 - len(shape))
            w(f"    /* {name:52s} */ "
              f"{{ {d[0]}, {d[1]}, {d[2]}, {d[3]}, "
              f"{int(np.prod(shape))}u, 0x{off:04x}u, {foff}u }},\n")
        w("};\n\n")

        w("/* Element offset into kws_param_f32, meaningful only for the float\n")
        w(" * tensors. */\n")
        w("const uint16_t kws_param_off[] = {\n")
        for name, shape, off, po, cnt in i8_meta:
            w(f"    /* {name:52s} */ {po}u,\n")
        w("};\n\n")

        w("const float kws_wscale[] = {\n")
        for (name, shape, off, po, cnt), s in zip(i8_meta, scales):
            w(f"    /* {name:52s} */ {float(s).hex()},\n")
        w("};\n\n")

        w(f"/* {pf32.size} float32 values: the tensors listed in FLOAT_TENSORS.\n")
        w(" * Generated as values, not a blob, because it is small and every one\n")
        w(" * of them is read on every inference. */\n")
        w(f"__attribute__((aligned(16))) const float kws_param_f32[{pf32.size}] = {{\n")
        for i in range(0, pf32.size, 8):
            w("    " + ", ".join(float.hex(float(v)) for v in pf32[i:i + 8]) + ",\n")
        w("};\n\n")

        w("/* The generated tables must stay the same length as the enum in\n")
        w(" * kws_model.h. A mismatch means the checkpoint changed shape without\n")
        w(" * the firmware enum being updated, which would silently read the\n")
        w(" * wrong weights. */\n")
        w("_Static_assert(sizeof(kws_tensors) / sizeof(kws_tensors[0])\n")
        w("               == KWS_NUM_TENSORS,\n")
        w('              "kws_tensors length != KWS_NUM_TENSORS");\n')
        w("_Static_assert(sizeof(kws_param_off) / sizeof(kws_param_off[0])\n")
        w("               == KWS_NUM_TENSORS, \"kws_param_off length\");\n")
        w("_Static_assert(sizeof(kws_wscale) / sizeof(kws_wscale[0])\n")
        w("               == KWS_NUM_TENSORS, \"kws_wscale length\");\n")

    print(f"wrote {fpath}  ({fblob.size * 4:,} bytes, {len(header)} tensors, "
          f"host-only float32 reference)")
    print("wrote " + os.path.join(OUT_DIR, "kws_model_data.h"))

    if args.int8:
        i8 = (np.concatenate(i8_parts) if i8_parts else np.zeros(0, np.int8))
        assert len(i8) == i8_len, (len(i8), i8_len)
        ipath = os.path.join(OUT_DIR, "kws_model_i8.bin")
        i8.tofile(ipath)
        scal = np.asarray(scales, dtype=np.float32)
        scal.tofile(os.path.join(OUT_DIR, "kws_scales.bin"))

        nzero = int(sum(int((p == 0).sum()) for p in i8_parts))
        pad_total = i8_len - sum(int(p.size) for p in i8_parts)
        print(f"wrote {ipath}  ({i8_offset:,} B weights + {pad_total} B alignment "
              f"and read slack = {i8_len / 1024:.1f} KB; float32 would be "
              f"{(fblob.size * 4) / 1024:.1f} KB)")
        print(f"wrote {os.path.join(OUT_DIR, 'kws_scales.bin')}  "
              f"({scal.size} float32 scales)")
        print(f"float parameters: {pf32.size:,} values "
              f"({pf32.size * 4:,} B, {len(FLOAT_TENSORS)} tensors kept in float32)")
        print(f"  zero int8 weights: {100.0 * nzero / max(i8_offset, 1):.1f}%")

    if args.dump:
        tags = [t.strip() for t in args.fixtures.split(",") if t.strip()]
        dump_reference(model, params, args.wav, tags)

    emit_frontend_data()


# ---------------------------------------------------------------------------
# Fixtures. Each is a 1.000 s window at 16 kHz, the model's input contract.
# ---------------------------------------------------------------------------

def make_fixture(name):
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
    raise ValueError(f"unknown fixture {name}")


def emit_frontend_data():
    """Emit the window and the sparse mel filterbank used by kws_frontend.c.

    These are computed with numpy here rather than in C so the firmware sees
    exactly the values the training pipeline saw. Recomputing np.hanning and the
    filterbank on the device would introduce small differences at the log()
    epsilon floor, where the features are most sensitive.
    """
    from features import AudioFeatureExtractor
    ex = AudioFeatureExtractor()

    window = np.asarray(ex.window, dtype=np.float32)

    # mel_fb is (257, 40). Keep only the nonzero span per filter.
    spans = []
    packed = []
    cursor = 0
    for m in range(ex.mel_fb.shape[1]):
        col = ex.mel_fb[:, m]
        nz = np.nonzero(col)[0]
        if nz.size == 0:
            spans.append((0, 0, cursor))
            continue
        lo, hi = int(nz[0]), int(nz[-1]) + 1
        spans.append((lo, hi - lo, cursor))
        packed.extend(col[lo:hi].astype(np.float32).tolist())
        cursor += hi - lo

    packed = np.asarray(packed, dtype=np.float32)
    path = os.path.join(OUT_DIR, "kws_frontend_data.h")
    with open(path, "w", newline="\n") as f:
        f.write("/* Generated by tools/export_weights.py - do not edit by hand.\n")
        f.write(" * Values produced by models/features.py so the firmware sees\n")
        f.write(" * exactly the window and mel filterbank used during training.\n")
        f.write(" * Include AFTER kws_frontend.h. */\n")
        f.write("#pragma once\n\n")

        f.write(f"const float kws_window[{window.size}] = {{\n")
        for i in range(0, window.size, 8):
            body = ", ".join(float.hex(float(v)) for v in window[i:i + 8])
            f.write(f"    {body},\n")
        f.write("};\n\n")

        f.write("const kws_mel_span kws_mel_spans[40] = {\n")
        for lo, n, c in spans:
            f.write(f"    {{ {lo}, {n} }},\n")
        f.write("};\n\n")

        f.write(f"const float kws_mel_weights[{packed.size}] = {{\n")
        for i in range(0, packed.size, 8):
            body = ", ".join(float.hex(float(v)) for v in packed[i:i + 8])
            f.write(f"    {body},\n")
        f.write("};\n")

    total = sum(n for _, n, _ in spans)
    print(f"wrote {path}")
    print(f"  window  {window.size} values")
    print(f"  mel spans {len(spans)}, packed weights {packed.size} "
          f"(dense 257x40 would be {257 * 40})")
    print(f"  per-frame filterbank MACs: {total} vs {257 * 40} dense "
          f"({257 * 40 / max(total, 1):.1f}x less)")


def dump_reference(model, params, wav_path=None, tags=("noise",)):
    """Write golden tensors the C implementation is checked against."""
    from features import AudioFeatureExtractor
    ref_dir = os.path.join(OUT_DIR, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    ex = AudioFeatureExtractor()

    for tag in tags:
        if wav_path and tag == "wav":
            import soundfile as sf
            audio, sr = sf.read(wav_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            assert sr == 16000, sr
            wav = audio[:16000].astype(np.float32)
            tag = os.path.splitext(os.path.basename(wav_path))[0]
        else:
            wav = make_fixture(tag)

        spec = ex.compute_spectrogram(wav)
        x = jnp.asarray(spec)[None, ..., None]
        logits = model.apply({"params": params}, x, train=False)
        probs = jax.nn.softmax(logits, axis=-1)

        np.save(os.path.join(ref_dir, f"{tag}_spec.npy"), spec)
        np.save(os.path.join(ref_dir, f"{tag}_logits.npy"), np.asarray(logits))
        np.save(os.path.join(ref_dir, f"{tag}_prob.npy"), np.asarray(probs))
        print(f"  reference {tag:8s} spec{spec.shape} "
              f"P(keyword)={float(probs[0, 1]):.8f}")


if __name__ == "__main__":
    main()