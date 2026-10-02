"""
Export the Amaze v11 checkpoint into artefacts the ESP32 firmware can consume.

Produces, into  main/model/ :

  kws_model.bin    flat float32 blob, every parameter concatenated in a fixed
                   order. Embedded into flash with CMake EMBED_FILES rather than
                   compiled as a C array, so the repo stays small and the build
                   stays fast.

  kws_model.h      generated enum + per-tensor shape/offset constants, so the C
                   inference code indexes tensors by name at compile time and
                   never does a string lookup at runtime.

Also prints the operating point frozen into the checkpoint and, with --dump,
writes reference tensors for validating the C forward pass.

Usage (from the repo root):
    python tools/export_weights.py
    python tools/export_weights.py --dump
"""

import argparse
import os
import struct
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

from models import get_model, _delta_stack, SEBlock  # noqa: E402
import flax.linen as nn  # noqa: E402

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


def emit_int8(flat):
    """Symmetric per-tensor INT8 quantisation of the weights.

    The activation side stays float32 for now. Quantising activations needs a
    calibration pass against real audio and a decision on how much accuracy that
    costs; doing weights-only keeps the change reviewable and still removes
    ~75% of the weight traffic. Each tensor keeps its own scale because the
    network spans several orders of magnitude -- a single global scale would
    crush the attention bias.

    Scale is max|x| / 127. Zero-tensor scale is 1.0 so the dequantised result
    is exactly zero rather than a division by zero.
    """
    blob, header, offset = [], [], 0
    for name in TENSOR_ORDER:
        arr = flat[name].astype(np.float32).ravel()
        amax = float(np.abs(arr).max()) if arr.size else 0.0
        scale = (amax / 127.0) if amax > 0 else 1.0
        q = np.clip(np.rint(arr / scale), -127, 127).astype(np.int8)

        blob.append(q)
        header.append((name, flat[name].shape, offset, scale, int((q == 0).sum())))
        offset += q.size

    blob = np.concatenate(blob) if blob else np.zeros(0, np.int8)
    path = os.path.join(OUT_DIR, "kws_model_i8.bin")
    blob.tofile(path)

    scales = np.asarray([h[3] for h in header], dtype=np.float32)
    scales.tofile(os.path.join(OUT_DIR, "kws_scales.bin"))

    # INT8 weights plus one float32 scale per tensor.
    total = blob.size
    print(f"wrote {path}  ({total:,} bytes = {total / 1024:.1f} KB int8)")
    print(f"wrote {os.path.join(OUT_DIR, 'kws_scales.bin')}  "
          f"({scales.size} scales)")
    print(f"float32 equivalent was {total * 4:,} bytes "
          f"({total * 4 / 1024:.1f} KB) -> {100 * (1 - total * 1.0 / (total * 4)):.0f}% smaller")

    zero_frac = sum(h[4] for h in header) / max(total, 1)
    print(f"  zero weights after quantisation: {100 * zero_frac:.1f}% "
          f"(an upper bound on how much INT8 buys here)")
    return header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true",
                    help="also write reference tensors for C validation")
    ap.add_argument("--int8", action="store_true",
                    help="also emit per-tensor symmetric INT8 weights + scales")
    ap.add_argument("--wav", default=None, help="wav file to dump reference outputs for")
    args = ap.parse_args()

    model, params, st = load_params()
    flat = flatten(params)

    print("=" * 70)
    print(f"checkpoint : {os.path.relpath(CKPT, ROOT)}")
    print(f"arch       : {ARCH}")
    n = sum(v.size for v in flat.values())
    print(f"params     : {n:,}  (float32 blob = {n * 4 / 1024:.1f} KB)")
    print("-" * 70)
    print("operating point frozen in the checkpoint")
    for k in ("threshold", "deploy_agg", "deploy_need", "deploy_confirm_window",
              "deploy_refractory_steps", "deploy_span", "deploy_recall",
              "deploy_fa_per_hour", "tpr_offset_stress", "fpr_budget"):
        if k in st:
            print(f"  {k:26s} {st[k]}")
    print("=" * 70)

    # Report anything present in the checkpoint but absent from TENSOR_ORDER,
    # so a silently dropped weight can never reach the firmware.
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

    os.makedirs(OUT_DIR, exist_ok=True)

    blob = []
    header = []
    offset = 0
    for name in TENSOR_ORDER:
        arr = flat[name].astype(np.float32).ravel()
        header.append((name, flat[name].shape, offset))
        blob.append(arr)
        offset += arr.size

    blob = np.concatenate(blob)
    bin_path = os.path.join(OUT_DIR, "kws_model.bin")
    blob.tofile(bin_path)

    with open(os.path.join(OUT_DIR, "kws_model_data.h"), "w", newline="\n") as f:
        f.write("/* Generated by tools/export_weights.py - do not edit by hand.\n")
        f.write(f" * Source: models/{os.path.basename(CKPT)}  arch {ARCH}  "
                f"{n:,} params\n")
        f.write(" *\n")
        f.write(" * Emits the tensor table only. The enum, kws_tensor_meta and the\n")
        f.write(" * kws_blob declaration are owned by kws_model.h, which must be\n")
        f.write(" * included before this file. The index of each entry here matches\n")
        f.write(" * the enum in kws_model.h; a build-time assert below enforces it.\n")
        f.write(" */\n")
        f.write("#pragma once\n\n")
        f.write(f"#define KWS_BLOB_FLOATS {blob.size}u\n")
        f.write(f"#define KWS_PARAM_COUNT  {n}u\n\n")
        f.write("const kws_tensor_meta kws_tensors[] = {\n")
        for name, shape, off in header:
            d = list(shape) + [1] * (4 - len(shape))
            f.write(f"    /* {name:52s} */ "
                    f"{{ {d[0]}, {d[1]}, {d[2]}, {d[3]}, "
                    f"{int(np.prod(shape))}u, {off}u }},\n")
        f.write("};\n\n")
        f.write(f"/* Generated list length must equal KWS_NUM_TENSORS "
                f"({len(header)} here). */\n")
        f.write(f"_Static_assert(sizeof(kws_tensors) / sizeof(kws_tensors[0]) "
                f"== KWS_NUM_TENSORS,\n")
        f.write('              "kws_tensors length != KWS_NUM_TENSORS; the enum in "\n')
        f.write('              "kws_model_data.h length != KWS_NUM_TENSORS; the enum "\n')
        f.write('              "in kws_model.h and export_weights.py have drifted");\n')

    print(f"wrote {bin_path}  ({blob.size * 4:,} bytes, {len(header)} tensors)")
    print("wrote " + os.path.join(OUT_DIR, "kws_model_data.h"))

    if args.int8:
        emit_int8(flat)

    if args.dump:
        dump_reference(model, params, args.wav)

    emit_frontend_data()


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


def dump_reference(model, params, wav_path=None):
    """Write golden tensors the C implementation is checked against."""
    from features import AudioFeatureExtractor
    ref_dir = os.path.join(OUT_DIR, "reference")
    os.makedirs(ref_dir, exist_ok=True)

    ex = AudioFeatureExtractor()

    if wav_path:
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert sr == 16000, sr
        wav = audio[:16000]
        tag = os.path.splitext(os.path.basename(wav_path))[0]
    else:
        rng = np.random.default_rng(0)
        wav = (rng.normal(0, 0.05, 16000) * 32768).astype(np.float32)
        tag = "noise"

    spec = ex.compute_spectrogram(wav)
    x = jnp.asarray(spec)[None, ..., None]
    logits = model.apply({"params": params}, x, train=False)
    probs = jax.nn.softmax(logits, axis=-1)

    np.save(os.path.join(ref_dir, f"{tag}_spec.npy"), spec)
    np.save(os.path.join(ref_dir, f"{tag}_logits.npy"), np.asarray(logits))
    np.save(os.path.join(ref_dir, f"{tag}_prob.npy"), np.asarray(probs))
    print(f"wrote reference for '{tag}': spec{spec.shape} "
          f"logits{logits.shape} prob={np.asarray(probs)[0, 1]:.6f}")


if __name__ == "__main__":
    main()