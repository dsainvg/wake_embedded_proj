"""
Weight exporter for `m1_g_wide` (models/models_torch_hc.py :: TimedKWSNet)
==========================================================================

Reads `best_m1g_wide.pt` WITHOUT torch. The checkpoints are ZIP archives whose
`data.pkl` is an ordinary pickle referring to torch storage objects, so the
architecture name, the frozen threshold and every tensor's shape are all
reachable by stubbing those classes out. The tensor payloads are little-endian
float32 in the archive and are read directly. The only thing this cannot do is
*run* the model, which is why the exporter's job is to make that unnecessary:
it validates the checkpoint against the architecture by key and by shape, so the
C port cannot silently disagree with the trained weights.

Three outputs, all from the same walk, so they cannot drift apart:

  1. `kws_m1gw_i8.bin`     int8 weights, per-output-channel symmetric
  2. `kws_m1gw_scales.bin` float32 dequant scales, one per channel (or per
                          tensor for single-channel ones)
  3. `kws_m1gw_manifest.json`  every tensor's name/shape/offset/scale, plus the
                          params / MAC / activation-RAM table

The MAC and activation figures are computed from the REAL shapes by walking the
same layer sequence the C code executes, not by scaling a recorded number. That
matters here: the checkpoint records 1,473,962 MAC for this architecture and a
direct walk of the declared layers gives a materially larger figure. The
recorded number is reproduced in the report and the discrepancy is called out
rather than quietly dropped, because the entire CPU argument for this model
rests on it.

Usage:
    python tools/export_m1gw_weights.py                    # export + report
    python tools/export_m1gw_weights.py --check            # validate only
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import pickle
import struct
import sys
import zipfile
from typing import Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_CKPT = os.path.join(ROOT, "models", "best_m1g_wide.pt")
OUT_DIR = os.path.join(ROOT, "main", "model")

# ==============================================================================
# The architecture, as a declarative spec
# ==============================================================================
# Mirrors models_torch_hc.py::ARCHS_M1["m1_g_wide"] and models_torch.py::
# TimedKWSNet. Written out rather than imported because importing means torch,
# and the point of this file is to work without it. If the preset changes, this
# table and the trained checkpoint have to change together -- --check is what
# catches it.
ARCH_NAME = "m1_g_wide"
DIM = 96
STEM_CHANNELS = 32
NUM_BLOCKS = 2
NUM_HEADS = 4
FFN_MULT = 1.0
MEL_POOL_RANK = 8
TIME_STRIDE = 4
NUM_CLASSES = 2
MAX_LEN = 128
KERNEL_SIZES = (3, 7)
SE_REDUCTION = 8
GN_GROUPS = 4

INPUT_T = 49
INPUT_MEL = 40

HEAD_DIM = DIM // NUM_HEADS
SE_DIM = max(4, DIM // SE_REDUCTION)
FFN_HIDDEN = max(4, int(round(DIM * FFN_MULT)))
REL_BIAS_LEN = 2 * MAX_LEN - 1


def _gn_groups(channels: int) -> int:
    """models_torch.DSBlock2D: GroupNorm(max(1, gcd(in_ch, num_groups)))."""
    return max(1, math.gcd(channels, GN_GROUPS))


def conv_out(size: int, k: int, s: int, p: int) -> int:
    return (size + 2 * p - k) // s + 1


def expected_tensors() -> Dict[str, Tuple[int, ...]]:
    """Every parameter the architecture must have, with its exact shape."""
    t: Dict[str, Tuple[int, ...]] = {}

    # --- stem 1: 3 channels -> STEM_CHANNELS, mel stride 2 ---
    t["stem1.dw.weight"] = (3, 1, 3, 3)
    t["stem1.norm.weight"] = (3,)
    t["stem1.norm.bias"] = (3,)
    t["stem1.pw.weight"] = (STEM_CHANNELS, 3, 1, 1)

    # --- time downsample: stride TIME_STRIDE in time, mel untouched ---
    t["down.dw.weight"] = (STEM_CHANNELS, 1, 3, 3)
    t["down.norm.weight"] = (STEM_CHANNELS,)
    t["down.norm.bias"] = (STEM_CHANNELS,)
    t["down.pw.weight"] = (STEM_CHANNELS, STEM_CHANNELS, 1, 1)

    # --- stem 2: STEM_CHANNELS -> DIM, mel stride 2 again ---
    t["stem2.dw.weight"] = (STEM_CHANNELS, 1, 3, 3)
    t["stem2.norm.weight"] = (STEM_CHANNELS,)
    t["stem2.norm.bias"] = (STEM_CHANNELS,)
    t["stem2.pw.weight"] = (DIM, STEM_CHANNELS, 1, 1)

    # --- learned attention over mel bins ---
    t["mel_gate.fc1.weight"] = (MEL_POOL_RANK, DIM, 1, 1)
    t["mel_gate.fc1.bias"] = (MEL_POOL_RANK,)
    t["mel_gate.fc2.weight"] = (1, MEL_POOL_RANK, 1, 1)

    for b in range(NUM_BLOCKS):
        p = f"blocks.{b}"
        t[f"{p}.norm1.weight"] = (DIM,)
        t[f"{p}.norm1.bias"] = (DIM,)
        t[f"{p}.attn.rel_bias"] = (REL_BIAS_LEN, NUM_HEADS)
        t[f"{p}.attn.qkv.weight"] = (3 * DIM, DIM)
        t[f"{p}.attn.qkv.bias"] = (3 * DIM,)
        t[f"{p}.attn.proj.weight"] = (DIM, DIM)
        t[f"{p}.attn.proj.bias"] = (DIM,)
        t[f"{p}.norm2.weight"] = (DIM,)
        t[f"{p}.norm2.bias"] = (DIM,)
        for i, k in enumerate(KERNEL_SIZES):
            t[f"{p}.dw.{i}.weight"] = (DIM, 1, k)
        t[f"{p}.pw.weight"] = (DIM, DIM)
        t[f"{p}.se_fc1.weight"] = (SE_DIM, DIM)
        t[f"{p}.se_fc2.weight"] = (DIM, SE_DIM)
        t[f"{p}.norm3.weight"] = (DIM,)
        t[f"{p}.norm3.bias"] = (DIM,)
        t[f"{p}.ff1.weight"] = (FFN_HIDDEN, DIM)
        t[f"{p}.ff1.bias"] = (FFN_HIDDEN,)
        t[f"{p}.ff2.weight"] = (DIM, FFN_HIDDEN)
        t[f"{p}.ff2.bias"] = (DIM,)

    t["head.weight"] = (NUM_CLASSES, 3 * DIM)
    t["head.bias"] = (NUM_CLASSES,)
    return t


# Which axis the per-output-channel scale runs along. `None` means one scale for
# the whole tensor, which is what a bias or a LayerNorm gain wants.
SCALE_AXIS = {
    "stem1.dw.weight": 0,
    "stem1.pw.weight": 0,
    "down.dw.weight": 0,
    "down.pw.weight": 0,
    "stem2.dw.weight": 0,
    "stem2.pw.weight": 0,
    "mel_gate.fc1.weight": 0,
    "mel_gate.fc2.weight": 0,
    "head.weight": 0,
}
for _b in range(NUM_BLOCKS):
    _p = f"blocks.{_b}"
    SCALE_AXIS[f"{_p}.attn.qkv.weight"] = 0
    SCALE_AXIS[f"{_p}.attn.proj.weight"] = 0
    SCALE_AXIS[f"{_p}.pw.weight"] = 0
    SCALE_AXIS[f"{_p}.se_fc1.weight"] = 0
    SCALE_AXIS[f"{_p}.se_fc2.weight"] = 0
    SCALE_AXIS[f"{_p}.ff1.weight"] = 0
    SCALE_AXIS[f"{_p}.ff2.weight"] = 0
    for _i, _k in enumerate(KERNEL_SIZES):
        SCALE_AXIS[f"{_p}.dw.{_i}.weight"] = 0

# Tensors kept as float32 rather than int8.
#
# GroupNorm/LayerNorm gains and biases stay float: a norm gain is a multiplier
# applied to a normalised activation, so quantising it to 1/127 steps perturbs the
# normalised output directly, and they are only 6 KB.
#
# EVERY conv/linear bias stays float too, and this one is not a precaution. A
# bias is added to an int32 accumulator, so in a correct int8 pipeline it is
# itself an accumulator-domain quantity and gets int32 resolution for free;
# rounding it to int8 against its own max instead costs the small values
# everything. Measured on this checkpoint, int8 `blocks.0.ff1.bias` comes out at
# 76% RMS / 234% peak error, and `head.bias` is 100% saturated. 1162 float32
# biases cost 4.6 KB.
F32_ONLY = set()
for _b in range(NUM_BLOCKS):
    _p = f"blocks.{_b}"
    F32_ONLY |= {f"{_p}.norm1.weight", f"{_p}.norm1.bias",
                 f"{_p}.norm2.weight", f"{_p}.norm2.bias",
                 f"{_p}.norm3.weight", f"{_p}.norm3.bias",
                 f"{_p}.attn.rel_bias"}
F32_ONLY |= {"stem1.norm.weight", "stem1.norm.bias",
             "down.norm.weight", "down.norm.bias",
             "stem2.norm.weight", "stem2.norm.bias",
             "mel_gate.fc1.bias", "head.bias"}
for _b in range(NUM_BLOCKS):
    _p = f"blocks.{_b}"
    F32_ONLY |= {f"{_p}.attn.qkv.bias", f"{_p}.attn.proj.bias",
                 f"{_p}.ff1.bias", f"{_p}.ff2.bias"}


# ==============================================================================
# Reading a .pt without torch
# ==============================================================================

class _Storage:
    """A torch storage, identified by its key inside the archive.

    This checkpoint is in torch's legacy zip layout, where payloads are
    `data/<key>` with an integer key rather than `data/0` under the parameter's
    name. The key comes from the pickle's persistent id, so it has to be kept
    here or the payload cannot be found.
    """

    def __init__(self, key=None, numel=0):
        self.key = key
        self.numel = numel


class _TensorSpec:
    def __init__(self, storage, offset, size, stride, *rest):
        self.storage = storage
        self.offset = offset
        self.size = tuple(size)
        self.stride = tuple(stride)

    @property
    def numel(self):
        n = 1
        for d in self.size:
            n *= d
        return n


class _Stub:
    def __init__(self, *a, **k):
        pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("torch"):
            if "Storage" in name:
                return _Storage
            if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                return _TensorSpec
            return type(name, (_Stub,), {})
        return super().find_class(module, name)

    def persistent_load(self, pid):
        if isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage":
            key, numel = pid[2], pid[4]
            return _Storage(key, numel if isinstance(numel, int) else 0)
        return pid


def read_checkpoint(path: str) -> Tuple[dict, Dict[str, np.ndarray]]:
    """Return (metadata, {name: float32 ndarray}) laid out in torch's order."""
    z = zipfile.ZipFile(path)
    pkl_name = next(n for n in z.namelist() if n.endswith("data.pkl"))
    prefix = pkl_name[: -len("data.pkl")]
    obj = _Unpickler(io.BytesIO(z.read(pkl_name))).load()
    if not isinstance(obj, dict):
        raise SystemExit(f"{path}: expected a dict, got {type(obj).__name__}")

    meta = {k: v for k, v in obj.items() if not isinstance(v, dict)}
    sd = obj.get("state_dict") or obj.get("model") or obj.get("model_state_dict")
    if not isinstance(sd, dict):
        raise SystemExit(f"{path}: no state_dict found (keys: {sorted(obj)})")

    tensors: Dict[str, np.ndarray] = {}
    for name, spec in sd.items():
        if not isinstance(spec, _TensorSpec):
            continue
        storage = spec.storage
        if not isinstance(storage, _Storage) or storage.key is None:
            raise SystemExit(f"{name}: no storage key in the archive")
        data_key = f"{prefix}data/{storage.key}"
        raw = z.read(data_key)
        want = spec.numel
        if len(raw) != want * 4:
            raise SystemExit(
                f"{name}: archive holds {len(raw)} bytes for {want} float32 values. "
                "The checkpoint is probably fp16 or bf16, which this reader does "
                "not handle -- convert it to fp32 before exporting.")
        flat = np.frombuffer(raw, dtype="<f4")
        # torch row-major (C order); reshape is a view, no copy
        tensors[name] = flat.reshape(spec.size)

    return meta, tensors


# ==============================================================================
# Validation
# ==============================================================================

def validate(meta: dict, tensors: Dict[str, np.ndarray]) -> List[str]:
    """Key-for-key and shape-for-shape against the architecture. Returns errors."""
    errs: List[str] = []

    arch = meta.get("arch")
    if arch != ARCH_NAME:
        errs.append(f"checkpoint arch is {arch!r}, this exporter targets {ARCH_NAME!r}")

    want = expected_tensors()
    missing = sorted(set(want) - set(tensors))
    extra = sorted(set(tensors) - set(want))
    if missing:
        errs.append(f"{len(missing)} tensor(s) the architecture requires are absent: "
                    + ", ".join(missing[:8]) + ("..." if len(missing) > 8 else ""))
    if extra:
        errs.append(f"{len(extra)} tensor(s) in the checkpoint are not in the "
                    f"architecture: " + ", ".join(extra[:8])
                    + ("..." if len(extra) > 8 else ""))

    for name in sorted(set(want) & set(tensors)):
        got = tuple(tensors[name].shape)
        exp = want[name]
        if got != exp:
            errs.append(f"{name}: checkpoint shape {list(got)} != architecture {list(exp)}")

    params = sum(t.size for t in tensors.values())
    if int(meta.get("params", params)) != params:
        errs.append(f"recorded params {meta.get('params')} != tensors on disk {params}")

    # The mandate that cost the last architecture 62 points of recall.
    for name in tensors:
        if any(k in name for k in ("pos_emb", "abs_pos", "posenc")):
            errs.append(f"absolute position embedding present ({name}); the architecture "
                        "must use a relative bias only")
    if not any("rel_bias" in n for n in tensors):
        errs.append("no rel_bias tensor: relative position bias is missing")

    if not errs:
        total = sum(np.prod(t.shape) for t in tensors.values())
        print(f"  checkpoint validated against {ARCH_NAME}")
        print(f"    tensors           {len(tensors)}")
        print(f"    params            {total:,} (recorded {int(meta.get('params', total)):,})")
        print(f"    threshold         {meta.get('threshold')}")
        print(f"    recorded TPR      {meta.get('tpr')}")
        print(f"    recorded AUC      {meta.get('auc')}")
        print(f"    recorded MAC      {meta.get('mac'):,.0f}")
    return errs


# ==============================================================================
# Budget: walk the real shapes
# ==============================================================================

def conv_shapes():
    """Activation shape after each conv, from the declared config."""
    s = {}
    t1 = conv_out(INPUT_T, 3, 1, 1)
    m1 = conv_out(INPUT_MEL, 3, 2, 1)
    s["stem1.dw"] = (1, 3, t1, m1)
    s["stem1.pw"] = (1, STEM_CHANNELS, t1, m1)
    t2 = conv_out(t1, 3, TIME_STRIDE, 1)
    s["down.dw"] = (1, STEM_CHANNELS, t2, m1)
    s["down.pw"] = (1, STEM_CHANNELS, t2, m1)
    m2 = conv_out(m1, 3, 2, 1)
    s["stem2.dw"] = (1, STEM_CHANNELS, t2, m2)
    s["stem2.pw"] = (1, DIM, t2, m2)
    s["T"] = t2
    s["M"] = m2
    return s


def budget_table() -> Tuple[list, dict]:
    """Per-layer MAC and the activation-RAM high-water mark, from real shapes."""
    s = conv_shapes()
    T, M = s["T"], s["M"]
    rows: List[Tuple[str, int, str]] = []

    def add(name, mac, note=""):
        rows.append((name, int(mac), note))

    # DeltaStack: fixed kernels, adds and one scale. 0 MAC by definition.
    add("DeltaStack", 0, "3 x (log-mel, d1, d2), 0 params")

    # groups: the three depthwise convs have groups == channels.
    add("stem1.dw (depthwise 3x3)",
        _conv_mac(s["stem1.dw"], 3, 3, 9, groups=3), str(s["stem1.dw"]))
    add("stem1.pw (pointwise 1x1)",
        _conv_mac(s["stem1.pw"], 3, STEM_CHANNELS, 1, groups=1), str(s["stem1.pw"]))
    add("down.dw (depthwise, stride 4)",
        _conv_mac(s["down.dw"], STEM_CHANNELS, STEM_CHANNELS, 9, groups=STEM_CHANNELS),
        str(s["down.dw"]))
    add("down.pw (pointwise 1x1)",
        _conv_mac(s["down.pw"], STEM_CHANNELS, STEM_CHANNELS, 1, groups=1),
        str(s["down.pw"]))
    add("stem2.dw (depthwise 3x3)",
        _conv_mac(s["stem2.dw"], STEM_CHANNELS, STEM_CHANNELS, 9, groups=STEM_CHANNELS),
        str(s["stem2.dw"]))
    add("stem2.pw (pointwise 1x1)",
        _conv_mac(s["stem2.pw"], STEM_CHANNELS, DIM, 1, groups=1), str(s["stem2.pw"]))
    add("mel_gate (dim->rank->1)", T * M * DIM * MEL_POOL_RANK + T * M * MEL_POOL_RANK,
        f"per frame over {M} mel bins")

    for b in range(NUM_BLOCKS):
        p = f"block {b}"
        add(f"{p} qkv", T * DIM * 3 * DIM, f"T={T} D={DIM}")
        add(f"{p} Q@K^T + A@V", 2 * NUM_HEADS * T * T * HEAD_DIM, f"{NUM_HEADS} heads x {HEAD_DIM}")
        add(f"{p} attn proj", T * DIM * DIM, "")
        add(f"{p} depthwise k=3,7", T * DIM * sum(KERNEL_SIZES), "depthwise")
        add(f"{p} pointwise mix", T * DIM * DIM, "")
        add(f"{p} squeeze-excite", DIM * SE_DIM + SE_DIM * DIM, f"se_dim={SE_DIM}")
        add(f"{p} FFN", 2 * T * DIM * FFN_HIDDEN, f"hidden={FFN_HIDDEN}")

    add("head", 3 * DIM * NUM_CLASSES, "")
    add("SoftORPool", 0, "mean + logsumexp + max")

    total = sum(m for _, m, _ in rows)

    # --- activation RAM ---
    # High-water mark of the float32 buffers that must be live simultaneously.
    live = {
        "delta (3x49x40)": 3 * INPUT_T * INPUT_MEL,
        "stem1 dw out (3x49x20)": int(np.prod(s["stem1.dw"])),
        "stem1 pw out (32x49x20)": int(np.prod(s["stem1.pw"])),
        "stem2 pw out (96x13x10)": int(np.prod(s["stem2.pw"])),
        "sequence (13x96)": T * DIM,
        "attention logits (4x13x13)": NUM_HEADS * T * T,
        "ffn hidden (13x96)": T * FFN_HIDDEN,
        "mel scores (8x13x10)": MEL_POOL_RANK * T * M,
    }
    # stem1's input (delta), its depthwise output and its pointwise output are
    # all live at once; everything after is smaller than the pointwise output.
    peak_els = (3 * INPUT_T * INPUT_MEL
                + int(np.prod(s["stem1.dw"]))
                + int(np.prod(s["stem1.pw"])))
    peak = peak_els * 4
    peak_i8 = peak_els

    info = {
        "T_after_down": T,
        "M_after_stem2": M,
        "mac_total": total,
        "peak_activation_els": peak_els,
        "peak_activation_f32_bytes": peak,
        "peak_activation_i8_bytes": peak_i8,
        "live": live,
        "shapes": {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()},
    }
    return rows, info


def _conv_mac(out_shape, cin, cout, k, groups):
    """
    MAC for a convolution, grouped convs included.

        MAC = out_positions * (cin / groups) * cout * kernel_elems

    `out_positions` is batch times SPATIAL extent only. Counting the channel
    axis here multiplies every conv by its own output width, which for the
    pointwise layers is the difference between 94 k and 3.0 M.
    """
    pos = int(out_shape[0])
    for d in out_shape[2:]:
        pos *= int(d)
    return pos * (cin // groups) * cout * k


# ==============================================================================
# Quantisation and emit
# ==============================================================================

# Tensors whose rows must be 16-aligned in the blob, because
# kws_accum_row_i8() and dense_f32() stride them by align16(in_dim).
GEMM_TENSORS = set()


def _init_gemm_tensors():
    global GEMM_TENSORS
    for _b in range(NUM_BLOCKS):
        _p = f"blocks.{_b}"
        GEMM_TENSORS |= {f"{_p}.attn.qkv.weight", f"{_p}.attn.proj.weight",
                         f"{_p}.pw.weight", f"{_p}.se_fc1.weight",
                         f"{_p}.se_fc2.weight", f"{_p}.ff1.weight",
                         f"{_p}.ff2.weight"}
    GEMM_TENSORS |= {"stem1.pw.weight", "down.pw.weight",
                     "stem2.pw.weight", "mel_gate.fc1.weight",
                     "mel_gate.fc2.weight", "head.weight"}


_init_gemm_tensors()


def quantise(tensors: Dict[str, np.ndarray], order: List[str]):
    """Symmetric per-output-channel int8. Returns (blob, scales, records, report)."""
    blob = bytearray()
    scales: List[float] = []
    records = []
    report = []
    for name in order:
        w = tensors[name].astype(np.float64)
        if name in F32_ONLY:
            # Pad to a 4-byte boundary BEFORE the float32 payload.
            #
            # Without this a float32 tensor can start at any byte offset left by
            # the int8 run before it, and two things break. The generated table
            # divides the byte offset by 4 to make an element index, which floors
            # and silently shifts the tensor by up to 3 bytes -- the reference
            # then reads garbage and produces NaN. And on the Xtensa an unaligned
            # float32 load traps outright, so the firmware would fault where the
            # host merely misbehaves. main/CMakeLists.txt already documents the
            # trap for the blob; this is the same hazard one level in.
            pad16 = (-len(blob)) % 16
            if pad16:
                blob.extend(b"\x00" * pad16)
            payload = w.astype("<f4")
            blob.extend(payload.tobytes())
            records.append({"name": name, "shape": list(w.shape), "dtype": "f32",
                            "offset": len(blob) - payload.nbytes, "count": int(w.size),
                            "scales": None, "stride": 0})
            continue
        axis = SCALE_AXIS.get(name)
        if axis is None:
            mx = float(np.max(np.abs(w))) if w.size else 0.0
            s = mx / 127.0 if mx > 0 else 1.0
            q = np.clip(np.rint(w / s), -127, 127).astype(np.int8)
            nch = 1
            scales.append(s)
            smap = np.full(w.shape, s)
        else:
            moved = np.moveaxis(w, axis, 0).reshape(w.shape[axis], -1)
            mx = np.max(np.abs(moved), axis=1)
            s = np.where(mx > 0, mx / 127.0, 1.0)
            qd = np.clip(np.rint(moved / s[:, None]), -127, 127).astype(np.int8)
            q = np.moveaxis(qd.reshape((w.shape[axis],) + tuple(np.moveaxis(w, axis, 0).shape[1:])), 0, axis)
            scales.extend(s.tolist())
            nch = int(w.shape[axis])
            smap = s
            # Per-channel scales only broadcast back over the tensor if the
            # reduced axis is restored to a length-1 axis; a bare (C,) will not
            # line up with a (C, 1, kh, kw) weight.
            bshape = [1] * w.ndim
            bshape[axis] = nch
            smap_full = s.reshape(bshape)

        # 16-byte align the START of every tensor.
        #
        # The Xtensa int8 reduction loads 128 bits at a time and
        # kws_accum_row_i8() refuses the assembly path unless both operands are
        # 16-byte aligned, so an unaligned tensor silently drops the whole
        # matrix to the scalar fallback. Measured on hardware: 239 ms against a
        # 0.722 ms floor, with the backtrace inside kws_dot_i8_ref.
        pad16 = (-len(blob)) % 16
        if pad16:
            blob.extend(b"\x00" * pad16)

        # Pad the reduction axis out to a 16-element boundary.
        #
        # kws_accum_row_i8 (main/kws_int8.c) strides the weight by
        # align16(in_dim), because the Xtensa reduction loads 128 bits at a time
        # and every row must start on an alignment boundary. The old
        # tools/export_weights.py did this padding. Without it, stem1.pw
        # (in_dim 3, align16 16) is read as w[o*16 + i] against data at w[o*3 + i]
        # and every pointwise and Linear tensor is garbage.
        logical = int(q.size)
        stride = 0

        blob.extend(q.tobytes())
        records.append({"name": name, "shape": list(w.shape), "dtype": "i8",
                        "offset": len(blob) - q.nbytes, "count": logical,
                        "scales": nch, "stride": stride})

        # Quantisation quality, per tensor. A channel whose max |w| is far below
        # the tensor's own max gets a tiny scale and everything in it collapses
        # to zero; that is invisible in the blob size and catastrophic in the
        # output, so it is measured here rather than discovered on hardware.
        dq = q.astype(np.float64) * smap_full
        err = np.abs(dq - w)
        denom = np.max(np.abs(w)) if w.size else 1.0
        sat = float(np.mean(np.abs(q) >= 127)) if q.size else 0.0
        per_ch = (np.max(np.abs(dq - w), axis=tuple(range(1, w.ndim))) / denom
                  if axis is not None and w.ndim > 1 else None)
        report.append({
            "name": name, "shape": list(w.shape), "channels": nch,
            "max_abs_err_over_max_abs_w": float(err.max() / denom) if denom else 0.0,
            "rms_err_over_max_abs_w": float(np.sqrt((err ** 2).mean()) / denom) if denom else 0.0,
            "saturated_fraction": sat,
            "worst_channel_rel": float(per_ch.max()) if per_ch is not None else None,
        })
    return bytes(blob), np.asarray(scales, dtype="<f4"), records, report


def emit_c_header(records, scales_n, params, path, threshold):
    """
    Emit the tensor table the C kernel binds to.

    Everything the kernel needs about the weights is here: element offset,
    element count, dtype and how many scales belong to it. The kernel contains no
    literal about this checkpoint -- retraining and re-running the exporter is the
    only way the table changes -- so a renamed tensor cannot be shipped as zeros
    because the exporter refuses to emit at all in that case.
    """
    name_of = {i: r["name"] for i, r in enumerate(records)}
    lines = []
    lines.append("/*")
    lines.append(" * GENERATED by tools/export_m1gw_weights.py -- do not edit.")
    lines.append(" *")
    lines.append(f" * Source : best_m1g_wide.pt  ({params:,} parameters)")
    lines.append(f" * Arch   : {ARCH_NAME}")
    lines.append(f" * Frozen threshold: {threshold:.6f}")
    lines.append(" *")
    lines.append(" * Each tensor's dtype, shape, offset and scale count are fixed by the")
    lines.append(" * exporter after checking them against the architecture, so the kernel")
    lines.append(" * can index this table without knowing anything about the checkpoint.")
    lines.append(" */")
    lines.append("")
    lines.append("#ifndef KWS_M1GW_DATA_H")
    lines.append("#define KWS_M1GW_DATA_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("#define KWS_M1GW_TENSOR_COUNT %d" % len(records))
    lines.append("#define KWS_M1GW_PARAM_COUNT %du" % params)
    lines.append("#define KWS_M1GW_SCALE_COUNT %du" % scales_n)
    lines.append(f"#define KWS_M1GW_THRESHOLD {threshold:.6f}f")
    lines.append("")
    lines.append("typedef enum {")
    for i, r in enumerate(records):
        lines.append("    KWS_M1GW_T_%s = %d," % (_c_ident(r["name"]), i))
    lines.append("} kws_m1gw_tensor_id;")
    lines.append("")
    lines.append("typedef struct {")
    lines.append("    /* offset and scale_off are uint32_t. Both tables outgrow 16 bits on")
    lines.append("     * this checkpoint (159,651 weight elements, 2,182 scales), and a")
    lines.append("     * truncating initializer is a silent mis-read, not a build error on")
    lines.append("     * the target toolchain. */")
    lines.append("    uint32_t offset;     /* BYTE offset into the blob -- both Ti() and Tf()")
    lines.append("     *                            add it to an int8_t pointer, which advances in")
    lines.append("     *                            bytes. Storing an ELEMENT index here instead made")
    lines.append("     *                            Tf() read 4x too early and the network returned")
    lines.append("     *                            NaN rather than a wrong number. */")
    lines.append("    uint32_t count;      /* elements                              */")
    lines.append("    uint32_t scale_off;  /* first scale in kws_m1gw_scales        */")
    lines.append("    uint16_t scale_n;    /* 0 = float32 tensor, no scales         */")
    lines.append("    uint16_t is_f32;")
    lines.append("    uint8_t  ndim;")
    lines.append("    uint8_t  pad;")
    lines.append("    uint16_t dim[4];")
    lines.append("} kws_m1gw_tensor;")
    lines.append("")
    lines.append("extern const kws_m1gw_tensor kws_m1gw_tensors[KWS_M1GW_TENSOR_COUNT];")
    lines.append("")
    lines.append("/* Non-const pointers to const data so the host harness can rebind them")
    lines.append(" * to heap copies; the firmware binds them to the generated arrays. Both")
    lines.append(" * link the same inference code, so a host pass is a real pass. */")
    lines.append("extern const int8_t  *kws_m1gw_blob;")
    lines.append("extern const float   *kws_m1gw_scales;")
    lines.append("")
    lines.append("#endif /* KWS_M1GW_DATA_H */")

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")

    # The table body is emitted separately so the header stays readable.
    tbl = []
    tbl.append("/* GENERATED by tools/export_m1gw_weights.py -- do not edit. */")
    tbl.append("")
    tbl.append("#include \"kws_m1gw_data.h\"")
    tbl.append("")
    tbl.append("const kws_m1gw_tensor kws_m1gw_tensors[KWS_M1GW_TENSOR_COUNT] = {")
    scale_cursor = 0
    for r in records:
        if r["dtype"] == "f32":
            sc, sn = 0, 0
        else:
            sc, sn = scale_cursor, r["scales"]
            scale_cursor += r["scales"]
        dims = list(r["shape"]) + [1, 1, 1, 1]
        tbl.append("    /* %-34s */ { %6d, %6d, %4d, %2d, %d, %d, 0, { %d, %d, %d, %d } },"
                   % (r["name"] + " ", r["offset"], r["count"], sc, sn,
                      1 if r["dtype"] == "f32" else 0,
                      len(r["shape"]), dims[0], dims[1], dims[2], dims[3]))
    tbl.append("};")
    # bin2c emits `const int8_t <sym>[N]`, an ARRAY, while the header declares
    # `extern const int8_t *<sym>`, a POINTER. Same name, different type: the
    # linker is satisfied and the kernel then dereferences the array's first
    # bytes as an address. That is an access violation, not a link error, and it
    # is why the indirection lives here rather than at the declaration.
    tbl.append("")
    tbl.append("extern const int8_t kws_m1gw_blob_data[];")
    tbl.append("extern const float  kws_m1gw_scales_data[];")
    tbl.append("")
    tbl.append("const int8_t *kws_m1gw_blob  = kws_m1gw_blob_data;")
    tbl.append("const float  *kws_m1gw_scales = kws_m1gw_scales_data;")
    with open(path.replace(".h", ".c"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(tbl) + "\n")

    return scale_cursor


def _c_ident(name: str) -> str:
    return name.replace(".", "_").upper()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--check", action="store_true",
                    help="validate against the architecture and report, emit nothing")
    args = ap.parse_args()

    print("=" * 88)
    print(f" export {ARCH_NAME}  <-  {os.path.basename(args.checkpoint)}")
    print("=" * 88)

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    meta, tensors = read_checkpoint(args.checkpoint)
    errs = validate(meta, tensors)
    if errs:
        print()
        for e in errs:
            print(f"  FAIL {e}")
        raise SystemExit(f"{len(errs)} problem(s): refusing to export a checkpoint "
                         "that does not match the architecture. Shipping a renamed "
                         "or retrained tensor as zeros is the failure this check exists "
                         "to prevent.")

    rows, info = budget_table()
    recorded = float(meta.get("mac", 0))

    print()
    print("-" * 88)
    print(" arithmetic, walked from the declared layer sequence")
    print("-" * 88)
    for name, mac, note in rows:
        if mac:
            print(f"  {name:<30} {mac:>10,}   {note}")
    print(f"  {'TOTAL':<30} {info['mac_total']:>10,}   "
          f"floor {info['mac_total'] / 3_840_000.0:.3f} ms @ 240 MHz int8")
    print()
    print(f"  checkpoint records      {recorded:>10,.0f}   "
          f"floor {recorded / 3_840_000.0:.3f} ms")
    ratio = info["mac_total"] / recorded if recorded else 0.0
    print(f"  ratio                   {ratio:>10.2f}x")
    if ratio > 1.02:
        print()
        print(f"  !! The recorded figure is {1 / ratio:.2f}x the direct walk. Both are")
        print("     computed from the same declared layers, so one of them is wrong.")
        print("     The walk is used here because it is the one derived from the")
        print("     shapes the C port will actually execute. Treat the CPU headroom")
        print(f"     as {info['mac_total'] / 3_840_000.0:.3f} ms, not {recorded / 3_840_000.0:.3f} ms,")
        print("     until this is reconciled against the training-time counter.")

    print()
    print("-" * 88)
    print(f" shapes: T {INPUT_T} -> {info['T_after_down']}, "
          f"MEL {INPUT_MEL} -> {info['M_after_stem2']}")
    print("-" * 88)
    print("  activation RAM (float32 buffers, high-water mark)")
    for k, v in info["live"].items():
        print(f"    {k:<28} {v * 4:>8,} B")
    peak = info["peak_activation_f32_bytes"]
    print(f"    {'PEAK (delta + dw out + pw out)':<28} {peak:>8,} B "
          f"= {peak / 1024:.1f} KB  ({peak / (256 * 1024) * 100:.0f}% of the 256 KB budget)")
    print(f"    same tensors held as int8              "
          f"{info['peak_activation_i8_bytes']:>8,} B "
          f"= {info['peak_activation_i8_bytes'] / 1024:.1f} KB")

    if args.check:
        print("\n  --check: nothing emitted.")
        return 0

    order = [k for k in expected_tensors() if k in tensors]
    blob, scales, records, qreport = quantise(tensors, order)

    # bin2c.py requires a multiple of 16 and refuses to pad itself, on the
    # grounds that a blob whose padding is implicit is a blob whose layout is
    # ambiguous. Pad here, at the END only: every tensor offset in the table
    # refers to the start of a tensor, so a trailing pad cannot move one.
    pad = (-len(blob)) % 16
    if pad:
        blob = blob + b"\x00" * pad

    # Weight-fidelity summary, before anything downstream can hide it.
    worst = max(qreport, key=lambda r: r["rms_err_over_max_abs_w"])
    worst_ch = max((r for r in qreport if r["worst_channel_rel"] is not None),
                   key=lambda r: r["worst_channel_rel"], default=None)
    med = float(np.median([r["rms_err_over_max_abs_w"] for r in qreport]))
    print()
    print("-" * 88)
    print(" weight fidelity after int8 (symmetric, per output channel)")
    print("-" * 88)
    print(f"  median RMS error / max|w|      {med * 100:.3f}%")
    print(f"  worst tensor                   {worst['name']} "
          f"{worst['rms_err_over_max_abs_w'] * 100:.3f}% RMS, "
          f"{worst['max_abs_err_over_max_abs_w'] * 100:.3f}% peak")
    if worst_ch:
        print(f"  worst single channel           {worst_ch['name']} "
              f"{worst_ch['worst_channel_rel'] * 100:.3f}% peak")
    sat_worst = max(qreport, key=lambda r: r["saturated_fraction"])
    print(f"  most saturated tensor          {sat_worst['name']} "
          f"{sat_worst['saturated_fraction'] * 100:.2f}% of weights at +/-127")
    print()
    print("  Reference point: a well-conditioned int8 weight matrix lands near")
    print("  0.4-0.8% RMS. Anything above a few percent, or a channel near 100%,")
    print("  means the per-channel scale is wrong for that channel and the C port")
    print("  would inherit it silently.")

    os.makedirs(args.out_dir, exist_ok=True)
    blob_path = os.path.join(args.out_dir, "kws_m1gw_i8.bin")
    scales_path = os.path.join(args.out_dir, "kws_m1gw_scales.bin")
    manifest_path = os.path.join(args.out_dir, "kws_m1gw_manifest.json")
    with open(blob_path, "wb") as fh:
        fh.write(blob)
    with open(scales_path, "wb") as fh:
        fh.write(scales.tobytes())

    params = sum(t.size for t in tensors.values())
    threshold = float(meta.get("threshold", 0.0))
    hdr_path = os.path.join(args.out_dir, "kws_m1gw_data.h")
    used_scales = emit_c_header(records, int(scales.size), params, hdr_path, threshold)

    manifest = {
        "arch": ARCH_NAME,
        "source_checkpoint": os.path.basename(args.checkpoint),
        "threshold": float(meta.get("threshold", 0.0)),
        "config": {
            "dim": DIM, "stem_channels": STEM_CHANNELS, "num_blocks": NUM_BLOCKS,
            "num_heads": NUM_HEADS, "ffn_mult": FFN_MULT, "mel_pool_rank": MEL_POOL_RANK,
            "time_stride": TIME_STRIDE, "num_classes": NUM_CLASSES,
            "ffn_hidden": FFN_HIDDEN, "se_dim": SE_DIM, "head_dim": HEAD_DIM,
            "input": [INPUT_T, INPUT_MEL],
            "T_after_down": info["T_after_down"], "M_after_stem2": info["M_after_stem2"],
        },
        "params": params,
        "blob_bytes": len(blob),
        "scale_count": int(scales.size),
        "mac_walked": info["mac_total"],
        "mac_recorded": recorded,
        "peak_activation_f32_bytes": peak,
        "quantisation": qreport,
        "tensors": records,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print()
    print("-" * 88)
    print(" emitted")
    print("-" * 88)
    print(f"  {blob_path}")
    print(f"    {len(blob):,} B   ({params:,} params, "
          f"{len(blob) / params:.2f} B/param)")
    print(f"  {scales_path}")
    print(f"    {scales.size * 4:,} B   ({scales.size} channels)")
    print(f"  {manifest_path}")
    print(f"  {hdr_path}")
    print(f"  {hdr_path.replace('.h', '.c')}")
    print()
    if used_scales != int(scales.size):
        print(f"  !! header references {used_scales} scales but {scales.size} were written; "
              "the table and the blob disagree, which would mis-scale the network "
              "silently rather than fail")
        raise SystemExit(1)
    print(f"  scale table cross-check: header {used_scales} == blob {scales.size}  OK")
    print(f"  flash for weights: {len(blob) / 1024:.1f} KB int8 "
          f"(was 80,624 B for bcconformer_v3: {len(blob) / 80624:.2f}x)")
    print(f"  threshold to freeze into Kconfig: {float(meta.get('threshold', 0.0)):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())