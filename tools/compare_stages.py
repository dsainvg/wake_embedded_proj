"""
Compare the C forward pass against the JAX reference, stage by stage.

Prints, for each stage, the max absolute difference, the RMS difference and the
magnitude of the reference, so a structural bug (large, early) is obvious from
a rounding difference (tiny, late).

Usage:
    tools\\test_kws_host.bat
    python tools/compare_stages.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STAGES = os.path.join(ROOT, "main", "model", "reference", "stages")

NAMES = {1: "stem", 2: "conv2", 3: "gn3swish", 4: "seq",
         5: "block1", 6: "block2", 7: "block3", 8: "feat", 9: "logits"}


def read_npy(path):
    with open(path, "rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"{path}: not a .npy file")
        major = f.read(1)[0]
        f.read(1)                                    # minor
        # v1 has a uint16 header length, v2+ a uint32.
        if major == 1:
            hlen = int.from_bytes(f.read(2), "little")
        else:
            hlen = int.from_bytes(f.read(4), "little")
        hdr = f.read(hlen).decode("latin1")
        shape = ()
        if "'shape':" in hdr:
            body = hdr.split("'shape':")[1]
            body = body[body.index("(") + 1:body.index(")")]
            for tok in body.split(","):
                tok = tok.strip().rstrip(",").strip()
                if tok:
                    shape += (int(tok),)
        data = np.frombuffer(f.read(), dtype="<f4")
    return data.reshape(shape) if shape else data


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "noise"
    worst = None

    print(f"{'stage':<10}{'count':>8}{'ref |max|':>12}{'max abs':>12}"
          f"{'rms abs':>12}{'rel rms':>10}")
    print("-" * 64)

    for st in range(1, 10):
        npy = os.path.join(STAGES, f"{tag}_{st}_{NAMES[st]}.npy")
        bin_ = os.path.join(STAGES, f"c_{st}_{NAMES[st]}.bin")
        if not (os.path.exists(npy) and os.path.exists(bin_)):
            print(f"{NAMES[st]:<10}  (missing)")
            continue

        ref = read_npy(npy).ravel().astype(np.float64)
        got = np.fromfile(bin_, dtype="<f4").astype(np.float64)

        n = min(ref.size, got.size)
        if n == 0:
            print(f"{NAMES[st]:<10}  (empty)")
            continue
        ref, got = ref[:n], got[:n]

        d = np.abs(ref - got)
        rms = float(np.sqrt((d ** 2).mean()))
        refmag = float(np.abs(ref).max())
        rel = rms / max(refmag, 1e-12)
        print(f"{NAMES[st]:<10}{n:>8}{refmag:>12.4f}{d.max():>12.4e}"
              f"{rms:>12.4e}{rel:>10.2e}")

        if worst is None or rel > worst[1]:
            worst = (NAMES[st], rel)

    print("-" * 64)
    if worst is None:
        print("no stages compared")
        return 1
    print(f"worst relative error: {worst[0]} at {worst[1]:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())