"""
Device-side latency analyzer
---------------------------
Consumes the `LAT ...` line the firmware logs on every detection, plus the WAV
that same detection saved, and reports the device half of the metric:

    keyword end -> detection -> clip start

Everything here is derived from anchors the device can actually observe. The
keyword's acoustic end is *not* one of them -- on hardware it is an event inside
a 1 s analysis window, and the firmware can only bracket it. So the analyzer
narrows the bracket with an energy-based endpoint detector run over the saved
audio, and reports how confident that resolution is instead of presenting the
result as exact.

What this cannot measure is the cloud term. That needs the streaming path, which
does not exist in firmware yet; for it see `esp32_emulator.py`, which measures
the whole chain against synthesized audio with a known keyword end.

Usage
-----
    idf.py monitor | python analyze_device_latency.py --slot recorded/slot03.wav
    python analyze_device_latency.py --log capture.log --slot recorded/slot03.wav

`--log` may be a monitor transcript or a saved one. Without `--slot` the
analyzer reports the brackets only, which is still worth having: a bracket wider
than the acceptable latency is a finding on its own.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

SAMPLE_RATE = 16000

LAT_RE = re.compile(
    r"LAT\s+t_us=(?P<t_us>-?\d+)\s+"
    r"sample=(?P<sample>-?\d+)\s+"
    r"kw_end_lo=(?P<lo>-?\d+)\s+"
    r"kw_end_hi=(?P<hi>-?\d+)\s+"
    r"clip_start=(?P<clip_start>-?\d+)\s+"
    r"infer_us=(?P<infer_us>-?\d+)\s+"
    r"drift_us=(?P<drift_us>-?\d+)"
)

SAVED_RE = re.compile(r"saved\s+(?P<sec>[\d.]+)\s*s clip -> slot\s+(?P<slot>\d+)")

DRIFT_RE = re.compile(r"sample clock is\s+(?P<drift>-?\d+)\s+ms behind")
LISTEN_RE = re.compile(r"listening\s+p=[\d.]+\s+hold=[\d.]+\s+infer=(?P<infer>\d+)\s+ms"
                       r"\s+hop=(?P<hop>\d+)\s+ms\s+duty=[\d.]+%\s+drift=(?P<drift>-?\d+)\s+ms")


@dataclass
class Detection:
    t_us: int
    sample: int
    kw_end_lo: int
    kw_end_hi: int
    clip_start: int
    infer_us: int
    drift_us: int

    # Filled in once the saved audio has been analysed.
    kw_end: Optional[int] = None
    kw_end_method: str = "unresolved"
    slot: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    @property
    def bracket_ms(self) -> float:
        return (self.kw_end_hi - self.kw_end_lo) * 1000.0 / SAMPLE_RATE

    @property
    def detect_to_clip_ms(self) -> float:
        """Detection -> first sample of the saved clip. Negative means the clip
        starts before the verdict, which is what the pre-roll is for."""
        return (self.clip_start - self.sample) * 1000.0 / SAMPLE_RATE


def parse_log(text: str) -> tuple[List[Detection], List[dict]]:
    detections: List[Detection] = []
    for line in text.splitlines():
        m = LAT_RE.search(line)
        if m:
            detections.append(Detection(
                t_us=int(m.group("t_us")),
                sample=int(m.group("sample")),
                kw_end_lo=int(m.group("lo")),
                kw_end_hi=int(m.group("hi")),
                clip_start=int(m.group("clip_start")),
                infer_us=int(m.group("infer_us")),
                drift_us=int(m.group("drift_us")),
            ))

    drift_warnings = [int(m.group("drift")) for m in DRIFT_RE.finditer(text)]
    listening = []
    for m in LISTEN_RE.finditer(text):
        listening.append({"infer_ms": int(m.group("infer")),
                          "hop_ms": int(m.group("hop")),
                          "drift_ms": int(m.group("drift"))})

    slots = {int(m.group("slot")) for m in SAVED_RE.finditer(text)}
    return detections, {"drift_warnings": drift_warnings, "listening": listening,
                        "saved_slots": sorted(slots)}


def find_keyword_end(pcm, lo_sample: int, clip_start: int, rel: float = 0.06,
                     smooth_ms: float = 20.0) -> tuple[Optional[int], str, List[str]]:
    """
    Locate the end of speech inside the saved clip by frame energy.

    Returns (absolute sample index, method, notes).

    The bracket from the firmware says the keyword ended somewhere in a 1 s
    window that ends at the first above-threshold evaluation. Inside the saved
    clip the speech is followed by silence, so the boundary is a step in short-
    term energy: find the last frame inside the bracket whose energy is still a
    meaningful fraction of the peak, and take its end.

    The honest caveat is in the notes, not buried here: this finds the end of
    *loud speech*, which for a keyword with a quiet final consonant or a room
    noise floor above the threshold is the end of the audible energy, not the
    end of the phoneme. It is reported as a bracket of its own whenever the
    energy near the decision is not a clean step.
    """
    notes: List[str] = []
    if len(pcm) == 0:
        return None, "no audio", ["saved clip is empty"]

    x = pcm.astype("float64")
    frame = max(1, int(SAMPLE_RATE * smooth_ms / 1000.0))
    hop = frame
    n = (len(x) - frame) // hop + 1
    if n <= 0:
        return None, "clip too short", ["clip shorter than one energy frame"]

    frames = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop][:n]
    energy = np.sqrt((frames ** 2).mean(axis=1))

    peak = float(energy.max())
    if peak <= 0.0:
        return None, "silence", ["clip contains no energy at all"]
    floor = float(np.percentile(energy, 10))
    if floor > peak * 0.5:
        return None, "no step", [f"energy floor {floor:.1f} is {floor / peak:.0%} of "
                                 f"peak {peak:.1f}: continuous background, no endpoint"]

    thr = floor + (peak - floor) * rel

    # Restrict to the bracket, but allow a margin: the bracket edge can sit a
    # frame or two either side of the true boundary.
    lo_idx = max(0, (lo_sample - clip_start) // hop - 2)
    hi_idx = min(len(energy), (lo_sample - clip_start) // hop
                 + int(SAMPLE_RATE * 1.1 / hop) + 2)
    if hi_idx <= lo_idx:
        notes.append("bracket fell outside the saved clip")
        return None, "bracket outside clip", notes

    window = energy[lo_idx:hi_idx]
    loud = np.flatnonzero(window >= thr)
    if len(loud) == 0:
        return None, "below threshold", notes + [
            "no frame inside the firmware's bracket exceeded the energy threshold; "
            "the bracket or the clip start do not line up"]
    last = lo_idx + loud[-1]
    end = (last + 1) * hop

    # Confidence: is the boundary a step, or a ramp?
    before = float(energy[max(0, last - 1)])
    after_idx = min(len(energy) - 1, last + 3)
    after = float(energy[after_idx])
    ratio = after / max(before, 1e-9)
    method = "energy step"
    if ratio > 0.5:
        method = "energy ramp (low confidence)"
        notes.append(f"energy falls only to {ratio:.0%} of the last loud frame "
                     f"{hop * 3} frames after the cut: the endpoint is a ramp, "
                     "not a step, so this is an upper bound on where speech ends")
    elif ratio > 0.15:
        method = "energy decay"
        notes.append(f"energy decays to {ratio:.0%} after the cut: plausible but "
                     "not a clean boundary")

    return clip_start + end, method, notes


def analyse(args) -> int:
    text = open(args.log, encoding="utf-8", errors="replace").read()
    detections, aux = parse_log(text)

    print("=" * 74)
    print(" device log")
    print("=" * 74)
    print(f"  detections in log : {len(detections)}")
    print(f"  clips saved       : {aux['saved_slots'] or 'none'}")
    if aux["drift_warnings"]:
        print(f"  !! device reported sample loss {len(aux['drift_warnings'])} time(s), "
              f"worst {max(aux['drift_warnings'])} ms behind the wall clock")
    if aux["listening"]:
        worst = max(aux["listening"], key=lambda r: r["drift_ms"])
        print(f"  steady state      : infer={worst['infer_ms']} ms, hop={worst['hop_ms']} ms, "
              f"drift={worst['drift_ms']} ms")

    if not detections:
        print("\n  No LAT lines: the firmware predates this instrumentation, or no")
        print("  detection fired. Both are worth knowing -- the second one is why")
        print("  there is no latency number to report.")
        return 2

    pcm = None
    clip_start = None
    if args.slot:
        import soundfile as sf
        pcm, sr = sf.read(args.slot, dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm[:, 0]
        if sr != SAMPLE_RATE:
            print(f"  {args.slot} is {sr} Hz, expected {SAMPLE_RATE}")
            return 2
        print(f"  clip analysed     : {args.slot} ({len(pcm) / SAMPLE_RATE:.2f} s)")

    print()
    print("=" * 74)
    print(" per-detection latency")
    print("=" * 74)

    records = []
    for i, d in enumerate(detections[-args.last:], start=1):
        slot = None
        if aux["saved_slots"]:
            slot = aux["saved_slots"][-1] if len(aux["saved_slots"]) == 1 else None

        if pcm is not None:
            # The clip's own sample 0 maps to this detection's clip_start.
            kw, method, notes = find_keyword_end(pcm, d.kw_end_lo, d.clip_start)
            d.kw_end = kw
            d.kw_end_method = method
            d.notes.extend(notes)
            d.slot = slot

        print(f"\n  [{i}] t_us={d.t_us}  sample={d.sample}  infer={d.infer_us / 1000.0:.0f} ms")
        print(f"      keyword end bracket : {d.kw_end_lo} .. {d.kw_end_hi} "
              f"({d.bracket_ms:.0f} ms wide)")
        if d.kw_end is not None:
            print(f"      keyword end resolved: sample {d.kw_end} ({d.kw_end_method})")
            edge = (d.sample - d.kw_end) * 1000.0 / SAMPLE_RATE
            print(f"      EDGE LATENCY        : {edge:.1f} ms  "
                  f"(keyword end -> detection)")
            if not (d.kw_end_lo <= d.kw_end <= d.kw_end_hi):
                print(f"      !! resolved end falls outside the firmware's bracket -- "
                      f"one of the two is wrong")
        else:
            print(f"      keyword end resolved: NO ({d.kw_end_method})")
        print(f"      detection -> clip   : {d.detect_to_clip_ms:.1f} ms "
              f"(negative = pre-roll)")
        print(f"      drift at detection  : {d.drift_us / 1000.0:.0f} ms behind the wall clock")
        for note in d.notes:
            print(f"      ! {note}")

        records.append({
            "t_us": d.t_us,
            "sample": d.sample,
            "kw_end_bracket": [d.kw_end_lo, d.kw_end_hi],
            "kw_end_resolved": d.kw_end,
            "kw_end_method": d.kw_end_method,
            "edge_latency_ms": None if d.kw_end is None
                              else round((d.sample - d.kw_end) * 1000.0 / SAMPLE_RATE, 1),
            "detect_to_clip_ms": round(d.detect_to_clip_ms, 1),
            "infer_ms": round(d.infer_us / 1000.0, 1),
            "drift_ms": round(d.drift_us / 1000.0, 1),
            "notes": d.notes,
        })

    resolved = [r["edge_latency_ms"] for r in records if r["edge_latency_ms"] is not None]
    if resolved:
        print()
        print("=" * 74)
        print(f" EDGE LATENCY over {len(resolved)} detection(s): "
              f"min {min(resolved):.1f} ms  median {sorted(resolved)[len(resolved) // 2]:.1f} ms  "
              f"max {max(resolved):.1f} ms")
        print("=" * 74)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"detections": records,
                       "drift_warnings": aux["drift_warnings"],
                       "listening": aux["listening"]}, fh, indent=2)
        print(f"\n[json] wrote {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Device-side keyword-end latency from a firmware log and its clip",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--log", required=True, help="idf.py monitor transcript")
    parser.add_argument("--slot", default=None,
                        help="WAV saved by the detection being analysed")
    parser.add_argument("--last", type=int, default=10, help="report only the last N")
    parser.add_argument("--out", default=None, help="write the records to this JSON file")
    return analyse(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())