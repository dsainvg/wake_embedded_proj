"""
ESP32-S3 firmware emulator for edge->cloud latency
--------------------------------------------------
Runs the *real* detector against audio whose keyword end is known exactly, and
streams the result to `server.py` as if it came off the board.

Why an emulator rather than a log parser
---------------------------------------
The metric is a delta between the instant the keyword ends and the instant the
cloud holds the audio that follows it. Both endpoints have to be known, and the
keyword end is the hard one: on hardware it is an acoustic event inside a 1 s
window, so it can only be located to within the hop, by a VAD, or by hand. Here
it is a sample index, because the emulator generates the audio, so the edge term
is exact rather than estimated and everything above it can be attributed.

The timing model is the firmware's, not an idealised one. `kws_task` reads a
hop, then blocks for the inference, then decides:

    t_decision(k) = max(t_ready(k), t_decision(k-1)) + infer

which is the whole loop. When `infer > hop` that recursion is the story: the
device falls behind real time, the gap between the audio it has consumed and
the audio the DMA ring has captured grows without bound, and once it exceeds the
ring's slack the oldest samples are overwritten before they are ever read. A
single inference shorter than the ring looks safe and the cumulative backlog
does not, which is why this is modelled rather than asserted -- see
`README.latency.md`.

Usage
-----
    python esp32_emulator.py                       # synthesize, stream, report
    python esp32_emulator.py --offline              # device-side terms only
    python esp32_emulator.py --wav clip.wav --kw-end-sec 3.4
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latency import build_breakdown  # noqa: E402

SAMPLE_RATE = 16000


# --------------------------------------------------------------------------
# Firmware timing model
# --------------------------------------------------------------------------


@dataclass
class LoopConfig:
    """
    Mirrors wake_word_main.c's macros. Defaults are the measured/derived values
    from that file: a 200 ms hop, the 334 ms inference measured on the S3, a
    1.024 s DMA ring (16 x 1024) and a 1 s pre-roll ring.
    """

    hop_ms: int = 200
    infer_ms: int = 334
    ring_slack_ms: int = 1024
    preroll_ms: int = 1000
    refractory_ms: int = 1500

    @property
    def hop_samples(self) -> int:
        return SAMPLE_RATE * self.hop_ms // 1000

    @property
    def ring_slack_samples(self) -> int:
        return SAMPLE_RATE * self.ring_slack_ms // 1000

    @property
    def preroll_samples(self) -> int:
        return SAMPLE_RATE * self.preroll_ms // 1000

    @property
    def refractory_steps(self) -> int:
        return max(1, self.refractory_ms // self.hop_ms)


@dataclass
class Hop:
    """One iteration of the firmware's loop."""

    index: int
    t_ready: float        # device time the hop's last sample leaves the mic
    t_read_done: float    # device time the read returns
    t_decision: float     # device time the detector's verdict exists
    first_sample: int     # absolute sample index of the first sample READ
    drop: int             # samples of this hop already overwritten when read
    lag_ms: float         # how far behind real time the read cursor is


class FirmwareLoop:
    """
    The read -> infer -> detect cycle, as a state machine over device time.

    Three details are load-bearing and all of them are easy to get wrong:

    * The read is what blocks, not the infer. `i2s_channel_read` returns only
      once the hop's worth of samples exists, so a hop that is ready before the
      previous inference finished is served immediately and costs nothing.
    * Once `infer > hop` the recurrence never catches up: each iteration costs
      `infer` of wall time and consumes only `hop` of audio, so the read cursor
      falls behind real time by `infer - hop` on every hop, forever. That is
      `lag`, and it grows without bound.
    * The read serves whatever the DMA ring still holds, and the ring is finite.
      `lag` is not the DMA backlog; the DMA backlog is `min(lag, ring_slack)`,
      because past that point the ring has already overwritten the oldest
      samples. `drop` is the difference -- the audio the device asked for and
      will never get.

    A single inference shorter than the ring looks perfectly safe, and that is
    exactly why this is modelled instead of asserted from the inference time:
    wake_word_main.c warns when one inference exceeds the ring, which cannot
    happen here, and stays silent while the cumulative lag walks straight past
    it.
    """

    def __init__(self, cfg: LoopConfig, t0: float) -> None:
        self.cfg = cfg
        self.t0 = t0                    # device time at which sample 0 was captured
        self._prev_decision = t0
        self._prev_read_done = t0
        self._read_pos = 0              # absolute index of the next unread sample
        self.samples_read = 0           # samples actually delivered to the model
        self.samples_dropped = 0
        self.max_lag_ms = 0.0
        self.first_drop_index: Optional[int] = None
        self._index = 0

    def next_hop(self) -> Optional[Hop]:
        cfg = self.cfg
        hop = cfg.hop_samples
        self._index += 1

        # The data this hop wants exists once the mic has delivered it, whether
        # or not the task is there to collect it.
        t_ready = self.t0 + (self._read_pos + hop) / SAMPLE_RATE
        t_read_done = max(t_ready, self._prev_decision)

        captured = int(round((t_read_done - self.t0) * SAMPLE_RATE))
        lag_samples = captured - self._read_pos
        drop = max(0, lag_samples - cfg.ring_slack_samples)
        if drop and self.first_drop_index is None:
            self.first_drop_index = self._index
        self.samples_dropped += drop

        lag_ms = (t_read_done - self.t0) - (self._read_pos / SAMPLE_RATE)
        self.max_lag_ms = max(self.max_lag_ms, lag_ms * 1000.0)

        # The read position jumps over anything the ring already overwrote. It
        # is not the same as "samples read plus samples dropped" once drops
        # start: the position is where the device *is*, the counters are what it
        # got and what it lost, and mixing them makes the lag collapse.
        first = self._read_pos + drop
        self._read_pos = first + hop
        self.samples_read += hop
        self._prev_read_done = t_read_done
        self._prev_decision = t_read_done + cfg.infer_ms / 1000.0

        return Hop(
            index=self._index,
            t_ready=t_ready,
            t_read_done=t_read_done,
            t_decision=self._prev_decision,
            first_sample=first,
            drop=drop,
            lag_ms=lag_ms * 1000.0,
        )

    def captured_samples(self) -> int:
        """Samples the mic delivered by the last read completion."""
        return int(round((self._prev_read_done - self.t0) * SAMPLE_RATE))

    def sustain_factor(self) -> float:
        """
        Audio seconds the model actually saw per wall second. 1.0 is real time.

        Counted from samples read, not from the read *position*: a device that
        is permanently skipping audio still advances its position at the nominal
        hop rate, and scoring that as real-time would report a device losing two
        thirds of the room's audio as keeping up perfectly.

        Measured against the last *read completion*, not the last verdict: the
        final inference is still in flight when the run ends, and counting it
        would make a device that exactly keeps up look like it runs 2% slow.
        """
        elapsed = self._prev_read_done - self.t0
        if elapsed <= 0:
            return float("inf")
        return (self.samples_read / SAMPLE_RATE) / elapsed


# --------------------------------------------------------------------------
# Audio with a known keyword end
# --------------------------------------------------------------------------


async def synth_keyword(voice: str, text: str) -> np.ndarray:
    """TTS the keyword so its first and last sample are both known exactly."""
    try:
        import edge_tts
    except ImportError as exc:
        raise SystemExit("edge-tts is not installed; pip install edge-tts, "
                         "or pass --wav with --kw-end-sec") from exc
    communicate = edge_tts.Communicate(text, voice)
    chunks = []
    async for item in communicate.stream():
        if item["type"] == "audio":
            chunks.append(item["data"])
    if not chunks:
        raise SystemExit("TTS returned no audio (no network?)")
    raw = b"".join(chunks)

    import io
    import soundfile as sf
    from scipy import signal

    data, sr = sf.read(io.BytesIO(raw))
    if data.ndim > 1:
        data = data[:, 0]
    if sr != SAMPLE_RATE:
        data = signal.resample(data, int(len(data) * SAMPLE_RATE / sr))
    return data.astype(np.float32)


def load_wav(path: str) -> np.ndarray:
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    if sr != SAMPLE_RATE:
        raise SystemExit(f"{path} is {sr} Hz; the pipeline is fixed at {SAMPLE_RATE} Hz")
    return data.astype(np.float32)


def trim_silence(x: np.ndarray, rel: float = 0.02) -> np.ndarray:
    """
    Strip the padding a TTS engine wraps around an utterance.

    The whole harness rests on `kw_end_sample` being the end of the *word*, so a
    boundary that lands in the synthesiser's trailing silence is not a weaker
    measurement, it is the wrong measurement: it would report a keyword ending
    at a time the room never heard anything at. The cut is at 2% of peak, which
    is below the level of any speech formant and above the noise floor of the
    synthesiser.
    """
    if len(x) == 0:
        return x
    peak = float(np.max(np.abs(x)))
    if peak <= 0.0:
        return x
    loud = np.flatnonzero(np.abs(x) >= peak * rel)
    if len(loud) == 0:
        return x
    return x[loud[0]:loud[-1] + 1]


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    freqs[0] = 1.0
    spec = np.fft.rfft(white) / np.sqrt(freqs)
    return np.fft.irfft(spec, n=n).astype(np.float32)


async def build_audio(args) -> tuple[np.ndarray, int, dict]:
    """Return (float32 audio, absolute keyword-end sample index, provenance)."""
    rng = np.random.default_rng(args.seed)

    if args.wav:
        audio = load_wav(args.wav)
        if args.kw_end_sec is None:
            raise SystemExit("--wav requires --kw-end-sec: on a recording the "
                             "keyword end is not known exactly, which is the whole "
                             "reason this harness synthesizes instead")
        kw_end = int(round(args.kw_end_sec * SAMPLE_RATE))
        if not 0 < kw_end < len(audio):
            raise SystemExit(f"--kw-end-sec is outside the clip ({len(audio) / SAMPLE_RATE:.2f} s long)")
        return audio, kw_end, {"source": "wav", "path": args.wav}

    print("[audio] synthesizing keyword with edge-tts ...", flush=True)
    keyword = trim_silence(await synth_keyword(args.voice, args.text))
    print(f"[audio] keyword: {len(keyword) / SAMPLE_RATE:.3f} s ({args.voice}), "
          f"silence trimmed", flush=True)

    lead_s = args.lead_sec
    command = trim_silence(await synth_keyword(args.voice, args.command))
    tail_s = max(args.tail_sec, len(command) / SAMPLE_RATE + 0.2)

    parts = [
        np.zeros(int(lead_s * SAMPLE_RATE), dtype=np.float32),
        keyword,
        np.zeros(int(round(tail_s * SAMPLE_RATE)) - len(command), dtype=np.float32),
        command,
    ]
    total = sum(len(p) for p in parts)
    audio = np.concatenate(parts) + pink_noise(total, rng) * args.noise_rms
    peak = float(np.max(np.abs(audio)))
    if peak > 0.99:
        audio = audio * (0.99 / peak)

    kw_start = int(lead_s * SAMPLE_RATE)
    kw_end = kw_start + len(keyword)
    return audio, kw_end, {
        "source": "edge-tts",
        "voice": args.voice,
        "text": args.text,
        "keyword_start_sample": kw_start,
        "keyword_end_sample": kw_end,
        "keyword_seconds": round(len(keyword) / SAMPLE_RATE, 4),
    }


# --------------------------------------------------------------------------
# Clock probe (two-step, NTP sense)
# --------------------------------------------------------------------------


class ProbeRunner:
    """
    Runs NTP-style exchanges over the same socket the audio uses.

    Each probe piggybacks the previous probe's receive time, so one round trip
    yields one complete four-timestamp sample and the server can reduce the
    window on its own. The device never computes an offset -- it has no second
    clock to compute one against.
    """

    def __init__(self, ws) -> None:
        self.ws = ws
        self.seq = 0
        self.prev_seq: Optional[int] = None
        self.prev_t4: Optional[float] = None

    async def run(self, count: int, interval_s: float = 0.02) -> None:
        for _ in range(count):
            t1 = time.monotonic()
            await self.ws.send(json.dumps({
                "event": "time_probe",
                "seq": self.seq,
                "t1": t1,
                "prev_seq": self.prev_seq,
                "prev_t4": self.prev_t4,
            }))
            raw = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
            t4 = time.monotonic()
            ack = json.loads(raw)
            if ack.get("event") != "time_probe_ack":
                raise SystemExit(f"unexpected reply to probe: {raw!r}")
            self.prev_seq, self.prev_t4 = self.seq, t4
            self.seq += 1
            await asyncio.sleep(interval_s)


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


async def run(args) -> int:
    from kws_engine import StreamingKWSEngine

    audio, kw_end_sample, provenance = await build_audio(args)
    total_samples = len(audio)
    audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

    cfg = LoopConfig(hop_ms=args.hop_ms, infer_ms=args.infer_ms,
                     ring_slack_ms=args.ring_slack_ms, preroll_ms=args.preroll_ms,
                     refractory_ms=args.refractory_ms)

    print()
    print("=" * 74)
    print(" firmware loop configuration")
    print("=" * 74)
    print(f"  hop                : {cfg.hop_ms} ms ({cfg.hop_samples} samples)")
    print(f"  inference (modelled): {cfg.infer_ms} ms")
    print(f"  DMA ring            : {cfg.ring_slack_ms} ms ({cfg.ring_slack_samples} samples)")
    print(f"  pre-roll            : {cfg.preroll_ms} ms ({cfg.preroll_samples} samples)")
    print(f"  sustain factor      : {min(cfg.hop_ms, cfg.infer_ms) / cfg.infer_ms:.3f}x real time"
          f"  (capped by max(hop, infer))")
    if cfg.infer_ms > cfg.hop_ms:
        rings = cfg.ring_slack_ms / (cfg.infer_ms - cfg.hop_ms)
        print(f"  -> inference exceeds the hop by {cfg.infer_ms - cfg.hop_ms} ms per hop, so the")
        print(f"     DMA backlog grows {cfg.infer_ms - cfg.hop_ms} ms/hop and overflows after")
        print(f"     ~{rings:.1f} hops ({rings * cfg.hop_ms / 1000.0:.2f} s of audio).")
    print()

    engine = StreamingKWSEngine(checkpoint_path=args.model, arch=args.arch,
                                 confidence_threshold=args.threshold)
    # The firmware derives its refractory from the hop in milliseconds; the
    # reference engine reads it from the checkpoint in 100 ms strides. Same
    # intent, different units, so pin it to what the firmware would do.
    engine.refractory_steps = cfg.refractory_steps

    # The device clock is absolute, like esp_timer's free-running counter since
    # boot. Keeping it absolute matters: t0 is the instant the DMA stream opened,
    # and FirmwareLoop stamps times relative to it, so every comparison between a
    # loop timestamp and a reading of the clock has to be on the same scale.
    t0 = time.monotonic()          # device time at which sample 0 was captured
    loop = FirmwareLoop(cfg, t0)

    def dev() -> float:
        return time.monotonic()

    ring: collections.deque = collections.deque(maxlen=cfg.preroll_samples)

    print("=" * 74)
    print(" streaming loop")
    print("=" * 74)

    ws = None
    if not args.offline:
        import websockets
        ws = await websockets.connect(args.uri)
        print(f"[ws] connected to {args.uri}")
        probes = ProbeRunner(ws)
        await asyncio.sleep(0.2)
        await probes.run(args.probes)
        print(f"[ws] {probes.seq} clock probes exchanged "
              f"(offset is estimated server-side from these)")

    detected_at: Optional[Hop] = None
    kw_end_dev = t0 + kw_end_sample / SAMPLE_RATE
    sent_end = False
    first_write_dev: Optional[float] = None
    stream_start_sample = -1
    preroll_sent = 0
    cursor = 0
    next_send_sample = 0

    while True:
        hop = loop.next_hop()
        if hop is None:
            break
        chunk = audio_i16[hop.first_sample:hop.first_sample + cfg.hop_samples]
        if args.verbose:
            print(f"  hop {hop.index:3d} first={hop.first_sample:7d} len={len(chunk):5d} "
                  f"cursor={cursor:7d}/{total_samples} lag={hop.lag_ms:6.0f}ms "
                  f"drop={hop.drop:6d} now={dev() - t0:.3f}s "
                  f"dec={hop.t_decision - t0:.3f}s", flush=True)
        if len(chunk) == 0:
            if cursor >= total_samples:
                break
            raise SystemExit(
                f"hop {hop.index} produced no samples but the cursor is still "
                f"behind ({cursor} of {total_samples}); the loop model is skipping "
                "audio the file does not contain")

        # Block until the task would actually be scheduled.
        now = dev()
        if hop.t_read_done > now:
            await asyncio.sleep(hop.t_read_done - now)

        ring.extend(chunk.tolist())
        cursor = hop.first_sample + len(chunk)

        spotted, score, _ = engine.push_audio_chunk(chunk.astype(np.float32) / 32768.0)

        # Block for the modelled inference, so the backlog the firmware would
        # have accumulated is actually accumulated here.
        now = dev()
        if hop.t_decision > now:
            await asyncio.sleep(hop.t_decision - now)

        if spotted and detected_at is None:
            detected_at = hop
            print(f"\n>>> DETECTED at hop {hop.index}: keyword ended at "
                  f"{kw_end_dev - t0:.3f} s into the stream, verdict at "
                  f"{hop.t_decision - t0:.3f} s, score {score:.3f}")
            print(f"    keyword end -> detection: "
                  f"{(hop.t_decision - kw_end_dev) * 1000.0:.1f} ms")

            if ws is not None:
                preroll = np.fromiter(ring, dtype=np.int16,
                                      count=min(len(ring), cfg.preroll_samples))
                stream_start_sample = cursor - len(preroll)
                preroll_sent = len(preroll)

                await ws.send(json.dumps({
                    "event": "wake_detected",
                    "keyword": args.text,
                    "t_kw_end": kw_end_dev,
                    "t_detect": hop.t_decision,
                    "keyword_end_sample": kw_end_sample,
                    "stream_start_sample": stream_start_sample,
                    "preroll_samples": preroll_sent,
                    "samples_dropped": loop.samples_dropped,
                }))
                # The pre-roll write can only start once the metadata that gates
                # it is on the wire, so its timestamp cannot travel in that
                # metadata. It goes out with end_of_speech instead.
                await ws.send(preroll.tobytes())
                first_write_dev = dev()
                next_send_sample = cursor
                print(f"    sent {preroll_sent} pre-roll samples "
                      f"(keyword end is {kw_end_sample - stream_start_sample} into it)")

        # Stream the audio that follows, one hop at a time, as the firmware's
        # capture task would hand it to the transport.
        if detected_at is not None and ws is not None and not sent_end:
            if next_send_sample < cursor:
                block = audio_i16[next_send_sample:cursor]
                if len(block):
                    await ws.send(block.tobytes())
                    next_send_sample += len(block)

        # One exit, not three: the audio is finite, and once it is consumed the
        # turn is over whether or not anything was detected. An earlier version
        # keyed this on `detected_at is None`, so a detection with no socket
        # attached -- the --offline mode -- never terminated.
        if cursor >= total_samples:
            if detected_at is not None and ws is not None and not sent_end:
                await ws.send(json.dumps({"event": "end_of_speech",
                                          "t_first_write": first_write_dev}))
                sent_end = True
            break

    captured = loop.captured_samples()
    read = loop.samples_read
    lost = loop.samples_dropped
    print()
    print(f"  stream length      : {total_samples / SAMPLE_RATE:.2f} s, "
          f"{cursor} samples reached the model")
    print(f"  wall clock         : {dev() - t0:.2f} s")
    print(f"  mic delivered      : {captured} samples "
          f"({read} read + {lost} lost, {lost / max(1, captured) * 100.0:.1f}% lost, "
          f"{captured - read - lost} still buffered in DMA)")
    print(f"  max read lag       : {loop.max_lag_ms:.0f} ms behind real time")
    if loop.first_drop_index is not None:
        print(f"  first loss         : hop {loop.first_drop_index}, "
              f"{loop.first_drop_index * cfg.hop_ms} ms into the stream")
    print(f"  sustain factor     : {loop.sustain_factor():.3f}x real time "
          f"(hop {cfg.hop_ms} ms / infer {cfg.infer_ms} ms)")

    device_report = build_breakdown(
        keyword_end=kw_end_dev,
        detect=detected_at.t_decision if detected_at else float("nan"),
        cloud_first_audio=float("nan"),
        first_byte_write=first_write_dev,
        keyword_end_sample=kw_end_sample,
        stream_start_sample=stream_start_sample,
        preroll_samples=preroll_sent,
        samples_dropped=loop.samples_dropped,
        warnings=["offline: cloud terms not measured"] if args.offline else [],
    )

    print()
    print("=" * 74)
    print(" device-side terms")
    print("=" * 74)
    if detected_at is None:
        print(f"  no detection: the model never cleared {args.threshold:.2f} in "
              f"{total_samples / SAMPLE_RATE:.1f} s.")
        print("  This is a recall result, not a latency result: there is no measurement")
        print("  to report because the turn never happened. Check the checkpoint's")
        print("  stored streaming recall, lower --threshold, or supply audio that fires.")
        if ws is not None:
            await ws.close()
        return 2

    print(f"  {'keyword end -> detection':<32} {device_report.edge_detect_ms:8.2f} ms")
    if device_report.detect_write_ms is not None:
        print(f"  {'detection -> pre-roll written':<32} {device_report.detect_write_ms:8.2f} ms")
    print(f"  {'DETECTION LATENCY (device total)':<32} "
          f"{device_report.edge_detect_ms:8.2f} ms")

    cloud_record = None
    if ws is not None and sent_end:
        raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
        response = json.loads(raw)
        cloud_record = response.get("latency")
        await ws.close()

        print()
        print("=" * 74)
        print(" cloud record (server-side, clock-corrected)")
        print("=" * 74)
        if cloud_record is None:
            print("  server returned no latency record")
        else:
            print(f"  {'keyword end -> detection':<32} {cloud_record['edge_detect_ms']:8.2f} ms")
            if cloud_record.get("detect_write_ms") is not None:
                print(f"  {'detection -> socket write':<32} {cloud_record['detect_write_ms']:8.2f} ms")
            if cloud_record.get("network_ms") is not None:
                print(f"  {'socket write -> cloud ingest':<32} {cloud_record['network_ms']:8.2f} ms"
                      f" +/-{cloud_record['uncertainty_ms']:.2f}")
            print(f"  {'TOTAL (keyword end -> cloud)':<32} {cloud_record['total_ms']:8.2f} ms"
                  f" +/-{cloud_record['uncertainty_ms']:.2f}")
            print(f"  clock: offset {cloud_record['offset'] * 1000.0:+.3f} ms, "
                  f"spread {cloud_record['theta_spread'] * 1000.0:.3f} ms, "
                  f"{cloud_record['probe_samples']} probes")
            problems = cloud_record.get("problems") or []
            print()
            if problems:
                for problem in problems:
                    print(f"  ! {problem}")
            else:
                print("  no coherence problems in this record")

    out = {
        "config": {"hop_ms": cfg.hop_ms, "infer_ms": cfg.infer_ms,
                   "ring_slack_ms": cfg.ring_slack_ms, "preroll_ms": cfg.preroll_ms,
                   "threshold": args.threshold, "arch": args.arch},
        "audio": provenance,
        "device": device_report.as_dict(),
        "cloud": cloud_record,
        "sustain": {
            "samples_consumed": cursor,
            "samples_mic_delivered": loop.captured_samples(),
            "samples_read": loop.samples_read,
            "samples_dropped": loop.samples_dropped,
            "max_read_lag_ms": round(loop.max_lag_ms, 1),
            "first_drop_hop": loop.first_drop_index,
            "sustain_factor": round(loop.sustain_factor(), 4),
        },
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n[json] wrote {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emulate the ESP32-S3 wake pipeline and measure edge->cloud latency",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--uri", default="ws://localhost:8000/ws/audio")
    parser.add_argument("--offline", action="store_true",
                        help="skip the socket; report device-side terms only")
    parser.add_argument("--model", default="best_v11_production.flax")
    parser.add_argument("--arch", default="bcconformer_v3")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the checkpoint's frozen operating point")
    parser.add_argument("--probes", type=int, default=16, help="clock probes before streaming")

    g = parser.add_argument_group("firmware loop (mirrors wake_word_main.c)")
    g.add_argument("--hop-ms", type=int, default=200)
    g.add_argument("--infer-ms", type=int, default=334,
                   help="modelled inference time; 334 ms is the measured S3 figure")
    g.add_argument("--ring-slack-ms", type=int, default=1024)
    g.add_argument("--preroll-ms", type=int, default=1000)
    g.add_argument("--refractory-ms", type=int, default=1500)

    g = parser.add_argument_group("audio")
    g.add_argument("--wav", default=None, help="16 kHz mono clip instead of synthesis")
    g.add_argument("--kw-end-sec", type=float, default=None,
                   help="keyword end within --wav (required with --wav)")
    g.add_argument("--voice", default="en-US-AriaNeural")
    g.add_argument("--text", default="Amaze")
    g.add_argument("--command", default="turn on the kitchen lights")
    g.add_argument("--lead-sec", type=float, default=1.5,
                   help="audio before the keyword starts")
    g.add_argument("--tail-sec", type=float, default=2.5,
                   help="audio after the keyword ends, before the command")
    g.add_argument("--noise-rms", type=float, default=0.004)
    g.add_argument("--seed", type=int, default=7)

    parser.add_argument("--out", default=None, help="write the full record to this JSON file")
    parser.add_argument("--verbose", action="store_true", help="print every hop")
    args = parser.parse_args()

    # Checkpoints live one level up, next to kws_engine.py. Accept either a path
    # relative to the cwd or a bare filename, so the harness runs from either
    # directory without the caller having to know which one it is in.
    if not os.path.exists(args.model):
        candidate = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 args.model)
        if os.path.exists(candidate):
            args.model = candidate
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())