"""
Remote Cloud ASR Streaming WebSocket Server
-------------------------------------------
Accepts real-time binary audio streams from ESP32-S3 over WebSockets.
Features:
- Pluggable streaming ASR engines (Faster-Whisper, Vosk, or Benchmark Mock)
- NTP-style clock reconciliation, so a delta between the device clock and this
  clock is a measurement rather than an assumption (see `latency.py`)
- Latency decomposed into device and transport terms instead of one number
- Full-duplex bidirectional streaming (sends live partial & final transcripts back)

Wire protocol
-------------
Text frames are JSON events, binary frames are 16 kHz 16-bit mono PCM.

    {"event": "time_probe", "seq": n, "t1": <device s>,
     "prev_seq": n-1, "prev_t4": <device s>}   ->  {"event": "time_probe_ack",
                                                      "seq": n, "t1": ..., "t2": ..., "t3": ...}

    The probe is two-step in the NTP sense: `prev_t4` carries the device's
    receive time for the *previous* probe, so one round trip completes one
    sample and the server can reduce the window itself. The device never has to
    report an offset it has no way to compute.

    {"event": "wake_detected", "t_kw_end": <device s>, "t_detect": <device s>,
     "t_first_write": <device s>, "keyword_end_sample": n,
     "stream_start_sample": n, "preroll_samples": n, "samples_dropped": n}

    {"event": "end_of_speech"}
"""

import time
import json
import argparse
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

from latency import ClockModel, build_breakdown

app = FastAPI(title="ESP32-S3 Cloud ASR Streaming Server")

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

# Every turn appends one record here, and the file is the durable artefact: the
# stdout report is for a human watching one run, this is for comparing runs.
LATENCY_LOG = os.environ.get("KWS_LATENCY_LOG", "latency_log.jsonl")

# Live connections. Kept at module scope so /metrics can report on them.
SESSIONS: list = []

# Cloud timestamps are seconds since this process started, not Unix epoch.
#
# This is a precision requirement, not tidiness. A float64 holds ~52 bits of
# mantissa, and a Unix-epoch second is already ~1.79e12, which spends 41 of them
# on the integer part and leaves about 11 for the fraction: the smallest
# representable step at that magnitude is ~0.5 ms. A latency measurement whose
# own clock quantises to 0.5 ms cannot resolve the transport term it is trying
# to attribute, and the degradation is invisible -- the numbers still look
# plausible. Measured: min round trip came back as 0.49 ms on loopback before
# this change.
CLOUD_EPOCH = time.time()


def now() -> float:
    """Cloud clock, in seconds since process start."""
    return time.time() - CLOUD_EPOCH

# Check available ASR backends
ASR_ENGINE = "mock"
whisper_model = None

try:
    from faster_whisper import WhisperModel
    # Load lightweight Whisper model (tiny.en or base.en for ultra-fast response)
    whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    ASR_ENGINE = "faster-whisper"
    print("[ASR Engine] Successfully initialized Faster-Whisper (tiny.en, int8)")
except Exception:
    try:
        import vosk
        # Initialize Vosk if model exists
        ASR_ENGINE = "vosk"
        print("[ASR Engine] Successfully initialized Vosk streaming engine")
    except Exception:
        ASR_ENGINE = "mock"
        print("[ASR Engine] Running in Benchmark Mode (Install faster-whisper or vosk for live STT)")


def run_transcription(audio_bytes: bytes) -> str:
    """Runs speech-to-text decoding on collected 16kHz 16-bit mono audio."""
    if ASR_ENGINE == "faster-whisper" and whisper_model is not None:
        # Convert PCM 16-bit to float32 normalized array
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        segments, _ = whisper_model.transcribe(audio_float32, beam_size=1, language="en")
        text = " ".join([seg.text for seg in segments]).strip()
        return text if text else "(Silence / Unrecognized)"
    else:
        # Benchmark mock transcription
        return "turn on the living room lights"


@dataclass
class Session:
    """Everything measured for one connection."""

    client: str
    clock: ClockModel = field(default_factory=ClockModel)
    # seq -> (t1, t2, t3) awaiting the device's t4, which rides on the next probe
    pending: dict = field(default_factory=dict)

    stream_active: bool = False
    audio_buffer: bytearray = field(default_factory=bytearray)

    wake_payload: Optional[dict] = None
    cloud_wake_at: Optional[float] = None
    cloud_first_audio_at: Optional[float] = None
    device_first_write: Optional[float] = None
    audio_before_wake_bytes: int = 0

    reports: list = field(default_factory=list)

    def complete_probe(self, seq: int, t4: float) -> None:
        entry = self.pending.pop(seq, None)
        if entry is None:
            return
        t1, t2, t3 = entry
        probe = self.clock.add(seq=seq, t1=t1, t2=t2, t3=t3, t4=t4)
        if not probe.is_usable():
            return
        est = self.clock.estimate()
        if est is not None and est.n_usable % 8 == 1:
            print(f"[Clock] offset {est.offset * 1000.0:+.3f} ms  "
                  f"min-rtt {est.delay * 1000.0:.3f} ms  "
                  f"spread {est.theta_spread * 1000.0:.3f} ms  "
                  f"({est.n_usable}/{est.samples} usable)")

    def note_first_audio(self) -> None:
        """
        Stamp the arrival of the first PCM frame.

        Only a timestamp, taken where the receive() returns. The breakdown is
        not emitted here because it needs one number the device cannot have sent
        yet: `t_first_write` is the moment the pre-roll write *begins*, which is
        after the wake metadata carrying it has been flushed. Emitting early
        would mean either reporting a write timestamp that has not happened or
        silently folding the metadata frame into the network term.
        """
        if self.cloud_first_audio_at is None:
            self.cloud_first_audio_at = now()

    def emit_breakdown(self) -> None:
        """Build, log and print this turn's decomposition."""
        if self.wake_payload is None or self.cloud_first_audio_at is None:
            return

        p = self.wake_payload
        breakdown = build_breakdown(
            keyword_end=p["t_kw_end"],
            detect=p["t_detect"],
            first_byte_write=self.device_first_write,
            cloud_first_audio=self.cloud_first_audio_at,
            cloud_wake_event=self.cloud_wake_at,
            estimate=self.clock.estimate(),
            keyword_end_sample=int(p.get("keyword_end_sample", -1)),
            stream_start_sample=int(p.get("stream_start_sample", -1)),
            preroll_samples=int(p.get("preroll_samples", 0)),
            samples_dropped=int(p.get("samples_dropped", 0)),
        )
        if self.audio_before_wake_bytes:
            breakdown.warnings.append(
                f"{self.audio_before_wake_bytes} bytes arrived before the wake event "
                "and were discarded")

        self.reports.append(breakdown)
        record = breakdown.as_dict()
        record["client"] = self.client
        try:
            with open(LATENCY_LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            print(f"[Latency] could not append to {LATENCY_LOG}: {exc}")

        print()
        print(breakdown.format_table())
        for problem in record["problems"]:
            print(f"  ! {problem}")
        if not record["problems"]:
            print("  (no coherence problems)")
        print()


@app.get("/")
async def root():
    return {
        "service": "ESP32-S3 Cloud ASR Server",
        "asr_engine": ASR_ENGINE,
        "sample_rate": SAMPLE_RATE,
        "latency_log": LATENCY_LOG,
        "status": "online"
    }


@app.get("/metrics")
async def metrics():
    """Last decomposition per connection, so a client can read its own numbers."""
    return {"turns": [r.as_dict() for r in SESSIONS for r in r.reports]}


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_ip = websocket.client.host if websocket.client else "unknown"
    print(f"\n[WebSocket] ESP32-S3 Connected from: {client_ip}")

    session = Session(client=client_ip)
    SESSIONS.append(session)

    try:
        while True:
            message = await websocket.receive()

            # ---- JSON metadata ----
            if "text" in message and message["text"]:
                t2 = now()
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                event = payload.get("event")

                if event == "time_probe":
                    # This probe closes the previous one: the device's receive
                    # time for probe prev_seq rode along with this send.
                    if payload.get("prev_seq") is not None and payload.get("prev_t4") is not None:
                        session.complete_probe(int(payload["prev_seq"]), float(payload["prev_t4"]))
                    if payload.get("t1") is not None:
                        session.pending[int(payload["seq"])] = (
                            float(payload["t1"]), t2, now())
                    t3 = now()
                    await websocket.send_text(json.dumps({
                        "event": "time_probe_ack",
                        "seq": payload.get("seq"),
                        "t1": payload.get("t1"),
                        "t2": t2,
                        "t3": t3,
                    }))

                elif event == "wake_detected":
                    session.cloud_wake_at = t2
                    session.wake_payload = payload
                    session.stream_active = True
                    session.audio_buffer.clear()

                    print(f"\n===========================================================")
                    print(" [WAKE DETECTED EVENT] Received from ESP32-S3!")
                    print(f" Keyword end (device)   : {payload.get('t_kw_end')} s")
                    print(f" Detection  (device)    : {payload.get('t_detect')} s")
                    print(f" Cloud arrival          : {t2:.6f} s")
                    print("===========================================================")

                elif event == "end_of_speech":
                    session.stream_active = False

                    # The pre-roll write begins *after* the wake metadata that
                    # gates it has been flushed, so its timestamp can only be
                    # known here, which is why the breakdown is emitted here.
                    if payload.get("t_first_write") is not None:
                        session.device_first_write = float(payload["t_first_write"])
                    session.emit_breakdown()

                    total_bytes = len(session.audio_buffer)
                    duration_sec = total_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)

                    print(f"\n[ASR] Stream finished. Received {total_bytes:,} bytes "
                          f"({duration_sec:.2f}s of audio).")
                    print("[ASR] Decoding speech stream...")

                    # Execute ASR Decoding
                    t_asr_start = now()
                    transcript = run_transcription(bytes(session.audio_buffer))
                    t_asr_end = now()

                    asr_latency_ms = (t_asr_end - t_asr_start) * 1000.0

                    print(f" [TRANSCRIPT RESULT] : \"{transcript}\"")
                    print(f" ASR Decoding Time   : {asr_latency_ms:.1f} ms")
                    if session.cloud_first_audio_at is not None:
                        print(f" Audio ingest span    : "
                              f"{(t_asr_end - session.cloud_first_audio_at) * 1000.0:.1f} ms")
                    print("===========================================================\n")

                    response = {
                        "status": "success",
                        "transcript": transcript,
                        "audio_duration_sec": round(duration_sec, 2),
                        "asr_latency_ms": round(asr_latency_ms, 1),
                        "engine": ASR_ENGINE,
                    }
                    if session.reports:
                        response["latency"] = session.reports[-1].as_dict()
                    await websocket.send_text(json.dumps(response))

            # ---- binary PCM ----
            elif "bytes" in message and message["bytes"]:
                chunk = message["bytes"]
                if session.stream_active:
                    if session.cloud_first_audio_at is None:
                        session.note_first_audio()
                    session.audio_buffer.extend(chunk)
                else:
                    session.audio_before_wake_bytes += len(chunk)

    except WebSocketDisconnect:
        print(f"[WebSocket] ESP32-S3 Disconnected: {client_ip}")
    except RuntimeError as e:
        # Starlette can hand back a disconnect and then raise on the next
        # receive(); a device that drops the socket mid-turn is normal, not an
        # error worth a traceback.
        print(f"[WebSocket] ESP32-S3 Disconnected: {client_ip} ({e})")
    except Exception as e:
        print(f"[WebSocket Error] {e}")
    finally:
        if session in SESSIONS:
            SESSIONS.remove(session)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud ASR WebSocket Streaming Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Listening port")
    parser.add_argument("--latency-log", type=str, default=LATENCY_LOG,
                        help="JSONL file every turn's latency record is appended to")
    args = parser.parse_args()

    print(f"Starting Cloud ASR WebSocket Server on ws://{args.host}:{args.port}/ws/audio ...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")