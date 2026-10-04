"""
Realistic Multi-Speaker Streaming Simulation for Keyword: "Amaze"
-----------------------------------------------------------------
Simulates real-world continuous multi-speaker conversation:
1. [0.0s - 2.5s]: Background ambient chat (UK English) - "Hey, are you going to the concert tonight?"
2. [2.5s - 4.2s]: Hard Confusable Distractor (US English) - "I was amazed by the stage, it was crazy."
   -> Model MUST NOT trigger on "amazed" or "crazy"!
3. [4.2s - 5.4s]: Wake Word "Amaze" spoken with Indian / Australian accent!
   -> Model INSTANTLY triggers!
4. [5.4s - 8.0s]: Follow-up voice command: "turn on the kitchen lights and play jazz music"
5. Dispatches 300ms pre-roll + live audio to Cloud ASR server and measures latency!
"""

import os
import time
import asyncio
import numpy as np
import soundfile as sf
from scipy import signal
import edge_tts
from kws_engine import StreamingKWSEngine

SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 0.050 # 50ms chunks
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC) # 800 samples


async def synthesize_neural_phrase(text: str, voice: str) -> np.ndarray:
    """Synthesizes text using edge-tts and returns 16kHz float32 audio."""
    communicate = edge_tts.Communicate(text, voice)
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    raw_mp3 = b"".join(chunks)

    import io
    mp3_io = io.BytesIO(raw_mp3)
    data, orig_sr = sf.read(mp3_io)
    if len(data.shape) > 1:
        data = data[:, 0]
    if orig_sr != SAMPLE_RATE:
        num_target = int(len(data) * SAMPLE_RATE / orig_sr)
        data = signal.resample(data, num_target)
    return data.astype(np.float32)


def generate_ambient_pink_noise(num_samples: int) -> np.ndarray:
    white = np.random.normal(0, 1.0, num_samples)
    fft_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(num_samples, d=1.0 / SAMPLE_RATE)
    freqs[0] = 1.0
    pink_fft = fft_white / np.sqrt(freqs)
    pink = np.fft.irfft(pink_fft, n=num_samples)
    return (pink / (np.max(np.abs(pink)) + 1e-6) * 0.04).astype(np.float32)


async def build_multi_speaker_timeline() -> tuple[np.ndarray, float, float]:
    print("[1/3] Synthesizing multi-speaker neural audio stream...", flush=True)

    # Voice 1: UK Male (Conversation)
    print("  -> Generating background conversation (UK English: en-GB-RyanNeural)...", flush=True)
    seg1 = await synthesize_neural_phrase("Hey, are you going to the concert tonight?", "en-GB-RyanNeural")

    # Voice 2: US Female (Confusable Distractor containing 'amazed' and 'crazy')
    print("  -> Generating confusable distractor (US English: en-US-AriaNeural)...", flush=True)
    seg2 = await synthesize_neural_phrase("I was amazed by the stage, it was crazy.", "en-US-AriaNeural")

    # Voice 3: Indian Male (The Wake Word 'Amaze')
    print("  -> Generating wake word 'Amaze' (Indian English: en-IN-PrabhatNeural)...", flush=True)
    seg_wake = await synthesize_neural_phrase("Amaze", "en-IN-PrabhatNeural")

    # Voice 4: Follow-up command
    print("  -> Generating command (en-IN-PrabhatNeural)...", flush=True)
    seg_cmd = await synthesize_neural_phrase("turn on the kitchen lights and play jazz music", "en-IN-PrabhatNeural")

    # Assemble timeline with ambient noise & pauses
    pause_0_5s = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
    pause_1_0s = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)

    timeline_parts = [
        pause_0_5s,
        seg1,
        pause_0_5s,
        seg2,
        pause_0_5s,
        seg_wake,
        pause_0_5s,
        seg_cmd,
        pause_1_0s
    ]

    # Calculate exact wake word time boundary
    t_cursor = 0.5 + (len(seg1) / SAMPLE_RATE) + 0.5 + (len(seg2) / SAMPLE_RATE) + 0.5
    wake_start = t_cursor
    wake_end = wake_start + (len(seg_wake) / SAMPLE_RATE)

    full_audio = np.concatenate(timeline_parts)
    # Add ambient room pink noise
    ambient = generate_ambient_pink_noise(len(full_audio))
    full_audio = full_audio + ambient
    full_audio = full_audio / np.max(np.abs(full_audio)) * 0.90

    return full_audio, wake_start, wake_end


def run_streaming_simulation(checkpoint_path: str = "best_50k_kws_model.flax", arch: str = "bcconformer_50k", thresh: float = 0.65):
    full_audio, wake_start, wake_end = asyncio.run(build_multi_speaker_timeline())
    total_sec = len(full_audio) / SAMPLE_RATE

    print(f"\n[2/3] Multi-speaker audio stream ready: {total_sec:.2f}s total duration.", flush=True)
    print(f"      - [0.0s - {wake_start:.1f}s] Background chat & hard distractor ('amazed', 'crazy')", flush=True)
    print(f"      - [{wake_start:.2f}s - {wake_end:.2f}s] Target wake word 'Amaze'", flush=True)
    print(f"      - [{wake_end:.2f}s - {total_sec:.1f}s] Voice command & cloud streaming\n", flush=True)

    # Initialize KWS Engine
    engine = StreamingKWSEngine(checkpoint_path=checkpoint_path, arch=arch, confidence_threshold=thresh)

    print("\n[3/3] Executing real-time streaming inference loop (50ms chunks)...\n", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Time':^8} | {'State':^16} | {'RMS Energy':^12} | {'Confidence':^14} | {'Latency':^11}", flush=True)
    print("=" * 80, flush=True)

    total_chunks = len(full_audio) // CHUNK_SAMPLES
    state = "IDLE_LISTENING"
    cloud_buffer = []
    spotted_time = 0.0

    for i in range(total_chunks):
        chunk = full_audio[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
        cur_t = i * CHUNK_DURATION_SEC
        rms = np.sqrt(np.mean(chunk ** 2))

        if state == "IDLE_LISTENING":
            spotted, conf, lat = engine.push_audio_chunk(chunk)

            # Print log periodically or if confidence rises
            if (i % 8 == 0) or spotted or (conf > 0.25):
                bar = "#" * int(conf * 15)
                print(f" {cur_t:5.2f}s  | {state:^16} |    {rms:7.4f}   |  {conf:5.3f} {bar:<7} |  {lat:5.2f} ms", flush=True)

            if spotted:
                spotted_time = cur_t
                state = "STREAMING_CLOUD"
                print("\n" + "*" * 80, flush=True)
                print(f" >>> [KEYWORD 'AMAZE' SPOTTED at {cur_t:.2f}s!] Confidence: {conf:.3f} <<<", flush=True)
                print(f" True 'Amaze' audio boundary: {wake_start:.2f}s - {wake_end:.2f}s", flush=True)
                # The window fed to the model ends at cur_t + chunk duration, not
                # at cur_t, and 2-of-2 needs two hops before it can confirm, so
                # this is a coarse figure. It is not the deployment latency.
                print(f" Coarse detection lag: ~{((cur_t + CHUNK_DURATION_SEC) - wake_end) * 1000.0:.0f} ms "
                      f"from keyword completion (window edge, ignores confirmation)", flush=True)
                print("*" * 80 + "\n", flush=True)

                # Capture 300ms pre-roll
                preroll_samples = int(0.3 * SAMPLE_RATE)
                cloud_buffer.extend(engine.audio_buffer[-preroll_samples:])

        elif state == "STREAMING_CLOUD":
            cloud_buffer.extend(chunk)
            if i % 8 == 0:
                print(f" {cur_t:5.2f}s  | {state:^16} |    {rms:7.4f}   |   Streaming...    |       -", flush=True)

    # No latency number is printed here, and that is the point of the change.
    #
    # This script used to close with a fabricated block: "Edge-to-Cloud Delta
    # ~8.2 ms", "ASR Processing Time ~41.5 ms", a recognised command that was
    # never recognised by anything. No socket was opened, no audio left the
    # process, and those constants were typed into the print statement. A latency
    # figure that is not measured is worse than no figure, because it is
    # indistinguishable from one that was.
    #
    # The measurement lives in cloud_asr_server/, which does open a socket, does
    # reconcile the two clocks, and does decompose the total into its terms:
    #
    #     python cloud_asr_server/server.py
    #     python cloud_asr_server/esp32_emulator.py
    #
    # What this script can honestly report is what it collected:
    ingested_sec = len(cloud_buffer) / SAMPLE_RATE
    print("\n" + "=" * 80, flush=True)
    print(" [BUFFERED LOCALLY - NOTHING WAS SENT]", flush=True)
    print(f" Buffered after detection : {ingested_sec:.2f} s ({len(cloud_buffer):,} samples)", flush=True)
    print(" Edge-to-cloud latency    : not measured here, see cloud_asr_server/", flush=True)
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best_50k_kws_model.flax")
    parser.add_argument("--arch", type=str, default="bcconformer_50k")
    parser.add_argument("--thresh", type=float, default=0.65)
    args = parser.parse_args()

    run_streaming_simulation(checkpoint_path=args.model, arch=args.arch, thresh=args.thresh)
