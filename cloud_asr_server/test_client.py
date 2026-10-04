"""
Mock ESP32-S3 WebSocket Streaming Client
---------------------------------------
Simulates the ESP32-S3 microcontroller streaming audio to the Cloud ASR Server.
Used for verification, benchmark testing, and latency validation on PC without hardware.
"""

import time
import json
import wave
import asyncio
import argparse
import numpy as np

try:
    import websockets
except ImportError:
    print("[Error] websockets package is required. Run: pip install websockets")


async def simulate_esp32_stream(server_uri: str, audio_file: str = None, duration_sec: float = 2.5):
    print("=====================================================================")
    print(f" Simulating ESP32-S3 Audio Streaming to: {server_uri}")
    print("=====================================================================")

    async with websockets.connect(server_uri) as ws:
        print("[Connected] Persistent WebSocket handshake established!")
        time.sleep(0.5)

        # 1. Trigger Wake Word Spotted Event
        wake_timestamp_ms = int(time.time() * 1000)
        wake_event = {
            "event": "wake_detected",
            "keyword": "Activate Echo",
            "timestamp_ms": wake_timestamp_ms
        }
        t0 = time.time()
        await ws.send(json.dumps(wake_event))
        print(f"[Edge Event] Wake word spotted! Sent JSON metadata (T0 = {wake_timestamp_ms} ms)")

        # 2. Transmit Pre-Roll Buffer (300ms = 9600 bytes of 16kHz 16-bit PCM)
        preroll_bytes = bytes([0] * 9600)
        await ws.send(preroll_bytes)
        t_first_packet = time.time()
        edge_to_cloud_delta_ms = (t_first_packet - t0) * 1000.0
        print(f"[Latency Metric] Pre-roll dispatched. Edge-to-Cloud Delta: {edge_to_cloud_delta_ms:.2f} ms")

        # 3. Stream Live Audio Chunks (50ms chunks = 1600 bytes each)
        chunk_samples = int(16000 * 0.050) # 800 samples = 1600 bytes
        total_chunks = int(duration_sec / 0.050)

        print(f"[Streaming] Piping live audio: {total_chunks} chunks ({duration_sec}s total)...")
        for i in range(total_chunks):
            # Generate simulated speech sine tone + noise or send silence
            t = np.linspace(0, 0.050, chunk_samples, endpoint=False)
            sine = (0.3 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
            await ws.send(sine.tobytes())
            await asyncio.sleep(0.048) # 50ms pacing

        # 4. Send End of Speech
        end_event = {"event": "end_of_speech"}
        await ws.send(json.dumps(end_event))
        print("[Streaming] Dispatched end_of_speech to server. Waiting for ASR response...")

        # 5. Receive Server Response
        response = await ws.recv()
        result = json.loads(response)
        print("\n===========================================================")
        print(" [RECEIVED CLOUD ASR TRANSCRIPT RESPONSE]")
        print(f" Status     : {result.get('status')}")
        print(f" Transcript : \"{result.get('transcript')}\"")
        print(f" ASR Latency: {result.get('asr_latency_ms')} ms")
        print("===========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock ESP32-S3 Streamer")
    parser.add_argument("--uri", type=str, default="ws://localhost:8000/ws/audio", help="WebSocket URI")
    parser.add_argument("--duration", type=float, default=2.0, help="Stream duration in seconds")
    args = parser.parse_args()

    asyncio.run(simulate_esp32_stream(args.uri, duration_sec=args.duration))
