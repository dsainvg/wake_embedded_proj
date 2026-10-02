# Amaze — on-device wake word for ESP32-S3

An offline keyword spotter for the custom keyword **"amaze"**, running on an
ESP32-S3 N16R8 with an INMP441 I2S microphone. On a hit the LED turns green
and the following 10 seconds of audio are written to internal flash as a WAV.

No cloud, no assistant SDK, no pre-trained generic keyword. The model is
`best_v11_production.flax` from the [`wake_word_ml`](models) submodule, trained
by us in JAX/Flax, and hand-ported to C.

---

## Status

| | |
|---|---|
| Model | `bcconformer_v3`, 84,865 params, self-trained on "amaze" |
| App binary | **586,911 bytes** (was 904,304 with ESP-SR) |
| Internal DRAM | **93,211 B (27.3%)**, 248 KB free — meets the <256 KB budget |
| RAM budget | **met** |
| CPU budget (<10%) | **not met** — see [Performance](#performance) |
| Accuracy | not yet measured on hardware |
| Numerical parity | front end and stem verified; one known defect remains |

**Two known issues, stated up front:**

1. **The C forward pass does not yet match JAX exactly.** `build_feat` and
   `conv1_row` are verified bit-exact in isolation, but running them inside
   `kws_model_run` produces a different result at the first few frames. See
   [Validation](#validation).
2. **CPU is over budget.** 13.5 MMAC per inference in float32 cannot fit
   10% continuous duty at a useful hop. int8 kernels are not written yet.

---

## Hardware

| INMP441 | ESP32-S3 |
|---|---|
| `VDD` | `3V3` |
| `GND` | `GND` |
| `LR`  | `GND` |
| `WS`  | `GPIO40` |
| `SCK` | `GPIO41` |
| `SD`  | `GPIO42` |

`LR` tied to `GND` selects the left slot, which is what the `I2S_SLOT_MODE_MONO`
+ `I2S_STD_SLOT_LEFT` configuration expects.

GPIO 40/41/42 are the JTAG pins (MTCK/MTDO/MTDI). They work as I2S, but do
not attach a JTAG debugger at the same time. Octal PSRAM on N16R8 occupies
GPIO 33-37 only, so there is no conflict.

**Status LED** is the addressable RGB on GPIO48, driven by `led_strip` over RMT.
A plain GPIO toggle cannot light it. Colours carry the state:

| Colour | Meaning |
|---|---|
| dim orange, one pulse every 3 s | firmware alive, streaming from the mic |
| **bright green, solid 3 s** | **wake word detected** |
| amber, blinking 10 Hz | writing the clip to flash |

Set `EXAMPLE_LED_TYPE=GPIO` in menuconfig if you wired a discrete LED instead.

---

## Build and run

```bash
idf.py build
idf.py flash monitor
```

Expected serial output:

```
I wake: status LED: addressable RGB on GPIO48 (1 px)
I wake: INMP441 on WS=40 SCK=41 SD=42 @ 16000 Hz, left slot
I wake: I2S DMA ring: 16 descs x 1024 frames = 16384 samples (1.024 s)
I wake: Amaze v11 bcconformer_v3, 84865 params, threshold 0.68, 2-of-2 peak-hold
I wake: sliding window: 200 ms hop (3200 samples), refractory 1500 ms
I wake: running - say 'amaze'; LED turns green for 3 s on a hit
I wake: listening  p=0.031 hold=0.028  infer=41 ms  hop=200 ms  duty=20.5%  heap=198432
```

That `listening` line is the measurement that matters. It prints real inference
time and the resulting duty cycle, so the performance numbers below can be
replaced with measurements as soon as you run it.

---

## Architecture

```
INMP441
  -> I2S std RX, mono left, 32-bit slots
  -> int16 @ 16 kHz, 1.000 s sliding window (16000 samples)
  -> log-mel front end          (49 frames x 40 mels)
  -> bcconformer_v3             (49,40,1) -> P(keyword)
  -> peak-hold detector         threshold 0.68, 2-of-2, 1.5 s refractory
  -> on hit: LED green 3 s, capture next 10 s to flash
```

### Front end

Mirrors `models/features.py` exactly: 16 kHz, `n_fft` 512, Hanning 480,
hop 320 (20 ms), 40 triangular mel bands 100-7500 Hz, `log(mel + 1e-5)`.

Two optimisations carry the CPU cost:

- **Sparse mel filterbank.** Each triangular filter touches ~5-7 FFT bins, so the
  filterbank is stored as 418 packed weights with per-filter bin spans. A dense
  `(257 x 40)` matmul would be 10,280 MACs per frame — **24.6x more work**.
- **Frame reuse.** The window advances in whole feature hops, so every frame
  before the last few is bit-identical to the previous update. A 200 ms hop
  recomputes 10 of 49 frames and reuses 39.

The window and filterbank are *generated* by `tools/export_weights.py` from
numpy rather than recomputed on the device, so the firmware sees exactly the
values the training pipeline saw.

### Network

`bcconformer_v3` from `models/models.py`: delta-stack stem, two stride-2
convolutions, a mel attention gate, three relative-position conformer blocks,
and a soft-OR statistics pool. Full layer-by-layer description is in
`docs/resource_estimates.md`.

**The stem is never materialised.** `conv1` output is `(49,20,32)` float32 =
125 KB, which does not fit alongside the rest inside 256 KB. GroupNorm needs
statistics over every frame, which normally forces the tensor to exist, so
`conv1` is evaluated **three times** instead: once for its own GroupNorm
statistics, once for the frequency path's, and once to emit rows into a
three-frame rolling window feeding `conv2`. That costs ~1.88 MMAC and saves
125 KB.

`conv2` output *is* materialised, because recomputing it for its own GroupNorm
statistics would cost 6.8 MMAC — far too much to repeat.

---

## Detector

The operating point is read from the checkpoint, not guessed:

| Symbol | Value | Source |
|---|---|---|
| threshold | 0.68 | `deploy_threshold` |
| confirm | 2-of-2 | `deploy_need`, `deploy_confirm_window` |
| aggregation | peak-hold, decay 0.85 | `deploy_agg` |
| refractory | 1.5 s | `deploy_refractory_steps` |
| measured | 78.0% recall, 4.42 FA/h | `deploy_recall`, `deploy_fa_per_hour` |

Confirmation uses a **decaying peak hold**, not an EMA. The checkpoint's own
notes record why: at alpha 0.6 an EMA needs roughly five consecutive high
frames, but a 0.6 s wake word only produces two or three good frames in a 1 s
window, so an EMA-confirmed detector discards most real detections.

The refractory is expressed in **milliseconds**, not steps, so it stays correct
when the hop differs from the reference model's 100 ms.

---

## Performance

Full derivations, assumptions and per-layer tables are in
[`docs/resource_estimates.md`](docs/resource_estimates.md).

**13.51 MMAC per inference**, of which `conv2` alone is **50.1%**.

| path | ms/window | 100 ms | 200 ms | 300 ms | 500 ms |
|---|---|---|---|---|---|
| float32 (current) | ~56 | 56% | 28% | 19% | 11% |
| int8 (projected) | ~35 | 35% | 17% | 12% | 7% |

These are analytical (MAC count ÷ speedup at 240 MHz), **not measurements**.
Treat the int8 row as a range, not a figure.

Two corrections worth recording, both against my own earlier claims:

- **int8 does not halve the operation count.** The LX7 has no dot-product
  unit, so it executes the same 13.51 MMAC — only cheaper per op. Honest
  speedup is ~1.5-1.8x. The real win is that conv2's 55 KB int8 kernel
  becomes cache-resident in the 64 KB D-cache.
- **Inference does not lengthen the hop.** The I2S DMA ring is
  `16 x 1024 = 16384` samples — **1.024 s of slack** — and fills continuously,
  so stalling inside `kws_model_run` does not stall the microphone. Samples are
  lost only if a single inference exceeds that budget, which the firmware now
  warns about explicitly.

Meeting <10% continuous CPU at a 200 ms hop needs ≤3.3 MMAC, a 4x cut. With
`conv2` at 50% of the work, it is the only layer worth attacking.

---

## RAM

| | bytes |
|---|---|
| Internal DRAM | 93,211 (27.3% of 341,760) |
| Weights, flash | 339,460 fp32 / 84,865 int8 |
| Activations | PSRAM via `EXT_RAM_BSS_ATTR` |
| Capture ring + clip, PSRAM | ~640 KB |

The activations carry `EXT_RAM_BSS_ATTR`, which resolves to PSRAM. Note that
this attribute expands to **nothing** unless
`CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY` is set — it is, in
`sdkconfig.defaults`. Without it the whole activation set lands in internal
DRAM and DIRAM reads 96.8%.

The trade is PSRAM bandwidth inside the `conv2` inner loop, so measured
inference time will be worse than a DRAM-only build would be.

---

## Flash layout

| Partition | Offset | Size | Purpose |
|---|---|---|---|
| `nvs` | `0x9000` | 20 KB | boot vars, clip cursor |
| `otadata` | `0xE000` | 8 KB | OTA state |
| `phy_init` | `0x10000` | 4 KB | PHY calibration |
| `factory` | `0x20000` | 3 MB | application |
| `model` | `0x320000` | 2 MB | reserved for the upstream ESP-SR path |
| `storage` | `0x520000` | 8 MB | rolling WAV clips, 25 slots |

The `model` partition is now unused — ESP-SR is gone — but it is left in place
so the table does not move under existing devices.

Each slot holds one 10 s 16 kHz 16-bit mono WAV (~313 KB), used round-robin;
the cursor lives in NVS so it survives reboots. Retrieve clips with:

```powershell
.\tools\pull_clips.ps1 -Port COM15
```

The script reads the whole `storage` partition in one pass, scans for
`RIFF`/`WAVE` headers, and writes `slotNN.wav` per clip.

---

## Tooling

The offline pipeline lives in `tools/` and is run with the project venv:

```bash
python tools/export_weights.py --dump --int8   # weights + frontend data + references
python tools/dump_stages.py                   # per-layer JAX activations
tools\test_kws_host.bat                       # compile the firmware sources for host
python tools/compare_stages.py                # layer-by-layer C vs JAX diff
```

`export_weights.py` aborts if a checkpoint parameter is missing or
unexpected, so a renamed tensor can never be silently shipped as zeros.

### Validation

`tools/test_kws_host.bat` compiles the *exact* firmware sources for the host,
binds a heap copy of the weights, and diffs against the JAX model layer by
layer. This is the safety net, and it earned its place before anything reached
hardware — it found four porting bugs:

1. **`delta_stack` differences run over time, not mel.** Its pad spec
   `((0,0),(1,1),(0,0))` acts on axis 1 of a `(B,T,M)` array, which is frames.
2. **`conv1` output width** was `ROW_W/2` when `ROW_W` was already post-stride.
3. **JAX `SAME` padding uses `pad_lo = pad_total // 2`** (floor). Ceil shifted
   every feature one mel bin.
4. **A fixed five-row gather was too small** to derive `d2`, and `conv1`'s
   `SAME` padding is *zeros*, not edge-replicated.

Validation-only buffers are behind `#ifdef KWS_HOST_TEST`. Together they are
~316 KB and must never exist in a firmware build — they overflowed internal
DRAM before the `EXT_RAM_BSS_ATTR` change.

**Known defect.** `build_feat` and `conv1_row` are each verified bit-exact in
isolation, but calling them from inside `kws_model_run` yields a different
result for the first four frames. Buffer overlap, compiler optimisation,
input corruption and frame misalignment have all been ruled out. Host output is
`P=0.0892` against a JAX reference of `0.0272` on a noise fixture. Until this
is resolved the detector is not trustworthy.

---

## Configuration

| Symbol | Default | Notes |
|---|---|---|
| `EXAMPLE_HOP_MS` | 200 | main CPU dial; must be a multiple of 20 ms |
| `EXAMPLE_THRESHOLD` | 0.68 | from the checkpoint |
| `EXAMPLE_NEED` | 2 | consecutive hits to confirm |
| `EXAMPLE_HOLD_DECAY` | 0.85 | peak-hold decay |
| `EXAMPLE_REFRACTORY_MS` | 1500 | lockout after a hit |
| `EXAMPLE_CLIP_SECONDS` | 10 | WAV length |
| `EXAMPLE_PRE_ROLL_SECONDS` | 1 | keyword tail kept at the front |
| `EXAMPLE_MIC_SAMPLE_SHIFT` | 16 | INMP441 left-justified 24-bit in a 32-bit slot |

The hop must be a multiple of 20 ms because the front end only reuses frames
when the hop is a whole number of feature hops (320 samples).

`EXAMPLE_MIC_SAMPLE_SHIFT=16` discards both the 8 unused low bits of the I2S
slot and the 8 low bits of the real payload, giving a correct full-scale
`int16`. Use 14 for +6 dB gain if detection is too insensitive.

---

## Troubleshooting

**`inference took N ms, longer than the hop`** — the DMA ring (1.024 s) is
being overrun and audio is being dropped. Lower the hop or reduce model cost.

**`duty=` much higher than expected** — inference is genuinely slow. Confirm
`infer=` first, then check PSRAM placement did not hurt the `conv2` loop.

**No detections, `p` pinned near 0** — either the microphone is dead or the
numerical defect above is biting. Check `heap` and the `p` trace first: if `p`
never moves at all, the front end is not seeing audio.

**LED never lights** — GPIO48 is addressable on most devkits; confirm
`EXAMPLE_LED_TYPE` matches your board.

---

## Licence and provenance

The firmware here is ours. `models/` is a submodule of
<https://github.com/dsainvg/wakeword_ml.git>, also ours, containing the
training pipeline and the `amaze` checkpoint. No proprietary or
commercially-licensed wake-word SDK is used anywhere in this project, and no
pre-trained generic assistant keyword is involved.
