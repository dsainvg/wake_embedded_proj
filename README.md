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
| Weights in flash | **80,624 B int8** (was 339,460 B float32) |
| App binary | **355,408 bytes** (was 586,911) |
| Internal DRAM | **104,651 B (30.6%)**, 237 KB free — meets the <256 KB budget |
| RAM budget | **met** |
| CPU budget (<10%) | kernels written and building; **not yet measured on hardware** |
| Numerical parity | **verified** — worst error 1.1e-3 against JAX across 6 fixtures |

**What the parity means.** `tools/host_check.c` runs the firmware's own
`kws_model.c` and `kws_int8.c` on the host, bound to the exported weight blobs,
against the probability `model.apply` gives for the identical 1 s window. Every
fixture agrees to within 0.001 of a probability whose decision threshold is
0.68, so the port introduces about 0.15% of the margin it needs.

| fixture | JAX | float weights | int8 weights | error |
|---|---|---|---|---|
| noise    | 0.02721036 | 0.02679202 | 0.02620327 | 1.0e-3 |
| tone1k   | 0.00004640 | 0.00004753 | 0.00005002 | 3.6e-6 |
| sweep    | 0.00061473 | 0.00062528 | 0.00070142 | 8.7e-5 |
| silence  | 0.00008417 | 0.00008792 | 0.00009462 | 1.0e-5 |
| quiet    | 0.01193262 | 0.01166933 | 0.01098841 | 9.4e-4 |
| loud     | 0.02950655 | 0.02980416 | 0.02930226 | 2.0e-4 |

These are all negatives, which is the demanding direction: the two logits differ
by about 3.6 nats, so a 1% perturbation of the pooled feature vector moves the
probability by tens of percent. The absolute error is what the threshold sees.

**One thing is not yet measured.** The INT8 vector kernel in
`main/kws_xtensa.S` has been verified to assemble and to disassemble to the
intended instruction sequence, and it self-tests against the portable C loop at
boot, demoting itself if it disagrees — but nothing has run it on silicon. Until
the board is connected, treat the CPU figure as unknown rather than as met.

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
I wake: int8 kernel: xtensa vector
I wake: Amaze v11 bcconformer_v3, 84865 params, 80624 B int8 weights, threshold 0.68, 2-of-2 peak-hold
I wake: sliding window: 200 ms hop (3200 samples), refractory 1500 ms
I wake: running - say 'amaze'; LED turns green for 3 s on a hit
I wake: listening  p=0.031 hold=0.028  infer=15 ms  hop=200 ms  duty=7.5%  heap=198432
```

That `listening` line is the measurement that matters, and it prints real
inference time and the resulting duty cycle. **The `infer=` and `duty=` values
above are placeholders** — they show the format, not a result. Nothing has run
this build on hardware; see [Performance](#performance).

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

**11.24 MMAC per inference** — the float32 port executed 13.51 MMAC because it
evaluated `conv1` three times through the rolling ring; the materialised int8
stem computes it once. `conv2` is **60.3%** of the total and the conformer Dense
layers another 25%, so those two decide the CPU budget. They are the two that
the hardware can actually do fast.

Where the MACs go, and what each runs on:

| stage | MMAC | share | arithmetic |
|---|---|---|---|
| `conv1` | 0.85 | 7.6% | float32 weights |
| `f_conv` depthwise | 0.09 | 0.8% | float32 weights |
| **`conv2`** | **6.77** | **60.3%** | **int8, vector** |
| mel gate + weighted sum | 0.05 | 0.4% | float32 |
| conformer Dense | 2.78 | 24.7% | **int8, vector** |
| attention `QKᵀ` and `AV` | 0.69 | 6.1% | float32 |
| soft-OR pool + classifier | 0.06 | 0.5% | int8 / float32 |

**Why the arithmetic column is not a preference.** On this core there is no
dot-product unit and no FMA. A scalar int8 multiply-accumulate costs a byte
load, a two-instruction sign extension (there is no sign-extending byte load),
a multiply and an add — about four instructions per MAC, which is *worse* than
float32's hardware `FMUL.S`/`FADD.S`. Writing "int8" in C and expecting a
speedup gets you nothing.

The vector unit is the only thing that changes the arithmetic, and
`EE.VMULAS.S8.ACCX` is sixteen signed 8×8 multiply-accumulates per instruction
off two 128-bit loads. That is a ~4× instruction reduction against the scalar
int8 path and ~10× against float32, on the 85% of the work that is vectorised.

Both operands have to be 16-byte aligned for that, which is arranged offline:
`tools/export_weights.py` pads every weight row to a multiple of 16 and starts
every tensor on a 16-byte boundary, and `kws_model.c` pads every quantised
activation row the same way. The zero pad bytes contribute nothing to the sum.
It costs 25 bytes of weights.

| path | instructions/MAC | est. ms/window | est. duty @200 ms hop |
|---|---|---|---|
| float32 | ~3 | ~56 | ~28% |
| int8, scalar C | ~4 | ~75 | ~37% |
| **int8, vector** | **~0.4** | **~15** | **~7-8%** |

**The bottom row is an estimate and the one above it is a measurement.** The
56 ms figure was measured on hardware before the port. Nothing has run on
hardware since, so treat 15 ms as a projection: it assumes roughly one
instruction per cycle out of the vector path and ignores PSRAM traffic and the
swish cost. The boot log prints the real per-inference time and the running
average, so one `idf.py monitor` session settles it.

Two corrections worth recording, both against my own earlier claims:

- **int8 does not halve the operation count.** The MACs are the same; only the
  cost per MAC changes, and only if the vector unit is used. Plain C int8 is
  slower than float32 on this part.
- **Inference does not lengthen the hop.** The I2S DMA ring is
  `16 x 1024 = 16384` samples — **1.024 s of slack** — and fills continuously,
  so stalling inside `kws_model_run` does not stall the microphone. Samples are
  lost only if a single inference exceeds that budget, which the firmware warns
  about explicitly.

---

## RAM

| | bytes |
|---|---|
| Internal DRAM | 104,651 (30.6% of 341,760) |
| Weights, flash | 80,624 int8 + 22,844 float32 parameters |
| Activations, PSRAM | ~340 KB via `EXT_RAM_BSS_ATTR` |
| Per-frame scratch, internal | ~15 KB |
| Capture ring + clip, PSRAM | ~640 KB |

The activations carry `EXT_RAM_BSS_ATTR`, which resolves to PSRAM. Note that
this attribute expands to **nothing** unless
`CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY` is set — it is, in
`sdkconfig.defaults`. Without it the whole activation set lands in internal
DRAM and DIRAM reads 96.8%.

The per-frame scratch buffers are deliberately left in internal DRAM rather than
moved to PSRAM to save 15 KB: each is 2.5 KB, is read and written three times
across the stem passes, and internal DRAM is 30.6% used. Spending bandwidth to
save space nobody is short of would be the wrong trade.

The trade that does exist is PSRAM bandwidth inside the `conv2` inner loop, so
measured inference time will be worse than a DRAM-only build would be.

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

```bash
python tools/export_weights.py --int8 --dump   # weights, frontend data, references
python tools/capture_jax.py noise              # what the real Flax model computes
tools\build_host.bat                           # compile the firmware sources for host
build\host\kws_host.exe                        # parity against those references
```

`export_weights.py` aborts if a checkpoint parameter is missing or unexpected,
so a renamed tensor can never be silently shipped as zeros. It also owns two
offline transforms that are easy to get wrong and impossible to debug at runtime:
reversing every tensor's axes so the reduction axis is contiguous, and padding
every reduction to a multiple of 16 so the vector kernel's 128-bit loads land
aligned. Both are asserted, and `build\host\kws_host.exe` fails loudly if the
firmware's stride disagrees with them.

### Validation

Two tools, and the split matters.

**`tools/capture_jax.py` wraps the live Flax modules and runs `model.apply`
once**, recording what each one computed. It re-implements nothing. That is
deliberate: this project had two hand-written numpy/jax reimplementations of
`BCConformerV3` as its "reference", and both disagreed with the model they were
supposed to describe — one averaged GroupNorm statistics across all four groups
instead of per group, and one used `epsilon=1e-5` where `flax.linen.GroupNorm`
defaults to `1e-6`. Both produced plausible-looking numbers. Comparing against
the real modules removes that entire class of error.

**`tools/host_check.c` compiles the exact firmware sources for the host**, binds
a heap copy of the exported blobs, and runs the six fixtures the exporter
dumped. Same `kws_model.c`, same `kws_int8.c`, same weights, same layout — a
host pass is a real pass.

| fixture | JAX | float weights | int8 weights | error |
|---|---|---|---|---|
| noise    | 0.02721036 | 0.02679202 | 0.02620327 | 1.0e-3 |
| tone1k   | 0.00004640 | 0.00004753 | 0.00005002 | 3.6e-6 |
| sweep    | 0.00061473 | 0.00062528 | 0.00070142 | 8.7e-5 |
| silence  | 0.00008417 | 0.00008792 | 0.00009462 | 1.0e-5 |
| quiet    | 0.01193262 | 0.01166933 | 0.01098841 | 9.4e-4 |
| loud     | 0.02950655 | 0.02980416 | 0.02930226 | 2.0e-4 |

The harness also runs each fixture twice and compares, because a model that
reads an uninitialised buffer returns a different probability every call, and
that presents as a threshold problem rather than as a bug.

**The float-weights column is not decoration.** It is the same code path with
the float32 blob bound instead of the int8 one, so the difference between the
two columns isolates what the quantised *weights* cost from what the int8
*activations* cost, without maintaining a second model.

#### Bugs this caught, in order

1. **`spec_at` read out of bounds.** `clamp_frame` was applied to the frame
   index inside `d1_at` but not inside `spec_at`, so `spec_at(spec, -1, m)`
   computed `(size_t)(-1) * 40 + m` and returned garbage. It surfaced as conv1
   outputs of 1e27 at specific mel bins, not as a crash.
2. **`gn_reset()` for the second GroupNorm zeroed the first one's statistics.**
   They shared one global, and GroupNorm-1 is applied inside the loop that
   accumulates GroupNorm-2. The stem was being normalised by nothing.
3. **The conv window was packed in the wrong order.** The gather produced
   `(kt, kw, channel)` and the transposed weights are ordered
   `(channel, kw, kt)`. Every output element had the right magnitude and no
   relation to the reference — the signature of a layout bug rather than an
   arithmetic one.
4. **`epsilon` in GroupNorm.** 1e-5 instead of Flax's 1e-6.

Plus the four from the float32 port, listed in the git history.

### On-device checks

Two things are checked on the device at boot, because neither can be checked on
a host and both are the kind of failure that does not announce itself:

- **`kws_kernel_selftest()`** runs the assembly vector reduction against the
  portable C loop over every reduction length the network uses, with random
  data, full-scale alternating sign, and one non-zero element at a time at every
  offset. If they disagree, the portable loop takes over and the boot log says
  so. A kernel that assembles is not a kernel that computes the right thing, and
  a wrong one returns a plausible probability rather than an error.
- **`kws_expf_selftest()`** measures the fast `exp` against the library's over
  `[-30, 30]` and falls back to `expf` if the relative error exceeds 1e-4. The
  claimed bound is 2e-5; it is re-measured on every build rather than assumed.

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

**`int8 kernel: ... SELF TEST FAILED, fell back to portable C`** — the assembly
reduction disagreed with the portable loop and was demoted. Detection is still
correct, just slower; check the CPU figure before assuming the fallback is
harmless.

**`inference took N ms, longer than the hop`** — the DMA ring (1.024 s) is
being overrun and audio is being dropped. Lower the hop or reduce model cost.

**`duty=` much higher than expected** — inference is genuinely slow. Confirm
`infer=` first, then check PSRAM placement did not hurt the `conv2` loop.

**No detections, `p` pinned near 0** — check `heap` and the `p` trace first: if
`p` never moves at all, the front end is not seeing audio. If it moves but never
crosses the threshold, that is the model's false-reject rate on your acoustics,
not the port — parity with JAX is verified, so the answer is in the training.

**LED never lights** — GPIO48 is addressable on most devkits; confirm
`EXAMPLE_LED_TYPE` matches your board.

---

## Licence and provenance

The firmware here is ours. `models/` is a submodule of
<https://github.com/dsainvg/wakeword_ml.git>, also ours, containing the
training pipeline and the `amaze` checkpoint. No proprietary or
commercially-licensed wake-word SDK is used anywhere in this project, and no
pre-trained generic assistant keyword is involved.

`main/kws_xtensa.S` implements `EE.VMULAS.S8.ACCX` from the documented Xtensa
LX7 vector instruction set. Espressif's `esp-nn` component was used as the
reference for the instruction semantics and for the two-cycle gap before
reading the accumulator; the loop is our own.
