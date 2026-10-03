# Compliance analysis: `bcconformer_v3` against the stated KWS boundaries

Prepared from measurements taken on the physical board (ESP32-S3, 240 MHz, 8 MB
octal PSRAM @ 80 MHz) plus arithmetic on the model's own MAC count. Every
hardware number below was measured, not estimated; every projection is labelled
as one.

The four boundaries in the brief:

| # | Boundary | Status | Measured |
|---|---|---|---|
| 1 | Model size, RAM/flash footprint | **AT RISK / FAIL** | 221 KB internal + 259 KB PSRAM = **480 KB** vs a 256 KB limit |
| 2 | CPU < 10% while idling in continuous listening | **FAIL** | **296 ms** per inference vs a **22.2 ms** budget — **13.3x over** |
| 3 | High true-positive rate, near-zero false activations | **UNVERIFIED** | 79.8% streaming recall claimed offline; **no positive has ever been run on-device** |
| 4 | Latency from keyword end to cloud ASR | **NOT BUILT** | no streaming/transport layer exists |

The headline: **the model itself fits; the implementation does not.** And on the
plain ESP32 the brief names as an example target, it fails far harder, because
two of its speed assumptions are S3-specific.

---

## 1. CPU: the model fits, the code is 101x off its floor

The brief allows 10% CPU while listening. With a 200 ms hop, duty is
`infer / (hop + infer)`, so 10% means:

```
infer <= hop / 9
  hop = 200 ms  ->  22.2 ms      (this is the budget to beat)
  hop = 100 ms  ->  11.1 ms      (the cadence the model was validated at)
```

Measured: **296 ms**. Reduction required: **13.3x**.

Now the arithmetic floor, which is the important part of this document:

```
11.24 MMAC / 16 MAC per EE.VMULAS.S8   =    702,500 instructions
240 MHz = 240,000 cycles per ms        =      2.93 ms
```

**2.93 ms.** The budget of 22.2 ms is **7.6x** that floor. A properly int8,
properly vectorised implementation of *this exact network* has enough headroom
to comply with room to spare.

The current implementation is **101x** the floor. So the problem is not that the
model is too big for the part — at 84,865 parameters and 11.24 MMAC it is a small
model, and it is nowhere near "a heavy or uncompressed pre-trained transformer".
The problem is that **87% of the runtime is float32** on a core with no float
SIMD:

| stage | ms | share | representation |
|---|---|---|---|
| blocks (conformer) | 106.7 | 35.9% | int8 Dense, **float32 attention / norms / swish** |
| stem1 + stem2 + stem3 | 121.3 | 40.9% | **float32** |
| gn3 + swish | 24.0 | 8.1% | **float32** |
| conv2 | 40.7 | 13.7% | **int8** |
| melgate | 3.0 | 1.0% | float32 |
| pool + head | 1.2 | 0.4% | mixed |

Only conv2 is int8. ~86,000 scalar swishes per inference, plus attention matmuls,
LayerNorm and softmax, all on a core whose only vector unit is integer.

**Honest caveat:** the 101x gap is larger than the float32 share alone explains.
The measured PSRAM ceiling is ~60 MB/s, and I could not account for the residual
from first principles. That unexplained part is itself a finding: it means there
is more recoverable performance here than "quantise the float32", and it should be
measured rather than assumed.

### 1a. One concrete, quantified win: conv2 is flash-bound, not compute-bound

conv2 measures 40.7 ms for 6.77 MMAC — whose arithmetic cost is **1.76 ms**. That
is 23x overhead, and the cause is identifiable:

```
weight matrix (13,824 B) is re-read once per frame:  49 passes = 661.5 KB
661.5 KB / 40.7 ms                                      = 16.3 MB/s
SPI flash, DIO, 80 MHz (theoretical)                   = 20.0 MB/s
```

conv2 is reading its weight matrix out of flash 49 times per inference, and the
time is the flash bandwidth. The weights live in `.rodata`; the S3's flash cache
is not holding a 13.8 KB matrix across a 49-iteration loop that also streams
activations through it.

The fix is to make the weight pass more expensive per unit of gather, which is
just a bigger gather window — 2,880 B of scratch per frame today:

| frames gathered per weight pass | weight passes | scratch | projected conv2 |
|---|---|---|---|
| 1 (current) | 49 | 2,880 B | 40 ms |
| 2 | 25 | 5,760 B | 20 ms |
| **4** | **13** | **11,520 B** | **~10.6 ms** |
| 8 | 7 | 23,040 B | ~5.7 ms |

DIRAM has 115 KB free, so a 4-frame gather (11.5 KB) is affordable immediately.
This is the single best-understood item on the list: **conv2 40.7 -> ~11 ms,
saving ~30 ms** for a loop-boundary change and 11.5 KB of RAM.

---

## 2. RAM: passes only if PSRAM is not counted, and the margin is thin

| region | bytes | vs 256 KB |
|---|---|---|
| Internal DIRAM (`.data` + `.bss`) | 226,539 (221 KB) | **86.4%** — 35 KB free |
| External PSRAM (`.bss`) | 265,408 (259 KB) | 101% on its own |
| **Combined** | **491,947 (480 KB)** | **1.88x over** |

Flash is not a constraint: app 358,416 B + weights 80,624 B = **429 KB**, against
a 3 MB app partition (89% free).

The 256 KB figure in the brief is ambiguous about external RAM, and the two
readings give opposite verdicts:

- **Internal only:** 221 KB — compliant, with 35 KB of headroom. Thin but real.
- **Internal + external:** 480 KB — **1.88x over, fails.**

If PSRAM counts, the reduction needed is large. The largest single tensor is
`s_c2`, conv2's float32 output at **94,080 B**. Quantising it to int8 per row
would cut it to ~23,520 B (70 KB saved) and also halve the traffic of the two
stages that read it — `gn3+sw` (24.0 ms) and `melgate` (3.0 ms), both of which
are a single streaming pass over it. The original handoff argued against this on
*arithmetic* grounds ("only 47k float operations against 6.8 MMAC"); that
argument is wrong now that the bottleneck is known to be bandwidth, not
arithmetic.

---

## 3. Accuracy: the least-verified area, and the one the brief weighs most

The brief asks for a high true-positive rate with near-zero false activations.

- Offline, the checkpoint reports **79.8% streaming recall** at threshold 0.68
  with 2-of-2 peak-hold, **91.8%** clean-audio detection, FA 4.4/hour, with an
  FPR budget of 0.005.
- **All six parity fixtures are negatives.** Every fixture in the harness is
  silence, noise, tone or sweep. Worst int8-vs-JAX error is 5.738e-4 against a
  5e-3 budget.
- **No positive has ever been run through this pipeline**, on host or device.
  `recorded/` holds six 10-second INMP441 clips from a session where detection
  did fire, which is the obvious test, and it has not been done.

So the int8 port is verified to be *faithful to the float32 port*, and the float32
port is verified against JAX — but **whether the network fires on real recorded
speech at all, on this hardware, is untested.** In the device logs, `p` sits at
0.017-0.027 in a quiet room, which is consistent with the negative fixtures, so
there is no evidence of a fault; there is also no evidence of a working detector.

Two further accuracy risks that optimisation would introduce:

- **The firmware's 200 ms hop is already coarser than the validated 100 ms.** The
  model was validated at 100 ms (1,600-sample chunks). The firmware evaluates
  every 200 ms. That halves the temporal sampling and works against the 2-of-2
  peak-hold, which needs two consecutive frames above threshold.
- **79.8% recall is arguably below "high TPR"**, and it is the number a
  submission would be judged on.

### 3a. The hop is not a free parameter

The 2-of-2 detector requires two *consecutive evaluations* above threshold, and
consecutive evaluations are one hop apart. So raising the hop to fit a CPU budget
directly attacks recall:

| hop | 10% CPU budget | consecutive-hit spacing |
|---|---|---|
| 100 ms (as validated) | 11.1 ms | 100 ms |
| 200 ms (current) | 22.2 ms | 200 ms |
| 500 ms | 55.6 ms | 500 ms — a ~500 ms keyword barely spans 2 frames |

The hop cannot be raised as a performance shortcut without re-validating recall.

---

## 4. Latency: not implemented

The brief's defining feature is streaming the audio that *follows* the keyword to
a cloud ASR with minimal overhead. There is no network, transport or ASR client
in this project. What exists is the capture half:

- a rolling 10-second pre-roll buffer in PSRAM, and a 10-second post-keyword
  capture written to the internal `storage` partition as a WAV (`recorded/`).

The capture machinery was **dead** until this session — `ring_push` and
`capture_push` were defined, documented as "called from the feed task for every
AFE frame", and never called. The compiler's unused-function warnings were the
only evidence. They are now wired into the feed loop, but this has never been
observed to work end-to-end, because no detection has ever fired.

Writing to internal flash is also the wrong shape for the actual requirement: a
cloud handoff wants a stream, not a file written after the fact. The 25-slot
storage partition and the WAV framing would be replaced by a ring buffer feeding
a socket.

---

## 5. Hardware: two speed assumptions are ESP32-S3-specific

The brief names "Raspberry Pi or ESP32". The performance case above depends
entirely on the S3:

| property | ESP32-S3 (measured here) | plain ESP32 (LX6) |
|---|---|---|
| `EE.VMULAS.S8` int8 dot product | yes, 16 MAC/instruction | **no** |
| PSRAM | 8 MB, octal | none by default |
| Internal SRAM | 512 KB | 520 KB, ~300 KB usable |
| Consequence | int8 kernels vectorised | **int8 kernels run scalar** |

The portable fallback is what the self-test demotes to when the assembly is
unavailable, and it was measured on the S3 at roughly **3x slower** — conv2 went
from 40.7 ms to 485.6 ms when a fault demoted the whole kernel. On a plain ESP32
the assembly does not exist at all, so that scalar path is the only path, and the
265 KB of PSRAM-resident buffers could not be allocated.

**On a plain ESP32 this design fails both the RAM bound and the CPU bound, by a
wide margin.** Any submission targeting that part needs a different plan.

---

## 6. What compliance would actually take

Three options, with the honest assessment of each.

### Option A — drive the existing model to its floor

1. conv2 flash re-read fix (§1a): 40.7 -> ~11 ms. *Understood, low risk.*
2. Quantise the conformer attention matmuls, LayerNorm/GroupNorm and swish.
   The **9x error headroom** (5.738e-4 against a 5e-3 budget) is the currency
   here — the original float32 choice was made before the budget was tightened.
3. Quantise `s_c2` (§2): also attacks `gn3+sw` and `melgate`.

Projection: steps 2-3 apply to ~255 ms of float32 work. If that yields 4-6x —
plausible but **unverified, and I have been wrong on every projection so far** —
the total lands at **50-75 ms**. That is still **2.4-3.4x over** the 22.2 ms
budget.

**Option A alone does not comply.** It is necessary, not sufficient.

### Option B — retrain a smaller model

To have comfortable headroom, target ~1/3 of the current work: two conformer
blocks instead of three, or dimension 32 instead of 48, or 25 frames instead of
49. The MAC floor would drop to ~1 ms, leaving ~22x of overhead allowance inside
the budget.

Cost: a full retrain and revalidation, against a checkpoint currently at 79.8%
recall that has **never been confirmed to fire on real recorded speech**. The
first priority there is not a smaller model — it is establishing a working
positive on hardware.

### Option C — combine

Option A, then retrain smaller, then re-validate recall at whatever hop the
optimised timing supports.

---

## 7. Recommendation, in order

1. **Establish a positive.** Before any further optimisation, run a recorded
   "amaze" clip through the firmware and confirm the LED fires and a clip is
   written. Nothing else in this document can be settled while the detector's
   actual behaviour is unmeasured. It is also the cheapest experiment available:
   `recorded/` already holds six real clips.
2. **Do the conv2 flash fix** (§1a). ~30 ms for a loop-boundary change and 11.5 KB
   of RAM, fully understood from measurement.
3. **Spend the 9x error headroom** on int8 attention and norms (Option A step 2).
   This is the only remaining lever that changes the order of magnitude.
4. **Only then consider a smaller model** (Option B), because until step 3 is done
   there is no evidence about how much of the 101x gap is recoverable.
5. **Decide the PSRAM question early.** If the evaluator counts external RAM, the
   design needs ~225 KB removed before any of the above matters. That is a
   different conversation from optimising CPU.