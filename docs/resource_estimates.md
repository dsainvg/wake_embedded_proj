# Resource estimates — `bcconformer_v3` keyword spotter on ESP32-S3 (N16R8)

**Wake word:** "amaze" · **Model:** `bcconformer_v3`, 84,865 parameters, fp32 hand-port
**Target:** ESP32-S3 N16R8, dual-core Xtensa LX7 @ 240 MHz, 512 KB internal SRAM
(~380 KB usable as general DRAM), 8 MB octal PSRAM @ 80 MHz, 16 MB flash
**Project constraints:** **< 256 KB RAM**, **< 10 % CPU**, minimise latency from
keyword end to ASR receipt
**Source of truth for the model:** `main/kws_model.c`, `main/kws_frontend.c`
**Source of truth for the operating point:** `sdkconfig.defaults`, `models/agents.md`

---

## 0. How to read this document

**Every number below is a prediction, not a measurement.** They are derived
analytically from the source and from stated chip assumptions, and the chip
assumptions in §5.1 are the weakest link — the effective MAC rate, `expf()` cost
and flash XIP behaviour are all *estimates* that have not been run on this board.
The two I trust least are flagged **[UNMEASURED]**.

Where a value is derived arithmetically from the source it is marked *derived*.
Where it is an estimate it is marked *estimated* and the reasoning is given.
Where it is a range it is given as a range, because **a wrong confident number is
worse than a stated range**.

Method summary:

1. Recount every MAC independently from tensor shapes (`§1`).
2. Count the non-MAC operations separately, because on the LX7 they are a large
   and *non-quantisable* fraction of the cost (`§3`).
3. Build an itemised buffer ledger and take a pessimistic peak (`§4`).
4. Apply a stated cycles-per-MAC model and propagate uncertainty into ranges (`§5`).
5. Check what actually meets the two hard constraints (`§6`, `§7`).

Notation: **MAC** = one multiply-accumulate. **FLOP** = 2 × MAC. Dimensions:
`T=49` frames, `M=40` mel bands, `C1=32`, `SM=10` stem mel bins, `C2=48`,
`D=48`, `H=4` heads, `dh=12`, `B=3` blocks, `SE=6`, `PH=24` pool hidden,
`F=97` pooled features (`main/kws_model.h:20-29`).

---

## 1. Operation count — independent verification

### 1.1 The figures to verify

| Stage | Claimed MAC/inference | Independent derivation | Agrees? |
|---|---|---|---|
| `conv1` (one pass) | 846,720 | `49 × 20 × 32` outputs × `3·3·3` = 31,360 × 27 | ✅ |
| `conv1` (3 passes) | 2,540,160 | 846,720 × 3 | ✅ |
| `f_conv` (one pass) | 94,080 | 31,360 outputs × `3` taps (depthwise, 32 groups) | ✅ |
| `f_conv` (3 passes) | 282,240 | 94,080 × 3 | ✅ |
| `conv2` | 6,773,760 | `49 × 10 × 48` = 23,520 outputs × `3·3·32` = 288 | ✅ |
| mel gate | 23,520 | 490 outputs × 48 | ✅ (but incomplete — see §1.2) |
| 3 × RelConformerBlock | 3,875,472 | 3 × 1,270,656 = 3,811,968 | ⚠️ +63,504 |
| SoftORStatsPool | 56,448 | 49 × 48 × 24 | ✅ (but incomplete — see §1.2) |
| classifier | 194 | 97 × 2 | ✅ |
| **TOTAL** | **13,551,794** | **13,512,986** | ✅ within 0.29 % |

The headline "~13.55 MMAC" is **correct to three significant figures**. Two
line-item corrections and one reclassification follow.

### 1.2 Corrections

**C1 — the conformer block figure includes LayerNorm multiplies, not just MACs.**
`3,875,472 − 3,811,968 = 63,504`, which is exactly `3 blocks × 3 LayerNorms ×
49 rows × 48 elements × 3 multiplies` (the `d*d` variance term, the `× inv_std`
term and the `× scale` term in `layer_norm()`, `main/kws_model.c:137-168`).
Those *are* real multiplies, but folding them into a "MAC" line hides the fact
that they are not dot products and not int8-friendly. §1.3 separates them.

**C2 — the mel gate costs 2× the claimed number.** The claimed 23,520 counts only
the 1×1 projection (490 outputs × 48). The *weighted sum* that actually
materialises the `(49,48)` sequence is another `49 × 10 × 48 = 23,520`
multiply-accumulates (`main/kws_model.c:696-700`). Mel gate = **47,040**.

**C3 — the pool has a second projection.** The soft-OR scalar is
`w·Dense(x)` over 49 frames × 24 hidden = **1,176** MAC
(`main/kws_model.c:586`), on top of the 56,448 from the `48→24` Dense. Pool =
**57,624**.

### 1.3 Corrected per-layer MAC table (fp32 path as committed)

| # | Stage | Shape out | MAC/inference | % of total | Source |
|---|---|---|---|---|---|
| 1 | `build_feat` (delta stack) | (49,40,3) | 0 (subtraction only) | 0.0 % | `kws_model.c:289` |
| 2 | `conv1` × 3 | (49,20,32) | 2,540,160 | 18.79 % | `kws_model.c:360` |
| 3 | GroupNorm-1 (2 accum + 2 apply) | (49,20,32) | 0 MAC — 384,160 multiplies | — | `kws_model.c:178,208` |
| 4 | `f_conv` × 3 | (49,20,32) | 282,240 | 2.09 % | `kws_model.c:392` |
| 5 | GroupNorm-2 (2 accum + 1 apply) | (49,20,32) | 0 MAC — 125,440 multiplies | — | as above |
| 6 | stem residual `h + f + mean(f)` | (49,20,32) | 0 MAC — 62,720 adds | — | `kws_model.c:651-658` |
| 7 | **`conv2`** | **(49,10,48)** | **6,773,760** | **50.12 %** | `kws_model.c:411` |
| 8 | GroupNorm-3 (1 accum + 1 apply) | (49,10,48) | 0 MAC — 70,560 multiplies | — | `kws_model.c:670-674` |
| 9 | mel gate projection | (49,10) | 23,520 | 0.17 % | `kws_model.c:686-689` |
| 10 | mel gate weighted sum | (49,48) | 23,520 | 0.17 % | `kws_model.c:696-700` |
| 11 | 3 × RelConformerBlock | (49,48) | 3,811,968 | 28.21 % | `kws_model.c:530` |
| 11a | ↳ attention (qkv, QKᵀ, AV, proj) | | 2,046,240 | 15.14 % | `kws_model.c:445` |
| 11b | ↳ depthwise k3+k7, pointwise | | 409,248 | 3.03 % | `kws_model.c:494,512` |
| 11c | ↳ SE gate (48→6→48) | | 1,728 | 0.01 % | `kws_model.c:530` |
| 11d | ↳ FFN 48→96→48 | | 1,354,752 | 10.03 % | `kws_model.c:551-554` |
| 12 | SoftORStatsPool | (97,) | 57,624 | 0.43 % | `kws_model.c:574` |
| 13 | classifier + softmax | (2,) | 194 | 0.00 % | `kws_model.c:716` |
| | **TOTAL** | | **13,512,986** | 100 % | |

### 1.4 `conv2` is half the network — say it out loud

> **`conv2` — a single 3×3×32→48 convolution — is 50.12 % of all arithmetic in
> `bcconformer_v3`.**

It is 8× the cost of `conv1` and 1.8× the entire 3-block conformer stack, and it
is a single dense 3×3 convolution. This is **the** optimisation target. Every
other layer combined is 49.88 %.

Secondary hot spots, in order: `conv1` (18.79 %, and 2/3 of it is the GroupNorm
recompute artefact), attention (15.14 %), the FFN (10.03 %).

### 1.5 FLOPs

| Quantity | fp32 path | Notes |
|---|---|---|
| MAC | 13,512,986 | *derived* |
| FLOP from MAC (×2) | 27,025,972 | multiply + add |
| FLOP from `expf` (127,371 × ~10) | 1,273,710 | range 7–15 FLOP each |
| FLOP from divide (83,937 × ~14) | 1,175,118 | LX7 float divide is a short Newton–Raphson sequence |
| FLOP from `logf`/`rsqrt` (855 × ~11) | 9,405 | |
| FLOP from GroupNorm arithmetic | ~1,536,640 | 384,160 mul + 384,160 add + 384,160 sub/affine |
| FLOP from residual adds, max, compare, memmove | ~1,400,000 | *estimated* |
| **Total** | **≈ 32.4 MFLOP** | say **32 MFLOP ± 3** |

---

## 2. int8: the operation count does not change

**This is the single most important honesty statement in the document.**

The ESP32-S3's Xtensa LX7 has:

* a single-precision FPU (FSE/FSIP pipelines) with fused multiply-add;
* a 32-bit integer pipeline;
* **no int8 SIMD and no dot-product / matrix-multiply instruction.**

The ESP32-P4 does have dot-product instructions, which is where the familiar
"int8 is 4× faster" figure comes from. **The LX7 does not have them, and quoting
that 4× here would be wrong.**

Therefore, for a per-tensor-symmetric int8 quantisation of this network:

| | fp32 | int8 |
|---|---|---|
| Number of MACs performed | 13,512,986 | **13,512,986 — identical** |
| Operand width | 32 bit | 8 bit in, **32 bit accumulate** |
| Executed on | FSE pipeline | integer pipeline (same core, different pipe) |
| What actually changes | — | **cycles per MAC**, memory traffic, quantise/dequantise overhead |

An int8 MAC on the LX7 is *not* one instruction. It is roughly:

| Step | Instructions | Why it is not free |
|---|---|---|
| Load int8 weight + int8 activation | 2 | The LX7 has no byte-wide load-with-scale |
| Widen both to 32 bit | ~2 (`slli`+`srai`, or `sext`) | The integer datapath is 32-bit; there is no int8×int8→int16 accumulate |
| Multiply (`mull`) | 1 | 3-cycle latency, 1/cycle issue |
| Accumulate into int32 (`add`) | 1 | 1-cycle latency |
| Amortised loop / address / bounds | ~0.5 | — |
| **Total** | **≈ 5–6 int ops / MAC** | vs 1 FFMA for fp32 |

With four independent int32 accumulators to break the `mull` latency chain, the
*issue* rate is ~2.5–3.0 instructions/MAC, and the integer pipeline (which is
distinct from the FSE pipe) can overlap it with the fp32 transcendentals.
**Net arithmetic effect of int8 on this chip: ~1.2–1.5×, not 4×.** §5.4 derives
this properly, and §7.2 explains where int8 *does* pay.

**What int8 does buy, decisively, is memory traffic** — see §4.4 and §7.2.

---

## 3. Non-MAC operations (transcendental / normalisation)

These are 6–8 % of the *operation* count but a much larger share of the *cycles*,
and **none of them are int8-friendly**: they are float operations on quantities
(`std`, variance, softmax sums, exp arguments) whose dynamic range int8 destroys.

### 3.1 `expf` calls per inference

| Site | Count | Code |
|---|---|---|
| swish after GroupNorm-1 (graph-faithful only) | 31,360 | see §3.3 note |
| swish after GroupNorm-2 (graph-faithful only) | 31,360 | see §3.3 note |
| swish on `dwconv k3` output, per block | 2,352 × 3 = 7,056 | `kws_model.c:543` |
| swish on `dwconv k7` output, per block | 2,352 × 3 = 7,056 | `kws_model.c:543` |
| SE hidden swish, per block | 6 × 3 = 18 | `kws_model.c:556` |
| SE gate sigmoid, per block | 2,352 × 3 = 7,056 | `kws_model.c:560` |
| FFN hidden swish (49×96), per block | 4,704 × 3 = 14,112 | `kws_model.c:565` |
| attention softmax (4 heads × 49 × 49), per block | 9,604 × 3 = 28,812 | `kws_model.c:476` |
| mel gate softmax (49 × 10) | 490 | `kws_model.c:693` |
| classifier softmax | 2 | `kws_model.c:718` |
| pool log-sum-exp (49) | 49 | `kws_model.c:603` |
| **TOTAL `expf`** | **127,371** (graph-faithful) / 64,671 (as committed) | |

### 3.2 Other float primitives

| Primitive | Count/inference | Notes |
|---|---|---|
| float divide (the `x/(1+exp(-x))` in swish, `1/√var`, `1/Σexp`) | 83,937 | graph-faithful |
| `logf` — pool | 2 | `kws_model.c:604` |
| `logf` — front end, `log(mel + 1e-5)` | 400 (10 frames × 40) at a 200 ms hop | `kws_frontend.c:265` |
| `1/√` (rsqrt) — 9 LayerNorms + 3 GroupNorms | 9×49 + 3×4 = 453 | |
| max / compare | ≈ 31,700 | softmax max scans, pool max |
| GroupNorm multiply + add + subtract | 1,152,480 | §1.3 rows 3, 5, 8 |

### 3.3 Note on the as-committed stem (accuracy, not resources)

The trained graph applies `swish` after GroupNorm-1 and after GroupNorm-2. The
C port applies **neither** — passes 1–3 of `kws_model_run`
(`main/kws_model.c:625-666`) call `gn_apply()` and then go straight into
`fconv_row()` / the residual. This is a *numerical* deviation from training, not
a resource one, and it is why the as-committed `expf` count is 64,671 rather
than 127,371. Resource tables below use the **graph-faithful** count, because
that is what a correct build must execute, and note the as-committed figure
where the difference is material. This deviation is worth a separate fix and a
re-measured TPR; it is flagged here only so the two MAC/exp tables reconcile.

### 3.4 Front-end operations per recomputed frame

| Operation | Count/frame | Source |
|---|---|---|
| Hanning window multiply | 480 | `kws_frontend.c:252` |
| 256-point complex FFT (radix-2, 8 stages × 128 butterflies) | 4,096 real MAC + 6,144 add | `kws_frontend.c:75` |
| real-FFT unpack + twist + power | 3,084 real MAC | `kws_frontend.c:121` |
| **sparse mel filterbank** | **418 MAC** | `kws_frontend.c:257-265` |
| `logf` per mel band | 40 | `kws_frontend.c:265` |
| **Total** | **≈ 8,100 real MAC + 8,700 add + 40 `logf`** | |

Two front-end optimisations already in place, verified:

* **Sparse mel filterbank** — 40 triangular filters × 5–7 FFT bins = 418 MAC,
  versus a dense `257 × 40` matmul of 10,280 MAC. **24.6× reduction.** *Derived.*
* **Frame reuse** — with `n` new samples pushed, `keep = 49 − n/320` frames are
  bit-identical to the previous window and are only `memmove`d
  (`kws_frontend.c:229-245`). *Derived.*

Front-end cost per inference therefore scales with **frames recomputed**, not
with 49:

| Hop | Samples | Frames recomputed | Front-end cost **[UNMEASURED]** |
|---|---|---|---|
| 100 ms | 1,600 | 5 | ~0.55 ms |
| 200 ms | 3,200 | 10 | ~1.10 ms |
| 300 ms | 4,800 | 15 | ~1.65 ms |
| 500 ms | 8,000 | 25 | ~2.75 ms |
| 1000 ms | 16,000 | 49 (no reuse) | ~5.40 ms |

(Derived from 8,100 MAC × 2.2 cyc/MAC + 8,700 adds × 1 cyc + 40 `logf` × 40 cyc
≈ 32,000 cycles/frame = 0.133 ms/frame.)

---

## 4. RAM

### 4.1 Overlap assumptions (stated explicitly)

| ID | Assumption |
|---|---|
| **A1** | **No buffer reuse.** Where two buffers' live ranges do not obviously not overlap, both are charged. This over-counts — see A2 note. |
| **A2** | Genuine, provable overlaps that are nevertheless **charged** because the current code does not exploit them: `s_ln` and `s_attn` and `s_blk` are all `(49,48)` and are used as strict sequence stages inside `conformer_block`; `s_row`/`s_frow`/`s_hring` overlap with the conv2 emit. Charging them all is the pessimistic reading. |
| **A3** | PSRAM allocations are reported **separately** in §4.6, then **added back** in §4.7 under the stated assumption that PSRAM counts against the 256 KB figure. |
| **A4** | The I2S DMA pool and the hop-sized read buffer are internal (DMA-capable internal by default; `CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL=65536` also forces anything ≤ 64 KB internal, `sdkconfig.defaults:16`). |
| **A5** | FreeRTOS task stacks and TCBs are internal; none are configured for PSRAM (`main/wake_word_main.c:641-645`). |
| **A6** | The validation-only buffers (`s_stem_dbg` 125,440 B, `s_delta_dbg` 70,560 B, `kws_debug_buf` 131,072 B — 327,072 B of `.bss`) are **excluded**. They are behind `#ifdef KWS_HOST_TEST` (`main/kws_model.c:83-91`) and are correctly absent from the firmware build. |
| **A7** | The float32 weight blob is `const` in flash (XIP), not RAM. *Derived* from `kws_model.c:49`. |

### 4.2 Model activation buffers — fp32, firmware build

All from `main/kws_model.c:67-81`.

| Buffer | Shape | Elements | Bytes | Note |
|---|---|---|---|---|
| `s_feat` | 3 × (40,3) | 360 | 1,440 | delta-stack rows t−1, t, t+1 |
| `s_row` | 20 × 32 | 640 | 2,560 | stem row |
| `s_frow` | 20 × 32 | 640 | 2,560 | f_conv row |
| `s_hring` | 3 × (20,32) | 1,920 | 7,680 | 3-frame conv2 ring |
| **`s_c2`** | **(49,10,48)** | **23,520** | **94,080** | **conv2 output — the single largest buffer** |
| `s_seq` | (49,48) | 2,352 | 9,408 | conformer working sequence |
| `s_ln` | (49,48) | 2,352 | 9,408 | LayerNorm output |
| `s_qkv` | (49,144) | 7,056 | 28,224 | fused qkv projection |
| `s_attn` | (49,48) | 2,352 | 9,408 | attention output / conv-mod scratch |
| `s_blk` | (49,48) | 2,352 | 9,408 | conv-mod output |
| `s_ffn` | (49,96) | 4,704 | 18,816 | FFN hidden |
| `s_logits` | (49,) | 49 | 196 | attention logits row |
| `s_pool_s` | (49,) | 49 | 196 | soft-OR scores |
| `s_gn_sum`, `s_gn_sumsq` | 4 + 4 | 8 | 32 | GroupNorm statistics |
| | | | **193,416 B** | **= 188.9 KB** |

### 4.3 Front-end buffers

All from `main/kws_frontend.c:41-163`.

| Buffer | Shape | Elements | Bytes | Note |
|---|---|---|---|---|
| `s_tw_re`, `s_tw_im` | 128 each | 256 | 1,024 | 256-point FFT twiddles |
| `s_tw2_re`, `s_tw2_im` | 257 each | 514 | 2,056 | 512-point twist |
| `s_bitrev` | 256 × `uint8` | 256 | 256 | bit-reversal table |
| `s_pre`, `s_pim` | 256 each | 512 | 2,048 | 256-point complex FFT workspace |
| `s_mel_frames` | (49,40) | 1,960 | 7,840 | log-mel feature grid |
| `s_frame` | (512,) | 512 | 2,048 | windowed frame |
| `s_power` | (257,) | 257 | 1,028 | power spectrum |
| `s_audio` | (16,000 + 3,200) × `int16` | 19,200 | **38,400** | 1 s sliding window + 200 ms append room |
| | | | **54,700 B** | **= 53.4 KB** |

Const flash tables (not RAM): `kws_window[480]` = 1,920 B,
`kws_mel_spans[40]` = 160 B, `kws_mel_weights[418]` = 1,672 B, weight blob
339,460 B. Total rodata ≈ 343 KB. Flash is 16 MB; irrelevant to RAM.

### 4.4 Application and system RAM (fp32)

| Item | Bytes | Placement | Source |
|---|---|---|---|
| model activations | 193,416 | internal `.bss` | §4.2 |
| front-end state | 54,700 | internal `.bss` | §4.3 |
| `static float spec[49*40]` in `kws_task` | 7,840 | internal `.bss` | `wake_word_main.c:506` |
| **static subtotal** | **255,956** | | **= 249.96 KB** |
| I2S DMA pool: 16 descriptors × 1,024 frames × 4 B (mono int16 in a 32-bit slot) | ~65,700 | internal, DMA | `wake_word_main.c:458-459` |
| `raw` hop buffer: 3,200 × `int16` (200 ms hop) | 6,400 | internal, DMA | `wake_word_main.c:505` |
| semaphores (3 × `SemaphoreHandle_t` + control blocks) | ~250 | heap | `wake_word_main.c:612-624` |
| task stacks: `led` 3,072 + `store` 4,096 + `kws` 8,192, + idle ~2,560 + timer ~2,048 | ~19,968 | internal | `wake_word_main.c:641-645` |
| 3 × TCB + FreeRTOS heap bookkeeping | ~1,000 | internal | |
| **heap subtotal** | **~93,318** | | |
| **PEAK INTERNAL SRAM (fp32)** | **~349,274 B** | | **= 341.1 KB** |

> ### The float32 path **does not fit the 256 KB constraint.** It needs ~341 KB —
> **33 % over budget**, and that is *before* the LwIP/WiFi stack, which the
> current firmware does not build but the ASR design requires.

341 KB is still inside the chip's ~380 KB of usable general DRAM, so the fp32
build will link and run — it just leaves ~39 KB for everything else, which is not
enough for WiFi.

### 4.5 Weights

| | fp32 | int8 (per-tensor symmetric) |
|---|---|---|
| Quantised parameters | 84,865 | 84,865 |
| Bytes | **339,460** | **84,865** |
| Quantisation scales (76 tensors) | — | 76 × 4 B = **304** |
| Total flash footprint | 339,460 B | **85,169 B** |
| RAM cost | 0 (XIP, `const`) | 0 (XIP, `const`) |

**Weights are flash, not RAM, in both cases.** The RAM win from int8 is entirely
in the *activation* tables below. The *flash/XIP bandwidth* win is large and is
discussed in §5.4 and §7.2.

### 4.6 int8 activation layout

Design rule: int8 for the conv/linear accumulators; **fp32 kept wherever the
maths is a normalisation or a softmax**, because int8 there is a recall bug, not
a rounding error.

**int8 w+a, 3-pass stem (stem *not* materialised — direct analogue of the fp32
structure):**

| Buffer | Elements | int8 B | Kept fp32? |
|---|---|---|---|
| `s_feat` | 360 | 360 | no |
| `s_row`, `s_frow` | 1,280 | 1,280 | no |
| `s_hring` | 1,920 | 1,920 | no |
| `s_c2` | 23,520 | **23,520** | no — the headline saving (94,080 → 23,520) |
| `s_seq` | 2,352 | 2,352 | no |
| `s_qkv` | 7,056 | 7,056 | no |
| `s_blk` | 2,352 | 2,352 | no |
| `s_ffn` | 4,704 | 4,704 | no |
| `s_ln` | 2,352 | — | **9,408 fp32** (mean/variance need range) |
| `s_attn` | 2,352 | — | **9,408 fp32** (softmax probabilities) |
| conv2 row accumulator (10×48 `int32`) | 480 | — | 1,920 `int32` |
| `s_logits`, `s_pool_s` | 98 | — | 392 fp32 |
| `s_gn_*` | 8 | — | 32 fp32 |
| **subtotal** | | **64,704 B** | **= 63.2 KB** |

**int8 w+a, single-pass stem (the recommended build).** With int8 activations the
stem tensor `(49,20,32)` is **31,360 B instead of 125,440 B**, which fits
comfortably and removes the 3-pass GroupNorm workaround entirely (§7.3):

| Delta | Bytes |
|---|---|
| `s_stem` (new, int8, materialised) | **+31,360** |
| `s_row`, `s_frow`, `s_hring` (retired) | **−3,200** |
| **subtotal** | **92,864 B = 90.7 KB** |

`static int8 spec[49*40]` in `kws_task` drops from 7,840 → 1,960 B.

### 4.7 Peak RAM summary and the 256 KB verdict

Front-end (54,700 B) is unchanged in every int8 configuration — a log-mel
spectrogram is not quantisable without destroying the front end's dynamic range.

| Configuration | static `.bss` | + heap/system | **PEAK internal** | vs 256 KB |
|---|---|---|---|---|
| **fp32, 3-pass stem** | 255,956 (250.0 KB) | +93,318 | **349,274 B = 341.1 KB** | **✗ over by 85 KB (33 %)** |
| int8 w only (fp32 activations) | ≈ 255,900 | +93,318 | **≈ 349,200 B = 341.0 KB** | ✗ over — **int8 weights alone buy no RAM** |
| **int8 w+a, 3-pass stem** | 121,364 (118.5 KB) | +93,318 | **214,682 B = 209.7 KB** | **✓ 44 KB headroom** |
| **int8 w+a, single-pass stem** | 149,524 (146.0 KB) | +93,318 | **242,842 B = 237.2 KB** | **✓ 19 KB headroom (tight)** |
| int8 w+a, 1-pass stem, + WiFi/LwIP (~40 KB) | 149,524 | +133,318 | **282,842 B = 276.2 KB** | ✗ over once WiFi is added |

Two structural notes on the headroom:

* **`CONFIG_SPIRAM_MALLOC_RESERVE_INTERNAL=98304`** (`sdkconfig.defaults:17`)
  already reserves 96 KB of the internal heap against PSRAM encroachment, and
  **`CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL=65536`** (line 16) forces any allocation
  ≤ 64 KB into internal SRAM. Both are in the *right* direction but they are
  *not* reflected in the `.bss` column, so the static subtotal is a hard floor
  that no heap policy can relieve.
* The tightest item is `s_stem` at 31,360 B — 34 % of the int8 model budget. It
  exists only because single-pass GroupNorm needs the whole tensor, and it is
  affordable *only* because int8 made it affordable. If the WiFi stack has to
  share internal SRAM, the 3-pass stem returns and costs 1.88 MMAC again.

### 4.8 Does PSRAM count against the 256 KB?

**Assumption: yes.** PSRAM is external DRAM on the same memory bus, behind an
80 MHz octal interface; the CPU pays cache misses for it and the system pays
capacity for it. Treating it as free would be indefensible. So:

| PSRAM allocation | Bytes | Source |
|---|---|---|
| pre-roll ring, 1 s × 16 kHz × 2 B | 32,000 | `wake_word_main.c:616` (`MALLOC_CAP_SPIRARAM`) |
| capture buffer, 10 s × 16 kHz × 2 B | 320,000 | `wake_word_main.c:621` |
| flash-write staging buffer, 10 s × 2 B | 320,000 | `wake_word_main.c:399` (plain `malloc`; 320 KB > 64 KB so `ALWAYSINTERNAL` sends it to PSRAM) |
| ESP-IDF / WiFi / LwIP (when added) | ~40,000 | *estimated* |
| **PSRAM total** | **~712,000 B = 695 KB** | of 8 MB — 8.7 % |

| Configuration | internal | PSRAM | **total DRAM** | vs 256 KB |
|---|---|---|---|---|
| fp32 | 341.1 KB | 695 KB | **1,036 KB** | **✗ 4.05× over** |
| int8 w+a, 1-pass stem | 237.2 KB | 695 KB | **932 KB** | **✗ 3.64× over** |

> ### If PSRAM counts, **no configuration fits 256 KB**, and the reason is not
> the model.
>
> The capture path is 672 KB of the 712 KB: a 10 s WAV recorder plus its
> staging copy. The graded metric is *"time from keyword end to the ASR server
> receiving audio"* — and the current firmware **has no network path at all**
> (`main/wake_word_main.c` writes a WAV to flash and stops). The flash recorder
> and the ASR streaming path are competing for the same memory budget, and the
> flash recorder is not what is being graded.

**Recommended scope statement:** budget the 256 KB against the **inference path**
— front end + model activations + tasks + I2S — and either (a) make the flash
recorder optional and off in the graded configuration, or (b) drop the 320 KB
staging copy by writing the clip sector-by-sector as it fills. Under that
scope, **int8 w+a with a single-pass stem fits at 237.2 KB with 19 KB of
headroom**, and the fp32 path does not fit at all.

---

## 5. CPU / latency

### 5.1 Cycle model and assumptions

| Assumption | Value | Status | Reasoning |
|---|---|---|---|
| Clock | 240 MHz | *given* | `sdkconfig.defaults:20` |
| FPU issue | 1 FSE op/cycle, 4-cycle result latency | **estimated** | LX7 single-precision FPU with FFMA |
| fp32 MAC, **as committed** | **2.2 cycles/MAC** (0.45 MAC/cyc) | **[UNMEASURED]** | `dense()` (`kws_model.c:117`), `conv1_row` (`:360`) and `conv2_row` (`:411`) all accumulate into **one scalar** (`yr[o]`, `acc`), so the FFMA chain is fully serial at 4 cycles. The credit above 4.0 is the load/store pipeline overlapping. |
| fp32 MAC, restructured (4 partial accumulators, unrolled) | 1.1 cycles/MAC (0.9 MAC/cyc) | *estimated* | FPU issue-bound floor |
| fp32 MAC, pessimistic (no overlap) | 4.0 cycles/MAC | *estimated* | hard serial bound |
| int8 MAC, int8 weights + int8 activations | **1.45 cycles/MAC** (0.69 MAC/cyc) | **[UNMEASURED]** | 5–6 int ops/MAC ÷ 4 independent int32 accumulators, integer pipe overlapping the FSE pipe |
| int8 MAC, **int8 weights only**, fp32 activations | 3.0 cycles/MAC | *estimated* | a float→int conversion on every load eats the entire win |
| `expf` | 32 cycles (range 20–45) | **[UNMEASURED]** | ESP-IDF newlib/esp-libm software `expf` for Xtensa |
| float divide | 12 cycles (range 8–20) | **[UNMEASURED]** | FSIP short Newton–Raphson sequence |
| `logf` | 40 cycles | **[UNMEASURED]** | |
| `rsqrt` | 14 cycles | **[UNMEASURED]** | |
| Memory / branch / loop overhead factor | fp32 **1.20**, int8 **1.10** | *estimated* | int8 touches ~4× less flash, so less overhead |
| Flash XIP bandwidth | 25–30 MB/s sustained | *estimated* | 80 MHz quad SPI, flash and PSRAM on separate SPI peripherals on the S3 |

**The memory term deserves emphasis.** Per inference, weight bytes *touched*
(assuming no cache reuse) is:

| Layer | fp32 bytes touched | int8 bytes touched |
|---|---|---|
| `conv2` (55,296 B kernel re-read for each of 49 rows) | 2,709,504 | 677,376 |
| FFN × 3 blocks | 5,418,624 | 1,354,656 |
| qkv projection × 3 | 4,064,256 | 1,016,064 |
| attention output projection × 3 | 1,354,944 | 338,736 |
| `conv1` × 3 | 169,344 | 42,336 |
| pointwise × 3 | 1,354,944 | 338,736 |
| pool, dwconv, mel gate, f_conv, classifier | ~590,000 | ~148,000 |
| **total (no cache reuse)** | **≈ 15.7 MB** | **≈ 3.9 MB** |

The L3-equivalent here is a **64 KB data cache**
(`CONFIG_ESP32S3_DATA_CACHE_64KB`). The `conv2` kernel is 55,296 B in fp32 — it
*nearly fills the entire data cache* and is competing with 94 KB of streaming
activations. In int8 it is 13,824 B and fits many times over.

> **This is the real argument for int8 on the LX7, and it is a memory argument,
> not an arithmetic one.** int8 makes the `conv2` kernel cache-resident where
> fp32 cannot.

### 5.2 fp32 cycles per inference (as committed)

| Component | Count | Cycles/MAC or /op | Mcycles |
|---|---|---|---|
| MAC (all layers) | 13,512,986 | 2.2 | 29.73 |
| GroupNorm multiplies | 384,160 | 2.2 | 0.85 |
| GroupNorm adds/subs (integer pipe, overlapped) | ~768,320 | 0.3 eff. | 0.23 |
| `expf` | 127,371 | 32 | 4.08 |
| float divide | 83,937 | 12 | 1.01 |
| `logf` + rsqrt | 855 | ~30 | 0.03 |
| residual adds / max / memmove | — | — | 0.60 |
| **subtotal** | | | **36.53** |
| overhead × 1.20 | | | **43.84** |
| **fp32, as committed** | | | **43.84 Mcycles = 183 ms** |
| **fp32, restructured accumulators (1.1 cyc/MAC)** | | | **25.3 Mcycles = 105 ms** |
| **fp32, pessimistic (4.0 cyc/MAC)** | | | **73.3 Mcycles = 305 ms** |

> **fp32, as committed: 183 ms per inference (range 105–305 ms depending on how
> well the FPU pipelines and how well the weights cache).**

### 5.3 int8 cycles per inference

| Variant | MACs | Mcycles (MAC) | Mcycles (non-MAC) | Overhead | **Total** |
|---|---|---|---|---|---|
| int8 w only, fp32 activations | 13,512,986 | 40.5 | 6.0 | 1.20 | **46.5 Mcyc = 194 ms — SLOWER than fp32** |
| int8 w+a, 3-pass stem | 13,512,986 | 19.59 | 6.00 | 1.10 | **28.2 Mcyc = 117 ms** |
| **int8 w+a, single-pass stem** | **11,631,386** | 16.87 | 6.00 | 1.10 | **25.2 Mcyc = 105 ms** |
| int8 w+a, 1-pass stem, attention kept fp32 | 11,631,386 | 19.9 | 6.00 | 1.10 | **28.5 Mcyc = 119 ms** |
| int8 w+a, 1-pass stem, `conv2` → (3,1,32,48) | 7,116,506 | 10.32 | 6.00 | 1.10 | **18.0 Mcyc = 75 ms** |
| **int8 w+a, 1-pass stem, `conv2` → depthwise-separable** | **6,645,146** | 9.64 | 6.00 | 1.10 | **17.2 Mcyc = 72 ms** |

> **int8 w+a, single-pass stem: 105 ms. With `conv2` reduced: 72 ms.**

### 5.4 Justifying the int8 speedup — honestly

The brief asked for a 2×–4× range. **On an LX7, 4× is not available, and 2× is
only available if you also fix the fp32 code.** The honest decomposition:

| Term | Share of fp32 time | int8 factor | Contribution to overall |
|---|---|---|---|
| Arithmetic (the 13.5 MMAC) | 29.7 / 43.8 = **68 %** | **1.22×** (2.2 → 1.45 cyc/MAC) | 1.22× on that term |
| Flash XIP / weight traffic | (folded into the 1.20 overhead, est. 15–25 % of total) | **~3.5×** (15.7 MB → 3.9 MB) | 1.10–1.30× overall |
| `expf` / divide / `logf` / softmax | 5.1 / 43.8 = **12 %** | **1.00×** — must stay fp32 | 0 % |
| GroupNorm / LayerNorm | 1.1 / 43.8 = **3 %** | **1.00×** — must stay fp32 | 0 % |
| Quantise/dequantise overhead | 0 % | adds ~0.3 Mcyc | −2 % |

**Resulting range:**

| Scenario | Overall speedup | Time | When you get it |
|---|---|---|---|
| int8 w+a vs **fp32 as committed** (2.2 cyc/MAC) | **1.5×–1.8×** | 117 ms vs 183 ms | current code, no restructuring |
| int8 w+a vs **fp32 restructured** (1.1 cyc/MAC) | **1.0×–1.2×** | 105 ms vs 105 ms | both optimised — int8 is a *wash on arithmetic* |
| int8 w+a, 1-pass stem, vs fp32 as committed | **1.75×** | 105 ms vs 183 ms | recommended |
| int8 w+a + `conv2` reduced, vs fp32 as committed | **2.55×** | 72 ms vs 183 ms | recommended, needs retraining |
| **4× (the ESP32-P4 figure)** | **not available on LX7** | — | requires `mulp`/dot-product instructions the S3 lacks |

**The honest one-line summary:** *on the ESP32-S3, int8 is a **RAM and
cache** optimisation worth ~1.5–1.8× end-to-end, and an **arithmetic**
optimisation worth only ~1.2×. Its real value here is that it is the only thing
that makes the 3-pass stem unnecessary and the 256 KB budget reachable (§4.7,
§7.3) — not that it makes the MACs cheaper.*

**Additional int8 cost that is easy to forget:** int32 accumulation over a
288-term inner loop (in `conv2`) requires widening *every* product, because the
LX7 has no int8→int16 accumulate path. This is precisely why the rate is
1.45 cyc/MAC and not 0.5. And the requantise cost between tensors is small
(≈130,000 elements × 2 ops ≈ 0.3 Mcycles, ~1 % of the total) because the
tensors are small — that is the one place where this network is unusually
quantisation-friendly.

### 5.5 Quantisation risk (must be said)

The network is **not** uniformly safe to quantise. Recommended hybrid:

| Keep in int8 | Keep in fp32 | Why |
|---|---|---|
| `conv1`, `f_conv`, `conv2` | all 3 GroupNorms | GroupNorm re-normalises, so int8 input to it is well-conditioned |
| pointwise, dwconv, SE | all 3×3 = 9 LayerNorms | variance over 48 elements needs range |
| FFN 48→96→48 | all softmaxes | attention logits carry the relative position bias, which is *small*; int8 logits flip softmax branches |
| pool `Dense`, classifier | `expf`/`logf`/divide | |
| | **QKᵀ and AV** (2,046,240 MAC, 15 %) | cheapest place to buy numerical safety — costs 14 ms |

**Cost of that hybrid: 105 ms → 119 ms (+13 %).** It is cheap insurance. The
`GroupNorm` after `conv2` immediately before the conformer stack means the
quantisation noise injected by `conv2` is substantially normalised away; the
attention path has no such normalisation and is where accuracy will be lost
first. **Quantisation-aware fine-tuning with the same 24-bucket corpus and the
same TPR-at-FPR objective (`models/agents.md` §5) is a prerequisite, not an
afterthought** — and the result must be re-frozen against measured FA/hour on
real audio, because `models/agents.md` §9.1 is explicit that validation FPR does
not predict deployment FA.

---

## 6. Duty cycle

`duty = (inference time + front-end time) / hop period`, with front-end time from
§3.4. **This is the "the loop takes exactly one hop" model — see §6.1, which is
where the real firmware differs.**

| Hop | Front end | **fp32 committed (183 ms)** | fp32 restructured (105 ms) | int8 3-pass (117 ms) | **int8 1-pass (105 ms)** | int8 + `conv2`(3,1) (75 ms) | **int8 + `conv2` DS (72 ms)** |
|---|---|---|---|---|---|---|---|
| **100 ms** | 0.55 ms | **185.6 %** | 105.6 % | 117.6 % | 105.6 % | 75.6 % | 72.6 % |
| **200 ms** | 1.10 ms | **92.1 %** | 53.1 % | 59.1 % | **53.1 %** | 38.1 % | **36.6 %** |
| **300 ms** | 1.65 ms | **61.6 %** | 35.6 % | 39.6 % | **35.6 %** | 25.6 % | **24.6 %** |
| **500 ms** | 2.75 ms | **37.2 %** | 21.6 % | 24.0 % | **21.6 %** | 15.6 % | **15.0 %** |
| **1000 ms** | 5.40 ms | **18.8 %** | 11.0 % | 12.2 % | **11.0 %** | 8.0 % | **7.7 %** |

### 6.0 Which combinations meet < 10 % CPU?

| Configuration | Meets < 10 %? |
|---|---|
| fp32, as committed, any hop | **No.** Best case 18.8 % at a 1000 ms hop. |
| fp32, restructured accumulators | **No.** Best case 11.0 % at a 1000 ms hop. |
| int8 w+a, 3-pass stem | **No.** Best case 12.2 % at a 1000 ms hop. |
| int8 w+a, single-pass stem | **No.** Best case 11.0 % at a 1000 ms hop. |
| **int8 w+a, single-pass stem, `conv2` → depthwise-separable** | **Yes, at a 1000 ms hop only: 7.7 %.** |
| int8 w+a, single-pass stem, `conv2` → (3,1) | **Yes, at a 1000 ms hop only: 8.0 %.** |

> ### Verdict: **the < 10 % continuous-CPU constraint is not met at any hop the
> firmware can actually configure, except after reducing `conv2`, and then only
> at a 1000 ms hop** — which is itself blocked (§6.2) and which would break the
> model (§6.3).

### 6.1 The real firmware is worse than this table, and it is a bug

`kws_task` (`main/wake_word_main.c:527-548`) **blocks** on
`i2s_channel_read()` for a full hop, then pushes, then infers:

```c
i2s_channel_read(s_rx_chan, raw, hop * sizeof(int16_t), &n, portMAX_DELAY);  // :527-538
kws_frontend_push(raw, hop);                                                // :540
const float prob = kws_model_run(spec);                                     // :546
```

The period is therefore **`hop + t_infer`, not `hop`**, and the window advances
slower than the audio. At 183 ms inference and a 200 ms hop the real period is
383 ms, so the true duty is 48 % (not 92 %) and the effective hop is 383 ms
(not 200). The firmware's own log line reports `ema_inf_ms / HOP_MS`
(`wake_word_main.c:594`), which **understates the real duty by the factor
`(hop + t_infer) / hop` — 1.9× at the fp32/200 ms operating point.** Any
measurement taken from that log line is a measurement of the wrong quantity.

The consequence for latency is worse than a duty-cycle problem: because audio
backlogs in the 256 ms I2S DMA pool, the *frame grid itself* stretches, so hop
quantisation is measured against a 383 ms grid rather than 200 ms (§7.5).

**Fix:** drain I2S on core 0 on a timer, run inference on core 1, publish the
result through a queue. The current task layout pins `kws` and `led` to core 0
and only `store` to core 1 (`wake_word_main.c:641-645`), so **core 1 is idle and
available for this today.**

### 6.2 The firmware cannot currently use a hop larger than 200 ms

`KWS_MAX_HOP` is 3,200 samples (`main/kws_frontend.h:27`) and `kws_task` rejects
any hop above it (`main/wake_word_main.c:497-501`). That is 200 ms.
300 ms = 4,800 samples, 500 ms = 8,000, 1000 ms = 16,000 — **all rejected.**

The only cell in §6.0 that meets < 10 % is therefore unreachable without raising
`KWS_MAX_HOP`, which also grows `s_audio` (`main/kws_frontend.c:163`) by
`(hop − 3,200) × 2` bytes:

| Hop | `s_audio` size | Δ RAM |
|---|---|---|
| 200 ms (current) | 19,200 samples = 38,400 B | — |
| 300 ms | 24,000 samples = 48,000 B | +9,600 B |
| 500 ms | 24,000 samples = 48,000 B | +9,600 B |
| 1000 ms | 32,000 samples = 64,000 B | +25,600 B |

### 6.3 A 1000 ms hop is the wrong answer anyway

Two independent reasons, both fatal:

1. **Frame reuse collapses.** At a 1000 ms hop, `pending/320 = 50 > 49`, so
   `keep = 0` (`main/kws_frontend.c:229-238`): all 49 frames are recomputed and
   the sparse-filterbank/reuse optimisation buys nothing. The front end goes from
   1.10 ms to 5.40 ms (§3.4).
2. **The model was trained on arbitrary keyword positions inside a 1 s window.**
   A 1 s hop over a 1 s window means non-overlapping blocks: the keyword is
   clipped at the window edge roughly half the time, and the "roll the keyword
   along the time axis" stress test that drove model selection
   (`models/agents.md` §1.1) is never exercised. The measured consequence of
   temporal-position sensitivity on this model family is **26.5 points of
   streaming recall** — far more than any quantisation or `conv2` change will
   cost. Do not go past ~300–400 ms.

### 6.4 Energy gate — the honest way to read the < 10 % budget

`models/docs/ARCHITECTURE.md:20-24` specifies an RMS energy gate that skips
inference in silence. Under that design the < 10 % figure is an **idle-listening**
budget, not a continuous-speech budget, and the two differ by an order of
magnitude.

Break-even speech occupancy for a 10 % budget at a 200 ms hop (20 ms budget,
front end 1.10 ms):

| Configuration | 20 ms budget holds while speech occupies ≤ |
|---|---|
| fp32 as committed | 10.3 % of wall-clock |
| fp32 restructured | 18.2 % |
| int8 w+a, 3-pass stem | 15.4 % |
| **int8 w+a, single-pass stem** | **17.5 %** |
| **int8 w+a, 1-pass stem, `conv2` reduced** | **26.7 %** |

| Gated duty (int8 w+a, 1-pass stem, 200 ms hop) | Mean duty |
|---|---|
| Speech occupies 5 % of wall-clock | 2.7 % |
| Speech occupies 10 % | 5.9 % |
| Speech occupies 20 % | 11.2 % ✗ |
| Speech occupies 100 % (worst case) | 53.1 % ✗ |

**One implementation requirement:** gate the **model, not the front end**.
Gating the front end resets `s_primed`/frame reuse and forces the first
post-gate inference to rebuild all 49 frames (5.40 ms of front end plus a full
model pass). Keeping the mel grid live and gating only `kws_model_run()` costs
0 extra RAM and keeps every subsequent inference at the 10-frame cost.

### 6.5 Core split

| Core | Today | Recommended |
|---|---|---|
| Core 0 | `led` (p3), `kws` (p6) | I2S read + front end |
| Core 1 | `store` (p2), then idle | **`kws_model_run` + detector** |

This fixes §6.1 for free and keeps the WiFi task (when added) off the DSP core.
It also means the §6.0 percentages are **per-core**; if "< 10 % CPU" is read as
10 % of the *chip* (both cores), every table above is halved and the int8 +
`conv2`-reduced build at a 500 ms hop reaches 7.5 % of the chip. **State which
reading the requirement uses** — it is worth a factor of two.

---

## 7. End-to-end latency

**Metric:** keyword **end** → ASR server has the audio.

> **Status caveat:** the current firmware has **no network path**
> (`main/wake_word_main.c` captures to a WAV and writes it to flash). The
> transport row below is a *design projection* taken from
> `models/docs/ARCHITECTURE.md:100-117`, not a measurement. Everything above
> the transport is a prediction about the firmware as it stands.

### 7.1 Latency terms

| # | Term | Mean | p95 | Worst | Notes |
|---|---|---|---|---|---|
| L1 | **Hop quantisation** — the window must *close* after the keyword ends | 100 ms | 190 ms | 200 ms | 1 s window, 200 ms hop. Uniform. |
| L2 | **Inference** — the decision cannot exist until `kws_model_run` returns | 75 ms | 80 ms | 90 ms | int8 1-pass + `conv2` reduced |
| L3 | **2-of-2 confirmation** — the *second* hit is one full hop later, and its inference must also finish | **275 ms** | 275 ms | 275 ms | `CONFIG_EXAMPLE_NEED=2`, consecutive windows (`wake_word_main.c:566`) |
| L4 | **Refractory (1.5 s)** | **0** | 0 | 0 | `lockout` is set *at* the fire (`wake_word_main.c:571`), so it never delays the first detection. It delays a **repeat** by up to 7 hops = 1.4 s. |
| L5 | Detector bookkeeping + peak-hold | <0.01 ms | <0.01 ms | <0.01 ms | `wake_word_main.c:557-576` |
| L6 | **Capture opens** — `ring_peek_recent` (3,200 B memcpy) + waiting for the next 20 ms PCM frame | 10 ms | 20 ms | 20 ms | `wake_word_main.c:273`; the clip's first post-keyword sample arrives on the next feed frame |
| L7 | Core-1 → core-0 notify | 0.1 ms | 0.2 ms | 0.5 ms | design (`ARCHITECTURE.md:112`) |
| L8 | Pre-roll packet → WiFi airtime → server | 7 ms | 12 ms | 25 ms | 3,200 B; pre-established socket, no handshake |

### 7.2 Two designs

**Design A — current firmware, blocking I2S read.** The hop grid stretches to
`200 + 75 = 275 ms`, so L1 and L3 both grow.

| | Mean | p95 | Worst |
|---|---|---|---|
| L1 (grid = 275 ms) | 137 ms | 261 | 275 |
| L2 | 75 | 80 | 90 |
| L3 | 275 | 275 | 275 |
| L4–L5 | 0 | 0 | 0 |
| L6 | 10 | 20 | 20 |
| L7–L8 | 7 | 12 | 25 |
| **TOTAL** | **504 ms** | **648 ms** | **685 ms** |

**Design B — decoupled: I2S on core 0, inference on core 1, true 200 ms grid.**

| | Mean | p95 | Worst |
|---|---|---|---|
| L1 | 100 ms | 190 | 200 |
| L2 | 75 | 80 | 90 |
| L3 | 275 | 275 | 275 |
| L4–L5 | 0 | 0 | 0 |
| L6 | 10 | 20 | 20 |
| L7–L8 | 7 | 12 | 25 |
| **TOTAL** | **392 ms** | **502 ms** | **535 ms** |

**Design C — Design B with 1-of-2 confirmation** (removes L3 entirely).

| | Mean | p95 | Worst |
|---|---|---|---|
| **TOTAL** | **117 ms** | **227 ms** | **260 ms** |

### 7.3 Which term dominates?

| Term | Worst-case share (Design B) |
|---|---|
| **L1 + L3 (the detector's hop grid)** | **475 ms = 89 %** |
| L2 (inference) | 90 ms = 17 % |
| L6 (capture opens) | 20 ms = 4 % |
| L7–L8 (transport) | 25 ms = 5 % |

> **The dominant term is the detector's hop grid, not the model.** Three hops
> (quantisation + two-inference confirmation) is 89 % of the worst case.
> Cutting inference from 183 ms to 72 ms saves 111 ms; halving the hop saves
> 100 ms *and* doubles CPU. The model is the expensive half of the problem, but
> the **latency** is decided by the detector.

Reducing in order of effect:

1. **L3 — change the confirmation policy.** 2-of-2 costs a full hop *plus* an
   inference. `models/agents.md` §10.2 measures this directly: 1-of-2 buys
   **16–20 points of streaming recall**, at the cost of FA rising from 4.42 to
   16.08 FA/h on v11. If the FA budget is the binding constraint, 2-of-3 over
   non-consecutive windows keeps the false-alarm rate but does **not** reduce
   latency (still one hop). **L3 cannot be reduced without either accepting the
   FA cost or accepting non-consecutive windows.**
2. **Decouple inference from the I2S read** (Design B). Removes 112 ms of mean
   latency for zero compute. Free.
3. **L1 — smaller hop.** Half the hop costs half the quantisation and doubles
   CPU. At 100 ms, L1+L3 falls from 475 ms to 237 ms — but §6.0 shows nothing
   runs in 100 ms of budget.
4. **L2 — inference.** 105 ms → 72 ms via §7.4. Worth doing, worth 33 ms.
5. **L6 — open the capture from the ring, not from the live feed.** The 1 s
   pre-roll already contains 0.9 s of post-keyword audio in the common case
   (the window is 1 s long, so the ring holds up to 1 s *ending* at the window
   close). Seeding the clip entirely from the ring and only switching to live
   PCM when the ring runs out removes most of L6. Worth ~10 ms.

### 7.4 The hop / latency tension, and the recommended operating point

CPU and latency pull in opposite directions and cannot both be satisfied:

| Hop | Mean latency (Design B, 2-of-2) | Duty (int8 + `conv2` DS) |
|---|---|---|
| 100 ms | 192 ms | 72.6 % |
| 200 ms | **392 ms** | **36.6 %** |
| 300 ms | 542 ms | 24.6 % |
| 500 ms | 842 ms | 15.0 % |
| 1000 ms | 1,592 ms | 7.7 % |

**Recommended operating point: 200 ms hop.** Reasoning:

* Below 200 ms, latency improves linearly but CPU degrades linearly, and
  §6.0 shows nothing is close to viable. 100 ms is 72.6 % of a core.
* Above 200 ms, CPU improves but latency degrades *linearly* and §6.3 shows the
  model degrades too, because frame reuse collapses and the keyword gets clipped
  at the window edge.
* 200 ms is the only hop the firmware can actually configure without raising
  `KWS_MAX_HOP` (`main/kws_frontend.h:27`), and it is the `sdkconfig.defaults:39`
  value, and it is the largest multiple of 320 samples inside 3,200.
* Mean latency 392 ms with a worst case of 535 ms. The metric is
  keyword-end-to-server; 392 ms is dominated by a term (L3) that a detector
  policy change, not an engineering change, can address.

**If the graded latency target is tighter than ~250 ms, the lever is the
detector policy (1-of-2), not the model.** Dropping to 1-of-2 takes the mean to
117 ms with the model unchanged. That is a deliberate FA-for-latency trade and it
should be made explicitly and re-measured on real audio
(`models/agents.md` §10.2, §9.1), not by accident.

---

## 8. Recommendations

### 8.1 The realistic operating point

| Parameter | Value | Rationale |
|---|---|---|
| Precision | **int8 weights + int8 activations**, per-tensor symmetric, 76 scales; attention QKᵀ/AV and all norms/softmaxes in fp32 | §4.7, §5.5 |
| Stem | **single pass, int8 `(49,20,32)` = 31,360 B materialised** | §7.3 |
| `conv2` | **depthwise-separable `(3,3,32)` DW + `(1,1,32→48)` PW** — requires retraining | §1.4 |
| Weights | 85,169 B flash | §4.5 |
| RAM | **237.2 KB internal peak** (146.0 KB `.bss` + 91.2 KB heap/system) | §4.7 |
| Hop | **200 ms** (3,200 samples) | §7.4 |
| Core split | I2S + front end on core 0, inference + detector on core 1 | §6.5 |
| Flash recorder | **off in the graded configuration** | §4.8 |
| Detector | 2-of-2 peak-hold at 0.68 (revisit 1-of-2 if latency is graded) | §7.3, `models/agents.md` §10.1 |
| **Expected** | **6.65 MMAC, 72 ms inference, 36.6 % of one core, 1.10 ms front end** | §5.3, §6.0 |

### 8.2 Does it meet the constraints?

| Constraint | Verdict |
|---|---|
| **< 256 KB RAM** | ✅ **Yes, at 237.2 KB** (int8 w+a, single-pass stem), *provided* the flash recorder is disabled and WiFi's ~40 KB fits in the 19 KB remaining headroom — which it does not. **Tight to the point of needing one more cut**; see §8.3 rank 6. |
| **< 10 % CPU (continuous speech)** | ❌ **No at any hop the firmware can configure.** 7.7 % at a 1000 ms hop, which is blocked and which breaks the model. |
| **< 10 % CPU (gated, idle listening)** | ✅ **Yes: 0.55 %** (front end only, 1.10 ms / 200 ms), and ≤ 10 % while speech occupies ≤ 26.7 % of wall-clock. |
| **Latency** | ⚠️ 392 ms mean / 535 ms worst with 2-of-2; **117 ms / 260 ms with 1-of-2**. 89 % of it is the hop grid, not the model. |

> ### The single most important sentence in this document
>
> **`bcconformer_v3` at 84,865 parameters is roughly 4× too large to run
> continuously on an ESP32-S3 within 10 % CPU at a hop short enough to be
> useful.** To hit 10 % *continuously* at a 200 ms hop the network must come
> down to **≤ 3.3 MMAC** (4.1× reduction) — which means cutting the conformer
> stack, not just `conv2`. `models/agents.md` §9.3 is explicit that capacity is
> not the bottleneck for this model family (the 138 K-parameter variant tracked
> the 50 K one at 2.5× the cost), so a 4× cut will cost recall, and the FA budget
> will be the thing that breaks first.
>
> The defensible reading is that **< 10 % is an idle-listening budget** (which
> the gated design meets at 0.55 %), and that the honest statement to make to
> whoever set the constraint is: *continuous-speech CPU at a 200 ms hop is 37 %
> of one core for this model, and 2–3× that is not recoverable by
> quantisation.*

### 8.3 Optimisations, ranked by measured effect

| Rank | Optimisation | Effect | Cost / risk |
|---|---|---|---|
| **1** | **int8 weights + int8 activations** | RAM 341.1 → 237.2 KB (**30 %**), inference 183 → 105 ms (**1.75×**), `conv2` kernel 55,296 → 13,824 B so it becomes data-cache-resident | QAT + re-freeze on real audio. **Never do int8 weights alone** — 3.0 cyc/MAC, slower than fp32 and zero RAM saving. |
| **2** | **Reduce `conv2` (50 % of all compute)** | 6.77 → 1.79 MMAC (depthwise-separable) or 2.26 MMAC ((3,1) full). Total 11.63 → **6.65 MMAC**; inference 105 → **72 ms** (**−31 %** overall) | **Requires retraining.** The single highest-leverage change by a wide margin. `conv2` at (3,3,32,48) is a generic early-layer pattern with no evidence it needs full channel mixing at every tap. |
| **3** | **Single-pass stem** (enabled *by* #1) | −1.88 MMAC (**−14 %**), and 2 fewer GN accumulate/apply passes | Costs 28,160 B. **Not available in fp32** — the tensor is 125,440 B. |
| **4** | **Decouple inference from the I2S read loop; move it to core 1** | **Zero compute. −112 ms mean latency, and the duty figure stops being wrong.** | Core 1 is already idle. Cheapest item on this list. |
| **5** | **Energy-gate the model, not the front end** | Idle duty 53 % → **0.55 %**; the < 10 % budget is met for the state it was written for | Gating the front end destroys frame reuse and costs 5.4 ms per post-gate inference. |
| **6** | **Attack `s_stem` / activation RAM further** | 31,360 B is 34 % of the int8 activation budget; the 19 KB headroom over 256 KB is not enough for WiFi's ~40 KB | Overlap `s_ln`/`s_attn`/`s_blk` (all `(49,48)`, strictly sequenced — assumption A2 currently charges all three) → **−18,816 B**. Cheap, provable, no accuracy cost. Move `s_audio` (38,400 B) to PSRAM — it is read sequentially and never DMA'd → **−38,400 B**. Together: **57,216 B**, which turns 19 KB of headroom into 76 KB. |
| **7** | **Change the confirmation policy** (not a cost optimisation — a latency one) | **−275 ms**, 70 % of total latency | FA 4.42 → 16.08 FA/h on v11 (`models/agents.md` §10.2). Explicit trade, re-measured. |
| **8** | **Pruning** | — | **Ranked last deliberately.** `models/agents.md` §9.3 shows capacity is not this model's bottleneck. Unstructured pruning of a hand-ported C forward pass is hard to realise on the LX7 anyway (no 2:4 sparsity support). The one exception: structured pruning of `conv2`'s **output** channels, which is exactly what depthwise-separable decomposition (rank 2) achieves for free. |
| **9** | **Restructure the fp32 accumulators** (4 partial sums) | 183 → 105 ms in fp32, no retraining | **Only worth it if you stay in fp32.** Once int8 lands, fp32 arithmetic and int8 arithmetic converge to within 20 % (§5.4). |

### 8.4 What I would not do

| Tempting | Why not |
|---|---|
| int8 weights only | 3.0 cyc/MAC — slower than fp32 *and* zero RAM saving. Strictly dominated. |
| Hop → 1000 ms to hit 10 % CPU | Blocked by `KWS_MAX_HOP`; collapses frame reuse; clips the keyword at the window edge. Measured recall cost in this model family: 26.5 points. |
| Unstructured pruning to hit 3.3 MMAC | §9.3 of `models/agents.md`: capacity is not the bottleneck. You would spend recall and get nothing back. |
| int8 the softmaxes and LayerNorms to save a few KB | 3 % of cycles and a silent recall regression. Keep them fp32. |
| Treat PSRAM as free to clear the 256 KB budget | The 672 KB capture path still has to exist somewhere, and the ASR path needs the memory more than the WAV recorder does. |
| Adopt the ARCHITECTURE.md "12 ms via ESP-SR" figure | That is a different model on a different code path, and ESP-SR has been removed from this build. `models/docs/ARCHITECTURE.md:24` is stale. |

---

## 9. How to verify this

**These are predictions. Every number in §5 and §6 must be measured on the
ESP32-S3 before it is believed.** Ranked by how much it would change the
conclusions.

### 9.1 The two measurements that decide everything

**M1 — effective cycles/MAC for `conv2_row`.** This single number separates
"int8 is 1.8×" from "int8 is 1.0×", and it is the one I am least sure of.

```c
/* tools/, or a temporary hook in kws_model.c:411 */
uint32_t c0 = esp_cpu_get_cycle_count();
for (int rep = 0; rep < 1000; rep++) {
    conv2_row(a, b, c, out);          /* 6,773,760 MAC in the real call,
                                         138,240 MAC in one row           */
}
uint32_t c1 = esp_cpu_get_cycle_count();
printf("%.3f cyc/MAC\n", (double)(c1 - c0) / (1000.0 * 138240.0));
```

Expect 2.2 for the committed single-accumulator loop, ~1.1 if you restructure.
`esp_cpu_get_cycle_count()` is in `<esp_rom_sys.h>`; it reads the Xtensa
`CCOUNT` register and counts core cycles, not instructions.

**M2 — the cache-miss story.** If `conv2_row`'s inner loop is weight-fetch-bound,
int8 wins big. If it is FPU-bound, int8 wins 1.2× and the whole int8 argument
weakens. Read the data-cache statistics either side of the call:

```c
esp_cache_stats_t st;                  /* <esp_cache.h> */
esp_cache_get_stats(ESP_DATA_CACHE, &st);
```

If the miss count is proportional to 2.7 MB (the `conv2` weight traffic), the
memory argument holds. Also try `CONFIG_ESP32S3_DATA_CACHE_64KB=y` against a
smaller cache, or move the `conv2` kernel to PSRAM-and-cache — the effect on
`cycles/MAC` is the measurement.

### 9.2 What the firmware already measures

`kws_task` already brackets `kws_model_run` with `esp_timer_get_time()`
(`main/wake_word_main.c:545-551`) and logs an EMA plus a duty figure every 5 s
(`:588-597`):

```
listening  p=0.031 hold=0.026  infer=NNN ms  hop=200 ms  duty=NN.N%  heap=NNNNN
```

**Read `infer=NNN ms` and use it directly** — it is a µs-resolution measurement
of exactly L2 in §7.1. **Ignore `duty=NN.N%`** — it is `ema_inf_ms / HOP_MS`
(`:594`), which omits the `t_infer` component of the period (§6.1) and therefore
understates the real duty by `(hop + t_infer) / hop`.

### 9.3 Per-stage breakdown

Add `CCOUNT` brackets inside `kws_model_run` (`main/kws_model.c:621`) at these
boundaries, and compare against §5.2's line items:

| # | Bracket | Predicted (fp32) |
|---|---|---|
| 1 | stem pass 1 (`:625-629`) | 846,720 × 2.2 = 1.86 Mcyc |
| 2 | stem pass 2 (`:634-640`) | 940,800 × 2.2 = 2.07 Mcyc |
| 3 | stem pass 3 + conv2 (`:644-666`) | 7.71 Mcyc |
| 4 | GroupNorm-3 + swish (`:670-677`) | 23,520 × 2.2 + 23,520 × 32 = 0.80 Mcyc |
| 5 | mel gate (`:681-702`) | 0.05 Mcyc + 490 × 32 = 0.02 Mcyc |
| 6 | each conformer block (`:706-709`) | 1.27 MMAC × 2.2 = 2.80 Mcyc each |
| 7 | pool + classifier (`:712-720`) | 0.13 Mcyc |

Also bracket `kws_frontend_compute` (`main/wake_word_main.c:541`) to confirm
§3.4's 1.10 ms at a 200 ms hop, and check that it really only recomputes 10
frames.

### 9.4 Transcendental costs

```c
uint32_t c0 = esp_cpu_get_cycle_count();
volatile float s = 0.0f;
for (int i = 0; i < 100000; i++) { s += expf(-0.001f * i); }
uint32_t c1 = esp_cpu_get_cycle_count();
printf("expf: %.1f cyc\n", (double)(c1 - c0) / 100000.0);
```

§5.1 assumes 32 cycles. At 127,371 calls, every 1 cycle of error here is
127 kcycles = 0.53 ms per inference. Do the same for `logf` (assumed 40) and
float divide (assumed 12) — with 83,937 divides, a 2× error is 1.0 ms.

### 9.5 FreeRTOS run-time stats

Enable `configGENERATE_RUN_TIME_STATS` and `configUSE_TRACE_FACILITY` with
`esp_timer_get_time` as the counter, then `uxTaskGetRunTimeStats()` once a
second. This gives the **true** per-task duty including everything the
`esp_timer` bracket misses, and it is the number to put in the acceptance
report. Also run `vTaskGetRunTimeStats` with inference on core 0 vs core 1 to
confirm the split in §6.5.

### 9.6 RAM

```c
ESP_LOGI(TAG, "internal free=%u  psram free=%u  min-ever-internal=%u",
         heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
         heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
         heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL));
```

* Cross-check §4.4 / §4.7 against the linker map: sum the `.bss` symbols from
  `build/wake.map`. The predicted fp32 figure is 255,956 B and the predicted int8
  1-pass figure is 149,524 B.
* `heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL)` is the one that catches
  a transient peak the snapshot misses.
* **Re-run the whole ledger with `MALLOC_CAP_SPIRAM` printed separately** —
  §4.8's PSRAM total of 695 KB is an arithmetic prediction from the clip lengths
  and has not been measured.
* Watch the 96 KB `SPIRAM_MALLOC_RESERVE_INTERNAL` interaction
  (`sdkconfig.defaults:17`): if the internal heap is *not* shrinking as PSRAM is
  allocated, the reserve is doing its job; if it is, the reserve is set too low.

### 9.7 Accuracy gate — the check that matters most

Any quantisation or `conv2` change is a **model** change, and `models/agents.md`
§9.1 is emphatic that validation metrics do not predict deployment behaviour.

1. Run the host harness `tools/test_kws_host.c` against the JAX reference and
   confirm bit-parity (or the expected quantisation error) **before** touching
   the firmware. The debug buffers are already `#ifdef KWS_HOST_TEST`
   (`main/kws_model.c:83-91`) so this costs nothing on the device.
2. Re-freeze the operating point on the real corpus: the 24-bucket
   `dataset_v5`, TPR-at-FPR objective, **and** the 4.97 h soundscape + 0.576 h
   LibriSpeech FA measurement. A quantised model that looks fine on validation
   and fires on continuous speech is the documented failure mode here.
3. Re-measure the streaming recall **at the deployed hop** with the deployed
   detector. The offset-stress column in `models/agents.md` §1.1 is the proxy;
   a retrained depthwise-separable `conv2` will have a *different* temporal
   position profile, so the proxy must be re-measured, not inherited.
4. Compare against the frozen v11 point (threshold 0.68, 2-of-2, 37.0 %
   streaming recall at 5.81 FA/h). Any variant that exceeds **6 FA/h** at
   threshold 0.68 is rejected, whatever it does to latency.

### 9.8 Latency

Once the network path exists, instrument the *actual* path, not the design in
`ARCHITECTURE.md`:

* Timestamp with `esp_timer_get_time()` at: window close, `kws_model_run`
  return, `hits >= 2` satisfied (`wake_word_main.c:566`), `capture_start()`
  return (`:284`), first byte handed to the socket.
* On the server, log receive time minus the client's T0 header timestamp
  (`ARCHITECTURE.md:29` already carries `timestamp_ms`).
* **Sample over thousands of trials and report mean *and* the distribution.**
  L1 is uniform on `[0, hop]`, so the mean is `hop/2` but the p95 is `0.95·hop` —
  reporting only the mean hides the tail, and the tail is what a user
  experiences.

---

## 10. Out-of-scope observations (found while costing, not resource issues)

Recorded because they materially affect the validity of the numbers above, and
flagged rather than fixed since this document's scope is estimation.

1. **Missing swish in the stem.** `kws_model_run` (`main/kws_model.c:625-666`)
   applies `swish` after neither GroupNorm-1 nor GroupNorm-2, but the trained
   graph applies both. The stem output that feeds `conv2` is therefore on a
   different function than the one the weights were fitted for. This is
   independent of the missing diagnostic block, which *did* apply swish. The
   §3.1 `expf` table uses the graph-faithful count; fixing this will roughly
   double stem transcendental cost and change measured TPR. **Worth measuring
   before any quantisation work** — a quantised model fitted to the un-swished
   stem is quantising the wrong function.
2. **`kws_debug_build_feat` documents the SAME-padding contract as zeros**
   (`main/kws_model.c:283-304`) and the code matches, but the `delta_stack`
   derivation quoted in the comment computes `d1`/`d2` with `mode="edge"`
   clamping. Those are two different boundary conditions at the first and last
   frame and both are exercised. Host-harness parity is the arbiter; §9.7 step 1.
3. **The 3-pass stem is only needed because of RAM, and RAM is the constraint
   being optimised.** This is not a defect — it is a coherent trade — but it
   means rank 3 in §8.3 (single-pass stem) is only reachable *through* rank 1
   (int8 activations). They are not independent choices.

---

## Appendix A — constants

| Constant | Value | Source |
|---|---|---|
| Sample rate | 16,000 Hz | `kws_frontend.h:16` |
| `n_fft` / `win` / `hop` | 512 / 480 / 320 | `kws_frontend.h:17-19` |
| Mel bands | 40 (100–7,500 Hz) | `kws_frontend.h:20` |
| Frames per window | 49 | `kws_frontend.h:21` |
| Window samples | 16,000 (1.000 s) | `kws_frontend.h:22` |
| `KWS_MAX_HOP` | 3,200 samples (200 ms) | `kws_frontend.h:27` |
| Mel weights (sparse) | 418 floats | `main/model/kws_frontend_data.h` |
| Parameters | 84,865 | `main/model/kws_model_data.h:11` |
| `D` / `H` / `dh` | 48 / 4 / 12 | `kws_model.h:22-24` |
| `C1` / `C2` / stem mels | 32 / 48 / 10 | `kws_model.h:25-27` |
| Blocks / SE / pool hidden / features | 3 / 6 / 24 / 97 | `kws_model.h:28-31` |
| Threshold / need / hold decay / refractory / hop | 0.68 / 2 / 0.85 / 1500 ms / 200 ms | `sdkconfig.defaults:35-39` |
| Clip / pre-roll | 10 s / 1 s | `sdkconfig.defaults:48-49` |
| Flash / PSRAM / clock | 16 MB / 8 MB octal @ 80 MHz / 240 MHz | `sdkconfig.defaults:5,13,20` |
| Data cache / I-cache | 64 KB / 32 KB | `sdkconfig.defaults:22-23` |
| PSRAM malloc policy | always-internal ≤ 64 KB, reserve 96 KB | `sdkconfig.defaults:16-17` |

## Appendix B — summary of headline numbers

| Quantity | fp32 | int8 w+a | Source |
|---|---|---|---|
| MAC / inference | 13,512,986 | **13,512,986 (identical)** | §1.3, §2 |
| MAC with single-pass stem | — | 11,631,386 | §8.1 |
| MAC with `conv2` depthwise-separable | — | 6,645,146 | §8.3 |
| `expf` / inference | 127,371 | 127,371 (must stay fp32) | §3.1 |
| FLOP / inference | ≈ 32.4 M | ≈ 32.4 M | §1.5 |
| Cycles / inference | 10.5–73.3 M (central 43.8 M) | 15.4–31.9 M (central 22.7 M) | §5.2, §5.3 |
| **ms / inference @ 240 MHz** | **183 (105–305)** | **105 (86–150)** | §5.2, §5.3 |
| Weights (flash) | 339,460 B | 85,169 B | §4.5 |
| Model activations (RAM) | 193,416 B | 92,864 B (1-pass) | §4.2, §4.6 |
| Static `.bss` total | 255,956 B | 149,524 B | §4.7 |
| **Peak internal SRAM** | **349,274 B = 341.1 KB** | **242,842 B = 237.2 KB** | §4.7 |
| PSRAM | ~695 KB | ~695 KB | §4.8 |
| Duty @ 200 ms hop | 92.1 % | 53.1 % (36.6 % with `conv2` DS) | §6.0 |
| Latency, mean / worst (200 ms hop, 2-of-2) | 504 / 685 ms | 392 / 535 ms | §7.2 |
| Latency, mean / worst (1-of-2) | 229 / 410 ms | 117 / 260 ms | §7.2 |
