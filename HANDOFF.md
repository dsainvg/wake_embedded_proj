# HANDOFF

State as of the int8 rewrite. The float32 parity defect this file used to lead
with is fixed and verified; what remains is on-device measurement, which needs
a board that is not currently connected.

---

## 1. Where things stand

| | |
|---|---|
| Numerical parity vs JAX | **verified**, worst error 5.63e-4 on a 0.68 threshold |
| Bit-identical determinism | **verified**, host harness, all 6 fixtures, repeat and across processes |
| Inference time on hardware | **334 ms measured** -- 167% of the 200 ms hop, over budget |
| Weights in flash | 80,624 B int8 (was 339,460 B float32) |
| App binary | 358,416 B `wake.bin` (was 586,911); 89% of the 3 MB app partition free |
| Internal DRAM | 226,539 B (66%) -- 115,221 B free |
| Firmware builds | **yes**, `idf.py build` clean |
| Hardware run | **yes**, twice: numbers below, LED never fired |
| PSRAM static (.bss) | 265,408 B of 8 MB |

Verify all of the above yourself before trusting it:

```bash
tools\build_host.bat
build\host\kws_host.exe
```

---

## 2. What changed, and why

### 2.1 The forward pass was rewritten (`main/kws_model.c`)

The old version computed `conv1` three times, feeding `conv2` through a
three-frame rolling ring so the `(49,20,32)` float32 activation — 125 KB — never
had to exist. The ring could not represent the two boundary cases `conv2`
actually needs, and reading a ring slot that had not been written yet was the
class of bug that made the port disagree with training.

In int8 the same tensor is 31,360 bytes, so materialising it costs nothing that
matters and deletes the ring, both edge cases, and the failure mode.
GroupNorm takes its statistics over every frame, so materialising is also what
makes the stem three straight passes instead of two extra evaluations of `conv1`.

The stem is four separate int8 buffers rather than one reused buffer:

| buffer | contents | bytes |
|---|---|---|
| `s_c1q` | raw `conv1` output | 31,360 |
| `s_h1q` | after GroupNorm-1 + swish | 31,360 |
| `s_fq` | raw depthwise mel conv | 31,360 |
| `s_stq` | the tensor `conv2` reads | 31,360 |

Sharing one would save 31 KB out of 8 MB of PSRAM and cost a pass whose only job
would be to be got right.

### 2.2 Custom int8 kernels (`main/kws_int8.c`, `main/kws_xtensa.S`)

Two reductions cover the whole network: a conv over `(time, mel)` and a GEMM.
Both are `out[o] += sum_i q[i] * w[o][i]`, with the input channel as the
reduction axis.

**The layout decision is the important one.** Flax stores kernels `(in, out)`,
which makes the reduction strided by `out` — impossible to feed four MACs from
one 32-bit load. `tools/export_weights.py` therefore reverses every tensor's
axes at export time, and pads every reduction to a multiple of 16 so both
operands of the dot product are 16-byte aligned. That buys the whole speedup;
nothing else does.

**Activation scales are measured per row, per inference.** There is no
calibration set and there should not be one: a scale fitted in one room is wrong
in the next, and the failure mode is a wake word that stops firing at a level it
used to fire at. The cost is one pass over at most 288 bytes against a reduction
of at least 288 MACs.

**`conv2` is 60% of the network's 11.24 MMAC** and the conformer Dense layers
another 25%, so those are the two that are int8. `conv1`, `f_conv` and the
depthwise convolutions stay float32: together 7.6% of the work, and keeping
`conv1` exact means the stem is a clean reference.

**`EE.VMULAS.S8.ACCX` is the only thing that makes int8 faster than float32 on
this core.** There is no dot-product unit and no FMA, so a scalar int8 MAC costs
a byte load, a two-instruction sign extension, a multiply and an add — about
four instructions, *worse* than float32's hardware `FMUL.S`/`FADD.S`. The vector
instruction is sixteen MACs per instruction off two 128-bit loads.

GCC cannot allocate a Q register for an inline-asm operand, so the loop is a
`.S` file. `kws_int8.c` re-checks both preconditions (alignment, length multiple
of 16) before entering it and falls back to the portable loop if either fails,
so the assembly is an optimisation and never a correctness requirement.

### 2.3 What stayed float32, on purpose

| tensor | why |
|---|---|
| GroupNorm / LayerNorm scale and bias | added to or multiplied by *every* activation, so their quantisation error arrives intact at the logits rather than averaging down |
| the attention relative position bias | added straight into a softmax; a 0.01 logit step is a 1% error in one frame's attention weight |
| `conv1`, `f_conv`, depthwise-over-time | 7.6% of the work, depthwise or tiny |
| LayerNorm itself | an error here is a directly proportional error in which frames the network listens to |
| attention `QKᵀ`, `AV` | the softmax sits between them |

All of it is 22,844 bytes of flash. Not worth compromising the conditioning of
everything downstream for.

---

## 3. Files

| file | role |
|---|---|
| `main/kws_model.c/.h` | the network; owns the tensors and the activation buffers |
| `main/kws_int8.c/.h` | quantiser, conv and GEMM kernels, float32 twins, boot self-tests |
| `main/kws_xtensa.S` | the vector reduction |
| `main/model/kws_model_data.h` | **generated** tensor table, scales, float parameters |
| `main/model/kws_model_i8.bin` | **generated** int8 weight blob |
| `main/model/kws_model.bin` | **generated** float32 blob — host harness only, not in the firmware |
| `main/model/reference/*.npy` | **generated** fixture spectrograms and reference probabilities |
| `tools/export_weights.py` | the exporter; owns the layout and padding rules |
| `tools/capture_jax.py` | wraps the live Flax modules to get ground truth |
| `tools/host_check.c` | parity harness; builds the firmware sources for the host |

`tools/dump_stages.py`, `tools/compare_stages.py`, `tools/eval_int8.py`,
`tools/stage_ref.py`, `tools/test_feat_only.c` and `tools/test_kws_host.c/.bat`
were **deleted**. The first four were hand-written reimplementations of the model
in numpy/jax and every one of them disagreed with the checkpoint; keeping them
was worse than having no reference at all.

---

## 4. Verified numbers

Parity, `build\host\kws_host.exe`, int8 weights. The harness is built with
`-msse2 -mfpmath=sse`: the 32-bit MinGW default is x87, whose 80-bit registers
carry extra precision past the float return, which made the harness's own
bit-identical determinism check report sub-ULP differences that were compiler
artifacts, not model nondeterminism. SSE arithmetic is also what the Xtensa
target's IEEE-single math looks like. Both harness lines `fast exp:` and
`fast rcp:` report whether the two approximations are in use or fell back -- a
silent fallback is indistinguishable from a slow build, so it is printed.

| fixture | JAX | C | error |
|---|---|---|---|
| noise    | 0.02721036 | 0.02664748 | 5.6e-4 |
| tone1k   | 0.00004640 | 0.00004754 | 1.1e-6 |
| sweep    | 0.00061473 | 0.00070501 | 9.0e-5 |
| silence  | 0.00008417 | 0.00009702 | 1.3e-5 |
| quiet    | 0.01193262 | 0.01159632 | 3.4e-4 |
| loud     | 0.02950655 | 0.03003751 | 5.3e-4 |

Budget: 5e-3. Worst observed: 5.6e-4. The float-weights column in the harness
output isolates the cost of quantising the *weights* from the cost of quantising
the *activations*. Determinism: the harness's repeat check (same input twice,
bit for bit) passes on every fixture, and separate process runs produce
identical numbers.

---

## 4a. Measured on hardware, and what it cost

Two `idf.py flash monitor` sessions. Boot verdicts on the second run, all three
fast paths live and none fallen back:

```
int8 kernel: xtensa vector
fast exp:   in use (worst rel 2.86e-06)
fast rcp:   in use (worst rel 2.18e-07)
fast rsqrt: in use (worst rel 5.05e-07)
```

Per-stage, averaged over the runs between stats lines (`infer=334 ms`):

| stage | ms | share |
|---|---|---|
| blocks (3 conformer blocks) | 100.9 | **30.1%** |
| conv2 | 81.6 | **24.4%** |
| stem2 (GroupNorm-1 + swish + depthwise) | 49.3 | 14.7% |
| stem3 (GroupNorm-2 + residual + broadcast) | 39.2 | 11.7% |
| stem1 (conv1 + quantise) | 35.6 | 10.6% |
| gn3+sw (GroupNorm-3 + swish) | 24.0 | 7.2% |
| melgate | 3.2 | 0.9% |
| pool+head | 1.0 | 0.3% |

Three things follow, and only the first is obvious.

1. **It is not arithmetic-bound.** Removing ~1M operations per inference (the
   conv2 requantisation, 86,000 divisions, 294 `sqrtf`) moved the total from
   315 ms to 334 ms -- i.e. not at all. The fast paths are confirmed in use, so
   the work really is gone. Work removal that buys nothing means the time is
   going somewhere other than the arithmetic.

3. **It is memory-bound for ACTIVATIONS, at about 60 MB/s, and call-bound in
   conv2.** The mel gate is the clean measurement: 47,000 float MACs in 3.1 ms
   is ~16 cycles per MAC where FMUL+FADD should be ~2, and in the same 3.1 ms
   it reads 188 KB of conv2 output. 188 KB / 3.1 ms = 60 MB/s, which is what
   this part's PSRAM delivers on a streaming pass. Separately, conv2 was making
   211,680 calls into the assembly reduction for two useful vector MACs each.

   The obvious suspect -- weight matrices re-read once per output element, ~9.5
   MB per inference -- turned out **not** to be it: reversing those loops moved
   334 -> 330 ms. The weights are in flash `.rodata`, which this part caches.
   The 60 MB/s ceiling applies to the activation tensors, which are in PSRAM and
   are 94 KB each.

Also fixed from these logs: the kws task now subscribes to the task watchdog and
resets it in its loop. The watchdog was only being fed indirectly, by IDLE0, so
a task that never yields starved it and tripped it without anything hanging.

The LED logic is not implicated: `wake_word_main.c` lights it and calls
`capture_start()` on every detection, and `p` never rose above 0.03.

Projected CPU, `~0.4` instructions per MAC on the 85% that is vectorised:
**~15 ms per inference, ~7-8% duty at a 200 ms hop**. This is an arithmetic
estimate, not a measurement. The previous float32 figure of ~56 ms / 28% was
measured on hardware.

---

## 5. CPU: what was measured, what is done, what is left

The ~15 ms projection in section 4 was arithmetic on the MAC count and it was
wrong by 20x. It counted instructions per MAC and never counted the scalar
float work wrapped around the int8 reductions. On a core with no FMA, no dot
product unit and a software `__divsf3`, that wrapped work is the program.

The int8 rewrite was *slower* than the float32 build it replaced (~56 ms
measured on hardware). That is not inherent to int8 -- it is per-element
requantisation and division stacked on top of a reduced MAC count.

**Round two, after the stage breakdown showed the problem was memory, not arithmetic:**

| change | what it removes |
|---|---|
| conv2 loop order | the 13,824-byte weight matrix was fetched once per output element -- 490 times, 6.46 MB per inference. Now the frame is outermost, one frame's 10 mel windows are gathered into 2,880 B of DIRAM scratch, and each output channel's weights are fetched **once** and applied across all 10. 49 passes instead of 490 |
| GEMM loop order | same inversion in `kws_gemm_i8`/`_f32`: output channel outermost, so a weight row is fetched once and walked across all 49 input rows while hot. Was 49 passes over each Dense matrix, ~3 MB per inference |
| buffer placement | measured, not guessed. The conformer working set (`s_seq`, `s_ln`, `s_qkv`, `s_attn`, `s_blk`, `s_ffn`, 47 KB) and `s_stq` (31 KB, read 490 times by conv2) moved from PSRAM to internal DIRAM. DIRAM is now 66% / 115 KB free; PSRAM .bss fell 385 KB -> 265 KB |

The buffer move was tried at full size first -- all four stem int8 tensors in
DIRAM -- and came out at **97% of DIRAM with 9 KB free**, which is not a safe
place to be with DMA buffers and task stacks still to allocate. `s_c1q`,
`s_h1q` and `s_fq` went back to PSRAM: they are touched twice each and do not
earn 94 KB of DIRAM that `s_stq` and the conformer set use far better.

**Third round, from the first stage breakdown (see 4a). conv2 was 71.8 ms
against a 1.8 ms arithmetic ideal, and the reason was call count:**

conv2 reduces nine taps of 32 channels per output channel. Done as nine calls,
that is 49 x 48 x 10 x 9 = **211,680 calls per inference for 423,360 useful
vector MACs** -- two `EE.VMULAS.S8` each, behind a ~15-instruction prologue
(`entry`, `zero.accx`, block count, prime, the two-cycle accumulator read,
`retw.n`). A 7:1 overhead-to-work ratio.

`kws_asm_dot_taps_i8` in `kws_xtensa.S` walks all nine taps inside **one** call
and writes nine int32 partials; the caller applies the per-tap scales. 23,520
calls instead of 211,680. It has no FPU opcodes and uses only a2..a7 -- the
assembler rejects `t0`-`t3` outside ESP-IDF's macro headers, `loopgtz` needs a
register the caller cannot see, and `addi` takes an immediate, not a register.

It is covered by `kws_kernel_selftest` over tap counts {1,3,9} x strides
{16,32,48}, on random and on full-scale alternating-sign data, against the
portable loop. A disagreement demotes it to portable on its own flag
(`s_use_asm_taps`) without touching the rest of the kernel.

**What the third-round log also settled:** the loop reorder in round two was
almost worthless (334 -> 330 ms) and the reason was wrong in the note above.
The weights are in flash `.rodata`, which this part caches, not in PSRAM. The
PSRAM ceiling measured from the mel gate is real but it applies to *activation*
tensors, not to the weight matrices.

**Fourth round: an attempt to run the model across both cores, which FAILED.
Removed. Recorded here because the reasoning error is worth not repeating.**

`kws_model_run()` was split into `kws_model_front()` (delta stack, conv1, both
stem GroupNorms, conv2, GroupNorm-3, mel gate) and `kws_model_back()` (conformer
stack, soft-OR pool, classifier), and the firmware ran the back half on core 1.

Measured halves: front 189 ms, back 108 ms. The claim was that running them
concurrently costs the larger of the two -- 192 ms instead of 296 ms.

It measured **299 ms, unchanged**, with every stage identical to the single-core
run. The halves are independent in *memory* and strictly sequential in *time*:
`s_seq` is written by the mel gate, which is the LAST thing the front half does,
so the back half cannot begin until the front half has completely finished.
`max()` was never on the table; only the sum ever was.

The mistake was checking that the halves shared no buffers -- which they do not,
correctly -- and reading that as independence. Sharing memory is necessary for
parallelism and is nowhere near sufficient; the data dependency is what matters,
and it is total. `kws_model_front()`/`kws_model_back()` are kept because the seam
is real and useful to name, but there is no threading and `kws_model_run()` is
the whole of it again.

Round one, verified against the JAX fixtures after each change:

| change | what it removes |
|---|---|
| `swishf` reciprocal (`kws_rcp_raw`) | one `__divsf3` per swish, ~86,000 per inference, replaced by a seed plus three Newton steps (~12 instructions against ~60) |
| reciprocal square root (`kws_rsqrt_raw`) | `sqrtf` plus a division in `layer_norm` (294 rows) and `gn_finalize`, same seed, four Newton steps |
| conv2 per-tap gather | conv2 requantised all 288 window values for each of its 490 output elements -- a dequantise, a max-abs pass and a quantise pass, ~420,000 scalar float ops per inference. The window is now gathered as bytes (`memcpy`, 32 per tap) and reduced tap by tap |
| conv2 tap-major weights | per-tap activation scales need each tap's channels contiguous, which the plain axis reversal does not give. `TAP_MAJOR` in the exporter stores conv2 as `(out, kF, kT, in)`; every other tensor keeps the reversal |
| `kws_dot_taps_scaled` | one reduction call per output channel instead of nine. Nine calls is nine prologues and nine accumulator setups for the same eighteen `EE.VMULAS`, 23,520 times over -- it made conv2 *slower* than the single call it replaced, which is what the second hardware run showed |
| `kws_accum_row_i8` unrolled by 4 | the conformer's Dense layers reduce 48 MACs (three `EE.VMULAS`) per call, 7,056 calls per layer, so call overhead exceeded the multiply. Four output channels share the operand checks |

All three approximations are gated by self-tests measured over their real input
ranges, with the library call left in place as a fallback, and the verdicts are
printed in both the harness and the boot log:

```
fast exp:   in use (worst rel 2.86e-006, tol 1e-4)
fast rcp:   in use (worst rel 2.31e-007, tol 1e-6)
fast rsqrt: in use (worst rel 4.21e-007, tol 1e-6)
```

Accuracy improved rather than degraded: worst error 1.131e-3 -> 5.7e-4.

**Not yet done, in expected order of value.** Almost all of the remaining time is
float32, and this core has no float SIMD -- 86,000 scalar swishes plus attention,
norms and softmax is most of the 296 ms. Only conv2 (40.7 ms) is int8.

1. **Spend some of the 9x error headroom.** Worst error is 5.738e-4 against a
   5e-3 budget, so there is room to be roughly nine times less accurate in
   exchange for speed, and the handoff's original choice to keep LayerNorm,
   GroupNorm, swish and the attention matmuls in float32 was made before the
   budget was tightened. Quantising the conformer attention (float, 4 heads x
   49x49) and the norms are the two places where int8 would move the most time.
   This is the only lever left that changes the order of magnitude rather than
   the constant factor.
2. **Fine-grained parallelism inside the front half.** Not the front/back split
   -- that was sequential. But conv2's 48 output channels and gn3's 490 rows are
   independent, and the mel gate's 49 frames are independent, so ~68 ms of
   independent work could be split across the two cores. Only worth doing after
   item 1, because it is a constant factor on whatever is left.
3. **The stem's three passes.** 121 ms for a 31,360-value tensor, because
   GroupNorm-1 and GroupNorm-2 need whole-tensor statistics before the swish.
   Fusing them needs two frame buffers instead of four tensors and removes two
   of three crossings. Whether a streaming Welford pass can replace the three-pass
   structure is the open question, and it is a real algorithmic change rather
   than a scheduling one.
4. **`task_wdt` idle-task checking.** Fixed properly by subscribing the kws
   task, but `CONFIG_ESP_TASK_WDT_CHECK_IDLE_TASK_CPU0` is still on, and it is
   the reason a starved IDLE0 was reported at all. Harmless now.

---

## 6. Traps, for whoever hits them

- **Do not trust a reimplementation of the model as a reference.** Two of them
  were wrong in ways that produced plausible numbers. `capture_jax.py` exists
  because of that; use it.
- **Flax's `GroupNorm` statistics are per group**, over every axis *except* the
  group axis. Averaging over the group axis as well collapses all four groups
  into one mean and variance — a different normalisation that looks right.
- **`epsilon` is 1e-6**, matching `flax.linen.GroupNorm`'s default. 1e-5 is a
  silent change to the normalisation of every tensor in the network.
- **`_delta_stack` differences along TIME**, not mel. `models.py` pads axis 1 of
  a `(B,T,M)` array, which is the frame axis.
- **`SAME` padding is zeros**, not edge-replicated. For the delta stack that
  means `conv1` sees a genuinely zero phantom frame at both ends; for `d1` it
  means `d1[-1] == d1[0]`. Those are different rules and both are needed.
- **`pad_lo = pad_total // 2`** (floor). Ceil shifts every feature one mel bin.
- **Weight rows are 16-byte padded.** If you add a kernel whose reduction
  length is not a multiple of 16, it must be added to `REDUCE` in
  `export_weights.py` *and* the activation row must be padded to match, or the
  reduction will read into the next row.
- **The `{b}` placeholder in `REDUCE`** has to be expanded when matching a
  concrete tensor name. Getting that wrong leaves the block kernels unpadded
  and silently mis-strided.
- **conv2 is not axis-reversed like everything else.** It is tap-major, so its
  window and its weight row are both `(kF, kT, in)`. Reading it as
  `(in, kF, kT)` produces a correctly shaped tensor that is not the
  convolution, and the error looks like a quantisation problem rather than a
  layout one. `TAP_MAJOR` in the exporter is the single place that decides it.
- **A strided reduction cannot be vectorised.** Per-tap activation scales are
  only affordable because each tap's channels are contiguous. If the layout ever
  goes back to channel-major, the per-tap scales have to go with it -- the kernel
  would silently run scalar and lose the entire benefit of `EE.VMULAS.S8`.
- **The self tests are load-bearing, not decoration.** All three of the fast
  paths were wrong on the first attempt. The reciprocal overflowed to infinity
  for divisors above about 1e19. The rsqrt halved the *biased* exponent field
  instead of the unbiased one, so every result was off by a power of two and the
  test reported a relative error of exactly 1.0 -- and three Newton steps turned
  out to bottom out at 5.3e-4, inside "looks fine" and outside the 1e-6 the test
  enforces. Each was caught on the host in seconds and the fallback kept the
  build correct in the meantime. Keep the tolerances, keep the ranges, keep
  printing the verdicts.
- **A weight matrix fetched inside the row loop is not the expensive mistake it
  looks like.** The weights live in flash `.rodata` and this part caches them;
  removing 6.46 MB of *weight* re-reads per inference bought 12 ms. The same
  argument applied to activation tensors is a different matter, because those are
  in PSRAM at 60 MB/s. Know which memory a pointer is in before reasoning about
  its traffic.
- **Two useful vector MACs per call is not vectorisation.** conv2 reduced nine
  32-channel taps as nine separate calls and spent 7 instructions of prologue on
  every 2 of `EE.VMULAS.S8`. The MAC count looked fine; the call count was the
  entire cost.
- **Removing arithmetic from a memory-bound kernel buys exactly nothing.** The
  reciprocal, the rsqrt and the conv2 requantisation removal together were worth
  about 1M operations per inference and changed the total from 315 ms to 334 ms.
  The fast-path verdicts in the boot log are what made that conclusion safe to
  draw -- without them the other reading was "the optimisation did not work".
- **Three is not four.** Squared-error reasoning says three Newton steps suffice
  for rsqrt. Measured, three gives 5.3e-4 and four gives 4.2e-7.