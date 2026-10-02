# HANDOFF — Amaze wake word, ESP32-S3

Written at the end of the model-integration session so the next pass can start
clean. Everything here is committed and pushed; `git log` is the source of truth.

---

## 1. Where things actually stand

| | |
|---|---|
| Firmware builds, runs your model, ESP-SR removed | **done** |
| App binary | 586,912 B |
| Internal DRAM | 93,211 B (27.3%) — meets the <256 KB budget |
| Model deterministic | **fixed** (was returning a different probability every call) |
| **Numerical parity vs JAX** | **NOT met** — see §3 |
| CPU <10% idle | not met — ~28% float32 at 200 ms hop |
| int8 kernels | not started — see §5 |
| README | done, and honest about the above |

The firmware is in a shippable-build state. The detector is **not** trustworthy:
it will not reliably fire on the keyword until parity is reached.

Do not flash this expecting "amaze" detection to work.

---

## 2. System layout

```
INMP441 → I2S std RX (mono left, 32-bit slots)
        → int16 16 kHz, 1.000 s sliding window (16000 samples)
        → log-mel front end        (49, 40)
        → bcconformer_v3           → P(keyword)
        → peak-hold detector: thr 0.68, 2-of-2, decay 0.85, 1.5 s refractory
        → on hit: LED green 3 s, capture next 10 s to flash as WAV
```

| File | Role |
|---|---|
| `main/wake_word_main.c` | I2S, detector state machine, LED, flash clip storage |
| `main/kws_frontend.c/.h` | FFT, sparse mel filterbank, sliding window |
| `main/kws_model.c/.h` | the network — **this is what needs rewriting** |
| `main/model/kws_model.bin` | 84,865 float32 weights, 76 tensors |
| `main/model/kws_model_i8.bin` | per-tensor int8 weights (84,865 B) — exported, unused |
| `main/model/kws_model_data.h` | generated tensor table (shapes + offsets) |
| `main/model/kws_frontend_data.h` | generated window + mel filterbank, from numpy |
| `tools/export_weights.py` | checkpoint → weights, frontend data, references |
| `tools/dump_stages.py` | per-layer JAX activations |
| `tools/test_feat_only.c` | **capture-free** host harness — the reliable one |
| `tools/test_kws_host.c/.bat` | stage-dumping harness (buggy, see §7) |
| `tools/compare_stages.py` | layer-by-layer C vs JAX diff |
| `tools/eval_int8.py` | int8 accuracy measurement |
| `tools/pull_clips.ps1` | pull WAV clips off the device |

Model lives in the `models/` submodule (`wake_word_ml`): `best_v11_production.flax`,
`models.py`, `features.py`, `kws_engine.py`.

---

## 3. The open bug

`kws_model_run` returns **P = 0.0849** where the JAX reference gives
**P = 0.0272** on the same noise fixture.

Localised: `conv1` is bit-exact (7.6e-6, pure float32 rounding) for frames 4-48
and wrong only at **mel bin 0 of frames 0-3**. 2.15% RMS overall.

### Ruled out, each by direct experiment

| Hypothesis | Result |
|---|---|
| Uninitialised memory | **fixed** — six consecutive calls now agree |
| Undefined behaviour | identical at `-O0`, `-O2`, `-O2 -fno-strict-aliasing` |
| `spec` pointer wrong inside the model | captured it; low 32 bits match `ref_spec` |
| Inference mutates `spec` | byte-identical before/after every call, 0 elements changed |
| Buffer overlap | dumped all addresses; `s_row` ends exactly where `s_feat` begins |
| Extra `memcpy`s in pass 1 | removed one, no change |
| Intrusive `s_feat`/`spec` captures | removed them, no change |
| `gn_accum` trapped in an `#ifdef` | verified correct |
| `spec` shadowed | none |

### The paradox

`build_feat(spec, 0)` returns the correct delta stack when called from the
isolation hook, and an incorrect one when called from `kws_model_run`'s pass 1.
Same function, same binary, same spec pointer. Exactly the **first two floats**
of the spec buffer read as garbage from *inside* pass 1 and correct immediately
outside it.

### The test that actually works

```python
kws_model_init(); p1 = kws_model_run(spec)
                 p2 = kws_model_run(spec)   # compare!
```

That one line found the determinism bug in seconds after hours of reading stage
diffs. **Use it first, every time.**

---

## 4. Why the rewrite is the right fix

The stem is currently computed in three passes feeding conv2 through a
**three-frame rolling ring**, so the `(49,20,32)` float32 activation (125 KB)
never has to exist. That structure is where the bug lives:

- the ring is indexed one iteration before the row it needs exists
- output row 0 needs a zero-padded stem row −1, row 48 needs row 49 — the ring
  cannot hold either
- getting either wrong yields **uninitialised memory**, which is exactly the
  class of bug that has defeated every targeted experiment

Materialising the stem deletes the ring, both edge cases and the whole class.

In int8 the stem is `(49,20,32)` = **31,360 B**, so two stem buffers cost 63 KB
instead of 250 KB. The RAM saving that forced the ring is no longer needed.

---

## 5. The int8 design

### Weights — already measured safe

`tools/eval_int8.py`, per-tensor symmetric int8, float32 activations:

| fixture | float32 | int8 | error |
|---|---|---|---|
| noise | 0.027210 | 0.026972 | 0.000239 |
| tone1k | 0.000037 | 0.000039 | 0.000002 |
| sweep | 0.001971 | 0.002381 | 0.000410 |

Error is 1e-4 to 4e-4 against a **0.68** threshold. Weight quantisation is
effectively free. Per-channel came out *worse* (0.0328 on noise) — use per-tensor.

Weights: 339,460 B → 84,865 B, **75% smaller**.

### Activations — the actual accuracy risk

Unmeasured. Use **dynamic per-row quantisation**, which needs no calibration set
and cannot drift:

```c
/* per row: scale = max|x|/127, accumulate int32, dequantise once per output */
static float q_row(const float *x, int n, int8_t *q)
{
    float m = 0.0f;
    for (int i = 0; i < n; i++) { const float a = fabsf(x[i]); if (a > m) m = a; }
    if (m < 1e-12f) { memset(q, 0, (size_t)n); return 1e-12f; }
    const float s = m / 127.0f;
    for (int i = 0; i < n; i++) q[i] = (int8_t)lrintf(x[i] / s);
    return s;
}
```

Then `dot += (int32_t)w[i] * q[i]` and dequantise with `act_scale * weight_scale`.

### Suggested split

| Tensor | Type | Bytes |
|---|---|---|
| weights | int8 + float scale per tensor | 84,865 + 304 |
| stem, conv1 out `(49,20,32)` | int8 | 31,360 each |
| conv2 out `(49,10,48)` | int8 | 23,520 |
| qkv, ffn hidden, seq, ln, attn, blk | float32 | ~75 KB total |

Keep the conformer tensors float32 — they are small and the softmax/attention
precision matters most there.

All of these should carry `EXT_RAM_BSS_ATTR` so they land in PSRAM.

---

## 6. How to build and validate

```powershell
$env:IDF_TOOLS_PATH="C:\Espressif"      # NOT C:\Espressif — see §7
& "C:\esp\v6.1\esp-idf\export.ps1" | Out-Null
idf.py build
```

ML toolchain (JAX 0.11.2, flax 0.12.10) lives in an isolated venv:

```
C:\Users\dsain\AppData\Local\Temp\kilo\mlenv\Scripts\python.exe
```

Validation loop, in this order:

```bash
python tools/export_weights.py --dump --int8   # weights + frontend data + refs
python tools/dump_stages.py                   # per-layer JAX activations
gcc -O2 -std=gnu11 -DKWS_HOST_TEST -Imain -Imain\model \
    main/kws_frontend.c main/kws_model.c tools/test_feat_only.c -o t.exe -lm
t.exe
```

`test_feat_only.c` prints P and compares against the reference. **That is the
number that matters.** Only reach for `test_kws_host.bat` + `compare_stages.py`
when the whole-model number is right and you need to find *where*.

Accept when `P` matches JAX to <1e-4 on all fixtures.

---

## 7. Mistakes that cost time — do not repeat

1. **Never regex-patch `main/kws_model.c`.** Two attempts at scripted edits
   produced cascading compile errors and I reverted both. Write the file in one
   pass with the `write` tool. It has been through enough automated patching
   that its current structure is not fully trustworthy even though it compiles.

2. **`test_kws_host.bat` needs `-DKWS_HOST_TEST`.** Without it `kws_blob` is
   defined twice and the harness references `kws_weights`, which only exists in
   a firmware build. Capture machinery needs a *separate* switch,
   `-DKWS_CAPTURE_ON` — conflating the two once made me chase an instrument
   artefact as if it were a model bug.

3. **`EXT_RAM_BSS_ATTR` expands to nothing** unless
   `CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY=y`. It is set in
   `sdkconfig.defaults`. Without it the whole activation set lands in internal
   DRAM and DIRAM reads 96.8%.

4. **`add_custom_command(OUTPUT ...)` is "not scriptable"** in an IDF component
   scope — no target exists yet. `main/CMakeLists.txt` generates the weight
   array at configure time with `execute_process` instead.

5. **A stale capture file looks exactly like a real bug.** A cleanup pass
   deleted every `KWS_CAPTURE` call site, so a "frame 0 corruption" I chased
   for a while was a comparison against a file nothing was writing. **Delete the
   output files before trusting a diff.**

6. **Never trust a single `kws_model_run` call.** Always two, and compare.

---

## 8. Constraints being judged against

- **<256 KB RAM** — currently met at 93 KB internal. Recheck after the rewrite.
- **<10% CPU while idling** — not met. 13.51 MMAC/inference, `conv2` alone is
  **50.1%**. int8 plus `esp-nn` style kernels is the lever; full derivation in
  `docs/resource_estimates.md`.
- **No proprietary SDKs, no pre-trained generic keyword** — met. ESP-SR is
  removed entirely; only `led_strip` remains as a dependency.
- **Latency, keyword end → ASR receipt** — not built. The 10 s flash recorder is
  local test scaffolding and is *not* what the metric measures; streaming
  replaces it. Latency is ~89% hop-grid, ~17% model.

---

## 9. Suggested order for the next session

1. Write `main/kws_model.c` **in full, by hand**, int8 weights + materialised
   int8 stem + dynamic per-row activation quantisation. Delete the ring.
2. `test_feat_only.c` → match JAX to <1e-4.
3. Only then re-run `test_kws_host.bat` / `compare_stages.py` to localise.
4. Rebuild firmware, re-measure DIRAM.
5. Update the README's known-defect section once parity is real.
6. Then int8 CPU measurement, and decide about TFLite Micro (§10).

---

## 10. Open question for you

We are **not** using TinyML / TensorFlow Lite Micro. The inference is hand-written
C ported from a Flax checkpoint; training is JAX/Flax.

The PS says *"Recommended tools include TensorFlow Lite for Microcontrollers,
PyTorch Mobile or similar"* — recommended, not mandatory — and separately
mandates open-source and no pre-trained generic keywords, both of which the
hand-written path satisfies.

But a judge looking for a TinyML stack will look for TFLM and not find it. My
recommendation is to **keep hand-written and document why** — the `<256 KB RAM`
and `<10% CPU` constraints are what hand-written int8 actually hits — and add a
"Why not TFLite Micro" section to the README with measured figures.

Switching is reversible if you prefer the badge: it needs TensorFlow + jax2tf to
produce the `.tflite`, adds ~30-50 KB of code, and spends some RAM headroom.
