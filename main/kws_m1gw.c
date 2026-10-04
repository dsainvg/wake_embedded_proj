/*
 * m1_g_wide int8 forward pass. See kws_m1gw.h for the architecture, the
 * arithmetic, the RAM budget, and why ESP-DL was rejected in favour of reusing
 * this repo's own kws_int8.h.
 *
 * Layout
 * ------
 * int8 stem activations are [position][channel] with the channel axis innermost,
 * so the reduction axis of every GEMM is contiguous in both operands and
 * kws_accum_row_i8() applies without a transpose. Positions are (time * mel)
 * flattened. torch's own weight layouts already match: a 1x1 conv is (Cout, Cin)
 * and a Linear is (out, in), both [out_dim][in_dim] with the reduction innermost.
 * Nothing is transposed at runtime.
 *
 * The conformer blocks keep float32 activations. At 13 frames x 96 channels they
 * are a few KB, and float is what makes LayerNorm, the softmax and logsumexp
 * straightforward; the tensors that would be expensive are exactly the stem
 * tensors, and those are int8.
 *
 * Requantisation
 * --------------
 * Each GEMM row is requantised on its own scale, stored alongside as
 * s1_sc/dn_sc/s2_sc. A single tensor-wide scale would require computing the
 * whole tensor twice to discover its maximum, which would push the stem GEMMs
 * from 0.90 MMAC to 1.80 and cost more than the scales save. Per-row scales
 * cost 1,370 floats and keep the pipeline single-pass.
 */

#include "kws_m1gw.h"

#include <math.h>
#include <string.h>

#ifdef KWS_HOST_TEST
/* The host harness compiles this same file off the board. esp_timer and
 * esp_cpu_get_cycle_count are the only IDF dependencies, and both are used for
 * stage timings, which are diagnostics rather than anything load-bearing -- so
 * they are shimmed rather than removed, leaving the code under test identical.
 *
 * clock() rather than clock_gettime(): MinGW does not declare clock_gettime
 * under -std=gnu11 without _POSIX_C_SOURCE, and these numbers are diagnostics on
 * the host, not a measurement of anything real.
 *
 * The cycle shim is deliberately COARSE (1 MHz of "cycles" per second of
 * clock()). Its purpose is to keep the reporting path exercised and its
 * arithmetic correct on the host; the ratio it produces is not a cycle count
 * and nothing on the host should be read as one. */
#include <stdio.h>
#include <time.h>
static inline int64_t esp_timer_get_time(void)
{
    return (int64_t)((double)clock() * (1e6 / CLOCKS_PER_SEC));
}
static inline uint32_t esp_cpu_get_cycle_count(void)
{
    return (uint32_t)((double)clock() * (1e6 / CLOCKS_PER_SEC));
}
#else
#include "esp_cpu.h"
#include "esp_timer.h"
#endif

#include "kws_fastmath.h"
#include "kws_int8.h"
#include "kws_m1gw_data.h"

/*
 * Shortest reduction worth sending to the vector unit.
 *
 * MEASURED, both directions, on hardware:
 *
 *     in_dim=96   portable 12.04 ms   vector 2.55 ms   -> vector, 4.7x
 *     in_dim=32   portable 37.93 ms   vector 44.94 ms   -> PORTABLE
 *
 * The 96 case is qkv/proj/conv_pw/ff1/ff2. The 32 case is down.pw and stem2.pw,
 * and there the assembly is 18% SLOWER: 32 elements is two EE.VMULAS, and the
 * fixed cost of entering the windowed ABI (`entry a1, 32` saves 8 registers) plus
 * reading the 64-bit accumulator back with its two-cycle penalty is more than the
 * two fused instructions save.
 *
 * So the crossover is a length, not a capability. Below this the portable loop
 * wins; above it the vector unit wins by nearly 5x.
 */
#define KWS_M1GW_ASM_MIN_IN_DIM 64

/*
 * Whether to use the Xtensa assembly int8 reduction for the GEMMs.
 *
 * ENABLED, and the earlier evidence for disabling it was wrong.
 *
 * That comparison used the `down` stage, which bundles the depthwise convolution
 * with the pointwise GEMM -- so a change to the reduction was being read against
 * a number that also contained a strided 9-tap convolution. Per-projection
 * instrumentation settles it. Four projections with identical geometry
 * (13 rows, 96 in, 96 out, two blocks) each cost:
 *
 *     proj      12.032 ms    2,887,623 cyc
 *     conv_pw   12.050 ms    2,891,989 cyc
 *     ff1       12.037 ms    2,888,928 cyc
 *     ff2       12.051 ms    2,892,324 cyc
 *
 * All four within 0.2% of each other, which is what you expect when the cost is
 * the inner loop and nothing else. That works out at 12.05 cycles per
 * multiply-accumulate -- 0.83 MAC/cycle on a part whose vector unit does 16
 * MACs per EE.VMULAS.S8.ACCX. The scalar loop is the problem, and
 * kws_asm_dot_i8() is the fix.
 *
 * So this is a clean A/B for the next flash: identical weights, identical
 * activations, one line changed. proj at 12.03 ms and qkv at 33.92 ms are the
 * baselines. If the vector path engages, those drop by roughly an order of
 * magnitude; if they do not, the guard in kws_accum_row_i8() is rejecting it and
 * that condition is the next thing to print.
 */

#define KWS_M1GW_USE_ASM_REDUCTION 1

/* torch.nn.GroupNorm and nn.LayerNorm both default to 1e-5. */
#define KWS_M1GW_EPS 1e-5f

/* The softmax's normaliser over the 13 frames of the sequence, precomputed once
 * at compile time rather than per inference. */
#define KWS_M1GW_LOG_T ((float)log((double)KWS_M1GW_T2))

static kws_m1gw_arena s_arena;

#ifdef KWS_HOST_TEST
/* Stage-wise finiteness probe. A NaN does not announce itself: it compares
 * false against every threshold, so a NaN probability reads as "below
 * threshold" and the detector simply never fires, which looks identical to a
 * model that does not work. This names the stage where one first appears. */
static int s_nan_stage = -1;
static void chk_finite(const char *what, const float *v, int n)
{
    for (int i = 0; i < n; i++) {
        if (!isfinite(v[i])) {
            printf("NON-FINITE in %s at %d of %d: %g\n", what, i, n, (double)v[i]);
            s_nan_stage = 1;
            return;
        }
    }
}
#define CHK(what, buf, n) chk_finite(what, (buf), (n))
#else
#define CHK(what, buf, n) ((void)0)
#endif

/* ------------------------------------------------------------- timing ----- */

static const char *const s_stage_name[KWS_M1GW_STAGE_COUNT] = {
    "delta", "stem1", "down", "stem2", "mel_gate", "blocks", "pool+head",
};

/*
 * Multiply-accumulates per stage, same order as s_stage_name. Walked from the
 * declared layer sequence by tools/export_m1gw_weights.py; the total is
 * 2,771,692, i.e. 2.77 MMAC against the checkpoint's recorded 1,473,962 -- see
 * the exporter's report for why the recorded figure is 1.88x low.
 *
 * Carrying MACs beside measured cycles is the point: a stage spending far more
 * cycles per MAC than its neighbours is an implementation problem (a misaligned
 * reduction, a fallback path) and that ratio is visible here and nowhere else.
 */
static const uint32_t s_stage_mac[KWS_M1GW_STAGE_COUNT] = {
    0,        /* DeltaStack: fixed kernels, adds and one scale, 0 MAC */
    120540,   /* stem1    dw 3x3 s(1,2) + pw 3->32 over 980 positions */
    341120,   /* down     dw 3x3 s(4,1) + pw 32->32 over 260 positions */
    436800,   /* stem2    dw 3x3 s(1,2) + pw 32->96 over 130 positions */
    100880,   /* mel_gate dim->8->1 over 13x10, plus the weighted sum */
    1771776,  /* blocks   both RelConformerBlocks: attention, depthwise+SE and
               *           the FFN. Timed as ONE slot because they are one loop
               *           and interleaved -- the previous 8-slot layout timed
               *           both blocks entirely into 'attn' and reported zero
               *           for conv+se and ffn+pool while they demonstrably
               *           ran. */
    576,      /* pool+head SoftORPool is 0 MAC; the 288->2 head is 576 */
};

static const char *const s_proj_name[KWS_M1GW_PROJ_COUNT] = {
    "qkv", "proj", "conv_pw", "ff1", "ff2",
};
static uint64_t s_proj_us[KWS_M1GW_PROJ_COUNT];
static uint64_t s_proj_cycles[KWS_M1GW_PROJ_COUNT];

static uint64_t s_stage_us[KWS_M1GW_STAGE_COUNT];
static uint64_t s_stage_cycles[KWS_M1GW_STAGE_COUNT];
static uint64_t s_stage_runs;
static uint32_t s_last_us;
static uint32_t s_last_cycles;
static uint32_t s_peak_us;
static uint64_t s_total_us;
static uint64_t s_total_cycles;

static int64_t us_now(void) { return esp_timer_get_time(); }

/*
 * CPU cycles rather than timer ticks, because cycles are what the core actually
 * spent and do not move if the clock is not what you think it is. The two are
 * reported together and a disagreement between them is a misconfigured
 * frequency, which is otherwise invisible until a latency budget is missed.
 * esp_cpu_get_cycle_count() is a 32-bit counter on this part, so it wraps about
 * every 17 s at 240 MHz; every reading here spans microseconds, so a wrap only
 * ever shows up as a nonsensical per-stage delta, not a plausible wrong total.
 */
static inline uint32_t cyc_now(void) { return esp_cpu_get_cycle_count(); }

/* ---------------------------------------------------------------- utils -- */

/* Round-to-nearest with saturation to +/-127, so negation stays exact and the
 * scale stays symmetric (128 would make -128 unrepresentable). */
static inline int8_t sat8(float v)
{
    int32_t q = (int32_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    return (int8_t)q;
}

/* Symmetric scale of one row. An all-zero row returns 1.0f so the requantise
 * divides by a real number instead of producing a NaN that propagates silently
 * into the logits and reads as a probability. */
static float row_scale(const float *y, int n)
{
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        const float a = fabsf(y[i]);
        if (a > mx) mx = a;
    }
    return mx > 0.0f ? mx / 127.0f : 1.0f;
}

static void quantise_row(const float *y, int n, float scale, int8_t *q)
{
    const float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) q[i] = sat8(y[i] * inv);
}

/*
 * SiLU: x / (1 + exp(-x)).
 *
 * Both halves of this were a libm call, and this is the hottest function in the
 * model by a wide margin -- roughly 29,000 activations per inference.
 *
 *   expf   newlib's is software on this part, of the order of a thousand cycles
 *   /      there is NO float divide instruction, so this went through __divsf3
 *          in ROM at roughly sixty instructions -- which is what your crash PC
 *          was sitting in
 *
 * Together those are ~1,300 cycles per activation, about 37 M cycles per
 * inference, which is most of the 225 ms this model was taking while its
 * multiply-accumulates accounted for under a millisecond of the budget.
 *
 * kws_expf() and kws_rcp() fall back to the libm versions until their self-tests
 * pass, so a fast path that turns out to be wrong costs speed, not accuracy.
 */
static inline float silu(float x)
{
    return x * kws_rcp(1.0f + kws_expf(-x));
}

static void softmax_inplace(float *x, int n)
{
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = kws_expf(x[i] - mx); sum += x[i]; }
    const float inv = sum > 0.0f ? kws_rcp(sum) : 0.0f;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* -------------------------------------------------------------- tensors -- */

static inline const float *Tf(int id)
{
    return (const float *)(kws_m1gw_blob + kws_m1gw_tensors[id].offset);
}

static inline const int8_t *Ti(int id)
{
    return kws_m1gw_blob + kws_m1gw_tensors[id].offset;
}

static inline const float *Tsc(int id)
{
    const kws_m1gw_tensor *t = &kws_m1gw_tensors[id];
    return kws_m1gw_scales + t->scale_off;
}

/* ----------------------------------------------------------------- GEMM -- */

/*
 * One int8 row against a whole weight matrix, per-output-channel dequantised,
 * then requantised onto its own scale.
 *
 *   xq  [rows][in_dim] int8 with one scale for the tensor
 *   w   [out_dim][in_dim] int8, per-row scale wsc[out_dim]
 *   yq  [rows][out_dim] int8, one scale per row in ysc[rows]
 *
 * kws_accum_row_i8() folds a single scalar into the reduction and adds the bias,
 * so the per-output-channel weight scale is applied afterwards.
 */
#ifdef KWS_HOST_TEST
static int s_gemm_trace = 0;
#endif

static void gemm_row_q(const int8_t *xq, float xsc, int rows, int in_dim,
                       int tid_w, int tid_bias, int out_dim,
                       int8_t *yq, float *ysc)
{
    const int8_t *w = Ti(tid_w);
    const float *wsc = Tsc(tid_w);
    const float *bias = (tid_bias >= 0) ? Tf(tid_bias) : NULL;
    float *row = s_arena.row;

    /* kws_accum_row_i8() strides the weight by align16(in_dim), not by in_dim,
         * because the Xtensa reduction loads 128 bits at a time. The blob stores
         * rows unpadded, so the two agree only when in_dim is already a multiple
         * of 16. Where it is not -- stem1.pw (3), mel_gate.fc2 (8), se_fc2 (12)
         * -- the assembly kernel reads w[o*16 + i] against data at w[o*3 + i] and
         * returns garbage for every row after the first. Those three tensors are
         * 3% of the arithmetic, so they take a portable reduction rather than
         * forcing the whole blob into a padded layout. */
    for (int p = 0; p < rows; p++) {
        /*
         * The PORTABLE reduction, deliberately, for every shape.
         *
         * This used to call kws_accum_row_i8() -- and therefore the Xtensa
         * assembly -- whenever in_dim was a multiple of 16. Measured on hardware,
         * with the same weights and the same MAC count and the only difference
         * being which reduction ran:
         *
         *     stage    portable    assembly    MAC
         *     down       29.46 ms    47.19 ms   341,120
         *     stem2      17.80 ms    44.44 ms   436,800
         *     total      47.26 ms    91.63 ms   777,920
         *
         * The assembly is 1.94x SLOWER here. kws_kernel_selftest() checks that
         * the two AGREE and never checks which is quicker, and the boot banner
         * reports what was selected, so nothing objected.
         *
         * Why it loses is not established -- 32-element reductions are short
         * enough that the setup costs dominate, and a -O2 scalar loop over int8
         * is hard to beat. What is established is the measurement above, on this
         * part, for these shapes.
         *
         * Revisit if the reduction is retuned: kws_accum_row_i8() is still the
         * right entry point, and the portable branch below is the same loop it
         * falls back to.
         */
        /* Two independent conditions, and dropping either one is a silent
         * numerical failure rather than a slow path.
         *
         * The switch, which is a performance choice.
         *
         * AND `in_dim` being a multiple of 16, which is a CORRECTNESS
         * requirement. kws_accum_row_i8() strides the weight by
         * align16(in_dim), because the vector loads are 128 bits wide. This
         * blob stores rows unpadded, so for stem1.pw -- 3 inputs per row,
         * align16 giving 16 -- it would read weight bytes 0,1,2 of row 0 and
         * then start row 1 at byte 16 instead of byte 3. Every row after the
         * first would be garbage, the stem output would be noise, and the
         * detector would simply never fire while still reporting a plausible
         * probability.
         *
         * That is exactly what happened when the per-shape condition was
         * replaced by the switch alone: host discrimination collapsed from 12.6x
         * to 1.0x. The host caught it because it runs the same source, and that
         * is the argument for keeping the host harness wired to every build.
         */
        if (KWS_M1GW_USE_ASM_REDUCTION && (in_dim & 15) == 0
            && in_dim >= KWS_M1GW_ASM_MIN_IN_DIM) {
            kws_accum_row_i8(row, xq + (size_t)p * in_dim, w, bias, xsc,
                             out_dim, in_dim);
        } else {
            /* restrict is load-bearing, not decoration.
             *
             * xr and wr point into two different arrays -- one the activation
             * arena, one the weight blob in flash -- and there is no path by
             * which they can overlap. Without saying so, gcc has to assume they
             * might and so cannot keep the accumulator in a register across
             * iterations, which costs about 3.5x on these lengths: a 96-element
             * reduction was measuring 1366 cycles.
             */
            const int8_t *restrict xr = xq + (size_t)p * in_dim;
            for (int o = 0; o < out_dim; o++) {
                const int8_t *restrict wr = w + (size_t)o * in_dim;
                int32_t acc = 0;
                for (int i = 0; i < in_dim; i++) acc += (int32_t)xr[i] * (int32_t)wr[i];
                row[o] = (float)acc * xsc + (bias ? bias[o] : 0.0f);
            }
        }
        for (int o = 0; o < out_dim; o++) row[o] *= wsc[o];
#ifdef KWS_HOST_TEST
        if (s_gemm_trace < 6) {
            printf("GEMM t%d rows=%d in=%d out=%d xsc=%g p0: xr=[%d %d %d] w0=[%d %d %d] "
                   "acc=%d wsc0=%g row0=%g\n", tid_w, rows, in_dim, out_dim,
                   (double)xsc, (int)xq[0], (int)xq[1], (in_dim > 2 ? (int)xq[2] : 0),
                   (in_dim > 0 ? (int)w[0] : 0), (in_dim > 1 ? (int)w[1] : 0),
                   (in_dim > 2 ? (int)w[2] : 0),
                   (in_dim == 3 ? (int)((int32_t)xq[0] * (int32_t)w[0] +
                                         (int32_t)xq[1] * (int32_t)w[1] +
                                         (int32_t)xq[2] * (int32_t)w[2]) : 0),
                   (double)wsc[0], (double)row[0]);
            s_gemm_trace++;
        }
#endif
        const float s = row_scale(row, out_dim);
        ysc[p] = s;
        quantise_row(row, out_dim, s, yq + (size_t)p * out_dim);
    }
}

/* Dense on float activations, output float. Used inside the conformer blocks,
 * where the tensors are small and float avoids a requantise per layer.
 *
 * x and y MAY be the same buffer. Each output row is staged in the arena's
 * dense_row before being copied back, because writing y[p][o] directly would
 * overwrite x[p][o] while a later o' still has to read it -- which corrupts the
 * result silently instead of failing. */
static void dense_f32(const float *x, int rows, int dim, int out_dim,
                      int tid_w, int tid_bias, float *y)
{
    const int8_t *w = Ti(tid_w);
    const float *wsc = Tsc(tid_w);
    const float *bias = (tid_bias >= 0) ? Tf(tid_bias) : NULL;
    float *stage = s_arena.dense_row;

    for (int p = 0; p < rows; p++) {
        const float *xr = x + (size_t)p * dim;
        for (int o = 0; o < out_dim; o++) {
            const int8_t *wr = w + (size_t)o * dim;
            const float ws = wsc[o];
            float acc = bias ? bias[o] : 0.0f;
            for (int i = 0; i < dim; i++) acc += (float)wr[i] * ws * xr[i];
            stage[o] = acc;
        }
        memcpy(y + (size_t)p * out_dim, stage, (size_t)out_dim * sizeof(float));
    }
}

/* ------------------------------------------------------------- normals --- */

/* GroupNorm over [rows][channels] (channel innermost), then SiLU, in place on a
 * float buffer.
 *
 * ONE mean and ONE variance per GROUP, over every element of that group -- all
 * its channels and all rows. Per-channel statistics would be InstanceNorm, a
 * different layer: with 1 group over the 3 stem channels that is 3 statistics
 * where torch uses 1, and with 4 groups over 32 channels it is 32 where torch
 * uses 4. Neither fails loudly; both just make the network nearly
 * input-insensitive, which is exactly how this got found. */
static void groupnorm_silu(float *x, int rows, int channels, int groups,
                           const float *gw, const float *gb)
{
    const int per_group = channels / groups;
    const float n = (float)rows * (float)per_group;

    for (int g = 0; g < groups; g++) {
        float sum = 0.0f, sq = 0.0f;
        for (int p = 0; p < rows; p++) {
            const float *row = x + (size_t)p * channels + g * per_group;
            for (int c = 0; c < per_group; c++) { sum += row[c]; sq += row[c] * row[c]; }
        }
        const float mean = sum / n;
        float var = sq / n - mean * mean;
        if (var < 0.0f) var = 0.0f;
        const float inv_std = 1.0f / sqrtf(var + KWS_M1GW_EPS);

        for (int p = 0; p < rows; p++) {
            float *row = x + (size_t)p * channels + g * per_group;
            for (int c = 0; c < per_group; c++) {
                const float v = (row[c] - mean) * inv_std * gw[g * per_group + c]
                              + (gb ? gb[g * per_group + c] : 0.0f);
                row[c] = silu(v);
            }
        }
    }
}

/* LayerNorm over the innermost axis, row by row. */
static void layernorm(const float *x, float *y, int rows, int dim,
                      const float *gw, const float *gb)
{
#ifdef KWS_HOST_TEST
    chk_finite("layernorm gw", gw, dim);
    chk_finite("layernorm gb", gb, dim);
    chk_finite("layernorm x", x, rows * dim);
#endif
    for (int p = 0; p < rows; p++) {
        const float *xr = x + (size_t)p * dim;
        float *yr = y + (size_t)p * dim;
        float sum = 0.0f, sq = 0.0f;
        for (int i = 0; i < dim; i++) { sum += xr[i]; sq += xr[i] * xr[i]; }
        const float inv_n = 1.0f / (float)dim;
        const float mean = sum * inv_n;
        float var = sq * inv_n - mean * mean;
        if (var < 0.0f) var = 0.0f;
        const float inv_std = 1.0f / sqrtf(var + KWS_M1GW_EPS);
        for (int i = 0; i < dim; i++) {
            yr[i] = (xr[i] - mean) * inv_std * gw[i] + (gb ? gb[i] : 0.0f);
        }
    }
}

/*
 * Quantise a float [rows][dim] tensor row by row, then fold the per-row scales
 * onto a single tensor scale, because kws_accum_row_i8() takes one scalar.
 *
 * Per-row first, then folded: the alternative is one tensor-wide scale, which
 * needs the whole tensor computed twice to find its maximum. Returns the tensor
 * scale.
 */
static float requantise_rows(const float *yf, int8_t *yq, int rows, int dim)
{
    /* row_scale() ALREADY divides by 127, so the tensor scale is just the largest
     * row scale. Dividing it by 127 again -- which this did -- makes the tensor
     * scale 127x too small, every fold factor 127x too large, and every int8
     * value saturate at +-127. The result is not a scale error, it is a clipped
     * tensor: s1 correlated 0.48 with the reference instead of ~1. */
    float mx = 0.0f;
    for (int p = 0; p < rows; p++) {
        const float s = row_scale(yf + (size_t)p * dim, dim);
        if (s > mx) mx = s;
        quantise_row(yf + (size_t)p * dim, dim, s, yq + (size_t)p * dim);
    }
    const float ts = mx > 0.0f ? mx : 1.0f;
    for (int p = 0; p < rows; p++) {
        const float f = row_scale(yf + (size_t)p * dim, dim) / ts;   /* in [0,1] */
        for (int c = 0; c < dim; c++)
            yq[(size_t)p * dim + c] = sat8((float)yq[(size_t)p * dim + c] * f);
    }
    return ts;
}

/* --------------------------------------------------------- depthwise 3x3 --
 *
 * Conv2d(C, C, 3x3, groups=C), zero padded, strided in time and/or mel.
 *
 * Reads the int8 activation tensor DIRECTLY, with its per-position scales.
 * The obvious implementation dequantises the input into a float copy first; an
 * earlier draft did that and needed 125,440 B for the stem1 input alone, which
 * put the translation unit at 353 KB of .bss. Folding `in_sc[ip]` into the
 * accumulate costs one extra multiply per tap and removes the copy entirely.
 *
 * `in` is [rows][C] with rows = time*mel, so each output gathers 9 taps from a
 * strided input. `out` is float [out_rows][C] and lands in the arena.
 */
static void depthwise_i8(const int8_t *in, const float *in_sc,
                         int in_t, int in_m, int C,
                         int tid_w, int stride_t, int stride_m,
                         int out_t, int out_m, float *out)
{
    const int8_t *w = Ti(tid_w);
    const float *wsc = Tsc(tid_w);

    for (int ot = 0; ot < out_t; ot++) {
        for (int om = 0; om < out_m; om++) {
            const int op = ot * out_m + om;
            float *orow = out + (size_t)op * C;
            for (int c = 0; c < C; c++) {
                float acc = 0.0f;
                for (int kt = -1; kt <= 1; kt++) {
                    const int it = ot * stride_t + kt;
                    if (it < 0 || it >= in_t) continue;
                    for (int km = -1; km <= 1; km++) {
                        const int im = om * stride_m + km;
                        if (im < 0 || im >= in_m) continue;
                        const int ip = it * in_m + im;
                        acc += (float)w[(size_t)c * 9 + (kt + 1) * 3 + (km + 1)]
                               * (float)in[(size_t)ip * C + c] * in_sc[ip];
                    }
                }
                orow[c] = acc * wsc[c];
            }
        }
    }
}

/* --------------------------------------------------------- the forward --- */

float kws_m1gw_run(const float *spec)
{
    kws_m1gw_arena *A = &s_arena;
    float s_delta_scale = 1.0f;   /* set by DeltaStack, read by stem1 */
    const int64_t t0 = us_now();
    int64_t t_stage = t0;
    uint32_t c_stage = cyc_now();
    const uint32_t c_begin = c_stage;
    int stage = 0;

/* Time one projection. Same shape as STAGE_END() so the two read alike. */
#define PROJ_BEGIN(slot_, tag_)                                          \
    const int64_t pu_##tag_ = us_now();                                 \
    const uint32_t pc_##tag_ = cyc_now();                               \
    (void)pu_##tag_; (void)pc_##tag_

#define PROJ_END(slot_, tag_)                                            \
    do {                                                                 \
        s_proj_us[slot_] += (uint64_t)(us_now() - pu_##tag_);           \
        s_proj_cycles[slot_] += (uint64_t)(cyc_now() - pc_##tag_);       \
    } while (0)

#define STAGE_END()                                                     \
    do {                                                                \
        const int64_t t_now = us_now();                                 \
        const uint32_t c_now = cyc_now();                               \
        s_stage_us[stage] += (uint64_t)(t_now - t_stage);               \
        s_stage_cycles[stage] += (uint64_t)(c_now - c_stage);           \
        t_stage = t_now;                                                \
        c_stage = c_now;                                                \
        if (++stage >= KWS_M1GW_STAGE_COUNT) stage = KWS_M1GW_STAGE_COUNT - 1; \
    } while (0)



    /* ---- DeltaStack: [log-mel, d1, d2], central difference in time --------
     * Fixed kernels, no parameters, and zero MAC by definition -- these are
     * adds and one scale, so they do not appear in the arithmetic budget at all.
     * Stored channel-major [3][49][40] because the stem1 depthwise gathers one
     * channel at a time.
     *
     * Two passes: find the maximum over all three planes, then quantise. Doing
     * it in one pass means quantising against an unknown scale and rescaling
     * afterwards, which silently clips any plane whose magnitude already exceeds
     * 127 -- and a loud room produces exactly that. */
    {
        const int n = KWS_M1GW_T_IN * KWS_M1GW_M_IN;
        int8_t *q = A->delta;

        float mx = 0.0f;
        for (int t = 0; t < KWS_M1GW_T_IN; t++) {
            const int tp = (t > 0) ? t - 1 : 0;
            const int tn = (t + 1 < KWS_M1GW_T_IN) ? t + 1 : t;
            const int tn2 = (tn + 1 < KWS_M1GW_T_IN) ? tn + 1 : tn;
            const int tp2 = (tp > 0) ? tp - 1 : tp;
            for (int m = 0; m < KWS_M1GW_M_IN; m++) {
                const float v  = spec[t * KWS_M1GW_M_IN + m];
                const float d1 = 0.5f * (spec[tn * KWS_M1GW_M_IN + m]
                                         - spec[tp * KWS_M1GW_M_IN + m]);
                const float d2 = 0.5f * (
                    0.5f * (spec[tn2 * KWS_M1GW_M_IN + m]
                            - spec[tn * KWS_M1GW_M_IN + m])
                    - 0.5f * (spec[tn * KWS_M1GW_M_IN + m]
                              - spec[tp2 * KWS_M1GW_M_IN + m]));
                const float a = fabsf(v), b = fabsf(d1), c = fabsf(d2);
                if (a > mx) mx = a;
                if (b > mx) mx = b;
                if (c > mx) mx = c;
            }
        }
        s_delta_scale = mx > 0.0f ? mx / 127.0f : 1.0f;
        const float inv = 1.0f / s_delta_scale;

        for (int t = 0; t < KWS_M1GW_T_IN; t++) {
            const int tp = (t > 0) ? t - 1 : 0;
            const int tn = (t + 1 < KWS_M1GW_T_IN) ? t + 1 : t;
            const int tn2 = (tn + 1 < KWS_M1GW_T_IN) ? tn + 1 : tn;
            const int tp2 = (tp > 0) ? tp - 1 : tp;
            for (int m = 0; m < KWS_M1GW_M_IN; m++) {
                const float v  = spec[t * KWS_M1GW_M_IN + m];
                const float d1 = 0.5f * (spec[tn * KWS_M1GW_M_IN + m]
                                         - spec[tp * KWS_M1GW_M_IN + m]);
                const float d2 = 0.5f * (
                    0.5f * (spec[tn2 * KWS_M1GW_M_IN + m]
                            - spec[tn * KWS_M1GW_M_IN + m])
                    - 0.5f * (spec[tn * KWS_M1GW_M_IN + m]
                              - spec[tp2 * KWS_M1GW_M_IN + m]));
                q[(size_t)t * KWS_M1GW_M_IN + m] = sat8(v * inv);
                q[n + (size_t)t * KWS_M1GW_M_IN + m] = sat8(d1 * inv);
                q[2 * n + (size_t)t * KWS_M1GW_M_IN + m] = sat8(d2 * inv);
            }
        }
    }
    STAGE_END();
    /* ---- stem1: dw(1,2) -> GN(1 group over 3) -> SiLU -> pw 3->32 ---------
     * DeltaStack is channel-major [3][49][40] because the depthwise reads one
     * channel at a time, so this stage gets its own inline gather rather than
     * depthwise_i8(). Its output is only 2,940 floats, so it goes in dwf. */
    {
        const int red_pos = KWS_M1GW_MAX_POS;
        const int8_t *w = Ti(KWS_M1GW_T_STEM1_DW_WEIGHT);
        const float *wsc = Tsc(KWS_M1GW_T_STEM1_DW_WEIGHT);
        const float dsc = s_delta_scale;

        for (int op = 0; op < red_pos; op++) {
            const int ot = op / KWS_M1GW_M1;
            const int om = op - ot * KWS_M1GW_M1;
            for (int c = 0; c < 3; c++) {
                float acc = 0.0f;
                for (int kt = -1; kt <= 1; kt++) {
                    const int it = ot + kt;
                    if (it < 0 || it >= KWS_M1GW_T_IN) continue;
                    for (int km = -1; km <= 1; km++) {
                        const int im = om * 2 + km;
                        if (im < 0 || im >= KWS_M1GW_M_IN) continue;
                        acc += (float)w[(size_t)c * 9 + (kt + 1) * 3 + (km + 1)]
                               * (float)A->delta[(size_t)c * (KWS_M1GW_T_IN * KWS_M1GW_M_IN)
                                                 + it * KWS_M1GW_M_IN + im];
                    }
                }
                A->dwf[op * 3 + c] = acc * wsc[c] * dsc;
            }
        }
        groupnorm_silu(A->dwf, red_pos, 3, 1,
                       Tf(KWS_M1GW_T_STEM1_NORM_WEIGHT),
                       Tf(KWS_M1GW_T_STEM1_NORM_BIAS));
#ifdef KWS_HOST_TEST
        memcpy(A->snap, A->dwf, sizeof(A->snap));
#endif
        const float gs = requantise_rows(A->dwf, A->gnq, red_pos, 3);
#ifdef KWS_HOST_TEST
        A->snap_scale = gs;
        memcpy(A->snapq, A->gnq, sizeof(A->snapq));
#endif
        gemm_row_q(A->gnq, gs, red_pos, 3,
                   KWS_M1GW_T_STEM1_PW_WEIGHT, -1, KWS_M1GW_STEM,
                   A->s1, A->s1_sc);
    }
    STAGE_END();

    /* ---- down: dw(time stride 4) -> GN(4 groups over 32) -> SiLU -> pw -----
     * 49 frames at stride 4 with a 3-tap kernel gives 13, which is the whole
     * reason this model is cheap: every downstream tensor is 13 frames deep. */
    {
        const int out_pos = KWS_M1GW_T2_POS;
        depthwise_i8(A->s1, A->s1_sc, KWS_M1GW_T_IN, KWS_M1GW_M1,
                     KWS_M1GW_STEM, KWS_M1GW_T_DOWN_DW_WEIGHT, 4, 1,
                     KWS_M1GW_T2, KWS_M1GW_M1, A->dwf);
        groupnorm_silu(A->dwf, out_pos, KWS_M1GW_STEM, 4,
                       Tf(KWS_M1GW_T_DOWN_NORM_WEIGHT),
                       Tf(KWS_M1GW_T_DOWN_NORM_BIAS));
        const float gs = requantise_rows(A->dwf, A->gnq, out_pos, KWS_M1GW_STEM);
        gemm_row_q(A->gnq, gs, out_pos, KWS_M1GW_STEM,
                   KWS_M1GW_T_DOWN_PW_WEIGHT, -1, KWS_M1GW_STEM,
                   A->dn, A->dn_sc);
    }
    STAGE_END();

    /* ---- stem2: dw(1,2) -> GN(4 over 32) -> SiLU -> pw 32->96 -------------- */
    {
        const int out_pos = KWS_M1GW_S2_POS;
        depthwise_i8(A->dn, A->dn_sc, KWS_M1GW_T2, KWS_M1GW_M1,
                     KWS_M1GW_STEM, KWS_M1GW_T_STEM2_DW_WEIGHT, 1, 2,
                     KWS_M1GW_T2, KWS_M1GW_M2, A->dwf);
        groupnorm_silu(A->dwf, out_pos, KWS_M1GW_STEM, 4,
                       Tf(KWS_M1GW_T_STEM2_NORM_WEIGHT),
                       Tf(KWS_M1GW_T_STEM2_NORM_BIAS));
        const float gs = requantise_rows(A->dwf, A->gnq, out_pos, KWS_M1GW_STEM);
        gemm_row_q(A->gnq, gs, out_pos, KWS_M1GW_STEM,
                   KWS_M1GW_T_STEM2_PW_WEIGHT, -1, KWS_M1GW_DIM,
                   A->s2, A->s2_sc);
    }
    STAGE_END();

    /* ---- MelAttnPool: learned softmax over the 10 mel bins, per frame ------
     * Mean-pooling over mel was measurably worse in training: it throws away
     * formant structure, and WHICH bin carries the energy is the entire point
     * of a log-mel front end. Reads the int8 tensor directly. */
    {
        const int T = KWS_M1GW_T2, M = KWS_M1GW_M2;
        const int8_t *w1 = Ti(KWS_M1GW_T_MEL_GATE_FC1_WEIGHT);
        const float *w1s = Tsc(KWS_M1GW_T_MEL_GATE_FC1_WEIGHT);
        const float *b1 = Tf(KWS_M1GW_T_MEL_GATE_FC1_BIAS);
        const int8_t *w2 = Ti(KWS_M1GW_T_MEL_GATE_FC2_WEIGHT);
        const float w2s = Tsc(KWS_M1GW_T_MEL_GATE_FC2_WEIGHT)[0];
        float wts[KWS_M1GW_M2];

        for (int t = 0; t < T; t++) {
            for (int m = 0; m < M; m++) {
                const int p = t * M + m;
                const float xs = A->s2_sc[p];
                const int8_t *xr = A->s2 + (size_t)p * KWS_M1GW_DIM;
                for (int r = 0; r < KWS_M1GW_MEL_RANK; r++) {
                    float acc = b1[r];
                    const int8_t *wr = w1 + (size_t)r * KWS_M1GW_DIM;
                    for (int c = 0; c < KWS_M1GW_DIM; c++)
                        acc += (float)wr[c] * w1s[r] * (float)xr[c] * xs;
                    /* tanh via the fast exp rather than libm tanhf, which is
                     * itself an expf plus a divide: 1,040 of these per
                     * inference, and this stage measured 11 ms largely for them.
                     * tanh(x) = (1 - e^-2x) / (1 + e^-2x), arranged so the
                     * exponential argument is non-positive and cannot overflow. */
                    const float e2 = kws_expf(-2.0f * acc);
                    A->mel[(size_t)p * KWS_M1GW_MEL_RANK + r] =
                        (1.0f - e2) * kws_rcp(1.0f + e2);
                }
            }
            /* fc2 collapses the rank axis to one score per mel bin, then those
             * M scores are softmaxed against each other. */
            for (int m = 0; m < M; m++) {
                const float *sr = A->mel + (size_t)(t * M + m) * KWS_M1GW_MEL_RANK;
                float acc = 0.0f;
                for (int r = 0; r < KWS_M1GW_MEL_RANK; r++)
                    acc += (float)w2[r] * w2s * sr[r];
                wts[m] = acc;
            }
            softmax_inplace(wts, M);
            /* weighted sum back over mel, into the transposed sequence layout */
            for (int c = 0; c < KWS_M1GW_DIM; c++) {
                float acc = 0.0f;
                for (int m = 0; m < M; m++) {
                    const int p = t * M + m;
                    acc += wts[m] * (float)A->s2[(size_t)p * KWS_M1GW_DIM + c]
                               * A->s2_sc[p];
                }
                A->seq[(size_t)t * KWS_M1GW_DIM + c] = acc;
            }
        }
    }
    STAGE_END();


#ifdef KWS_HOST_TEST
    memcpy(A->snapseq, A->seq, sizeof(A->snapseq));
#endif
    CHK("seq after mel_gate", A->seq, KWS_M1GW_T2 * KWS_M1GW_DIM);

/* ---- 2 x RelConformerBlock, int8 GEMMs -----------------------------------
 *
 * The float stages that genuinely need float stay float: LayerNorm needs the
 * mean and variance, the attention needs a softmax, SiLU and the SE gate are
 * transcendentals. Everything in between -- qkv, proj, the conv pointwise, the
 * two FFN projections -- is an int8 GEMM through the same
 * kws_accum_row_i8() the stems use.
 *
 * That is the whole 99 ms. dense_f32() ran 1,771,776 MAC as scalar float
 * multiply-adds at 13.4 cycles/MAC on a core whose whole reason for existing
 * here is an int8 vector reduction; at the reduction's measured rate the same
 * arithmetic is a couple of milliseconds.
 *
 * Quantisation is per row, folded to a tensor scale for the GEMM, exactly the
 * scheme requantise_rows() already uses in the stems -- so the numerics are the
 * numerics that were verified against the float reference.
 *
 * ONE loop per block: attention -> depthwise+SE -> FFN. These are three
 * dependent stages of one residual block, not three passes. An earlier version
 * ran every block's attention before any block's FFN, which is a different
 * network and also mis-timed half the work into the wrong stage slot.
 */
    for (int b = 0; b < KWS_M1GW_BLOCKS; b++) {
        const int T = KWS_M1GW_T2, D = KWS_M1GW_DIM;
        const int rowbase = b * 20;   /* 20 tensors per block in the export table */

        /* --- attention ------------------------------------------------- */
        layernorm(A->seq, A->blk, T, D,
                  Tf(KWS_M1GW_T_BLOCKS_0_NORM1_WEIGHT + rowbase),
                  Tf(KWS_M1GW_T_BLOCKS_0_NORM1_BIAS + rowbase));

        /* qkv: int8 in, int8 out. */
        PROJ_BEGIN(0, q);
        const float xsc = requantise_rows(A->blk, A->gq, T, D);
        gemm_row_q(A->gq, xsc, T, D,
                   KWS_M1GW_T_BLOCKS_0_ATTN_QKV_WEIGHT + rowbase,
                   KWS_M1GW_T_BLOCKS_0_ATTN_QKV_BIAS + rowbase,
                   3 * D, A->gi, A->gsc);
        PROJ_END(0, q);

        /* The attention itself is small (2 x h x T x T x head_dim = 32 k MAC),
         * so it reads dequantised values rather than being forced through int8:
         * the softmax and the Q.K product are float anyway, and hoisting the
         * dequantise here is cheaper than requantising between them. */
        for (int i = 0; i < T * 3 * D; i++) {
            A->qkv[i] = (float)A->gi[i] * A->gsc[i / (3 * D)];
        }

        const float inv_sqrt = 1.0f / sqrtf((float)KWS_M1GW_HEAD_DIM);
        const float *rb = Tf(KWS_M1GW_T_BLOCKS_0_ATTN_REL_BIAS + rowbase);
        for (int h = 0; h < KWS_M1GW_HEADS; h++) {
            float *L = A->logits + (size_t)h * T * T;
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    float acc = 0.0f;
                    for (int d = 0; d < KWS_M1GW_HEAD_DIM; d++) {
                        acc += A->qkv[(size_t)i * 3 * D + h * KWS_M1GW_HEAD_DIM + d]
                             * A->qkv[(size_t)j * 3 * D + D + h * KWS_M1GW_HEAD_DIM + d];
                    }
                    /* Relative only, indexed by the SIGNED distance shifted by
                     * T-1. An absolute position embedding here cost 62 points
                     * of recall in training; this is why it is relative. */
                    L[i * T + j] = acc * inv_sqrt
                                  + rb[(size_t)(i - j + (T - 1)) * KWS_M1GW_HEADS + h];
                }
            }
            for (int i = 0; i < T; i++) softmax_inplace(L + i * T, T);

            for (int i = 0; i < T; i++) {
                for (int d = 0; d < KWS_M1GW_HEAD_DIM; d++) {
                    float acc = 0.0f;
                    for (int j = 0; j < T; j++) {
                        acc += L[i * T + j]
                             * A->qkv[(size_t)j * 3 * D + 2 * D + h * KWS_M1GW_HEAD_DIM + d];
                    }
                    A->blk[(size_t)i * D + h * KWS_M1GW_HEAD_DIM + d] = acc;
                }
            }
        }

        /* proj: int8 in, int8 out, then dequantise straight into the residual. */
        PROJ_BEGIN(1, p);
        const float psc = requantise_rows(A->blk, A->gq, T, D);
        gemm_row_q(A->gq, psc, T, D,
                   KWS_M1GW_T_BLOCKS_0_ATTN_PROJ_WEIGHT + rowbase,
                   KWS_M1GW_T_BLOCKS_0_ATTN_PROJ_BIAS + rowbase,
                   D, A->gi, A->gsc);
        PROJ_END(1, p);
        for (int i = 0; i < T * D; i++) {
            A->seq[i] += (float)A->gi[i] * A->gsc[i / D];
        }

        /* --- depthwise k=3 and k=7 over time, SiLU'd, then summed ---------- */
        layernorm(A->seq, A->blk, T, D,
                  Tf(KWS_M1GW_T_BLOCKS_0_NORM2_WEIGHT + rowbase),
                  Tf(KWS_M1GW_T_BLOCKS_0_NORM2_BIAS + rowbase));

        const int8_t *w3 = Ti(KWS_M1GW_T_BLOCKS_0_DW_0_WEIGHT + rowbase);
        const int8_t *w7 = Ti(KWS_M1GW_T_BLOCKS_0_DW_1_WEIGHT + rowbase);
        const float *s3 = Tsc(KWS_M1GW_T_BLOCKS_0_DW_0_WEIGHT + rowbase);
        const float *s7 = Tsc(KWS_M1GW_T_BLOCKS_0_DW_1_WEIGHT + rowbase);

        /* The taps gather from A->blk and land in A->gq: in place would let
         * frame t's write clobber the neighbours frame t+1 is still reading. */
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < D; c++) {
                float a3 = 0.0f, a7 = 0.0f;
                for (int kt = -1; kt <= 1; kt++) {
                    const int it = t + kt;
                    if (it < 0 || it >= T) continue;
                    a3 += (float)w3[(size_t)c * 3 + (kt + 1)] * s3[c]
                          * A->blk[(size_t)it * D + c];
                }
                for (int kt = -3; kt <= 3; kt++) {
                    const int it = t + kt;
                    if (it < 0 || it >= T) continue;
                    a7 += (float)w7[(size_t)c * 7 + (kt + 3)] * s7[c]
                          * A->blk[(size_t)it * D + c];
                }
                A->dwf[(size_t)t * D + c] = silu(a3) + silu(a7);
            }
        }

        /* conv pointwise: int8 in, int8 out. */
        PROJ_BEGIN(2, c);
        const float csc = requantise_rows(A->dwf, A->gq, T, D);
        gemm_row_q(A->gq, csc, T, D,
                   KWS_M1GW_T_BLOCKS_0_PW_WEIGHT + rowbase, -1, D,
                   A->gi, A->gsc);
        PROJ_END(2, c);
        for (int i = 0; i < T * D; i++) {
            A->blk[i] = (float)A->gi[i] * A->gsc[i / D];
        }

        /* squeeze-excite over time, in float: it is 2,304 MAC total and needs a
         * sigmoid, which is not worth pushing through int8. */
        {
            float mean[KWS_M1GW_DIM];
            for (int c = 0; c < D; c++) {
                float acc = 0.0f;
                for (int t = 0; t < T; t++) acc += A->blk[(size_t)t * D + c];
                mean[c] = acc / (float)T;
            }
            float z[KWS_M1GW_SE];
            {
                const int8_t *w = Ti(KWS_M1GW_T_BLOCKS_0_SE_FC1_WEIGHT + rowbase);
                const float *ws = Tsc(KWS_M1GW_T_BLOCKS_0_SE_FC1_WEIGHT + rowbase);
                for (int r = 0; r < KWS_M1GW_SE; r++) {
                    float acc = 0.0f;
                    for (int c = 0; c < D; c++) {
                        acc += (float)w[(size_t)r * D + c] * ws[r] * mean[c];
                    }
                    z[r] = silu(acc);
                }
            }
            {
                const int8_t *w = Ti(KWS_M1GW_T_BLOCKS_0_SE_FC2_WEIGHT + rowbase);
                const float *ws = Tsc(KWS_M1GW_T_BLOCKS_0_SE_FC2_WEIGHT + rowbase);
                for (int c = 0; c < D; c++) {
                    float acc = 0.0f;
                    for (int r = 0; r < KWS_M1GW_SE; r++) {
                        acc += (float)w[(size_t)c * KWS_M1GW_SE + r] * ws[c] * z[r];
                    }
                    const float g = kws_rcp(1.0f + kws_expf(-acc));
                    for (int t = 0; t < T; t++) A->blk[(size_t)t * D + c] *= g;
                }
            }
        }
        for (int i = 0; i < T * D; i++) A->seq[i] += A->blk[i];

        /* --- Swish FFN + residual ------------------------------------------ */
        layernorm(A->seq, A->blk, T, D,
                  Tf(KWS_M1GW_T_BLOCKS_0_NORM3_WEIGHT + rowbase),
                  Tf(KWS_M1GW_T_BLOCKS_0_NORM3_BIAS + rowbase));

        PROJ_BEGIN(3, f);
        const float fsc = requantise_rows(A->blk, A->gq, T, D);
        gemm_row_q(A->gq, fsc, T, D,
                   KWS_M1GW_T_BLOCKS_0_FF1_WEIGHT + rowbase,
                   KWS_M1GW_T_BLOCKS_0_FF1_BIAS + rowbase,
                   KWS_M1GW_FFN, A->gi, A->gsc);
        PROJ_END(3, f);

        /* SiLU, then straight back to int8 for ff2. Dequantise -> SiLU ->
         * requantise rather than keeping a float hidden layer alive, which is
         * what made the FFN expensive in the first place. */
        for (int i = 0; i < T * KWS_M1GW_FFN; i++) {
            A->dwf[i] = silu((float)A->gi[i] * A->gsc[i / KWS_M1GW_FFN]);
        }
        PROJ_BEGIN(4, g);
        const float f2sc = requantise_rows(A->dwf, A->gq, T, KWS_M1GW_FFN);
        gemm_row_q(A->gq, f2sc, T, KWS_M1GW_FFN,
                   KWS_M1GW_T_BLOCKS_0_FF2_WEIGHT + rowbase,
                   KWS_M1GW_T_BLOCKS_0_FF2_BIAS + rowbase,
                   D, A->gi, A->gsc);
        PROJ_END(4, g);
        for (int i = 0; i < T * D; i++) {
            A->seq[i] += (float)A->gi[i] * A->gsc[i / D];
        }
    }
    CHK("seq after blocks", A->seq, KWS_M1GW_T2 * KWS_M1GW_DIM);
    STAGE_END();

    /* ---- SoftORPool: mean + logsumexp + max, then the 288->2 head ---------
     * All three statistics are permutation-invariant over time by construction,
     * which is the point: a softmax attention pool would let one frame win and
     * reintroduce the positional prior the relative bias exists to remove. */
    {
        const int T = KWS_M1GW_T2, D = KWS_M1GW_DIM;
        for (int c = 0; c < D; c++) {
            float sum = 0.0f, mx = A->seq[c];
            for (int t = 0; t < T; t++) {
                const float v = A->seq[(size_t)t * D + c];
                sum += v;
                if (v > mx) mx = v;
            }
            const float mean = sum / (float)T;

            /* logsumexp, accumulated in double and seeded at zero.
             *
             * The natural float form is `lse = -INFINITY; lse += expf(v - mx)`,
             * and it is what this used. It is fragile in a way that matters
             * here: seeding with -INFINITY makes the result depend on every term
             * being non-zero, so a single underflowed expf -- and float expf
             * underflows to 0 below about -88 -- poisons the whole statistic.
             * The max-subtraction already makes each term <= 1, so the sum needs
             * no such seed. Double accumulation also keeps 13 terms exact.
             */
            /* Single precision, deliberately.
             *
             * This was double. The ESP32-S3 has one single-precision FPU and
             * NO double-precision hardware at all: every double operation is a
             * software routine. 1,248 double exp() plus 96 double log() is the
             * whole 12.4 ms this stage was reporting for 576 multiply-accumulates
             * -- 5167 cycles each -- which is why replacing libm's expf with a
             * fast one moved this stage by 0.01 ms and looked like a null
             * result. It was never the expf.
             *
             * Float is safe here because the maximum is already subtracted, so
             * every term is in (0, 1] and the sum is at least 1. That is the
             * invariant the earlier -INFINITY seed was breaking; removing the
             * seed and keeping the max subtraction removes the problem the
             * double was working around.
             */
            float lse = 0.0f;
            for (int t = 0; t < T; t++) {
                lse += kws_expf(A->seq[(size_t)t * D + c] - mx);
            }
            A->pool[c] = mean;
            A->pool[D + c] = mx + logf(lse) - KWS_M1GW_LOG_T;
            A->pool[2 * D + c] = mx;
        }
        CHK("pool", A->pool, KWS_M1GW_POOL);
        dense_f32(A->pool, 1, KWS_M1GW_POOL, KWS_M1GW_CLASSES,
                  KWS_M1GW_T_HEAD_WEIGHT, KWS_M1GW_T_HEAD_BIAS, A->head);
        softmax_inplace(A->head, KWS_M1GW_CLASSES);
    }
    STAGE_END();

#undef STAGE_END

    const int64_t t_end = us_now();
    const uint32_t c_end = cyc_now();
    s_last_us = (uint32_t)(t_end - t0);
    s_last_cycles = c_end - c_begin;
    s_total_us += s_last_us;
    s_total_cycles += s_last_cycles;
    if (s_last_us > s_peak_us) s_peak_us = s_last_us;
    s_stage_runs++;
    /* Class 1 is the keyword; the old model's head was Dense(->2) -> softmax and
     * P(keyword) was the positive logit. */
    return A->head[1];
}

/* ------------------------------------------------------------ accessors -- */

uint32_t kws_m1gw_last_us(void) { return s_last_us; }
uint64_t kws_m1gw_total_us(void) { return s_total_us; }
void     kws_m1gw_reset_timing(void) { s_total_us = 0; }

const char *kws_m1gw_stage_name(int slot)
{
    return (slot >= 0 && slot < KWS_M1GW_STAGE_COUNT) ? s_stage_name[slot] : "?";
}

uint64_t kws_m1gw_stage_us(int slot)
{
    return (slot >= 0 && slot < KWS_M1GW_STAGE_COUNT) ? s_stage_us[slot] : 0;
}

uint64_t kws_m1gw_stage_cycles(int slot)
{
    return (slot >= 0 && slot < KWS_M1GW_STAGE_COUNT) ? s_stage_cycles[slot] : 0;
}

uint32_t kws_m1gw_stage_mac(int slot)
{
    return (slot >= 0 && slot < KWS_M1GW_STAGE_COUNT) ? s_stage_mac[slot] : 0;
}

uint32_t kws_m1gw_total_mac(void)
{
    uint32_t tot = 0;
    for (int i = 0; i < KWS_M1GW_STAGE_COUNT; i++) tot += s_stage_mac[i];
    return tot;
}

uint32_t kws_m1gw_cpu_hz(void)
{
    /* Derived from the measurements rather than read from a config symbol.
     *
     * esp_cpu_get_freq_hz() is not public in IDF 6.1 (the frequency lives
     * behind esp_private/esp_clk.h), and reading the configured value would be
     * the wrong answer anyway: what matters is the rate the core ACTUALLY ran
     * at, which differs from the configured one under frequency scaling and
     * power management. Dividing the accumulated cycles by the accumulated
     * microseconds measures it, and the discrepancy against
     * CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ is itself worth seeing in the log. */
    if (s_total_us == 0) return 0;
    return (uint32_t)((s_total_cycles * 1000000ULL) / s_total_us);
}

uint32_t kws_m1gw_last_cycles(void) { return s_last_cycles; }

uint32_t kws_m1gw_peak_us(void) { return s_peak_us; }

uint64_t kws_m1gw_stage_runs(void) { return s_stage_runs; }

const char *kws_m1gw_proj_name(int slot)
{
    return (slot >= 0 && slot < KWS_M1GW_PROJ_COUNT) ? s_proj_name[slot] : "?";
}

uint64_t kws_m1gw_proj_us(int slot)
{
    return (slot >= 0 && slot < KWS_M1GW_PROJ_COUNT) ? s_proj_us[slot] : 0;
}

uint64_t kws_m1gw_proj_cycles(int slot)
{
    return (slot >= 0 && slot < KWS_M1GW_PROJ_COUNT) ? s_proj_cycles[slot] : 0;
}

void kws_m1gw_proj_reset(void)
{
    memset(s_proj_us, 0, sizeof(s_proj_us));
    memset(s_proj_cycles, 0, sizeof(s_proj_cycles));
}

void kws_m1gw_stage_reset(void)
{
    memset(s_stage_us, 0, sizeof(s_stage_us));
    memset(s_stage_cycles, 0, sizeof(s_stage_cycles));
    s_stage_runs = 0;
    s_total_us = 0;
    s_total_cycles = 0;
    s_peak_us = 0;
}

uint32_t kws_m1gw_ram_bytes(void) { return KWS_M1GW_RAM_BYTES; }

float kws_m1gw_threshold(void) { return KWS_M1GW_THRESHOLD; }

#ifdef KWS_HOST_TEST
/* Debug taps. Comparing these against tools/m1gw_reference.py pinpoints the
 * FIRST layer where the two implementations disagree, which is the only way to
 * bisect a port this size -- the final probabilities just say "wrong". */
void kws_m1gw_dbg_s2(const int8_t **q, const float **sc, int *n)
{
    *q = s_arena.s2;
    *sc = s_arena.s2_sc;
    *n = KWS_M1GW_S2_POS * KWS_M1GW_DIM;
}

void kws_m1gw_dbg_s1(const int8_t **q, const float **sc, int *n)
{
    *q = s_arena.s1;
    *sc = s_arena.s1_sc;
    *n = KWS_M1GW_MAX_POS * KWS_M1GW_STEM;
}

void kws_m1gw_dbg_snapq(const int8_t **q, float *sc, int *n)
{
    *q = s_arena.snapq;
    *sc = s_arena.snap_scale;
    *n = KWS_M1GW_MAX_POS * 3;
}

void kws_m1gw_dbg_dwf(const float **v, int *n)
{
    *v = s_arena.snap;
    *n = KWS_M1GW_MAX_POS * 3;      /* the stem1 stage: 980 rows x 3 channels */
}

void kws_m1gw_dbg_dn(const int8_t **q, const float **sc, int *n)
{
    *q = s_arena.dn;
    *sc = s_arena.dn_sc;
    *n = KWS_M1GW_T2_POS * KWS_M1GW_STEM;
}

void kws_m1gw_dbg_melseq(const float **v, int *n)
{
    *v = s_arena.snapseq;
    *n = KWS_M1GW_T2 * KWS_M1GW_DIM;
}

void kws_m1gw_dbg_seq(const float **v, int *n)
{
    *v = s_arena.seq;
    *n = KWS_M1GW_T2 * KWS_M1GW_DIM;
}

void kws_m1gw_dbg_pool(const float **v, int *n)
{
    *v = s_arena.pool;
    *n = KWS_M1GW_POOL;
}
#endif