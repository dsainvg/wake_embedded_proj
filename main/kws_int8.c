/*
 * Custom INT8 kernels for the Amaze wake-word network.
 *
 * Read kws_int8.h first: it explains why the weights are transposed at export
 * time and why the activation scales are measured per row instead of being
 * calibrated offline.
 *
 * Two implementations of the inner reduction exist and they must agree exactly:
 *
 *   portable  plain C, and the definition of the arithmetic. Runs on the host
 *             harness, which is where every numerical claim about this port was
 *             measured.
 *   xtensa    hand-written assembly driving EE.VMULAS.S8.QACC, four int8
 *             multiply-accumulates per instruction. Reachable only on the
 *             ESP32-S3, and selected automatically whenever the target is
 *             Xtensa, so the fast path cannot be silently off. The host test
 *             compares it against the portable loop over random data; the
 *             firmware runs the same check once at startup and refuses to arm
 *             the detector if they disagree.
 */

#include "kws_int8.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__XTENSA__)
#define KWS_XTENSA 1
extern int32_t kws_asm_dot_i8(const int8_t *a, const int8_t *b, int n);
extern void kws_asm_dot_taps_i8(const int8_t *a, const int8_t *b,
                                int n, int ntaps, int32_t *out);
#else
#define KWS_XTENSA 0
#endif

/* One frame's worth of gathered conv2 windows, so the weight matrix is fetched
 * once per frame instead of once per output element. out_w is 10 and a window is
 * 288 bytes, so 2,880 bytes -- small enough to stay in internal RAM. */
#define KWS_MAX_WINDOWS 10
/* conv2 is a 3x3 kernel, so nine taps. Bounds the per-tap scale array and the
 * output of the windowed reduction. */
#define KWS_MAX_TAPS    9

/* Scratch buffers. The kernels run one inference deep inside one FreeRTOS task,
 * so file-static scratch costs nothing in stack terms and saves 1.4 KB of task
 * stack, which on this part is the scarcer resource.
 *
 * Both are 16-byte aligned because the vector reduction loads 128 bits at a
 * time and the Xtensa will not take a misaligned vector load. */
static int8_t s_gat_win[KWS_MAX_WINDOWS * KWS_KERN_MAX_IN] __attribute__((aligned(16)));
static float  s_tap_sc[KWS_MAX_WINDOWS * KWS_MAX_TAPS];
/* Where the windowed assembly reduction leaves its per-tap results. */
#if KWS_XTENSA
static int32_t s_dot_out[KWS_MAX_TAPS];
#endif

/*
 * Reduction stride, in elements.
 *
 * tools/export_weights.py pads every weight row and every quantised activation
 * row out to a multiple of 16, zero filled. Both operands of the reduction are
 * therefore 16-byte aligned and the vector path needs no shifting, no tail, and
 * no unaligned-load handling: one EE.VMULAS.S8.ACCX instruction per 16
 * multiply-accumulates. The zero pad contributes nothing to the dot product.
 *
 * Getting this wrong is not a rounding error. The KWS_ALIGN constant is
 * asserted against the value the exporter used, so a mismatch is a build
 * failure rather than a plausible-looking wrong answer.
 */
#define KWS_ALIGN KWS_ALIGN16

static inline int align16(int n) { return (n + (KWS_ALIGN - 1)) & ~(KWS_ALIGN - 1); }

/*
 * Backend selection. Default is "use the assembly reduction"; kws_kernel_selftest
 * demotes to the portable one if it fails. Nothing else writes this, and the
 * demotion is one-way.
 */
#if KWS_XTENSA
static int s_use_asm = 1;
#else
static int s_use_asm = 0;
#endif

/* Set by kws_kernel_selftest if the windowed reduction disagrees with the
 * portable loop, in which case conv2 falls back to portable as well. */
#if KWS_XTENSA
static int s_use_asm_taps = 1;

/* Whether the windowed reduction conv2 uses is the assembly one. Reported
 * separately from kws_kernel_using_xtensa() because it is demoted on its own
 * evidence: a tap-loop fault costs conv2 speed and nothing else, and a boot log
 * that only said "xtensa vector" would hide that entirely. */
int kws_kernel_taps_using_xtensa(void) { return s_use_asm_taps; }
#else
int kws_kernel_taps_using_xtensa(void) { return 0; }
#endif

int kws_kernel_uses_xtensa(void) { return KWS_XTENSA; }
int kws_kernel_using_xtensa(void) { return s_use_asm; }
void kws_kernel_force_portable(void) { s_use_asm = 0; }

/*
 * Compare the assembly reduction against the portable one on data that has to
 * break it if it is wrong: full-scale values that make every product the largest
 * it can be, values of alternating sign so the accumulator really does change
 * sign, and lengths that are and are not multiples of four because the
 * assembly path is unrolled by four.
 *
 * The reduction axis in this network is at most 288 long, and the accumulator
 * is 32-bit signed. 288 * 127 * 127 = 4.6e6, so overflow is not a risk, but the
 * full-scale pattern is what catches a dropped lane or a lane pair counted
 * twice.
 */
int kws_kernel_selftest(void)
{
#if !KWS_XTENSA
    return 1;
#else
    /*
     * The assembly reduction is only ever entered for a 16-byte-aligned
     * operand pair and a multiple-of-16 length, so those are the cases tested.
     * The lengths are the ones this network actually reduces over: conv2 now
     * reduces one 32-channel tap at a time rather than a whole 288-element
     * window, and the conformer and classifier reduce over 48, 96, 6 and 112,
     * plus the two shortest multiples of 16 and one longer than anything used.
     */
    static const int lens[] = { 16, 32, 48, 64, 96, 112, 144, 288, 512 };
    static int8_t a[512] __attribute__((aligned(16)));
    static int8_t b[512] __attribute__((aligned(16)));
    uint32_t r = 0x12345678u;
    size_t li;
    int i;

    for (li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
        const int n = lens[li];

        /* xorshift32, so the host harness can reproduce the exact pattern. */
        for (i = 0; i < n; i++) {
            r ^= r << 13; r ^= r >> 17; r ^= r << 5;
            a[i] = (int8_t)((int)(r % 255u) - 127);
            b[i] = (int8_t)((int)((r >> 11) % 255u) - 127);
        }
        if (kws_asm_dot_i8(a, b, n) != kws_dot_i8(a, b, n)) { goto fail; }

        /* Full-scale with alternating sign: the largest accumulator this
         * reduction can reach, and one that changes sign at every element. A
         * lane that is dropped or counted twice cannot hide inside it. */
        for (i = 0; i < n; i++) {
            a[i] = (int8_t)((i & 1) ? -127 : 127);
            b[i] = (int8_t)((i & 1) ? 127 : -127);
        }
        if (kws_asm_dot_i8(a, b, n) != kws_dot_i8(a, b, n)) { goto fail; }

        /* One non-zero element at a time, at every offset within a block.
         * This is what pins down which lanes each of the two half-word
         * accumulator halves covers: an instruction that only multiplies the
         * low eight bytes agrees with the reference on random data often enough
         * to be worth a run of single-element probes. */
        for (i = 0; i < n; i++) {
            memset(a, 0, sizeof(a));
            memset(b, 0, sizeof(b));
            a[i] = (int8_t)(100 - (i & 63));
            b[i] = (int8_t)((i & 1) ? -3 : 3);
            if (kws_asm_dot_i8(a, b, n) != kws_dot_i8(a, b, n)) { goto fail; }
        }
    }

    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));
    if (kws_asm_dot_i8(a, b, 512) != 0) { goto fail; }
    a[0] = 100; b[0] = -3;
    if (kws_asm_dot_i8(a, b, 16) != -300) { goto fail; }

    /*
     * The windowed variant, which is the one conv2 actually uses and which has
     * more moving parts than the plain dot: it walks taps, it carries a float
     * scale per tap, and it advances the two operand pointers by a stride the
     * caller chose. Each of those can be wrong in a way that produces plausible
     * numbers, so it is checked against the portable loop on the tap counts and
     * strides this network uses -- (9 taps of 32 channels) for conv2, and the
     * single- and few-tap shapes a future caller might reach for.
     *
     * The scales are deliberately not all 1.0: a scale of exactly 1.0 makes a
     * missing multiply indistinguishable from a correct one.
     */
    {
        static const int taps[] = { 1, 3, 9 };
        static const int strides[] = { 16, 32, 48 };
        int32_t got[KWS_MAX_TAPS];
        size_t ti, si, k;

        for (ti = 0; ti < sizeof(taps) / sizeof(taps[0]); ti++) {
            for (si = 0; si < sizeof(strides) / sizeof(strides[0]); si++) {
                const int nt = taps[ti];
                const int st = strides[si];
                if (nt > KWS_MAX_TAPS) { continue; }
                const int total = nt * st;
                if (total > (int)sizeof(a)) { continue; }

                /* Random data, then full-scale alternating sign: the largest
                 * sums, and sums that change sign between taps, which is where
                 * a result written to the wrong slot would show up. */
                for (i = 0; i < total; i++) {
                    r ^= r << 13; r ^= r >> 17; r ^= r << 5;
                    a[i] = (int8_t)((int)(r % 255u) - 127);
                    r ^= r << 13; r ^= r >> 17; r ^= r << 5;
                    b[i] = (int8_t)((int)(r % 255u) - 127);
                }
                kws_asm_dot_taps_i8(a, b, st, nt, got);
                for (k = 0; k < (size_t)nt; k++) {
                    if (got[k] != kws_dot_i8(a + k * st, b + k * st, st)) {
                        goto fail_taps;
                    }
                }

                for (k = 0; k < (size_t)nt; k++) {
                    for (i = 0; i < st; i++) {
                        const int o = (int)k * st + i;
                        a[o] = (int8_t)((i & 1) ? -127 : 127);
                        b[o] = (int8_t)((k & 1) ? -127 : 127);
                    }
                }
                kws_asm_dot_taps_i8(a, b, st, nt, got);
                for (k = 0; k < (size_t)nt; k++) {
                    if (got[k] != kws_dot_i8(a + k * st, b + k * st, st)) {
                        goto fail_taps;
                    }
                }
            }
        }
    }
    return 1;

fail_taps:
    /*
     * Only the windowed reduction is demoted. It arrived after the plain dot,
     * it is used by conv2 alone, and taking the whole kernel down with it
     * would cost the conformer and the classifier -- which the plain dot still
     * does correctly -- a 3x slowdown for a bug in a different function. This
     * distinction was learned the hard way: a tap-loop bug demoted everything
     * and the result was an inference 6.8x SLOWER than before the fix.
     */
    s_use_asm_taps = 0;
    return 1;

fail:
    s_use_asm = 0;
    s_use_asm_taps = 0;
    return 0;
#endif
}

/* ------------------------------------------------------------------ */
/* Quantisation                                                         */
/* ------------------------------------------------------------------ */

float kws_quant_row(const float *x, int n, int8_t *q)
{
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        const float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > m) { m = a; }
    }
    if (m < 1e-20f) {
        memset(q, 0, (size_t)n);
        return 1.0f;
    }
    const float scale = m * (1.0f / 127.0f);
    const float inv   = 127.0f / m;
    for (int i = 0; i < n; i++) {
        const float v = x[i] * inv;
        /* Round half away from zero, then clamp. A cast plus a bias rather than
         * a library call: this runs once per activation element, which is
         * roughly 300k times per inference. */
        int32_t t = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
        if (t >  127) { t =  127; }
        if (t < -127) { t = -127; }
        q[i] = (int8_t)t;
    }
    return scale;
}

/* ------------------------------------------------------------------ */
/* Dot product                                                          */
/* ------------------------------------------------------------------ */

int32_t kws_dot_i8(const int8_t *a, const int8_t *b, int n)
{
    int32_t acc = 0;
    for (int i = 0; i < n; i++) {
        acc += (int32_t)a[i] * (int32_t)b[i];
    }
    return acc;
}

int32_t kws_dot_i8_ref(const int8_t *a, const int8_t *b, int n)
{
#if KWS_XTENSA
    /* The assembly reduction has two preconditions: both operands 16-byte
     * aligned, and a length that is a multiple of 16. Every call site in this
     * network satisfies both, but the check is here rather than trusted, because
     * a kernel that quietly reads past the end of an activation row is not a
     * performance bug, it is a wrong answer. */
    if (s_use_asm && (n & (KWS_ALIGN - 1)) == 0 &&
        ((((uintptr_t)a) | ((uintptr_t)b)) & (KWS_ALIGN - 1u)) == 0) {
        return kws_asm_dot_i8(a, b, n);
    }
#endif
    return kws_dot_i8(a, b, n);
}

/* ------------------------------------------------------------------ */
/* Row against a whole weight matrix                                    */
/* ------------------------------------------------------------------ */

/*
 * `in_dim` is the LOGICAL reduction length; the weight rows are
 * align16(in_dim) apart. Padding bytes are zero in both operands, so the
 * accumulator sees only live values.
 *
 * Four output channels at a time.
 *
 * The reduction is 48 MACs for the conformer's Dense layers -- three
 * EE.VMULAS -- and there are 7,056 of them per layer. At that ratio the call
 * prologue, the alignment test and the accumulator setup cost more than the
 * multiply, so a row-at-a-time reduction spends most of its time setting up
 * rather than reducing.
 *
 * Unrolling by four keeps the four partial sums in registers and hoists the
 * operand checks out of all four, which is the part that does not scale with
 * out_dim. The reduction is re-read for each of the four, so this trades
 * bandwidth for ALU work: on this part the ALU is the scarcer resource, and the
 * `q` row is hot in cache for all four.
 */
void kws_accum_row_i8(float *y, const int8_t *q, const int8_t *w,
                      const float *bias, float scale,
                      int out_dim, int in_dim)
{
    const int stride = align16(in_dim);
    const int8_t *wp = w;
    int o = 0;

#if KWS_XTENSA
    if (s_use_asm && (stride & (KWS_ALIGN - 1)) == 0 && stride > 0 &&
        ((((uintptr_t)q) | ((uintptr_t)w)) & (KWS_ALIGN - 1u)) == 0) {
        for (; o + 4 <= out_dim; o += 4, wp += 4 * stride) {
            const int32_t a0 = kws_asm_dot_i8(q, wp,                stride);
            const int32_t a1 = kws_asm_dot_i8(q, wp + stride,        stride);
            const int32_t a2 = kws_asm_dot_i8(q, wp + 2 * stride,    stride);
            const int32_t a3 = kws_asm_dot_i8(q, wp + 3 * stride,    stride);
            y[o + 0] = (float)a0 * scale + (bias != NULL ? bias[o + 0] : 0.0f);
            y[o + 1] = (float)a1 * scale + (bias != NULL ? bias[o + 1] : 0.0f);
            y[o + 2] = (float)a2 * scale + (bias != NULL ? bias[o + 2] : 0.0f);
            y[o + 3] = (float)a3 * scale + (bias != NULL ? bias[o + 3] : 0.0f);
        }
    }
#endif

    for (; o < out_dim; o++, wp += stride) {
        const int32_t a = kws_dot_i8_ref(q, wp, stride);
        y[o] = (float)a * scale + (bias != NULL ? bias[o] : 0.0f);
    }
}

/* ------------------------------------------------------------------ */
/* GEMM                                                                 */
/* ------------------------------------------------------------------ */

void kws_gemm_i8(float *y, const int8_t *xq, const float *xsc,
                 const int8_t *w, const float *bias, float wscale,
                 int rows, int in_dim, int out_dim)
{
    /*
     * Output channel OUTSIDE, row inside.
     *
     * The obvious order -- for each row, reduce every output channel -- re-reads
     * the entire weight matrix once per row. With 49 rows that is 49 passes over
     * a matrix small enough to sit in cache, and this part has no cache worth
     * the name on the far side of PSRAM: the matrix is read from PSRAM 49 times
     * when it could be read once. Measured on hardware, that traffic alone was
     * a large share of the conformer's cost.
     *
     * Reversing the loops makes the weight row the stationary operand. One row
     * is fetched, then walked across every input row while it is still hot, and
     * the next output channel follows. Same arithmetic, same order of
     * accumulation within a row -- only the memory traffic changes.
     *
     * The scales still belong to the input rows, so they are applied per (row,
     * channel) on the way out rather than hoisted, and the bias likewise.
     */
    const int stride = align16(in_dim);
    for (int o = 0; o < out_dim; o++) {
        const int8_t *wo = w + (size_t)o * stride;
        const float b = (bias != NULL) ? bias[o] : 0.0f;
        float *yc = y + o;

        for (int r = 0; r < rows; r++) {
            const int32_t a = kws_dot_i8_ref(xq + (size_t)r * in_dim, wo, stride);
            yc[(size_t)r * out_dim] = (float)a * (xsc[r] * wscale) + b;
        }
    }
}

void kws_gemm_f32(float *y, const int8_t *xq, const float *xsc,
                  const float *w, const float *bias,
                  int rows, int in_dim, int out_dim)
{
    /* Same output-channel-outermost order as the int8 GEMM, for the same reason:
     * the float matrix is four times larger, so re-reading it per row costs four
     * times as much traffic. */
    const int stride = align16(in_dim);
    for (int o = 0; o < out_dim; o++) {
        const float *wo = w + (size_t)o * stride;
        const float b = (bias != NULL) ? bias[o] : 0.0f;
        float *yc = y + o;
        for (int r = 0; r < rows; r++) {
            const int8_t *q = xq + (size_t)r * in_dim;
            float a = 0.0f;
            for (int i = 0; i < stride; i++) {
                a += (float)q[i] * wo[i];
            }
            yc[(size_t)r * out_dim] = a * xsc[r] + b;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Convolution                                                          */
/* ------------------------------------------------------------------ */

/*
 * Gather one output element's window into a contiguous int8 buffer, with no
 * dequantisation and no requantisation, and record the scale of each tap.
 *
 * The input is int8 with one scale per (frame, mel) row, so the nine taps carry
 * nine distinct scales and the window needs nine accumulators -- one per tap.
 * That costs nothing in the reduction itself: nine 32-wide dot products are the
 * same eighteen EE.VMULAS as one 288-wide one, because the reduction axis is
 * still contiguous and 16-byte aligned. The scale is applied once per tap per
 * output channel, 432 multiplies for a whole output element.
 *
 * This is why the scale granularity did not have to be traded away. The earlier
 * version collapsed the window to a single scale and requantised all 288 values
 * for each of the 490 output elements: three scalar loops where a byte copy will
 * do, ~420,000 scalar float operations per inference against 6.8 MMAC of int8
 * reduction that the vector unit does sixteen at a time. It was the single
 * largest cost in the network, and per-tap scales remove all of it.
 *
 * The packing order is TAP FIRST, then channel, and that is not a free choice:
 * the exporter stores conv2 tap-major, (out, kF, kT, in), precisely so that
 * each tap's input channels are contiguous. Under the plain axis reversal the
 * row would be (out, in, kF, kT) and each tap strided by kT*kW, which no
 * EE.VMULAS.S8 can consume -- the kernel would have to run scalar and lose the
 * entire point of the vector reduction.
 *
 * Taps that fall outside the tensor contribute exact zero bytes, which reduce to
 * zero and carry a zero scale, so the zero padding never has to be
 * special-cased.
 */
static void gather_window_i8(const int8_t *xq, const float *xsc,
                             int t, int ow,
                             int frames, int in_w, int in_c,
                             int kT, int kW, int pad_w,
                             int8_t *s_gat, float *tap_scale)
{
    /* Tap scales first: nine of them, indexed kw*kT+kt, and independent of the
     * channel loop. Out-of-range taps get scale 0 to match their zero bytes. */
    for (int kw = 0; kw < kW; kw++) {
        const int iw = ow * 2 + kw - pad_w;
        for (int kt = 0; kt < kT; kt++) {
            const int tt = t + kt - kT / 2;
            tap_scale[kw * kT + kt] =
                (tt >= 0 && tt < frames && iw >= 0 && iw < in_w)
                    ? xsc[(size_t)(tt * in_w + iw)] : 0.0f;
        }
    }

    int n = 0;
    for (int kw = 0; kw < kW; kw++) {
        const int iw = ow * 2 + kw - pad_w;
        for (int kt = 0; kt < kT; kt++) {
            const int tt = t + kt - kT / 2;
            if (tt < 0 || tt >= frames || iw < 0 || iw >= in_w) {
                memset(s_gat + n, 0, (size_t)in_c);
                n += in_c;
                continue;
            }
            const int8_t *src = xq + (size_t)(tt * in_w + iw) * in_c;
            memcpy(s_gat + n, src, (size_t)in_c);
            n += in_c;
        }
    }
}

void kws_conv2d_i8(float *y, const int8_t *xq, const float *xsc,
                   const int8_t *w, float wscale,
                   int frames, int in_w, int out_w,
                   int in_c, int out_c, int kT, int kW)
{
    const int pad_w = ((out_w - 1) * 2 + kW - in_w) / 2;
    const int n_in = align16(kT * kW * in_c);
    const int blk = align16(in_c);
    const int ntaps = kT * kW;

    /*
     * Frame OUTSIDE, output channel inside.
     *
     * conv2 has 490 output elements and a 13,824-byte weight matrix, so looping
     * elements outermost re-reads that matrix 490 times -- 6.46 MB of PSRAM
     * traffic per inference, which measured as the largest single line item on
     * hardware. Here one frame's out_w windows are gathered first (2,880 B of
     * scratch, not 490 windows' worth), and then each output channel's weights
     * are fetched ONCE and applied across all out_w of them while hot. That is
     * 49 passes over the matrix instead of 490.
     *
     * The frames are outermost because the whole point is to amortise the
     * weight fetch over as many output elements as possible, and one frame is
     * as many as fit in scratch without spilling to PSRAM -- which would trade
     * the traffic win back for a latency loss.
     */
    const int win_sz = n_in;

    for (int t = 0; t < frames; t++) {
        /* Gather this frame's mel windows, once, for all 48 channels. */
        for (int ow = 0; ow < out_w; ow++) {
            int8_t *gat = s_gat_win + (size_t)ow * win_sz;
            float *tsc = s_tap_sc + (size_t)ow * KWS_MAX_TAPS;
            gather_window_i8(xq, xsc, t, ow, frames, in_w, in_c,
                             kT, kW, pad_w, gat, tsc);
        }

        float *yt = y + (size_t)t * out_w * out_c;
        for (int o = 0; o < out_c; o++) {
            const int8_t *wo = w + (size_t)o * n_in;
            for (int ow = 0; ow < out_w; ow++) {
                const int8_t *gat = s_gat_win + (size_t)ow * win_sz;
                const float *tsc = s_tap_sc + (size_t)ow * KWS_MAX_TAPS;
                float a;

                /* ONE call for the whole 288-byte window, not nine for its
                 * taps. The taps need separate scales, which is why this was
                 * nine calls in the first place -- but each of those calls
                 * reduced 32 bytes, so it paid a ~15-instruction prologue for
                 * two EE.VMULAS. Nine prologues per output channel, 211,680
                 * times per inference, is the overhead the windowed
                 * reduction in kws_xtensa.S exists to remove. */
#if KWS_XTENSA
                if (s_use_asm_taps &&
                    ((((uintptr_t)gat) | ((uintptr_t)wo)) & (KWS_ALIGN - 1u)) == 0) {
                    kws_asm_dot_taps_i8(gat, wo, blk, ntaps, s_dot_out);
                    a = 0.0f;
                    for (int k = 0; k < ntaps; k++) {
                        a += (float)s_dot_out[k] * tsc[k];
                    }
                } else
#endif
                {
                    a = 0.0f;
                    for (int k = 0; k < ntaps; k++) {
                        const int32_t d =
                            kws_dot_i8_ref(gat + k * in_c, wo + k * in_c, blk);
                        a += (float)d * tsc[k];
                    }
                }
                yt[(size_t)ow * out_c + o] = a * wscale;
            }
        }
    }
}

void kws_conv2d_f32(float *y, const int8_t *xq, const float *xsc,
                    const float *w,
                    int frames, int in_w, int out_w,
                    int in_c, int out_c, int kT, int kW)
{
    const int pad_w = ((out_w - 1) * 2 + kW - in_w) / 2;
    const int n_in = align16(kT * kW * in_c);
    const int blk = align16(in_c);
    const int ntaps = kT * kW;
    const int win_sz = n_in;

    /* Same loop order as the int8 kernel -- frame, then channel, then mel window
     * -- so the two differ only in the weights and the reduction, which is the
     * whole point of running both in the harness. */
    for (int t = 0; t < frames; t++) {
        for (int ow = 0; ow < out_w; ow++) {
            int8_t *gat = s_gat_win + (size_t)ow * win_sz;
            float *tsc = s_tap_sc + (size_t)ow * KWS_MAX_TAPS;
            gather_window_i8(xq, xsc, t, ow, frames, in_w, in_c,
                             kT, kW, pad_w, gat, tsc);
        }

        float *yt = y + (size_t)t * out_w * out_c;
        for (int o = 0; o < out_c; o++) {
            const float *wo = w + (size_t)o * n_in;
            for (int ow = 0; ow < out_w; ow++) {
                const int8_t *gat = s_gat_win + (size_t)ow * win_sz;
                const float *tsc = s_tap_sc + (size_t)ow * KWS_MAX_TAPS;
                float a = 0.0f;
                for (int k = 0; k < ntaps; k++) {
                    float d = 0.0f;
                    for (int i = 0; i < blk; i++) {
                        d += (float)gat[k * in_c + i] * wo[k * in_c + i];
                    }
                    a += d * tsc[k];
                }
                yt[(size_t)ow * out_c + o] = a;
            }
        }
    }
}