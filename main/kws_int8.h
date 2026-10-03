/*
 * Custom INT8 kernels for the Amaze wake-word network.
 *
 * WHY HAND-WRITTEN
 * ----------------
 * There is no accelerator on the ESP32-S3, so the multiply-accumulate rate is
 * set entirely by how tightly the inner loop is written. Plain C cannot reach
 * it: on this core an int8 multiply-accumulate needs a load, a sign extension
 * (there is no sign-extending byte load) and an add, so ~4 instructions per
 * MAC -- slower than float32, which has hardware FADD.S/FMUL.S. Only the vector
 * unit's 8-bit multiply-accumulate, four products per instruction, beats that.
 * See kws_xtensa.S for the hand-written assembly that drives it, and
 * KWS_USE_XTENSA below.
 *
 * THE LAYOUT DECISION THAT MAKES ANY OF IT FAST
 * ---------------------------------------------
 * Every reduction here is `out[o] += sum_i q[i] * w[o][i]`: the input channel
 * is the reduction axis. Flax stores kernels the other way round, (i, o), which
 * makes the reduction strided by `out` and impossible to feed four MACs from
 * one 32-bit load. tools/export_weights.py therefore transposes every tensor
 * with two or more axes at export time, so both operands of the dot product are
 * contiguous. This is the single most important thing about the kernels.
 *
 * ACTIVATION QUANTISATION IS DYNAMIC AND PER ROW
 * ----------------------------------------------
 * There is no calibration set, and there should not be one: a scale fitted on
 * one room or one noise bank is wrong in the next, and the failure mode is a
 * wake word that stops firing at a level it used to fire at. Instead the
 * maximum magnitude of each row is measured on the row itself, immediately
 * before the reduction, and the dequantisation scale is derived from that. It
 * cannot drift.
 *
 * One row is the reduction unit, so the scale is shared across exactly the
 * elements that share an accumulator. The largest unit in the network is conv2's
 * 3x3x32 window = 288 samples, still comfortably tight.
 */

#ifndef KWS_INT8_H
#define KWS_INT8_H

#include <stdint.h>

/* Largest activation row any kernel in this network reduces over:
 * conv2's 3 time x 3 mel x 32 channel window. Kernels take a caller-supplied
 * scratch buffer of this size rather than a VLA, because the stack of the
 * inference task is already the smallest of the three. */
#define KWS_KERN_MAX_IN 288

/* Element stride granularity of every INT8 activation row and every weight
 * row. Both are padded out to a multiple of this and 16-byte aligned, so the
 * vector reduction loads 128 bits at a time with no shifting and no tail. Set
 * here and in kws_int8.c and tools/export_weights.py; they must agree. */
#define KWS_ALIGN16 16

/* Quantise n values to int8, returning the scale so x[i] ~= q[i] * scale.
 *
 * An all-zero row -- a fully padded convolution tap, or a silent channel --
 * returns 1.0f and an all-zero output rather than dividing by zero. Values are
 * rounded to nearest and clamped to +/-127; 128 is excluded so negation stays
 * representable and the scale stays symmetric.
 */
float kws_quant_row(const float *x, int n, int8_t *q);

/*
 * The inner reduction, and its portable twin. Both are exported so the host
 * harness and the firmware boot check can compare them over random data; if
 * they ever disagree the assembly kernel is not used.
 */
int32_t kws_dot_i8(const int8_t *a, const int8_t *b, int n);
int32_t kws_dot_i8_ref(const int8_t *a, const int8_t *b, int n);

/*
 * Which reduction the kernels will actually call, and a one-shot check that the
 * assembly reduction agrees with the portable one over random data of every
 * length the network uses. Returns 1 on agreement.
 *
 * kws_kernel_selftest() runs before the detector is armed. A kernel that
 * assembles is not a kernel that computes the right thing, and a wake word
 * detector that silently returns a wrong probability is worse than one that
 * refuses to start, so the answer is checked rather than assumed.
 */
int kws_kernel_uses_xtensa(void);
int kws_kernel_selftest(void);

/* Whether conv2's windowed reduction is the assembly one. It is self-tested and
 * demoted on its own evidence, separately from the plain dot, because a fault in
 * the tap loop costs conv2 speed and nothing else. That has to be visible: the
 * only symptom is a slower conv2, which is easy to misread as "the
 * optimisation just did not help". */
int kws_kernel_taps_using_xtensa(void);

/* Force the portable reduction. Used by the self-test and by the fallback. */
void kws_kernel_force_portable(void);

/* 1 = assembly reduction in use, 0 = portable. Read-only after init. */
int kws_kernel_using_xtensa(void);

/*
 * y[o] = bias[o] + scale * sum_i q[i] * w[o][i], one row against a whole weight
 * matrix. Both operands are contiguous over the reduction axis i.
 */
void kws_accum_row_i8(float *y, const int8_t *q, const int8_t *w,
                      const float *bias, float scale,
                      int out_dim, int in_dim);

/*
 * GEMM over rows of quantised activations:
 *   xq  [rows][in_dim] int8, transposed-dependent layout, row-major
 *   xsc [rows]        float, dequantisation scale of each row
 *   w   [out_dim][in_dim] int8 (or float32 in the *_f32 twin)
 *   y   [rows][out_dim]   float32
 */
void kws_gemm_i8(float *y, const int8_t *xq, const float *xsc,
                 const int8_t *w, const float *bias, float wscale,
                 int rows, int in_dim, int out_dim);

/*
 * 2-D convolution over (time, mel): stride 1 in time, stride 2 in mel, SAME
 * padding with ZEROS, no bias. The time axis is the ROW axis.
 *   xq  [frames*in_w][in_c] int8, ONE SCALE PER FRAME (xsc[frames])
 *   w   [kT][kW][out_c][in_c] int8, already transposed by the exporter
 *   y   [frames*out_w][out_c] float32
 *
 * The per-frame scale is what lets the window be gathered as bytes. One scale
 * per (frame, mel) row is tighter, but then each of the 490 output elements
 * needs its own 288-element requantisation, which measured as the single most
 * expensive thing in the network. Three accumulators, one per frame the kernel
 * reaches, buys the same contiguous reduction with no requantisation at all.
 */
void kws_conv2d_i8(float *y, const int8_t *xq, const float *xsc,
                   const int8_t *w, float wscale,
                   int frames, int in_w, int out_w,
                   int in_c, int out_c, int kT, int kW);

/*
 * float32 twins. Identical loop structure, float32 weights, same quantised
 * activations. They exist so the host harness can measure what the INT8 WEIGHTS
 * cost from one code path, and so the int8 kernels have something to be
 * compared against element for element.
 */
void kws_gemm_f32(float *y, const int8_t *xq, const float *xsc,
                  const float *w, const float *bias,
                  int rows, int in_dim, int out_dim);

void kws_conv2d_f32(float *y, const int8_t *xq, const float *xsc,
                    const float *w,
                    int frames, int in_w, int out_w,
                    int in_c, int out_c, int kT, int kW);

#endif /* KWS_INT8_H */