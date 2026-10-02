/*
 * bcconformer_v3 forward pass -- port of models/models.py :: BCConformerV3.
 *
 * Pipeline:
 *   delta_stack        (49,40,1) -> (49,40,3)  edge-padded d1/d2
 *   conv1  (3,3,3)->32  stride (1,2) SAME, no bias      -> (49,20,32)
 *   GroupNorm(4) + swish
 *   f_conv depthwise (1,3) over mel, no bias
 *   GroupNorm(4) + swish
 *   h = h + f_conv + mean_over_mel(f_conv)
 *   conv2  (3,3,32)->48 stride (1,2) SAME, no bias     -> (49,10,48)
 *   GroupNorm(4) + swish
 *   mel gate: 1x1 conv 48->1, softmax over mel, weighted sum -> (49,48)
 *   3 x RelConformerBlock (attention w/ relative pos bias, dw convs, SE, FFN)
 *   soft-OR pool (mean + logsumexp + max)              -> (97,)
 *   Dense(97->2) -> softmax                              -> P(keyword)
 *
 * WHY conv1 IS EVALUATED THREE TIMES
 *
 * The stem output (49,20,32) float32 is 125 KB. The task budget is 256 KB of
 * RAM and the rest of the network needs ~130 KB, so it cannot be materialised.
 *
 * GroupNorm, however, takes its statistics over EVERY frame, which normally
 * forces the whole tensor to exist. Instead conv1 is recomputed per pass:
 *
 *   pass 1  conv1 rows            -> accumulate GroupNorm-1 statistics
 *   pass 2  conv1 + GN1 + f_conv  -> accumulate GroupNorm-2 statistics
 *   pass 3  conv1 + GN1 + GN2     -> emit rows into a 3-frame ring that
 *                                     feeds conv2
 *
 * conv1 is 0.85 MMAC, so paying it twice more costs ~1.7 MMAC and saves 125 KB.
 * conv2 output (49,10,48) = 94 KB IS materialised, because recomputing it for
 * its own GroupNorm statistics would cost 6.8 MMAC -- far too much to repeat.
 */

#include "kws_model.h"
#include "kws_model_data.h"

#ifndef KWS_HOST_TEST
#include <esp_attr.h>
#endif
/* On the host there is no PSRAM section; the buffers are ordinary .bss. */
#ifndef EXT_RAM_BSS_ATTR
#define EXT_RAM_BSS_ATTR
#endif
#include <math.h>
#include <string.h>

/*
 * Bind the weight blob. The firmware points at the bin2c-generated array; the
 * host test rebinds it to a heap copy, so the identical inference code runs in
 * both places and a host pass is a real pass.
 */
#ifndef KWS_HOST_TEST
extern const float kws_weights[];
const float *kws_blob = kws_weights;
#endif
/* On the host there is no PSRAM section; the buffers are ordinary .bss. */
#ifndef EXT_RAM_BSS_ATTR
#define EXT_RAM_BSS_ATTR
#endif

/* ------------------------------------------------------------------ */
/* Activations                                                         */
/* ------------------------------------------------------------------ */

#define ROW_W 20   /* stem mel bins  */
#define ROW_H (ROW_W * KWS_C1_OUT)

#define SEQ_N   ((size_t)KWS_FRAMES * KWS_DIM)
#define FFN_N   ((size_t)KWS_FRAMES * 2 * KWS_DIM)
#define C2_N    ((size_t)KWS_FRAMES * KWS_STEM_MELS * KWS_C2_OUT)

/* All activation buffers are FLAT. A 2-D float array decays to float (*)[n]
 * and silently mismatches every float* parameter, and the model is row-major
 * throughout anyway. */

static float s_feat[3][KWS_MELS * 3];   /* delta-stack rows t-1, t, t+1 */
static float s_row[ROW_H];               /* stem row    20 x 32 */
static float s_frow[ROW_H];              /* f_conv row  20 x 32 */
EXT_RAM_BSS_ATTR static float s_hring[3][ROW_H];
/* SAME padding over time: the rows just outside the window are zeros. */
static float s_hrow_zero[ROW_H];

EXT_RAM_BSS_ATTR static float s_c2[C2_N];                /* conv2 output, 94 KB - the big one */
EXT_RAM_BSS_ATTR static float s_seq[SEQ_N];

EXT_RAM_BSS_ATTR static float s_ln[SEQ_N];
EXT_RAM_BSS_ATTR static float s_qkv[(size_t)KWS_FRAMES * 3 * KWS_DIM];
EXT_RAM_BSS_ATTR static float s_attn[SEQ_N];
EXT_RAM_BSS_ATTR static float s_blk[SEQ_N];
EXT_RAM_BSS_ATTR static float s_ffn[FFN_N];
static float s_logits[KWS_FRAMES];
static float s_pool_s[KWS_FRAMES];

#if defined(KWS_HOST_TEST) && defined(KWS_CAPTURE_ON)
/* Validation-only buffers. They are ~316 KB of .bss together, which overflows
 * internal DRAM on its own and would also blow the 256 KB RAM budget, so they
 * must never exist in a firmware build. */
volatile int    kws_debug_stage = 0;
volatile size_t kws_debug_n = 0;
float           kws_debug_buf[KWS_DEBUG_MAX];
static float    s_stem_dbg[KWS_FRAMES * ROW_H];
static float    s_delta_dbg[KWS_FRAMES * 360];

#define KWS_CAPTURE(stage, src, count)                                   \
    do {                                                                 \
        if (kws_debug_stage == (stage)) {                                \
            const size_t n_ = (size_t)(count);                           \
            if (n_ <= KWS_DEBUG_MAX) {                                   \
                memcpy(kws_debug_buf, (src), n_ * sizeof(float));        \
                kws_debug_n = n_;                                        \
            }                                                            \
        }                                                                \
    } while (0)
#else
#define KWS_CAPTURE(stage, src, count) do { } while (0)
#endif
/* On the host there is no PSRAM section; the buffers are ordinary .bss. */
#ifndef EXT_RAM_BSS_ATTR
#define EXT_RAM_BSS_ATTR
#endif

/* ------------------------------------------------------------------ */
/* Primitives                                                          */
/* ------------------------------------------------------------------ */

static inline float swishf(float x)
{
    return x / (1.0f + expf(-x));
}

/* y = x @ kernel (+ bias), kernel laid out [in][out]. */
static void dense(const float *x, const float *kernel, const float *bias,
                  int rows, int in_dim, int out_dim, float *y)
{
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * in_dim;
        float *yr = y + (size_t)r * out_dim;
        for (int o = 0; o < out_dim; o++) {
            yr[o] = (bias != NULL) ? bias[o] : 0.0f;
        }
        for (int i = 0; i < in_dim; i++) {
            const float v = xr[i];
            const float *k = kernel + (size_t)i * out_dim;
            for (int o = 0; o < out_dim; o++) {
                yr[o] += v * k[o];
            }
        }
    }
}

/* Flax LayerNorm over the last axis, epsilon 1e-6. */
static void layer_norm(const float *x, const float *scale, const float *bias,
                       int rows, int dim, float *y)
{
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * dim;
        float *yr = y + (size_t)r * dim;

        float mean = 0.0f;
        for (int i = 0; i < dim; i++) { mean += xr[i]; }
        mean /= (float)dim;

        float var = 0.0f;
        for (int i = 0; i < dim; i++) {
            const float d = xr[i] - mean;
            var += d * d;
        }
        var /= (float)dim;

        const float inv = 1.0f / sqrtf(var + 1e-6f);
        for (int i = 0; i < dim; i++) {
            yr[i] = (xr[i] - mean) * inv * scale[i] + bias[i];
        }
    }
}

/*
 * GroupNorm with 4 groups over the last axis.
 *
 * Flax computes the statistics per (sample, group) across every OTHER axis, so
 * for a (49,20,32) tensor it averages over 49*20*8 elements per group. That
 * coupling is why the stem cannot be streamed a row at a time. Accumulation is
 * kept separate from application so a tensor can be normalised in place.
 */
static float s_gn_sum[4];
static float s_gn_sumsq[4];

static void gn_reset(void)
{
    for (int g = 0; g < 4; g++) { s_gn_sum[g] = 0.0f; s_gn_sumsq[g] = 0.0f; }
}

static void gn_accum(const float *x, int rows, int dim)
{
    const int per = dim / 4;
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * dim;
        for (int g = 0; g < 4; g++) {
            float a = 0.0f, b = 0.0f;
            for (int j = 0; j < per; j++) {
                const float v = xr[g * per + j];
                a += v;
                b += v * v;
            }
            s_gn_sum[g] += a;
            s_gn_sumsq[g] += b;
        }
    }
}

static void gn_finalize(int total_elements_per_group)
{
    const float n = (float)total_elements_per_group;
    for (int g = 0; g < 4; g++) {
        const float mean = s_gn_sum[g] / n;
        /* Keep the mean/variance split so gn_apply can use it per row without
         * recomputing statistics. */
        s_gn_sum[g] = mean;
        s_gn_sumsq[g] = s_gn_sumsq[g] / n - mean * mean;
    }
}

static void gn_apply(float *x, int rows, int dim,
                     const float *scale, const float *bias, float eps)
{
    const int per = dim / 4;
    for (int r = 0; r < rows; r++) {
        float *xr = x + (size_t)r * dim;
        for (int g = 0; g < 4; g++) {
            const float inv = 1.0f / sqrtf(s_gn_sumsq[g] + eps);
            const float mean = s_gn_sum[g];
            for (int j = 0; j < per; j++) {
                const int c = g * per + j;
                xr[c] = (xr[c] - mean) * inv * scale[c] + bias[c];
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Stem                                                                */
/* ------------------------------------------------------------------ */

/*
 * Fill the three delta-stack rows conv1 needs, plus the first and second
 * TIME derivatives of the middle one.
 *
 * The derivatives are along TIME, not mel. models/models.py::_delta_stack does
 *
 *     t     = x[:, :, :, 0]                    # (B, T, M)
 *     d1_pad = jnp.pad(t, ((0,0),(1,1),(0,0)), mode="edge")
 *     d1    = 0.5 * (d1_pad[:, 2:, :] - d1_pad[:, :-2, :])
 *
 * and that pad spec's middle axis is T (not the mel axis), so the difference is
 * over frames. Computing it over mel instead shifts the model onto features it
 * was never trained on, and the error is invisible until the logits disagree.
 *
 * d1[t] needs rows t-1..t+1; d2[t] needs rows t-2..t+2, so five rows are
 * gathered with edge clamping (jnp.pad mode="edge").
 */
static inline float spec_at(const float *spec, int frame, int m)
{
    if (frame < 0) { frame = 0; }
    if (frame > KWS_FRAMES - 1) { frame = KWS_FRAMES - 1; }
    return spec[(size_t)frame * KWS_MELS + m];
}

static inline int clamp_frame(int f)
{
    if (f < 0) { return 0; }
    if (f > KWS_FRAMES - 1) { return KWS_FRAMES - 1; }
    return f;
}

/* d1 at frame f, with the FRAME clamped before differencing. That clamp must
 * happen here rather than inside spec_at: numpy pads the d1 array itself with
 * mode="edge", so d1[-1] is d1[0], not a difference of two clamped raw frames
 * (which would evaluate to zero). */
static inline float d1_at(const float *spec, int f, int m)
{
    f = clamp_frame(f);
    return 0.5f * (spec_at(spec, f + 1, m) - spec_at(spec, f - 1, m));
}

/*
 * Build the three delta-stack rows conv1 consumes at output frame t, i.e.
 * frames t-1, t and t+1.
 *
 * Two details that are easy to get wrong and were both wrong here:
 *
 * 1. The derivatives run along TIME, not mel. models/models.py::_delta_stack
 *    pads axis 1 of a (B, T, M) array, which is the frame axis. Differencing
 *    over mel instead shifts the model onto features it never saw.
 *
 * 2. conv1 uses SAME padding, which pads with ZEROS, not by replicating the
 *    edge. So for t = 0 the t-1 row and for t = the last frame the t+1 row must
 *    be all zeros. Edge-replicating them (as the delta stack itself does
 *    internally) feeds conv1 a non-zero phantom frame at both boundaries.
 *
 * d2 needs d1 at f+1 and f-1, which in turn needs frames f-2 .. f+2. Indexing
 * the spec with clamping expresses that directly and avoids the off-by-one that
 * a fixed-size row gather produced.
 */
static void build_feat(const float *spec, int t)
{
    for (int i = 0; i < 3; i++) {
        const int f = t - 1 + i;
        float *dst = s_feat[i];

        if (f < 0 || f > KWS_FRAMES - 1) {
            memset(dst, 0, sizeof(s_feat[i]));    /* SAME padding: zeros */
            continue;
        }

        for (int m = 0; m < KWS_MELS; m++) {
            const float v  = spec_at(spec, f, m);
            const float d1 = d1_at(spec, f, m);
            const float d2 = 0.5f * (d1_at(spec, f + 1, m) -
                                     d1_at(spec, f - 1, m));
            dst[(size_t)m * 3 + 0] = v;
            dst[(size_t)m * 3 + 1] = d1;
            dst[(size_t)m * 3 + 2] = d2;
        }
    }
}

/*
 * conv1 for one time row: 3 rows of (40 mel x 3 chan) -> (20 mel x 32 chan).
 * Kernel is (kT=3, kF=3, in=3, out=32), stride (1,2) over mel, SAME, no bias.
 *
 * JAX "SAME": pad_total = (out-1)*stride + k - in, then pad_lo = pad_total // 2
 * (FLOOR) and pad_hi = pad_total - pad_lo. Using ceil here silently shifts the
 * whole feature by one mel bin.
 * For mel: (20-1)*2 + 3 - 40 = 1, so pad_lo = 0 and one sample of padding on
 * the right.
 *
 * Time needs kT/2 = 1 sample of padding either side, which build_feat supplies
 * by clamping to the frame edges.
 */
#ifdef KWS_HOST_TEST
void kws_debug_build_feat(const float *spec, int t, float *out)
{
    build_feat(spec, t);
    memcpy(out, s_feat, sizeof(s_feat));
}
#endif
/* On the host there is no PSRAM section; the buffers are ordinary .bss. */
#ifndef EXT_RAM_BSS_ATTR
#define EXT_RAM_BSS_ATTR
#endif

static void conv1_row(float *out);

#ifdef KWS_HOST_TEST
void kws_debug_addrs(const char **names, const void **ptrs, int n)
{
    const void *tab[] = { s_feat, s_row, s_frow, s_hring,
#if defined(KWS_HOST_TEST) && defined(KWS_CAPTURE_ON)
                          s_stem_dbg, s_delta_dbg,
#endif
                          s_c2, s_seq, s_ln, s_qkv, s_attn,
                          s_blk, s_ffn,
#if defined(KWS_HOST_TEST) && defined(KWS_CAPTURE_ON)
                          kws_debug_buf
#else
                          NULL
#endif
                          };
    const char *nm[] = { "s_feat", "s_row", "s_frow", "s_hring", "s_stem_dbg",
                         "s_delta_dbg", "s_c2", "s_seq", "s_ln", "s_qkv",
                         "s_attn", "s_blk", "s_ffn", "kws_debug_buf" };
    const int lim = (n < 14) ? n : 14;
    for (int i = 0; i < lim; i++) {
        names[i] = nm[i];
        ptrs[i] = tab[i];
    }
}
#endif
/* On the host there is no PSRAM section; the buffers are ordinary .bss. */
#ifndef EXT_RAM_BSS_ATTR
#define EXT_RAM_BSS_ATTR
#endif

#ifdef KWS_HOST_TEST
void kws_debug_conv1(const float *spec, int t, float *out)
{
    build_feat(spec, t);
    conv1_row(out);
}
#endif
/* On the host there is no PSRAM section; the buffers are ordinary .bss. */
#ifndef EXT_RAM_BSS_ATTR
#define EXT_RAM_BSS_ATTR
#endif

static void conv1_row(float *out)
{
    const float *k = W(KWS_T_CONV1);
    const int out_w = ROW_W;                       /* 40 mel / stride 2 = 20 */
    const int pad_total = (out_w - 1) * 2 + 3 - KWS_MELS;
    const int pad_m = pad_total / 2;

    memset(out, 0, sizeof(float) * (size_t)out_w * KWS_C1_OUT);

    for (int ow = 0; ow < out_w; ow++) {
        for (int oc = 0; oc < KWS_C1_OUT; oc++) {
            float acc = 0.0f;
            for (int kt = 0; kt < 3; kt++) {
                const float *fr = s_feat[kt];
                for (int kw = 0; kw < 3; kw++) {
                    const int iw = ow * 2 + kw - pad_m;
                    if (iw < 0 || iw >= KWS_MELS) { continue; }
                    for (int ic = 0; ic < 3; ic++) {
                        const float xv = fr[(size_t)iw * 3 + ic];
                        acc += xv * k[(((size_t)kt * 3 + kw) * 3 + ic) * KWS_C1_OUT + oc];
                    }
                }
            }
            out[(size_t)ow * KWS_C1_OUT + oc] = acc;
        }
    }
}

/* Depthwise conv over mel, SAME, no bias.
 * Flax stores a grouped conv kernel as (kT, kF, in/groups, out/groups), so
 * feature_group_count == channels leaves the middle axis 1 wide: the checkpoint
 * shape is (1,3,1,32) and the weight index is simply kw * C + c. */
static void fconv_row(const float *in, float *out)
{
    const float *k = W(KWS_T_FCONV);       /* (1, 3, 1, 32) */
    const int pad = (3 - 1) / 2;

    for (int iw = 0; iw < ROW_W; iw++) {
        for (int c = 0; c < KWS_C1_OUT; c++) {
            float acc = 0.0f;
            for (int kw = 0; kw < 3; kw++) {
                const int jw = iw + kw - pad;
                if (jw < 0 || jw >= ROW_W) { continue; }
                acc += in[(size_t)jw * KWS_C1_OUT + c] * k[(size_t)kw * KWS_C1_OUT + c];
            }
            out[(size_t)iw * KWS_C1_OUT + c] = acc;
        }
    }
}

/* conv2 for one time row: 3 stem rows (20x32) -> (10 mel x 48). */
static void conv2_row(const float *r0, const float *r1, const float *r2,
                      float *out)
{
    const float *k = W(KWS_T_CONV2);      /* (3,3,32,48) */
    const int out_w = KWS_STEM_MELS;
    const int pad_total = (out_w - 1) * 2 + 3 - ROW_W;
    const int pad_m = pad_total / 2;       /* floor, per JAX SAME */

    memset(out, 0, sizeof(float) * (size_t)out_w * KWS_C2_OUT);

    for (int ow = 0; ow < out_w; ow++) {
        for (int oc = 0; oc < KWS_C2_OUT; oc++) {
            float acc = 0.0f;
            for (int kt = 0; kt < 3; kt++) {
                const float *row = (kt == 0) ? r0 : ((kt == 1) ? r1 : r2);
                for (int kw = 0; kw < 3; kw++) {
                    const int iw = ow * 2 + kw - pad_m;
                    if (iw < 0 || iw >= ROW_W) { continue; }
                    for (int ic = 0; ic < KWS_C1_OUT; ic++) {
                        const float xv = row[(size_t)iw * KWS_C1_OUT + ic];
                        acc += xv *
                               k[(((size_t)kt * 3 + kw) * KWS_C1_OUT + ic) * KWS_C2_OUT + oc];
                    }
                }
            }
            out[(size_t)ow * KWS_C2_OUT + oc] = acc;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Attention                                                           */
/* ------------------------------------------------------------------ */

static void rel_attention(const float *x, const float *qkv_k, const float *qkv_b,
                          const float *bias, const float *proj_k, const float *proj_b,
                          float *out)
{
    const int T = KWS_FRAMES, H = KWS_HEADS, D = KWS_DIM, dh = KWS_HEAD_DIM;
    const float inv = 1.0f / sqrtf((float)dh);

    dense(x, qkv_k, qkv_b, T, D, 3 * D, s_qkv);

    const float *qb = s_qkv;
    const float *kb = s_qkv + D;
    const float *vb = s_qkv + 2 * D;

    memset(s_attn, 0, sizeof(s_attn));

    for (int h = 0; h < H; h++) {
        for (int i = 0; i < T; i++) {
            float *row = s_logits;

            for (int j = 0; j < T; j++) {
                float acc = 0.0f;
                for (int d = 0; d < dh; d++) {
                    acc += qb[(size_t)i * D + h * dh + d] * kb[(size_t)j * D + h * dh + d];
                }
                acc = acc * inv + bias[(size_t)(i - j + (T - 1)) * H + h];
                row[j] = acc;
            }

            float mx = row[0];
            for (int j = 1; j < T; j++) { if (row[j] > mx) { mx = row[j]; } }
            float sum = 0.0f;
            for (int j = 0; j < T; j++) { row[j] = expf(row[j] - mx); sum += row[j]; }
            const float isum = 1.0f / sum;
            for (int j = 0; j < T; j++) { row[j] *= isum; }

            for (int d = 0; d < dh; d++) {
                float acc = 0.0f;
                for (int j = 0; j < T; j++) {
                    acc += row[j] * vb[(size_t)j * D + h * dh + d];
                }
                s_attn[(size_t)i * D + h * dh + d] = acc;
            }
        }
    }

    dense(s_attn, proj_k, proj_b, T, D, D, out);
}

/* Depthwise conv over TIME, per channel. kernel [K][D], SAME, no bias. */
static void dwconv_t(const float *in, const float *k, int K, float *out)
{
    const int T = KWS_FRAMES, D = KWS_DIM;
    const int pad = (K - 1) / 2;

    for (int t = 0; t < T; t++) {
        for (int c = 0; c < D; c++) {
            float acc = 0.0f;
            for (int j = 0; j < K; j++) {
                const int tt = t + j - pad;
                if (tt < 0 || tt >= T) { continue; }
                acc += in[(size_t)tt * D + c] * k[(size_t)j * D + c];
            }
            out[(size_t)t * D + c] = acc;
        }
    }
}

static void pointwise(const float *in, const float *k, float *out)
{
    const int T = KWS_FRAMES, D = KWS_DIM;
    for (int t = 0; t < T; t++) {
        const float *xr = in + (size_t)t * D;
        float *yr = out + (size_t)t * D;
        for (int o = 0; o < D; o++) {
            float acc = 0.0f;
            for (int c = 0; c < D; c++) { acc += xr[c] * k[(size_t)c * D + o]; }
            yr[o] = acc;
        }
    }
}

/* ------------------------------------------------------------------ */
/* One RelConformerBlock, in place on x                                */
/* ------------------------------------------------------------------ */

static void conformer_block(float *x, int b)
{
    const int T = KWS_FRAMES, D = KWS_DIM;
    const int base = KWS_T_BLOCKS + b * KWS_BLOCK_TENSORS;

    layer_norm(x, W(base + 0), W(base + 1), T, D, s_ln);
    rel_attention(s_ln, W(base + 2), W(base + 3), W(base + 4),
                  W(base + 5), W(base + 6), s_attn);
    for (int i = 0; i < T * D; i++) { x[i] += s_attn[i]; }

    layer_norm(x, W(base + 7), W(base + 8), T, D, s_ln);
    dwconv_t(s_ln, W(base + 9), 3, s_blk);
    dwconv_t(s_ln, W(base + 10), 7, s_attn);
    for (int i = 0; i < T * D; i++) { s_blk[i] = swishf(s_blk[i]) + swishf(s_attn[i]); }

    pointwise(s_blk, W(base + 11), s_attn);

    float z[KWS_DIM];
    for (int c = 0; c < D; c++) { z[c] = 0.0f; }
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < D; c++) { z[c] += s_attn[(size_t)t * D + c]; }
    }
    for (int c = 0; c < D; c++) { z[c] /= (float)T; }

    float zr[KWS_SE];
    dense(z, W(base + 12), NULL, 1, D, KWS_SE, zr);
    for (int i = 0; i < KWS_SE; i++) { zr[i] = swishf(zr[i]); }
    dense(zr, W(base + 13), NULL, 1, KWS_SE, D, z);

    for (int i = 0; i < T * D; i++) {
        x[i] += s_attn[i] / (1.0f + expf(-z[i % D]));
    }

    layer_norm(x, W(base + 14), W(base + 15), T, D, s_ln);
    dense(s_ln, W(base + 16), W(base + 17), T, D, 2 * D, s_ffn);
    for (int i = 0; i < T * 2 * D; i++) { s_ffn[i] = swishf(s_ffn[i]); }
    dense(s_ffn, W(base + 18), W(base + 19), T, 2 * D, D, s_attn);
    for (int i = 0; i < T * D; i++) { x[i] += s_attn[i]; }
}

/* ------------------------------------------------------------------ */
/* Soft-OR pooling + classifier                                        */
/* ------------------------------------------------------------------ */

static void soft_or_pool(const float *x, float *out)
{
    const int T = KWS_FRAMES, D = KWS_DIM, H = KWS_POOL_HIDDEN;
    const float *proj_k = W(KWS_T_POOL_PROJ);
    const float *proj_b = W(KWS_T_POOL_PROJ + 1);
    const float *w = W(KWS_T_OR_W);
    const float b = *W(KWS_T_OR_B);

    float sum[D], mx[D];
    for (int c = 0; c < D; c++) { sum[c] = 0.0f; mx[c] = -1e30f; }
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < D; c++) {
            const float v = x[(size_t)t * D + c];
            sum[c] += v;
            if (v > mx[c]) { mx[c] = v; }
        }
    }
    for (int c = 0; c < D; c++) { sum[c] /= (float)T; }

    dense(x, proj_k, proj_b, T, D, H, s_ffn);

    float mxs = -1e30f;
    for (int t = 0; t < T; t++) {
        float s = b;
        for (int h = 0; h < H; h++) { s += s_ffn[(size_t)t * H + h] * w[h]; }
        s_pool_s[t] = s;
        if (s > mxs) { mxs = s; }
    }
    float lse = 0.0f;
    for (int t = 0; t < T; t++) { lse += expf(s_pool_s[t] - mxs); }
    lse = mxs + logf(lse) - logf((float)T);

    for (int c = 0; c < D; c++) { out[c] = sum[c]; }
    out[D] = lse;
    for (int c = 0; c < D; c++) { out[D + 1 + c] = mx[c]; }
}

/* ------------------------------------------------------------------ */
/* Entry point                                                         */
/* ------------------------------------------------------------------ */

void kws_model_init(void)
{
    /* Zero every activation buffer, not just the ones a previous call was
     * thought to have written. Reading a buffer that some early-exit path never
     * touched makes the result depend on whatever was in that RAM, which
     * presented as a model that returned a different probability on every
     * call. Clearing everything makes the run deterministic, which is also
     * what makes it debuggable. */
    memset(s_feat, 0, sizeof(s_feat));
    memset(s_row, 0, sizeof(s_row));
    memset(s_frow, 0, sizeof(s_frow));
    memset(s_hring, 0, sizeof(s_hring));
    memset(s_hrow_zero, 0, sizeof(s_hrow_zero));
    memset(s_c2, 0, sizeof(s_c2));
    memset(s_seq, 0, sizeof(s_seq));
    memset(s_ln, 0, sizeof(s_ln));
    memset(s_qkv, 0, sizeof(s_qkv));
    memset(s_attn, 0, sizeof(s_attn));
    memset(s_blk, 0, sizeof(s_blk));
    memset(s_ffn, 0, sizeof(s_ffn));
    memset(s_logits, 0, sizeof(s_logits));
    memset(s_pool_s, 0, sizeof(s_pool_s));
    gn_reset();
}

float kws_model_run(const float *spec)
{
    /* --- pass 1: GroupNorm-1 statistics over conv1 output --------------- */
    gn_reset();
    for (int t = 0; t < KWS_FRAMES; t++) {
        build_feat(spec, t);
#if defined(KWS_HOST_TEST) && defined(KWS_CAPTURE_ON)
        /* stage 20: s_feat at the exact call site, no accumulation in between */
        if (t == 0) {
            KWS_CAPTURE(20, s_feat, 360);
            KWS_CAPTURE(22, spec, 8);   /* what spec[0..7] holds right here */
            KWS_CAPTURE(23, (const float *)(const void *)&spec, 2);  /* the pointer itself */
        }
        /* stage 21: s_feat again, immediately after the memcpy */
        KWS_CAPTURE(21, s_feat, 360);
#endif
        conv1_row(s_row);
#if defined(KWS_HOST_TEST) && defined(KWS_CAPTURE_ON)
        memcpy(&s_stem_dbg[(size_t)t * ROW_H], s_row, sizeof(s_row));
#endif
        gn_accum(s_row, ROW_W, KWS_C1_OUT);
    }
#if defined(KWS_HOST_TEST) && defined(KWS_CAPTURE_ON)
    KWS_CAPTURE(14, s_delta_dbg, KWS_FRAMES * 360);   /* delta stack */
    KWS_CAPTURE(10, s_stem_dbg, KWS_FRAMES * ROW_H);  /* conv1 output */
#endif
    gn_finalize(KWS_FRAMES * ROW_W * (KWS_C1_OUT / 4));

    /* --- pass 2: GroupNorm-2 statistics over f_conv output -------------- */
    gn_reset();
    for (int t = 0; t < KWS_FRAMES; t++) {
        build_feat(spec, t);
        conv1_row(s_row);
        gn_apply(s_row, ROW_W, KWS_C1_OUT, W(KWS_T_GN1_SCALE), W(KWS_T_GN1_BIAS), 1e-5f);
        fconv_row(s_row, s_frow);
        gn_accum(s_frow, ROW_W, KWS_C1_OUT);
    }
    gn_finalize(KWS_FRAMES * ROW_W * (KWS_C1_OUT / 4));

    /* --- pass 3: emit stem rows, drive conv2 through the 3-frame ring ---- */
    for (int t = 0; t < KWS_FRAMES; t++) {
        build_feat(spec, t);
        conv1_row(s_row);
        gn_apply(s_row, ROW_W, KWS_C1_OUT, W(KWS_T_GN1_SCALE), W(KWS_T_GN1_BIAS), 1e-5f);
        fconv_row(s_row, s_frow);
        gn_apply(s_frow, ROW_W, KWS_C1_OUT, W(KWS_T_GN2_SCALE), W(KWS_T_GN2_BIAS), 1e-5f);

        for (int c = 0; c < KWS_C1_OUT; c++) {
            float m = 0.0f;
            for (int iw = 0; iw < ROW_W; iw++) { m += s_frow[(size_t)iw * KWS_C1_OUT + c]; }
            m /= (float)ROW_W;
            for (int iw = 0; iw < ROW_W; iw++) {
                s_frow[(size_t)iw * KWS_C1_OUT + c] += s_row[(size_t)iw * KWS_C1_OUT + c] + m;
            }
        }

        memcpy(s_hring[t % 3], s_frow, sizeof(s_frow));

        /*
         * conv2's output row o consumes stem rows o-1, o, o+1. After storing
         * frame t the complete outputs are o = t-1 (needing t-2, t-1, t).
         *
         * Two edges need care and both were wrong before:
         *
         *  - o = 0 needs stem row -1, which SAME padding supplies as ZEROS.
         *    Reading a never-written ring slot here is what made the model
         *    return a different probability on every call.
         *  - o = 48 needs stem row 49, also zeros, and can only be completed
         *    after the loop. Skipping it left the last output row uninitialised
         *    for the whole network downstream.
         */
        if (t >= 1) {
            const float *r0 = (t >= 2) ? s_hring[(t + 1) % 3] : s_hrow_zero;
            const float *r1 = s_hring[(t + 2) % 3];      /* t-1 */
            const float *r2 = s_hring[t % 3];            /* t   */
            conv2_row(r0, r1, r2,
                      s_c2 + (size_t)(t - 1) * KWS_STEM_MELS * KWS_C2_OUT);
        }
    }

    /* Final output row: stem rows 47, 48 and a zero-padded 49. The ring still
     * holds those two because 47 % 3 == 2 and 48 % 3 == 0. */
    conv2_row(s_hring[(KWS_FRAMES - 2) % 3],
              s_hring[(KWS_FRAMES - 1) % 3],
              s_hrow_zero,
              s_c2 + (size_t)(KWS_FRAMES - 1) * KWS_STEM_MELS * KWS_C2_OUT);

    /* --- GroupNorm-3 + swish over conv2 output, in place ---------------- */
    gn_reset();
    gn_accum(s_c2, KWS_FRAMES * KWS_STEM_MELS, KWS_C2_OUT);
    gn_finalize(KWS_FRAMES * KWS_STEM_MELS * (KWS_C2_OUT / 4));
    gn_apply(s_c2, KWS_FRAMES * KWS_STEM_MELS, KWS_C2_OUT,
             W(KWS_T_GN3_SCALE), W(KWS_T_GN3_BIAS), 1e-5f);
    for (int i = 0; i < KWS_FRAMES * KWS_STEM_MELS * KWS_C2_OUT; i++) {
        s_c2[i] = swishf(s_c2[i]);
    }


    /* --- mel gate ------------------------------------------------------- */
    {
        const float *mg = W(KWS_T_MEL_GATE);      /* (48,) */
        for (int t = 0; t < KWS_FRAMES; t++) {
            float w[KWS_STEM_MELS];
            float mx = -1e30f;
            for (int iw = 0; iw < KWS_STEM_MELS; iw++) {
                float acc = 0.0f;
                for (int c = 0; c < KWS_C2_OUT; c++) { acc += s_c2[((size_t)t * KWS_STEM_MELS + iw) * KWS_C2_OUT + c] * mg[c]; }
                w[iw] = acc;
                if (acc > mx) { mx = acc; }
            }
            float sum = 0.0f;
            for (int iw = 0; iw < KWS_STEM_MELS; iw++) { w[iw] = expf(w[iw] - mx); sum += w[iw]; }
            for (int iw = 0; iw < KWS_STEM_MELS; iw++) { w[iw] /= sum; }

            for (int c = 0; c < KWS_C2_OUT; c++) { s_seq[(size_t)t * KWS_DIM + c] = 0.0f; }
            for (int iw = 0; iw < KWS_STEM_MELS; iw++) {
                const float ww = w[iw];
                for (int c = 0; c < KWS_C2_OUT; c++) { s_seq[(size_t)t * KWS_DIM + c] += s_c2[((size_t)t * KWS_STEM_MELS + iw) * KWS_C2_OUT + c] * ww; }
            }
        }
    }


    /* --- conformer stack ------------------------------------------------ */
    for (int b = 0; b < KWS_BLOCKS; b++) {
        conformer_block(s_seq, b);
        KWS_CAPTURE(5 + b, s_seq, SEQ_N);
    }

    /* --- soft-OR pool + classifier --------------------------------------- */
    float feat[KWS_FEAT];
    soft_or_pool(s_seq, feat);

    float logits[2];
    dense(feat, W(KWS_T_HEAD_K), W(KWS_T_HEAD_B), 1, KWS_FEAT, 2, logits);

    const float mx = (logits[0] > logits[1]) ? logits[0] : logits[1];
    const float e0 = expf(logits[0] - mx);
    const float e1 = expf(logits[1] - mx);
    return e1 / (e0 + e1);
}
