/*
 * bcconformer_v3 forward pass -- port of models/models.py :: BCConformerV3.
 *
 * PIPELINE
 * --------
 *   delta stack       (49,40,1) -> (49,40,3)  edge-padded d1/d2, TIME axis
 *   conv1   (3,3,3)->32   stride (1,2) SAME, no bias            float weights
 *   GroupNorm(4) + swish
 *   f_conv  depthwise (1,3) over mel, no bias                  float weights
 *   GroupNorm(4) + swish
 *   h = h + f_conv + mean_over_mel(f_conv)
 *   conv2   (3,3,32)->48  stride (1,2) SAME, no bias            INT8
 *   GroupNorm(4) + swish
 *   mel gate: 1x1 conv 48->1, softmax over mel, weighted sum -> (49,48)
 *   3 x RelConformerBlock                                      INT8 Dense
 *   soft-OR pool (mean + logsumexp + max)              -> (97,)
 *   Dense(97->2) -> softmax                             -> P(keyword)
 *
 * WHY THE STEM IS MATERIALISED AND THE OLD RING IS GONE
 * -----------------------------------------------------
 * The previous version computed conv1 three times, feeding conv2 through a
 * three-frame rolling ring so that the (49,20,32) float32 activation -- 125 KB
 * -- never had to exist. That structure could not represent the two boundary
 * cases conv2 actually needs (stem row -1 and stem row 49, both zero-padded),
 * and reading a ring slot that had not been written yet was the class of bug
 * that made this port disagree with the training reference.
 *
 * In INT8 the same tensor is 31,360 bytes, so materialising it costs nothing
 * that matters and deletes the ring, both edge cases and the whole failure
 * mode. GroupNorm takes its statistics over every frame, so materialising is
 * also what lets the stem be three straight passes instead of two extra
 * evaluations of conv1.
 *
 * WHAT IS INT8 AND WHY
 * --------------------
 * conv2 is 60% of the network's 11.2 MMAC and the conformer Dense layers are
 * another 25%, so those two are where the CPU budget is decided. They are the
 * ones the hardware can actually do fast: see kws_int8.h for why plain C
 * int8 cannot, and why the weights are transposed at export time.
 *
 * Everything else stays float32 on purpose:
 *
 *   conv1, f_conv, the depthwise convolutions   7.6% of the MACs, depthwise or
 *                                                tiny. int8 buys nothing worth
 *                                                the requantisation plumbing,
 *                                                and float keeps conv1 exact so
 *                                                the stem is a clean reference.
 *   LayerNorm, GroupNorm, the attention softmax  these decide the conditioning
 *                                                of everything downstream. A
 *                                                quantised softmax error is a
 *                                                directly proportional error in
 *                                                which frames the network
 *                                                listens to.
 *   the attention Q.K^T and A.V products         6% of the MACs, and the softmax
 *                                                sits between them.
 */

#include "kws_model.h"
#include "kws_model_data.h"
#include "kws_int8.h"

#ifndef KWS_HOST_TEST
#include <esp_attr.h>
#include <esp_timer.h>
#endif
#include <math.h>
#include <stdint.h>
#include <string.h>
#ifdef KWS_HOST_TEST
#include <time.h>
#endif

/* EXT_RAM_BSS_ATTR expands to nothing unless
 * CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY is set, and it is what keeps the
 * activation set out of internal DRAM. It does not exist on the host. */
#ifndef EXT_RAM_BSS_ATTR
#define EXT_RAM_BSS_ATTR
#endif

/*
 * Stage tracing. Off by default because fprintf costs more than the whole
 * stem; defined in the host build only. The failure it exists for is the one
 * where the output stops depending on the input -- every fixture then returns
 * the same probability, which looks like a bad threshold rather than a stage
 * that silently produced zeros.
 */
#ifdef KWS_MODEL_TRACE
#include <stdio.h>
#define TRACE(...) fprintf(stderr, __VA_ARGS__)
#define TR_N 10
static double s_tr[TR_N];
static long   s_trn[TR_N];

static void tr_acc(int slot, const float *x, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        s_tr[slot] += (double)x[i] * (double)x[i];
    }
    s_trn[slot] += (long)n;
}

static double tr_rms(int slot)
{
    return s_trn[slot] ? sqrt(s_tr[slot] / (double)s_trn[slot]) : 0.0;
}

static void tr_dump(const char *const *names)
{
    for (int i = 0; i < TR_N; i++) {
        fprintf(stderr, "  %-10s rms %12.5f  n %8ld\n", names[i],
                tr_rms(i), s_trn[i]);
    }
    for (int i = 0; i < TR_N; i++) { s_tr[i] = 0.0; s_trn[i] = 0; }
}

/* Write one tensor for tools/compare_stage_tensor.py. */
static void tr_save(const char *path, const float *x, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "  cannot write %s\n", path); return; }
    fwrite(x, sizeof(float), n, f);
    fclose(f);
    fprintf(stderr, "  saved %s (%zu floats)\n", path, n);
}

static float trace_rms(const float *x, size_t n)
{
    double acc = 0.0;
    for (size_t i = 0; i < n; i++) { acc += (double)x[i] * (double)x[i]; }
    return (float)sqrt(acc / (double)n);
}
#else
#define TRACE(...) do { } while (0)
#define tr_acc(...) do { } while (0)
#define tr_rms(...)  (0.0)
#define tr_dump(...) do { } while (0)
#define tr_save(...) do { } while (0)
#define trace_rms(x, n) (0.0f)
#endif

/*
 * Bind the weight blobs. The firmware points at the bin2c-generated arrays; the
 * host harness rebinds them to heap copies, so the identical inference code
 * runs in both places and a host pass is a real pass.
 */
#ifndef KWS_HOST_TEST
extern const int8_t kws_weights_i8[];
const int8_t *kws_blob_i8 = kws_weights_i8;
const float  *kws_blob_f32 = NULL;   /* the float32 twin is not built into the
                                      * firmware; there is nothing for it to
                                      * point at and 309 KB of flash it does not
                                      * need. */
#endif

/* ------------------------------------------------------------------ */
/* Tensor access                                                        */
/* ------------------------------------------------------------------ */

/*
 * Float32 view of a tensor. Tensors the exporter kept in float32 live in the
 * generated kws_param_f32; everything else is in the host-only float32 blob.
 * Both hold the identical values in the identical reversed layout, which is
 * what makes the float-weight mode an A/B rather than a second model.
 */
static inline const float *WF(int i)
{
    const kws_tensor_meta *m = &kws_tensors[i];
    return (m->offset == 0xFFFFu) ? (kws_param_f32 + kws_param_off[i])
                                  : (kws_blob_f32 + m->foff);
}

/* INT8 view. Never called for a tensor the exporter kept in float32. */
static inline const int8_t *WI8(int i)
{
    return kws_blob_i8 + kws_tensors[i].offset;
}

static inline float WS(int i)
{
    return kws_wscale[i];
}

/* ------------------------------------------------------------------ */
/* Shapes                                                              */
/* ------------------------------------------------------------------ */

#define ROW_W    20                          /* stem mel bins            */
#define C1_ROW   (ROW_W * KWS_C1_OUT)        /* 640                      */
#define C2_ROW   (KWS_STEM_MELS * KWS_C2_OUT)/* 480                      */
#define C1_ROWS  ((size_t)KWS_FRAMES * ROW_W)
#define C2_ROWS  ((size_t)KWS_FRAMES * KWS_STEM_MELS)
#define SEQ_N    ((size_t)KWS_FRAMES * KWS_DIM)
#define QKV_N    ((size_t)KWS_FRAMES * 3 * KWS_DIM)
#define FFN_N    ((size_t)KWS_FRAMES * 2 * KWS_DIM)

/* Flax's GroupNorm and LayerNorm both default to epsilon 1e-6. Using 1e-5
 * here is a silent change to the normalisation of every tensor in the network
 * and shows up only as a probability that is a few times too high. */
#define KWS_LN_EPS 1e-6f

/* ------------------------------------------------------------------ */
/* Activations                                                         */
/* ------------------------------------------------------------------ */

/*
 * Buffer placement is chosen by MEASUREMENT, not by size.
 *
 * These tensors are in internal DIRAM, not PSRAM, and the reason is a number
 * from the hardware: the mel gate reads 188 KB of conv2 output in 3.1 ms,
 * which is about 60 MB/s -- the ceiling this part's PSRAM delivers when a pass
 * streams it. Every tensor below is walked by several stages per inference, so
 * a tensor that sits in PSRAM is paid at that ceiling every time it is touched.
 *
 * DIRAM is 341 KB total and the budget is under 50%, so this placement is the
 * reason to watch that number if anything else grows. There is no fallback
 * path for any of it: a buffer that cannot fit is a buffer that has to be
 * re-cut, not one that silently moves.
 */

/* Stem. Four int8 tensors of the same shape, each with one scale per
 * (frame, mel) row of 32 values:
 *   c1  raw conv1 output          31,360 B
 *   h1  after GroupNorm-1+swish   31,360 B
 *   f   raw depthwise mel conv    31,360 B
 *   st  the tensor conv2 reads    31,360 B
 * 125 KB in total, against 250 KB for the same four in float32, and the RAM
 * saving is not the point -- the int8 stem is what lets conv2 itself be int8.
 *
 * The four are separate rather than shared on purpose. A reused buffer would
 * save 31 KB out of 8 MB of PSRAM and cost a pass whose only job would be to
 * be got right.
 *
 * Only st is in DIRAM. It is written by pass 3 and then read 490 times by
 * conv2, which is the hottest tensor in the network by a wide margin; the other
 * three are touched twice each and cost 94 KB of DIRAM they do not earn. That
 * split is the measured budget, not a rule of thumb -- c1, h1 and f went to
 * PSRAM because moving all four took DIRAM to 97% with 9 KB free, which is not
 * a safe place to be with DMA buffers and task stacks still to allocate.
 */
EXT_RAM_BSS_ATTR static int8_t s_c1q[(size_t)C1_ROWS * KWS_C1_OUT];
EXT_RAM_BSS_ATTR static int8_t s_h1q[(size_t)C1_ROWS * KWS_C1_OUT];
EXT_RAM_BSS_ATTR static int8_t s_fq[(size_t)C1_ROWS * KWS_C1_OUT];
static int8_t s_stq[(size_t)C1_ROWS * KWS_C1_OUT] __attribute__((aligned(16)));
EXT_RAM_BSS_ATTR static float  s_c1s[C1_ROWS];
EXT_RAM_BSS_ATTR static float  s_h1s[C1_ROWS];
EXT_RAM_BSS_ATTR static float  s_fs[C1_ROWS];
/* One scale per (frame, mel) row, as everywhere else. conv2 needs nine of them
 * per window, but nine taps are nine 32-wide dots either way -- the same
 * eighteen EE.VMULAS as one 288-wide dot -- so per-tap scales cost nothing in
 * the reduction and the window is gathered as raw bytes with no requantisation.
 * See kws_conv2d_i8. */
static float  s_sts[C1_ROWS];

/* conv2 output, float32: 94 KB. Quantising it would save 70 KB and halve the
 * traffic, but the only two things downstream are a 48-long dot per mel bin and
 * a 10-element weighted sum -- 47k float operations against 6.8 MMAC. Keeping
 * it float removes an entire requantisation stage for no measurable cost.
 *
 * Stays in PSRAM: it is written once by conv2 and read twice (GroupNorm-3, then
 * the mel gate), so three crossings is already close to the minimum, and at 94 KB
 * it is the largest thing in the model. It is the first candidate to move if the
 * DIRAM budget ever allows. */
EXT_RAM_BSS_ATTR static float s_c2[(size_t)C2_ROWS * KWS_C2_OUT];

/* The conformer's working set. Each is written by one stage and read by the
 * next, several times per block, three blocks over -- so these are the tensors
 * that pay the PSRAM ceiling hardest and the reason they are in DIRAM.
 * 5 x 9,408 B = 47 KB, and the quantised rows below are what the vector
 * reduction actually reads. */
static float s_seq[SEQ_N];
static float s_ln[SEQ_N];
static float s_qkv[QKV_N];
static float s_attn[SEQ_N];
static float s_blk[SEQ_N];
static float s_ffn[FFN_N];

/*
 * Quantised activation rows that feed the vector reduction directly.
 * 16-byte aligned because kws_int8.c loads 128 bits at a time and every row
 * stride is a multiple of 16 -- see KWS_ALIGN16.
 */
#define KWS_QBUF __attribute__((aligned(16)))
EXT_RAM_BSS_ATTR static int8_t s_lnq[SEQ_N]    KWS_QBUF;
EXT_RAM_BSS_ATTR static int8_t s_attnq[SEQ_N]  KWS_QBUF;
EXT_RAM_BSS_ATTR static int8_t s_blkq[SEQ_N]   KWS_QBUF;
EXT_RAM_BSS_ATTR static int8_t s_seqq[SEQ_N]   KWS_QBUF;
EXT_RAM_BSS_ATTR static int8_t s_ffnq[FFN_N]   KWS_QBUF;
/* squeeze-and-excite row (48), its expanded row (6 padded to 16), and the
 * classifier's 97-value feature row padded to 112 */
static float s_rowsc[KWS_FRAMES];      /* per-frame dequantisation scales */
static float s_zq_sc[4];
/* The two squeeze-and-excite rows sit back to back: the 48-value channel
 * average occupies [0,48) and the 6-value expansion [48,64), each padded to a
 * multiple of 16 inside quant_rows. The classifier's 97-value row is longer
 * still and is quantised into the same buffer afterwards. */
#define KWS_SE_ROW0 0
#define KWS_SE_ROW1 KWS_DIM
static int8_t s_zq[128] KWS_QBUF;

static float s_mg[KWS_C2_OUT];           /* mel-gate kernel, dequantised */
static int s_mg_ready;
static const char *s_tag = "noise";      /* fixture name, for trace dumps */
static float s_pool_s[KWS_FRAMES];

/* Small per-frame scratch. Internal DRAM, because it is touched three times per
 * stem pass and is only 2.5 KB each. */
static float s_feat[3][KWS_MELS * 3];    /* delta-stack rows t-1, t, t+1 */
static float s_row[C1_ROW];               /* one stem frame                */
static float s_h1f[C1_ROW];
static float s_fraw[C1_ROW];
static float s_f2f[C1_ROW];
static float s_stf[C1_ROW];
static float s_bcast[KWS_C1_OUT];
static float s_z[KWS_DIM];
#ifdef KWS_MODEL_TRACE
static float s_trf[(size_t)C1_ROWS * KWS_C1_OUT];   /* dequantised for dumps */
#endif
static float s_zr[KWS_SE];
static float s_featv[KWS_FEAT];
static float s_logits[2];

/* ------------------------------------------------------------------ */
/* Mode and timing                                                      */
/* ------------------------------------------------------------------ */

static int s_f32_weights = 0;
static uint32_t s_last_us;
static uint64_t s_total_us;
static uint32_t s_front_us;

int kws_model_set_float_weights(int on)
{
    const int prev = s_f32_weights;
    s_f32_weights = on ? 1 : 0;
    return prev;
}

void kws_model_set_tag(const char *tag) { s_tag = (tag != NULL) ? tag : "noise"; }

uint32_t kws_model_last_us(void) { return s_last_us; }
uint64_t kws_model_total_us(void) { return s_total_us; }

static uint64_t kws_now_us(void)
{
#ifdef KWS_HOST_TEST
    return (uint64_t)(clock() * 1000000.0 / CLOCKS_PER_SEC);
#else
    return (uint64_t)esp_timer_get_time();
#endif
}

/*
 * Per-stage cost accounting. Eight timestamps per inference on a 200 ms hop is
 * free, and it is the difference between optimising the stage that is actually
 * slow and optimising the one that reads as slow.
 */
static const char *const s_stage_name[KWS_STAGE_COUNT] = {
    "stem1", "stem2", "stem3", "conv2", "gn3+sw",
    "melgate", "blocks", "pool+head", "front"
};
static uint64_t s_stage_us[KWS_STAGE_COUNT];
static uint64_t s_stage_runs;

const char *kws_model_stage_name(int slot)
{
    return (slot >= 0 && slot < KWS_STAGE_COUNT) ? s_stage_name[slot] : "?";
}

uint64_t kws_model_stage_us(int slot)
{
    return (slot >= 0 && slot < KWS_STAGE_COUNT) ? s_stage_us[slot] : 0;
}

uint64_t kws_model_stage_runs(void) { return s_stage_runs; }

void kws_model_stage_reset(void)
{
    for (int i = 0; i < KWS_STAGE_COUNT; i++) { s_stage_us[i] = 0; }
    s_stage_runs = 0;
}

/* Close the stage that just ended and open the next one. */
#define STAGE(slot)                                  \
    do {                                              \
        const uint64_t now_ = kws_now_us();           \
        s_stage_us[slot] += now_ - ts_;               \
        ts_ = now_;                                   \
    } while (0)

/* ------------------------------------------------------------------ */
/* Primitives                                                          */
/* ------------------------------------------------------------------ */

/*
 * Fast exp for the activations.
 *
 * swish runs about 86,000 times per inference and the softmaxes and gates add
 * another few hundred, so this is the single hottest transcendental in the
 * model. The ESP32-S3 has no FPU-accelerated exp in ROM and newlib's expf
 * costs the better part of a hundred cycles, which at this call count is
 * milliseconds of the CPU budget.
 *
 * exp(x) = 2^(x * log2 e), split into an integer exponent n and a fractional
 * part r in (-1, 1). 2^n comes from constructing the IEEE-754 exponent field
 * directly; 2^r is a degree-7 Taylor series in r whose next term is 1.5e-5, so
 * the relative error stays under 2e-5 across the whole range.
 *
 * That bound is claimed, not assumed: kws_expf_selftest() below compares against
 * the real expf over the range swish actually sees and refuses to enable the
 * fast path if it is worse than 1e-4. The host harness runs that check on every
 * fixture, so the claim is re-measured on every change to this code.
 */
static inline float kws_expf_raw(float x)
{
    /* exp underflows to 0 below about -88 and overflows float above about 88.
     * Clamping rather than returning a sentinel keeps swish monotone at the
     * ends instead of folding it, which would make the activation a step. */
    if (x < -88.0f) { x = -88.0f; }
    else if (x > 88.0f) { x = 88.0f; }

    const float z = x * 1.44269504f;               /* x * log2(e) */
    const int n = (int)z;                          /* toward zero: r in (-1,1) */
    const float r = z - (float)n;

    const float p =
        1.0f + r * (0.69314718f + r * (0.24022651f + r * (0.05550411f +
               r * (0.00961813f + r * (0.00133336f + r * (0.00015404f +
               r * 0.00001525f))))));

    union { float f; int32_t i; } u;
    u.i = ((int32_t)n + 127) << 23;               /* 2^n */
    return p * u.f;
}

/* -1 until the self test has run, then 1 if the fast exp is good enough. */
static float kws_expf_ok = -1.0f;
static float kws_expf_worst = 1.0f;

#define KWS_EXP_REL_TOL 1.0e-4f

static inline float kws_expf_use(float x)
{
    return (kws_expf_ok > 0.5f) ? kws_expf_raw(x) : expf(x);
}

int kws_expf_selftest(void)
{
    /* swish sees arguments well inside [-30, 30]; beyond that the function
     * saturates and an error there cannot reach a logit. */
    float worst = 0.0f;
    for (int i = -3000; i <= 3000; i++) {
        const float x = (float)i * 0.01f;
        const float want = expf(x);
        if (want < 1e-30f) { continue; }
        const float rel = fabsf(kws_expf_raw(x) - want) / want;
        if (rel > worst) { worst = rel; }
    }
    kws_expf_worst = worst;
    kws_expf_ok = (worst < KWS_EXP_REL_TOL) ? 1.0f : 0.0f;
    return kws_expf_ok > 0.5f;
}

float kws_expf_worst_relerr(void) { return kws_expf_worst; }

/*
 * Reciprocal, by seed plus Newton-Raphson.
 *
 * swish is x / (1 + exp(-x)) and this core has no float divide: every one of
 * the ~86,000 swishes an inference performs went through __divsf3, a ROM
 * routine of roughly sixty instructions. One reciprocal here costs about
 * twelve, and the difference over 86,000 calls is milliseconds.
 *
 * The seed inverts the exponent field and the top of the mantissa, which lands
 * within about 5% for the cost of an integer subtract, and each Newton step
 * squares the relative error: 5% -> 0.1% -> 1e-5 -> 1e-9, so three steps end
 * below float32 resolution.
 *
 * The divisor here is always 1 + something positive, so it is >= 1 and never
 * zero, denormal or negative. That is not an accident of this one call site: a
 * reciprocal is only safe because of the range its callers guarantee, which is
 * why the range is part of the self test rather than a comment.
 *
 * As with the fast exp, the accuracy claim is measured rather than asserted:
 * kws_rcp_selftest() compares against 1.0f/d over the whole range the model
 * produces and refuses the fast path if it is worse than 1e-6, leaving the
 * division in place. The host harness runs that on every fixture.
 */
static float kws_rcp_ok = -1.0f;
static float kws_rcp_worst = 1.0f;
static float kws_rsqrt_worst = 1.0f;

/* Defined below, next to kws_rcp_use, but the self test has to measure it. */
static inline float kws_rsqrt_raw(float v);

#define KWS_RCP_REL_TOL 1.0e-6f

static inline float kws_rcp_raw(float d)
{
    union { float f; int32_t i; } u;
    u.f = d;

    /* Reduce d to a mantissa in [1, 2) and take the exponent out by
     * construction, so the Newton iteration below cannot overflow on a large
     * divisor. Subtracting the whole bit pattern, which is the cheaper trick,
     * overflows the very first multiply for d above about 1e19 and returns
     * garbage: d*g becomes inf, 2-inf is -inf, and -inf*inf poisons the result.
     * The self test covers exactly that range, which is how it was caught. */
    const int32_t bits = u.i;
    const int32_t expo = (bits >> 23) & 0xff;
    union { float f; int32_t i; } mv;
    mv.i = (bits & 0x007fffff) | 0x3f800000;       /* d = m * 2^(expo-127) */
    const float m = mv.f;                           /* m in [1, 2)         */

    /* Seed by the chord of 1/m over [1, 2): 1.5 - 0.5m is exact at both ends
     * and at most 12.5% high in between. */
    float g = 1.5f - 0.5f * m;

    /* Newton-Raphson squares the relative error each step:
     * 12.5% -> 1.6% -> 2.4e-4 -> 5.9e-8, below float32 resolution. */
    g = g * (2.0f - m * g);
    g = g * (2.0f - m * g);
    g = g * (2.0f - m * g);

    /* Put the exponent back as a reciprocal. 2^(127-expo) is a valid normal
     * float for every divisor swish can produce; where the true reciprocal is
     * subnormal this underflows toward zero, which is the correct answer. */
    union { float f; int32_t i; } p;
    p.i = (254 - expo) << 23;
    return g * p.f;
}

static inline float kws_rcp_use(float d)
{
    return (kws_rcp_ok > 0.5f) ? kws_rcp_raw(d) : 1.0f / d;
}

int kws_rcp_selftest(void)
{
    /* 1 + exp(-x) for x in [-88, 88] is 1 .. 1.65e38, and swish only ever calls
     * this with a divisor in that span. The sweep is log-spaced so the large
     * divisors are sampled as densely as the small ones, which a linear sweep
     * from 1 would not do. */
    float worst = 0.0f;
    for (int i = 0; i <= 20000; i++) {
        const float x = -88.0f + 176.0f * ((float)i / 20000.0f);
        const float d = 1.0f + kws_expf_raw(-x);
        const float want = 1.0f / d;
        const float got = kws_rcp_raw(d);
        const float rel = fabsf(got - want) / want;
        if (rel > worst) { worst = rel; }
    }
    /* And the divisors a LayerNorm or a softmax produce, which are O(1). */
    for (int i = 1; i <= 4096; i++) {
        const float d = (float)i * (1.0f / 4096.0f) * 4.0f;
        const float want = 1.0f / d;
        const float rel = fabsf(kws_rcp_raw(d) - want) / want;
        if (rel > worst) { worst = rel; }
    }

    /* The reciprocal square root shares the seed and so shares the verdict: it
     * is only trusted when the reciprocal above it passed. What it needs is its
     * own check over the variances a norm actually produces -- small positive
     * numbers, and the whole exponent range below one, because a quiet channel's
     * variance is many orders of magnitude under a loud one's. */
    float worst_sqrt = 0.0f;
    for (int e = -30; e <= 4; e++) {
        for (int i = 1; i <= 512; i++) {
            const float v = ldexpf((float)i / 512.0f, e);
            const float want = 1.0f / sqrtf(v);
            const float got = kws_rsqrt_raw(v);
            const float rel = fabsf(got - want) / want;
            if (rel > worst_sqrt) { worst_sqrt = rel; }
        }
    }
    kws_rsqrt_worst = worst_sqrt;

    kws_rcp_worst = worst;
    kws_rcp_ok = (worst < KWS_RCP_REL_TOL) ? 1.0f : 0.0f;
    if (kws_rcp_ok > 0.5f && worst_sqrt >= KWS_RCP_REL_TOL) {
        kws_rcp_ok = 0.0f;      /* reciprocal is fine, rsqrt is not: use neither */
    }
    return kws_rcp_ok > 0.5f;
}

float kws_rsqrt_worst_relerr(void) { return kws_rsqrt_worst; }

float kws_rcp_worst_relerr(void) { return kws_rcp_worst; }

/*
 * Reciprocal square root, by seed plus two Newton steps on the reciprocal.
 *
 * LayerNorm and GroupNorm both need 1/sqrt(v), which on this part is sqrtf --
 * another library routine -- followed by a division, another. Neither is hot
 * enough to matter on its own: 294 LayerNorm rows and four GroupNorm finalises
 * per inference. But both are on the path that decides which frames the network
 * listens to, so they are the last places to spend a cycle, and the machinery
 * to avoid the library call is already here and already self-tested.
 *
 * Seeded from the same chord as the reciprocal, so the initial error is the
 * same 12.5%, and each Newton step squares it: 12.5% -> 1.6% -> 2.4e-4, with
 * the last step pushed a second time to land below float32 resolution.
 */
static inline float kws_rsqrt_raw(float v)
{
    union { float f; int32_t i; } u;
    u.f = v;

    /* The UNBIASED exponent, and it is halved with an arithmetic shift that
     * rounds toward negative infinity. Getting this wrong is invisible in a
     * spot check and catastrophic in the sweep: the first version shifted the
     * biased field by the wrong sign and the self test reported a relative
     * error of exactly 1.0 for every input, which is the signature of a result
     * that is off by a power of two rather than merely inaccurate.
     */
    const int32_t eu = ((u.i >> 23) & 0xff) - 127;

    union { float f; int32_t i; } mv;
    mv.i = (u.i & 0x007fffff) | 0x3f800000;       /* m in [1, 2) */
    const float m = mv.f;

    /* Iterate on the MANTISSA only. The seed approximates 1/sqrt(m) from m
     * alone, so the iteration has to converge on 1/sqrt(m); running it against
     * the full v and scaling afterwards makes the two solve different problems.
     *
     * Four steps, not three: each squares the relative error, but from this
     * seed three steps bottom out at 5.3e-4 -- inside the range where a norm
     * still works and outside the range where it is correct. Four reaches
     * 4.2e-7, which is under the tolerance the self test enforces. */
    float g = 1.5f - 0.5f * m;                    /* chord for 1/sqrt(m) */
    g = g * (1.5f - 0.5f * m * g * g);
    g = g * (1.5f - 0.5f * m * g * g);
    g = g * (1.5f - 0.5f * m * g * g);
    g = g * (1.5f - 0.5f * m * g * g);

    /* 1/sqrt(v) = 1/sqrt(m) * 2^(-eu/2). The shift divides eu by two rounding
     * down, so an odd eu has half a power left over: multiply by 2^(-1/2)
     * before applying the even part. */
    if (eu & 1) { g = g * 0.70710678f; }
    union { float f; int32_t i; } p;
    p.i = (127 - (eu >> 1)) << 23;
    return g * p.f;
}

static inline float kws_rsqrt_use(float v)
{
    return (kws_rcp_ok > 0.5f) ? kws_rsqrt_raw(v) : 1.0f / sqrtf(v);
}

static inline float swishf(float x)
{
    return x * kws_rcp_use(1.0f + kws_expf_use(-x));
}

/* Quantise `rows` consecutive runs of `cols` values into a 16-column-aligned
 * buffer: one scale per run, and the row stride rounded up to a multiple of 16
 * with the pad bytes zero.
 *
 * The pad is not optional. kws_int8.c reduces 16 bytes at a time with a vector
 * load, so a row whose stride is 97 would be read as 112 and the 15 bytes past
 * the end belong to the next row. Zero is the only value that keeps the sum
 * correct, and the stride is what keeps the next row's data out of this one. */
static void quant_rows(const float *src, int rows, int cols,
                       int8_t *dst, float *dsc)
{
    const int stride = (cols + 15) & ~15;
    for (int r = 0; r < rows; r++) {
        const float *s = src + (size_t)r * cols;
        int8_t *d = dst + (size_t)r * stride;
        dsc[r] = kws_quant_row(s, cols, d);
        for (int i = cols; i < stride; i++) { d[i] = 0; }
    }
}

/*
 * GEMM through whichever weight representation is selected. Both variants take
 * the same quantised activations and write float32, so the difference the
 * harness measures is the weights and nothing else.
 */
static void dense_i8(float *y, const int8_t *xq, const float *xsc, int idx,
                     const float *bias, int rows, int in_dim, int out_dim)
{
    if (s_f32_weights) {
        kws_gemm_f32(y, xq, xsc, WF(idx), bias, rows, in_dim, out_dim);
    } else {
        kws_gemm_i8(y, xq, xsc, WI8(idx), bias, WS(idx),
                    rows, in_dim, out_dim);
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

        const float inv = kws_rsqrt_use(var + KWS_LN_EPS);
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
 * coupling across the whole tensor is the reason the stem is materialised
 * rather than streamed a row at a time, and the reason the stem is three passes:
 * one to accumulate the statistics, one to apply them and produce the input the
 * next convolution needs, and one to apply the second normalisation and combine.
 */
/*
 * GroupNorm with 4 groups over the last axis.
 *
 * Flax computes the statistics per (sample, group) across every OTHER axis, so
 * for a (49,20,32) tensor it averages over 49*20*8 elements per group. That
 * coupling across the whole tensor is why the stem is materialised rather than
 * streamed a row at a time, and why the stem is three passes: one to accumulate
 * the statistics, one to apply them and produce the input the next convolution
 * needs, and one to apply the second normalisation and combine.
 *
 * The statistics are a value, not a global, because the stem needs two sets at
 * once. GroupNorm-1 has to be applied inside the same loop that accumulates
 * GroupNorm-2, so holding them in one global pair means the second reset
 * silently zeroes the first -- which produces a tensor that is normalised by
 * nothing, and a probability that is wrong rather than obviously broken.
 */
typedef struct {
    float mean[4];
    float inv[4];
    float n;
    float sum[4];
    float sumsq[4];
} kws_gn;

static void gn_reset(kws_gn *g)
{
    for (int i = 0; i < 4; i++) {
        g->mean[i] = 0.0f;
        g->inv[i] = 1.0f;
        g->sum[i] = 0.0f;
        g->sumsq[i] = 0.0f;
    }
    g->n = 0.0f;
}

static void gn_accum(kws_gn *g, const float *x, int rows, int dim)
{
    const int per = dim / 4;
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * dim;
        for (int i = 0; i < 4; i++) {
            float a = 0.0f, b = 0.0f;
            for (int j = 0; j < per; j++) {
                const float v = xr[i * per + j];
                a += v;
                b += v * v;
            }
            g->sum[i] += a;
            g->sumsq[i] += b;
        }
    }
    g->n += (float)rows * (float)per;
}

static void gn_finalize(kws_gn *g)
{
    /* Every GroupNorm here averages over every OTHER axis: 49 x 20 x (dim/4)
     * per group for the stem, 49 x 10 x 12 for the third. g->n accumulates that
     * count as the statistics are gathered, so it cannot disagree with what was
     * actually summed. */
    const float n = g->n;
    for (int i = 0; i < 4; i++) {
        const float mean = g->sum[i] / n;
        g->mean[i] = mean;
        g->inv[i] = kws_rsqrt_use(g->sumsq[i] / n - mean * mean + KWS_LN_EPS);
    }
}

/*
 * Dequantise one stem tensor, apply a set of statistics, and swish, writing
 * float32. This is the operation that turns a stored int8 tensor back into the
 * float32 the depthwise convolution and the residual add need, and it is where
 * the accuracy cost of storing the stem in INT8 is paid.
 */
static void gn_dq_swish(const kws_gn *g, const int8_t *q, const float *sc,
                        float *dst, int rows, int dim,
                        const float *scale, const float *bias)
{
    const int per = dim / 4;
    for (int r = 0; r < rows; r++) {
        const float s = sc[r];
        const int8_t *qr = q + (size_t)r * dim;
        float *dr = dst + (size_t)r * dim;
        for (int c = 0; c < dim; c++) {
            const float v = (float)qr[c] * s;
            dr[c] = swishf((v - g->mean[c / per]) * g->inv[c / per]
                           * scale[c] + bias[c]);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Stem                                                                */
/* ------------------------------------------------------------------ */

static inline int clamp_frame(int f)
{
    if (f < 0) { return 0; }
    if (f > KWS_FRAMES - 1) { return KWS_FRAMES - 1; }
    return f;
}

/*
 * spec[f][m], with the frame CLAMPED.
 *
 * The clamp is not an optimisation and not defensive padding: it is what numpy's
 * mode="edge" means. Without it, asking for frame -1 computes
 * (size_t)(-1) * 40 + m, reads megabytes outside the buffer, and hands the
 * result to a convolution -- which is a plausible-looking number rather than a
 * crash, so it survives far longer than it should.
 */
static inline float spec_at(const float *spec, int frame, int m)
{
    if (frame < 0) { frame = 0; }
    else if (frame > KWS_FRAMES - 1) { frame = KWS_FRAMES - 1; }
    return spec[(size_t)frame * KWS_MELS + m];
}

/*
 * First time derivative at frame f, with the FRAME clamped before the
 * difference. That clamp belongs here as well as in spec_at: numpy pads the d1
 * array itself with mode="edge", so d1[-1] is d1[0], not a difference of two
 * clamped raw frames (which would evaluate to zero).
 */
static inline float d1_at(const float *spec, int f, int m)
{
    f = clamp_frame(f);
    return 0.5f * (spec_at(spec, f + 1, m) - spec_at(spec, f - 1, m));
}

/*
 * Fill the three delta-stack rows conv1 consumes at output frame t, i.e. frames
 * t-1, t and t+1, each as [value, d1, d2] over mel.
 *
 * Three details that are easy to get wrong and have all been wrong here:
 *
 * 1. The derivatives run along TIME, not mel. models.py::_delta_stack pads
 *    axis 1 of a (B, T, M) array, which is the frame axis. Differencing over
 *    mel instead shifts the model onto features it never saw, and the only
 *    symptom is a probability that does not match training.
 *
 * 2. conv1 uses SAME padding, which pads with ZEROS, not by replicating the
 *    edge. For t = 0 the t-1 row, and for t = the last frame the t+1 row, must
 *    be all zeros. Edge-replicating them (as the delta stack does internally)
 *    feeds conv1 a non-zero phantom frame at both boundaries.
 *
 * 3. d2 needs d1 at f+1 and f-1, which in turn needs frames f-2 .. f+2.
 *    Indexing the spec with clamping expresses that directly.
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
            dst[(size_t)m * 3 + 0] = spec_at(spec, f, m);
            dst[(size_t)m * 3 + 1] = d1_at(spec, f, m);
            dst[(size_t)m * 3 + 2] = 0.5f * (d1_at(spec, f + 1, m)
                                           - d1_at(spec, f - 1, m));
        }
    }
}

/*
 * conv1 for one frame: 3 rows of (40 mel x 3 chan) -> (20 mel x 32 chan).
 * Kernel (3,3,3,32), stride (1,2) over mel, SAME, no bias.
 *
 * Exported reversed, so the weight is [oc][ic][kw][kt] and the two reduction
 * axes, channel and kernel, are both innermost.
 */
static void conv1_f32(const float *sfeat, float *out)
{
    const float *w = WF(KWS_T_CONV1);
    /* JAX "SAME": pad_total = (out-1)*stride + k - in, pad_lo = pad_total // 2
     * (FLOOR). Using ceil here silently shifts the whole feature by one mel
     * bin. For mel: (20-1)*2 + 3 - 40 = 1, so pad_lo = 0 and a single sample of
     * padding on the right. The time axis needs kT/2 = 1 of padding either
     * side, which build_feat supplies. */
    const int pad_m = ((ROW_W - 1) * 2 + 3 - KWS_MELS) / 2;

    for (int ow = 0; ow < ROW_W; ow++) {
        for (int oc = 0; oc < KWS_C1_OUT; oc++) {
            const float *wo = w + oc * 27;
            float acc = 0.0f;
            for (int kt = 0; kt < 3; kt++) {
                const float *fr = sfeat + kt * KWS_MELS * 3;
                for (int kw = 0; kw < 3; kw++) {
                    const int iw = ow * 2 + kw - pad_m;
                    if (iw < 0 || iw >= KWS_MELS) { continue; }
                    const float *xp = fr + iw * 3;
                    const float *wp = wo + kw * 3 + kt;
                    acc += xp[0] * wp[0];
                    acc += xp[1] * wp[9];
                    acc += xp[2] * wp[18];
                }
            }
            out[(size_t)ow * KWS_C1_OUT + oc] = acc;
        }
    }
}

/*
 * Depthwise convolution over mel for one frame, SAME, no bias.
 * Flax stores a grouped conv kernel as (kT, kF, in/groups, out/groups), so
 * feature_group_count == channels leaves the channel axis 1 wide: the checkpoint
 * shape is (1,3,1,32), reversed it is [c][0][k] and the weight index is
 * simply c * 3 + k.
 */
static void fconv_f32(const float *in, float *out)
{
    const float *w = WF(KWS_T_FCONV);
    for (int iw = 0; iw < ROW_W; iw++) {
        for (int c = 0; c < KWS_C1_OUT; c++) {
            float acc = 0.0f;
            for (int kw = 0; kw < 3; kw++) {
                const int jw = iw + kw - 1;
                if (jw < 0 || jw >= ROW_W) { continue; }
                acc += in[(size_t)jw * KWS_C1_OUT + c] * w[c * 3 + kw];
            }
            out[(size_t)iw * KWS_C1_OUT + c] = acc;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Conformer                                                          */
/* ------------------------------------------------------------------ */

/*
 * Relative-position multi-head self-attention over the 49 frames.
 *
 * logits[i][j] = q_i . k_j / sqrt(dh) + bias[h][i - j + (T-1)]
 *
 * The bias is indexed by signed distance, so it expresses "prefer nearby
 * context" and structurally cannot encode "the keyword belongs at frame 14" --
 * which is exactly the positional prior this architecture dropped.
 */
static void rel_attention(const float *bias)
{
    const int T = KWS_FRAMES, H = KWS_HEADS, D = KWS_DIM, dh = KWS_HEAD_DIM;
    const float inv = 1.0f / sqrtf((float)dh);

    static float w[KWS_FRAMES];

    for (int h = 0; h < H; h++) {
        const float *bh = bias + h * (2 * T - 1);
        for (int i = 0; i < T; i++) {
            const float *qi = s_qkv + (size_t)i * 3 * D + h * dh;
            float mx = -3.0e38f;
            for (int j = 0; j < T; j++) {
                const float *kj = s_qkv + (size_t)j * 3 * D + D + h * dh;
                float acc = 0.0f;
                for (int d = 0; d < dh; d++) { acc += qi[d] * kj[d]; }
                acc = acc * inv + bh[i - j + (T - 1)];
                w[j] = acc;
                if (acc > mx) { mx = acc; }
            }
            float sum = 0.0f;
            for (int j = 0; j < T; j++) { w[j] = kws_expf_use(w[j] - mx); sum += w[j]; }
            const float isum = 1.0f / sum;

            float *oi = s_attn + (size_t)i * D + h * dh;
            for (int d = 0; d < dh; d++) { oi[d] = 0.0f; }
            for (int j = 0; j < T; j++) {
                const float wj = w[j] * isum;
                const float *vj = s_qkv + (size_t)j * 3 * D + 2 * D + h * dh;
                for (int d = 0; d < dh; d++) { oi[d] += wj * vj[d]; }
            }
        }
    }
}

/* Depthwise convolution over TIME, per channel. Kernel (K,1,D) reversed to
 * [c][0][k], so the weight index is c * K + j. SAME, no bias. */
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
                acc += in[(size_t)tt * D + c] * k[c * K + j];
            }
            out[(size_t)t * D + c] = acc;
        }
    }
}

/* One RelConformerBlock, in place on x. */
static void conformer_block(float *x, int b)
{
    const int T = KWS_FRAMES, D = KWS_DIM;
    const int base = KWS_T_BLOCKS + b * KWS_BLOCK_TENSORS;
    int i;

    /* --- attention ---------------------------------------------------- */
    layer_norm(x, WF(base + 0), WF(base + 1), T, D, s_ln);
    TRACE("[b%d.0] LN_0   %.5f\n", b, trace_rms(s_ln, SEQ_N));
    quant_rows(s_ln, T, D, s_lnq, s_rowsc);
    dense_i8(s_qkv, s_lnq, s_rowsc, base + 2, WF(base + 3), T, D, 3 * D);
    TRACE("[b%d.1] qkv    %.5f\n", b, trace_rms(s_qkv, QKV_N));
    rel_attention(WF(base + 4));
    TRACE("[b%d.2] attn   %.5f\n", b, trace_rms(s_attn, SEQ_N));
    quant_rows(s_attn, T, D, s_attnq, s_rowsc);
    dense_i8(s_attn, s_attnq, s_rowsc, base + 5, WF(base + 6), T, D, D);
    TRACE("[b%d.3] proj   %.5f\n", b, trace_rms(s_attn, SEQ_N));
    for (i = 0; i < (int)SEQ_N; i++) { x[i] += s_attn[i]; }

    /* --- multi-scale depthwise conv + pointwise + SE ------------------- */
    layer_norm(x, WF(base + 7), WF(base + 8), T, D, s_ln);
    TRACE("[b%d.4] LN_1   %.5f\n", b, trace_rms(s_ln, SEQ_N));
    dwconv_t(s_ln, WF(base + 9), 3, s_blk);
    dwconv_t(s_ln, WF(base + 10), 7, s_attn);
    for (i = 0; i < (int)SEQ_N; i++) { s_blk[i] = swishf(s_blk[i]) + swishf(s_attn[i]); }
    TRACE("[b%d.5] dw     %.5f\n", b, trace_rms(s_blk, SEQ_N));
    quant_rows(s_blk, T, D, s_blkq, s_rowsc);
    dense_i8(s_attn, s_blkq, s_rowsc, base + 11, NULL, T, D, D);
    TRACE("[b%d.6] pw     %.5f\n", b, trace_rms(s_attn, SEQ_N));

    for (int c = 0; c < D; c++) { s_z[c] = 0.0f; }
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < D; c++) { s_z[c] += s_attn[(size_t)t * D + c]; }
    }
    for (int c = 0; c < D; c++) { s_z[c] /= (float)T; }

    quant_rows(s_z, 1, D, s_zq + KWS_SE_ROW0, s_zq_sc);
    dense_i8(s_zr, s_zq, s_zq_sc, base + 12, NULL, 1, D, KWS_SE);
    for (int c = 0; c < KWS_SE; c++) { s_zr[c] = swishf(s_zr[c]); }
    quant_rows(s_zr, 1, KWS_SE, s_zq + KWS_SE_ROW1, s_zq_sc + 1);
    dense_i8(s_z, s_zq + KWS_SE_ROW1, s_zq_sc + 1, base + 13, NULL, 1, KWS_SE, D);

    for (i = 0; i < (int)SEQ_N; i++) {
        x[i] += s_attn[i] / (1.0f + kws_expf_use(-s_z[i % D]));
    }

    /* --- Swish feed-forward -------------------------------------------- */
    layer_norm(x, WF(base + 14), WF(base + 15), T, D, s_ln);
    TRACE("[b%d.9] LN_2   %.5f\n", b, trace_rms(s_ln, SEQ_N));
    quant_rows(s_ln, T, D, s_lnq, s_rowsc);
    dense_i8(s_ffn, s_lnq, s_rowsc, base + 16, WF(base + 17), T, D, 2 * D);
    TRACE("[b%d.10] ffn1  %.5f\n", b, trace_rms(s_ffn, FFN_N));
    for (i = 0; i < (int)FFN_N; i++) { s_ffn[i] = swishf(s_ffn[i]); }
    quant_rows(s_ffn, T, 2 * D, s_ffnq, s_rowsc);
    dense_i8(s_attn, s_ffnq, s_rowsc, base + 18, WF(base + 19), T, 2 * D, D);
    TRACE("[b%d.11] ffn2  %.5f\n", b, trace_rms(s_attn, SEQ_N));
    for (i = 0; i < (int)SEQ_N; i++) { x[i] += s_attn[i]; }
}

/* ------------------------------------------------------------------ */
/* Soft-OR pooling + classifier                                        */
/* ------------------------------------------------------------------ */

/*
 * mean + log-sum-exp 'soft OR' + max over time.
 *
 * A softmax attention pool is free to collapse onto one preferred frame, which
 * is the mechanism behind a positional prior: the network learns "trust frame
 * 14" instead of "trust whichever frame looks like the keyword". A soft OR asks
 * the deployed question -- is there ANY frame carrying the keyword -- and being
 * symmetric in the frames it cannot encode where that evidence was.
 */
static void soft_or_pool(const float *x, float *out)
{
    const int T = KWS_FRAMES, D = KWS_DIM, H = KWS_POOL_HIDDEN;
    const float *w = WF(KWS_T_OR_W);
    const float b = *WF(KWS_T_OR_B);

    float sum[D], mx[D];
    for (int c = 0; c < D; c++) { sum[c] = 0.0f; mx[c] = -3.0e38f; }
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < D; c++) {
            const float v = x[(size_t)t * D + c];
            sum[c] += v;
            if (v > mx[c]) { mx[c] = v; }
        }
    }
    for (int c = 0; c < D; c++) { sum[c] /= (float)T; }

    quant_rows(x, T, D, s_seqq, s_rowsc);
    dense_i8(s_blk, s_seqq, s_rowsc, KWS_T_POOL_PROJ, WF(KWS_T_POOL_BIAS),
             T, D, H);

    float mxs = -3.0e38f;
    for (int t = 0; t < T; t++) {
        float s = b;
        for (int h = 0; h < H; h++) { s += s_blk[(size_t)t * H + h] * w[h]; }
        s_pool_s[t] = s;
        if (s > mxs) { mxs = s; }
    }
    float lse = 0.0f;
    for (int t = 0; t < T; t++) { lse += kws_expf_use(s_pool_s[t] - mxs); }
    lse = mxs + logf(lse) - logf((float)T);

    for (int c = 0; c < D; c++) { out[c] = sum[c]; }
    out[D] = lse;
    for (int c = 0; c < D; c++) { out[D + 1 + c] = mx[c]; }
}

/* ------------------------------------------------------------------ */
/* Entry points                                                        */
/* ------------------------------------------------------------------ */

void kws_model_init(void)
{
    /*
     * Zero every activation buffer, not just the ones a previous call was
     * thought to have written. Reading a buffer that some path never touched
     * makes the result depend on whatever was in that RAM, which presents as a
     * model that returns a different probability on every call. Clearing
     * everything makes the run deterministic, which is also what makes it
     * debuggable.
     */
    memset(s_c1q, 0, sizeof(s_c1q));
    memset(s_h1q, 0, sizeof(s_h1q));
    memset(s_fq, 0, sizeof(s_fq));
    memset(s_stq, 0, sizeof(s_stq));
    memset(s_c1s, 0, sizeof(s_c1s));
    memset(s_h1s, 0, sizeof(s_h1s));
    memset(s_fs, 0, sizeof(s_fs));
    memset(s_sts, 0, sizeof(s_sts));
    memset(s_c2, 0, sizeof(s_c2));
    memset(s_seq, 0, sizeof(s_seq));
    memset(s_ln, 0, sizeof(s_ln));
    memset(s_qkv, 0, sizeof(s_qkv));
    memset(s_attn, 0, sizeof(s_attn));
    memset(s_blk, 0, sizeof(s_blk));
    memset(s_ffn, 0, sizeof(s_ffn));
    memset(s_lnq, 0, sizeof(s_lnq));
    memset(s_attnq, 0, sizeof(s_attnq));
    memset(s_blkq, 0, sizeof(s_blkq));
    memset(s_seqq, 0, sizeof(s_seqq));
    memset(s_ffnq, 0, sizeof(s_ffnq));
    memset(s_feat, 0, sizeof(s_feat));
    memset(s_row, 0, sizeof(s_row));
    memset(s_h1f, 0, sizeof(s_h1f));
    memset(s_fraw, 0, sizeof(s_fraw));
    memset(s_f2f, 0, sizeof(s_f2f));
    memset(s_stf, 0, sizeof(s_stf));
    memset(s_bcast, 0, sizeof(s_bcast));
    memset(s_z, 0, sizeof(s_z));
    memset(s_zr, 0, sizeof(s_zr));
    memset(s_featv, 0, sizeof(s_featv));
    memset(s_logits, 0, sizeof(s_logits));
    memset(s_mg, 0, sizeof(s_mg));
    memset(s_pool_s, 0, sizeof(s_pool_s));

    /* The mel gate is a 48-long dot per mel bin, 490 of them: 23k operations
     * against 11.2 MMAC. Dequantising its 48 weights once is cheaper and simpler
     * than threading a scale through that loop. Done lazily as well as here,
     * because a host harness may bind the blob after calling init. */
    s_mg_ready = 0;

    s_last_us = 0;
    s_total_us = 0;
    kws_model_stage_reset();
    (void)kws_expf_selftest();
    (void)kws_rcp_selftest();
}

/*
 * First half: delta stack, conv1, both stem GroupNorms, conv2, GroupNorm-3,
 * and the mel gate that reduces the stem output to the 49x48 the conformer
 * stack consumes. Leaves s_seq ready and returns; kws_model_back() consumes it.
 */
void kws_model_front(const float *spec)
{
    const uint64_t t_start = kws_now_us();
    uint64_t ts_ = t_start;
    kws_gn gn1, gn2, gn3;
    int t;

    if (!s_mg_ready) {
        const int8_t *mg = WI8(KWS_T_MEL_GATE);
        const float mgs = WS(KWS_T_MEL_GATE);
        for (int c = 0; c < KWS_C2_OUT; c++) { s_mg[c] = (float)mg[c] * mgs; }
        s_mg_ready = 1;
    }

    /* ---- stem pass 1: conv1 -> int8, and the GroupNorm-1 statistics ---- */
    gn_reset(&gn1);
    for (t = 0; t < KWS_FRAMES; t++) {
        build_feat(spec, t);
        conv1_f32((const float *)s_feat, s_row);
        tr_acc(0, s_feat[0], KWS_MELS * 3);
        tr_acc(1, s_row, C1_ROW);
        gn_accum(&gn1, s_row, ROW_W, KWS_C1_OUT);
        quant_rows(s_row, ROW_W, KWS_C1_OUT,
                   s_c1q + (size_t)t * C1_ROW, s_c1s + (size_t)t * ROW_W);
    }
    gn_finalize(&gn1);
    STAGE(0);
    TRACE("[1] conv1   rms %.5f  gn1 mean %.5f inv %.4f\n",
          trace_rms(s_row, C1_ROW), gn1.mean[0], gn1.inv[0]);

    /* ---- stem pass 2: GroupNorm-1 + swish, depthwise mel conv -> int8,
     *                  and the GroupNorm-2 statistics -------------------- */
    gn_reset(&gn2);
    for (t = 0; t < KWS_FRAMES; t++) {
        gn_dq_swish(&gn1, s_c1q + (size_t)t * C1_ROW, s_c1s + (size_t)t * ROW_W,
                    s_h1f, ROW_W, KWS_C1_OUT,
                    WF(KWS_T_GN1_SCALE), WF(KWS_T_GN1_BIAS));
#ifdef KWS_MODEL_TRACE
        if (t == 3) {
            const float sc0 = s_c1s[(size_t)t * ROW_W];
            const int8_t *q0 = s_c1q + (size_t)t * C1_ROW;
            TRACE("[2.t3] sc0 %.5f q %d %d %d %d | gn1 %.4f %.4f %.4f %.4f"
                  " | h1f %.4f %.4f %.4f %.4f\n",
                  (double)sc0, (int)q0[0], (int)q0[1], (int)q0[2], (int)q0[3],
                  (double)((q0[0] * sc0 - gn1.mean[0]) * gn1.inv[0]
                           * WF(KWS_T_GN1_SCALE)[0] + WF(KWS_T_GN1_BIAS)[0]),
                  (double)((q0[1] * sc0 - gn1.mean[1]) * gn1.inv[1]
                           * WF(KWS_T_GN1_SCALE)[1] + WF(KWS_T_GN1_BIAS)[1]),
                  (double)((q0[2] * sc0 - gn1.mean[0]) * gn1.inv[0]
                           * WF(KWS_T_GN1_SCALE)[2] + WF(KWS_T_GN1_BIAS)[2]),
                  (double)((q0[3] * sc0 - gn1.mean[0]) * gn1.inv[0]
                           * WF(KWS_T_GN1_SCALE)[3] + WF(KWS_T_GN1_BIAS)[3]),
                  (double)s_h1f[0], (double)s_h1f[1], (double)s_h1f[2],
                  (double)s_h1f[3]);
        }
#endif
        fconv_f32(s_h1f, s_fraw);
        tr_acc(2, s_h1f, C1_ROW);
        tr_acc(3, s_fraw, C1_ROW);
        gn_accum(&gn2, s_fraw, ROW_W, KWS_C1_OUT);
        quant_rows(s_fraw, ROW_W, KWS_C1_OUT,
                   s_fq + (size_t)t * C1_ROW, s_fs + (size_t)t * ROW_W);
        quant_rows(s_h1f, ROW_W, KWS_C1_OUT,
                   s_h1q + (size_t)t * C1_ROW, s_h1s + (size_t)t * ROW_W);
    }
    gn_finalize(&gn2);
    STAGE(1);
    TRACE("[2] h1f rms %.5f  fraw rms %.5f  gn2 mean %.5f inv %.4f\n",
          trace_rms(s_h1f, C1_ROW), trace_rms(s_fraw, C1_ROW),
          gn2.mean[0], gn2.inv[0]);

    /* ---- stem pass 3: GroupNorm-2 + swish, residual, broadcast add, and
     *                  the int8 tensor conv2 reads ----------------------- */
    for (t = 0; t < KWS_FRAMES; t++) {
        gn_dq_swish(&gn2, s_fq + (size_t)t * C1_ROW, s_fs + (size_t)t * ROW_W,
                    s_f2f, ROW_W, KWS_C1_OUT,
                    WF(KWS_T_GN2_SCALE), WF(KWS_T_GN2_BIAS));
        tr_acc(4, s_f2f, C1_ROW);

        /* h = h + f_conv + mean_over_mel(f_conv). The broadcast term is the
         * mean of the ALREADY normalised and swished f_conv, which is why it
         * comes out of pass 3 and not out of the raw convolution. */
        for (int c = 0; c < KWS_C1_OUT; c++) {
            float m = 0.0f;
            for (int iw = 0; iw < ROW_W; iw++) {
                m += s_f2f[(size_t)iw * KWS_C1_OUT + c];
            }
            s_bcast[c] = m / (float)ROW_W;
        }

        for (int iw = 0; iw < ROW_W; iw++) {
            const int8_t *h1 = s_h1q + (size_t)t * C1_ROW
                               + (size_t)iw * KWS_C1_OUT;
            const float h1s = s_h1s[(size_t)t * ROW_W + iw];
            for (int c = 0; c < KWS_C1_OUT; c++) {
                s_stf[(size_t)iw * KWS_C1_OUT + c] =
                    (float)h1[c] * h1s
                    + s_f2f[(size_t)iw * KWS_C1_OUT + c]
                    + s_bcast[c];
            }
        }
        /* Per (frame, mel) row scales, each row written 16-byte aligned: that is the
         * layout conv2's vector reduction needs, and nine taps of nine scales is
         * free once the window is gathered as bytes instead of requantised. */
        quant_rows(s_stf, ROW_W, KWS_C1_OUT,
                   s_stq + (size_t)t * C1_ROW, s_sts + (size_t)t * ROW_W);
        tr_acc(5, s_stf, C1_ROW);
    }
    STAGE(2);
    TRACE("[3] stem    rms %.5f  stq[0] %d s %.4g\n",
          trace_rms(s_stf, C1_ROW), (int)s_stq[0], (double)s_sts[0]);

    /* ---- conv2 + GroupNorm-3 + swish ---------------------------------- */
    if (s_f32_weights) {
        kws_conv2d_f32(s_c2, s_stq, s_sts, WF(KWS_T_CONV2),
                       KWS_FRAMES, ROW_W, KWS_STEM_MELS,
                       KWS_C1_OUT, KWS_C2_OUT, 3, 3);
    } else {
        kws_conv2d_i8(s_c2, s_stq, s_sts, WI8(KWS_T_CONV2), WS(KWS_T_CONV2),
                      KWS_FRAMES, ROW_W, KWS_STEM_MELS,
KWS_C1_OUT, KWS_C2_OUT, 3, 3);
    }
    STAGE(3);

    gn_reset(&gn3);
    gn_accum(&gn3, s_c2, (int)C2_ROWS, KWS_C2_OUT);
    gn_finalize(&gn3);
#ifdef KWS_MODEL_TRACE
    {
        char p[256];
        snprintf(p, sizeof(p), "main/model/reference/stages/c_%s_conv2.bin", s_tag);
        tr_save(p, s_c2, (size_t)C2_ROWS * KWS_C2_OUT);
    }
#endif
    {
        const float *gscale = WF(KWS_T_GN3_SCALE);
        const float *gbias  = WF(KWS_T_GN3_BIAS);
        const int per = KWS_C2_OUT / 4;
        for (int r = 0; r < (int)C2_ROWS; r++) {
            float *row = s_c2 + (size_t)r * KWS_C2_OUT;
            for (int c = 0; c < KWS_C2_OUT; c++) {
                const int g = c / per;
                row[c] = swishf((row[c] - gn3.mean[g]) * gn3.inv[g]
                                * gscale[c] + gbias[c]);
            }
        }
    }
    TRACE("[4] conv2   rms %.5f  gn3 mean %.5f inv %.4f\n",
          trace_rms(s_c2, (size_t)C2_ROWS * KWS_C2_OUT),
          gn3.mean[0], gn3.inv[0]);
    tr_acc(7, s_c2, (size_t)C2_ROWS * KWS_C2_OUT);
    STAGE(4);
#ifdef KWS_MODEL_TRACE
    {
        char p[256];
        snprintf(p, sizeof(p), "main/model/reference/stages/c_%s_gn3.bin", s_tag);
        tr_save(p, s_c2, (size_t)C2_ROWS * KWS_C2_OUT);
    }
#endif

    /* ---- mel gate: a learned softmax over the 10 mel bins -------------- */
    for (t = 0; t < KWS_FRAMES; t++) {
        const float *ht = s_c2 + (size_t)t * C2_ROW;
        float mx = -3.0e38f;
        for (int iw = 0; iw < KWS_STEM_MELS; iw++) {
            const float *hr = ht + (size_t)iw * KWS_C2_OUT;
            float acc = 0.0f;
            for (int c = 0; c < KWS_C2_OUT; c++) { acc += hr[c] * s_mg[c]; }
            s_pool_s[iw] = acc;
            if (acc > mx) { mx = acc; }
        }
        float sum = 0.0f;
        for (int iw = 0; iw < KWS_STEM_MELS; iw++) {
            s_pool_s[iw] = kws_expf_use(s_pool_s[iw] - mx);
            sum += s_pool_s[iw];
        }
        const float isum = 1.0f / sum;

        float *sq = s_seq + (size_t)t * KWS_DIM;
        for (int c = 0; c < KWS_C2_OUT; c++) { sq[c] = 0.0f; }
        for (int iw = 0; iw < KWS_STEM_MELS; iw++) {
            const float *hr = ht + (size_t)iw * KWS_C2_OUT;
            const float ww = s_pool_s[iw] * isum;
            for (int c = 0; c < KWS_C2_OUT; c++) { sq[c] += hr[c] * ww; }
        }
    }
    TRACE("[5] seq     rms %.5f\n", trace_rms(s_seq, SEQ_N));
    tr_acc(8, s_seq, SEQ_N);
    STAGE(5);

    s_front_us = (uint32_t)(kws_now_us() - t_start);
}

/*
 * The second half: conformer stack, soft-OR pool, classifier, softmax.
 *
 * Split from the first half at the only clean seam in the model. Everything
 * before this point writes s_seq and reads nothing the stack touches; everything
 * after reads s_seq and tensors derived from it. No buffer is written by one half
 * and read by the other, which is what lets the firmware run the two halves on
 * separate cores with a single handoff on s_seq and no other synchronisation.
 *
 * kws_model_run() remains the sequential composition of the two, and that is what
 * the host harness calls -- so the parity numbers describe exactly the arithmetic
 * both cores execute, with no threading inside the measurement.
 */
float kws_model_back(void)
{
    const uint64_t t_start = kws_now_us();
    uint64_t ts_ = t_start;

#ifdef KWS_MODEL_TRACE
    {
        static char path[256];
        size_t r;
        for (r = 0; r < (size_t)C1_ROWS; r++) {
            const float s = s_c1s[r];
            const int8_t *q = s_c1q + r * KWS_C1_OUT;
            for (int c = 0; c < KWS_C1_OUT; c++) { s_trf[r * KWS_C1_OUT + c] = q[c] * s; }
        }
        snprintf(path, sizeof(path), "main/model/reference/stages/c_%s_conv1.bin", s_tag);
        tr_save(path, s_trf, (size_t)C1_ROWS * KWS_C1_OUT);
        for (r = 0; r < (size_t)C1_ROWS; r++) {
            const float s = s_fs[r];
            const int8_t *q = s_fq + r * KWS_C1_OUT;
            for (int c = 0; c < KWS_C1_OUT; c++) { s_trf[r * KWS_C1_OUT + c] = q[c] * s; }
        }
        snprintf(path, sizeof(path), "main/model/reference/stages/c_%s_fconv.bin", s_tag);
        tr_save(path, s_trf, (size_t)C1_ROWS * KWS_C1_OUT);
    }
#endif

    /* ---- conformer stack ------------------------------------------------ */
    for (int b = 0; b < KWS_BLOCKS; b++) { conformer_block(s_seq, b); }
    TRACE("[6] blocks  rms %.5f\n", trace_rms(s_seq, SEQ_N));
    tr_acc(9, s_seq, SEQ_N);
    STAGE(6);

    /* ---- soft-OR pool + classifier -------------------------------------- */
    soft_or_pool(s_seq, s_featv);
    TRACE("[7] feat    rms %.5f  lse %.5f\n",
          trace_rms(s_featv, KWS_FEAT), (double)s_featv[KWS_DIM]);

    {
        float feat_scale;
        quant_rows(s_featv, 1, KWS_FEAT, s_zq + KWS_SE_ROW0, &feat_scale);
        dense_i8(s_logits, s_zq + KWS_SE_ROW0, &feat_scale, KWS_T_HEAD_K, WF(KWS_T_HEAD_B),
                 1, KWS_FEAT, 2);
    }

    const float mx = (s_logits[0] > s_logits[1]) ? s_logits[0] : s_logits[1];
    const float e0 = kws_expf_use(s_logits[0] - mx);
    const float e1 = kws_expf_use(s_logits[1] - mx);

    s_last_us = (uint32_t)((kws_now_us() - t_start) + s_front_us);
    s_stage_us[7] += kws_now_us() - ts_;
    s_stage_runs++;
    s_total_us += s_last_us;

    tr_dump((const char *const[]){
        "delta", "conv1", "gn1+sw", "fconv", "gn2+sw", "stem",
        "conv2", "gn3+sw", "seq", "blocks" });

#ifdef KWS_MODEL_TRACE
    {
        char p[256];
        snprintf(p, sizeof(p), "main/model/reference/stages/c_%s_seq.bin", s_tag);
        tr_save(p, s_seq, SEQ_N);
        snprintf(p, sizeof(p), "main/model/reference/stages/c_%s_feat.bin", s_tag);
        tr_save(p, s_featv, KWS_FEAT);
    }
#endif

    return e1 / (e0 + e1);
}

/*
 * Both halves in sequence, on one core.
 *
 * This is the reference path and the one the host harness measures: same
 * arithmetic, same buffers, no threading, so a parity regression here is a
 * numerical change and nothing else. The firmware may instead run the halves on
 * two cores -- kws_model_front() and kws_model_back() are the same two pieces of
 * code either way.
 */
float kws_model_run(const float *spec)
{
    kws_model_front(spec);
    return kws_model_back();
}
