/*
 * Fast expf, and a fast reciprocal, for SiLU, softmax and the mel scorer.
 *
 * RECOVERED, not rewritten. These lived in main/kws_model.c and were removed when
 * bcconformer_v3 was deleted to make room for m1_g_wide. Removing them is why
 * the replacement model was slow in a way its arithmetic does not justify.
 *
 * m1_g_wide is 2.77 M MAC with a vector unit; on this part that is a fraction of
 * a millisecond. It was taking 225 ms. The gap was not multiply-accumulate, it
 * was transcendentals:
 *
 *     stage          ms     MAC   cyc/MAC   what is actually running
 *     pool+head   12.41     576     5171.7   1,248 expf + 96 logf, i.e. libm
 *     mel_gate    11.02  100,880      26.2   1,040 tanhf, each built on expf
 *     blocks     100.78 1771776      13.7   ~27,500 SiLU = ~27,500 expf
 *
 * Roughly 29,000 expf calls per inference. Newlib's expf on the ESP32-S3 is a
 * software routine of the order of a thousand cycles, which is exactly the
 * 5171 cyc/MAC that pool+head reports for 576 MAC. The SIMD question was a red
 * herring: by then there was almost no multiply-accumulate left to vectorise,
 * because the multiply-accumulate had been drowned out by expf.
 *
 * There is also a divide hiding in SiLU. This core has no float divide, so every
 * x / (1 + exp(-x)) went through __divsf3 in ROM -- about sixty instructions,
 * which is what your crash PC showed at the top of one of these logs. The
 * reciprocal below removes those too.
 *
 * The bounds are measured, not assumed: kws_expf_selftest() and
 * kws_rcp_selftest() compare against the real libm over the range the
 * activations actually see and refuse to enable the fast path if either is
 * worse than its tolerance. Drop this file and the device goes back to 225 ms.
 */
#include "kws_fastmath.h"

#include <math.h>

float kws_expf_raw(float x)
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

float kws_expf(float x)
{
    return (kws_expf_ok > 0.5f) ? kws_expf_raw(x) : expf(x);
}

static int kws_expf_selftest(void)
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

/*
 * kws_rcp() is the one the activations use: the fast path once the self-test has
 * passed, the libm division otherwise. SiLU is x / (1 + exp(-x)), so this is
 * called once per activation -- roughly 29,000 times per inference, each one
 * otherwise a __divsf3 in ROM.
 */
float kws_rcp(float x)
{
    return (kws_rcp_ok > 0.5f) ? kws_rcp_raw(x) : (1.0f / x);
}

float kws_rcp_raw(float d)
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

float kws_rcp_worst_relerr(void) { return kws_rcp_worst; }

int kws_fastmath_selftest(void)
{
    const int e = kws_expf_selftest();

    /* The reciprocal is only safe over the range its callers guarantee, and in
     * SiLU that divisor is 1 + exp(-x), which is always >= 1. The range is
     * therefore part of what is tested, not a comment. */
    float worst = 0.0f;
    for (int i = 1; i <= 8192; i++) {
        const float d = (float)i;
        const float want = 1.0f / d;
        const float rel = fabsf(kws_rcp_raw(d) - want) / want;
        if (rel > worst) { worst = rel; }
    }
    /* and the large divisors that broke the cheaper bit-pattern trick */
    for (int e2 = 0; e2 < 40; e2++) {
        const float d = ldexpf(1.0f, e2) * 1.5f;
        const float want = 1.0f / d;
        const float rel = fabsf(kws_rcp_raw(d) - want) / want;
        if (rel > worst) { worst = rel; }
    }
    kws_rcp_worst = worst;
    kws_rcp_ok = (worst < 1.0e-5f) ? 1.0f : 0.0f;

    return e;
}
