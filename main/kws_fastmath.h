#ifndef KWS_FASTMATH_H
#define KWS_FASTMATH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * exp(x) split into an integer exponent taken straight from the IEEE-754
 * exponent field and a degree-7 Taylor series in the fraction; the next term is
 * 1.5e-5, so the relative error stays under 2e-5 across the range.
 *
 * Returns the libm expf until the self-test has run and passed. That ordering
 * matters: a fast path that is enabled before it has been measured against
 * libm is a fast path nobody checked.
 */
float kws_expf_raw(float x);
float kws_expf(float x);          /* fast when enabled, else libm expf */

/*
 * Reciprocal by seed plus Newton-Raphson.
 *
 * SiLU is x / (1 + exp(-x)) and this core has NO float divide, so every
 * activation went through __divsf3 in ROM -- about sixty instructions. At the
 * ~29,000 SiLU calls an inference performs, that is not a rounding detail.
 */
float kws_rcp_raw(float x);
float kws_rcp(float x);           /* fast when enabled, else 1/x */

int   kws_fastmath_selftest(void);  /* runs both; 1 if the fast paths are usable */
float kws_expf_worst_relerr(void);
float kws_rcp_worst_relerr(void);

#ifdef __cplusplus
}
#endif

#endif /* KWS_FASTMATH_H */
