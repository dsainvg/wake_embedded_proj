/*
 * Host parity check for the m1_g_wide int8 kernel.
 *
 * Runs main/kws_m1gw.c -- the same file the firmware builds, not a copy -- on a
 * log-mel fixture and prints P(keyword) for tools/m1gw_reference.py to compare
 * against. Compiled with -DKWS_HOST_TEST, which shims esp_timer and nothing else.
 *
 * What this establishes and what it does not:
 *
 *   ESTABLISHES that the C kernel computes the same function as the numpy
 *   reference built from the same checkpoint: no transposed weight, no wrong
 *   stride, no padding convention that disagrees with the trainer, no
 *   per-channel-vs-per-group normalisation mistake. Those errors produce a
 *   plausible-looking probability and nothing else.
 *
 *   DOES NOT establish that the quantised model detects the keyword. The
 *   reference is float32; agreement here means the port is right up to int8
 *   rounding, and it says nothing about whether the trained model's TPR survives
 *   quantisation. That needs real keyword audio and an on-device sweep.
 *
 * Usage:
 *   build\host\m1gw_host.exe build\host\m1gw\spec.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../main/kws_m1gw.h"
#include "kws_m1gw_data.h"   /* generated, into main/model/ */

#ifdef KWS_HOST_TEST
void kws_m1gw_dbg_s2(const int8_t **q, const float **sc, int *n);
void kws_m1gw_dbg_s1(const int8_t **q, const float **sc, int *n);
void kws_m1gw_dbg_dn(const int8_t **q, const float **sc, int *n);
void kws_m1gw_dbg_dwf(const float **v, int *n);
void kws_m1gw_dbg_snapq(const int8_t **q, float *sc, int *n);
void kws_m1gw_dbg_melseq(const float **v, int *n);
void kws_m1gw_dbg_seq(const float **v, int *n);
void kws_m1gw_dbg_pool(const float **v, int *n);
#endif

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: m1gw_host <spec.bin> [more.bin ...]\n");
        return 2;
    }

    printf("model      %s\n", KWS_M1GW_TENSOR_COUNT ? "m1_g_wide" : "?");
    printf("params     %u\n", (unsigned)KWS_M1GW_PARAM_COUNT);
    printf("scales     %u\n", (unsigned)KWS_M1GW_SCALE_COUNT);
    printf("threshold  %.6f\n", (double)KWS_M1GW_THRESHOLD);
    printf("arena      %u B (%.1f KB, %.1f%% of 256 KB)\n",
           (unsigned)kws_m1gw_ram_bytes(),
           kws_m1gw_ram_bytes() / 1024.0,
           kws_m1gw_ram_bytes() / 262144.0 * 100.0);
    printf("shapes     T2=%d M1=%d M2=%d\n",
           KWS_M1GW_T2, KWS_M1GW_M1, KWS_M1GW_M2);

    enum { NSPEC = KWS_M1GW_T_IN * KWS_M1GW_M_IN };
    static float spec[NSPEC];

    int rc = 0;
    for (int a = 1; a < argc; a++) {
        FILE *f = fopen(argv[a], "rb");
        if (!f) { printf("cannot open %s\n", argv[a]); return 2; }
        size_t got = fread(spec, sizeof(float), NSPEC, f);
        fclose(f);
        if (got != (unsigned long)NSPEC) {
            printf("%-40s SHORT READ %u of %d\n", argv[a], (unsigned)got, NSPEC);
            rc = 1;
            continue;
        }

        const float p = kws_m1gw_run(spec);

        /* A NaN here would read as a probability and sail past every threshold
         * comparison, because NaN compares false against everything. A detector
         * that emits NaN is worse than one that emits zero, so it is checked
         * explicitly rather than left to show up as a missing detection. */
        if (!isfinite(p)) {
            printf("%-40s NON-FINITE P=%g\n", argv[a], (double)p);
            rc = 1;
            continue;
        }
        printf("%-40s P=%.9f\n", argv[a], (double)p);

        /* Dump the stem output and the pooled vector so the reference can be
         * compared layer by layer rather than only at the probability. */
        if (a == argc - 1) {
            FILE *f;
            const int8_t *q; const float *sc; int n, ch;
            const float *dwf; int dn_;
            kws_m1gw_dbg_dwf(&dwf, &dn_);
            f = fopen("build/host/m1gw/c_dwf.bin", "wb");
            if (f) { fwrite(dwf, 4, dn_, f); fclose(f); }
            const int8_t *sqq; float sqs; int sqn;
            kws_m1gw_dbg_snapq(&sqq, &sqs, &sqn);
            f = fopen("build/host/m1gw/c_snapq.bin", "wb");
            if (f) {
                for (int i = 0; i < sqn; i++) { float v = (float)sqq[i] * sqs; fwrite(&v, 4, 1, f); }
                fclose(f);
            }
            printf("DBG snap_scale=%g\n", (double)sqs);
            kws_m1gw_dbg_s1(&q, &sc, &n); ch = KWS_M1GW_STEM;
            f = fopen("build/host/m1gw/c_s1.bin", "wb");
            if (f) {
                for (int i = 0; i < n; i++) { float v = (float)q[i] * sc[i / ch]; fwrite(&v, 4, 1, f); }
                fclose(f);
            }
            kws_m1gw_dbg_dn(&q, &sc, &n); ch = KWS_M1GW_STEM;
            f = fopen("build/host/m1gw/c_dn.bin", "wb");
            if (f) {
                for (int i = 0; i < n; i++) { float v = (float)q[i] * sc[i / ch]; fwrite(&v, 4, 1, f); }
                fclose(f);
            }
            kws_m1gw_dbg_s2(&q, &sc, &n); ch = KWS_M1GW_DIM;
            f = fopen("build/host/m1gw/c_s2.bin", "wb");
            if (f) {
                for (int i = 0; i < n; i++) { float v = (float)q[i] * sc[i / ch]; fwrite(&v, 4, 1, f); }
                fclose(f);
            }
            const float *msq; int msn;
            kws_m1gw_dbg_melseq(&msq, &msn);
            f = fopen("build/host/m1gw/c_melseq.bin", "wb");
            if (f) { fwrite(msq, 4, msn, f); fclose(f); }
            const float *sq; int sn;
            kws_m1gw_dbg_seq(&sq, &sn);
            f = fopen("build/host/m1gw/c_seq.bin", "wb");
            if (f) { fwrite(sq, 4, sn, f); fclose(f); }
            const float *pl; int pn;
            kws_m1gw_dbg_pool(&pl, &pn);
            f = fopen("build/host/m1gw/c_pool.bin", "wb");
            if (f) { fwrite(pl, 4, pn, f); fclose(f); }
        }
    }

    printf("host_us    last=%u total=%lu\n", (unsigned)kws_m1gw_last_us(),
           (unsigned long)kws_m1gw_total_us());
    for (int s = 0; s < KWS_M1GW_STAGE_COUNT; s++) {
        printf("  stage %-12s %10lu us\n", kws_m1gw_stage_name(s),
               (unsigned long)kws_m1gw_stage_us(s));
    }
    return rc;
}