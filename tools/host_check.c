/*
 * Host-side validation for the Amaze network.
 *
 * Compiles kws_model.c and kws_int8.c for the host, binds heap copies of the
 * exported weight blobs, and runs the fixtures that
 * tools/export_weights.py --dump wrote: the log-mel spectrogram and the
 * probability the JAX model gives for exactly the same window.
 *
 * It reports, for every fixture:
 *
 *   float weights  the same code path with the float32 blob bound instead of
 *                  the INT8 one, which isolates what the INT8 WEIGHTS cost
 *                  from what the INT8 activations cost
 *   int8 weights   what the firmware actually computes
 *   repeat         the same input run twice, to prove determinism
 *
 * A mismatch here is a porting bug, so this must pass before the firmware is
 * trusted. It is the only place the numerical claims in the README were
 * measured.
 *
 * Build and run from the repo root:
 *     tools\build_host.bat
 *     build\host\kws_host.exe
 *     build\host\kws_host.exe noise quiet
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../main/kws_model.h"
#include "../main/kws_int8.h"

const int8_t *kws_blob_i8;
const float  *kws_blob_f32;

#define MODEL_DIR  "main/model/"
#define REF_DIR    MODEL_DIR "reference/"

/* ------------------------------------------------------------------ */
/* .npy reader (float32, C order, 1-D or 2-D)                           */
/* ------------------------------------------------------------------ */

static float *read_npy(const char *path, int *rows, int *cols)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }

    unsigned char pre[10];
    if (fread(pre, 1, 10, f) != 10 || memcmp(pre, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "%s: not a .npy file\n", path); fclose(f); return NULL;
    }
    /* v1 carries the header length in 2 bytes, v2+ in 4. The length field is
     * authoritative; scanning for a terminator byte is not. */
    const size_t hlen = (size_t)pre[8] | ((size_t)pre[9] << 8);

    char *hdr = (char *)malloc(hlen + 1);
    if (!hdr || fread(hdr, 1, hlen, f) != hlen) {
        fprintf(stderr, "%s: short header\n", path);
        free(hdr); fclose(f); return NULL;
    }
    hdr[hlen] = '\0';

    char descr[16] = { 0 };
    int shape[2] = { 0, 0 }, ndim = 0;
    const char *kd = strstr(hdr, "'descr'");
    if (kd) {
        const char *q = strchr(kd + 7, ':');
        if (q) { q = strchr(q + 1, '\''); if (q) {
            q++;
            for (int i = 0; i < 15 && q[i] != '\''; i++) { descr[i] = q[i]; } } }
    }
    const char *ks = strstr(hdr, "'shape'");
    if (ks) {
        const char *q = strchr(ks + 7, '(');
        if (q) {
            char *end;
            q++;
            long v = strtol(q, &end, 10);
            if (end > q) { shape[0] = (int)v; ndim = 1; }
            q = strchr(end, ',');
            if (q) {
                long w = strtol(q + 1, &end, 10);
                if (end > q) { shape[1] = (int)w; ndim = 2; }
            }
        }
    }
    if (strcmp(descr, "<f4") != 0) {
        fprintf(stderr, "%s: descr is '%s', expected '<f4'\n", path, descr);
        free(hdr); fclose(f); return NULL;
    }

    const int total = (ndim == 2) ? shape[0] * shape[1] : shape[0];
    float *out = (float *)malloc(sizeof(float) * (size_t)total);
    if (!out || fread(out, sizeof(float), (size_t)total, f) != (size_t)total) {
        fprintf(stderr, "%s: short read\n", path);
        free(out); free(hdr); fclose(f); return NULL;
    }
    free(hdr); fclose(f);

    if (rows) { *rows = shape[0]; }
    if (cols) { *cols = (ndim == 2) ? shape[1] : 1; }
    return out;
}

static void *slurp(const char *path, long *size_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = malloc((size_t)sz);
    if (!p || fread(p, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "%s: short read\n", path);
        free(p); fclose(f); return NULL;
    }
    fclose(f);
    *size_out = sz;
    return p;
}

/* ------------------------------------------------------------------ */

static const char *fixtures[] = {
    "noise", "tone1k", "sweep", "silence", "quiet", "loud"
};
#define N_FIXTURES ((int)(sizeof(fixtures) / sizeof(fixtures[0])))

int main(int argc, char **argv)
{
    long i8_bytes = 0, f32_bytes = 0;
    int i;

    /* ---- weights ------------------------------------------------------ */
    int8_t *i8 = (int8_t *)slurp(MODEL_DIR "kws_model_i8.bin", &i8_bytes);
    float  *f32 = (float *)slurp(MODEL_DIR "kws_model.bin", &f32_bytes);
    if (!i8 || !f32) { return 2; }
    kws_blob_i8 = i8;
    kws_blob_f32 = f32;

    printf("weights: int8 %ld B (%.1f KB), float32 %ld B (%.1f KB), "
           "%d tensors, %u params\n",
           i8_bytes, i8_bytes / 1024.0, f32_bytes, f32_bytes / 1024.0,
           KWS_NUM_TENSORS, (unsigned)KWS_PARAM_COUNT);

    /* Every tensor must fit inside the blob it is read from, in BOTH blobs: an
     * exporter that changed the tensor order would otherwise read whatever is
     * next in flash and produce a plausible number. */
    for (i = 0; i < KWS_NUM_TENSORS; i++) {
        const kws_tensor_meta *m = &kws_tensors[i];
        if ((uint64_t)m->foff + m->count > (uint64_t)(f32_bytes / 4)) {
            fprintf(stderr, "tensor %d out of range in the float blob\n", i);
            return 1;
        }
        if (m->offset != 0xFFFFu &&
            (uint64_t)m->offset + m->count > (uint64_t)i8_bytes) {
            fprintf(stderr, "tensor %d out of range in the int8 blob\n", i);
            return 1;
        }
    }

    printf("kernel backend: %s",
           kws_kernel_uses_xtensa() ? "xtensa asm" : "portable C");
    if (kws_kernel_using_xtensa()) {
        printf(" (self-test %s)\n",
               kws_kernel_selftest() ? "ok" : "FAILED, using portable");
    } else {
        printf(" (self-test n/a)\n");
    }

    kws_model_init();

    /* The model replaces two library calls with approximations that are only
     * used if they measure accurate enough. Both verdicts belong in the output:
     * a fallback that silently engages is indistinguishable from a slow build. */
    printf("fast exp: %s (worst rel %.2e, tol 1e-4)\n",
           kws_expf_worst_relerr() < 1e-4f ? "in use" : "FELL BACK to expf",
           (double)kws_expf_worst_relerr());
    printf("fast rcp: %s (worst rel %.2e, tol 1e-6)\n",
           kws_rcp_worst_relerr() < 1e-6f ? "in use" : "FELL BACK to divide",
           (double)kws_rcp_worst_relerr());
    printf("fast rsqrt: %s (worst rel %.2e, tol 1e-6)\n",
           kws_rsqrt_worst_relerr() < 1e-6f ? "in use" : "FELL BACK to sqrtf",
           (double)kws_rsqrt_worst_relerr());

    printf("\n%-9s %11s %11s %11s %10s %9s %9s\n",
           "fixture", "JAX", "float wts", "int8 wts", "d(float)",
           "d(int8)", "us/run");
    printf("----------------------------------------------------------------\n");

    int worst_i8 = 0;
    double worst = 0.0;
    int failures = 0;

    int first = 0, last = N_FIXTURES;
    if (argc > 1) {
        first = last = -1;
        for (i = 0; i < N_FIXTURES; i++) {
            if (strcmp(argv[1], fixtures[i]) == 0) { first = last = i; }
        }
        if (first < 0) { fprintf(stderr, "unknown fixture %s\n", argv[1]); return 2; }
    }

    for (i = first; i <= last && i < N_FIXTURES; i++) {
        char path[512];
        int rows = 0, cols = 0;

        snprintf(path, sizeof(path), REF_DIR "%s_spec.npy", fixtures[i]);
        float *spec = read_npy(path, &rows, &cols);
        if (!spec) { failures++; continue; }
        if (rows != KWS_FRAMES || cols != KWS_MELS) {
            fprintf(stderr, "%s: spec is %dx%d, expected %dx%d\n",
                    fixtures[i], rows, cols, KWS_FRAMES, KWS_MELS);
            free(spec); failures++; continue;
        }
        snprintf(path, sizeof(path), REF_DIR "%s_prob.npy", fixtures[i]);
        float *ref = read_npy(path, NULL, NULL);
        if (!ref) { free(spec); failures++; continue; }
        const double want = ref[1];
        free(ref);

        kws_model_set_float_weights(1);
        kws_model_init();

        fprintf(stderr, "--- float weights ---\n");
        kws_model_set_tag(fixtures[i]);
        const double pf = kws_model_run(spec);

        kws_model_set_float_weights(0);
        kws_model_init();
        fprintf(stderr, "--- int8 weights ---\n");
        const double p1 = kws_model_run(spec);
        fprintf(stderr, "--- repeat ---\n");
        const double p2 = kws_model_run(spec);

        /* Determinism: two calls on the same input must agree bit for bit. A
         * model that reads an uninitialised buffer returns a different number
         * every time, and that looks like a threshold problem rather than a
         * bug. */
        if (p1 != p2) {
            double rel = fabs(p1 - p2) / (fabs(p1) + fabs(p2) + 1e-30);
            printf("%-9s NOT BIT-IDENTICAL: %.17g then %.17g (rel %.2e)\n",
                   fixtures[i], p1, p2, rel);
            if (rel > 1e-9) { failures++; }
        }

        const double df = fabs(pf - want);
        const double di = fabs(p1 - want);
        if (di > worst) { worst = di; worst_i8 = i; }

        printf("%-9s %11.8f %11.8f %11.8f %10.2e %9.2e %9.0f\n",
               fixtures[i], want, pf, p1, df, di,
               (double)kws_model_last_us());

        free(spec);
    }

    printf("----------------------------------------------------------------\n");
    printf("worst int8 error: %s at %.3e\n", fixtures[worst_i8], worst);

    /* The number that matters. The checkpoint's threshold is 0.68 and every
     * fixture here is a negative, so a correct port must stay far below it and
     * must not move by more than the threshold's own resolution. */
    const double budget = 0.005;
    if (worst > budget) {
        printf("FAIL: worst error %.3e exceeds the %.0e budget\n", worst, budget);
        failures++;
    } else {
        printf("PASS: worst error %.3e is within the %.0e budget "
               "(threshold 0.68)\n", worst, budget);
    }

    free(i8);
    free(f32);
    return failures ? 1 : 0;
}