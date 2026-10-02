/*
 * Host-side validation for the Amaze front end and network.
 *
 * Compiles kws_frontend.c and kws_model.c for the host, feeds them the golden
 * tensors produced by tools/export_weights.py --dump, and compares against the
 * JAX reference. Any mismatch here is a porting bug, so this must pass before
 * the firmware is trusted.
 *
 * Build and run from the repo root:
 *     python tools/export_weights.py --dump
 *     tools\test_kws_host.sh
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../main/kws_frontend.h"
#include "../main/kws_model.h"

/* Bound to a heap copy of the weights by main() before inference runs. */
const float *kws_blob;

/* ---- minimal .npy reader (float32, C order, 1-D or 2-D) -------------- */

static void *read_npy(const char *path, int *rows, int *cols)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }

    /* .npy v1/v2: magic(6) version(2) HEADER_LEN(2, little endian) then the
     * ASCII dict, then the raw array. Scanning for a terminator byte is wrong;
     * the length field is authoritative. */
    unsigned char pre[10];
    if (fread(pre, 1, 10, f) != 10 ||
        memcmp(pre, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "%s: not a .npy file\n", path);
        fclose(f);
        return NULL;
    }
    const size_t hlen = (size_t)pre[8] | ((size_t)pre[9] << 8);

    char *hdr = (char *)malloc(hlen + 1);
    if (!hdr || fread(hdr, 1, hlen, f) != hlen) {
        fprintf(stderr, "%s: short header\n", path);
        free(hdr);
        fclose(f);
        return NULL;
    }
    hdr[hlen] = '\0';

    /* parse descr/fortran_order/shape out of the ASCII header */
    char descr[16] = {0};
    int fortran = 0;
    int shape[2] = {0, 0};
    int ndim = 0;
/* The dict looks like {'descr': '<f4', 'fortran_order': False,
     * 'shape': (49, 40), }. Matching the KEY must not be mistaken for the
     * VALUE, so skip past the key's closing quote and the colon first. */
    const char *k_descr = strstr(hdr, "'descr'");
    const char *k_fort  = strstr(hdr, "'fortran_order'");
    const char *k_shape = strstr(hdr, "'shape'");

    if (k_descr) {
        const char *q = strchr(k_descr + 7, ':');
        if (q) {
            q = strchr(q + 1, '\'');
            if (q) {
                q++;
                for (int k = 0; k < 15 && q[k] != '\''; k++) { descr[k] = q[k]; }
            }
        }
    }
    if (k_fort) {
        const char *q = strchr(k_fort + 14, ':');
        if (q) { fortran = (q[2] == 'T'); }
    }
    if (k_shape) {
        const char *q = strchr(k_shape + 7, '(');
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
        fclose(f);
        return NULL;
    }
    if (fortran) {
        fprintf(stderr, "%s: fortran order not supported\n", path);
        fclose(f);
        return NULL;
    }

    const int total = (ndim == 2) ? shape[0] * shape[1] : shape[0];
    float *out = (float *)malloc(sizeof(float) * (size_t)total);
    if (!out) { fclose(f); return NULL; }
    if (fread(out, sizeof(float), (size_t)total, f) != (size_t)total) {
        fprintf(stderr, "%s: short read\n", path);
        free(out);
        fclose(f);
        return NULL;
    }
    fclose(f);

    if (rows) { *rows = (ndim == 2) ? shape[0] : shape[0]; }
    if (cols) { *cols = (ndim == 2) ? shape[1] : 1; }
    return out;
}

/* ------------------------------------------------------------------ */

static const char *weights_bin = "main/model/kws_model.bin";
static const char *ref_dir = "main/model/reference";


int main(int argc, char **argv)
{
    const char *tag = (argc > 1) ? argv[1] : "noise";

    /* --- load weights ------------------------------------------------- */
    FILE *f = fopen(weights_bin, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s (run tools/export_weights.py first)\n", weights_bin);
        return 2;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    const uint32_t n = (uint32_t)(sz / 4);

    float *heap = (float *)malloc(sizeof(float) * n);
    if (!heap) { fclose(f); return 2; }
    if (fread(heap, sizeof(float), n, f) != n) { fclose(f); free(heap); return 2; }
    fclose(f);
    printf("weights: %s  %u floats (%.1f KB)\n", weights_bin, n, sz / 1024.0);
    printf("tensors: %d\n", KWS_NUM_TENSORS);

    for (int i = 0; i < KWS_NUM_TENSORS; i++) {
        const kws_tensor_meta *m = &kws_tensors[i];
        printf("  [%2d] {%4d,%4d,%4d,%4d} count=%6u off=%6u\n",
               i, m->dim0, m->dim1, m->dim2, m->dim3, m->count, m->offset);
        if (m->offset + m->count > n) {
            fprintf(stderr, "  TENSOR %d OUT OF RANGE\n", i);
            free(heap);
            return 1;
        }
    }

    kws_blob = heap;
    kws_frontend_init();
    kws_model_init();
    kws_blob = heap;
    kws_frontend_init();
    kws_model_init();

    /* --- load reference ------------------------------------------------ */
    char path[512];
    snprintf(path, sizeof(path), "%s/%s_spec.npy", ref_dir, tag);
    int rows = 0, cols = 0;
    float *ref_spec = (float *)read_npy(path, &rows, &cols);
    if (!ref_spec) { free(heap); return 2; }
    printf("\nreference spec: %s  %d x %d\n", path, rows, cols);
    if (rows != KWS_FRAMES || cols != KWS_MELS) {
        fprintf(stderr, "unexpected spec shape\n");
        free(ref_spec);
        free(heap);
        return 1;
    }

snprintf(path, sizeof(path), "%s/%s_prob.npy", ref_dir, tag);
    float *ref_prob = (float *)read_npy(path, NULL, NULL);
    if (!ref_prob) { free(ref_spec); free(heap); return 2; }
    const float expect = ref_prob[1];

    /* Dump what the model actually receives, so a reader that parses the .npy
     * header differently from Python shows up as data rather than mystery. */
    {
        FILE *sf = fopen("main/model/reference/c_spec.bin", "wb");
        if (sf) {
            fwrite(ref_spec, sizeof(float), (size_t)rows * cols, sf);
            fclose(sf);
        }
    }
    printf("expected P(keyword) = %.8f\n", expect);
    printf("ADDR ref_spec=%p  kws_debug_buf=%p  span=%u bytes\n",
           (void *)ref_spec, (void *)kws_debug_buf,
           (unsigned)(KWS_DEBUG_MAX * sizeof(float)));

    /* Does kws_model_run write through its const spec pointer? Compare the
     * caller's buffer before and after, element by element. */
    {
        FILE *o = fopen("main/model/reference/c_spec_after.bin", "wb");
        if (o) {
            fwrite(ref_spec, sizeof(float), (size_t)rows * cols, o);
            fclose(o);
            double worst = 0.0;
            int worst_i = -1;
            for (int i = 0; i < rows * cols; i++) {
                const double d = fabs((double)ref_spec[i]);
                if (d > worst) { worst = d; worst_i = i; }
            }
            printf("spec after inference: max |value| = %.6f at index %d "
                   "(frame %d mel %d)\n",
                   worst, worst_i,
                   (worst_i >= 0 ? worst_i / cols : -1),
                   (worst_i >= 0 ? worst_i % cols : -1));
        }
    }

    /* --- run ------------------------------------------------------------ */
    float got = kws_model_run(ref_spec);
    printf("C        P(keyword) = %.8f\n", got);
    printf("abs diff           = %.3e\n", fabsf(got - expect));

    /* --- also exercise the real front end on a synthetic int16 signal --- */
    static int16_t audio[KWS_AUDIO_SAMPLES + KWS_MAX_HOP];
    for (int i = 0; i < KWS_AUDIO_SAMPLES; i++) {
        audio[i] = (int16_t)(3000 * sinf(2.0f * 3.14159f * 440.0f * i / 16000.0f));
    }
    for (int c = 0; c < 3; c++) {
        kws_frontend_reset();
        int16_t chunk[KWS_MAX_HOP];
        for (int i = 0; i < KWS_MAX_HOP; i++) { chunk[i] = audio[c * KWS_MAX_HOP + i]; }
        kws_frontend_push(chunk, KWS_MAX_HOP);
    }
    float spec[KWS_FRAMES * KWS_MELS];
    if (kws_frontend_compute(spec)) {
        double diff = 0.0;
        for (int i = 0; i < KWS_FRAMES * KWS_MELS; i++) {
            const double d = (double)spec[i] - (double)ref_spec[i];
            diff += d * d;
        }
        printf("\nfront end on a 440 Hz tone vs reference spec: rms diff = %.4e\n",
               sqrt(diff / (KWS_FRAMES * KWS_MELS)));
    } else {
        printf("\nfront end not primed (expected: the tone is not the reference)\n");
    }

free(ref_spec);
    free(ref_prob);

    /* --- isolated build_feat: the delta stack on its own, nothing else ---- */
    {
        char out[512];
        snprintf(out, sizeof(out), "%s/c_feat.bin", ref_dir);
        FILE *o = fopen(out, "wb");
        if (o) {
            static float tmp[360];
            for (int t = 0; t < KWS_FRAMES; t++) {
                kws_debug_build_feat(ref_spec, t, tmp);
                fwrite(tmp, sizeof(float), 360, o);
            }
            fclose(o);
            printf("build_feat -> %s (49 x 360)\n", out);
        }
    }

    /* --- dump every stage for the layer-by-layer bisect ------------------ */
    {
        const char *sdir = "main/model/reference/stages";
        const char *names[] = {"", "stem", "conv2", "gn3swish", "seq",
                               "block1", "block2", "block3", "feat", "logits"};
        for (int st = 10; st <= 22; st++) {
            char out[512];
            snprintf(out, sizeof(out), "%s/c_%d_site.bin", sdir, st);
            FILE *o = fopen(out, "wb");
            if (!o) break;
            {
                float before0 = ref_spec[0], before1 = ref_spec[1];
                kws_debug_stage = st;
                (void)kws_model_run(ref_spec);
                kws_debug_stage = 0;
                if (ref_spec[0] != before0 || ref_spec[1] != before1) {
                    printf("  stage %d CORRUPTED spec[0..1]: %.6g,%.6g -> %.6g,%.6g\n",
                           st, (double)before0, (double)before1,
                           (double)ref_spec[0], (double)ref_spec[1]);
                }
            }            fwrite(kws_debug_buf, sizeof(float), kws_debug_n, o);
            fclose(o);
        }
        for (int st = 1; st <= 9; st++) {
            char out[512];
            snprintf(out, sizeof(out), "%s/c_%d_%s.bin", sdir, st, names[st]);
            FILE *o = fopen(out, "wb");
            if (!o) { fprintf(stderr, "cannot write %s\n", out); break; }
            {
                float before0 = ref_spec[0], before1 = ref_spec[1];
                kws_debug_stage = st;
                (void)kws_model_run(ref_spec);
                kws_debug_stage = 0;
                if (ref_spec[0] != before0 || ref_spec[1] != before1) {
                    printf("  stage %d CORRUPTED spec[0..1]: %.6g,%.6g -> %.6g,%.6g\n",
                           st, (double)before0, (double)before1,
                           (double)ref_spec[0], (double)ref_spec[1]);
                }
            }            fwrite(kws_debug_buf, sizeof(float), kws_debug_n, o);
            fclose(o);
            printf("stage %d %-8s %6zu floats -> %s\n", st, names[st],
                   kws_debug_n, out);
        }
    }

    free(heap);
    return 0;
}

