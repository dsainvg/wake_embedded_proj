/*
 * Minimal isolation test for build_feat() -- nothing else runs.
 *
 * If the delta stack is correct here but wrong inside the full harness, then
 * something in kws_model_run is writing through a pointer it should not, and
 * this test will also catch it by dumping `spec` before and after.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../main/kws_frontend.h"
#include "../main/kws_model.h"

const float *kws_blob;

static float *read_npy(const char *path, int *rows, int *cols)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    unsigned char pre[10];
    if (fread(pre, 1, 10, f) != 10 || memcmp(pre, "\x93NUMPY", 6) != 0) {
        fclose(f); return NULL;
    }
    size_t hlen = (size_t)pre[8] | ((size_t)pre[9] << 8);
    char *hdr = (char *)malloc(hlen + 1);
    if (fread(hdr, 1, hlen, f) != hlen) { free(hdr); fclose(f); return NULL; }
    hdr[hlen] = 0;

    char descr[16] = {0};
    const char *kd = strstr(hdr, "'descr'");
    if (kd) {
        const char *q = strchr(kd + 7, ':');
        if (q) { q = strchr(q + 1, '\''); if (q) { q++;
            for (int i = 0; i < 15 && q[i] != '\''; i++) descr[i] = q[i]; } }
    }
    int shape[2] = {0, 0}, ndim = 0;
    const char *ks = strstr(hdr, "'shape'");
    if (ks) {
        const char *q = strchr(ks + 7, '(');
        if (q) {
            char *end;
            q++;
            long v = strtol(q, &end, 10);
            if (end > q) { shape[0] = (int)v; ndim = 1; }
            q = strchr(end, ',');
            if (q) { long w = strtol(q + 1, &end, 10);
                     if (end > q) { shape[1] = (int)w; ndim = 2; } }
        }
    }
    if (strcmp(descr, "<f4") != 0) { fprintf(stderr, "descr='%s'\n", descr);
                                       free(hdr); fclose(f); return NULL; }

    const int total = (ndim == 2) ? shape[0] * shape[1] : shape[0];
    float *out = (float *)malloc(sizeof(float) * total);
    if (fread(out, sizeof(float), total, f) != (size_t)total) {
        free(out); free(hdr); fclose(f); return NULL;
    }
    free(hdr); fclose(f);
    if (rows) *rows = (ndim == 2) ? shape[0] : shape[0];
    if (cols) *cols = (ndim == 2) ? shape[1] : 1;
    return out;
}

static void dump(const char *path, const float *p, size_t n)
{
    FILE *o = fopen(path, "wb");
    if (o) { fwrite(p, sizeof(float), n, o); fclose(o); }
}

int main(void)
{
    /* conv1 needs the kernel, so bind the blob first. */
    FILE *wf = fopen("main/model/kws_model.bin", "rb");
    if (!wf) { fprintf(stderr, "cannot open kws_model.bin\n"); return 2; }
    fseek(wf, 0, SEEK_END);
    long wsz = ftell(wf);
    fseek(wf, 0, SEEK_SET);
    float *blob = (float *)malloc((size_t)wsz);
    if (!blob || fread(blob, sizeof(float), (size_t)wsz / 4, wf) != (size_t)wsz / 4) {
        fprintf(stderr, "blob load failed\n"); return 2;
    }
    fclose(wf);
    kws_blob = blob;
    printf("blob %ld bytes\n", wsz);

    {
        const char *nm[16]; const void *pt[16];
        kws_debug_addrs(nm, pt, 16);
        for (int i = 0; i < 14; i++) { printf("  %-14s %p\n", nm[i], pt[i]); }
    }

    int rows = 0, cols = 0;
    float *spec = read_npy("main/model/reference/noise_spec.npy", &rows, &cols);
    if (!spec) { fprintf(stderr, "read failed\n"); return 2; }
    printf("spec %d x %d\n", rows, cols);

    dump("main/model/reference/feat_spec_before.bin", spec, (size_t)rows * cols);

    static float tmp[360];
    FILE *rowsf = fopen("main/model/reference/feat_rows.bin", "wb");
    if (!rowsf) { fprintf(stderr, "cannot write rows\n"); return 2; }
    for (int t = 0; t < KWS_FRAMES; t++) {
        kws_debug_build_feat(spec, t, tmp);
        fwrite(tmp, sizeof(float), 360, rowsf);
    }
    fclose(rowsf);

    dump("main/model/reference/feat_spec_after.bin", spec, (size_t)rows * cols);

    {
        static float crow[640];
        FILE *cf = fopen("main/model/reference/conv1_rows.bin", "wb");
        if (cf) {
            for (int t = 0; t < KWS_FRAMES; t++) {
                kws_debug_conv1(spec, t, crow);
                fwrite(crow, sizeof(float), 640, cf);
            }
            fclose(cf);
            printf("conv1 rows written\n");
        }
    }

    double d = 0.0;
    for (int i = 0; i < rows * cols; i++) {
        d += fabs(spec[i]);
    }
    printf("spec checksum after loop = %.6f\n", d);
    free(spec);
    return 0;
}