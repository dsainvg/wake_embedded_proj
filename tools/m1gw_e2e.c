/*
 * End-to-end host test: does the detector actually FIRE?
 *
 * The parity harness proves main/kws_m1gw.c matches a float reference. It does
 * not prove the thing that matters in production, which is that the deployed
 * threshold is crossed by the keyword and not by anything else. That needs real
 * audio through the real front end, and the only keyword audio in reach is
 * synthesised.
 *
 * This runs the FULL pipeline off the board -- the firmware's own
 * kws_frontend.c, not a reimplementation -- over a PCM file, one hop at a time,
 * exactly as kws_task does:
 *
 *     push(hop) -> compute(spec) -> kws_m1gw_run(spec)
 *
 * It prints the peak probability per window and how many windows crossed
 * CONFIG_EXAMPLE_THRESHOLD. Silence, noise and a tone should stay far below;
 * "amaze" should go over.
 *
 * Usage: m1gw_e2e.exe <pcm_s16le_16k.bin> [threshold]
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../main/kws_frontend.h"
#include "../main/kws_m1gw.h"

#define HOP 3200
#define REFRACTORY_MS 1500

/* Passed on the command line so the harness tracks sdkconfig rather than
 * hard-coding a second copy of the operating point. */
static float DECAY = 0.85f;
static int NEED = 3;            /* 200 ms, the firmware's hop */

static float s_spec[KWS_FRAMES * KWS_MELS];

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: m1gw_e2e <pcm_s16le_16k.bin> [threshold]\n");
        return 2;
    }
    float thr = (argc > 2) ? (float)atof(argv[2]) : kws_m1gw_threshold();
    if (argc > 3) NEED = atoi(argv[3]);
    if (argc > 4) DECAY = (float)atof(argv[4]);

    FILE *f = fopen(argv[1], "rb");
    if (!f) { printf("cannot open %s\n", argv[1]); return 2; }
    static int16_t pcm[1 << 20];
    size_t got = fread(pcm, sizeof(int16_t), sizeof(pcm) / sizeof(int16_t), f);
    fclose(f);
    printf("audio      %s: %.2f s @ 16 kHz (%u samples)\n", argv[1],
           got / 16000.0, (unsigned)got);
    printf("threshold  %.6f\n", (double)thr);

    kws_frontend_init();

    /* Mirror the firmware's peak-hold exactly (main/wake_word_main.c):
     *
     *     hold = max(p, hold * decay)
     *     if (hold >= threshold) { if (++hits >= NEED && lockout == 0) fire }
     *     else hits = 0
     *
     * Reporting raw per-window scores instead would measure a different machine
     * than the one that ships. NEED is a real part of the detector, and at
     * NEED=3 it is exactly what separates a sustained keyword from one peak.
     */
    const int hop_ms = HOP * 1000 / 16000;
    const int lockout_steps = (REFRACTORY_MS / hop_ms) > 0 ? (REFRACTORY_MS / hop_ms) : 1;

    int windows = 0, over = 0, fired = 0;
    float peak = 0.0f, hold = 0.0f;
    double sum = 0.0;
    int first_fire = -1;
    int hits = 0, lockout = 0;

    for (size_t off = 0; off + HOP <= (size_t)got; off += HOP) {
        kws_frontend_push(pcm + off, HOP);
        if (!kws_frontend_compute(s_spec)) continue;
        windows++;
        const float p = kws_m1gw_run(s_spec);
        sum += p;
        if (p > peak) peak = p;

        hold = (p > hold * DECAY) ? p : hold * DECAY;
        if (hold >= thr) {
            over++;
            if (++hits >= NEED && lockout == 0) {
                if (!fired) first_fire = (int)((off / 16000.0) * hop_ms / 1000);
                fired++;
                hits = 0;
                hold = 0.0f;
                lockout = lockout_steps;
            }
        } else {
            hits = 0;
        }
        if (lockout > 0) lockout--;
    }

    printf("windows    %d  (hop %d ms, need %d, decay %.2f, refractory %d ms)\n",
           windows, hop_ms, NEED, (double)DECAY, lockout_steps * hop_ms);
    printf("mean P     %.6f\n", windows ? sum / windows : 0.0);
    printf("peak P     %.6f\n", (double)peak);
    printf("over thr   %d/%d window(s)\n", over, windows);
    printf("CONFIRMED  %d detection(s)%s\n", fired,
           fired ? "" : "   <-- NEVER CONFIRMED");
    if (fired) printf("first fire at %.2f s\n", first_fire * hop_ms / 1000.0);

    printf("\nstage timings (host clock, diagnostic only):\n");
    for (int s = 0; s < KWS_M1GW_STAGE_COUNT; s++) {
        printf("  %-12s %8lu\n", kws_m1gw_stage_name(s),
               (unsigned long)kws_m1gw_stage_us(s) / (unsigned long)(windows ? windows : 1));
    }
    return fired ? 0 : 1;
}