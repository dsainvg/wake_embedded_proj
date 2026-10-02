/*
 * Log-mel front end for the Amaze keyword spotter.
 *
 * Numerically mirrors models/features.py (AudioFeatureExtractor):
 *
 *   sample rate   16000 Hz
 *   n_fft         512   (480-sample window, zero-padded by 32)
 *   win_length    480   Hanning with numpy's np.hanning denominator (M-1)
 *   hop           320   (20 ms)
 *   n_mels        40    triangular, 100 Hz .. 7500 Hz
 *   compression   log(mel + 1e-5)
 *   output        49 frames x 40 mels per 1.000 s window
 *
 * Two things make this cheap enough to sit inside a 10% CPU budget:
 *
 * 1. The mel filterbank is stored SPARSE. A triangular filter touches only
 *    ~5-7 FFT bins, so a dense 257x40 multiply would be 10280 MACs per frame
 *    when roughly 240 are actually nonzero -- a ~40x reduction.
 *
 * 2. Frames are reused across updates. Walking the window by HOP_WALK samples
 *    shifts the feature grid by exactly HOP_WALK/320 frame slots, so every
 *    frame before the last few is bit-identical to the previous update. A
 *    200 ms walk therefore costs 10 FFTs instead of 49.
 */

#include "kws_frontend.h"

#ifndef KWS_HOST_TEST
#include <esp_attr.h>
#endif
/* On the host there is no PSRAM section; the buffers are ordinary .bss. */
#ifndef EXT_RAM_BSS_ATTR
#define EXT_RAM_BSS_ATTR
#endif
#include <math.h>
#include <string.h>

/* Window and sparse mel filterbank, generated from models/features.py. */
#include "kws_frontend_data.h"

/* ------------------------------------------------------------------ */
/* 512-point real FFT, built on a 256-point complex FFT                */
/* ------------------------------------------------------------------ */

#define M        (KWS_NFFT / 2)          /* 256 */
#define LOG2M    8

static float   s_tw_re[M / 2];           /* 256-point transform twiddles  */
static float   s_tw_im[M / 2];
static float   s_tw2_re[M + 1];          /* 512-point twist, k = 0..M     */
static float   s_tw2_im[M + 1];
static uint8_t s_bitrev[M];
static bool    s_fft_ready = false;

static void fft_init(void)
{
    if (s_fft_ready) {
        return;
    }
    /* Forward transform sign convention (exp(-2*pi*i*k/N)), matching numpy. */
    for (int k = 0; k < M / 2; k++) {
        s_tw_re[k] = cosf(-2.0f * (float)M_PI * (float)k / (float)M);
        s_tw_im[k] = sinf(-2.0f * (float)M_PI * (float)k / (float)M);
    }
    for (int k = 0; k <= M; k++) {
        s_tw2_re[k] = cosf(-2.0f * (float)M_PI * (float)k / (float)KWS_NFFT);
        s_tw2_im[k] = sinf(-2.0f * (float)M_PI * (float)k / (float)KWS_NFFT);
    }
    for (int i = 0; i < M; i++) {
        unsigned r = 0;
        for (int b = 0; b < LOG2M; b++) {
            if (i & (1u << b)) {
                r |= 1u << (LOG2M - 1 - b);
            }
        }
        s_bitrev[i] = (uint8_t)r;
    }
    s_fft_ready = true;
}

/* In-place radix-2 iterative 256-point complex FFT. re/im length M. */
static void fft256(float *re, float *im)
{
    for (int i = 0; i < M; i++) {
        const int j = s_bitrev[i];
        if (j > i) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= M; len <<= 1) {
        const int half = len >> 1;
        const int step = M / len;
        for (int i = 0; i < M; i += len) {
            for (int j = 0; j < half; j++) {
                const int k = j * step;
                const float wr = s_tw_re[k];
                const float wi = s_tw_im[k];
                const int a = i + j;
                const int b = a + half;
                const float xr = re[b] * wr - im[b] * wi;
                const float xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
            }
        }
    }
}

EXT_RAM_BSS_ATTR static float s_pre[M];
EXT_RAM_BSS_ATTR static float s_pim[M];

/*
 * Real 512-point FFT of a real signal -> |X[k]|^2 for k = 0..256 (257 bins).
 *
 * The even and odd samples are packed into one M-point complex transform, then
 * unpacked with
 *
 *     E[k] = (Z[k] + conj(Z[M-k])) / 2
 *     O[k] = (Z[k] - conj(Z[M-k])) / (2i)
 *     X[k] = E[k] + exp(-2*pi*i*k/512) * O[k]        k = 0..M
 *
 * k = 0 and k = M fall out of the same formula (the twist is 1 and -1
 * respectively), so there is no special case.
 */
static void real_fft512(const float *x, float *power)
{
    for (int k = 0; k < M; k++) {
        s_pre[k] = x[2 * k];
        s_pim[k] = x[2 * k + 1];
    }
    fft256(s_pre, s_pim);

    for (int k = 0; k <= M; k++) {
        const int ki = (k == M) ? 0 : k;
        const int mk = (M - k) % M;

        const float zr = s_pre[ki], zi = s_pim[ki];
        const float cr = s_pre[mk], ci = -s_pim[mk];   /* conj(Z[M-k]) */

        const float er = 0.5f * (zr + cr);
        const float ei = 0.5f * (zi + ci);
        const float dr = 0.5f * (zr - cr);
        const float di = 0.5f * (zi - ci);

        /* O = d / (2i) = d * (-i/2)  ->  (re, im) = (di, -dr) */
        const float orr = di;
        const float oii = -dr;

        const float twr = s_tw2_re[k];
        const float twi = s_tw2_im[k];

        const float xr = er + (orr * twr - oii * twi);
        const float xi = ei + (orr * twi + oii * twr);
        power[k] = xr * xr + xi * xi;
    }
}

/* ------------------------------------------------------------------ */
/* Front end state                                                    */
/* ------------------------------------------------------------------ */

EXT_RAM_BSS_ATTR static float s_mel_frames[KWS_FRAMES][KWS_MELS];
EXT_RAM_BSS_ATTR static float s_frame[KWS_NFFT];
EXT_RAM_BSS_ATTR static float s_power[KWS_NFFT / 2 + 1];

/* Sliding window: KWS_AUDIO_SAMPLES retained, KWS_MAX_HOP of append room. */
EXT_RAM_BSS_ATTR static int16_t s_audio[KWS_AUDIO_SAMPLES + KWS_MAX_HOP];
static int      s_filled = 0;       /* valid samples in s_audio            */
static int      s_pending = 0;      /* pushed since the last compute       */
static bool     s_primed = false;

bool kws_frontend_init(void)
{
    fft_init();
    memset(s_mel_frames, 0, sizeof(s_mel_frames));
    memset(s_audio, 0, sizeof(s_audio));
    s_filled = 0;
    s_pending = 0;
    s_primed = false;
    return true;
}

void kws_frontend_reset(void)
{
    memset(s_mel_frames, 0, sizeof(s_mel_frames));
    memset(s_audio, 0, sizeof(s_audio));
    s_filled = 0;
    s_pending = 0;
    s_primed = false;
}

bool kws_frontend_primed(void)
{
    return s_primed;
}

/* Append n new samples to the sliding window, discarding the oldest. */
void kws_frontend_push(const int16_t *samples, int n)
{
    if (n <= 0 || n > KWS_MAX_HOP) {
        return;
    }
    memmove(s_audio, s_audio + n, KWS_AUDIO_SAMPLES * sizeof(int16_t));
    memcpy(s_audio + KWS_AUDIO_SAMPLES, samples, n * sizeof(int16_t));

    s_filled += n;
    s_pending += n;
    if (s_filled > KWS_AUDIO_SAMPLES) {
        s_filled = KWS_AUDIO_SAMPLES;
    }
}

/*
 * Emit the (49, 40) log-mel for the current window.
 *
 * Returns false until the window has been primed with KWS_AUDIO_SAMPLES.
 *
 * Frame t occupies samples [t*320, t*320 + 480) from the start of the window,
 * matching compute_spectrogram after it pads the window to exactly 16000
 * samples.
 *
 * Reuse: after the window advanced by n samples, new frame t is the frame that
 * used to sit at t + n/320, so the first (KWS_FRAMES - n/320) frames are
 * unchanged and only the rest need recomputing. If n is not a whole number of
 * feature hops the frame grid no longer aligns, so every frame is recomputed.
 */
bool kws_frontend_compute(float *out)
{
    if (s_filled < KWS_AUDIO_SAMPLES) {
        return false;
    }

    int keep = 0;
    if (s_primed && s_pending > 0 && (s_pending % KWS_HOP) == 0) {
        keep = KWS_FRAMES - (s_pending / KWS_HOP);
        if (keep < 0) {
            keep = 0;
        }
        if (keep > KWS_FRAMES) {
            keep = KWS_FRAMES;
        }
    }

    const int shift = KWS_FRAMES - keep;

    if (keep > 0) {
        memmove(s_mel_frames[0], s_mel_frames[shift],
                keep * KWS_MELS * sizeof(float));
    }

    for (int t = shift; t < KWS_FRAMES; t++) {
        const int start = t * KWS_HOP;

        memset(s_frame, 0, sizeof(s_frame));
        for (int i = 0; i < KWS_WIN; i++) {
            s_frame[i] = (float)s_audio[start + i] * kws_window[i];
        }

        real_fft512(s_frame, s_power);

        for (int m = 0; m < KWS_MELS; m++) {
            const float *w = &kws_mel_weights[kws_mel_spans[m].start];
            const int16_t sbin = kws_mel_spans[m].start;
            const int16_t n = kws_mel_spans[m].n;
            float acc = 0.0f;
            for (int i = 0; i < n; i++) {
                acc += w[i] * s_power[sbin + i];
            }
            s_mel_frames[t][m] = logf(acc + 1e-5f);
        }
    }

    s_pending = 0;
    s_primed = true;
    memcpy(out, s_mel_frames, sizeof(s_mel_frames));
    return true;
}