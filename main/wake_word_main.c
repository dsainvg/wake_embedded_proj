/*
 * INMP441 I2S microphone + ESP-SR WakeNet wake word detector
 *
 * Target : ESP32-S3 N16R8 (16 MB flash, 8 MB octal PSRAM)
 * Mic    : INMP441, WS=40, SCK=41, SD=42, LR=GND (left slot)
 *
 * Pipeline:  INMP441 -> I2S std RX -> 16-bit -> ESP-SR AFE -> WakeNet
 *
 * On a wake word hit the LED goes green for 3 s and the audio that FOLLOWS the
 * keyword is captured; once CONFIG_EXAMPLE_CLIP_SECONDS have been captured the
 * clip is written to the "storage" flash partition as a WAV file.
 */

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "esp_attr.h"
#include "esp_check.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_system.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "nvs_flash.h"

#include "kws_frontend.h"
#include "kws_int8.h"
#include "kws_fastmath.h"
#include "kws_m1gw.h"
#include "kws_m1gw_data.h"

#if CONFIG_EXAMPLE_LED_TYPE_WS2812
#include "led_strip.h"
#endif


static const char *TAG = "wake";

#define SAMPLE_RATE          CONFIG_EXAMPLE_SAMPLE_RATE
/* I2S DMA ring depth, in milliseconds. This -- not the hop -- is the budget an
 * inference has to fit inside: the read blocks until the hop's samples have
 * arrived, so the ring only ever holds the time spent in kws_m1gw_run. Sized in
 * i2s_init from the same numbers the driver is given. */
#define RING_DESC_NUM        16
#define RING_FRAME_NUM       1024
#define RING_SLACK_MS        ((RING_DESC_NUM * RING_FRAME_NUM * 1000) / SAMPLE_RATE)
#define CLIP_SECONDS         CONFIG_EXAMPLE_CLIP_SECONDS
#define CLIP_SAMPLES         (SAMPLE_RATE * CLIP_SECONDS)
#define PRE_ROLL_SAMPLES     (SAMPLE_RATE * CONFIG_EXAMPLE_PRE_ROLL_SECONDS)
#define WAV_HEADER_LEN       44
#define FLASH_SECTOR_SIZE    4096
#define ALIGN_UP(v, a)       (((v) + (a)-1) / (a) * (a))

/* ------------------------------------------------------------------ */
/* Globals                                                            */
/* ------------------------------------------------------------------ */


static i2s_chan_handle_t s_rx_chan = NULL;

/*
 * Monotonic count of every sample the DMA has handed to this task.
 *
 * This is the device's only trustworthy timebase for a latency measurement: it
 * is derived from the same reads the analysis consumes, so it survives a reboot
 * of the scheduler, a change of the hop, or a network stack that buffers, none
 * of which a wall clock would. Absolute device time comes from esp_timer, which
 * the capture path already stamps; this counter ties that time to a sample
 * index, and sample index to sample index is what makes the numbers comparable
 * to anything measured off the audio.
 */
static volatile int64_t s_sample_clock = 0;   /* samples delivered to kws_task */

/* Short pre-roll ring, used only to seed a clip with the tail of the keyword. */
static int16_t *s_ring = NULL;
static size_t s_ring_wpos = 0;
static size_t s_ring_filled = 0;
static SemaphoreHandle_t s_ring_lock = NULL;

/* Post-detection capture buffer. */
static int16_t *s_clip = NULL;
static size_t s_capture_len = 0;          /* valid samples in s_clip   */
static volatile bool s_capture_active = false;
static volatile int64_t s_capture_start_sample = 0;
static SemaphoreHandle_t s_capture_lock = NULL;
static SemaphoreHandle_t s_write_req = NULL;   /* given when a clip is full */

/* LED state */
typedef enum {
    LED_IDLE = 0,
    LED_DETECTED,
    LED_SAVING,
} led_state_t;

static volatile led_state_t s_led_state = LED_IDLE;
static volatile int64_t s_led_until = 0;      /* esp_timer time, us */

#if CONFIG_EXAMPLE_LED_TYPE_WS2812
static led_strip_handle_t s_led = NULL;
#endif

/* ------------------------------------------------------------------ */
/* LED                                                                */
/* ------------------------------------------------------------------ */

static void led_write(int on)
{
#if CONFIG_EXAMPLE_LED_TYPE_GPIO
    const int level = CONFIG_EXAMPLE_LED_ACTIVE_LOW ? !on : !!on;
    gpio_set_level(CONFIG_EXAMPLE_LED_GPIO, level);
#else
    static const uint32_t idle_rgb[3]   = { 2, 1, 0 };    /* dim orange */
    static const uint32_t detect_rgb[3] = { 0, 60, 0 };   /* green      */
    static const uint32_t saving_rgb[3] = { 50, 30, 0 };  /* amber      */
    const uint32_t *c = (s_led_state == LED_DETECTED) ? detect_rgb
                       : (s_led_state == LED_SAVING)  ? saving_rgb
                                                      : idle_rgb;
    uint32_t scale = on ? 1 : 0;
    led_strip_set_pixel(s_led, 0, c[0] * scale, c[1] * scale, c[2] * scale);
    led_strip_refresh(s_led);
#endif
}

static void led_init(void)
{
#if CONFIG_EXAMPLE_LED_TYPE_WS2812
    led_strip_rmt_config_t rmt_cfg = {
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .resolution_hz = 10 * 1000 * 1000,
        .flags.with_dma = false,
    };
    led_strip_config_t strip_cfg = {
        .strip_gpio_num = CONFIG_EXAMPLE_LED_GPIO,
        .max_leds = CONFIG_EXAMPLE_LED_LED_COUNT,
        .led_model = LED_MODEL_WS2812,
        .color_component_format = LED_STRIP_COLOR_COMPONENT_FMT_GRB,
        .flags.invert_out = false,
    };
    ESP_ERROR_CHECK(led_strip_new_rmt_device(&strip_cfg, &rmt_cfg, &s_led));
    ESP_LOGI(TAG, "status LED: addressable RGB on GPIO%d (%u px)",
             CONFIG_EXAMPLE_LED_GPIO, (unsigned)CONFIG_EXAMPLE_LED_LED_COUNT);
#elif CONFIG_EXAMPLE_LED_TYPE_GPIO
    gpio_config_t cfg = {
        .pin_bit_mask = 1ULL << CONFIG_EXAMPLE_LED_GPIO,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&cfg));
    ESP_LOGI(TAG, "status LED: plain GPIO%d (active %s)",
             CONFIG_EXAMPLE_LED_GPIO,
             CONFIG_EXAMPLE_LED_ACTIVE_LOW ? "low" : "high");
#endif
    led_write(0);
}

static void led_task(void *arg)
{
    int64_t next_heartbeat = esp_timer_get_time();
    int pulse = 0;

    for (;;) {
        int64_t now = esp_timer_get_time();

        switch (s_led_state) {
        case LED_DETECTED:
            /* Solid green for the full 3 s acknowledgement window. */
            if (now < s_led_until) {
                led_write(1);
            } else {
                s_led_state = LED_IDLE;
                led_write(0);
                next_heartbeat = now;
            }
            break;

        case LED_SAVING:
            /* Clip is being erased/written to flash (~10 s after the hit). */
            led_write((now / 100000) & 1);
            if (now >= s_led_until) {
                s_led_state = LED_IDLE;
                led_write(0);
                next_heartbeat = now;
            }
            break;

        case LED_IDLE:
        default:
            led_write(0);
            if (now >= next_heartbeat) {
                pulse = !pulse;
                led_write(pulse);
                vTaskDelay(pdMS_TO_TICKS(60));
                led_write(0);
                next_heartbeat = now + 3000000;    /* 3 s */
            } else {
                vTaskDelay(pdMS_TO_TICKS(20));
            }
            break;
        }

        if (s_led_state != LED_IDLE) {
            vTaskDelay(pdMS_TO_TICKS(20));
        }
    }
}

/* ------------------------------------------------------------------ */
/* Pre-roll ring + post-detection capture                             */
/* ------------------------------------------------------------------ */

static void *alloc_capture(size_t bytes)
{
    void *p = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM);
    if (p == NULL) {
        ESP_LOGW(TAG, "PSRAM alloc failed (%u bytes), using internal RAM",
                 (unsigned)bytes);
        p = heap_caps_malloc(bytes, MALLOC_CAP_DMA);
    }
    return p;
}

static void ring_push(const int16_t *src, size_t n)
{
    xSemaphoreTake(s_ring_lock, portMAX_DELAY);

    size_t first = PRE_ROLL_SAMPLES - s_ring_wpos;
    if (first > n) {
        first = n;
    }
    if (first > 0) {
        memcpy(&s_ring[s_ring_wpos], src, first * sizeof(int16_t));
    }
    if (n > first) {
        memcpy(&s_ring[0], src + first, (n - first) * sizeof(int16_t));
    }

    s_ring_wpos = (s_ring_wpos + n) % PRE_ROLL_SAMPLES;
    s_ring_filled += n;
    if (s_ring_filled > PRE_ROLL_SAMPLES) {
        s_ring_filled = PRE_ROLL_SAMPLES;
    }

    xSemaphoreGive(s_ring_lock);
}

/* Copy up to `want` most recent samples, oldest first, into dst. */
static size_t ring_peek_recent(int16_t *dst, size_t want)
{
    if (want > PRE_ROLL_SAMPLES) {
        want = PRE_ROLL_SAMPLES;
    }

    xSemaphoreTake(s_ring_lock, portMAX_DELAY);

    size_t n = (s_ring_filled < want) ? s_ring_filled : want;
    size_t start = (s_ring_wpos + PRE_ROLL_SAMPLES - n) % PRE_ROLL_SAMPLES;
    size_t first = PRE_ROLL_SAMPLES - start;
    if (first > n) {
        first = n;
    }
    memcpy(dst, &s_ring[start], first * sizeof(int16_t));
    if (n > first) {
        memcpy(dst + first, &s_ring[0], (n - first) * sizeof(int16_t));
    }

    xSemaphoreGive(s_ring_lock);
    return n;
}

/*
 * Called on an accepted detection. Seeds the clip with the tail of the keyword
 * so the saved audio actually contains it, then opens the window for the audio
 * that follows.
 *
 * The sample index the clip starts at is recorded, because it is the only way
 * to relate what was captured to when the verdict was reached. Without it the
 * saved WAV has no position in the device's audio timeline, and the offset
 * between "the keyword ended" and "the clip begins" -- which is where the
 * latency actually is -- cannot be recovered from the file after the fact.
 */
static void capture_start(int64_t detected_at_sample)
{
    xSemaphoreTake(s_capture_lock, portMAX_DELAY);

    if (s_capture_active) {
        /* A capture is already running; let it finish rather than truncating. */
        xSemaphoreGive(s_capture_lock);
        return;
    }

#if CONFIG_EXAMPLE_PRE_ROLL_SECONDS > 0
    s_capture_len = ring_peek_recent(s_clip, PRE_ROLL_SAMPLES);
#else
    s_capture_len = 0;
#endif
    s_capture_start_sample = detected_at_sample - (int64_t)s_capture_len;
    s_capture_active = true;

    xSemaphoreGive(s_capture_lock);

    ESP_LOGI(TAG, "capturing %d s following the keyword (%d s pre-roll)",
             CONFIG_EXAMPLE_CLIP_SECONDS, CONFIG_EXAMPLE_PRE_ROLL_SECONDS);
}

/* Called from the feed task for every AFE frame. */
static void capture_push(const int16_t *pcm, size_t n)
{
    if (!s_capture_active) {
        return;
    }

    xSemaphoreTake(s_capture_lock, portMAX_DELAY);

    if (s_capture_len < CLIP_SAMPLES) {
        size_t room = CLIP_SAMPLES - s_capture_len;
        size_t take = (n < room) ? n : room;
        memcpy(&s_clip[s_capture_len], pcm, take * sizeof(int16_t));
        s_capture_len += take;
    }

    bool complete = (s_capture_len >= CLIP_SAMPLES);
    if (complete) {
        s_capture_active = false;
    }

    xSemaphoreGive(s_capture_lock);

    if (complete) {
        xSemaphoreGive(s_write_req);
    }
}

/* ------------------------------------------------------------------ */
/* Flash WAV storage                                                  */
/* ------------------------------------------------------------------ */

static void wav_header(uint8_t *h, uint32_t data_bytes)
{
    const uint32_t riff_size = 36 + data_bytes;
    const uint32_t rate = SAMPLE_RATE;
    const uint16_t channels = 1;
    const uint16_t bits = 16;
    const uint16_t block_align = channels * bits / 8;
    const uint32_t byte_rate = rate * block_align;

    memcpy(h + 0,  "RIFF", 4);
    memcpy(h + 4,  &riff_size, 4);
    memcpy(h + 8,  "WAVEfmt ", 8);
    memcpy(h + 16, &((uint32_t){16}), 4);
    memcpy(h + 20, &((uint16_t){1}), 2);            /* PCM */
    memcpy(h + 22, &channels, 2);
    memcpy(h + 24, &rate, 4);
    memcpy(h + 28, &byte_rate, 4);
    memcpy(h + 32, &block_align, 2);
    memcpy(h + 34, &bits, 2);
    memcpy(h + 36, "data", 4);
    memcpy(h + 40, &data_bytes, 4);
}

static uint32_t storage_cursor_get(void)
{
    nvs_handle_t h;
    uint32_t v = 0;
    if (nvs_open("wake", NVS_READWRITE, &h) != ESP_OK) {
        return 0;
    }
    if (nvs_get_u32(h, "slot", &v) != ESP_OK) {
        v = 0;
    }
    nvs_close(h);
    return v;
}

static esp_err_t storage_cursor_set(uint32_t v)
{
    nvs_handle_t h;
    esp_err_t err = nvs_open("wake", NVS_READWRITE, &h);
    if (err != ESP_OK) {
        return err;
    }
    err = nvs_set_u32(h, "slot", v);
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    nvs_close(h);
    return err;
}

static void storage_task(void *arg)
{
    const esp_partition_t *part =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY,
                                 CONFIG_EXAMPLE_STORAGE_PARTITION);
    if (part == NULL) {
        ESP_LOGE(TAG, "partition '%s' not found - storage disabled",
                 CONFIG_EXAMPLE_STORAGE_PARTITION);
        vTaskDelete(NULL);
        return;
    }

    const size_t slot_size = ALIGN_UP(WAV_HEADER_LEN + CLIP_SAMPLES * sizeof(int16_t),
                                      FLASH_SECTOR_SIZE);
    const size_t part_size = part->size;
    const size_t slot_count = part_size / slot_size;
    const size_t clip_bytes = CLIP_SAMPLES * sizeof(int16_t);

    uint32_t slot = storage_cursor_get();
    if (slot >= slot_count) {
        slot = 0;
    }

    ESP_LOGI(TAG, "storage: partition '%s' %u KB, %u slots x %u KB",
             CONFIG_EXAMPLE_STORAGE_PARTITION,
             (unsigned)(part_size / 1024),
             (unsigned)slot_count,
             (unsigned)(slot_size / 1024));

    uint8_t header[WAV_HEADER_LEN];
    uint8_t *stage = malloc(clip_bytes);
    if (stage == NULL) {
        ESP_LOGE(TAG, "no memory for %u byte staging buffer", (unsigned)clip_bytes);
        vTaskDelete(NULL);
        return;
    }

    for (;;) {
        xSemaphoreTake(s_write_req, portMAX_DELAY);

        xSemaphoreTake(s_capture_lock, portMAX_DELAY);
        size_t n = s_capture_len;
        memcpy(stage, s_clip, n * sizeof(int16_t));
        s_capture_len = 0;
        xSemaphoreGive(s_capture_lock);

        wav_header(header, n * sizeof(int16_t));

        const size_t offset = (size_t)slot * slot_size;

        s_led_state = LED_SAVING;
        s_led_until = esp_timer_get_time() + 5000000;

        esp_err_t err = esp_partition_erase_range(part, offset, slot_size);
        if (err == ESP_OK) {
            err = esp_partition_write(part, offset, header, WAV_HEADER_LEN);
        }
        if (err == ESP_OK) {
            err = esp_partition_write(part, offset + WAV_HEADER_LEN, stage,
                                      n * sizeof(int16_t));
        }

        if (err != ESP_OK) {
            ESP_LOGE(TAG, "slot %u write failed: %s", (unsigned)slot,
                     esp_err_to_name(err));
        } else {
            ESP_LOGI(TAG, "saved %.1f s clip -> slot %u (flash 0x%06" PRIx32
                     ", %u bytes)",
                     (double)n / SAMPLE_RATE, (unsigned)slot,
                     (uint32_t)(part->address + offset),
                     (unsigned)(WAV_HEADER_LEN + n * sizeof(int16_t)));
        }

        slot = (slot + 1) % slot_count;
        storage_cursor_set(slot);

        s_led_state = LED_IDLE;
        led_write(0);
    }
}

/* ------------------------------------------------------------------ */
/* I2S + AFE feed                                                     */
/* ------------------------------------------------------------------ */

static void i2s_init(void)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = RING_DESC_NUM;
    chan_cfg.dma_frame_num = RING_FRAME_NUM;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &s_rx_chan));

    /*
     * INMP441 drives 24 bits of two's complement audio left-justified inside a
     * 32-bit I2S slot. LR is tied to GND so only the left slot carries data.
     */
    i2s_std_slot_config_t slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
        I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO);
    slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = slot_cfg,
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = CONFIG_EXAMPLE_I2S_SCK_GPIO,
            .ws   = CONFIG_EXAMPLE_I2S_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din  = CONFIG_EXAMPLE_I2S_SD_GPIO,
            .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
        },
    };

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_rx_chan, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(s_rx_chan));

    ESP_LOGI(TAG, "INMP441 on WS=%d SCK=%d SD=%d @ %d Hz, left slot",
             CONFIG_EXAMPLE_I2S_WS_GPIO, CONFIG_EXAMPLE_I2S_SCK_GPIO,
             CONFIG_EXAMPLE_I2S_SD_GPIO, SAMPLE_RATE);
    /* The DMA ring is what decouples inference from capture: it fills
     * continuously, so stalling in kws_m1gw_run does not stall the mic.
     * 16 x 1024 samples = 1.024 s of slack, far more than one inference, so
     * blocking between hops does NOT lengthen the hop period. Samples are only
     * lost if a single inference ever exceeds that budget, which the
     * "inference longer than hop" warning below makes visible. */
    ESP_LOGI(TAG, "I2S DMA ring: %d descs x %d frames = %d samples (%.3f s)",
             chan_cfg.dma_desc_num, chan_cfg.dma_frame_num,
             chan_cfg.dma_desc_num * chan_cfg.dma_frame_num,
             (double)chan_cfg.dma_desc_num * chan_cfg.dma_frame_num / SAMPLE_RATE);
}

static void kws_task(void *arg)
{
    /* The window walks in hop-sized steps. KWS_MAX_HOP bounds it, and the front
     * end only reuses feature frames when the hop is a whole number of feature
     * hops (320 samples), so the hop must be a multiple of 320. */
    const int hop = CONFIG_EXAMPLE_HOP_MS * SAMPLE_RATE / 1000;

    if (hop <= 0 || hop > KWS_MAX_HOP || (hop % KWS_HOP) != 0) {
        ESP_LOGE(TAG, "hop of %d samples is not a multiple of %d (<= %d)",
                 hop, KWS_HOP, KWS_MAX_HOP);
        vTaskDelete(NULL);
        return;
    }

    /* i2s_channel_read() destination must be DMA capable. */
    int16_t *raw = heap_caps_malloc(hop * sizeof(int16_t), MALLOC_CAP_DMA);
    static float spec[KWS_FRAMES * KWS_MELS];
    ESP_ERROR_CHECK(raw ? ESP_OK : ESP_ERR_NO_MEM);

    ESP_LOGI(TAG, "sliding window: %d ms hop (%d samples), refractory %d ms",
             CONFIG_EXAMPLE_HOP_MS, hop, CONFIG_EXAMPLE_REFRACTORY_MS);

    int refractory_steps = CONFIG_EXAMPLE_REFRACTORY_MS / CONFIG_EXAMPLE_HOP_MS;
    if (refractory_steps < 1) {
        refractory_steps = 1;
    }

    float hold = 0.0f;
    int hits = 0;
    int lockout = 0;

    /* Anchors for the latency record. The keyword's acoustic end is not
     * observable here -- it is somewhere inside the 1 s window, and the window
     * is 1 s long -- so what this logs is the interval it must lie in, and the
     * host analyzer narrows that against the saved audio. Reporting a single
     * "keyword ended at" number would mean inventing one. */
    int64_t win_end_sample = 0;
    int64_t first_hit_end = -1;

    int64_t next_stat = esp_timer_get_time();
    int64_t t_loop_start = 0;
    int64_t samples_at_start = 0;
    int64_t ema_us = 0;
    int32_t ema_inf_ms = 0;

    /*
     * Subscribe this task to the task watchdog and reset it in the loop.
     *
     * Without this the task is watched only indirectly, through IDLE0: the
     * watchdog is reset by idle tasks, so a task that never blocks starves them
     * and trips the watchdog even though nothing has hung. That is exactly what
     * happens here -- while the DMA ring stays full the read at the top of the
     * loop returns immediately, the task never yields, and IDLE0 does not run.
     *
     * Subscribing the task that is actually doing the work states the real
     * requirement, which is "an inference must not take longer than the
     * watchdog period", and stops the watchdog from being a proxy for whether
     * the scheduler got a turn.
     */
    esp_task_wdt_add(NULL);
    ESP_LOGI(TAG, "kws task subscribed to the task watchdog");

    for (;;) {
        esp_task_wdt_reset();
        const size_t want = (size_t)hop * sizeof(int16_t);
        size_t got = 0;
        while (got < want) {
            size_t n = 0;
            esp_err_t err = i2s_channel_read(s_rx_chan, (uint8_t *)raw + got,
                                             want - got, &n, portMAX_DELAY);
            if (err != ESP_OK) {
                ESP_LOGW(TAG, "i2s read: %s (after %u of %u bytes)",
                         esp_err_to_name(err), (unsigned)got, (unsigned)want);
                vTaskDelay(pdMS_TO_TICKS(10));
                break;
            }
            if (n == 0) {
                /* No data and no error. Yield rather than spin, so the idle
                 * task still gets the core -- the task watchdog subscribes
                 * IDLE0 on this part, and a spin here starves it. */
                taskYIELD();
                continue;
            }
            got += n;
        }

        /*
         * Advance the clock by what was ACTUALLY read, not by a nominal hop.
         *
         * This was `s_sample_clock += hop`, unconditionally. Combined with the
         * `break` above it meant a short or failed read still counted a whole
         * hop, so the sample clock ran ahead of the wall clock: drift went
         * negative and kept going (-278 -> -7106 ms over 25 s), which is the
         * signature of the analysis being fed time-compressed audio rather than
         * the microphone. It also made the drift watchdog report the opposite of
         * what was happening, since a genuine overrun would have pushed it
         * positive.
         *
         * Only whole hops are analysed: a partial read would leave the front end
         * mid-hop, which its 20 ms feature alignment rejects anyway.
         */
        const size_t have = got / sizeof(int16_t);
        s_sample_clock += (int64_t)have;
        win_end_sample = s_sample_clock;
        if (t_loop_start == 0) {
            t_loop_start = esp_timer_get_time();
            samples_at_start = s_sample_clock;
        }

        if (have != (size_t)hop) {
            /* Counted honestly above; nothing to analyse this iteration. */
            continue;
        }

        #if CONFIG_EXAMPLE_PRE_ROLL_SECONDS > 0
        ring_push(raw, (size_t)hop);
#endif
#if CONFIG_EXAMPLE_STORAGE_ENABLED
        capture_push(raw, (size_t)hop);
#endif

        kws_frontend_push(raw, hop);
        if (!kws_frontend_compute(spec)) {
            continue;               /* still filling the first window */
        }

        const int64_t t0 = esp_timer_get_time();
        const float prob = kws_m1gw_run(spec);
        const int64_t dt = esp_timer_get_time() - t0;
        /* An inference this long is the thing the watchdog is watching for, so
         * reset after it as well as before the read. */
        esp_task_wdt_reset();

        /* Exponential average of inference time, so the reading is stable. */
        ema_us = (ema_us == 0) ? dt : (ema_us * 7 + dt) / 8;
        ema_inf_ms = (int32_t)(ema_us / 1000);

        /* A single long inference is not what costs audio. The read blocks until the
         * hop's samples exist, so the recurrence is
         *
         *     period(k) = max(hop, period(k-1)) + infer
         *
         * and when infer > hop it never catches up: every iteration falls
         * further behind real time and the DMA ring's backlog grows by
         * (infer - hop) per hop until the ring is full, after which the oldest
         * samples are overwritten before they are ever read. The per-inference
         * warning below cannot see that -- 334 ms against a 1024 ms ring is
         * under the threshold -- so the running drift is measured against the
         * sample clock instead, which is the quantity that actually goes wrong. */
        if (dt > (int64_t)RING_SLACK_MS * 1000) {
            ESP_LOGW(TAG, "inference took %d ms, longer than the %d ms DMA ring "
                          "-- audio is being dropped",
                     (int)(dt / 1000), RING_SLACK_MS);
        } else if (dt > (int64_t)CONFIG_EXAMPLE_HOP_MS * 1000) {
            ESP_LOGW(TAG, "inference took %d ms, longer than the %d ms hop - "
                          "period is now ~%d ms (no samples lost, ring is %d ms)",
                     (int)(dt / 1000), CONFIG_EXAMPLE_HOP_MS,
                     (int)(dt / 1000) + CONFIG_EXAMPLE_HOP_MS, RING_SLACK_MS);
        }

        /* Peak-hold confirmation, exactly as models/kws_engine.py does it. An
         * EMA at alpha 0.6 cannot confirm a realistic burst: two frames at
         * p=0.90 lift it only to 0.71, which never clears the 0.68 operating
         * point after a dip. The hold confirms on a genuine 0.5 s keyword. */
        hold = (prob > hold * CONFIG_EXAMPLE_HOLD_DECAY)
                   ? prob : hold * CONFIG_EXAMPLE_HOLD_DECAY;

        if (lockout > 0) {
            lockout--;
        }

        bool detected = false;
        if (hold >= CONFIG_EXAMPLE_THRESHOLD) {
            if (++hits >= CONFIG_EXAMPLE_NEED) {
                if (lockout == 0) {
                    detected = true;
                    hits = 0;
                    hold = 0.0f;
                    lockout = refractory_steps;
                }
            }
        } else {
            hits = 0;
        }

        /* First window over threshold, whether or not it confirmed. This is the
         * tightest on-device bound available on where the keyword ended. */
        if (hold >= CONFIG_EXAMPLE_THRESHOLD && first_hit_end < 0) {
            first_hit_end = win_end_sample;
        }

        if (detected) {
            const int64_t t_detect_us = esp_timer_get_time();

            ESP_LOGW(TAG, "WAKE WORD 'amaze'  p=%.3f  (%d ms inference)",
                     (double)prob, ema_inf_ms);

            s_led_state = LED_DETECTED;
            s_led_until = t_detect_us + 3000000;
#if CONFIG_EXAMPLE_STORAGE_ENABLED
            capture_start(win_end_sample);
#endif

            /*
             * The latency record, on one machine-parseable line, logged after
             * capture_start so `clip_start` is this turn's and not the last one.
             *
             * `kw_end_lo`/`kw_end_hi` bracket where the keyword ended. The
             * first above-threshold window is the earliest instant the model had
             * seen the whole keyword, so it cannot end later than that window's
             * trailing edge; and no earlier window scored, so it cannot end
             * before the earliest of them. The interval is a full second wide,
             * which is why the host analyzer resolves it acoustically from the
             * saved WAV instead of the device pretending to know.
             *
             * `clip_start_sample` is where the saved WAV's first sample sits in
             * the device timeline, which is what ties that analysis to this
             * detection.
             */
            ESP_LOGW(TAG, "LAT t_us=%" PRId64 " sample=%" PRId64
                          " kw_end_lo=%" PRId64 " kw_end_hi=%" PRId64
                          " clip_start=%" PRId64 " infer_us=%" PRId64
                          " drift_us=%" PRId64,
                     t_detect_us, win_end_sample,
                     (first_hit_end >= 0 ? first_hit_end : win_end_sample) - KWS_AUDIO_SAMPLES,
                     win_end_sample, s_capture_start_sample, dt,
                     (t_detect_us - t_loop_start)
                         - (int64_t)((double)(win_end_sample - samples_at_start)
                                     * 1000000.0 / SAMPLE_RATE));

            first_hit_end = -1;
        }

        const int64_t now = esp_timer_get_time();
        if (now >= next_stat) {
            /* Drift is the measurement the per-inference warning cannot make:
             * how far the samples the task has consumed have fallen behind the
             * wall clock. Zero means the device is keeping up with the room.
             * Once it exceeds the DMA ring, audio is being lost silently and
             * every downstream number -- including any latency figure taken from
             * this log -- describes a timeline with holes in it. */
            const int64_t drift_us = (now - t_loop_start)
                - (int64_t)((double)(s_sample_clock - samples_at_start)
                            * 1000000.0 / SAMPLE_RATE);

            ESP_LOGI(TAG, "listening  p=%.3f hold=%.3f  infer=%d ms  hop=%d ms  "
                          "duty=%.1f%%  drift=%" PRId64 " ms  heap=%u",
                     (double)prob, (double)hold, ema_inf_ms,
                     CONFIG_EXAMPLE_HOP_MS,
                     100.0f * (float)ema_inf_ms / (float)CONFIG_EXAMPLE_HOP_MS,
                     drift_us / 1000,
                     (unsigned)esp_get_free_heap_size());

            if (drift_us > (int64_t)RING_SLACK_MS * 1000) {
                ESP_LOGW(TAG, "sample clock is %" PRId64 " ms behind the wall "
                              "clock, past the %d ms DMA ring -- audio is being "
                              "dropped. Inference (%" PRId64 " ms) exceeds the "
                              "%d ms hop, so the task never catches up.",
                         drift_us / 1000, RING_SLACK_MS, dt / 1000,
                         CONFIG_EXAMPLE_HOP_MS);
            }

            /* Where the inference time goes, averaged over the runs since the
             * last line. An inference that overruns the hop has to be attacked
             * where the time actually is, and the stage names are the argument
             * for which stage that is. */
            const uint64_t runs = kws_m1gw_stage_runs();
            if (runs > 0) {
                uint64_t total_us = 0, total_cyc = 0;
                for (int s = 0; s < KWS_M1GW_STAGE_COUNT; s++) {
                    total_us += kws_m1gw_stage_us(s);
                    total_cyc += kws_m1gw_stage_cycles(s);
                }
                const double ms = (double)total_us / 1000.0 / (double)runs;
                const double duty = 100.0 * (double)total_us
                        / ((double)runs * (double)CONFIG_EXAMPLE_HOP_MS * 1000.0);

                ESP_LOGI(TAG, "  ---- resource summary over %llu inference(s) ----",
                         (unsigned long long)runs);
                ESP_LOGI(TAG, "  cpu      %8.2f ms/infer  %5.1f%% of a %d ms hop  "
                              "last %u us  peak %u us",
                         ms, duty, CONFIG_EXAMPLE_HOP_MS,
                         (unsigned)kws_m1gw_last_us(),
                         (unsigned)kws_m1gw_peak_us());
                ESP_LOGI(TAG, "  cycles   %8llu total  %.2f M/infer  (measured rate %llu Hz)",
                         (unsigned long long)total_cyc,
                         (double)total_cyc / (double)runs / 1e6,
                         (unsigned long long)kws_m1gw_cpu_hz());
                ESP_LOGI(TAG, "  work     %8u MAC/infer  %.3f MAC/us  floor %.3f ms "
                              "@ 16 MAC/instr",
                         (unsigned)kws_m1gw_total_mac(),
                         ms > 0.0 ? (double)kws_m1gw_total_mac() / (double)total_us : 0.0,
                         (double)kws_m1gw_total_mac() / 3.84e6);
                ESP_LOGI(TAG, "  ram      %8u B arena (%.1f%% of 256 KB)  free heap %u B",
                         (unsigned)kws_m1gw_ram_bytes(),
                         100.0 * kws_m1gw_ram_bytes() / (256.0 * 1024.0),
                         (unsigned)esp_get_free_heap_size());

                ESP_LOGI(TAG, "    %-10s %8s %8s %9s %7s",
                         "stage", "ms", "MAC", "cyc/MAC", "share");
                for (int s = 0; s < KWS_M1GW_STAGE_COUNT; s++) {
                    const uint64_t us = kws_m1gw_stage_us(s);
                    const uint64_t cyc = kws_m1gw_stage_cycles(s);
                    const uint32_t mac = kws_m1gw_stage_mac(s);
                    /* cycles per MAC is the diagnostic ratio: a stage spending
                     * far more than its neighbours is an implementation problem
                     * (misaligned reduction, fallback path), not a model one. */
                    char ratio[16] = "-";
                    if (mac > 0) {
                        snprintf(ratio, sizeof(ratio), "%.2f",
                                 (double)cyc / (double)mac / (double)runs);
                    }
                    ESP_LOGI(TAG, "    %-10s %8.2f %8u %9s %6.1f%%",
                             kws_m1gw_stage_name(s),
                             (double)us / 1000.0 / (double)runs,
                             (unsigned)mac, ratio,
                             100.0 * (double)us / (double)(total_us ? total_us : 1));
                }
                ESP_LOGI(TAG, "    projection breakdown (per inference):");
                for (int s = 0; s < KWS_M1GW_PROJ_COUNT; s++) {
                    ESP_LOGI(TAG, "      %-8s %8.3f ms %12llu cyc",
                             kws_m1gw_proj_name(s),
                             (double)kws_m1gw_proj_us(s) / 1000.0 / (double)runs,
                             (unsigned long long)(kws_m1gw_proj_cycles(s)
                                 / (unsigned long long)runs));
                }
                kws_m1gw_proj_reset();
                kws_m1gw_stage_reset();
            }
            next_stat = now + 5000000;
        }
    }
}

void app_main(void)
{
    led_init();

    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    s_ring_lock = xSemaphoreCreateMutex();
    ESP_ERROR_CHECK(s_ring_lock ? ESP_OK : ESP_ERR_NO_MEM);

#if CONFIG_EXAMPLE_PRE_ROLL_SECONDS > 0
    s_ring = alloc_capture(PRE_ROLL_SAMPLES * sizeof(int16_t));
    ESP_ERROR_CHECK(s_ring ? ESP_OK : ESP_ERR_NO_MEM);
#endif

#if CONFIG_EXAMPLE_STORAGE_ENABLED
    s_clip = alloc_capture(CLIP_SAMPLES * sizeof(int16_t));
    ESP_ERROR_CHECK(s_clip ? ESP_OK : ESP_ERR_NO_MEM);
    s_capture_lock = xSemaphoreCreateMutex();
    s_write_req = xSemaphoreCreateBinary();
    ESP_ERROR_CHECK(s_capture_lock ? ESP_OK : ESP_ERR_NO_MEM);
    ESP_ERROR_CHECK(s_write_req ? ESP_OK : ESP_ERR_NO_MEM);
    ESP_LOGI(TAG, "keeping a rolling %d s capture in PSRAM", CONFIG_EXAMPLE_CLIP_SECONDS);
#endif

    kws_frontend_init();

    /*
     * Verify the INT8 vector reduction against the portable one before arming
     * the detector.
     *
     * The assembly kernel in kws_xtensa.S is the difference between meeting the
     * CPU budget and missing it by a factor of four, and a kernel that computes
     * the wrong sum does not announce itself: it returns a plausible probability,
     * and the symptom is a wake word that never fires. A kernel that assembles is
     * not a kernel that computes the right thing, so the answer is checked here,
     * once, over every reduction length the network uses -- and the portable
     * loop takes over automatically if it disagrees.
     */
    const int kernel_ok = kws_kernel_selftest();
    /* The fast transcendentals are only enabled once they have been measured
     * against libm, and this is where that happens. m1_g_wide makes roughly
     * 29,000 of these calls per inference; at libm's ~1000 cycles each they were
     * most of a 225 ms inference, and the multiply-accumulates they drowned out
     * were under a millisecond of the budget. */
    const int fast_ok = kws_fastmath_selftest();
    ESP_LOGI(TAG, "fast exp: %s (worst rel %.2e, tol 1e-4)",
             fast_ok ? "in use" : "FELL BACK to expf",
             (double)kws_expf_worst_relerr());
    ESP_LOGI(TAG, "fast rcp: %s (worst rel %.2e, tol 1e-5)  [no __divsf3 per activation]",
             kws_rcp_worst_relerr() < 1.0e-5f ? "in use" : "FELL BACK to divide",
             (double)kws_rcp_worst_relerr());

    ESP_LOGI(TAG, "int8 kernel: %s%s", kws_kernel_using_xtensa() ? "xtensa vector"
                                                                 : "portable C",
             kernel_ok ? "" : " -- SELF TEST FAILED, fell back to portable C");

    /*
     * conv2's windowed reduction is demoted on its own evidence, so it gets its
     * own line. If this says portable while the line above says xtensa vector,
     * conv2 is running the scalar loop and the inference will be slow for a
     * reason that has nothing to do with the model.
     */
    ESP_LOGI(TAG, "conv2 window reduction: %s",
             kws_kernel_taps_using_xtensa() ? "xtensa vector"
                                            : "portable C -- SELFTEST FAILED");

    /*
     * The two transcendental replacements, reported with the error each one
     * measured. A fallback that engages silently is indistinguishable from a
     * build that was always this slow, so the verdict belongs in the boot log
     * next to the kernel's: if infer= comes back high, "exp: fell back" is the
     * first thing to check.
     */
    /* The fast exp/rcp/rsqrt probes lived in kws_model.c and went with it. m1_g_wide
     * has no hand-rolled fast path for them, so there is nothing to report here;
     * the kernel self-test above is still the gate on arithmetic. */
    ESP_LOGI(TAG, "Amaze m1_g_wide, %u params, %u B int8 weights, "
                  "threshold %.6f, %d-of-%d peak-hold, decay %.2f",
             (unsigned)KWS_M1GW_PARAM_COUNT,
             (unsigned)kws_m1gw_ram_bytes(),
             (double)kws_m1gw_threshold(),
             CONFIG_EXAMPLE_NEED, CONFIG_EXAMPLE_NEED,
             (double)CONFIG_EXAMPLE_HOLD_DECAY);

    i2s_init();

    xTaskCreatePinnedToCore(led_task,     "led",   3072, NULL, 3, NULL, 0);
#if CONFIG_EXAMPLE_STORAGE_ENABLED
    xTaskCreatePinnedToCore(storage_task, "store", 4096, NULL, 2, NULL, 1);
#endif
    xTaskCreatePinnedToCore(kws_task,     "kws",   8192, NULL, 6, NULL, 0);

    ESP_LOGI(TAG, "running - say 'amaze'; LED turns green for 3 s on a hit");
}