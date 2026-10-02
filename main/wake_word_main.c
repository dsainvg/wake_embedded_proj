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
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "nvs_flash.h"

#if CONFIG_EXAMPLE_LED_TYPE_WS2812
#include "led_strip.h"
#endif

#include "esp_afe_config.h"
#include "esp_afe_sr_iface.h"
#include "esp_afe_sr_models.h"
#include "esp_wn_iface.h"
#include "esp_wn_models.h"
#include "model_path.h"

static const char *TAG = "wake";

#define SAMPLE_RATE          CONFIG_EXAMPLE_SAMPLE_RATE
#define CLIP_SECONDS         CONFIG_EXAMPLE_CLIP_SECONDS
#define CLIP_SAMPLES         (SAMPLE_RATE * CLIP_SECONDS)
#define PRE_ROLL_SAMPLES     (SAMPLE_RATE * CONFIG_EXAMPLE_PRE_ROLL_SECONDS)
#define WAV_HEADER_LEN       44
#define FLASH_SECTOR_SIZE    4096
#define ALIGN_UP(v, a)       (((v) + (a)-1) / (a) * (a))

/* ------------------------------------------------------------------ */
/* Globals                                                            */
/* ------------------------------------------------------------------ */

static const esp_afe_sr_iface_t *s_afe_handle = NULL;
static esp_afe_sr_data_t *s_afe_data = NULL;

static i2s_chan_handle_t s_rx_chan = NULL;

/* Short pre-roll ring, used only to seed a clip with the tail of the keyword. */
static int16_t *s_ring = NULL;
static size_t s_ring_wpos = 0;
static size_t s_ring_filled = 0;
static SemaphoreHandle_t s_ring_lock = NULL;

/* Post-detection capture buffer. */
static int16_t *s_clip = NULL;
static size_t s_capture_len = 0;          /* valid samples in s_clip   */
static volatile bool s_capture_active = false;
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
 */
static void capture_start(void)
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
    chan_cfg.dma_desc_num = 16;
    chan_cfg.dma_frame_num = 1024;
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
}

static void feed_task(void *arg)
{
    const int chunk = s_afe_handle->get_feed_chunksize(s_afe_data);
    const int nch  = s_afe_handle->get_feed_channel_num(s_afe_data);
    const size_t n_samples = (size_t)chunk * nch;

    ESP_LOGI(TAG, "AFE feed: %d samples/frame, %d channel(s)", chunk, nch);

    int32_t *raw = heap_caps_malloc(n_samples * sizeof(int32_t), MALLOC_CAP_DMA);
    int16_t *pcm = heap_caps_malloc(n_samples * sizeof(int16_t), MALLOC_CAP_INTERNAL);
    ESP_ERROR_CHECK(raw ? ESP_OK : ESP_ERR_NO_MEM);
    ESP_ERROR_CHECK(pcm ? ESP_OK : ESP_ERR_NO_MEM);

    for (;;) {
        size_t got = 0;
        while (got < n_samples * sizeof(int32_t)) {
            size_t bytes_read = 0;
            esp_err_t err = i2s_channel_read(s_rx_chan, (uint8_t *)raw + got,
                                             n_samples * sizeof(int32_t) - got,
                                             &bytes_read, portMAX_DELAY);
            if (err != ESP_OK) {
                ESP_LOGW(TAG, "i2s read: %s", esp_err_to_name(err));
                vTaskDelay(pdMS_TO_TICKS(10));
                break;
            }
            got += bytes_read;
        }

        /* INMP441 24-bit payload sits in bits 31..8 of the 32-bit slot. */
        for (size_t i = 0; i < n_samples; i++) {
            pcm[i] = (int16_t)(raw[i] >> CONFIG_EXAMPLE_MIC_SAMPLE_SHIFT);
        }

        ring_push(pcm, n_samples);
        capture_push(pcm, n_samples);
        s_afe_handle->feed(s_afe_data, pcm);
    }
}

/* ------------------------------------------------------------------ */
/* AFE fetch: detection, LED, capture trigger                         */
/* ------------------------------------------------------------------ */

static void detect_task(void *arg)
{
    int64_t last_hit = 0;
    int64_t next_stat = esp_timer_get_time();
    float peak_dbfs = -100.0f;

    for (;;) {
        afe_fetch_result_t *res = s_afe_handle->fetch(s_afe_data);
        if (res == NULL || res->ret_value == ESP_FAIL) {
            continue;
        }

        if (res->data_volume > peak_dbfs) {
            peak_dbfs = res->data_volume;
        }

        int64_t now = esp_timer_get_time();

        if (res->wakeup_state == WAKENET_DETECTED &&
            now - last_hit >= (int64_t)CONFIG_EXAMPLE_DETECT_COOLDOWN_MS * 1000) {
            last_hit = now;

            ESP_LOGW(TAG, "WAKE WORD DETECTED (model %d, word %d, %.1f dBFS)",
                     res->wakenet_model_index, res->wake_word_index,
                     (double)res->data_volume);

            /* Green acknowledgement for 3 s. */
            s_led_state = LED_DETECTED;
            s_led_until = now + 3000000;

#if CONFIG_EXAMPLE_STORAGE_ENABLED
            /* Open the window on the audio that follows the keyword. */
            capture_start();
#endif
        }

        if (now >= next_stat) {
            ESP_LOGI(TAG, "listening: vad=%d peak=%.1f dBFS heap=%u",
                     res->vad_state, (double)peak_dbfs,
                     (unsigned)esp_get_free_heap_size());
            peak_dbfs = -100.0f;
            next_stat = now + 5000000;
        }
    }
}

/* ------------------------------------------------------------------ */
/* app_main                                                           */
/* ------------------------------------------------------------------ */

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
#endif

    srmodel_list_t *models = esp_srmodel_init("model");
    ESP_ERROR_CHECK(models ? ESP_OK : ESP_ERR_NOT_FOUND);

    for (int i = 0; i < models->num; i++) {
        if (strstr(models->model_name[i], ESP_WN_PREFIX) != NULL) {
            ESP_LOGI(TAG, "wake word model: %s (%s)",
                     models->model_name[i], models->model_info[i]);
        }
    }
    if (esp_srmodel_filter(models, ESP_WN_PREFIX, NULL) == NULL) {
        ESP_LOGE(TAG, "no WakeNet model flashed - pick one under "
                      "menuconfig > ESP Speech Recognition");
    }

    afe_config_t *cfg = afe_config_init("M", models, AFE_TYPE_SR, AFE_MODE_LOW_COST);
    ESP_ERROR_CHECK(cfg ? ESP_OK : ESP_ERR_NO_MEM);

    if (cfg->wakenet_model_name) {
        ESP_LOGI(TAG, "AFE wake word: %s", cfg->wakenet_model_name);
    }

    s_afe_handle = esp_afe_handle_from_config(cfg);
    ESP_ERROR_CHECK(s_afe_handle ? ESP_OK : ESP_ERR_NO_MEM);
    s_afe_data = s_afe_handle->create_from_config(cfg);
    ESP_ERROR_CHECK(s_afe_data ? ESP_OK : ESP_ERR_NO_MEM);
    afe_config_free(cfg);

    s_afe_handle->set_wakenet_threshold(s_afe_data, 1,
                                        CONFIG_EXAMPLE_WAKENET_THRESHOLD);

    i2s_init();

    xTaskCreatePinnedToCore(led_task,    "led",    3072, NULL, 3, NULL, 0);
#if CONFIG_EXAMPLE_STORAGE_ENABLED
    xTaskCreatePinnedToCore(storage_task, "store", 4096, NULL, 2, NULL, 1);
#endif
    xTaskCreatePinnedToCore(feed_task,   "feed",   8192, NULL, 6, NULL, 0);
    xTaskCreatePinnedToCore(detect_task, "detect", 4096, NULL, 5, NULL, 1);

    ESP_LOGI(TAG, "running - say your wake word; LED turns green for 3 s on a hit");
}