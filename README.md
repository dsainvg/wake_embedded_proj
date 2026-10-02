# ESP32-S3 + INMP441 wake word detector

ESP-SR WakeNet offline wake word detection on an ESP32-S3 N16R8, using an INMP441
I2S microphone. No SD card. Wake word hits light an LED and dump the previous 10
seconds of mic audio to internal flash as a WAV file.

## Hardware

| INMP441 | ESP32-S3 |
|---|---|
| `VDD`   | `3V3`    |
| `GND`   | `GND`    |
| `LR`    | `GND`    |
| `WS`    | `GPIO40` |
| `SCK`   | `GPIO41` |
| `SD`    | `GPIO42` |

`LR` tied to `GND` selects the **left** slot, which matches the
`I2S_STD_SLOT_LEFT` + `I2S_SLOT_MODE_MONO` I2S configuration.

GPIO 40/41/42 are the JTAG pins (MTCK/MTDO/MTDI). They work fine as I2S, but do
not run JTAG debugging at the same time. On N16R8, octal PSRAM only occupies
GPIO 33-37, so there is no conflict.

## Status LED

GPIO48 on most ESP32-S3 devkits is an **addressable RGB LED (WS2812)**, which a
plain `gpio_set_level()` cannot light - the driver has to push pixel data down a
single data line. `led_strip` (the `espressif/led_strip` component, driven over
RMT) is used by default, and the build below confirms it works.

Set **menuconfig > Wake Word Detection > Status LED type**:

| Option | Use when |
|---|---|
| `Addressable RGB LED (WS2812 / SK6812)` | default; onboard RGB on GPIO48 |
| `Plain GPIO LED` | you wired a discrete LED to the pin yourself |

Colours carry the state, so you don't have to count blinks:

| Colour | Meaning |
|---|---|
| dim orange, single pulse every 3 s | firmware alive, streaming from the mic |
| **bright green, solid 3 s** | **wake word detected** |
| amber, blinking 10 Hz | writing the finished clip to flash |
| off | between events |

`EXAMPLE_LED_LED_COUNT` sets the pixel count if you have more than one.

## Capture behaviour

On an accepted detection the capture window **opens at the keyword and closes
`EXAMPLE_CLIP_SECONDS` later**, so what you get is the keyword followed by the
next ~9 s of speech — not the 10 s that preceded it.

```
 |<-- 1 s -->|=================== 10 s total ===================|
   keyword tail                audio after the keyword
```

The 1 s of pre-roll is `EXAMPLE_PRE_ROLL_SECONDS`. It exists because a clip that
starts *after* detection loses the keyword itself, which is the part a human
listener wants first. Set it to `0` if you want the window to start strictly at
the detection instant.

The LED goes green for 3 s immediately on the hit; the clip is only written once
the full 10 s has elapsed, so the amber flash-write indicator appears roughly 10 s
after the green.

## Build result

```
wake.bin  binary size 0xdc9a0 bytes (903475 B), 71% of the 3 MB app partition free
DIRAM     117867 B / 341760 B static (34.5%)
Flash     695204 B code + 117632 B rodata
```

That 117 KB of static DIRAM leaves ~224 KB of internal SRAM for the AFE
pipeline, which allocates ~60 KB at runtime plus FreeRTOS heaps. PSRAM use at
runtime is the 740 KB AFE plus the 320 KB capture ring, out of 8 MB.

## Memory cost of a wake word

Measured on ESP32-S3 (ESP-SR benchmark):

| Configuration | Internal RAM | PSRAM | CPU |
|---|---|---|---|
| WakeNet9 bare (2-channel) | 16 KB | 324 KB | 3.0 ms per 32 ms frame |
| WakeNet9 bare (3-channel) | 20 KB | 347 KB | 4.3 ms per frame |
| WakeNet10 bare (3-channel) | 17 KB | 523 KB | 7.1 ms per frame |
| Full AFE (1 mic, SR, low cost) | 60 KB | 740 KB | 8.8% feed + 9.8% fetch of one core |

This project runs the full AFE pipeline, so budget roughly **60 KB internal RAM
and 800 KB PSRAM**, plus ~320 KB PSRAM for the 10-second capture ring. The
`wn9_hiesp` model itself occupies ~370 KB in the `model` flash partition.

## Flash layout

`partitions.csv` for the 16 MB module:

| Partition  | Offset    | Size     | Purpose |
|------------|-----------|----------|---------|
| `nvs`       | `0x9000`  | 20 KB    | boot variables, clip cursor |
| `otadata`   | `0xE000`  | 8 KB     | OTA state |
| `phy_init`  | `0x10000` | 4 KB     | PHY calibration |
| `factory`   | `0x20000` | 3 MB     | application |
| `model`     | auto      | 2 MB     | ESP-SR neural network models |
| `storage`   | auto      | 8 MB     | rolling WAV clips (16 slots) |

Each slot holds one 10-second 16 kHz 16-bit mono WAV (~313 KB). Slots are used
round-robin; the next free index is kept in NVS so it survives reboots. The most
recent clip overwrites the oldest.

Pull a clip off the device with `esptool.py`:

```
esptool.py --chip esp32s3 read_flash 0x520000 0x4E240 clip.wav
```

(the offset is `partition->address + slot * slot_size`, printed on the serial log
at each save).

## Build and flash

```bash
idf.py set-target esp32s3
idf.py build
idf.py flash monitor
```

Select a different wake word under **menuconfig > ESP Speech Recognition > Load
Multiple Wake Words (WakeNet9 or WakeNet10)**. Options include
`wn9_hilesp` (Hi, 乐鑫), `wn9_nihaoxiaozhi` (你好小智), `wn9_alexa`,
`wn10_hiesp`, and about 60 others.

## Using your own wake word

ESP-SR supports custom wake words two ways:

1. **External model path** - menuconfig > ESP Speech Recognition > model data path
   = external path, pointing at a directory of model folders. CMake packages them
   into the `model` partition.
2. **Bundled model** - drop the model folder into
   `esp-sr/model/wakenet_model` and add a matching `SR_WN_<NAME>` entry to its
   `Kconfig.projbuild`.

Training/serving details:
<https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/wake_word_engine/ESP_Wake_Words_Customization.html>

A ready alternative is MultiNet (`mn6_en` / `mn7_en`), which needs **no training at
all** - you just list up to 300 English phrases in menuconfig and it recognizes
them. It costs more RAM than WakeNet (mn6_en is ~4.1 MB PSRAM) but gives you
arbitrary commands instead of one fixed wake word.

## Tuning

| Symbol | Default | Effect |
|---|---|---|
| `EXAMPLE_WAKENET_THRESHOLD` | 0.5 | Raise for fewer false triggers, lower for better recall |
| `EXAMPLE_DETECT_COOLDOWN_MS` | 2000 | Ignore repeats after a hit |
| `EXAMPLE_MIC_SAMPLE_SHIFT` | 16 | Use 14 for +6 dB gain (may clip) |
| `EXAMPLE_CLIP_SECONDS` | 10 | Length of each saved clip |

The INMP441 puts 24 bits of two's complement audio left-justified in a 32-bit I2S
slot, i.e. in bits 31..8. A right shift of 16 discards both the 8 unused low bits
and the 8 low bits of the payload, giving a correct full-scale `int16`.

## Troubleshooting

**No wake word hits, `peak` stuck near -100 dBFS** - the mic is not delivering
data. Check `3V3`, that `LR` is on `GND`, and that `WS`/`SCK`/`SD` are not
swapped. GPIO 40/41/42 on the JTAG header need to be free of a debugger.

**`no WakeNet model flashed`** - set the model under menuconfig > ESP Speech
Recognition, then `idf.py fullclean && idf.py build`. The model is packaged into
the `model` partition by CMake and flashed by `idf.py flash`.

**Every prompt triggers it** - raise `EXAMPLE_WAKENET_THRESHOLD` toward 0.8.

**Rarely triggers, but works when close** - the INMP441 has no AGC. Lower
`EXAMPLE_MIC_SAMPLE_SHIFT` to 14 for gain, or enable `agc_init` in the AFE config.