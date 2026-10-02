idf_component_register(
    SRCS "wake_word_main.c"
    INCLUDE_DIRS "."
    REQUIRES
        esp-sr
        esp_driver_i2s
        esp_driver_gpio
        esp_timer
        nvs_flash
        spi_flash
)