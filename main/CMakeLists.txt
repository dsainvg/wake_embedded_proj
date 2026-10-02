# The weight blob is turned into a C array at build time by tools/bin2c.py
# rather than embedded with CMake's target_add_binary_data. A blob placed into
# .rodata by the assembler has no guaranteed 4-byte alignment, and the Xtensa
# LX7 traps on unaligned float loads instead of silently coping. A C array lets
# the compiler place it with explicit alignment.
#
# Generated at CONFIGURE time rather than as a build rule: an
# add_custom_command(OUTPUT ...) is "not scriptable" in an IDF component scope
# because no target exists yet when IDF runs its scriptable-configure pass.
# The file only changes when the model does, so it is guarded by a timestamp
# stamp file.
set(KWS_BIN ${CMAKE_CURRENT_LIST_DIR}/model/kws_model.bin)
set(KWS_SRC ${CMAKE_CURRENT_BINARY_DIR}/kws_weights.c)
set(KWS_BIN2C ${CMAKE_CURRENT_LIST_DIR}/../tools/bin2c.py)
set(KWS_STAMP ${CMAKE_CURRENT_BINARY_DIR}/kws_weights.stamp)

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${KWS_BIN} ${KWS_BIN2C})

file(TIMESTAMP ${KWS_BIN} KWS_BIN_MTIME "%Y%m%d%H%M%S")
set(KWS_NEED_GEN TRUE)
if(EXISTS ${KWS_STAMP})
    file(READ ${KWS_STAMP} KWS_STAMPED)
    if(KWS_STAMPED STREQUAL KWS_BIN_MTIME AND EXISTS ${KWS_SRC})
        set(KWS_NEED_GEN FALSE)
    endif()
endif()

if(KWS_NEED_GEN)
    execute_process(
        COMMAND ${PYTHON} ${KWS_BIN2C} ${KWS_BIN} ${KWS_SRC} kws_weights
        RESULT_VARIABLE KWS_RC
        OUTPUT_VARIABLE KWS_OUT
        ERROR_VARIABLE  KWS_ERR)
    if(NOT KWS_RC EQUAL 0)
        message(FATAL_ERROR "bin2c failed (${KWS_RC}): ${KWS_ERR}")
    endif()
    file(WRITE ${KWS_STAMP} ${KWS_BIN_MTIME})
    message(STATUS "bin2c: ${KWS_OUT}")
endif()

idf_component_register(
    SRCS "wake_word_main.c"
         "kws_frontend.c"
         "kws_model.c"
         "${KWS_SRC}"
    INCLUDE_DIRS "." "model"
    REQUIRES
        esp_driver_i2s
        esp_driver_gpio
        esp_timer
        nvs_flash
        spi_flash
)