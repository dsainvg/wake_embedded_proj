# The weight blobs are turned into C arrays at build time by tools/bin2c.py
# rather than embedded with CMake's target_add_binary_data. A blob placed into
# .rodata by the assembler has no guaranteed alignment, and the Xtensa LX7 traps
# on an unaligned load instead of silently coping. A C array lets the compiler
# place it with explicit alignment, which the INT8 vector reduction in
# main/kws_xtensa.S needs: it loads 128 bits at a time, so both the blob and
# every 16-byte row inside it have to be aligned.
#
# Two arrays come out:
#   kws_weights.c   the int8 weight blob, ~79 KB
#   kws_params.c    5,711 float32 values -- the normalisation gains and biases,
#                   the attention position bias, and the stem convolutions,
#                   which stay in float32 on purpose. See FLOAT_TENSORS in
#                   tools/export_weights.py for why those are not quantised.
#
# The files only change when the model does, so they are generated at CONFIGURE
# time rather than as build rules: an add_custom_command(OUTPUT ...) is "not
# scriptable" in an IDF component scope because no target exists yet when IDF
# runs its scriptable-configure pass. A timestamp stamp file guards the work.
set(KWS_BIN    ${CMAKE_CURRENT_LIST_DIR}/model/kws_model_i8.bin)
set(KWS_BIN2C  ${CMAKE_CURRENT_LIST_DIR}/../tools/bin2c.py)
set(KWS_WSRC   ${CMAKE_CURRENT_BINARY_DIR}/kws_weights.c)
set(KWS_PSRC   ${CMAKE_CURRENT_BINARY_DIR}/kws_params.c)
set(KWS_STAMP  ${CMAKE_CURRENT_BINARY_DIR}/kws_weights.stamp)

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             ${KWS_BIN} ${KWS_BIN2C})

file(TIMESTAMP ${KWS_BIN} KWS_BIN_MTIME "%Y%m%d%H%M%S")
set(KWS_NEED_GEN TRUE)
if(EXISTS ${KWS_STAMP})
    file(READ ${KWS_STAMP} KWS_STAMPED)
    if(KWS_STAMPED STREQUAL KWS_BIN_MTIME AND EXISTS ${KWS_WSRC} AND EXISTS ${KWS_PSRC})
        set(KWS_NEED_GEN FALSE)
    endif()
endif()

if(KWS_NEED_GEN)
    execute_process(
        COMMAND ${PYTHON} ${KWS_BIN2C} ${KWS_BIN} ${KWS_WSRC} kws_weights_i8 --int8
        RESULT_VARIABLE KWS_RC
        OUTPUT_VARIABLE KWS_OUT
        ERROR_VARIABLE  KWS_ERR)
    if(NOT KWS_RC EQUAL 0)
        message(FATAL_ERROR "bin2c failed (${KWS_RC}): ${KWS_ERR}")
    endif()
    message(STATUS "${KWS_OUT}")

    # The float parameters are emitted by the same exporter, as values inside
    # main/model/kws_model_data.h, so there is no second blob to convert here.
    file(WRITE ${KWS_STAMP} ${KWS_BIN_MTIME})
endif()

idf_component_register(
    SRCS "wake_word_main.c"
         "kws_frontend.c"
         "kws_model.c"
         "kws_int8.c"
         "kws_xtensa.S"
         "${KWS_WSRC}"
    INCLUDE_DIRS "." "model"
    REQUIRES
        esp_driver_i2s
        esp_driver_gpio
        esp_timer
        nvs_flash
        spi_flash
)