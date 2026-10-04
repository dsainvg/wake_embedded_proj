# The bcconformer_v3 blob is gone: kws_model.c is deleted and m1_g_wide
# replaces it. Its bin2c step, the weight target and the stamp went with it.

set(KWS_BIN2C  ${CMAKE_CURRENT_LIST_DIR}/../tools/bin2c.py)
set(KWS_M1GW_BIN    ${CMAKE_CURRENT_LIST_DIR}/model/kws_m1gw_i8.bin)
set(KWS_M1GW_SCBIN  ${CMAKE_CURRENT_LIST_DIR}/model/kws_m1gw_scales.bin)
set(KWS_M1GW_WSRC   ${CMAKE_CURRENT_BINARY_DIR}/kws_m1gw_blob.c)
set(KWS_M1GW_SCSRC  ${CMAKE_CURRENT_BINARY_DIR}/kws_m1gw_scales_gen.c)
set(KWS_M1GW_STAMP  ${CMAKE_CURRENT_BINARY_DIR}/kws_m1gw.stamp)

# --- m1_g_wide: int8 weights + per-channel dequant scales -------------------
file(TIMESTAMP ${KWS_M1GW_BIN} KWS_M1GW_MTIME "%Y%m%d%H%M%S")
file(TIMESTAMP ${KWS_M1GW_SCBIN} KWS_M1GW_SCMTIME "%Y%m%d%H%M%S")
set(KWS_M1GW_NEED_GEN TRUE)
if(EXISTS ${KWS_M1GW_STAMP})
    file(READ ${KWS_M1GW_STAMP} KWS_M1GW_STAMPED)
    if(KWS_M1GW_STAMPED STREQUAL KWS_M1GW_MTIME${KWS_M1GW_SCMTIME}
       AND EXISTS ${KWS_M1GW_WSRC} AND EXISTS ${KWS_M1GW_SCSRC})
        set(KWS_M1GW_NEED_GEN FALSE)
    endif()
endif()

if(KWS_M1GW_NEED_GEN)
    execute_process(
        COMMAND ${PYTHON} ${KWS_BIN2C} ${KWS_M1GW_BIN} ${KWS_M1GW_WSRC} kws_m1gw_blob_data --int8
        RESULT_VARIABLE KWS_M1GW_RC OUTPUT_VARIABLE KWS_M1GW_OUT ERROR_VARIABLE KWS_M1GW_ERR)
    if(NOT KWS_M1GW_RC EQUAL 0)
        message(FATAL_ERROR "bin2c failed on the m1_g_wide blob (${KWS_M1GW_RC}): ${KWS_M1GW_ERR}")
    endif()
    execute_process(
        COMMAND ${PYTHON} ${KWS_BIN2C} ${KWS_M1GW_SCBIN} ${KWS_M1GW_SCSRC} kws_m1gw_scales_data
        RESULT_VARIABLE KWS_M1GW_RC2 OUTPUT_VARIABLE KWS_M1GW_OUT2 ERROR_VARIABLE KWS_M1GW_ERR2)
    if(NOT KWS_M1GW_RC2 EQUAL 0)
        message(FATAL_ERROR "bin2c failed on the m1_g_wide scales (${KWS_M1GW_RC2}): ${KWS_M1GW_ERR2}")
    endif()
    message(STATUS "${KWS_M1GW_OUT}${KWS_M1GW_OUT2}")
    file(WRITE ${KWS_M1GW_STAMP} ${KWS_M1GW_MTIME}${KWS_M1GW_SCMTIME})
endif()

idf_component_register(
    SRCS "wake_word_main.c"
         "kws_frontend.c"
         "kws_fastmath.c"
         "kws_int8.c"
         "kws_xtensa.S"
         "kws_m1gw.c"
         "model/kws_m1gw_data.c"
         "${KWS_M1GW_WSRC}"
         "${KWS_M1GW_SCSRC}"
    INCLUDE_DIRS "." "model"
    REQUIRES
        esp_driver_i2s
        esp_driver_gpio
        esp_timer
        nvs_flash
        spi_flash
)