@echo off
rem Build the m1_g_wide host parity binary.
rem
rem Same reasoning as build_host.bat: gcc for the host cannot see the ESP-IDF
rem headers, and the point is that the SAME main\kws_m1gw.c the firmware builds
rem also runs off the board, so a host pass is a real pass of the real code.
rem
rem The blobs are turned into C exactly as the IDF build does, through the same
rem tools\bin2c.py, so the host is reading the same bytes the firmware will.
rem
rem   build\host\m1gw_host.exe build\host\m1gw\spec.bin
setlocal
cd /d "%~dp0\.."
if not exist build\host mkdir build\host
if not exist build\host\m1gw_gen mkdir build\host\m1gw_gen

python tools\bin2c.py main\model\kws_m1gw_i8.bin build\host\m1gw_gen\blob.c kws_m1gw_blob_data --int8
if errorlevel 1 exit /b 1
python tools\bin2c.py main\model\kws_m1gw_scales.bin build\host\m1gw_gen\scales.c kws_m1gw_scales_data
if errorlevel 1 exit /b 1

rem -msse2 -mfpmath=sse: the 32-bit MinGW default is x87, whose 80-bit registers
rem carry extra precision past the float return. SSE is also what the Xtensa
rem target's IEEE-single arithmetic looks like.
gcc -O2 -std=gnu11 -Wall -Wextra -msse2 -mfpmath=sse -DKWS_HOST_TEST ^
    -Imain -Imain\model ^
    main\kws_m1gw.c main\kws_int8.c main\kws_fastmath.c tools\m1gw_host_check.c ^
    build\host\m1gw_gen\blob.c build\host\m1gw_gen\scales.c main\model\kws_m1gw_data.c ^
    -o build\host\m1gw_host.exe -lm
if errorlevel 1 exit /b 1
echo built build\host\m1gw_host.exe

rem The end-to-end binary adds the firmware's own front end, so a real PCM clip
rem runs through push -> compute -> kws_m1gw_run exactly as kws_task does.
if not exist build\host\esp_attr mkdir build\host\esp_attr
rem kws_frontend.c guards its own esp_attr.h include with KWS_HOST_TEST, so no
rem shim header is actually needed; the directory is kept for older copies.
gcc -O2 -std=gnu11 -Wall -msse2 -mfpmath=sse -DKWS_HOST_TEST ^
    -Imain -Imain\model ^
    main\kws_m1gw.c main\kws_frontend.c main\kws_int8.c main\kws_fastmath.c tools\m1gw_e2e.c ^
    build\host\m1gw_gen\blob.c build\host\m1gw_gen\scales.c main\model\kws_m1gw_data.c ^
    -o build\host\m1gw_e2e.exe -lm
if errorlevel 1 exit /b 1
echo built build\host\m1gw_e2e.exe