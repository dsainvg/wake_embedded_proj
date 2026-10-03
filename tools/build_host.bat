@echo off
rem Build the host validation binary.
rem
rem Separate from the IDF build on purpose: gcc for the host cannot see the
rem ESP-IDF headers, and the point of this build is that the SAME inference
rem source compiles and runs off the board, so a host pass is a real pass.
rem
rem   build\host\kws_host.exe            all fixtures
rem   build\host\kws_host.exe noise      one fixture
rem
setlocal
cd /d "%~dp0\.."
if not exist build\host mkdir build\host

rem -msse2 -mfpmath=sse: the 32-bit MinGW default is x87, whose 80-bit
rem registers carry extra precision past the float return, which makes the
rem harness's own bit-identical determinism check misfire on compiler
rem artifacts. SSE is also what the Xtensa target's IEEE-single math looks like.
gcc -O2 -std=gnu11 -Wall -Wextra -msse2 -mfpmath=sse -DKWS_HOST_TEST ^
    -Imain -Imain\model ^
    main\kws_model.c main\kws_int8.c tools\host_check.c ^
    -o build\host\kws_host.exe -lm
if errorlevel 1 exit /b 1
echo built build\host\kws_host.exe