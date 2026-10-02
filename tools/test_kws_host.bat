@echo off
REM Build and run the host validation for the Amaze front end and network.
REM
REM Compiles the exact firmware sources for the host, binds the weights from
REM main/model/kws_model.bin, and compares against the golden tensors produced
REM by tools/export_weights.py --dump.
REM
REM Usage, from the repo root:
REM     tools\test_kws_host.bat
REM
REM Any diff here is a porting bug. Do not flash until it passes.

setlocal
cd /d "%~dp0\.."

set GCC=C:\MinGW\bin\gcc.exe
if not exist "%GCC%" (
    echo gcc not found at %GCC%
    exit /b 2
)

if not exist "main\model\kws_model.bin" (
    echo kws_model.bin missing - run:
    echo   python tools\export_weights.py --dump
    exit /b 2
)

if not exist "main\model\reference" (
    echo reference tensors missing - run:
    echo   python tools\export_weights.py --dump
    exit /b 2
)

echo === building host test ===
%GCC% -O2 -std=gnu11 -Wall -Wextra -Wno-unused-parameter ^
    -Imain -Imain\model ^
    main\kws_frontend.c main\kws_model.c tools\test_kws_host.c ^
    -o build_host_test.exe -lm
if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)

echo === running ===
build_host_test.exe %1
exit /b %errorlevel%