<#
.SYNOPSIS
    Pull recorded wake-word clips off the ESP32-S3 storage partition.

.DESCRIPTION
    Reads the whole "storage" partition in one esptool pass, then scans for
    valid RIFF/WAVE headers and writes each one out as clipNN.wav.

    Slot geometry must match main/wake_word_main.c:
      base      = 0x520000
      slot_size = ALIGN_UP(44 + 16000 * 2 * CLIP_SECONDS, 4096)
      clip_len  = 44 + 16000 * 2 * CLIP_SECONDS

.PARAMETER Port
    Serial port, e.g. COM15.

.PARAMETER OutDir
    Directory to write the .wav files into. Created if missing.

.PARAMETER ClipSeconds
    Must match CONFIG_EXAMPLE_CLIP_SECONDS (default 10).

.EXAMPLE
    .\tools\pull_clips.ps1 -Port COM15
#>
[CmdletBinding()]
param(
    [string]$Port = "COM15",
    [string]$OutDir = ".\clips",
    [int]$ClipSeconds = 10,
    [switch]$KeepImage
)

$ErrorActionPreference = "Stop"

$SampleRate = 16000
$Channels    = 1
$BitsPerSamp = 16
$ClipLen     = 44 + ($SampleRate * $Channels * $BitsPerSamp / 8) * $ClipSeconds
$SlotSize    = [int][math]::Ceiling($ClipLen / 4096) * 4096
$PartSize    = 0x800000
$PartBase    = 0x520000
# Floor, not an [int] cast: PowerShell's [int] rounds (25.9 -> 26) while C truncates,
# which would make this script scan a slot past the end of the partition.
$SlotCount   = [int][math]::Floor($PartSize / $SlotSize)

# NOTE: the format operator must be parenthesised here. Written as
#   Write-Host "x {0}" -f $v
# PowerShell binds -f to -ForegroundColor (it uniquely matches that parameter)
# and fails with "Cannot convert value 5373952 to System.ConsoleColor".

Write-Host ("storage partition : 0x{0:X6}  ({1} KB)" -f $PartBase, ($PartSize / 1024))
Write-Host ("clip length       : {0} bytes (0x{1:X})" -f $ClipLen, $ClipLen)
Write-Host ("slot size         : {0} bytes (0x{1:X})" -f $SlotSize, $SlotSize)
Write-Host ("slots in partition: {0}" -f $SlotCount)
Write-Host ""

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}
$OutDir = (Resolve-Path $OutDir).Path

$image = Join-Path $env:TEMP "wake_storage.bin"

Write-Host ("Reading {0} KB from 0x{1:X6} on {2} ..." -f ($PartSize / 1024), $PartBase, $Port)

# Use the module form: the bare `esptool.py` wrapper is deprecated and prints a
# warning to stderr, which trips $ErrorActionPreference = "Stop" on PowerShell
# 5.1 and aborts the read. Also temporarily relax the preference for the call
# itself, since esptool writes progress to stderr.
$ErrorActionPreference = "Continue"
& python -m esptool --chip esp32s3 -p $Port read_flash $PartBase $PartSize $image
$esptoolExit = $LASTEXITCODE
$ErrorActionPreference = "Stop"

if ($esptoolExit -ne 0) {
    throw "esptool read_flash failed with exit code $esptoolExit"
}

$bytes = [System.IO.File]::ReadAllBytes($image)
Write-Host ("Read {0} bytes, scanning for RIFF/WAVE headers ..." -f $bytes.Length)
Write-Host ""

$found = 0
for ($slot = 0; $slot -lt $SlotCount; $slot++) {

    $offset = $slot * $SlotSize
    if (($offset + $ClipLen) -gt $bytes.Length) { break }

    # "RIFF" then "WAVE" at +8
    if ($bytes[$offset]     -ne 0x52 -or $bytes[$offset + 1] -ne 0x49 -or
        $bytes[$offset + 2] -ne 0x46 -or $bytes[$offset + 3] -ne 0x46 -or
        $bytes[$offset + 8] -ne 0x57 -or $bytes[$offset + 9] -ne 0x41 -or
        $bytes[$offset+10] -ne 0x56 -or $bytes[$offset+11] -ne 0x45) {
        continue
    }

    # data chunk size at +40 tells us how much real audio is in here
    $dataLen = [BitConverter]::ToUInt32($bytes, $offset + 40)
    $write   = if ($dataLen + 44 -gt $ClipLen) { $ClipLen } else { $dataLen + 44 }

    $name = Join-Path $OutDir ("slot{0:D2}.wav" -f $slot)
    [System.IO.File]::WriteAllBytes($name, $bytes[$offset..($offset + $write - 1)])

    Write-Host ("  slot {0,-2}  flash 0x{1:X6}  {2,6:N1} s audio -> {3}" -f `
        $slot, ($PartBase + $offset), ($dataLen / 2 / $SampleRate), (Split-Path $name -Leaf))
    $found++
}

Write-Host ""
if ($found -eq 0) {
    Write-Host "No clips found. Either nothing has been detected yet, or the"
    Write-Host "partition layout differs - check the 'storage:' line in the serial log."
} else {
    Write-Host ("Wrote {0} clip(s) to {1}" -f $found, $OutDir)
}

if (-not $KeepImage) {
    Remove-Item $image -Force -ErrorAction SilentlyContinue
}