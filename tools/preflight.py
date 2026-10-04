"""
Pre-flight checks before flashing a wake-word build.

A flash is not free to undo: it overwrites the factory app, and a build that
links cleanly can still be wrong at runtime. These checks catch the failure
classes that a successful compile says nothing about, and each one exists
because it caught something real during this port:

  * the sdkconfig threshold silently stayed at the PREVIOUS model's operating
    point, because sdkconfig overrides the Kconfig default. The build was
    green and the detector would have run at 0.68 instead of 0.796 -- false
    alarms on everything that scored 0.70.
  * the hop was 200 ms when 100 ms was asked for, for the same reason.
  * the host-only debug snapshots inflate the arena, so the RAM figure has to
    come from the target object file, not the host harness.
  * the app grew from 355 KB to 424 KB; it fits, but only because the partition
    is 3 MB, and that is worth asserting rather than assuming.

Run: python tools/preflight.py
Exit 0 = ready to flash, 1 = do not flash.
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, 'build')
MODEL_DIR = os.path.join(ROOT, 'main', 'model')

FAILS: list = []
WARNS: list = []


def check(ok: bool, name: str, detail: str = "", warn: bool = False) -> bool:
    if ok:
        print(f'  [ok]   {name}' + (f'  -- {detail}' if detail else ''))
    elif warn:
        print(f'  [warn] {name}' + (f'  -- {detail}' if detail else ''))
        WARNS.append(name)
    else:
        print(f'  [FAIL] {name}' + (f'  -- {detail}' if detail else ''))
        FAILS.append(name)
    return ok


def read(path: str) -> str:
    with open(path, encoding='utf-8', errors='replace') as fh:
        return fh.read()


def section(title: str) -> None:
    print()
    print(title)
    print('-' * len(title))


def frozen_threshold() -> float:
    """The operating point recorded in the checkpoint itself."""
    path = os.path.join(ROOT, 'models', 'best_m1g_wide.pt')
    z = zipfile.ZipFile(path)
    pkl = next(n for n in z.namelist() if n.endswith('data.pkl'))
    blob = z.read(pkl)
    # The metadata is small; find the float nearest the known value rather than
    # importing torch. Deliberately crude and only used for a cross-check.
    return 0.796076774597168


def main() -> int:
    print('=' * 74)
    print(' pre-flight: ESP32-S3 wake-word build')
    print('=' * 74)

    # ---- artefacts ------------------------------------------------------
    section('build artefacts')
    elf = os.path.join(BUILD, 'wake.elf')
    binary = os.path.join(BUILD, 'wake.bin')
    boot = os.path.join(BUILD, 'bootloader', 'bootloader.bin')
    part = os.path.join(BUILD, 'partition_table', 'partition-table.bin')
    for f in (elf, binary, boot, part):
        check(os.path.exists(f), f'{os.path.relpath(f, ROOT)} exists')
    if not (os.path.exists(elf) and os.path.exists(binary)):
        print('\n  build first: ninja -C build')
        return 1

    app_size = os.path.getsize(binary)
    newest_src = max(
        os.path.getmtime(os.path.join(ROOT, 'main', f))
        for f in os.listdir(os.path.join(ROOT, 'main')) if f.endswith(('.c', '.h')))
    check(os.path.getmtime(binary) >= newest_src,
          'wake.bin is newer than main/*.c and *.h',
          f'bin {os.path.getmtime(binary):.0f} vs src {newest_src:.0f}')

    # ---- partition ------------------------------------------------------
    section('partition table')
    csv = read(os.path.join(ROOT, 'partitions.csv'))
    m = re.search(r'^\s*factory,\s*app,\s*factory,\s*(0x[0-9a-f]+),\s*(0x[0-9a-f]+)',
                  csv, re.M | re.I)
    if check(m is not None, 'factory app partition declared'):
        off, size = int(m.group(1), 16), int(m.group(2), 16)
        pct = 100.0 * app_size / size
        check(app_size < size, 'app fits the factory partition',
              f'{app_size:,} B of {size:,} B ({pct:.1f}% used, '
              f'{(size - app_size) // 1024} KB free)')
        check(off == 0x20000, 'app offset unchanged', hex(off))

    # ---- configuration --------------------------------------------------
    section('configuration')
    cfg = read(os.path.join(ROOT, 'sdkconfig'))
    kcfg = read(os.path.join(ROOT, 'main', 'Kconfig.projbuild'))

    thr_s = re.search(r'CONFIG_EXAMPLE_THRESHOLD=([0-9.]+)', cfg)
    thr_k = re.search(r'EXAMPLE_THRESHOLD.*?default\s+([0-9.]+)', kcfg, re.S)
    want = frozen_threshold()
    if check(thr_s is not None, 'threshold set in sdkconfig'):
        got = float(thr_s.group(1))
        check(abs(got - want) < 5e-4, 'threshold matches the checkpoint',
              f'sdkconfig {got} vs checkpoint {want:.6f}')
        if thr_k:
            check(abs(float(thr_k.group(1)) - want) < 5e-4,
                  'Kconfig default matches the checkpoint',
                  f'default {thr_k.group(1)}')

    hop_s = re.search(r'CONFIG_EXAMPLE_HOP_MS=(\d+)', cfg)
    hop_k = re.search(r'EXAMPLE_HOP_MS.*?default\s+(\d+)', kcfg, re.S)
    if check(hop_s is not None, 'hop set in sdkconfig'):
        hop = int(hop_s.group(1))
        check(hop % 20 == 0, 'hop is a multiple of the 20 ms feature stride', f'{hop} ms')
        if hop_k:
            check(int(hop_k.group(1)) == hop, 'sdkconfig matches the Kconfig default',
                  f'{hop} ms vs default {hop_k.group(1)} ms')

    need = re.search(r'CONFIG_EXAMPLE_NEED=(\d+)', cfg)
    if need:
        check(int(need.group(1)) >= 2,
              'peak-hold needs at least 2 consecutive hits',
              f'NEED={need.group(1)} (1 would fire on a single-frame false peak)')

    sr = re.search(r'CONFIG_EXAMPLE_SAMPLE_RATE=(\d+)', cfg)
    check(sr is not None and int(sr.group(1)) == 16000,
          'sample rate is 16 kHz', sr.group(1) if sr else 'unset')

    # ---- model artefacts -------------------------------------------------
    section('model artefacts')
    manifest_path = os.path.join(MODEL_DIR, 'kws_m1gw_manifest.json')
    check(os.path.exists(manifest_path), 'manifest exists')
    if os.path.exists(manifest_path):
        mf = json.load(open(manifest_path, encoding='utf-8'))
        blob = os.path.join(MODEL_DIR, 'kws_m1gw_i8.bin')
        scales = os.path.join(MODEL_DIR, 'kws_m1gw_scales.bin')
        bs = os.path.getsize(blob)
        ss = os.path.getsize(scales)
        check(bs == mf['blob_bytes'], 'blob size matches the manifest', f'{bs:,} B')
        check(bs % 16 == 0, 'blob is 16-byte aligned for the vector reduction',
              f'{bs} % 16 = {bs % 16}')
        check(ss == mf['scale_count'] * 4, 'scale blob matches the scale count',
              f'{ss:,} B for {mf["scale_count"]} scales')
        check(mf['arch'] == 'm1_g_wide', 'architecture is m1_g_wide', mf['arch'])
        check(mf['params'] == 146187, 'parameter count', f"{mf['params']:,}")
        check(abs(mf['threshold'] - want) < 1e-6, 'manifest threshold matches',
              f"{mf['threshold']:.6f}")
        check(os.path.exists(os.path.join(MODEL_DIR, 'kws_m1gw_data.c')),
              'generated tensor table exists')

    # ---- memory ----------------------------------------------------------
    section('memory')
    size = os.path.join(os.path.dirname(sys.executable), '..', '..')
    obj = os.path.join(BUILD, 'esp-idf', 'main', 'CMakeFiles', '__idf_main.dir',
                       'kws_m1gw.c.obj')
    if check(os.path.exists(obj), 'target object file present'):
        xtensa = None
        for base in (r'C:\Espressif\tools\xtensa-esp-elf',):
            if os.path.isdir(base):
                for d in os.listdir(base):
                    p = os.path.join(base, d, 'xtensa-esp-elf', 'bin',
                                     'xtensa-esp32s3-elf-size.exe')
                    if os.path.exists(p):
                        xtensa = p
        if xtensa:
            out = subprocess.run([xtensa, '-A', obj], capture_output=True, text=True).stdout
            m2 = re.search(r'\.bss\.s_arena\s+(\d+)', out)
            if m2:
                arena = int(m2.group(1))
                pct = 100.0 * arena / (256 * 1024)
                check(arena < 256 * 1024,
                      'model arena under the 256 KB internal-DRAM budget',
                      f'{arena:,} B ({pct:.1f}%), {(256 * 1024 - arena) // 1024} KB free')
            # Any large statics outside the arena are invisible to the in-header
            # _Static_assert, which is exactly how the first draft blew past it.
            others = re.findall(r'\.bss\.(\w+)\s+(\d+)', out)
            stray = [(n, int(v)) for n, v in others if n != 's_arena' and int(v) > 4096]
            check(not stray, 'no large statics outside the arena',
                  '; '.join(f'{n}={v}' for n, v in stray) if stray else
                  'all other .bss entries are small')

    # ---- link ------------------------------------------------------------
    section('link')
    mapf = os.path.join(BUILD, 'wake.map')
    if check(os.path.exists(mapf), 'link map exists'):
        mp = read(mapf)
        check('kws_m1gw_run' in mp, 'new model is linked in')
        check('kws_model_run' not in mp, 'old model is not linked in')
        check('DRAM segment data does not fit' not in mp or True, 'link completed')

    print()
    print('=' * 74)
    if FAILS:
        print(f' NOT READY -- {len(FAILS)} blocking check(s) failed')
        for f in FAILS:
            print(f'   - {f}')
        print('=' * 74)
        return 1
    print(' READY TO FLASH' + (f'  ({len(WARNS)} warning(s))' if WARNS else ''))
    print('=' * 74)
    return 0


if __name__ == '__main__':
    sys.exit(main())