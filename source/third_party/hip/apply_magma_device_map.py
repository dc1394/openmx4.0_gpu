#!/usr/bin/env python3
"""Install the scoped logical-device map into the pinned hipMAGMA interface.

Idempotent and incremental: preserves the existing Ozaki2 patch and all
unrelated generated sources, configurations and library build products.
"""
from pathlib import Path
import sys


def main():
    root = Path(sys.argv[1])
    source = root / 'interface_hip/interface.cpp'
    original = source.read_text()
    text = original
    replacements = (
        ('#include "error.h"', '#include "error.h"\n#include "magma_device_map.h"'),
        ('    cnt = min( cnt, int(size) );',
         '    if (openmx_magma_map_count) cnt = openmx_magma_map_count;\n'
         '    cnt = min( cnt, int(size) );'),
        ('    *device = dev;', '    *device = openmx_magma_map_to_logical(dev);'),
        ('    err = hipSetDevice( int(device) );',
         '    err = hipSetDevice( openmx_magma_map_to_physical(int(device)) );'),
    )
    for old, new in replacements:
        if new in text:
            continue
        if text.count(old) != 1:
            raise RuntimeError('Unexpected hipMAGMA interface; cannot apply: ' + old)
        text = text.replace(old, new)
    header = Path(__file__).with_name('magma_device_map.h').read_bytes()
    target = source.with_name('magma_device_map.h')
    changed = not target.exists() or target.read_bytes() != header
    if changed:
        target.write_bytes(header)
    if text != original:
        source.write_text(text)
    elif changed:
        source.touch()  # MAGMA's generated dependency rules may omit this header.


if __name__ == '__main__':
    main()
