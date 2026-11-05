#!/usr/bin/env python3
"""Rebuild and rerun the frozen SVE256 kernel experiments on this machine.

Do not run concurrently with an OpenMX benchmark. Each output is new; original
measurements and compiler commands are retained unchanged.
"""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
for name in ('grid-kernel-probe-v4', 'force-kernel-probe-v1',
             'force-kernel-probe-v2', 'force-grid-probe'):
    command = json.loads((root / (name + '-reproduce.json')).read_text())
    output = root / (name + '-reproduced.csv')
    with output.open('x') as stream:
        subprocess.run(command, check=True)
        subprocess.run(['taskset', '-c', '0', command[-1]], stdout=stream, check=True)
    print(output, flush=True)
