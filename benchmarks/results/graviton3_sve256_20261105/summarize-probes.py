#!/usr/bin/env python3
"""Extract representative measured cases; never substitute these for full runs."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent

def read(name):
    records = list(csv.DictReader((root / name).open()))
    assert all(x['bitwise_equal'] == '1' for x in records)
    return records

def pick(records, **keys):
    chosen = [x for x in records if all(str(v) == x[k] for k, v in keys.items())]
    assert len(chosen) == 1, (keys, len(chosen))
    return float(chosen[0]['seconds'])

grid = read('grid-kernels-v4.csv')
force = read('force-kernels.csv')
force2 = read('force-kernels-v2.csv')
fg = read('force-grid-kernels.csv')
cases = []

def add(name, a, b, shape):
    cases.append(dict(kernel=name, before_seconds=a, after_seconds=b,
                      speedup=a / b, shape=shape))

for stage, new in [('density', 8), ('hamiltonian', 6)]:
    keys = dict(stage=stage, rows=13, columns=13, spins=4)
    add('NC ' + stage, pick(grid, method=1, **keys), pick(grid, method=new, **keys),
        '13x13, 2003 grid points, 4 components')
for terms, method in [(1, 4), (2, 5)]:
    for orientation in [0, 1]:
        keys = dict(rows=13, columns=13, projectors=972, fp32=1,
                    terms=terms, lanes_rows=orientation)
        add(f'VNA FP32 terms={terms} lanes_rows={orientation}',
            pick(force, method=0, **keys), pick(force2, method=method, **keys),
            '13x13, 972 projectors')
add('force density contraction', pick(fg, rows=13, columns=13, method=0),
    pick(fg, rows=13, columns=13, method=3), '13x13')

summary = dict(bitwise_case_counts={'grid-v4': len(grid), 'VNA-v1': len(force),
                                    'VNA-v2': len(force2), 'force-density': len(fg)},
               representative_cases=cases)
(root / 'own-kernel-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
