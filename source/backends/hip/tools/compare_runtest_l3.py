#!/usr/bin/env python3
"""Compare built-in test runs, including every force component (not a signed sum)."""
import argparse
import json
import math
from pathlib import Path
import re


def read_output(path):
    text = path.read_text()
    energy = float(re.search(r'^\s*Utot\.\s+(\S+)', text, re.M)[1])
    block = re.search(r'<coordinates.forces\s*\n\s*(\d+)\s*\n', text)
    count = int(block[1])
    atoms, forces = [], []
    for line in text[block.end():].splitlines()[:count]:
        fields = line.split()
        atoms.append((int(fields[0]), fields[1], *map(float, fields[2:5])))
        forces.extend(map(float, fields[5:8]))
    if len(forces) != 3 * count or not all(map(math.isfinite, [energy, *forces])):
        raise ValueError(f'Incomplete or nonfinite output: {path}')
    grids = tuple(int(re.search(rf'^\s*Num.Grid{i}\.\s+(\d+)', text, re.M)[1]) for i in (1, 2, 3))
    return energy, atoms, forces, grids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path, nargs='?')
    parser.add_argument('--reference', action='store_true',
                        help='check one run directory against the bundled references')
    parser.add_argument('--energy-tolerance', type=float, default=1e-7)
    parser.add_argument('--force-tolerance', type=float, default=1e-7)
    args = parser.parse_args()
    if args.reference == (args.candidate is not None):
        parser.error('provide two run directories, or --reference and one run directory')
    if args.reference:
        args.candidate = args.baseline
    baseline = {} if args.reference else {
        r['case']: r for r in json.loads((args.baseline / 'results.json').read_text())}
    candidate = {r['case']: r for r in json.loads((args.candidate / 'results.json').read_text())}
    input_directories = {}
    for folder in (args.baseline, args.candidate):
        metadata = folder / 'command.json'
        info = json.loads(metadata.read_text()) if metadata.exists() else {}
        input_directories[folder] = info.get('input_directory', 'large3_example')
    if input_directories[args.baseline] != input_directories[args.candidate]:
        parser.error('Cannot compare different test suites')
    failed = False
    print('| Case | Before (s) | After (s) | Speedup | abs ΔE (Ha) | Max abs ΔF (Ha/bohr) | Status |')
    print('|---|---:|---:|---:|---:|---:|---|')
    for name in sorted(baseline.keys() | candidate.keys()):
        b = {'status': 'completed'} if args.reference else baseline.get(name, {})
        c = candidate.get(name, {})
        if b.get('status') != 'completed' or c.get('status') != 'completed':
            print(f"| {name} | — | — | — | — | — | {b.get('status', 'missing')} / {c.get('status', 'missing')} |")
            failed = True
            continue
        def output(folder, reference=False):
            input_directory = input_directories[folder]
            inp = next((folder / name / input_directory).glob('*.dat'))
            system = re.search(r'^System.Name\s+(\S+)', inp.read_text(), re.M | re.I)[1]
            parent = folder / name
            if reference:
                parent /= input_directory
            return parent / (system + '.out')
        be, ba, bf, bg = read_output(output(args.baseline, args.reference))
        ce, ca, cf, cg = read_output(output(args.candidate))
        # Bundled references print coordinates to five decimal places and
        # occasionally round the last digit differently from this release.
        position_tolerance = 1.000001e-5 if args.reference else 0.0
        same_atoms = len(ba) == len(ca) and all(
            x[:2] == y[:2] and all(abs(u-v) <= position_tolerance
                                  for u, v in zip(x[2:], y[2:]))
            for x, y in zip(ba, ca))
        if not same_atoms or bg != cg:
            raise ValueError(f'{name}: atom positions or integration grids differ')
        de = abs(be - ce)
        df = max(abs(x-y) for x, y in zip(bf, cf))
        ok = de <= args.energy_tolerance and df <= args.force_tolerance
        failed |= not ok
        ct = c['elapsed_seconds']
        bt = 'reference' if args.reference else f"{b['elapsed_seconds']:.2f}"
        speedup = '—' if args.reference else f"{b['elapsed_seconds']/ct:.2f}×"
        print(f"| {name} | {bt} | {ct:.2f} | {speedup} | {de:.3g} | {df:.3g} | {'OK' if ok else 'FAIL'} |")
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
