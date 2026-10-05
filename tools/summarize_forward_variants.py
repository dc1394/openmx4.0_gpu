#!/usr/bin/env python3
"""Markdown summary of tools/run_forward_variants.sh runs.

usage: summarize_forward_variants.py [--ref LABEL] <case dir> [<case dir> ...]

Each case directory holds one sub-directory per run variant with the OpenMX
input and output.  Every run is compared with the reference variant (default:
the first one): total energy, largest force-component difference, chemical
potential and total spin moment.  When the runs report the bridge's
forward-transform counters (OPENMX_GEMMUL8_FORWARD_TIMING=1), the wall time of
the two forward GEMMs and the preparation/reuse counts are listed too, and
with OPENMX_CLUSTER_PROFILE=1 the phases of the dense solve.  The first solve
of a run carries one-time costs (the preparation of X, the first use of a
library routine), so the phase times are means over the later solves.
"""
import math
import re
import sys
from pathlib import Path


def read_run(folder):
    outs = [p for p in folder.glob('*.out')]
    dats = [p for p in folder.glob('*.dat')]
    if not outs or not dats:
        return None
    text = outs[0].read_text(errors='replace')
    dat = dats[0].read_text(errors='replace')

    def key(name):
        found = re.findall(r'^\s*' + re.escape(name) + r'\s+(\S+)', dat, re.M | re.I)
        return found[-1] if found else ''

    def number(pattern):
        found = re.search(pattern, text, re.M)
        return float(found[1]) if found else None

    run = {
        'label': folder.name,
        'criterion': key('scf.criterion'),
        'gemmul8': key('scf.gemmul8.enable') or 'on',
        'max_iter': int(key('scf.maxIter') or 0),
        'scf': sum(1 for line in text.splitlines() if re.match(r'^ +SCF= +\d+ ', line)),
        'utot': number(r'^\s*Utot\.\s+(\S+)'),
        'uele': number(r'^\s*Uele\.\s+(\S+)'),
        'mu': number(r'^\s*Chemical potential \(Hartree\)\s+(\S+)'),
        'spin': number(r'^\s*Total spin moment \(muB\)\s+(\S+)'),
        'time': number(r'^\s*Elapsed\.Time\.\s+(\S+)'),
        'forces': None,
    }
    # one line per rank that ran forward products (two for separate spin worlds)
    run['forward'] = None
    stds = [p for p in folder.glob('*.std')]
    if stds:
        lines = re.findall(r'forward transform: mode=(\S+) moduli=(\d+) scaling=(\S+) reuse=(\d) unblocked=(\d); '
                           r'calls (\d+) \+ (\d+), prepared (\d+), reused (\d+), capacity fallbacks (\d+), '
                           r'retained ([\d.]+) MiB, wall (\S+) \+ (\S+) s', stds[0].read_text(errors='replace'))
        if lines:
            run['forward'] = {
                'setting': f"{lines[0][0]}" + (f" L={lines[0][1]} {lines[0][2]}" if lines[0][0] == 'gemmul8' else ''),
                'calls': sum(int(l[5]) + int(l[6]) for l in lines),
                'prepared': sum(int(l[7]) for l in lines),
                'reused': sum(int(l[8]) for l in lines),
                'fallbacks': sum(int(l[9]) for l in lines),
                'retained': sum(float(l[10]) for l in lines),
                'wall': sum(float(l[11]) + float(l[12]) for l in lines),
            }
    # OPENMX_CLUSTER_PROFILE=1: cumulative phase times, one line per solve and rank
    run['phases'] = None
    if stds:
        first, last = {}, {}
        for m in re.finditer(r'CLUSTERPROF rank=(\d+) solves=(\d+) n=\d+ forward=(\S+) eigen=(\S+) back=(\S+) evec_d2h=(\S+)',
                             stds[0].read_text(errors='replace')):
            values = [int(m[2])] + [float(m[i]) for i in (3, 4, 5, 6)]
            first.setdefault(m[1], values)
            last[m[1]] = values
        if last:
            later = sum(v[0] - first[k][0] for k, v in last.items())
            run['phases'] = {
                'solves': sum(v[0] for v in last.values()),
                'first': 1e3 * sum(v[1] / v[0] for v in first.values()) / len(first),
                # ms per solve: over the later solves, or over all of a single-solve run
                'mean': [1e3 * sum(v[i] - first[k][i] for k, v in last.items()) / later for i in range(1, 5)] if later
                        else [1e3 * sum(v[i] for v in last.values()) / sum(v[0] for v in last.values()) for i in range(1, 5)],
            }
    block = re.search(r'<coordinates.forces\s*\n\s*(\d+)\s*\n', text)
    if block:
        forces = []
        for line in text[block.end():].splitlines()[:int(block[1])]:
            forces.extend(map(float, line.split()[5:8]))
        run['forces'] = forces
    return run


def diff(a, b, spec='.1e'):
    if a is None or b is None:
        return '-'
    return format(a - b, '+' + spec)


def summarize(case, ref_label):
    runs = [r for r in (read_run(p) for p in sorted(case.iterdir()) if p.is_dir()) if r]
    if not runs:
        return
    ref = next((r for r in runs if r['label'] == ref_label), runs[0])
    print(f"## {case.name} (reference: {ref['label']})\n")
    forward = any(r['forward'] for r in runs)
    print('| run | scf.criterion | GEMMul8 | SCF steps | Utot (Ha) | dUtot (Ha) | max abs dF (Ha/bohr) | d mu (Ha) | d spin moment (muB) | time (s) |'
          + (' forward transform | forward GEMMs (s) | per SCF step (ms) | prepared / reused / fallbacks | retained X (MiB) |' if forward else ''))
    print('|---|---|---|---:|---:|---:|---:|---:|---:|---:|' + ('---|---:|---:|---:|---:|' if forward else ''))
    for r in runs:
        if r['utot'] is None:
            print(f"| {r['label']} | {r['criterion']} | {r['gemmul8']} | {r['scf']} | no result | | | | | |")
            continue
        if r['forces'] and ref['forces'] and len(r['forces']) == len(ref['forces']):
            df = format(max(abs(x - y) for x, y in zip(r['forces'], ref['forces'])), '.1e')
        else:
            df = '-'
        steps = f"{r['scf']}" + (' (cap)' if r['max_iter'] and r['scf'] >= r['max_iter'] else '')
        time = '-' if r['time'] is None else format(r['time'], '.1f')
        extra = ''
        if forward:
            f = r['forward']
            extra = (f" {f['setting']} | {f['wall']:.4f} | {1e3 * f['wall'] / max(1, r['scf']):.3f} | "
                     f"{f['prepared']} / {f['reused']} / {f['fallbacks']} | {f['retained']:.1f} |") if f else ' - | - | - | - | - |'
        print(f"| {r['label']} | {r['criterion']} | {r['gemmul8']} | {steps} | {r['utot']:.12f} | {diff(r['utot'], ref['utot'])} | "
              f"{df} | {diff(r['mu'], ref['mu'])} | {diff(r['spin'], ref['spin'])} | {time} |" + extra)
    print()
    if any(r['phases'] for r in runs):
        print('Phases of the dense solve (OPENMX_CLUSTER_PROFILE=1), ms per solve: the forward transform of the first solve, '
              'then means over the later solves:\n')
        print('| run | solves | forward, first solve | forward | eigensolver | back transform | eigenvector download | forward share |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|')
        for r in runs:
            if r['phases']:
                f, e, b, d = r['phases']['mean']
                print(f"| {r['label']} | {r['phases']['solves']} | {r['phases']['first']:.3f} | {f:.3f} | {e:.3f} | {b:.3f} | {d:.3f} | "
                      f"{100 * f / (f + e + b + d):.1f}% |")
        print()


def main():
    args = sys.argv[1:]
    ref_label = ''
    if len(args) >= 2 and args[0] == '--ref':
        ref_label = args[1]
        args = args[2:]
    if not args:
        sys.exit(__doc__)
    for arg in args:
        folder = Path(arg)
        if folder.is_dir() and any(p.is_dir() for p in folder.iterdir()):
            summarize(folder, ref_label)


if __name__ == '__main__':
    main()
