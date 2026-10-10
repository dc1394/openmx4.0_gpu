#!/usr/bin/env python3
"""Markdown summary of tools/run_forward_variants.sh runs.

usage: summarize_forward_variants.py [--ref LABEL] <case dir> [<case dir> ...]

Each case directory holds one sub-directory per run variant with the OpenMX
input and output.  Every run is compared with the reference variant (default:
the first one): total energy, largest force-component difference, chemical
potential and total spin moment.  When the runs report the bridge's
forward-transform counters (OPENMX_GEMMUL8_FORWARD_TIMING=1), the wall time of
the two forward GEMMs and the preparation/reuse counts are listed too, and
with OPENMX_CLUSTER_PROFILE=1 the phases of the dense solve (collinear and
non-collinear cluster solvers).  The first solve
of a run carries one-time costs (the preparation of X, the first use of a
library routine), so the phase times are means over the later solves; the
forward transform also gets the median of the later solves, which an
occasional slow solve does not move.  Runs
of the precision controller (scf.gemmul8.adaptive) also list the accepted
SCF steps per stage and the rejected trials.  With OPENMX_GEMMUL8_TIMING=1
the runs report every GEMM routed through the GEMMul8 bridge (count, time and
operations per rank, GEMMul8 and plain cuBLAS apart), listed in a further
table.
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
        first, last, steps = {}, {}, []
        for m in re.finditer(r'CLUSTERPROF rank=(\d+) solves=(\d+) n=\d+ forward=(\S+) eigen=(\S+) back=(\S+) evec_d2h=(\S+)',
                             stds[0].read_text(errors='replace')):
            values = [int(m[2])] + [float(m[i]) for i in (3, 4, 5, 6)]
            if m[1] in last:
                steps.append(values[1] - last[m[1]][1])
            first.setdefault(m[1], values)
            last[m[1]] = values
        if last:
            later = sum(v[0] - first[k][0] for k, v in last.items())
            steps.sort()
            run['phases'] = {
                'solves': sum(v[0] for v in last.values()),
                'first': 1e3 * sum(v[1] / v[0] for v in first.values()) / len(first),
                'median': 1e3 * (steps[len(steps) // 2] if len(steps) % 2 else 0.5 * sum(steps[len(steps) // 2 - 1:len(steps) // 2 + 1]))
                          if steps else None,
                # ms per solve: over the later solves, or over all of a single-solve run
                'mean': [1e3 * sum(v[i] - first[k][i] for k, v in last.items()) / later for i in range(1, 5)] if later
                        else [1e3 * sum(v[i] for v in last.values()) / sum(v[0] for v in last.values()) for i in range(1, 5)],
            }
    # OPENMX_GEMMUL8_TIMING=1: one line per rank and report, summed per rank
    run['general'] = None
    if stds:
        per_rank, setting = {}, ''
        for m in re.finditer(r'general GEMMs, rank (\S+): real (\d+) \+ (\d+) calls, (\S+) \+ (\S+) s, (\S+) \+ (\S+) GFlop; '
                             r'complex (\d+) \+ (\d+) calls, (\S+) \+ (\S+) s, (\S+) \+ (\S+) GFlop \(GEMMul8 \+ cuBLAS\); '
                             r'moduli (\d+) / (\d+), scaling (\S+) / (\S+)', stds[0].read_text(errors='replace')):
            v = per_rank.setdefault(m[1], [0, 0, 0.0, 0.0, 0.0])
            v[0] += int(m[2]) + int(m[3])
            v[1] += int(m[8]) + int(m[9])
            v[2] += float(m[4]) + float(m[5]) + float(m[10]) + float(m[11])
            v[3] += float(m[6]) + float(m[7]) + float(m[12]) + float(m[13])
            v[4] += float(m[5]) + float(m[11])
            setting = f'L={m[14]} {m[16]} / L={m[15]} {m[17]}'
        if per_rank:
            run['general'] = {
                'real': sum(v[0] for v in per_rank.values()),
                'complex': sum(v[1] for v in per_rank.values()),
                'seconds': sum(v[2] for v in per_rank.values()),
                'busiest': max(v[2] for v in per_rank.values()),
                'gflop': sum(v[3] for v in per_rank.values()),
                'cublas_seconds': sum(v[4] for v in per_rank.values()),
                'ranks': len(per_rank),
                'setting': setting,
            }
    # OPENMX_CLUSTER_PROFILE=1 in the non-collinear cluster solver: cumulative
    # phase times of the owner rank, one line per solve (older binaries print
    # no release field)
    run['ncphases'] = None
    if stds:
        lines = list(re.finditer(r'NCCLUSTERPROF rank=\d+ solves=(\d+) n2=(\d+) maxn=(\d+) overlap=(\S+) gather=(\S+) '
                                 r'forward=(\S+) eigen=(\S+) back=(\S+)(?: release=(\S+))? dm=(\S+)',
                                 stds[0].read_text(errors='replace')))
        if lines:
            values = lambda m: [float(m[i]) if m[i] is not None else 0.0 for i in range(4, 11)]
            cumulative = [values(m) for m in lines]
            # per-solve times of the solves after the first (the first rebuilds the
            # overlap and solves all n2 states)
            later = [[b - a for a, b in zip(x, y)] for x, y in zip(cumulative, cumulative[1:])]
            median = lambda v: sorted(v)[len(v) // 2] if len(v) % 2 else 0.5 * sum(sorted(v)[len(v) // 2 - 1:len(v) // 2 + 1])
            run['ncphases'] = {
                'solves': int(lines[-1][1]),
                'first': cumulative[0],
                'maxn': (int(lines[0][3]), int(lines[-1][3])),
                'release': lines[0][9] is not None,
                'median': [1e3 * median([t[i] for t in later]) for i in range(7)] if later else None,
                'max': [1e3 * max(t[i] for t in later) for i in range(7)] if later else None,
                'sum_median': 1e3 * median([sum(t) for t in later]) if later else None,
            }
    # scf.gemmul8.adaptive: one line per trial, "... stage 1 (moduli=12 fast reuse), eta=..., rejected"
    run['stages'] = None
    if stds:
        stages, rejected, eta = {}, 0, {}
        for m in re.finditer(r'forward GEMMs: stage \d+ \(([^)]*)\)(?:, eta=\s*(\S+?))?(, rejected)?\s*$',
                             stds[0].read_text(errors='replace'), re.M):
            name = re.sub(r'moduli=(\d+) (\w)\w*.*', r'L\1\2', m[1])
            if m[3]:
                rejected += 1
            else:
                stages[name] = stages.get(name, 0) + 1
            if m[2]:
                eta[name] = max(eta.get(name, 0.0), float(m[2]))
        if stages or rejected:
            run['stages'] = {'steps': stages, 'rejected': rejected, 'eta': eta}
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
    if any(r['general'] for r in runs):
        print('GEMMs routed through the GEMMul8 bridge (OPENMX_GEMMUL8_TIMING=1; each call timed between stream '
              'synchronizations): calls and time summed over the ranks, the busiest rank, and its share of the run time:\n')
        print('| run | GEMMul8 setting (real / complex) | real calls | complex calls | ranks | GFlop | time, all ranks (s) | '
              'of which cuBLAS (s) | busiest rank (s) | busiest rank / run time | busiest rank per SCF step (ms) |')
        print('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for r in runs:
            g = r['general']
            if g:
                share = '-' if not r['time'] else f"{100 * g['busiest'] / r['time']:.1f}%"
                print(f"| {r['label']} | {g['setting']} | {g['real']} | {g['complex']} | {g['ranks']} | {g['gflop']:.1f} | "
                      f"{g['seconds']:.3f} | {g['cublas_seconds']:.3f} | {g['busiest']:.3f} | {share} | "
                      f"{1e3 * g['busiest'] / max(1, r['scf']):.2f} |")
        print()
    if any(r['stages'] for r in runs):
        print('Precision controller: accepted SCF steps per stage of the forward transform (L<moduli><a|f>: accurate or '
              'fast scaling), rejected trials, and the largest error indicator seen in each stage:\n')
        print('| run | steps per stage | rejected trials | largest indicator |')
        print('|---|---|---:|---|')
        for r in runs:
            if r['stages']:
                t = r['stages']
                print(f"| {r['label']} | " + ', '.join(f'{k}: {v}' for k, v in t['steps'].items()) + f" | {t['rejected']} | "
                      + (', '.join(f'{k}: {v:.1e}' for k, v in t['eta'].items()) or '-') + ' |')
        print()
    if any(r['phases'] for r in runs):
        print('Phases of the dense solve (OPENMX_CLUSTER_PROFILE=1), ms per solve: the forward transform of the first solve, '
              'the median of the later ones, then means over the later solves:\n')
        print('| run | solves | forward, first solve | forward, median | forward | eigensolver | back transform | eigenvector download | '
              'forward share |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
        for r in runs:
            if r['phases']:
                f, e, b, d = r['phases']['mean']
                median = '-' if r['phases']['median'] is None else format(r['phases']['median'], '.3f')
                print(f"| {r['label']} | {r['phases']['solves']} | {r['phases']['first']:.3f} | {median} | {f:.3f} | {e:.3f} | {b:.3f} | "
                      f"{d:.3f} | {100 * f / (f + e + b + d):.1f}% |")
        print()
    print_ncphases(runs)


def print_ncphases(runs):
    if not any(r.get('ncphases') for r in runs):
        return
    print('Phases of the non-collinear dense solve (OPENMX_CLUSTER_PROFILE=1, owner rank): the first solve in s (it rebuilds '
          'the transformed overlap and solves all 2n states), then ms per later solve, median (maximum); "release" is the '
          'eigenvalue download and the release of the GEMMul8 workspace:\n')
    print('| run | solves | states, first / later | first solve (s) | overlap | gather | forward transform | eigensolver | '
          'back transform | release | density matrix | sum (median) |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in runs:
        t = r.get('ncphases')
        if not t:
            continue
        if t['median'] is None:
            cells = ['-'] * 8
        else:
            cells = [f"{a:.1f} ({b:.1f})" for a, b in zip(t['median'], t['max'])] + [f"{t['sum_median']:.1f}"]
            if not t['release']:
                cells[5] = '-'
        print(f"| {r['label']} | {t['solves']} | {t['maxn'][0]} / {t['maxn'][1]} | {sum(t['first']):.2f} | " + ' | '.join(cells) + ' |')
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
