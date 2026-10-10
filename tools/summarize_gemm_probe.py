#!/usr/bin/env python3
"""Markdown summary of tests/gemmul8_reuse_probe results.

usage: summarize_gemm_probe.py [--tag TAG] <dir> [<dir> ...]

Each directory holds probe<TAG>.jsonl (the --json output of the probe: wall
times, errors, bit comparisons, error indicator) and optionally
probe_prof<TAG>.jsonl (the same run of the GEMMul8_PROFILE=1 binary, for the
phase times).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def load(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fmt(value, spec='.2e'):
    return '-' if value is None else format(value, spec)


def summarize(case, tag=''):
    records = load(case / f'probe{tag}.jsonl')
    profile = load(case / f'probe_prof{tag}.jsonl')
    if not records:
        print(f'## {case.name}\n\nno probe{tag}.jsonl\n')
        return
    errors = [r for r in records if r['record'] == 'error']
    times = [r for r in records if r['record'] == 'time']
    n = errors[0]['n']
    hs = []
    for r in errors:
        if r['h'] not in hs:
            hs.append(r['h'])
    prepared = next((r['prepared_with'] for r in errors if 'prepared_with' in r), None)
    print(f"## {errors[0]['label']} (n = {n}, {len(hs)} Hamiltonians, X prepared with {prepared})\n")

    # reuse validity
    by = defaultdict(list)
    for r in errors:
        if r['path'] == 'reuse':
            by[(r['L'], r['fast'])].append(r)
    print('Bit mismatches of the reuse path against the normal path (elements of n^2 = %d):\n' % (n * n))
    print('| L | mode | prepare call B / C | same H: B / C / C (2nd GEMM alone) | other H, max: B / C / C (2nd GEMM alone) | max abs diff of C (Ha) |')
    print('|---:|---|---:|---:|---:|---:|')
    for (L, fast), rows in sorted(by.items()):
        same = [r for r in rows if r['h'] == prepared]
        other = [r for r in rows if r['h'] != prepared]
        s = same[0] if same else None
        print(f"| {L} | {'fast' if fast else 'accu'} | {rows[0]['prepare_bitsB']} / {rows[0]['prepare_bitsC']} | "
              + (f"{s['bitsB']} / {s['bitsC']} / {s['bitsC_gemm2_only']}" if s else '-') + ' | '
              + (f"{max(r['bitsB'] for r in other)} / {max(r['bitsC'] for r in other)} / {max(r['bitsC_gemm2_only'] for r in other)}" if other else '-')
              + f" | {max(r['maxdiffC'] for r in rows):.2e} |")
    print()

    # errors
    fp64 = [r for r in errors if r['path'] == 'fp64']
    ref_rel = max(r['errC_rel'] for r in fp64)
    ref_max = max(r['errC_max'] for r in fp64)
    print('Errors against the double-double reference on the sampled columns (maximum over the Hamiltonians; C = X^T H X):\n')
    print('| path | L | mode | B: relative | C: relative | C: max abs (Ha) | C relative / FP64 | asymmetry of C |')
    print('|---|---:|---|---:|---:|---:|---:|---:|')
    print(f"| cuBLAS FP64 | - | - | {max(r['errB_rel'] for r in fp64):.2e} | {ref_rel:.2e} | {ref_max:.2e} | 1.0 | {max(r['asymC_rel'] for r in fp64):.2e} |")
    table = defaultdict(list)
    for r in errors:
        if r['path'] in ('normal', 'reuse', 'blocked'):
            table[(r['L'], r['fast'], r['path'])].append(r)
    for (L, fast, path), rows in sorted(table.items()):
        rel = max(r['errC_rel'] for r in rows)
        print(f"| GEMMul8 {path} | {L} | {'fast' if fast else 'accu'} | {max(r['errB_rel'] for r in rows):.2e} | {rel:.2e} | "
              f"{max(r['errC_max'] for r in rows):.2e} | {rel / ref_rel:.3g} | {max(r['asymC_rel'] for r in rows):.2e} |")
    print()

    # error indicator (random directions), when the probe evaluated it
    if any('eta_full_b16' in r for r in errors):
        print('Error indicator eta = |C Omega - X^T (H (X Omega))|_F / |X^T (H (X Omega))|_F for the computed C '
              '(maximum over the Hamiltonians; "full" takes C as computed, "herm" the symmetric matrix built from its '
              'lower triangle; the last column is the wall time of the three thin FP64 products with 16 directions):\n')
        print('| path | L | mode | true relative error of C | eta full, 4 / 8 / 16 directions | eta herm, 16 directions | indicator (ms) |')
        print('|---|---:|---|---:|---:|---:|---:|')
        groups = defaultdict(list)
        for r in errors:
            if 'eta_full_b16' in r and r['path'] in ('fp64', 'normal'):
                groups[(r['path'] != 'fp64', r.get('L', 0), r.get('fast', 0), r['path'])].append(r)
        for (_, L, fast, path), rows in sorted(groups.items()):
            name = 'cuBLAS FP64' if path == 'fp64' else 'GEMMul8'
            print(f"| {name} | {L if path != 'fp64' else '-'} | {('fast' if fast else 'accu') if path != 'fp64' else '-'} | "
                  f"{max(r['errC_rel'] for r in rows):.2e} | {max(r['eta_full_b4'] for r in rows):.2e} / "
                  f"{max(r['eta_full_b8'] for r in rows):.2e} / {max(r['eta_full_b16'] for r in rows):.2e} | "
                  f"{max(r['eta_herm_b16'] for r in rows):.2e} | {1e3 * max(r['eta_wall_b16'] for r in rows):.3f} |")
        print()

    # times
    t64 = next((r for r in times if r['path'] == 'fp64'), None)
    pair64 = t64['wall_hx'] + t64['wall_xtb'] if t64 else None
    prof = {(r['L'], r['fast']): r for r in profile if r['record'] == 'time' and r['path'] == 'gemmul8'}
    print(f"Wall time of the two GEMMs on the last Hamiltonian (ms); cuBLAS FP64 pair: {fmt(1e3 * pair64 if pair64 else None, '.3f')}.  "
          'The scaling columns are GEMMul8 phase times of the profiling build (both GEMMs):\n')
    blocked = any(r.get('wall_hx_blocked') for r in times)
    print('| L | mode | normal | reuse | saved | prepare | normal / FP64 | scaling, normal | scaling, reuse | scaling share of normal | retained X (GiB) | scratch: normal / reuse (GiB) |'
          + (' blocked | blocked / normal | scaling, blocked |' if blocked else ''))
    print('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|' + ('---:|---:|---:|' if blocked else ''))
    for r in sorted((r for r in times if r['path'] == 'gemmul8'), key=lambda r: (r['L'], r['fast'])):
        normal = r['wall_hx_normal'] + r['wall_xtb_normal']
        reuse = r['wall_hx_reuse'] + r['wall_xtb_reuse']
        prepare = r['wall_hx_prepare'] + r['wall_xtb_prepare']
        p = prof.get((r['L'], r['fast']))
        if p and sum(p['phases_hx_normal']) > 0:
            sn = p['phases_hx_normal'][0] + p['phases_xtb_normal'][0]
            sr = p['phases_hx_reuse'][0] + p['phases_xtb_reuse'][0]
            total = sum(p['phases_hx_normal']) + sum(p['phases_xtb_normal'])
            scaling = f'{1e3 * sn:.3f} | {1e3 * sr:.3f} | {100 * sn / total:.1f}%'
        else:
            scaling = '- | - | -'
        gib = 1024.0 ** 3
        extra = ''
        if blocked:
            b = r['wall_hx_blocked'] + r['wall_xtb_blocked']
            sb = f"{1e3 * (p['phases_hx_blocked'][0] + p['phases_xtb_blocked'][0]):.3f}" if p and sum(p.get('phases_hx_blocked', [0])) > 0 else '-'
            extra = f" {1e3 * b:.3f} | {b / normal:.2f} | {sb} |"
        print(f"| {r['L']} | {'fast' if r['fast'] else 'accu'} | {1e3 * normal:.3f} | {1e3 * reuse:.3f} | {100 * (normal - reuse) / normal:.1f}% | "
              f"{1e3 * prepare:.3f} | {fmt(normal / pair64 if pair64 else None, '.2f')} | {scaling} | "
              f"{(r['bytes_keep_xr'] + r['bytes_keep_xl']) / gib:.2f} | {r['bytes_normal'] / gib:.2f} / {r['bytes_rest'] / gib:.2f} |" + extra)
    print()


def main():
    args = sys.argv[1:]
    tag = ''
    if len(args) >= 2 and args[0] == '--tag':
        tag = args[1]
        args = args[2:]
    if not args:
        sys.exit(__doc__)
    for arg in args:
        summarize(Path(arg), tag)


if __name__ == '__main__':
    main()
