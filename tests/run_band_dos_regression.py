#!/usr/bin/env python3
"""Compare the packed DOS and Fermi mesh with the previous band output."""

import argparse
from array import array
import math
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys

from run_scf_output_regression import input_text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from exx_output_checks import read_bxsf
from run_gpu_suites import stop_job

CASES = {
    'tetra': ('Cafcc_FS.dat', False, '4 3 5'),
    'gauss': ('Cafcc_FS.dat', True, '4 3 5'),
    'spin': ('Febcc2.dat', False, '3 2 4'),
    'nc-tetra': ('FePt-NC-SCF.dat', False, '3 2 4'),
    'nc-gauss': ('FePt-NC-SCF.dat', True, '3 2 4'),
}


def read_dos(folder):
    text = (folder / 'result.Dos.val').read_text()
    fields = {name: tuple(map(int, re.search(rf'^{name}\s+([\d ]+)$', text, re.M)[1].split()))
              for name in ('mode', 'NonCol', 'N', 'Nspin', 'irange', 'Kgrid')}
    rows = [line.split() for line in text.split('<Eigenvalues\n', 1)[1].split('Eigenvalues>', 1)[0].splitlines()]
    coordinates = [tuple(map(int, line[:3])) for line in rows]
    levels = [list(map(float, line[3:])) for line in rows]
    width = fields['N'][0] * (1 + fields['NonCol'][0])
    records = struct.Struct(f'=3i{width}f')
    weights = []
    for i, j, k, *values in records.iter_unpack((folder / 'result.Dos.vec').read_bytes()):
        weights.append(((i, j, k), values))
    return fields, coordinates, levels, weights


def require_close(actual, expected, tolerance, label):
    difference = max(abs(a-b) for a, b in zip(actual, expected, strict=True))
    assert all(math.isfinite(v) for v in actual) and difference <= tolerance, (label, difference)
    return difference


def compare_dos(reference, result, gaussian):
    fields, coords, levels, weights = read_dos(result)
    ref_fields, ref_coords, ref_levels, ref_weights = read_dos(reference)
    assert fields == ref_fields and coords == ref_coords and len(ref_weights) == len(weights), result
    energy_error = max(require_close(a, b, 2e-6, result) for a, b in zip(levels, ref_levels, strict=True))
    weight_error = 0.0
    if gaussian:
        for (ijk, row), (ref_ijk, ref_row) in zip(weights, ref_weights, strict=True):
            assert ijk == ref_ijk == (0, 0, 0), result
            weight_error = max(weight_error, require_close(row, ref_row, 2e-4, result))
    else:
        metadata = (reference / 'result.Dos.val').read_text()
        chemical = float(re.search(r'^ChemP\s+(\S+)', metadata, re.M)[1])
        lo, hi = map(float, re.search(r'^Erange\s+(\S+)\s+(\S+)', metadata, re.M).groups())
        band_min, band_max = fields['irange']
        order = len(weights[0][1])
        if fields['NonCol'][0]:
            assert coords[::2] == coords[1::2] and levels[::2] == levels[1::2], result
            assert ref_coords[::2] == ref_coords[1::2] and ref_levels[::2] == ref_levels[1::2], reference
            coords, ref_levels = coords[::2], ref_levels[::2]
        offset = 0
        for ijk, eigen in zip(coords, ref_levels, strict=True):
            first = 0
            while first < len(eigen):
                end = first+1
                # Individual eigenvectors can rotate inside a degenerate eigenspace.
                while end < len(eigen) and abs(eigen[end]-eigen[first]) <= 1.1e-6:
                    end += 1
                old = ref_weights[offset+first:offset+end]
                new = weights[offset+first:offset+end]
                assert all(coord == ijk for coord, _ in [*old, *new]), result
                for _, row in new:
                    assert abs(sum(row)-1) < 1e-5, (result, ijk, sum(row))
                # The padding at a truncated spectrum edge can publish only
                # part of a degenerate multiplet outside the requested range.
                truncated_edge = (first == 0 and band_min > 1 and eigen[end-1] < chemical+lo
                                  or end == len(eigen) and band_max < order and eigen[first] > chemical+hi)
                if not truncated_edge:
                    a = [sum(values) for values in zip(*(row for _, row in new))]
                    b = [sum(values) for values in zip(*(row for _, row in old))]
                    weight_error = max(weight_error, require_close(a, b, 2e-4, (result, ijk, first, end)))
                first = end
            offset += len(eigen)
        assert offset == len(weights), result
    return energy_error, weight_error


def compare_surfaces(reference, result):
    files = sorted(result.glob('*.bxsf'))
    assert files and [p.name for p in files] == sorted(p.name for p in reference.glob('*.bxsf')), result
    error = 0.0
    for path in files:
        old, new = read_bxsf(reference / path.name), read_bxsf(path)
        assert old['grid'] == new['grid'] and old['labels'] == new['labels'], path
        for ijk, row in new['values'].items():
            error = max(error, require_close(row, old['values'][ijk], 2.1e-5, path))
    return error


def check_atomic_projection(folder, noncollinear, polarized, atom_count):
    for spin in range(2 if polarized else 1):
        total = None
        for atom in range(1, atom_count+1):
            suffix = f'_a{atom}.frmsf' if noncollinear else f'_s{spin}_a{atom}.frmsf'
            path = folder / ('result.FermiSurf' + suffix)
            lines = path.read_text().splitlines()
            count = math.prod(map(int, lines[0].split())) * int(lines[2])
            assert count > 0 and len(lines) == 6+2*count, path
            values = array('d', map(float, lines[6+count:]))
            assert all(math.isfinite(v) for v in values), path
            total = values if total is None else array('d', (a+b for a,b in zip(total, values, strict=True)))
        require_close(total, [1.0]*len(total), 1e-5, folder)


def simulate(binary, folder, text, environment, command, threads):
    folder.mkdir(parents=True)
    (folder / 'input.dat').write_text(text)
    log_path = folder / 'run.log'
    with log_path.open('w') as log:
        job = subprocess.Popen([*command, str(binary), 'input.dat', '-nt', str(threads)], cwd=folder,
                               env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            job.wait(timeout=240)
        except BaseException:
            stop_job(job)
            raise
    log = log_path.read_text()
    assert job.returncode == 0 and 'The calculation was normally finished.' in log, log_path
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True, help='OpenMX build before the DOS change')
    parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--simd', choices=('avx2', 'avx512', 'auto'), default='auto')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launcher', default='mpirun')
    parser.add_argument('--ranks', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('cases', choices=tuple(CASES), nargs='*')
    args = parser.parse_args()
    binary, reference = args.binary.resolve(strict=True), args.reference.resolve(strict=True)
    gpu = args.backend == 'cuda'
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('OPENMX_', 'OMP_', 'MKL_', 'KMP_'))}
    environment.update(OPENMX_GPU=str(int(gpu)), OPENMX_BAND_GPU_SWITCH_NUM='1', OPENMX_HOST_SIMD=args.simd,
                       OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC='FALSE', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = shlex.split(args.launcher) + ['-np', str(args.ranks)]

    for name in args.cases or CASES:
        sample, gaussian, grid = CASES[name]
        template = (ROOT / 'work' / sample).read_text()
        atoms = int(re.search(r'^Atoms.Number\s+(\d+)', template, re.M)[1])
        settings = {'DATA.PATH': ROOT / 'DFT_DATA19', 'System.Name': 'result',
                    'level.of.stdout': 1, 'level.of.fileout': 0, 'scf.restart': 'off',
                    'scf.eigen.lib': 'gpusolver' if gpu else 'elpa2', 'scf.Kgrid': '2 2 2',
                    'scf.maxIter': 5, 'MD.Type': 'nomd', 'MD.maxIter': 1,
                    'Band.dispersion': 'off', 'MO.fileout': 'off',
                    'Dos.fileout': 'off' if gaussian else 'on', 'DosGauss.fileout': 'on' if gaussian else 'off',
                    'DosGauss.Num.Mesh': 72, 'DosGauss.Width': 0.35, 'Dos.Kgrid': grid,
                    'Dos.Erange': '-15 12' if gaussian else '-60 100',
                    'FermiSurfer.fileout': 'off' if gaussian else 'on'}
        text = input_text(template, settings)
        old = args.output / name / 'reference'
        new = args.output / name / args.backend
        simulate(reference, old, text, environment, command, args.threads)
        log = simulate(binary, new, text, environment, command, args.threads)
        backend = 'GPU owner(s) with CUDA' if gpu else 'CPU rank(s) with '
        assert re.search(r'^<Band_DFT_Dosout> DOS and Fermi surface: .*' + re.escape(backend), log, re.M), new
        energy, projection = compare_dos(old, new, gaussian)
        bxsf = compare_surfaces(old, new)
        if not gaussian:
            check_atomic_projection(new, name.startswith('nc-'), name == 'spin', atoms)
        print(f'PASS {args.backend} {name} Dos.val={energy:.3g} Dos.vec={projection:.3g} BXSF={bxsf:.3g}', flush=True)


if __name__ == '__main__':
    main()
