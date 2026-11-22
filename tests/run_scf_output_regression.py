#!/usr/bin/env python3
"""Exercise OpenMX forum reports 3559–3562 with the real MPI application.

Uses the O2 geometry in work/O2+0.dat and the bundled O6.0 basis. The
three-step calculations check input validation and output, including the
second SCF step, without requiring converged energies.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from run_gpu_suites import stop_job


def input_text(template, settings, *, large_basis=False, mo_points=False):
    keys = {key.lower() for key in settings}
    lines = [line for line in template.splitlines()
             if not line.split() or line.split()[0].lower() not in keys]
    lines.extend(f'{key} {value}' for key, value in settings.items())
    if mo_points:
        lines += ['<MO.kpoint', '0 0 0', '0.25 0 0', '0.5 0 0', 'MO.kpoint>']
    text = '\n'.join(lines) + '\n'
    if large_basis:
        text = text.replace('O7.0-s3p3d2f1', 'O6.0-s7p6d5f4')
    return text


def check_scfout(path, original_input):
    data = path.read_bytes()
    header = struct.Struct('=7i')
    atoms, spin_version, _, _, _, last_cell, order = header.unpack_from(data)
    assert (atoms, spin_version, order) == (2, 13, 1), path
    assert last_cell > 0, path
    offset = header.size

    def check_rows(record_format, count):
        nonlocal offset
        record = struct.Struct(record_format)
        for _ in range(count):
            assert record.unpack_from(data, offset)[0] == 0, (path, offset)
            offset += record.size

    check_rows('=4d', last_cell + 1)  # atv
    check_rows('=4i', last_cell + 1)  # atv_ijk
    count_format = f'={atoms}i'
    offset += struct.calcsize(count_format)  # basis sizes
    neighbors = struct.unpack_from(count_format, data, offset)
    offset += struct.calcsize(count_format)  # FNAN
    offset += 2 * sum(count + 1 for count in neighbors) * struct.calcsize('=i')
    check_rows('=4d', 3 + 3 + atoms)  # tv, rtv, Gxyz

    lines = original_input.encode().splitlines(keepends=True)
    record_length = 256
    count_offset = len(data) - len(lines) * record_length - struct.calcsize('=i')
    assert count_offset > offset, path
    assert struct.unpack_from('=i', data, count_offset)[0] == len(lines), path
    assert data[count_offset + struct.calcsize('=i'):] == b''.join(
        line.ljust(record_length, b'\0') for line in lines
    ), path
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'work/openmx')
    parser.add_argument('--launcher', default='mpirun --bind-to none')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    template = (ROOT / 'work/O2+0.dat').read_text()
    settings = {'DATA.PATH': str(ROOT / 'DFT_DATA19'), 'System.Name': 'result',
                'scf.energycutoff': '100.0', 'scf.maxIter': '3', 'scf.restart': 'off',
                'scf.eigen.lib': 'elpa2', 'level.of.fileout': '1'}
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('OPENMX_', 'OMP_', 'MKL_', 'KMP_'))}
    env.update(OPENMX_GPU='0', OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1')
    command = shlex.split(args.launcher) + ['-np', '2', str(binary), 'input.dat', '-nt', '1']

    def run(name, text, *, error=None, perturb='23'):
        case_dir = output / name
        case_dir.mkdir()
        (case_dir / 'input.dat').write_text(text)
        log_path = case_dir / 'run.log'
        with log_path.open('w') as log:
            process = subprocess.Popen(command, cwd=case_dir, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(env, MALLOC_PERTURB_=perturb), start_new_session=True)
            try:
                process.wait(timeout=180)
            except BaseException:
                stop_job(process)
                raise
        log = log_path.read_text()
        if error is None:
            assert process.returncode == 0, (process.returncode, log_path)
            assert 'The calculation was normally finished.' in log, log_path
            assert re.findall(r'MD=\s*1\s+SCF=\s*(\d+)', log) == ['1', '2', '3'], log_path
        else:
            assert process.returncode == 1, (process.returncode, log_path)
            assert error in log and not (case_dir / 'result.out').exists(), log_path
        print(f'PASS: {name} (exit {process.returncode})', flush=True)
        return case_dir, log

    soc = dict(settings, **{'scf.SpinOrbit.Coupling': 'on'})
    run('collinear-soc', input_text(template, soc),
        error='Spin-orbit coupling is not supported for collinear DFT calculations.')
    gamma_mo = dict(settings, **{'scf.EigenvalueSolver': 'Band', 'MO.fileout': 'on', 'MO.Nkpoint': '3'})
    _, gamma_log = run('gamma-multiple-mo-kpoints', input_text(template, gamma_mo, mo_points=True),
                       error='MO.Nkpoint should be 1 for the cluster calculation.')
    assert 'When only the gamma point is considered' in gamma_log

    large_basis = dict(settings, **{'level.of.fileout': '2'})
    large_dir, _ = run('collinear-o2-156', input_text(template, large_basis, large_basis=True))
    report = (large_dir / 'result.out').read_text()
    assert 'Eigenvalues (Hartree) and Eigenvectors for SCF KS-eq.' in report, large_dir
    # Index 6 is the seventh s radial function. Check both atoms even on a
    # libc that prints '(null)' for an unset string pointer without crashing.
    assert len(re.findall(r'(?m)^ +6 s +[-+0-9]', report)) >= 2, large_dir
    assert '(null)' not in report, large_dir

    gamma_mo.update({'MO.fileout': 'off', 'HS.fileout': 'on'})
    repeat_input = input_text(template, gamma_mo, mo_points=True)
    first, _ = run('gamma-scfout-first', repeat_input, perturb='23')
    second, _ = run('gamma-scfout-repeat', repeat_input, perturb='147')
    a = check_scfout(first / 'result.scfout', repeat_input)
    b = check_scfout(second / 'result.scfout', repeat_input)
    assert a == b, (first, second)
    print('PASS: valid gamma-only runs produce byte-identical .scfout files', flush=True)


if __name__ == '__main__':
    main()
