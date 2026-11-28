#!/usr/bin/env python3
"""Check EXX output levels with the real CPU, CUDA, or HIP application."""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from run_scf_output_regression import input_text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from exx_force_checks import forces_of
from exx_kpoint_checks import make_input
from exx_output_checks import PATH_END, PATH_START, band_input
from exx_scf_checks import number
from run_gpu_suites import stop_job


def check_output(path, level, gpu, fallback):
    text = path.read_text()
    lines = [line for line in text.splitlines() if line.startswith('<EXX>')]
    if level < 3:
        assert 'force(EXX)' not in text, path
    if level == 0:
        assert not lines, path
        return
    messages = '\n'.join(lines)
    assert not re.search(r'\b(?:CUDA|HIP)\b', messages, re.IGNORECASE), path
    assert messages.count('Preparing HSE06 exact exchange') == 1, path
    assert 'The outer iteration converged' in messages, path
    assert ('GPU dispatch enabled' in messages) == gpu, path
    assert ('(CPU matrix products)' in messages) == (not gpu), path
    assert ('GPU-accelerated exact exchange' in messages) == (gpu and not fallback), path
    assert ('matrix products ran on CPU' in messages) == fallback, path
    for detail in ('outer iteration ', 'local RI', 'time (s)', 'forces of the exact exchange',
                   'outputs after the SCF'):
        # The lowercase iteration heading is the per-build diagnostic.
        present = any(line.startswith('<EXX>  ' + detail) for line in lines)
        assert present == (level >= 2), (path, detail)
    if gpu and not fallback and level >= 2:
        assert re.search(r'GPU: [1-9]\d* calls of the exchange build on the device', messages), path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda', 'hip'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launcher', default='mpirun')
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    gpu = args.backend != 'cpu'
    # The silicon cell provides enough work to exercise the GPU exchange build.
    template = make_input('result', 'siz', (1, 1, 1), (4, 4, 4), 'HSE06',
                          extra=band_input([(PATH_START, PATH_END, 2)]))
    settings = {'DATA.PATH': str(ROOT / 'DFT_DATA19'), 'scf.restart': 'off',
                'scf.eigen.lib': 'gpusolver' if gpu else 'elpa2', 'scf.exx.gpu.min.work': '0'}
    env = {key: value for key, value in os.environ.items() if not key.startswith(('OPENMX_', 'OMP_', 'MKL_', 'KMP_'))}
    env.update(OPENMX_GPU='1' if gpu else '0', OPENMX_BAND_GPU_SWITCH_NUM='1',
               OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = shlex.split(args.launcher) + ['-np', '4', str(binary), 'input.dat', '-nt', '1']
    cases = [('summary', 1, 0, False), ('file-detail', 0, 2, False), ('stdout-detail', 2, 1, False)]
    if gpu:
        cases.append(('cpu-exchange', 1, 1, True))

    reference = None
    for name, stdout_level, file_level, fallback in cases:
        folder = output / name
        folder.mkdir()
        case_settings = dict(settings, **{'level.of.stdout': stdout_level, 'level.of.fileout': file_level})
        case_env = env.copy()
        if fallback:
            case_settings['scf.exx.gpu.min.work'] = '1e30'
            case_env['OPENMX_BAND_GPU_DIAG'] = '0'
        (folder / 'input.dat').write_text(input_text(template, case_settings))
        run_log = folder / 'run.log'
        with run_log.open('w') as log:
            process = subprocess.Popen(command, cwd=folder, env=case_env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                process.wait(timeout=180)
            except BaseException:
                stop_job(process)
                raise
        assert process.returncode == 0 and 'The calculation was normally finished.' in run_log.read_text(), run_log
        report = folder / 'result.out'
        check_output(run_log, stdout_level, gpu, fallback)
        check_output(report, max(file_level, 1), gpu, fallback)
        report_text = report.read_text()
        energies = [number(rf'^\s+{key}\.\s+(-?\d+\.\d+)', report_text) for key in ('Utot', 'Uexx')]
        forces = forces_of(folder, 'result')
        assert None not in energies and forces is not None and len(forces) == 2, report
        if reference is None:
            reference = energies, forces
        ref_energies, ref_forces = reference
        assert max(abs(a - b) for a, b in zip(energies, ref_energies)) <= 1e-7, report
        assert max(abs(a - b) for row, ref_row in zip(forces, ref_forces) for a, b in zip(row, ref_row)) <= 1e-7, report
        print(f'PASS {args.backend} {name}', flush=True)


if __name__ == '__main__':
    main()
