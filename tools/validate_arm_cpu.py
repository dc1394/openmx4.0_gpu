#!/usr/bin/env python3
"""Validate the CPU build with CTest and scalar/NEON/SVE MPI calculations.

Run in an ordinary Linux shell with MPI sockets and LeakSanitizer available.
Each calculation uses a fresh directory; the bundled inputs stay unchanged.
The report includes every planned run, including failures and interruptions.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from compare_gpu_suites import read_output
from run_gpu_suites import stop_job


CASES = {
    'Methane': 'work/input_example/Methane.dat',
    'Cr2': 'work/input_example/Cr2.dat',
    'Si8': 'work/Si8.dat',
    'DIA8_Krylov': 'work/ml_example/DIA8_Krylov.dat',
}
MODES = ('scalar', 'legacy', 'neon', 'sve')
LAYOUTS = ((1, 1), (4, 2))
SCF_CRITERION = 1e-10


def prepare_input(source, data, *, krylov_order=None):
    overrides = {
        'System.CurrrentDirectory': './', 'System.Name': 'result',
        'DATA.PATH': str(data), 'level.of.stdout': '2', 'level.of.fileout': '1',
        'MD.Type': 'nomd', 'MD.maxIter': '1', 'scf.maxIter': '200',
        'scf.criterion': str(SCF_CRITERION), 'scf.eigen.lib': 'elpa2',
        'scf.restart': 'off', 'memory.leak': 'off',
    }
    if krylov_order is not None:
        overrides['orderN.KrylovH.order'] = str(krylov_order)
    replaced = {key.lower() for key in overrides}
    lines = []
    for line in source.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].lower() not in replaced:
            lines.append(line)
    lines.extend(f'{key} {value}' for key, value in overrides.items())
    return '\n'.join(lines) + '\n'


def run(command, folder, env, log_path, timeout):
    start = time.monotonic()
    with log_path.open('w') as log:
        proc = subprocess.Popen(command, cwd=folder, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_job(proc)
            raise RuntimeError(f'timeout after {timeout} seconds; see {log_path}')
        except BaseException:
            stop_job(proc)
            raise
    return code, time.monotonic() - start


def check_scf(log, output):
    # DFT.c stops on a band-energy change and a bound on norm[0]. The
    # printed NormRD is sqrt(abs(norm[0])), not an energy in Hartree.
    # Use OpenMX's convergence flag; a normal program exit alone also
    # happens when the SCF iteration limit is reached.
    achieved = re.findall(r'The SCF was achieved at MD=\s*(\d+)', log)
    if achieved != ['1'] or 'The SCF was not fully achieved' in log:
        raise ValueError('OpenMX did not report SCF convergence for the single MD step')
    scf = re.findall(r'SCF=\s*(\d+)\s+NormRD=\s*(\S+)\s+Uele=\s*(\S+)', output)
    if len(scf) < 2:
        raise ValueError('missing SCF history')
    residual = float(scf[-1][1])
    energies = [float(row[2]) for row in scf[-2:]]
    if residual < 0 or not all(map(math.isfinite, [residual, *energies])):
        raise ValueError('nonfinite or invalid SCF history')
    return dict(scf_steps=int(scf[-1][0]), scf_residual=residual,
                scf_band_energy_change=abs(energies[-1]-energies[-2]))


def test_environment(environ):
    # GPU mock tests need their own default policy. OPENMX_GPU=0 belongs
    # only to the actual CPU calculations below, not the whole CTest suite.
    env = {key: value for key, value in environ.items() if not key.startswith('OPENMX_')}
    env.update(OPENBLAS_NUM_THREADS='1', BLIS_NUM_THREADS='1',
               OMP_NUM_THREADS='2', OMP_DYNAMIC='FALSE')
    # Do not let a sandbox-only workaround disable the external leak check.
    env.pop('ASAN_OPTIONS', None)
    env.pop('LSAN_OPTIONS', None)
    return env


def check_output(folder, expected_mode):
    log = (folder / 'run.log').read_text(errors='replace')
    if 'The calculation was normally finished.' not in log:
        raise ValueError('missing normal-completion message')
    reported = set(re.findall(r'host SIMD = (\S+)', log))
    if reported != {expected_mode}:
        raise ValueError(f'expected host SIMD {expected_mode}, reported {sorted(reported)}')
    output = folder / 'result.out'
    energy, atoms, forces, grids = read_output(output)
    scf = check_scf(log, output.read_text())
    if not all(math.isfinite(x) for atom in atoms for x in atom[2:]):
        raise ValueError('nonfinite atom positions')
    return dict(energy=energy, atoms=atoms, forces=forces, grids=grids,
                selected_mode=expected_mode, **scf)


def compare(reference, candidate, tolerance):
    if reference['atoms'] != candidate['atoms'] or reference['grids'] != candidate['grids']:
        raise ValueError('atom positions or integration grids differ')
    if len(reference['forces']) != len(candidate['forces']) or not reference['forces']:
        raise ValueError('force array sizes differ or are empty')
    de = abs(reference['energy'] - candidate['energy'])
    df = max(abs(x-y) for x, y in zip(reference['forces'], candidate['forces']))
    return dict(energy_difference=de, max_force_difference=df,
                passed=de <= tolerance and df <= tolerance)


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=repo / 'build/cpu')
    parser.add_argument('--binary', type=Path, default=repo / 'work/openmx')
    parser.add_argument('--output', type=Path, help='fresh directory for logs and report')
    parser.add_argument('--mpirun', default='mpirun')
    parser.add_argument('--timeout', type=float, default=1200, help='seconds per calculation')
    parser.add_argument('--tolerance', type=float, default=1e-7,
                        help='absolute energy (Ha) and force (Ha/bohr) tolerance')
    parser.add_argument('--krylov-order', type=int, default=600,
                        help='DIA8 local basis limit (default: 600, covering its full local space; '
                             '400 reproduces the original reduced-basis sensitivity)')
    parser.add_argument('--skip-ctest', action='store_true', help='only run the MPI calculations')
    parser.add_argument('--prepare-only', action='store_true', help='prepare all inputs without executing')
    args = parser.parse_args()
    if args.timeout <= 0 or not math.isfinite(args.timeout) or not 0 < args.tolerance < 1:
        parser.error('timeout must be finite and positive; tolerance must be between zero and one')
    if args.krylov_order < 1:
        parser.error('krylov-order must be positive')
    build, binary = args.build.resolve(), args.binary.resolve()
    core = binary.parent / 'lib/libopenmx_core.so'
    if not binary.is_file() or not core.is_file():
        parser.error('build the application and its matching lib/libopenmx_core.so first')
    api = ctypes.CDLL(str(core))
    api.openmx_core_host_simd_mode.argtypes = [ctypes.c_char_p]
    api.openmx_core_host_simd_mode.restype = ctypes.c_int
    api.openmx_core_host_simd_name.argtypes = [ctypes.c_int]
    api.openmx_core_host_simd_name.restype = ctypes.c_char_p
    api.openmx_core_host_simd_usable.argtypes = [ctypes.c_int]
    api.openmx_core_host_simd_usable.restype = ctypes.c_int
    available = {api.openmx_core_host_simd_name(i).decode(): bool(api.openmx_core_host_simd_usable(i))
                 for i in range(7)}
    selected = {mode: api.openmx_core_host_simd_name(api.openmx_core_host_simd_mode(mode.encode())).decode()
                for mode in MODES}
    if selected['neon'] != 'NEON' or not selected['sve'].startswith('SVE'):
        parser.error(f'this validation requires usable NEON and SVE variants: {available}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output or build / ('arm-validation-' + stamp)).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = test_environment(os.environ)
    report = dict(status='prepared', available_modes=available, selected_modes=selected,
                  binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  core_sha256=hashlib.sha256(core.read_bytes()).hexdigest(),
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                  tolerance=args.tolerance, scf_criterion=SCF_CRITERION, krylov_order=args.krylov_order,
                  ctest=dict(status='not_run'), runs=[], comparisons=[])
    (output / 'changes.patch').write_text(subprocess.check_output(['git', 'diff'], cwd=repo, text=True))

    def save():
        (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')

    for name, source in CASES.items():
        for ranks, threads in LAYOUTS:
            for mode in MODES:
                folder = output / name / f'{ranks}mpi-{threads}omp' / mode
                folder.mkdir(parents=True)
                order = args.krylov_order if name == 'DIA8_Krylov' else None
                (folder / 'case.dat').write_text(prepare_input(repo / source, repo / 'DFT_DATA19', krylov_order=order))
                command = [args.mpirun, '-np', str(ranks), '--bind-to', 'none',
                           str(binary), 'case.dat', '-nt', str(threads)]
                report['runs'].append(dict(case=name, ranks=ranks, threads=threads, mode=mode,
                                           directory=str(folder), command=command, status='not_run'))
    save()
    print(f'Logs and report: {output}', flush=True)
    print(f'Host variants: {available}; SVE request selects {selected["sve"]}', flush=True)
    if args.prepare_only:
        return 0
    report['status'] = 'running'
    try:
        if not args.skip_ctest:
            print('START full CTest (including MPI and leak checks)', flush=True)
            code, elapsed = run(['ctest', '--test-dir', str(build), '--output-on-failure', '--parallel', '4'],
                                repo, env, output / 'ctest.log', 3600)
            report['ctest'] = dict(status='passed' if code == 0 else 'failed', returncode=code,
                                   elapsed_seconds=elapsed)
            print(f'CTest: {report["ctest"]["status"]} ({elapsed:.1f}s)', flush=True)
            save()
        for item in report['runs']:
            label = f'{item["case"]} {item["ranks"]}MPI/{item["threads"]}OMP {item["mode"]}'
            print('START ' + label, flush=True)
            folder = Path(item['directory'])
            run_env = dict(env, OPENMX_GPU='0', OPENMX_HOST_SIMD=item['mode'],
                           OMP_NUM_THREADS=str(item['threads']))
            item['status'] = 'running'
            save()
            try:
                code, elapsed = run(item['command'], folder, run_env, folder / 'run.log', args.timeout)
                item.update(returncode=code, elapsed_seconds=elapsed)
                if code:
                    raise RuntimeError(f'MPI process exited with code {code}')
                item.update(check_output(folder, selected[item['mode']]))
                item['status'] = 'completed'
            except (OSError, RuntimeError, ValueError, AttributeError, IndexError) as error:
                item.update(status='failed', error=str(error))
            print(f'{item["status"].upper()} {label}: {item.get("error", "normal finish, SCF converged")}', flush=True)
            save()
        by_key = {(r['case'], r['ranks'], r['threads'], r['mode']): r for r in report['runs']}
        for item in report['runs']:
            # Compare every ISA to scalar at the same MPI layout. Also compare
            # the multirank scalar result to the single-rank scalar result.
            key = (item['case'], item['ranks'], item['threads'], 'scalar')
            if item['mode'] == 'scalar':
                if (item['ranks'], item['threads']) == LAYOUTS[0]:
                    continue
                key = (item['case'], *LAYOUTS[0], 'scalar')
            reference = by_key[key]
            comparison = dict(case=item['case'], candidate=item['directory'],
                              reference=reference['directory'], passed=False)
            try:
                if item['status'] != 'completed' or reference['status'] != 'completed':
                    raise ValueError('candidate or reference did not complete')
                comparison.update(compare(reference, item, args.tolerance))
            except ValueError as error:
                comparison['error'] = str(error)
            report['comparisons'].append(comparison)
        passed = (all(r['status'] == 'completed' for r in report['runs']) and
                  all(c['passed'] for c in report['comparisons']) and
                  (args.skip_ctest or report['ctest']['status'] == 'passed'))
        report['status'] = 'passed' if passed else 'failed'
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
    except (OSError, RuntimeError) as error:
        report.update(status='failed', error=str(error))
    finally:
        for item in report['runs']:
            if item['status'] == 'running':
                item['status'] = 'interrupted'
        save()
    complete = sum(r['status'] == 'completed' for r in report['runs'])
    matches = sum(c['passed'] for c in report['comparisons'])
    print(f'{report["status"].upper()}: calculations {complete}/{len(report["runs"])}, '
          f'comparisons {matches}/{len(CASES) * (len(LAYOUTS) * len(MODES) - 1)}', flush=True)
    print(f'Report: {output / "results.json"}', flush=True)
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
