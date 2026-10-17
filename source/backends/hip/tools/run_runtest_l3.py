#!/usr/bin/env python3
"""Run unmodified built-in test inputs in isolated directories.

Each MPI job contains one input, so a host-memory limit does not discard
the rest of the suite. Originals, references and existing work outputs stay
untouched. Results and the reason for any interruption are saved as JSON.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


def available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable is unavailable')


def group_rss(pgid):
    processes = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdecimal():
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                processes[int(entry.name)] = (int(fields[1]), int(fields[21]))
            except (OSError, ValueError, IndexError):
                pass
    family = {pgid}
    while True:
        children = {pid for pid, (ppid, _) in processes.items() if ppid in family}
        if children <= family:
            break
        family.update(children)
    return sum(processes[pid][1] for pid in family if pid in processes) * os.sysconf('SC_PAGE_SIZE')


def stop_job(proc):
    # Open MPI gives its ranks separate process groups inside the launcher's
    # session. Remember that session so a stuck rank cannot outlive a timeout.
    session = proc.pid
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    except ProcessLookupError:
        pass
    for entry in Path('/proc').iterdir():
        if entry.name.isdecimal():
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[3]) == session:
                    os.kill(int(entry.name), signal.SIGKILL)
            except (ProcessLookupError, FileNotFoundError):
                pass
    proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--suite', choices=('S', 'L', 'L2', 'L3'), default='L3',
                        help='built-in suite (S means -runtest; default: L3)')
    parser.add_argument('--ranks', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1,
                        help='OpenMP threads per MPI rank (also passed as -nt)')
    parser.add_argument('--mpirun', default='mpirun', help='MPI launcher matching the binary')
    parser.add_argument('--map-by', help='optional Open MPI mapping policy')
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--min-available-gib', type=float, default=16)
    parser.add_argument('--timeout', type=float, default=3600, help='seconds per input')
    parser.add_argument('--env', action='append', default=[], metavar='NAME=VALUE')
    args = parser.parse_args()
    if args.ranks < 1 or args.threads < 1 or args.timeout <= 0 or args.min_available_gib < 0:
        parser.error('ranks, threads and timeout must be positive; memory reserve must be nonnegative')
    repo = Path(__file__).resolve().parents[1]
    input_directory = {'S': 'input_example', 'L': 'large_example',
                       'L2': 'large2_example', 'L3': 'large3_example'}[args.suite]
    test_option = '-runtest' + ('' if args.suite == 'S' else args.suite)
    inputs = repo / 'work' / input_directory
    cases = sorted(inputs.glob('*.dat'))
    if args.cases:
        by_name = {p.stem: p for p in cases}
        if set(args.cases) - by_name.keys():
            parser.error('Unknown case in the selected suite')
        cases = [by_name[name] for name in dict.fromkeys(args.cases)]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'DFT_DATA19').symlink_to(repo / 'DFT_DATA19', target_is_directory=True)
    binary = args.binary.resolve()
    shutil.copy2(binary, output / 'openmx')
    binary = output / 'openmx'
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC='FALSE',
               OPENBLAS_NUM_THREADS='1', BLIS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for item in args.env:
        if '=' not in item or not item.split('=', 1)[0]:
            parser.error('--env requires NAME=VALUE')
        name, value = item.split('=', 1)
        env[name] = value
    results = []
    command = [args.mpirun, '-np', str(args.ranks), '--bind-to', 'core']
    if args.map_by:
        command += ['--map-by', args.map_by]
    elif args.threads > 1:
        command += ['--map-by', f'slot:PE={args.threads}']
    command += [str(binary), test_option, '-nt', str(args.threads)]
    (output / 'command.json').write_text(json.dumps(dict(
        command=command, suite=args.suite, input_directory=input_directory,
        env=args.env, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        environment={k: v for k, v in env.items() if k.startswith((
            'OPENMX_', 'OMP_', 'BLIS_', 'OPENBLAS_', 'MKL_', 'MAGMA_', 'GEMMUL8_',
            'HSA_', 'HIP_', 'ROCR_', 'LIBOMPTARGET_', 'GPU_MAX_HW_QUEUES'))},
        min_available_gib=args.min_available_gib, timeout=args.timeout), indent=2))
    for inp in cases:
        case = output / inp.stem
        testdir = case / input_directory
        testdir.mkdir(parents=True)
        (testdir / inp.name).symlink_to(inp)
        system = re.search(r'^System.Name\s+(\S+)', inp.read_text(), re.M | re.I)[1]
        (testdir / (system + '.out')).symlink_to(inputs / (system + '.out'))
        start = time.monotonic()
        peak = 0
        min_available = available_bytes()
        status = 'failed'
        print(f'START {inp.stem}', flush=True)
        with (case / 'run.log').open('w') as log:
            proc = subprocess.Popen(command, cwd=case, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while proc.poll() is None:
                    peak = max(peak, group_rss(proc.pid))
                    min_available = min(min_available, available_bytes())
                    if min_available < args.min_available_gib * 1024**3:
                        status = 'host_memory_guard'
                        stop_job(proc)
                        break
                    if time.monotonic() - start > args.timeout:
                        status = 'timeout'
                        stop_job(proc)
                        break
                    time.sleep(0.25)
            finally:
                if proc.poll() is None:
                    stop_job(proc)
        result_file = case / (test_option[1:] + '.result')
        row = result_file.read_text() if result_file.exists() else ''
        match = re.search(r'Elapsed time\(s\)=\s*([\d.]+)\s+diff Utot=\s*([\d.eE+-]+)\s+diff Force=\s*([\d.eE+-]+)', row)
        if proc.returncode == 0 and match:
            status = 'completed'
        result = dict(case=inp.stem, status=status, returncode=proc.returncode,
                      wall_seconds=time.monotonic()-start, peak_rss_gib=peak/1024**3,
                      min_available_gib=min_available/1024**3)
        if match:
            result.update(zip(('elapsed_seconds', 'diff_utot', 'diff_force'), map(float, match.groups())))
        results.append(result)
        (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    return int(any(r['status'] != 'completed' for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
