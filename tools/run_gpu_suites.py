#!/usr/bin/env python3
"""Run unmodified built-in test inputs in isolated directories.

By default each MPI job contains one input, so a host-memory limit does not
discard the rest of the suite. Use --together to run all selected inputs in
one MPI job and exercise cache cleanup between systems. Originals, references
and existing work outputs stay untouched. Results are saved as JSON. SIGINT
preserves completed cases and records the active case as interrupted; cases
not yet started remain listed in command.json.
"""
import argparse
import hashlib
import json
import math
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


def read_result_rows(path):
    """Read native suite rows by index: OpenMX truncates filenames to 30 chars."""
    if not path.exists():
        return {}
    rows = {}
    pattern = (r'^\s*(\d+)\s+.*?Elapsed time\(s\)=\s*(\S+)\s+'
               r'diff Utot=\s*(\S+)\s+diff Force=\s*(\S+)')
    for match in re.finditer(pattern, path.read_text(), re.M):
        try:
            values = tuple(map(float, match.groups()[1:]))
        except ValueError:
            values = None
        rows[int(match[1])] = values
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--suite', choices=('S', 'L', 'L2', 'L3'), default='L3',
                        help='built-in suite (S means -runtest; default: L3)')
    parser.add_argument('--ranks', type=int, default=8)
    parser.add_argument('--mpirun', default='mpirun', help='MPI launcher matching the binary')
    parser.add_argument('--map-by', help='optional Open MPI mapping policy')
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--together', action='store_true',
                        help='run selected inputs in one MPI job, in native filename order')
    parser.add_argument('--min-available-gib', type=float, default=16)
    parser.add_argument('--timeout', type=float, default=3600,
                        help='seconds per input; --together resets the timer after each result')
    parser.add_argument('--env', action='append', default=[], metavar='NAME=VALUE')
    args = parser.parse_args()
    if args.ranks < 1 or args.timeout <= 0 or args.min_available_gib < 0:
        parser.error('ranks and timeout must be positive; memory reserve must be nonnegative')
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
    if not cases:
        parser.error('No test inputs found')
    if args.together:
        cases.sort()  # Runtest.c sorts all input filenames before execution.
    systems = {}
    for inp in cases:
        system = re.search(r'^System.Name\s+(\S+)', inp.read_text(), re.M | re.I)
        if system is None or not (inputs / (system[1] + '.out')).is_file():
            parser.error(f'Missing reference for {inp.name}')
        systems[inp] = system[1]
    if args.together and len(set(systems.values())) != len(cases):
        parser.error('--together requires distinct System.Name values to preserve each output')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'DFT_DATA19').symlink_to(repo / 'DFT_DATA19', target_is_directory=True)
    binary = args.binary.resolve()
    shutil.copy2(binary, output / 'openmx')
    binary = output / 'openmx'
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', BLIS_NUM_THREADS='1')
    for item in args.env:
        if '=' not in item or not item.split('=', 1)[0]:
            parser.error('--env requires NAME=VALUE')
        name, value = item.split('=', 1)
        env[name] = value
    results = []
    command = [args.mpirun, '-np', str(args.ranks), '--bind-to', 'core']
    if args.map_by:
        command += ['--map-by', args.map_by]
    command += [str(binary), test_option, '-nt', '1']
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total,driver_version',
                          '--format=csv'], capture_output=True, text=True, check=False) if shutil.which('nvidia-smi') else None
    (output / 'command.json').write_text(json.dumps(dict(
        gpu_info=gpu.stdout.strip() if gpu and gpu.returncode == 0 else None,
        command=command, suite=args.suite, input_directory=input_directory,
        together=args.together, cases=[inp.stem for inp in cases],
        metrics_scope='job' if args.together else 'case',
        env=args.env, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        environment={k: v for k, v in env.items() if k.startswith((
            'OPENMX_', 'OMP_', 'BLIS_', 'OPENBLAS_', 'MAGMA_', 'GEMMUL8_',
            'CUDA_', 'CUBLAS_', 'CUSOLVER_', 'NVCOMPILER_'))},
        min_available_gib=args.min_available_gib, timeout=args.timeout), indent=2))
    groups = [cases] if args.together else [[inp] for inp in cases]
    process_failed = False
    interrupted = False

    def request_stop(signum, frame):
        # Defer Ctrl-C to the polling loop so cleanup and results.json finish
        # even if another SIGINT arrives while terminating the MPI session.
        nonlocal interrupted
        interrupted = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    (output / 'results.json').write_text('[]\n')
    for group in groups:
        if interrupted:
            break
        case = output / ('together' if args.together else group[0].stem)
        testdir = case / input_directory
        testdir.mkdir(parents=True)
        for inp in group:
            (testdir / inp.name).symlink_to(inp)
            system = systems[inp]
            (testdir / (system + '.out')).symlink_to(inputs / (system + '.out'))
        result_file = case / (test_option[1:] + '.result')
        start = time.monotonic()
        progress_time = start
        progress = 0
        peak = 0
        min_available = available_bytes()
        status = 'failed'
        print('START ' + ('--together ' if args.together else '') +
              ' '.join(inp.stem for inp in group), flush=True)
        if interrupted:
            break
        with (case / 'run.log').open('w') as log:
            proc = subprocess.Popen(command, cwd=case, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while proc.poll() is None:
                    if interrupted:
                        status = 'interrupted'
                        stop_job(proc)
                        break
                    peak = max(peak, group_rss(proc.pid))
                    min_available = min(min_available, available_bytes())
                    if args.together:
                        completed = len(read_result_rows(result_file))
                        if completed > progress:
                            progress = completed
                            progress_time = time.monotonic()
                    if min_available < args.min_available_gib * 1024**3:
                        status = 'host_memory_guard'
                        stop_job(proc)
                        break
                    if time.monotonic() - progress_time > args.timeout:
                        status = 'timeout'
                        stop_job(proc)
                        break
                    time.sleep(0.25)
            finally:
                if proc.poll() is None:
                    stop_job(proc)
        if interrupted:
            status = 'interrupted'
        process_failed |= proc.returncode != 0 or status in ('timeout', 'host_memory_guard')
        rows = read_result_rows(result_file)
        wall_seconds = time.monotonic() - start
        active_index = 1
        while active_index in rows:
            active_index += 1
        for index, inp in enumerate(group, 1):
            if interrupted and index > active_index:
                break  # Later --together inputs never started.
            values = rows.get(index)
            # A later input can fail after earlier native result rows were
            # flushed. Keep those completed cases, but return a failed job.
            completed = (values is not None and all(map(math.isfinite, values)) and
                         (case / (systems[inp] + '.out')).is_file() and
                         (proc.returncode == 0 or args.together or interrupted))
            case_status = 'failed' if interrupted and index in rows else status
            result = dict(case=inp.stem, status='completed' if completed else case_status,
                          returncode=proc.returncode, wall_seconds=wall_seconds,
                          peak_rss_gib=peak/1024**3, min_available_gib=min_available/1024**3)
            if values is not None and all(map(math.isfinite, values)):
                result.update(zip(('elapsed_seconds', 'diff_utot', 'diff_force'), values))
            results.append(result)
            print(json.dumps(result), flush=True)
        (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    signal.signal(signal.SIGINT, previous_sigint)
    if interrupted:
        return 130
    return int(process_failed or any(r['status'] != 'completed' for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
