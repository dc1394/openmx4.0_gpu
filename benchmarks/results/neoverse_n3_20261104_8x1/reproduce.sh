#!/usr/bin/env bash
# Debian 12 / Linux AArch64. Run as your ordinary user, NOT with sudo bash.
# sudo is used only for Debian packages; sources, builds and results are yours.
set -Eeuo pipefail
export LC_ALL=C
if [[ ${1:-} == --help ]]; then
  cat <<'HELP'
Usage: bash install_openmx_benchmark.sh [--deps-only]
Default: install dependencies, clone, build, install, test, benchmark.
--deps-only: install Debian dependencies only, then exit.

Optional environment variables:
  OPENMX_BENCH_ROOT       source directory (default: $HOME/openmx4.0_gpu)
  OPENMX_BENCH_PREFIX     installation (default: $HOME/.local/opt/openmx4.0-cpu)
  OPENMX_BENCH_REF        commit (default: inspected main, 6ad8d53e75db6453e70d7bccf34725c541fbfbae)
  OPENMX_BENCH_SKIP_APT=1  skip apt when dependencies have already been installed
  OPENMX_BENCH_JOBS=4     parallel build jobs
  OPENMX_BENCH_MIN_FREE_GIB=8   minimum free disk space before setup
  OPENMX_BENCH_RUNS=3     timed runs PER mode, in alternating order
  OPENMX_BENCH_WARMUPS=1  warmup runs PER mode (excluded from timing statistics)
  OPENMX_BENCH_TIMEOUT=7200   timeout seconds PER calculation
  OPENMX_BENCH_ENERGY_TOL=1e-7   absolute energy tolerance, Hartree
  OPENMX_BENCH_FORCE_TOL=1e-7    absolute force tolerance, Hartree/bohr
  OPENMX_BENCH_SCF_MAX_ITER     optional explicit change to the input's 25 iterations
  OPENMX_BENCH_REQUIRE_CONVERGED=1   make SCF nonconvergence an error

The driver stops on the first execution/output error; remaining planned runs
are reported as not_run. Each new invocation creates a fresh result directory.

"no SIMD" here means OPENMX_HOST_SIMD=scalar, versus sve128, using the
same installed binary. BLAS/ELPA/compiler optimizations are identical;
this is NOT a claim that all libraries or machine instructions are scalar.
The original input's SCF/mixing/geometry settings are preserved by default.
It may reach its 25-iteration limit without converging. Numerical agreement
and SCF convergence are reported separately. No GPU/CUDA SDK is needed.
HELP
  exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --deps-only ) ]]; then
  echo 'Unknown option. Use --help.' >&2; exit 2
fi
if (( EUID == 0 )); then
  echo 'Run as your ordinary user: bash install_openmx_benchmark.sh (without sudo).' >&2
  exit 2
fi
if [[ $(uname -s) != Linux || $(uname -m) != aarch64 ]]; then
  echo 'This script targets this machine: Linux AArch64 with SVE128.' >&2; exit 2
fi
root=${OPENMX_BENCH_ROOT:-$HOME/openmx4.0_gpu}
prefix=${OPENMX_BENCH_PREFIX:-$HOME/.local/opt/openmx4.0-cpu}
ref=${OPENMX_BENCH_REF:-6ad8d53e75db6453e70d7bccf34725c541fbfbae}
jobs=${OPENMX_BENCH_JOBS:-4}
[[ $jobs =~ ^[1-9][0-9]*$ ]] || { echo 'JOBS must be positive.' >&2; exit 2; }
minimum_gib=${OPENMX_BENCH_MIN_FREE_GIB:-8}
[[ $minimum_gib =~ ^[1-9][0-9]*$ ]] || { echo 'MIN_FREE_GIB must be positive.' >&2; exit 2; }
# The upstream ELPA Autoconf build does not safely support whitespace paths.
[[ $root = /* && $prefix = /* && ! $root =~ [[:space:]] && ! $prefix =~ [[:space:]] ]] || {
  echo 'Use absolute source/install paths without whitespace.' >&2; exit 2;
}
available_kib=$(df --output=avail -k "$HOME" | tail -n 1)
if (( available_kib < minimum_gib * 1024 * 1024 )); then
  echo "Insufficient free disk space: require at least $minimum_gib GiB before setup." >&2
  df -h "$HOME" >&2
  exit 2
fi
stamp=$(date -u +%Y%m%dT%H%M%SZ)-$$
logs=$HOME/openmx-benchmarks/$stamp
mkdir -p "$logs"
exec > >(tee -a "$logs/setup.log") 2>&1
trap 'rc=$?; echo "Stopped at line $LINENO (status $rc). Logs: $logs" >&2; exit "$rc"' ERR
echo "Logs: $logs"
if [[ ${OPENMX_BENCH_SKIP_APT:-0} != 1 ]]; then
  sudo -v
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential gfortran \
    autoconf automake libtool libtool-bin m4 pkg-config flex bison \
    openmpi-bin libopenmpi-dev libopenblas-pthread-dev \
    libscalapack-openmpi-dev libfftw3-dev \
    python3 python3-venv unzip patch
fi
if [[ ${1:-} == --deps-only ]]; then
  echo 'Debian dependencies installed. Run this script again to build and benchmark.'
  exit 0
fi
for program in git gcc g++ gfortran make mpicc mpirun python3 autoconf automake libtoolize; do
  command -v "$program" >/dev/null || { echo "Missing dependency: $program" >&2; exit 1; }
done
python3 - <<'PY'
import ctypes, os
libc = ctypes.CDLL(None, use_errno=True)
vl = libc.prctl(51, 0, 0, 0, 0)  # PR_SVE_GET_VL
if vl < 0:
    raise SystemExit('SVE is unavailable: cannot benchmark SVE128.')
if len(os.sched_getaffinity(0)) < 8:
    raise SystemExit('At least 8 available CPU cores are required.')
print('Current SVE VL:', (vl & 65535) * 8, 'bits; 128 bits will be selected in each rank.')
PY
# Debian 12 ships CMake 3.25.1, below this project's 3.25.2 minimum.
# A private venv avoids replacing Debian's Python or CMake installations.
venv=$HOME/.local/share/openmx-build-tools
[[ -x $venv/bin/python ]] || python3 -m venv "$venv"
"$venv/bin/python" -m pip install --disable-pip-version-check --only-binary=:all: 'cmake==3.31.6'
export PATH="$venv/bin:$PATH"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ FC=/usr/bin/gfortran
export OMPI_CC="$CC" OMPI_CXX="$CXX" OMPI_FC="$FC"
# Avoid inheriting OpenMX/ELPA policies or multithreaded BLAS from the shell.
while IFS= read -r name; do
  case "$name" in
    OPENMX_BENCH_*) ;;
    OPENMX_*|ELPA_DEFAULT_*|OMP_*|OPENBLAS_*|MKL_*|BLIS_*|GOTO_*) unset "$name" ;;
  esac
done < <(compgen -e)
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1 OMP_PROC_BIND=TRUE OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_DYNAMIC=FALSE

if [[ ! -e $root ]]; then
  git clone --depth 1 --single-branch https://github.com/dc1394/openmx4.0_gpu.git "$root"
  if [[ $(git -C "$root" rev-parse HEAD) != "$ref" ]]; then
    git -C "$root" fetch --depth 1 origin "$ref"
    git -C "$root" checkout --detach FETCH_HEAD
  fi
else
  [[ -d $root/.git ]] || { echo "Existing path is not a Git checkout: $root" >&2; exit 1; }
  [[ $(git -C "$root" rev-parse HEAD) == "$ref" ]] || {
    echo 'Existing checkout has a different commit. Set OPENMX_BENCH_ROOT to a fresh path.' >&2; exit 1;
  }
fi
[[ -z $(git -C "$root" status --porcelain --untracked-files=no) ]] || {
  echo 'Tracked source files are modified. Use a clean checkout.' >&2; exit 1;
}
git -C "$root" rev-parse HEAD > "$logs/git-head.txt"
git -C "$root" status --porcelain > "$logs/git-status.txt"
# CPU builds do not use GEMMul8 or the FFTW Git submodules. FFTW comes from apt.
for file in benchmarks/inputs/sidia333_nc_cluster.dat \
    DFT_DATA19/PAO/Si7.0.pao DFT_DATA19/VPS/Si_PBE19.vps \
    source/third_party/dist/elpa-new_release_2026_02_002.zip; do
  [[ -s $root/$file ]] || { echo "Required source/data missing: $root/$file" >&2; exit 1; }
done
build=$root/build/cpu-benchmark
cmake -S "$root" -B "$build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_Fortran_COMPILER="$FC" \
  -DOPENMX_GPU_BACKEND=CPU -DOPENMX_CORE_BACKEND=SERIAL \
  -DOPENMX_CPU_COMPILER=GCC -DOPENMX_CPU_MPI=OPENMPI -DOPENMX_CPU_MPI_ROOT=/usr \
  -DOPENMX_CPU_LINALG=OPENBLAS -DOPENMX_CPU_BUILD_SCALAPACK=OFF \
  -DOPENMX_HOST_SIMD_CXX_COMPILER="$CXX" \
  -DOPENMX_CORE_HOST_SIMD=ON -DOPENMX_CORE_HOST_SVE=ON \
  -DOPENMX_COPY_TO_WORK=OFF -DOPENMX_BUILD_JOBS="$jobs" -DBUILD_TESTING=ON
cmake --build "$build" --target openmx openmx_cpu_numerics --parallel "$jobs"
ctest --test-dir "$build" --output-on-failure --no-tests=error \
  -R '^(cpu-numerical-toolchain|core-host-sve-abi)$' | tee "$logs/ctest.log"
cmake --install "$build"
cp "$build/CMakeCache.txt" "$logs/CMakeCache.txt"
cp "$0" "$logs/install_openmx_benchmark.sh"
{
  uname -a
  lscpu
  free -h
  df -h "$root"
  gcc --version
  gfortran --version
  mpirun --version
  cmake --version
  dpkg-query -W git gcc g++ gfortran openmpi-bin libopenmpi-dev \
    libopenblas-pthread-dev libscalapack-openmpi-dev libfftw3-dev
  ldd "$prefix/bin/openmx"
} > "$logs/environment.txt"
echo "Installed: $prefix/bin/openmx"

# Keep the benchmark self-contained in the result directory. This same driver
# validates SIMD/VL/affinity in EVERY MPI rank immediately before exec(OpenMX).
cat > "$logs/benchmark.py" <<'PY'
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time


def dump(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    result = float(value.replace('D', 'E').replace('d', 'e'))
    if not math.isfinite(result):
        raise ValueError('nonfinite number in OpenMX output')
    return result


def rank_exec(binary, mode):
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    if int(os.environ['OMPI_COMM_WORLD_SIZE']) != 8:
        raise RuntimeError('MPI size must be 8')
    for key in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
        if os.environ[key] != '1':
            raise RuntimeError(f'{key} must be 1')
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        raise RuntimeError(f'rank {rank} is not bound to one CPU: {affinity}')
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SVE_SET_VL, PR_SVE_VL_INHERIT: retain VL=128 across exec in BOTH modes.
    if libc.prctl(50, 16 | (1 << 17), 0, 0, 0) < 0:
        raise RuntimeError(f'Cannot set SVE128: errno {ctypes.get_errno()}')
    vl = libc.prctl(51, 0, 0, 0, 0)
    if vl < 0 or (vl & 65535) != 16:
        raise RuntimeError('SVE128 is unavailable')
    api = ctypes.CDLL(str(binary.parent.parent / 'lib/libopenmx_core.so'))
    api.openmx_core_host_simd_mode.argtypes = [ctypes.c_char_p]
    api.openmx_core_host_simd_mode.restype = ctypes.c_int
    api.openmx_core_host_simd_name.argtypes = [ctypes.c_int]
    api.openmx_core_host_simd_name.restype = ctypes.c_char_p
    selected = api.openmx_core_host_simd_name(api.openmx_core_host_simd_mode(mode.encode())).decode()
    expected = {'scalar': 'scalar', 'sve128': 'SVE128'}[mode]
    if selected != expected:
        raise RuntimeError(f'SIMD fallback detected: expected {expected}, got {selected}')
    dump(Path(f'rank-{rank}.json'), dict(rank=rank, ranks=8, cpu_affinity=affinity,
         omp_threads=1, mode=selected, sve_bits=128, binary=str(binary)))
    os.execv(str(binary), [str(binary), 'case.dat', '-nt', '1'])


def prepare(source, data):
    overrides = {'System.CurrrentDirectory': './', 'System.Name': 'result',
                 'DATA.PATH': str(data), 'level.of.stdout': '1',
                 'level.of.fileout': '1', 'scf.eigen.lib': 'elpa2', 'scf.restart': 'off'}
    limit = os.environ.get('OPENMX_BENCH_SCF_MAX_ITER')
    if limit:
        if int(limit) < 1:
            raise ValueError('SCF_MAX_ITER must be positive')
        overrides['scf.maxIter'] = str(int(limit))
    replaced = {key.lower() for key in overrides}
    lines = [line for line in source.read_text().splitlines()
             if not line.split() or line.split()[0].lower() not in replaced]
    lines.extend(f'{key} {value}' for key, value in overrides.items())
    return '\n'.join(lines) + '\n', overrides


def parse(folder):
    log = (folder / 'run.log').read_text(errors='replace')
    if 'The calculation was normally finished.' not in log:
        raise ValueError('missing normal completion message')
    text = (folder / 'result.out').read_text(errors='replace')
    energy = re.findall(r'^\s*Utot\.\s+(\S+)', text, re.M)
    if not energy:
        raise ValueError('missing final Utot summary')
    block = re.search(r'<coordinates.forces\s*\n\s*(\d+)\s*\n', text)
    if not block or int(block[1]) != 216:
        raise ValueError('missing forces or wrong atom count (expected 216)')
    rows = text[block.end():].splitlines()[:216]
    if len(rows) != 216 or any(len(row.split()) < 8 for row in rows):
        raise ValueError('incomplete force table')
    atoms, forces = [], []
    for row in rows:
        fields = row.split()
        atoms.append([int(fields[0]), fields[1], *map(number, fields[2:5])])
        forces.extend(map(number, fields[5:8]))
    if [a[0] for a in atoms] != list(range(1, 217)):
        raise ValueError('atom index mismatch')
    grids = []
    for i in (1, 2, 3):
        match = re.search(rf'^\s*Num.Grid{i}\.\s+(\d+)', text, re.M)
        if not match:
            raise ValueError(f'missing grid {i}')
        grids.append(int(match[1]))
    history = re.findall(r'SCF=\s*(\d+)\s+NormRD=\s*(\S+)\s+Uele=\s*(\S+)', text)
    if not history:
        raise ValueError('missing SCF history')
    history = [[int(step), number(residual), number(energy)] for step, residual, energy in history]
    achieved = re.findall(r'The SCF was achieved at MD=\s*(\d+)', log)
    missed = re.findall(r'The SCF was not fully achieved at MD=\s*(\d+)', log)
    if len(achieved) + len(missed) != 1 or (achieved or missed) != ['1']:
        raise ValueError('expected exactly one SCF convergence report at MD=1')
    # Merge_LogFile appends .CompTime to result.out and removes .CompTime.
    # A normal completed run therefore usually has no standalone time file.
    elapsed = re.search(r'^\s*Elapsed\.Time\.\s+(\S+)', text, re.M)
    if not elapsed and (folder / 'result.CompTime').exists():
        elapsed = re.search(r'^\s*Elapsed\.Time\.\s+(\S+)',
                            (folder / 'result.CompTime').read_text(), re.M)
    if not elapsed:
        # The last time is the maximum across ranks, matching Elapsed.Time.
        elapsed = re.search(r'^\s*Total Computational Time\s*=\s*\d+\s+\S+\s+\d+\s+(\S+)', log, re.M)
    if not elapsed or number(elapsed[1]) <= 0:
        raise ValueError('missing or invalid native computation time')
    return dict(energy_Ha=number(energy[-1]), atoms=atoms, forces_Ha_bohr=forces,
                grids=grids, scf_history=history, scf_steps=history[-1][0],
                scf_converged=bool(achieved), native_seconds=number(elapsed[1]))


def compare(reference, candidate, etol, ftol):
    if reference['atoms'] != candidate['atoms'] or reference['grids'] != candidate['grids']:
        raise ValueError('geometry or grids differ')
    a, b = reference['scf_history'], candidate['scf_history']
    same_steps = [r[0] for r in a] == [r[0] for r in b]
    history_delta = max(abs(x[2] - y[2]) for x, y in zip(a, b)) if same_steps else None
    de = abs(reference['energy_Ha'] - candidate['energy_Ha'])
    df = max(abs(x-y) for x, y in zip(reference['forces_Ha_bohr'], candidate['forces_Ha_bohr']))
    same_convergence = reference['scf_converged'] == candidate['scf_converged']
    passed = (de <= etol and df <= ftol and same_steps and same_convergence
              and history_delta is not None and history_delta <= etol)
    return dict(matched=passed, energy_delta_Ha=de, max_force_delta_Ha_bohr=df,
                max_scf_band_energy_delta_Ha=history_delta, same_scf_steps=same_steps,
                same_convergence=same_convergence,
                printed_energy_and_forces_equal=(reference['energy_Ha'] == candidate['energy_Ha']
                    and reference['forces_Ha_bohr'] == candidate['forces_Ha_bohr']))


def run_job(command, folder, env, timeout):
    if shutil.disk_usage(folder).free < 1024**3:
        raise RuntimeError('Less than 1 GiB free disk space; stopping before the next calculation')
    start = time.monotonic()
    with (folder / 'run.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=folder, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            raise
    return code, time.monotonic() - start


def main(repo, binary, output):
    count = int(os.environ.get('OPENMX_BENCH_RUNS', '3'))
    warmups = int(os.environ.get('OPENMX_BENCH_WARMUPS', '1'))
    timeout = float(os.environ.get('OPENMX_BENCH_TIMEOUT', '7200'))
    etol = float(os.environ.get('OPENMX_BENCH_ENERGY_TOL', '1e-7'))
    ftol = float(os.environ.get('OPENMX_BENCH_FORCE_TOL', '1e-7'))
    require_scf = os.environ.get('OPENMX_BENCH_REQUIRE_CONVERGED', '0') == '1'
    if count < 1 or warmups < 0 or not all(math.isfinite(x) and x > 0 for x in (timeout, etol, ftol)):
        raise ValueError('invalid repeat count, warmup count, timeout or tolerance')
    # Allow 0.5 GiB per planned run for density/restart files, plus 1 GiB reserve.
    required_space = (1 + count + warmups) * 1024**3
    if shutil.disk_usage(output).free < required_space:
        raise RuntimeError(f'Insufficient benchmark disk space: need {required_space / 1024**3:g} GiB')
    source = repo / 'benchmarks/inputs/sidia333_nc_cluster.dat'
    data = repo / 'DFT_DATA19'
    case, overrides = prepare(source, data)
    (output / 'original.dat').write_bytes(source.read_bytes())
    dump(output / 'input-overrides.json', overrides)
    hashes = {str(p): sha(p) for p in (binary, binary.parent.parent / 'lib/libopenmx_core.so',
        binary.parent.parent / 'lib/libopenmx_host_simd.so', source,
        data / 'PAO/Si7.0.pao', data / 'VPS/Si_PBE19.vps')}
    report = dict(status='running', mpi_ranks=8, omp_threads=1, timed_runs_per_mode=count,
                  warmups_per_mode=warmups, energy_tolerance_Ha=etol,
                  force_tolerance_Ha_bohr=ftol, require_converged=require_scf,
                  hashes=hashes, input_overrides=overrides, runs=[], comparisons=[])
    # Prepare the complete plan first, so interruption cannot hide missing runs.
    for phase, repeats in [('warmup', warmups), ('timed', count)]:
        for repeat in range(1, repeats + 1):
            order = ('scalar', 'sve128') if repeat % 2 else ('sve128', 'scalar')
            for mode in order:
                folder = output / f'{phase}-{repeat:02d}-{mode}'
                folder.mkdir()
                (folder / 'case.dat').write_text(case)
                command = ['/usr/bin/mpirun', '-np', '8', '--map-by', 'core', '--bind-to', 'core',
                           '--nooversubscribe', '--report-bindings', sys.executable,
                           str(Path(__file__).resolve()), '--rank', str(binary), mode]
                report['runs'].append(dict(phase=phase, repeat=repeat, mode=mode,
                    directory=str(folder), status='not_run', command=command))

    def save():
        dump(output / 'results.json', report)

    save()
    try:
        for item in report['runs']:
            print(f"START {item['phase']} {item['repeat']} {item['mode']} (8 MPI x 1 OMP)", flush=True)
            item['status'] = 'running'
            save()
            env = dict(os.environ, OPENMX_GPU='0', OPENMX_HOST_SIMD=item['mode'])
            folder = Path(item['directory'])
            try:
                code, wall = run_job(item['command'], folder, env, timeout)
                item.update(returncode=code, wall_seconds=wall)
                if code:
                    raise RuntimeError(f'MPI exit status {code}; see {folder / "run.log"}')
                audits = [json.loads((folder / f'rank-{i}.json').read_text()) for i in range(8)]
                if len({tuple(a['cpu_affinity']) for a in audits}) != 8:
                    raise ValueError('MPI ranks share CPU cores')
                item.update(parse(folder), rank_audit=audits, status='completed')
                print(f"DONE {item['native_seconds']:.3f} s native, {wall:.3f} s wall, "
                      f"SCF steps={item['scf_steps']}, converged={item['scf_converged']}", flush=True)
            except Exception as error:
                item.update(status='failed', error=str(error))
                print(f'FAILED: {error}', flush=True)
            save()
            if item['status'] == 'failed':
                print('Stopping after the first error. Remaining planned runs were not started.', flush=True)
                break
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        for item in report['runs']:
            if item['status'] == 'running':
                item['status'] = 'interrupted'
        save()
        return 130

    return summarize(report, output)


def summarize(report, output, recovered=False):
    count = report['timed_runs_per_mode']
    warmups = report['warmups_per_mode']
    etol = report['energy_tolerance_Ha']
    ftol = report['force_tolerance_Ha_bohr']
    require_scf = report['require_converged']
    suffix = '-recovered' if recovered else ''

    def save():
        dump(output / f'results{suffix}.json', report)

    report['comparisons'] = []
    report.pop('sve128_speedup', None)
    complete = [r for r in report['runs'] if r['status'] == 'completed']
    reference = next((r for r in complete if r['phase'] == 'timed' and r['mode'] == 'scalar'), None)
    for item in complete:
        if reference is None:
            break
        comparison = dict(reference=reference['directory'], candidate=item['directory'])
        try:
            comparison.update(compare(reference, item, etol, ftol))
        except ValueError as error:
            comparison.update(matched=False, error=str(error))
        report['comparisons'].append(comparison)
    report['all_scf_converged'] = bool(complete) and all(r['scf_converged'] for r in complete)
    report['all_numerically_matched'] = (len(complete) == len(report['runs']) and reference is not None
        and len(report['comparisons']) == len(report['runs'])
        and all(c['matched'] for c in report['comparisons']))
    report['timings'] = {}
    lines = ['OpenMX sidia333_nc_cluster: 8 MPI x 1 OpenMP',
             'no SIMD = host scalar kernels; BLAS/ELPA optimizations unchanged.',
             f'Energy tolerance: {etol:g} Ha; force tolerance: {ftol:g} Ha/bohr.',
             f'Timed repetitions per mode: {count}; warmups per mode: {warmups}.', '']
    for mode in ('scalar', 'sve128'):
        rows = [r for r in complete if r['phase'] == 'timed' and r['mode'] == mode]
        if len(rows) != count:
            lines.append(f'{mode}: INCOMPLETE ({len(rows)}/{count})')
            continue
        native = [r['native_seconds'] for r in rows]
        wall = [r['wall_seconds'] for r in rows]
        report['timings'][mode] = dict(native_median_seconds=statistics.median(native),
            native_min_seconds=min(native), native_max_seconds=max(native),
            wall_median_seconds=statistics.median(wall))
        lines.append(f'{mode}: native median {statistics.median(native):.3f} s '
                     f'(min {min(native):.3f}, max {max(native):.3f}); '
                     f'wall median {statistics.median(wall):.3f} s')
    if len(report['timings']) == 2:
        speedup = (report['timings']['scalar']['native_median_seconds'] /
                   report['timings']['sve128']['native_median_seconds'])
        report['sve128_speedup'] = speedup
        lines.append(f'SVE128 speedup (scalar/SVE128 native medians): {speedup:.4f}x')
    if report['comparisons'] and all('energy_delta_Ha' in c for c in report['comparisons']):
        lines.extend([f"Max |delta E|: {max(c['energy_delta_Ha'] for c in report['comparisons']):.6g} Ha",
            f"Max |delta F|: {max(c['max_force_delta_Ha_bohr'] for c in report['comparisons']):.6g} Ha/bohr",
            'Energy/forces exactly equal at printed precision: ' +
            str(all(c['printed_energy_and_forces_equal'] for c in report['comparisons']))])
    lines.append('Numerical agreement (including SCF energy history): ' +
                 ('MATCH' if report['all_numerically_matched'] else 'FAIL / INCOMPLETE'))
    lines.append('SCF convergence: ' + ('ALL CONVERGED' if report['all_scf_converged']
                 else 'NOT ALL CONVERGED. The original input allows only 25 iterations.'))
    ok = report['all_numerically_matched'] and (not require_scf or report['all_scf_converged'])
    report['status'] = 'matched' if ok else 'failed'
    save()
    fields = ['phase', 'repeat', 'mode', 'status', 'native_seconds', 'wall_seconds',
              'energy_Ha', 'scf_steps', 'scf_converged', 'directory', 'error']
    with (output / f'timings{suffix}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['runs'])
    summary = '\n'.join(lines) + f'\nFull results and logs: {output}\n'
    (output / f'summary{suffix}.txt').write_text(summary)
    print(summary, flush=True)
    return 0 if ok else 1


def recover(output, wait=False):
    """Reparse completed calculations without changing/restarting OpenMX jobs."""
    previous = None
    deadline = time.monotonic() + 24 * 3600
    while True:
        try:
            report = json.loads((output / 'results.json').read_text())
        except (OSError, ValueError):
            if not wait or time.monotonic() >= deadline:
                raise
            time.sleep(5)
            continue
        signature = [(r['phase'], r['repeat'], r['mode'], r['status']) for r in report['runs']]
        if signature != previous:
            print(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                  'Benchmark progress:', signature, flush=True)
            previous = signature
        if report.get('status') == 'running':
            if not wait:
                raise RuntimeError('Benchmark is running. Use --recover-wait or wait for it to finish.')
            if time.monotonic() >= deadline:
                raise RuntimeError('Timed out waiting for the original benchmark driver.')
            time.sleep(10)
            continue
        break
    report['recovered_from'] = str(output / 'results.json')
    report['original_driver_status'] = report['status']
    for item in report['runs']:
        try:
            if item.get('returncode') != 0:
                raise ValueError('no recorded successful MPI exit; run cannot be recovered')
            folder = Path(item['directory'])
            audits = [json.loads((folder / f'rank-{i}.json').read_text()) for i in range(8)]
            expected = 'scalar' if item['mode'] == 'scalar' else 'SVE128'
            if any(a['rank'] != i or a['ranks'] != 8 or a['omp_threads'] != 1 or
                   a['sve_bits'] != 128 or a['mode'] != expected or len(a['cpu_affinity']) != 1
                   for i, a in enumerate(audits)):
                raise ValueError('rank audit mismatch')
            if len({tuple(a['cpu_affinity']) for a in audits}) != 8:
                raise ValueError('MPI ranks share CPU cores')
            item.update(parse(folder), rank_audit=audits, status='completed')
            item.pop('error', None)
        except (OSError, ValueError, KeyError) as error:
            item.update(status='failed', error=str(error))
    return summarize(report, output, recovered=True)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--rank':
        rank_exec(Path(sys.argv[2]), sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] in ('--recover', '--recover-wait'):
        raise SystemExit(recover(Path(sys.argv[2]).resolve(), sys.argv[1] == '--recover-wait'))
    elif len(sys.argv) == 4:
        raise SystemExit(main(*(Path(arg).resolve() for arg in sys.argv[1:])))
    else:
        raise SystemExit('Expected REPO BINARY RESULTS, or --rank BINARY MODE')
PY
python3 "$logs/benchmark.py" "$root" "$prefix/bin/openmx" "$logs"
echo "Finished. Summary: $logs/summary.txt"
