"""NVIDIA continuation regression for retained upstream float products.

These are matched-seed correctness checks, not cold-start benchmarks.
The preserved seed and reference belong to the 2026-11-08 campaign.
"""
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

A = Path(__file__).resolve().parent
ROOT = A.parent.parent
PREVIOUS = ROOT / 'build/cghelgeo-precision-20261108-163534'
MPI = Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.9/comm_libs/mpi/bin/mpirun')
VARIANTS = ['gpu_off_native', 'gpu_off_emulated', 'gpu_default', 'cpu_avx512_off']
sys.path.insert(0, str(ROOT / 'tools'))
from compare_gpu_suites import read_output
from run_gpu_suites import available_bytes, group_rss, stop_job

STOP = False
MPS = False


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stop(signum, frame):
    global STOP
    STOP = True


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def clean_environment():
    prefixes = ('OPENMX_', 'OMX_', 'GEMMUL8_', 'OMP_', 'KMP_', 'MKL_', 'I_MPI_',
                'CUDA_', 'CUBLAS_', 'CUSOLVER_', 'OMPI_', 'PMIX_', 'PRTE_',
                'NVCOMPILER_', 'OPENBLAS_', 'BLIS_', 'ACC_')
    excluded = {'LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT', 'CPATH',
                'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH'}
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes) and k not in excluded}
    env.update(OMP_NUM_THREADS='2', OMP_DYNAMIC='FALSE', OMP_MAX_ACTIVE_LEVELS='1',
               OMP_PLACES='threads', OMP_PROC_BIND='close', OMP_STACKSIZE='64M',
               OPENBLAS_NUM_THREADS='2', BLIS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               MKL_DYNAMIC='FALSE')
    return env


def mps_environment():
    env = clean_environment()
    env.update(CUDA_MPS_PIPE_DIRECTORY=str(A / 'mp'),
               CUDA_MPS_LOG_DIRECTORY=str(A / 'mps-log'), CUDA_VISIBLE_DEVICES='0')
    return env


def control(command):
    return subprocess.run(['nvidia-cuda-mps-control'], input=command + '\n',
                          env=mps_environment(), capture_output=True, text=True, timeout=20)


def start_mps():
    global MPS
    assert len(str(A / 'mp/control')) < 108
    (A / 'mp').mkdir()
    (A / 'mps-log').mkdir()
    result = subprocess.run(['nvidia-cuda-mps-control', '-d'], env=mps_environment(),
                            capture_output=True, text=True, timeout=20)
    save(A / 'mps-start.json', dict(returncode=result.returncode,
                                  stdout=result.stdout, stderr=result.stderr))
    result.check_returncode()
    MPS = True


def stop_mps():
    global MPS
    if MPS:
        result = control('quit')
        save(A / 'mps-stop.json', dict(returncode=result.returncode,
                                     stdout=result.stdout, stderr=result.stderr))
        result.check_returncode()
        MPS = False


def prepare():
    assert json.loads((A / 'build-status.json').read_text())['returncode'] == 0
    old = json.loads((PREVIOUS / 'provenance.json').read_text())
    for name in ['bin', 'lib', 'seed', 'runs']:
        (A / name).mkdir()
    built = ROOT / 'work/openmx'
    binary = A / 'bin/openmx'
    shutil.copy2(built, binary)
    output = subprocess.check_output(['readelf', '-d', str(binary)], text=True)
    rpath = re.search(r'\(RPATH\).*?\[(.*?)\]', output)[1]
    entries = rpath.split(':')
    first = ['$ORIGIN/lib', '$ORIGIN/../lib']
    new_rpath = ':'.join(first + [part for part in entries if part not in first])
    assert len(rpath) == len(new_rpath)
    blob = binary.read_bytes()
    assert blob.count(rpath.encode() + b'\0') == 1
    binary.write_bytes(blob.replace(rpath.encode() + b'\0', new_rpath.encode() + b'\0'))
    libraries = {
        'libopenmx_core.so': ROOT / 'build/cuda/core/libopenmx_core.so',
        'libopenmx_host_simd.so': ROOT / 'build/cuda/core/host-simd/lib/libopenmx_host_simd.so',
    }
    for name, source in libraries.items():
        shutil.copy2(source, A / 'lib' / name)
    for data in old['datasets']:
        relative = Path(data['path'])
        assert sha(ROOT / relative) == sha(PREVIOUS / relative) == data['sha256']
        (A / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PREVIOUS / relative, A / relative)
    original = ROOT / 'benchmarks/inputs/cghelgeo.dat'
    assert sha(original) == old['input_sha256']
    shutil.copy2(original, A / 'original-cghelgeo.dat')
    seed_files = json.loads((PREVIOUS / 'runs/gpu_full_fp64_same_seed/result.json').read_text())['restart_seed_files']
    seed = PREVIOUS / 'runs/gpu_default/cghelgeo_rst'
    assert sorted(str(p.relative_to(seed)) for p in seed.rglob('*') if p.is_file()) == sorted(d['path'] for d in seed_files)
    for data in seed_files:
        assert sha(seed / data['path']) == data['sha256']
    shutil.copytree(seed, A / 'seed/cghelgeo_rst')
    for name in ['rank.py', 'cusolver_audit.cpp', 'cusolver_audit.so']:
        shutil.copy2(PREVIOUS / name, A / name)
    assert sha(A / 'cusolver_audit.so') == old['cuSOLVER_audit']['library_sha256']
    core = ctypes.CDLL(str(A / 'lib/libopenmx_core.so'))
    core.openmx_core_host_simd_mode.argtypes = [ctypes.c_char_p]
    core.openmx_core_host_simd_mode.restype = ctypes.c_int
    core.openmx_core_host_simd_name.argtypes = [ctypes.c_int]
    core.openmx_core_host_simd_name.restype = ctypes.c_char_p
    selected = core.openmx_core_host_simd_name(core.openmx_core_host_simd_mode(b'avx512')).decode()
    assert selected.lower().replace('-', '') == 'avx512'
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    provenance = dict(source_commit=source_commit, criterion=1e-10, MPI_ranks=18, OMP_threads=2,
        purpose='Matched-seed continuation correctness regression; not a cold-start benchmark.',
        input_sha256=sha(original), datasets=old['datasets'],
        binaries=dict(built_sha256=sha(built), snapshot_sha256=sha(binary)),
        libraries={name: sha(A / 'lib' / name) for name in libraries},
        snapshot_rpath=dict(old=rpath, new=new_rpath,
            change='Equal-length RPATH reorder only; prefer frozen core libraries.'),
        selected_host_SIMD=selected, seed_files=seed_files,
        seed_source=str(seed), seed_source_commit=old['source_commit'],
        reference='benchmarks/results/rtx5080_cghelgeo_strict_20261108/upstream',
        reference_binary_sha256=old['binaries']['upstream_sha256'],
        audit_library_sha256=sha(A / 'cusolver_audit.so'),
        audit_source_sha256=sha(A / 'cusolver_audit.cpp'))
    save(A / 'provenance.json', provenance)
    return provenance


def replace_input(text, key, value):
    pattern = r'(?im)^\s*' + re.escape(key) + r'\s+[^\n]*$'
    if re.search(pattern, text):
        return re.sub(pattern, key + ' ' + value, text)
    return text + '\n' + key + ' ' + value + '\n'


def snapshot_ranks(folder):
    for path in folder.glob('rank-??.json'):
        data = json.loads(path.read_text())
        base = Path('/proc') / str(data['pid'])
        try:
            if Path(os.fsdecode((base / 'cmdline').read_bytes().split(b'\0')[0])).name != 'openmx':
                continue
            (folder / (path.stem + '-maps.txt')).write_text((base / 'maps').read_text())
            threads = []
            for task in (base / 'task').iterdir():
                try:
                    mask = re.search(r'^Cpus_allowed_list:\s*(.*)', (task / 'status').read_text(), re.M)[1]
                    threads.append(dict(tid=int(task.name), cpus=mask))
                except (OSError, TypeError):
                    pass
            save(folder / (path.stem + '-threads.json'), threads)
        except (OSError, TypeError, IndexError):
            pass


def run_one(variant, provenance):
    folder = A / 'runs' / variant
    folder.mkdir()
    for data in provenance['seed_files']:
        assert sha(A / 'seed/cghelgeo_rst' / data['path']) == data['sha256']
    shutil.copytree(A / 'seed/cghelgeo_rst', folder / 'cghelgeo_rst')
    text = (A / 'original-cghelgeo.dat').read_text()
    changes = {'DATA.PATH': str(A / 'DFT_DATA19'), 'scf.criterion': '1e-10',
               'level.of.stdout': '2', 'scf.restart': 'on'}
    if variant != 'gpu_default':
        changes['scf.fp32.enable'] = 'off'
    if variant == 'gpu_off_native':
        changes['scf.gemmul8.enable'] = 'off'
    if variant == 'cpu_avx512_off':
        changes['scf.eigen.lib'] = 'elpa2'
    for key, value in changes.items():
        text = replace_input(text, key, value)
    (folder / 'input.dat').write_text(text)
    env = clean_environment()
    if variant.startswith('gpu'):
        env.update(OPENMX_GPU='1', CUDA_VISIBLE_DEVICES='0',
                   CUDA_MPS_PIPE_DIRECTORY=str(A / 'mp'), CUDA_MPS_LOG_DIRECTORY=str(A / 'mps-log'),
                   OPENMX_CUSOLVER_VERBOSE='1', OPENMX_GEMMUL8_TIMING='1',
                   OMX_PRECISION_AUDIT_LIBRARY=str(A / 'cusolver_audit.so'))
        if variant == 'gpu_off_native':
            env['OPENMX_CUSOLVER_EMULATION'] = '0'
    else:
        env.update(OPENMX_GPU='0', OPENMX_HOST_SIMD='avx512')
    command = [str(MPI), '--bind-to', 'none', '-np', '18', sys.executable,
               str(A / 'rank.py'), str(A / 'bin/openmx'), 'input.dat', '-nt', '2']
    record = dict(variant=variant, status='running', command=command, criterion=1e-10,
        input_sha256=sha(folder / 'input.dat'), binary_sha256=sha(A / 'bin/openmx'),
        environment={k: v for k, v in env.items() if k.startswith(('OPENMX_', 'OMX_', 'OMP_', 'MKL_', 'OPENBLAS_', 'BLIS_', 'CUDA_'))},
        initialization='Same converged default-GPU seed (82 prior SCF steps); scf.restart on.',
        start_time=time.strftime('%Y-%m-%d %H:%M:%S %z'))
    save(folder / 'result.json', record)
    print('START', variant, flush=True)
    start = time.monotonic()
    peak = 0
    minimum = available_bytes()
    next_snapshot = 0
    samples = []
    process = None
    try:
        with (folder / 'run.log').open('w') as log:
            process = subprocess.Popen(command, cwd=folder, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                elapsed = time.monotonic() - start
                available = available_bytes()
                rss = group_rss(process.pid)
                peak = max(peak, rss)
                minimum = min(minimum, available)
                save(A / 'progress.json', dict(status='running', variant=variant, pid=process.pid,
                    elapsed_seconds=round(elapsed, 1), peak_RSS_gib=round(peak / 2**30, 3),
                    available_gib=round(available / 2**30, 3)))
                if elapsed >= next_snapshot:
                    snapshot_ranks(folder)
                    samples.append(dict(seconds=elapsed, rss_gib=rss / 2**30, available_gib=available / 2**30))
                    save(folder / 'resources.json', samples)
                    next_snapshot = elapsed + 15
                    if MPS:
                        mps = control('get_server_list')
                        save(A / 'mps-server.json', dict(returncode=mps.returncode, stdout=mps.stdout, stderr=mps.stderr))
                if STOP or (A / 'STOP').exists() or available < 8 * 2**30 or elapsed > 14400:
                    stop_job(process)
                    raise RuntimeError('Stopped or memory/time limit reached')
                time.sleep(2)
            code = process.wait()
        log = (folder / 'run.log').read_text()
        assert code == 0 and log.count('The calculation was normally finished.') == 1, (variant, code)
        assert log.count('The SCF was achieved at MD=') == 1 and 'The SCF was not fully achieved' not in log, variant
        assert '<Restart>  Found restart files' in log and '<Restart>  Could not find restart files' not in log
        energy, atoms, forces, grid = read_output(folder / 'cghelgeo.out')
        assert len(atoms) == 650 and len(forces) == 1950 and all(map(math.isfinite, [energy] + forces))
        phases = {name.strip(): dict(min=float(lo), max=float(hi)) for name, lo, hi in re.findall(
            r'^\s*([^=\n]+?)\s*=\s+\d+\s+([\d.]+)\s+\d+\s+([\d.]+)\s*$', log, re.M)}
        record.update(status='completed', returncode=code, converged=True,
            end_time=time.strftime('%Y-%m-%d %H:%M:%S %z'), wall_seconds=time.monotonic() - start,
            total_seconds=phases['Total Computational Time']['max'], phases_seconds=phases,
            scf_steps=len(re.findall(r'MD=\s*1\s+SCF=\s*\d+', log)), energy_hartree=energy,
            atoms=atoms, forces_hartree_per_bohr=forces, FFT_grid=grid, peak_RSS_gib=peak / 2**30,
            min_available_gib=minimum / 2**30, output_sha256=sha(folder / 'cghelgeo.out'))
        save(folder / 'result.json', record)
        print('DONE', variant, record['scf_steps'], energy, flush=True)
        return record
    finally:
        if process is not None and process.poll() is None:
            stop_job(process)


def main():
    provenance = prepare()
    results = []
    try:
        start_mps()
        for variant in VARIANTS:
            if variant.startswith('cpu'):
                stop_mps()
            assert sha(A / 'bin/openmx') == provenance['binaries']['snapshot_sha256']
            results.append(run_one(variant, provenance))
            save(A / 'results.json', results)
        save(A / 'progress.json', dict(status='completed', runs=len(results)))
    except BaseException as exc:
        save(A / 'failure.json', dict(error=str(exc), completed=len(results)))
        raise
    finally:
        stop_mps()


if __name__ == '__main__':
    main()
