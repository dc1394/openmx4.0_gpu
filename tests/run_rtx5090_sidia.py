#!/usr/bin/env python3
"""Two-GPU MAGMA/GEMMul8/FP32 checks for the supplied Si555/Si666 cells.

Source build/env.sh first. Use --cells 2 for a small numerical regression,
omit it for the exact supplied atom counts. All Band runs keep Regular
2x2x2 sampling. Cluster retains the periodic cell and uses Gamma only.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from types import SimpleNamespace

from run_mgpu_sidia_benchmark import silicon_input, descendant_rss
from run_mgpu_cluster_regression import REPO, digest, keyword, read_output, save, stop_job

CASES = ('cluster_col', 'cluster_nc', 'band_col', 'band_nc')


def prepare(case, mode, args):
    cells = args.cells or (6 if case == 'cluster_col' else 5)
    text, atoms = silicon_input(case, SimpleNamespace(
        cells=cells, datapath=REPO / 'DFT_DATA19', gpus=2, scf_iterations=args.steps))
    settings = {
        'System.Name': 'sidia', 'level.of.stdout': 1, 'level.of.fileout': 0,
        'scf.Generation.Kpoint': 'Regular',
        'scf.Kgrid': '2 2 2' if case.startswith('band') else '1 1 1',
        'scf.eigen.lib': 'gpusolver3', 'scf.mgpu.enable': 'on',
        'scf.fp32.enable': 'on' if mode == 'mixed' else 'off',
        'scf.gemmul8.enable': 'off' if mode == 'native' else 'on',
    }
    for key, value in settings.items():
        text = keyword(text, key, value)
    if args.scf_only:
        text = keyword(text, 'orbitalOpt.Force.Skip', 'on')
    text = text.replace('# Band is periodic with two GCenter k-points.',
                        '# Band keeps the supplied Regular 2x2x2 mesh.')
    text = text.replace("user's sidia555.dat", "user's sidia666.dat" if cells == 6 else "user's sidia555.dat")
    env = {k: v for k, v in os.environ.items() if not k.startswith(('OPENMX_', 'GEMMUL8_'))}
    controls = {
        'CUDA_VISIBLE_DEVICES': '0,1',
        'OMP_NUM_THREADS': str(args.threads), 'OPENBLAS_NUM_THREADS': str(args.threads),
        'MAGMA_NUM_THREADS': str(args.threads),
        'OPENMX_MGPU_BAND_SHARED': '1', 'OPENMX_MGPU_NUM_GPUS': '2',
        'OPENMX_MGPU_VERBOSE': '1', 'OPENMX_MAGMA_FP32': '1' if mode == 'mixed' else '0',
        'OPENMX_MAGMA_FP32_COMPLEX_MAX_N': str(args.complex_fp32_max_n),
        'OPENMX_MGPU_GEMM_WORKSPACE_MB': str(args.workspace_mb),
        'OPENMX_MGPU_GEMM_HOST_MB': str(args.workspace_mb),
        'OPENMX_GEMMUL8_MAX_WORKSPACE_MB': str(args.gemmul8_mb or (6144 if case.endswith('_col') else 2048)),
        'OPENMX_MGPU_GEMM_PAGEABLE': '1',
        'OPENMX_MAGMA_REFINE_STEPS': str(args.refine_steps),
        'OPENMX_MAGMA_REFINE_RESIDENT_MB': str(args.resident_mb),
        'OPENMX_MAGMA_REFINE_RESERVE_MB': '2048',
        # The bridge timer synchronizes each GPU before queuing the next one.
        # Keep it off for concurrent multi-GPU performance measurements.
        'OPENMX_GEMMUL8_TIMING': '0', 'OPENMX_GEMMUL8_VERBOSE': '1',
        'OPENMX_CLUSTER_PROFILE': '1', 'OPENMX_BAND_PROFILE': '1',
        'OPENMX_FORCE_PROFILE': '1',
        'OPENMX_BAND_GPU_MAX_CONCURRENT_K': '1',
        # Retain backtransformed states until the chemical potential is known.
        # The original 1 GiB cap otherwise repeats large eigensolves in DM.
        'OPENMX_BAND_COL_KCACHE_MB': str(args.col_kcache_mb),
        'OPENMX_BAND_NONCOL_KCACHE_MB': str(args.nc_kcache_mb),
        'OPENMX_BAND_CACHE_TRACE': '1', 'OPENMX_BAND_NONCOL_CACHE_TRACE': '1',
        'OPENMX_CLUSTER_GPU_SWITCH_NUM': '1', 'OPENMX_CLUSTER_NONCOL_GPU_SWITCH_NUM': '1',
        'OPENMX_BAND_GPU_SWITCH_NUM': '1', 'OPENMX_BAND_NONCOL_GPU_SWITCH_NUM': '1',
    }
    env.update(controls)
    return text, atoms, env, controls


def run(case, mode, args):
    folder = args.output / case / mode
    folder.mkdir(parents=True, exist_ok=False)
    text, atoms, env, controls = prepare(case, mode, args)
    (folder / 'input.dat').write_text(text)
    command = ['/usr/bin/time', '-f', '%e', '-o', 'wall-time.txt',
               'mpirun', '--bind-to', 'none', '-np', str(args.ranks),
               str(args.binary), 'input.dat', '-nt', str(args.threads)]
    result = dict(case=case, mode=mode, atoms=atoms, command=command, scf_only=args.scf_only,
                  controls=controls, input_sha256=digest(folder / 'input.dat'),
                  binary_sha256=digest(args.binary), folder=str(folder), status='running')
    save(folder / 'result.json', result)
    print(f'START {case}/{mode}: {atoms} atoms, {args.steps} SCF step(s)', flush=True)
    begin = time.monotonic()
    samples, peak_rss, early_error = [], 0, None
    with (folder / 'run.log').open('w') as log:
        process = subprocess.Popen(command, cwd=folder, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - begin > args.timeout:
                    early_error = 'calculation timeout'
                    break
                recent = (folder / 'run.log').read_text(errors='replace')
                if re.search(r'using CPU LAPACK|using CPU BLAS|FP32 refinement declined|GPU initialization failed|Falling back to native cuBLAS', recent):
                    early_error = 'requested accelerated path fell back; stopped before further computation'
                    break
                peak_rss = max(peak_rss, descendant_rss(process.pid))
                gpu = subprocess.check_output([
                    'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True)
                samples.append({'seconds': time.monotonic()-begin, 'gpus': gpu.strip()})
                time.sleep(5)
        finally:
            if process.poll() is None:
                stop_job(process)
            save(folder / 'gpu-samples.json', samples)
    result.update(returncode=process.returncode, wall_seconds=time.monotonic()-begin,
                  peak_process_rss_bytes=peak_rss)
    if (folder / 'wall-time.txt').exists() and process.returncode == 0:
        result['wall_seconds'] = float((folder / 'wall-time.txt').read_text().strip())
    log = (folder / 'run.log').read_text(errors='replace')
    try:
        if early_error:
            raise ValueError(early_error)
        if process.returncode or 'normally finished' not in log.lower():
            raise ValueError('OpenMX did not finish normally')
        if re.search(r'using CPU LAPACK|using CPU BLAS|FP32 refinement declined|GPU initialization failed|Falling back to native cuBLAS', log):
            raise ValueError('requested accelerated path fell back; inspect run.log')
        solves = re.findall(r'solver selected[^\n]*backend=magma[^\n]*valid=1', log)
        refined = re.findall(r'FP32 refined[^\n]*valid=1', log)
        if not solves or any('gpus=2 ' not in line for line in solves):
            raise ValueError('missing successful two-GPU MAGMA solve')
        planned_native = [line for line in solves if 'type=Z ' in line and
                          args.complex_fp32_max_n > 0 and
                          int(re.search(r'\bn=(\d+)', line)[1]) > args.complex_fp32_max_n]
        if mode == 'mixed' and len(refined) != len(solves)-len(planned_native):
            raise ValueError('MAGMA precision use differs from the requested policy')
        policies = re.findall(r'precision policy[^\n]*mode=FP64', log)
        if mode == 'mixed' and len(policies) != len(planned_native):
            raise ValueError('missing explicit precision-policy records')
        if case.startswith('band'):
            expected = 8 if case.endswith('_nc') else 4
            points = set(map(int, re.findall(r'solver selected[^\n]*kpoint=(\d+)', log)))
            if points != set(range(expected)):
                raise ValueError(f'unexpected k-point coverage: {points}')
        energy, coordinates, forces, grids = read_output(folder / 'sidia.out')
        if len(coordinates) != atoms or len(forces) != 3*atoms:
            raise ValueError('incomplete energy/force output')
        band_energies = list(map(float, re.findall(r'<DFT>\s+Uele\s*=\s*([-+\d.eE]+)', log)))
        if not band_energies:
            raise ValueError('missing SCF band energy')
        if not all(math.isfinite(v) for v in [energy, *forces, *band_energies]):
            raise ValueError('nonfinite energy or force output')
        iterations = sorted(set(map(int, re.findall(r'MD=\s*1\s+SCF=\s*(\d+)', log))))
        if not iterations or iterations != list(range(1, iterations[-1]+1)) or iterations[-1] > args.steps:
            raise ValueError(f'unexpected SCF iteration sequence: {iterations}')
        result.update(status='passed', energy_hartree=None if args.scf_only else energy,
                      band_energies_hartree=band_energies,
                      forces_hartree_per_bohr=[] if args.scf_only else forces, grids=grids,
                      scf_iterations=iterations,
                      magma_solves=len(solves), refined_solves=len(refined),
                      planned_fp64_solves=len(planned_native) if mode == 'mixed' else 0,
                      band_cache_profiles=[s for s in log.splitlines() if re.match(r'BAND_(?:COL|NONCOL)_CACHE ', s)],
                      phase_profiles=[s for s in log.splitlines() if 'PROF' in s])
        timing = re.search(r'Total Computational Time\s*=\s*\d+\s+([\d.]+)\s+\d+\s+([\d.]+)', log)
        if timing:
            result['application_seconds'] = float(timing[2])
    except Exception as error:
        result.update(status='failed', error=str(error))
    save(folder / 'result.json', result)
    print(f"{result['status'].upper()} {case}/{mode}: {result['wall_seconds']:.1f} s", flush=True)
    if result['status'] != 'passed':
        raise RuntimeError(f'{folder}: {result.get("error")}')
    if not args.keep_restart:
        # These benchmark-owned restart files can exceed 1 GiB per case.
        # Inputs, numerical outputs and logs remain for reproducibility.
        shutil.rmtree(folder / 'sidia_rst', ignore_errors=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=REPO / 'build/cuda/openmx')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--modes', nargs='+', choices=('native', 'gemmul8', 'mixed'), default=['native', 'mixed'])
    parser.add_argument('--cells', type=int)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--ranks', type=int, default=2)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--workspace-mb', type=int, default=24576)
    parser.add_argument('--gemmul8-mb', type=int, help='Default: 6144 for collinear, 2048 for noncollinear')
    parser.add_argument('--resident-mb', type=int, default=28672)
    parser.add_argument('--col-kcache-mb', type=int, default=4096)
    parser.add_argument('--nc-kcache-mb', type=int, default=24576)
    parser.add_argument('--refine-steps', type=int, choices=range(2, 6), default=3)
    parser.add_argument('--complex-fp32-max-n', type=int, default=16384,
                        help='Use native FP64 MAGMA above this complex dimension; 0 removes the limit')
    parser.add_argument('--timeout', type=int, default=14400)
    parser.add_argument('--keep-restart', action='store_true')
    parser.add_argument('--scf-only', action='store_true',
                        help='Skip force/total-energy evaluation; compare SCF Uele instead of total energy')
    args = parser.parse_args()
    args.output, args.binary = args.output.resolve(), args.binary.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    results, comparisons = [], []
    for case in args.cases:
        baseline = None
        for mode in args.modes:
            value = run(case, mode, args)
            results.append(value)
            save(args.output / 'results.json', results)
            if baseline is None:
                baseline = value
            else:
                de = abs(value['band_energies_hartree'][-1]-baseline['band_energies_hartree'][-1]) if args.scf_only else abs(value['energy_hartree']-baseline['energy_hartree'])
                df = None if args.scf_only else max(abs(a-b) for a,b in zip(value['forces_hartree_per_bohr'], baseline['forces_hartree_per_bohr']))
                item = dict(case=case, baseline=baseline['mode'], mode=mode,
                            energy_kind='SCF Uele' if args.scf_only else 'total energy',
                            speedup=baseline['wall_seconds']/value['wall_seconds'],
                            energy_error_hartree=de, energy_error_hartree_per_atom=de/value['atoms'],
                            max_force_error_hartree_per_bohr=df)
                comparisons.append(item)
                save(args.output / 'comparisons.json', comparisons)
                print(json.dumps(item), flush=True)
                if de/value['atoms'] > 1e-8 or (df is not None and df > 1e-7):
                    raise RuntimeError('Comparison exceeds 1e-8 Ha/atom or 1e-7 Ha/bohr tolerance')


if __name__ == '__main__':
    main()
