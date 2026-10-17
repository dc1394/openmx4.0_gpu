#!/usr/bin/env python3
"""Check native, automatic-capacity, forced-MAGMA and GPU-limit CPU fallback.

Run inside a four-GPU single-node allocation. A large *reserve*, not an
allocation, exercises genuine hipSOLVER admission failure on eight Si atoms.
The automatic cases deliberately omit scf.mgpu.enable; gpusolver3 is tested
with OPENMX_MGPU_SOLVER=single and OPENMX_MGPU_GEMM=0 in the environment.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

from run_mgpu_sidia_benchmark import CASES, silicon_input
from run_mgpu_cluster_regression import REPO, compare, digest, keyword, read_output, save, stop_job
from run_mgpu_band_regression import records


def run_case(args, case, mode):
    folder = args.output / case / mode
    folder.mkdir(parents=True)
    band = case.startswith('band_')
    limit = (3 if band else 1) if mode == 'limit' else (4 if band else 2)
    text, atoms = silicon_input(case, SimpleNamespace(
        cells=1, datapath=args.datapath, scf_iterations=1, gpus=limit))
    text = keyword(text, 'scf.eigen.lib', 'gpusolver3' if mode == 'forced' else 'gpusolver')
    (folder / 'input.dat').write_text(text)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('OPENMX_', 'GEMMUL8_'))}
    controls = {
        'OMP_NUM_THREADS': '2', 'MAGMA_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
        'OPENMX_EIGEN_REFINE': '0', 'OPENMX_MGPU_SOLVER': 'single',
        'OPENMX_MGPU_GEMM': '0', 'OPENMX_MGPU_NUM_GPUS': '4', 'OPENMX_MGPU_VERBOSE': '1',
        'OPENMX_CLUSTER_GPU_SWITCH_NUM': '1', 'OPENMX_CLUSTER_NONCOL_GPU_SWITCH_NUM': '1',
        'OPENMX_BAND_GPU_SWITCH_NUM': '1', 'OPENMX_BAND_NONCOL_GPU_SWITCH_NUM': '1',
    }
    if mode in ('auto', 'limit'):
        # Exceed one MI300A's shared memory as well as discrete-card VRAM.
        # This is an admission reserve, never a real allocation.
        controls['OPENMX_BAND_GPU_RESERVE_MB'] = '1048576'
        controls['OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB'] = '1048576'
    env.update(controls)
    cmd = [args.mpirun, '-np', '1', '--map-by', 'slot:PE=2', '--bind-to', 'core',
           str(args.binary), 'input.dat', '-nt', '2']
    result = dict(case=case, mode=mode, command=cmd, controls=controls, gpu_limit=limit,
                  input_sha256=digest(folder / 'input.dat'), binary_sha256=digest(args.binary))
    with (folder / 'openmx.log').open('w') as log:
        process = subprocess.Popen(cmd, cwd=folder, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=args.timeout)
        finally:
            if process.poll() is None:
                stop_job(process)
    result['returncode'] = process.returncode
    log = (folder / 'openmx.log').read_text(errors='replace')
    ops = records(log)
    result['operations'] = ops
    if mode == 'limit':
        if process.returncode != 0 or 'GPU limit will not be exceeded' not in log:
            raise ValueError(f'{case}: expected successful GPU-limit CPU fallback')
        if any(op['kind'] in ('solver', 'gemm') for op in ops):
            raise ValueError(f'{case}: insufficient pool must use CPU dense work')
        if 'using the CPU eigensolver and matrix products' not in log or 'normally finished' not in log:
            raise ValueError(f'{case}: missing successful CPU routing diagnostic')
    else:
        if process.returncode != 0 or 'normally finished' not in log:
            raise ValueError(f'{case}/{mode}: OpenMX failed')
        if mode == 'native':
            if any(op['kind'] in ('solver', 'gemm') for op in ops):
                raise ValueError(f'{case}: native hipSOLVER unexpectedly cooperated')
        else:
            banner = ('host-backed MAGMA capacity path.' if band else 'checking capacity recovery.') \
                if mode == 'auto' else 'scf.eigen.lib=gpusolver3'
            if banner not in log:
                raise ValueError(f'{case}/{mode}: missing routing diagnostic')
            solvers = [op for op in ops if op['kind'] == 'solver']
            if not solvers or any(op.get('backend') != 'magma' or op.get('storage') != 'host'
                                  or op.get('valid') != '1' for op in solvers):
                raise ValueError(f'{case}/{mode}: expected successful host MAGMA')
            dense = [op for op in ops if op['kind'] in ('solver', 'gemm')]
            for op in dense:
                devices = list(map(int, op['devices'].split(',')))
                if len(devices) != 2 or any(d < 0 or d >= limit for d in devices):
                    raise ValueError(f'{case}/{mode}: GPU limit/group violated: {devices}')
            if band and {op.get('devices') for op in solvers} != {'0,1', '2,3'}:
                raise ValueError(f'{case}/{mode}: incomplete k-point GPU groups')
    energy, coordinates, forces, grids = read_output(folder / 'sidia.out')
    if len(coordinates) != atoms or len(forces) != 3 * atoms:
        raise ValueError(f'{case}/{mode}: incomplete atoms/forces')
    components = {key: float(value.replace('D', 'E').replace('d', 'e'))
                  for key, value in re.findall(r'^\s*((?:U\w+|Enpy))\.\s+(\S+)',
                  (folder / 'sidia.out').read_text(), re.M)}
    if not components or not all(math.isfinite(v) for v in
            [energy] + list(forces) + list(components.values())):
        raise ValueError(f'{case}/{mode}: missing or nonfinite result')
    result.update(energy_hartree=energy, atoms=coordinates,
                  forces_hartree_per_bohr=forces, grids=grids, energy_components=components)
    save(folder / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datapath', type=Path, default=REPO / 'DFT_DATA19')
    parser.add_argument('--mpirun', default='mpirun')
    parser.add_argument('--timeout', type=float, default=90)
    args = parser.parse_args()
    args.binary, args.output, args.datapath = (p.resolve() for p in (args.binary, args.output, args.datapath))
    args.output.mkdir(parents=True, exist_ok=False)
    args.energy_tolerance = args.force_tolerance = 1e-7
    results, comparisons = [], []
    for case in CASES:
        baseline = None
        for mode in ('native', 'auto', 'forced', 'limit'):
            print(f'Checking {case}/{mode}', flush=True)
            result = run_case(args, case, mode)
            results.append(result)
            if mode == 'native':
                baseline = result
            else:
                comparison = compare(baseline, result, args)
                comparisons.append(dict(case=case, mode=mode, **comparison))
                if not comparison['passed']:
                    raise ValueError(f'{case}/{mode}: numerical comparison failed')
            save(args.output / 'results.json', results)
            save(args.output / 'comparisons.json', comparisons)
    save(args.output / 'summary.json', dict(passed=True, runs=len(results), comparisons=len(comparisons)))
    print(json.dumps(dict(passed=True, runs=len(results), comparisons=len(comparisons))), flush=True)


if __name__ == '__main__':
    main()
