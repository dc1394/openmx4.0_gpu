#!/usr/bin/env python3
"""Two-rank, two-GPU DC/DC-LNO/Krylov validation on the supplied Si cells.

These local solvers use gpusolver (FP64 cuSOLVER), not gpusolver3/MAGMA.
Mixed mode exercises FP32 double-float grid kernels and GEMMul8. Build the
test observer first; see doc/rtx5090_ordern.txt for commands and limitations.
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

CASES = ('dc_col', 'dclno_col', 'krylov_col', 'dc_nc', 'dclno_nc')
SOLVERS = {'dc': 'DC', 'dclno': 'DC-LNO', 'krylov': 'Krylov'}
FAILURE = re.compile(r'falling back to (?:the )?(?:LAPACK|CPU)|using the CPU eigensolver instead|'
                     r'GPU initialization failed|Falling back to native cuBLAS|'
                     r'Trying native cuBLAS instead|Retrying this DC Hamiltonian solve|'
                     r'eigensolve did not converge|ORDERNTRACE error', re.I)


def prepare(case, mode, args):
    cells = args.cells or (5 if case.endswith('_nc') or case == 'dclno_col' else 6)
    text, atoms = silicon_input('cluster_nc' if case.endswith('_nc') else 'cluster_col',
        SimpleNamespace(cells=cells, datapath=REPO / 'DFT_DATA19', gpus=2, scf_iterations=args.steps))
    method = case.split('_')[0]
    text = '\n'.join(line for line in text.splitlines() if not line.startswith('#')) + '\n'
    text = f'# Si diamond {cells}x{cells}x{cells}; local {SOLVERS[method]} solver validation.\n' + text
    settings = {'level.of.stdout': 2, 'level.of.fileout': 0,
                'scf.EigenvalueSolver': SOLVERS[method], 'scf.eigen.lib': 'gpusolver',
                'scf.fp32.enable': 'on' if mode == 'mixed' else 'off',
                'scf.gemmul8.enable': 'off' if mode == 'native' else 'on',
                'orderN.HoppingRanges': '8.0' if method == 'dclno' else '7.0',
                'orderN.NumHoppings': 2, 'orderN.KrylovH.order': 400}
    if args.scf_only:
        settings['orbitalOpt.Force.Skip'] = 'on'
    for key, value in settings.items():
        text = keyword(text, key, value)
    if method == 'dclno':
        text += '\n<LNOs.Num\n Si 4\nLNOs.Num>\n'
    env = {k:v for k,v in os.environ.items() if not k.startswith(('OPENMX_', 'GEMMUL8_'))}
    controls = {'CUDA_VISIBLE_DEVICES': '0,1', 'OMP_NUM_THREADS': str(args.threads),
                'OPENBLAS_NUM_THREADS': str(args.threads), 'MAGMA_NUM_THREADS': str(args.threads),
                'OPENMX_MGPU_NUM_GPUS': '2', 'OPENMX_BAND_PROFILE': '1',
                'OPENMX_DCLNO_PROFILE': '1', 'OPENMX_KRYLOV_GPU_PROFILE': '1',
                'OPENMX_GEMMUL8_TIMING': '1' if args.gemm_timing else '0',
                'OPENMX_GEMMUL8_VERBOSE': '1', 'OPENMX_GEMMUL8_MAX_WORKSPACE_MB': '2048'}
    env.update(controls)
    return text, atoms, env, controls


def run(case, mode, args):
    folder = args.output / case / mode
    folder.mkdir(parents=True, exist_ok=False)
    text, atoms, env, controls = prepare(case, mode, args)
    (folder / 'input.dat').write_text(text)
    # Preload only the application, not mpirun, time or nvidia-smi.
    preload = str(args.observer) + (':' + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
    command = ['/usr/bin/time', '-f', '%e', '-o', 'wall-time.txt',
               'mpirun', '--bind-to', 'none', '--tag-output', '-np', '2',
               'env', f'LD_PRELOAD={preload}', str(args.binary), 'input.dat', '-nt', str(args.threads)]
    result = dict(case=case, mode=mode, atoms=atoms, command=command, scf_only=args.scf_only,
                  controls=controls, input_sha256=digest(folder/'input.dat'),
                  binary_sha256=digest(args.binary), observer_sha256=digest(args.observer),
                  folder=str(folder), status='running')
    save(folder/'result.json', result)
    print(f'START {case}/{mode}: {atoms} atoms, {args.steps} SCF step(s)', flush=True)
    begin, samples, peak_rss, early_error = time.monotonic(), [], 0, None
    with (folder/'run.log').open('w') as log:
        process = subprocess.Popen(command, cwd=folder, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic()-begin > args.timeout:
                    early_error = 'calculation timeout'
                    break
                recent = (folder/'run.log').read_text(errors='replace')
                if FAILURE.search(recent):
                    early_error = 'accelerated path failed; inspect run.log'
                    break
                peak_rss = max(peak_rss, descendant_rss(process.pid))
                gpu = subprocess.check_output(['nvidia-smi',
                    '--query-gpu=index,uuid,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True)
                apps = subprocess.check_output(['nvidia-smi',
                    '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                    '--format=csv,noheader,nounits'], text=True)
                samples.append(dict(seconds=time.monotonic()-begin, gpus=gpu.strip(), applications=apps.strip()))
                time.sleep(5)
        finally:
            if process.poll() is None:
                stop_job(process)
            save(folder/'gpu-samples.json', samples)
    result.update(returncode=process.returncode, wall_seconds=time.monotonic()-begin,
                  peak_process_rss_bytes=peak_rss)
    if (folder/'wall-time.txt').exists() and process.returncode == 0:
        result['wall_seconds'] = float((folder/'wall-time.txt').read_text().strip())
    log = (folder/'run.log').read_text(errors='replace')
    try:
        if early_error:
            raise ValueError(early_error)
        if process.returncode or 'normally finished' not in log.lower():
            raise ValueError('OpenMX did not finish normally')
        if FAILURE.search(log):
            raise ValueError('accelerated path failed')
        density = [dict(zip(('rank','batches','mode'), map(int, row))) for row in re.findall(
            r'SETHOTFDENPROF id=(\d+) batches=(\d+) workspace_mib=\S+ mode=(\d+)', log)]
        expected_mode = 2 if mode == 'mixed' else 0
        if {r['rank'] for r in density if r['batches'] > 0 and r['mode'] == expected_mode} != {0,1}:
            raise ValueError(f'missing density kernels of mode={expected_mode} on both ranks')
        if any(r['mode'] != expected_mode for r in density):
            raise ValueError('density precision changed unexpectedly during the run')
        traces = [dict(zip(('rank','pid','device','real64','complex64','real32','complex32','errors'), map(int, row)))
                  for row in re.findall(r'ORDERNTRACE total rank=(\d+) pid=(\d+) device=(\d+) real64=(\d+) complex64=(\d+) real32=(\d+) complex32=(\d+) errors=(\d+)', log)]
        if {(r['rank'],r['device']) for r in traces} != {(0,0),(1,1)}:
            raise ValueError('missing actual cuSOLVER calls on both assigned GPUs')
        if any(r['errors'] or r['real32'] or r['complex32'] for r in traces):
            raise ValueError('unexpected local eigensolver precision or API error')
        gemms = [dict(zip(('rank','real_gemmul8','real_cublas','complex_gemmul8','complex_cublas'), map(int,row)))
                 for row in re.findall(r'general GEMMs, rank (\d+): real (\d+) \+ (\d+) calls[^\n]*?complex (\d+) \+ (\d+) calls',log)]
        # DC NC uses its own OpenACC contraction loops, not the GEMMul8
        # bridge. Do not mistake an input enable flag for an executed GEMM.
        if args.gemm_timing and case != 'dc_nc':
            field = 'cublas' if mode == 'native' else 'gemmul8'
            if {r['rank'] for r in gemms if r['real_'+field]+r['complex_'+field] > 0} != {0,1}:
                raise ValueError(f'missing {field} execution counters on both ranks')
        for r in traces:
            uuid = next(line.split(',')[1].strip() for line in samples[-1]['gpus'].splitlines()
                        if int(line.split(',')[0]) == r['device'])
            if not any(f'{uuid}, {r["pid"]},' in s['applications'] for s in samples):
                raise ValueError('missing physical GPU/process corroboration')
        energy, coordinates, forces, grids = read_output(folder/'sidia.out')
        band_energies = list(map(float, re.findall(r'<DFT>\s+Uele\s*=\s*([-+\d.eE]+)', log)))
        if len(coordinates) != atoms or len(forces) != 3*atoms or not band_energies:
            raise ValueError('incomplete numerical output')
        if not all(math.isfinite(v) for v in [energy, *forces, *band_energies]):
            raise ValueError('nonfinite energy or forces')
        iterations = sorted(set(map(int, re.findall(r'MD=\s*1\s+SCF=\s*(\d+)', log))))
        if iterations != list(range(1, args.steps+1)):
            raise ValueError(f'unexpected SCF iteration sequence: {iterations}')
        result.update(status='passed', energy_hartree=None if args.scf_only else energy,
                      band_energies_hartree=band_energies, grids=grids,
                      forces_hartree_per_bohr=[] if args.scf_only else forces,
                      scf_iterations=iterations, density_kernels=density, local_solver_calls=traces,
                      gemm_calls=gemms,
                      phase_profiles=[s for s in log.splitlines() if 'PROF' in s or 'profile>' in s or 'KRYFUSE' in s])
    except Exception as error:
        result.update(status='failed', error=str(error))
    save(folder/'result.json', result)
    print(f"{result['status'].upper()} {case}/{mode}: {result['wall_seconds']:.1f} s", flush=True)
    if result['status'] != 'passed':
        raise RuntimeError(f'{folder}: {result.get("error")}')
    shutil.rmtree(folder/'sidia_rst', ignore_errors=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=REPO/'build/cuda/openmx')
    parser.add_argument('--observer', type=Path, default=REPO/'build/rtx5090/ordern-observer.so')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES[:3]),
                        help='Default: three collinear solvers. NC DC/DC-LNO need much more host memory.')
    parser.add_argument('--modes', nargs='+', choices=('native','gemmul8','mixed'), default=['native','mixed'])
    parser.add_argument('--cells', type=int)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--timeout', type=int, default=14400)
    parser.add_argument('--scf-only', action='store_true')
    parser.add_argument('--gemm-timing', action='store_true')
    args = parser.parse_args()
    for name in ('output','binary','observer'):
        setattr(args, name, getattr(args,name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    results, comparisons = [], []
    for case in args.cases:
        baseline = None
        for mode in args.modes:
            value = run(case, mode, args)
            results.append(value)
            save(args.output/'results.json', results)
            if baseline is None:
                baseline = value
                continue
            de = abs(value['band_energies_hartree'][-1]-baseline['band_energies_hartree'][-1]) if args.scf_only else abs(value['energy_hartree']-baseline['energy_hartree'])
            df = None if args.scf_only else max(abs(a-b) for a,b in zip(value['forces_hartree_per_bohr'],baseline['forces_hartree_per_bohr']))
            item = dict(case=case, baseline=baseline['mode'], mode=mode,
                        energy_kind='SCF Uele' if args.scf_only else 'total energy',
                        speedup=baseline['wall_seconds']/value['wall_seconds'],
                        energy_error_hartree=de, energy_error_hartree_per_atom=de/value['atoms'],
                        max_force_error_hartree_per_bohr=df,
                        passed=de/value['atoms'] <= 1e-8 and (df is None or df <= 1e-7))
            comparisons.append(item)
            save(args.output/'comparisons.json',comparisons)
            print(json.dumps(item), flush=True)
    if any(not item['passed'] for item in comparisons):
        raise RuntimeError('numerical comparison failed; all requested comparisons retained')


if __name__ == '__main__':
    main()
