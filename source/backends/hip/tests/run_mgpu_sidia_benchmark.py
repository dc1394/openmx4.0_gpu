#!/usr/bin/env python3
"""Benchmark the supplied Si diamond supercell in four mainline solver paths.

Run inside an allocated single node; the caller owns MPI, GPU visibility and
the device allocation. Inputs, exact controls, binary hashes, memory samples, energies and forces
are retained. A failed or silently CPU-fallback run is never a GPU timing.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from run_mgpu_cluster_regression import REPO, compare, digest, keyword, read_output, save, stop_job
from run_mgpu_band_regression import records

CASES = ('cluster_col', 'cluster_nc', 'band_col', 'band_nc')
MODES = ('single', 'hybrid', 'magma')


def silicon_input(case, args):
    # Eight sites from the user's sidia555.dat, with x/y/z cell ordering.
    sites = ((1, 1, 1), (5, 1, 5), (1, 5, 5), (5, 5, 1),
             (7, 3, 3), (7, 7, 7), (3, 3, 7), (3, 7, 3))
    ncell, lattice = args.cells, 5.431
    nc, band = case.endswith('_nc'), case.startswith('band_')
    rows, positions = [], []
    for x in range(ncell):
        for y in range(ncell):
            for z in range(ncell):
                for site in sites:
                    pos = tuple(round(lattice * (cell + frac / 8), 7)
                                for cell, frac in zip((x, y, z), site))
                    positions.append(pos)
                    row = f'{len(rows)+1:5d} Si ' + ' '.join(f'{p:14.7f}' for p in pos) + '  2.0 2.0'
                    if nc:
                        row += '  45.0 15.0 45.0 15.0 1'
                    rows.append(row)
    atoms = 8 * ncell**3
    assert len(rows) == len(set(positions)) == atoms
    assert positions[0] == (0.678875, 0.678875, 0.678875)
    if ncell == 5:
        assert positions[40] == (0.678875, 6.109875, 0.678875)
        assert positions[200] == (6.109875, 0.678875, 0.678875)
        assert positions[-1] == (23.760625, 26.476125, 23.760625)
    side = lattice * ncell
    text = f'''# Derived from the user's sidia555.dat; coordinates/basis/cutoff retained.
# The supplied cell is retained; Cluster uses its Gamma-point solver.
# Band is periodic with two GCenter k-points.
System.CurrentDirectory ./
System.Name sidia
level.of.stdout 2
level.of.fileout 1
DATA.PATH {args.datapath}
Species.Number 1
<Definition.of.Atomic.Species
 Si Si7.0-s2p2d1 Si_PBE19
Definition.of.Atomic.Species>
Atoms.Number {atoms}
Atoms.SpeciesAndCoordinates.Unit Ang
<Atoms.SpeciesAndCoordinates
''' + '\n'.join(rows) + f'''
Atoms.SpeciesAndCoordinates>
Atoms.UnitVectors.Unit Ang
<Atoms.UnitVectors
 {side:.7f} 0.0 0.0
 0.0 {side:.7f} 0.0
 0.0 0.0 {side:.7f}
Atoms.UnitVectors>
scf.XcType GGA-PBE
scf.SpinPolarization {'nc' if nc else 'off'}
scf.SpinOrbit.Coupling {'on' if nc else 'off'}
scf.ElectronicTemperature 300.0
scf.energycutoff 200.0
scf.maxIter {args.scf_iterations}
scf.EigenvalueSolver {'Band' if band else 'Cluster'}
scf.Generation.Kpoint GCenter
scf.Kgrid {'2 1 1' if band else '1 1 1'}
scf.Mixing.Type rmm-diisk
scf.Init.Mixing.Weight 0.30
scf.Min.Mixing.Weight 0.001
scf.Max.Mixing.Weight 0.700
scf.Mixing.History 7
scf.Mixing.StartPulay 5
scf.criterion 1.0e-13
scf.restart off
scf.fp32.enable off
scf.Gpu.Num {args.gpus}
MD.Type nomd
MD.maxIter 1
MD.TimeStep 1
MD.Opt.criterion 1.0e-5
Band.dispersion off
MO.fileout off
Dos.fileout off
'''
    return text, atoms


def controls(mode, args):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('OPENMX_', 'GEMMUL8_'))}
    changes = {
        'OMP_NUM_THREADS': str(args.threads), 'OPENBLAS_NUM_THREADS': str(args.blas_threads),
        'MAGMA_NUM_THREADS': str(args.blas_threads), 'OPENMX_EIGEN_REFINE': '0',
        'OPENMX_MGPU_SOLVER': 'magma' if mode == 'magma' else 'single',
        'OPENMX_MGPU_GEMM': '0' if mode == 'single' else '1',
        'OPENMX_GPU_WORKERS': '0' if mode == 'single' else '1',
        'OPENMX_MGPU_NUM_GPUS': str(args.gpus), 'OPENMX_MGPU_VERBOSE': '1',
        # The production 2 GiB result cap is below even a 13000^2 complex
        # output. Raise only this cap; actual free-memory admission remains.
        'OPENMX_MGPU_GEMM_RESULT_MB': str(args.gemm_result_mib),
        'OPENMX_MGPU_GEMM_WORKSPACE_MB': '1024', 'OPENMX_MGPU_GEMM_HOST_MB': '1024',
        'OPENMX_BAND_COL_KCACHE_MB': '8192', 'OPENMX_BAND_NONCOL_KCACHE_MB': '8192',
        'OPENMX_BAND_COL_OVERLAP_CACHE_MB': '8192', 'OPENMX_BAND_NONCOL_OVERLAP_CACHE_MB': '8192',
        'OPENMX_CLUSTER_PROFILE': '1', 'OPENMX_BAND_PROFILE': '1',
        'OPENMX_BAND_GPU_MAX_CONCURRENT_K': '1',
    }
    if args.exercise_host and mode == 'hybrid':
        changes['OPENMX_MGPU_MEMORY'] = 'host'
    if args.cells < 5:
        for name in ('OPENMX_CLUSTER_GPU_SWITCH_NUM', 'OPENMX_CLUSTER_NONCOL_GPU_SWITCH_NUM',
                     'OPENMX_BAND_GPU_SWITCH_NUM', 'OPENMX_BAND_NONCOL_GPU_SWITCH_NUM'):
            changes[name] = '1'
    # Respect production memory admission; do not force an oversized GPU path.
    env.update(changes)
    return env, changes


def descendant_rss(root_pid):
    rows = subprocess.check_output(['ps', '-eo', 'pid=,ppid=,rss='], text=True)
    processes = [tuple(map(int, row.split())) for row in rows.splitlines()]
    children = {root_pid}
    while True:
        new = children | {pid for pid, parent, _ in processes if parent in children}
        if new == children:
            break
        children = new
    return sum(rss for pid, _, rss in processes if pid in children) * 1024


def inspect(folder, case, mode, atoms, args):
    log = (folder / 'openmx.log').read_text(errors='replace')
    if not re.search(r'normally finished', log, re.I):
        raise ValueError('OpenMX did not finish normally')
    if re.search(r'falling back to (?:the )?(?:ScaLAPACK|ELPA)|GPU initialization failed', log):
        raise ValueError('CPU eigensolver fallback: excluded from GPU comparison')
    steps = sorted(set(map(int, re.findall(r'\bMD=\s*1\s+SCF=\s*(\d+)', log))))
    if steps != list(range(1, args.scf_iterations + 1)):
        raise ValueError(f'Unexpected SCF iterations: {steps}')
    energy, coordinates, forces, grids = read_output(folder / 'sidia.out')
    if len(coordinates) != atoms or len(forces) != 3 * atoms:
        raise ValueError('Incomplete atoms/forces')
    if not all(math.isfinite(v) for v in [energy] + list(forces)):
        raise ValueError('Nonfinite numerical result')
    components = {key: float(value.replace('D', 'E').replace('d', 'e')) for key, value in
                  re.findall(r'^\s*((?:U\w+|Enpy))\.\s+(\S+)', (folder / 'sidia.out').read_text(), re.M)}
    if not components or not all(math.isfinite(v) for v in components.values()):
        raise ValueError('Missing or nonfinite energy components')
    if mode == 'magma' and 'scf.eigen.lib=gpusolver3:' not in log:
        raise ValueError('The new gpusolver3 input selector was not activated')
    operations = records(log)
    times = {name.strip(): float(maximum) for name, maximum in re.findall(
        r'^\s+([^=\n]+?)\s+=\s+\d+\s+[\d.]+\s+\d+\s+([\d.]+)\s*$', log, re.M)}
    elapsed = times.get('Total Computational Time')
    if elapsed is None or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('Missing application elapsed time')
    eigen = [r for r in operations if r['kind'] == 'solver']
    gemms = [r for r in operations if r['kind'] == 'gemm' and r.get('reason') == 'success']
    n = atoms * 13 * (2 if case.endswith('_nc') else 1)
    if mode == 'magma' and not any(r.get('backend') == 'magma' and int(r['n']) == n for r in eigen):
        raise ValueError('No successful MAGMA solve at the full problem dimension')
    if mode != 'single' and not gemms:
        raise ValueError('No successful cooperative matrix product')
    expected_gpus = args.gpus // 2 if case.startswith('band_') else args.gpus
    if any(int(r.get('gpus', '0')) != expected_gpus for r in gemms):
        raise ValueError('The cooperative product did not use the requested worker count')
    if mode == 'single' and (eigen or gemms):
        raise ValueError('Single mode unexpectedly used a cooperative operation')
    if case.startswith('band_') and mode != 'single':
        q = args.gpus // 2
        for k in range(2):
            expected = ','.join(str(k*q+j) for j in range(q))
            if not any(r.get('devices') == expected and r.get('kpoint') == str(k) for r in gemms):
                raise ValueError(f'Missing Band worker group {expected}')
    return {'energy_hartree': energy, 'energy_components': components,
            'atoms': coordinates, 'forces_hartree_per_bohr': forces,
            'grids': grids, 'scf_steps': steps, 'operations': operations,
            'application_elapsed_seconds': elapsed, 'application_times_seconds': times,
            'phase_profiles': [line for line in log.splitlines() if 'PROF' in line],
            'matrix_order': n}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, default=REPO / 'source/openmx')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--datapath', type=Path, default=REPO / 'DFT_DATA19')
    p.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    p.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    p.add_argument('--gpus', type=int, default=4)
    p.add_argument('--ranks', type=int, default=1)
    p.add_argument('--threads', type=int, default=32)
    p.add_argument('--blas-threads', type=int, default=32)
    p.add_argument('--scf-iterations', type=int, default=1)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--cells', type=int, default=5, help='5 retains all 1000 atoms; other sizes are explicitly recorded')
    p.add_argument('--timeout', type=float, default=1500)
    p.add_argument('--host-limit-gib', type=float, default=210)
    p.add_argument('--gemm-result-mib', type=int, default=12288)
    p.add_argument('--energy-tolerance', type=float, default=1e-7)
    p.add_argument('--force-tolerance', type=float, default=1e-7)
    p.add_argument('--mpirun', default='mpirun')
    p.add_argument('--exercise-host', action='store_true', help='Exercise default gpusolver host routing on a small regression system')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    if min(args.gpus, args.ranks, args.threads, args.blas_threads, args.cells,
           args.repeats, args.scf_iterations, args.gemm_result_mib) < 1:
        p.error('Counts must be positive')
    if not all(math.isfinite(x) and x > 0 for x in
               (args.timeout, args.host_limit_gib, args.energy_tolerance, args.force_tolerance)):
        p.error('Time, memory and numerical limits must be finite and positive')
    if any(c.startswith('band_') for c in args.cases) and any(m != 'single' for m in args.modes) and args.gpus < 4:
        p.error('Two Band k-points require at least four GPUs for cooperation')
    args.output = args.output.resolve()
    args.binary, args.datapath = args.binary.resolve(), args.datapath.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    plans = []
    for case in args.cases:
        template, atoms = silicon_input(case, args)
        (args.output / (case + '.dat')).write_text(keyword(template, 'scf.eigen.lib', 'gpusolver3'))
        for repeat in range(args.repeats):
            for mode in args.modes:
                folder = args.output / case / mode / str(repeat + 1)
                folder.mkdir(parents=True)
                text = keyword(template, 'scf.eigen.lib', 'gpusolver3' if mode == 'magma' else 'gpusolver')
                text = keyword(text, 'scf.mgpu.enable', 'off' if mode == 'single' else 'on')
                (folder / 'input.dat').write_text(text)
                command = [args.mpirun, '-np', str(args.ranks), '--map-by', f'slot:PE={args.threads}',
                           '--bind-to', 'core', str(args.binary), 'input.dat', '-nt', str(args.threads)]
                plans.append({'case': case, 'mode': mode, 'repeat': repeat+1, 'atom_count': atoms,
                              'folder': str(folder), 'command': command, 'controls': controls(mode, args)[1],
                              'input_sha256': digest(folder / 'input.dat')})
    save(args.output / 'plans.json', plans)
    if args.prepare_only:
        print(f'Prepared {len(plans)} runs: {args.output}', flush=True)
        return
    binary_hash = digest(args.binary)
    results = []
    for plan in plans:
        folder = Path(plan['folder'])
        result = dict(plan, binary_sha256=binary_hash, status='failed',
                      allocation={'pbs_job_id': os.environ.get('PBS_JOBID'),
                                  'hostname': socket.gethostname(),
                                  'hip_visible_devices': os.environ.get('HIP_VISIBLE_DEVICES')},
                      started_at_unix_seconds=time.time())
        started, peak = time.monotonic(), 0
        print(f"Running {plan['case']} {plan['mode']} repeat={plan['repeat']}", flush=True)
        try:
            with (folder / 'openmx.log').open('w') as log, (folder / 'memory.jsonl').open('w') as memory:
                process = subprocess.Popen(plan['command'], cwd=folder, env=controls(plan['mode'], args)[0],
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while process.poll() is None:
                        rss = descendant_rss(process.pid)
                        peak = max(peak, rss)
                        memory.write(json.dumps({'seconds': time.monotonic()-started, 'process_tree_rss_bytes': rss})+'\n')
                        memory.flush()
                        if rss > args.host_limit_gib * 2**30:
                            raise ValueError('Host-memory guard stopped this run')
                        diagnostic = (folder / 'openmx.log').read_text(errors='replace')
                        if re.search(r'falling back to (?:the )?(?:ScaLAPACK|ELPA)|GPU initialization failed', diagnostic):
                            raise ValueError('GPU memory/device admission declined; stopped before CPU fallback benchmark')
                        if time.monotonic()-started > args.timeout:
                            raise TimeoutError('Benchmark time limit reached')
                        time.sleep(2)
                finally:
                    if process.poll() is None:
                        stop_job(process)
                result['returncode'] = process.returncode
            if process.returncode:
                raise ValueError(f'MPI/OpenMX exited {process.returncode}')
            result.update(inspect(folder, plan['case'], plan['mode'], plan['atom_count'], args))
            result['status'] = 'completed'
        except (OSError, ValueError, TimeoutError) as error:
            result['error'] = str(error)
        result['elapsed_seconds'] = time.monotonic()-started
        result['finished_at_unix_seconds'] = time.time()
        result['peak_process_tree_rss_bytes'] = peak
        results.append(result)
        save(folder / 'result.json', result)
        save(args.output / 'results.json', results)
        print(json.dumps({k: result[k] for k in ('case', 'mode', 'status', 'elapsed_seconds', 'peak_process_tree_rss_bytes')}), flush=True)
        if result['status'] != 'completed':
            print(result['error'], flush=True)
    comparisons = []
    for case in args.cases:
        baseline = next((r for r in results if r['case'] == case and r['mode'] == 'single'
                         and r['status'] == 'completed'), None)
        if baseline is None:
            continue
        for result in results:
            if result['case'] != case or result['status'] != 'completed' or result is baseline:
                continue
            comparison = {'case': case, 'mode': result['mode'], 'repeat': result['repeat']}
            try:
                comparison.update(compare(baseline, result, args))
            except ValueError as error:
                comparison.update(passed=False, error=str(error))
            comparisons.append(comparison)
    save(args.output / 'comparisons.json', comparisons)
    passed = all(r['status'] == 'completed' for r in results) and all(c['passed'] for c in comparisons)
    save(args.output / 'summary.json', {'planned_runs': len(plans),
         'completed_runs': sum(r['status'] == 'completed' for r in results),
         'comparisons': len(comparisons), 'passed': passed})
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
