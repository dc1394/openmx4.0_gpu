#!/usr/bin/env python3
"""Compare periodic Band runs with equal GPU groups per actual k-point.

The caller supplies a single-node allocation and owns MPI. One collinear
Regular 2x1x1 mesh has one actual k-point after inversion reduction. Noncollinear
Band retains every mesh point, and a 1x1x1 input becomes periodic Cluster, so the
one-k-point suite deliberately contains only collinear off/on cases.
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

from run_mgpu_cluster_regression import (REPO, BACKENDS, CASES, keyword, digest,
                                         save, compare, read_output, stop_job)


def expected_groups(gpus, kpoints):
    width = gpus // kpoints
    if width < 2:
        return []
    return [list(range(k * width, (k + 1) * width)) for k in range(kpoints)]


def make_input(template, case, backend, args):
    mesh = args.kpoints if case == 'noncol_soc' else 2 * args.kpoints
    values = {
        'DATA.PATH': str(args.datapath), 'System.CurrentDirectory': './',
        'System.CurrrentDirectory': './', 'System.Name': 'band_' + case,
        'level.of.stdout': 3 if args.require_refinement else 2, 'level.of.fileout': 1,
        'scf.EigenvalueSolver': 'band', 'scf.eigen.lib': 'gpusolver3' if backend == 'magma' else 'gpusolver',
        'scf.Gpu.Num': args.gpus, 'scf.fp32.enable': args.fp32,
        'scf.maxIter': args.scf_iterations, 'scf.criterion': '1e-20', 'scf.restart': 'off',
        'scf.restart.filename': 'band_' + case, 'MD.Type': 'nomd', 'MD.maxIter': 1,
        'scf.Generation.Kpoint': 'Regular', 'scf.Kgrid': f'{mesh} 1 1',
        'scf.mgpu.enable': 'off' if backend == 'off' else 'on',
        'scf.SpinPolarization': {'col_off': 'off', 'col_on': 'on', 'noncol_soc': 'nc'}[case],
        'scf.SpinOrbit.Coupling': 'on' if case == 'noncol_soc' else 'off',
    }
    text = template
    for key, value in values.items():
        text = keyword(text, key, value)
    pattern = re.compile(r'(^\s*<Atoms\.SpeciesAndCoordinates[^\n]*\n)(.*?)(^\s*Atoms\.SpeciesAndCoordinates>[^\n]*)', re.I | re.M | re.S)
    match = pattern.search(text)
    if not match:
        raise ValueError('No coordinate block in periodic template')
    rows = []
    for line in match[2].splitlines():
        fields = line.split('#', 1)[0].split()
        if not fields:
            continue
        if len(fields) != 7:
            raise ValueError('Periodic template must have seven fields per collinear atom')
        if case != 'col_off':
            # Keep each species charge while exercising a nonzero spin density.
            total = float(fields[5]) + float(fields[6])
            fields[5:7] = [str(total / 2 + 0.1), str(total / 2 - 0.1)]
        if case == 'noncol_soc':
            fields += ['45', '15', '45', '15', '1']
        rows.append('  ' + ' '.join(fields))
    atom_count = int(re.search(r'^\s*Atoms.Number\s+(\d+)', text, re.I | re.M)[1])
    if len(rows) != atom_count:
        raise ValueError('Incomplete periodic atom block')
    text = text[:match.start(2)] + '\n'.join(rows) + '\n' + text[match.end(2):]
    return text, atom_count


def environment(backend, args):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('OPENMX_', 'GEMMUL8_'))}
    controls = {
        'OMP_NUM_THREADS': str(args.threads), 'OPENBLAS_NUM_THREADS': '1',
        'MAGMA_NUM_THREADS': '1', 'OPENMX_EIGEN_REFINE': str(args.refine),
        'OPENMX_BAND_GPU_SWITCH_NUM': '1', 'OPENMX_BAND_NONCOL_GPU_SWITCH_NUM': '1',
        'OPENMX_BAND_GPU_DIAG': '1', 'OPENMX_BAND_PROFILE': '1',
        'OPENMX_MGPU': '0' if backend == 'off' else '1',
        'OPENMX_MGPU_GEMM': '0' if backend == 'off' else '1',
        'OPENMX_MGPU_SOLVER': backend, 'OPENMX_MGPU_NUM_GPUS': str(args.gpus),
        'OPENMX_MGPU_SOLVER_MIN_N': '1', 'OPENMX_MGPU_GEMM_MIN_N': '1',
        'OPENMX_MGPU_VERBOSE': '1',
    }
    if args.warm is not None:
        controls['OPENMX_EIGEN_REFINE_WARM'] = str(args.warm)
        controls['OPENMX_EIGEN_REFINE_WARM_BAND'] = str(args.warm)
    if args.refine:
        controls['OPENMX_EIGEN_REFINE_UNTIL'] = '0'
    if args.max_concurrent_k is not None:
        controls['OPENMX_BAND_GPU_MAX_CONCURRENT_K'] = str(args.max_concurrent_k)
    env.update(controls)
    return env, controls


def records(log):
    result = []
    for line in log.splitlines():
        if '<OpenMX_MGPU>' not in line:
            continue
        kind = ('gemm' if '> gemm ' in line else 'solver' if '> solver selected ' in line
                else 'group' if '> band-group ' in line else None)
        if kind:
            fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
            result.append(dict(fields, kind=kind))
    return result


def check_groups(rows, backend, args):
    operations = [row for row in rows if row['kind'] != 'group']
    groups = expected_groups(args.gpus, args.kpoints)
    if backend == 'off':
        if operations:
            raise ValueError('Disabled Band run recorded a multi-GPU operation')
        return
    if not groups:
        if operations:
            raise ValueError('floor(G/K)<2 must disable cooperation even with explicit on')
        # Input setup may disable the runtime before scopes are entered. The
        # independent plan test checks exact K; inspect_run verifies Band use.
        return
    successful = {'gemm': set(), 'solver': set()}
    cross_group_origin = False
    used_pool = set(device for group in groups for device in group)
    for row in operations:
        if row.get('scope') != 'band' or int(row.get('kpoints', '-1')) != args.kpoints:
            raise ValueError('Operation lacks the requested Band/actual-k-point scope')
        kpoint = int(row.get('kpoint', '-1'))
        if not 0 <= kpoint < args.kpoints:
            raise ValueError('Out-of-range k-point in operation log')
        devices = [int(value) for value in row.get('devices', '').split(',') if value]
        if devices != groups[kpoint]:
            raise ValueError(f'k={kpoint}: expected equal GPU group {groups[kpoint]}, found {devices}')
        owner = int(row.get('owner', '-1'))
        if owner not in used_pool:
            raise ValueError('Dense matrix owner used a GPU outside the truncated pool')
        cross_group_origin |= owner not in devices
        if int(row.get('group_gpus', '-1')) != len(devices) or int(row.get('used_gpus', '-1')) != len(used_pool):
            raise ValueError('Reported group size/used GPU pool disagrees with floor(G/K)')
        if int(row.get('gpus', '0')) != len(devices):
            raise ValueError('Operation did not use every GPU of its group')
        if row['kind'] == 'gemm' and row.get('reason') == 'success' and row.get('type') == 'Z':
            successful['gemm'].add(kpoint)
        if row['kind'] == 'solver' and row.get('type') == 'Z' and (
                backend == 'auto' or row.get('backend') == backend):
            successful['solver'].add(kpoint)
    for kind, seen in successful.items():
        if seen != set(range(args.kpoints)):
            raise ValueError(f'Not every k-point exercised {kind}/{backend}: found {sorted(seen)}')
    if args.ranks == 1 and args.kpoints > 1 and not cross_group_origin:
        raise ValueError('Sequential one-rank test did not exercise an owner outside its worker group')


def inspect_run(plan, args):
    folder = Path(plan['directory'])
    log = (folder / 'openmx.log').read_text(errors='replace')
    if 'The calculation was normally finished.' not in log:
        raise ValueError('OpenMX did not finish normally')
    if "changed to 'Cluster'" in log:
        raise ValueError('Input was converted to Cluster instead of exercising Band')
    band_tag = '<Band_DFT_NonCol>' if plan['case'] == 'noncol_soc' else '<Band_DFT_Col>'
    capacity_tag = '<OpenMX_MGPU> Band NC n=' if plan['case'] == 'noncol_soc' else '<OpenMX_MGPU> Band collinear n='
    if band_tag not in log and capacity_tag not in log:
        raise ValueError('No Band GPU diagnostic confirms the requested solver path')
    steps = sorted(set(map(int, re.findall(r'\bMD=\s*1\s+SCF=\s*(\d+)', log))))
    if steps != list(range(1, args.scf_iterations + 1)):
        raise ValueError(f'Expected exactly {args.scf_iterations} SCF steps, found {steps}')
    refinement = re.findall(r'<Band_DFT_(?:Col|NonCol)>[^\n]*\brefinement steps?\b[^\n]*', log)
    if args.require_refinement and not refinement:
        raise ValueError('No Band refinement execution diagnostic; setting REFINE alone is insufficient')
    warm_attempts = re.findall(r'<(?:Band_DFT_(?:Col|NonCol)|eigen_refine_gpu)>[^\n]*warm start[^\n]*', log)
    output = folder / ('band_' + plan['case'] + '.out')
    energy, atoms, forces, grids = read_output(output)
    if len(atoms) != plan['atoms'] or len(forces) != 3 * plan['atoms']:
        raise ValueError('Incomplete atom/force output')
    components = {key: float(value.replace('D', 'E').replace('d', 'e')) for key, value in
                  re.findall(r'^\s*((?:U\w+|Enpy))\.\s+(\S+)', output.read_text(), re.M)}
    if not components or not all(math.isfinite(v) for v in components.values()):
        raise ValueError('Missing or nonfinite energy components')
    observed = records(log)
    check_groups(observed, plan['backend'], args)
    return {'energy_hartree': energy, 'energy_components': components,
            'atoms': atoms, 'forces_hartree_per_bohr': forces, 'grids': grids,
            'scf_steps': steps, 'backend_records': observed,
            'refinement_records': refinement, 'warm_start_records': warm_attempts}


def run_one(plan, args):
    folder, started = Path(plan['directory']), time.monotonic()
    result = dict(plan, status='failed')
    try:
        with (folder / 'openmx.log').open('w') as log:
            process = subprocess.Popen(plan['command'], cwd=folder, env=environment(plan['backend'], args)[0],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                stop_job(process)
                raise
        result['returncode'] = code
        if code:
            raise ValueError(f'MPI/OpenMX exited {code}')
        result.update(inspect_run(plan, args))
        result['status'] = 'completed'
    except (OSError, ValueError, TypeError, AttributeError, IndexError, subprocess.TimeoutExpired) as error:
        result['error'] = str(error)
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new private directory; must not exist')
    parser.add_argument('--template', type=Path, default=REPO / 'work/large_example/B2C62_Band.dat')
    parser.add_argument('--datapath', type=Path, default=REPO / 'DFT_DATA19')
    parser.add_argument('--kpoints', type=int, default=1, help='actual k-points after collinear inversion reduction')
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--ranks', type=int, default=2)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--scf-iterations', type=int, default=3)
    parser.add_argument('--fp32', choices=('off', 'on'), default='off')
    parser.add_argument('--refine', type=int, default=0, help='OPENMX_EIGEN_REFINE iterations')
    parser.add_argument('--warm', type=int, choices=(0, 1), help='explicit Band/eigensolver warm-start setting')
    parser.add_argument('--require-refinement', action='store_true', help='fail unless an actual Band refinement diagnostic appears')
    parser.add_argument('--max-concurrent-k', type=int, help='force a Band turn cap to exercise staged refinement')
    parser.add_argument('--mpirun', default='mpirun')
    parser.add_argument('--mpi-arg', action='append', default=[])
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--energy-tolerance', type=float, default=1e-7)
    parser.add_argument('--force-tolerance', type=float, default=1e-7)
    parser.add_argument('--cases', choices=CASES, nargs='+')
    parser.add_argument('--backends', choices=BACKENDS, nargs='+', default=list(BACKENDS))
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.kpoints < 1 or not 2 <= args.gpus <= 8 or args.ranks < 1 or args.threads < 1:
        parser.error('Require positive kpoints/ranks/threads and 2..8 GPUs')
    if args.scf_iterations < 1 or args.refine < 0 or (args.max_concurrent_k is not None and args.max_concurrent_k < 1):
        parser.error('SCF iterations/turn cap must be positive; refinement count must be nonnegative')
    if args.require_refinement and (args.fp32 != 'on' or args.refine < 1 or args.scf_iterations < 3):
        parser.error('--require-refinement needs --fp32 on, positive --refine, and at least three SCF steps')
    if not all(math.isfinite(v) and v > 0 for v in (args.timeout, args.energy_tolerance, args.force_tolerance)):
        parser.error('Timeout/tolerances must be finite and positive')
    args.cases = list(dict.fromkeys(args.cases or (CASES[:2] if args.kpoints == 1 else CASES)))
    args.backends = list(dict.fromkeys(args.backends))
    if args.kpoints == 1 and 'noncol_soc' in args.cases:
        parser.error('Noncollinear 1x1x1 becomes periodic Cluster; use --kpoints 2 --gpus 4 for NC Band')
    for name in ('baseline', 'binary', 'output', 'template', 'datapath'):
        setattr(args, name, getattr(args, name).resolve())
    for binary in (args.baseline, args.binary):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            parser.error(f'Not executable: {binary}')
    if not args.datapath.is_dir() or re.search(r'\s', str(args.datapath)):
        parser.error('DATA.PATH must be an existing directory without whitespace')
    template, template_hash = args.template.read_text(), digest(args.template)
    prepared = {(case, mode): make_input(template, case, mode, args)
                for case in args.cases for mode in ('off', *args.backends)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'binaries').mkdir()
    provenance = {}
    for name, source in (('baseline', args.baseline), ('candidate', args.binary)):
        snapshot = args.output / 'binaries' / name
        shutil.copy2(source, snapshot)
        provenance[name] = {'source': str(source), 'snapshot': str(snapshot), 'sha256': digest(snapshot)}
    plans = []
    for case in args.cases:
        for kind, mode in [('baseline', 'off')] + [('candidate', mode) for mode in args.backends]:
            directory = args.output / case / ('baseline' if kind == 'baseline' else mode)
            directory.mkdir(parents=True)
            text, atoms = prepared[case, mode]
            (directory / 'input.dat').write_text(text)
            command = [args.mpirun, '-np', str(args.ranks), '--bind-to', 'core', *args.mpi_arg,
                       provenance[kind]['snapshot'], 'input.dat', '-nt', str(args.threads)]
            plans.append({'case': case, 'kind': kind, 'backend': mode, 'atoms': atoms,
                          'directory': str(directory), 'command': command,
                          'controls': environment(mode, args)[1],
                          'input_sha256': digest(directory / 'input.dat')})
    manifest = {'template': str(args.template), 'template_sha256': template_hash, 'binaries': provenance,
                'datapath': str(args.datapath), 'ranks': args.ranks, 'gpus': args.gpus,
                'actual_kpoints': args.kpoints, 'expected_groups': expected_groups(args.gpus, args.kpoints),
                'scf_iterations': args.scf_iterations, 'fp32': args.fp32, 'refine': args.refine,
                'warm': args.warm, 'require_refinement': args.require_refinement,
                'max_concurrent_k': args.max_concurrent_k,
                'energy_tolerance_hartree': args.energy_tolerance,
                'force_tolerance_hartree_per_bohr': args.force_tolerance,
                'prepared_only': args.prepare_only, 'plans': plans}
    save(args.output / 'command.json', manifest)
    if args.prepare_only:
        print(f'Prepared {len(plans)} private Band runs: {args.output}/command.json')
        return 0
    results, comparisons, baselines = [], [], {}
    for plan in plans:
        print(f"Running Band {plan['case']} {plan['kind']}/{plan['backend']}", flush=True)
        result = run_one(plan, args)
        if result['kind'] == 'baseline':
            baselines[result['case']] = result
        else:
            comparison = {'case': result['case'], 'backend': result['backend'], 'passed': False}
            baseline = baselines.get(result['case'], {})
            if baseline.get('status') == result['status'] == 'completed':
                try:
                    comparison.update(compare(baseline, result, args))
                except ValueError as error:
                    comparison['error'] = str(error)
            else:
                comparison['error'] = result.get('error', baseline.get('error', 'Missing baseline'))
            comparisons.append(comparison)
            print(json.dumps(comparison, allow_nan=False), flush=True)
        results.append(result)
        save(args.output / 'results.json', results)
        save(args.output / 'comparisons.json', comparisons)
    completed = len(results) == len(plans) and all(row['status'] == 'completed' for row in results)
    unchanged = digest(args.template) == template_hash
    passed = completed and unchanged and bool(comparisons) and all(row['passed'] for row in comparisons)
    save(args.output / 'summary.json', {'passed': passed, 'completed': completed, 'planned_runs': len(plans),
         'completed_runs': len(results), 'comparisons': len(comparisons), 'template_unchanged': unchanged})
    print(f"{'PASS' if passed else 'FAIL'}: {len(comparisons)} Band comparisons; {args.output}")
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
