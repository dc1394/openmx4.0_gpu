#!/usr/bin/env python3
"""Compare private Pt63 Cluster runs against a supplied pre-MGPU binary.

Run inside an allocated single node. The caller owns GPU visibility, MPI setup
and device visibility. Nothing is submitted to a scheduler by this script.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
from compare_runtest_l3 import read_output
from run_runtest_l3 import stop_job

CASES = ('col_off', 'col_on', 'noncol_soc')
BACKENDS = ('off', 'magma', 'auto')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def keyword(text, key, value):
    pattern = re.compile(r'^\s*' + re.escape(key) + r'\s+[^\n]*$', re.I | re.M)
    replacement = f'{key:<32} {value}'
    if pattern.search(text):
        return pattern.sub(lambda _: replacement, text)
    return text.rstrip() + '\n' + replacement + '\n'


def make_input(template, case, backend, datapath, gpus):
    text = template
    options = {
        'DATA.PATH': str(datapath), 'System.CurrentDirectory': './',
        'System.CurrrentDirectory': './',
        'System.Name': 'mgpu_' + case, 'level.of.stdout': 2,
        'level.of.fileout': 1, 'scf.EigenvalueSolver': 'cluster',
        'scf.eigen.lib': 'gpusolver3' if backend == 'magma' else 'gpusolver', 'scf.Gpu.Num': gpus,
        'scf.fp32.enable': 'off', 'scf.maxIter': 3,
        'scf.criterion': '1e-20', 'scf.restart': 'off',
        'MD.Type': 'nomd', 'MD.maxIter': 1,
        'scf.mgpu.enable': 'off' if backend == 'off' else 'on',
        'scf.SpinPolarization': {'col_off': 'off', 'col_on': 'on', 'noncol_soc': 'nc'}[case],
        'scf.SpinOrbit.Coupling': 'on' if case == 'noncol_soc' else 'off',
    }
    for key, value in options.items():
        text = keyword(text, key, value)
    pattern = re.compile(r'(^\s*<Atoms\.SpeciesAndCoordinates\s*\n)(.*?)(^\s*Atoms\.SpeciesAndCoordinates>[^\n]*)', re.I | re.M | re.S)
    match = pattern.search(text)
    if not match:
        raise ValueError('No atom coordinate block in template')
    rows = []
    for line in match[2].splitlines():
        fields = line.split('#', 1)[0].split()
        if not fields:
            continue
        if len(fields) != 7:
            raise ValueError('Expected seven fields in each collinear Pt63 atom row')
        if case in ('col_on', 'noncol_soc'):
            # Preserve the 16-electron total while starting with nonzero spin.
            fields[5:7] = ['8.5', '7.5']
        if case == 'noncol_soc':
            fields += ['45', '15', '45', '15', '1']
        rows.append('  ' + ' '.join(fields))
    if len(rows) != 63:
        raise ValueError(f'Expected 63 atoms, found {len(rows)}')
    return text[:match.start(2)] + '\n'.join(rows) + '\n' + text[match.end(2):]


def environment(backend, args):
    # Keep site MPI/HIP environment, remove inherited OpenMX policy knobs.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('OPENMX_', 'GEMMUL8_'))}
    controls = {
        'OMP_NUM_THREADS': str(args.threads), 'OPENBLAS_NUM_THREADS': '1',
        'MAGMA_NUM_THREADS': '1', 'OPENMX_EIGEN_REFINE': '0',
        'OPENMX_CLUSTER_GPU_SWITCH_NUM': '1',
        'OPENMX_CLUSTER_NONCOL_GPU_SWITCH_NUM': '1',
        'OPENMX_MGPU': '0' if backend == 'off' else '1',
        'OPENMX_MGPU_GEMM': '0' if backend == 'off' else '1',
        'OPENMX_MGPU_SOLVER': backend, 'OPENMX_MGPU_NUM_GPUS': str(args.gpus),
        'OPENMX_MGPU_SOLVER_MIN_N': '1', 'OPENMX_MGPU_GEMM_MIN_N': '1',
        'OPENMX_MGPU_VERBOSE': '1',
    }
    env.update(controls)
    return env, controls


def backend_records(log):
    records = []
    for line in log.splitlines():
        if '<OpenMX_MGPU>' not in line:
            continue
        kind = 'gemm' if '> gemm ' in line else 'solver' if '> solver selected ' in line else None
        if kind:
            record = dict(re.findall(r'(\w+)=([^\s]+)', line))
            record['kind'] = kind
            records.append(record)
    return records


def inspect_run(folder, case, backend, gpus):
    text = (folder / 'openmx.log').read_text(errors='replace')
    steps = sorted(set(map(int, re.findall(r'\bMD=\s*1\s+SCF=\s*(\d+)', text))))
    if steps != [1, 2, 3]:
        raise ValueError(f'Expected exactly SCF steps 1,2,3, found {steps}')
    output = folder / ('mgpu_' + case + '.out')
    energy, atoms, forces, grids = read_output(output)
    if len(atoms) != 63 or len(forces) != 189:
        raise ValueError('Expected all 63 atoms and 189 force components')
    components = {key: float(value.replace('D', 'E').replace('d', 'e'))
                  for key, value in re.findall(r'^\s*((?:U\w+|Enpy))\.\s+(\S+)', output.read_text(), re.M)}
    if not components or not all(math.isfinite(v) for v in components.values()):
        raise ValueError('Missing or nonfinite energy components')
    records = backend_records(text)
    expected_type = 'Z' if case == 'noncol_soc' else 'D'
    gemms = [row for row in records if row['kind'] == 'gemm' and
             row.get('reason') == 'success' and row.get('type') == expected_type and
             int(row.get('gpus', '0')) >= gpus]
    selected = [row for row in records if row['kind'] == 'solver' and
                row.get('type') == expected_type and int(row.get('gpus', '0')) >= gpus]
    if backend == 'off' and records:
        raise ValueError('Disabled run unexpectedly recorded a multi-GPU operation')
    if backend != 'off' and not gemms:
        raise ValueError(f'No successful {expected_type} GEMM using at least {gpus} GPUs')
    if backend == 'magma' and not any(row.get('backend') == backend for row in selected):
        raise ValueError(f'No selected {backend} {expected_type} eigensolver using {gpus} GPUs')
    if backend == 'auto' and not selected:
        raise ValueError('Auto solver did not record its selected backend')
    return {'energy_hartree': energy, 'energy_components': components,
            'atoms': atoms, 'forces_hartree_per_bohr': forces, 'grids': grids,
            'scf_steps': steps, 'backend_records': records}


def compare(baseline, candidate, args):
    if baseline['atoms'] != candidate['atoms'] or baseline['grids'] != candidate['grids']:
        raise ValueError('Atom positions or integration grids differ from baseline')
    de = abs(baseline['energy_hartree'] - candidate['energy_hartree'])
    diffs = [abs(x-y) for x, y in zip(baseline['forces_hartree_per_bohr'], candidate['forces_hartree_per_bohr'])]
    be, ce = baseline['energy_components'], candidate['energy_components']
    if be.keys() != ce.keys():
        raise ValueError('Energy component names differ from baseline')
    return {'abs_energy_difference_hartree': de,
            'max_abs_force_difference_hartree_per_bohr': max(diffs),
            'rms_force_difference_hartree_per_bohr': math.sqrt(sum(v*v for v in diffs)/len(diffs)),
            'energy_component_differences_hartree': {key: abs(be[key]-ce[key]) for key in be},
            'passed': de <= args.energy_tolerance and max(diffs) <= args.force_tolerance}


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run_one(plan, args):
    folder = Path(plan['directory'])
    env, _ = environment(plan['backend'], args)
    started = time.monotonic()
    record = dict(plan, status='failed')
    try:
        with (folder / 'openmx.log').open('w') as log:
            proc = subprocess.Popen(plan['command'], cwd=folder, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                stop_job(proc)
                raise
        record['returncode'] = code
        if code:
            raise ValueError(f'MPI/OpenMX exit status {code}')
        record.update(inspect_run(folder, plan['case'], plan['backend'], args.gpus))
        record['status'] = 'completed'
    except (OSError, ValueError, AttributeError, TypeError, IndexError, subprocess.TimeoutExpired) as error:
        record['error'] = str(error)
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new private directory; must not exist')
    parser.add_argument('--template', type=Path, default=REPO / 'work/large_example/Pt63.dat')
    parser.add_argument('--datapath', type=Path, default=REPO / 'DFT_DATA19')
    parser.add_argument('--ranks', type=int, default=2)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--mpirun', default='mpirun')
    parser.add_argument('--mpi-arg', action='append', default=[], help='additional launcher argument; use --mpi-arg=VALUE')
    parser.add_argument('--timeout', type=float, default=600, help='seconds per MPI run')
    parser.add_argument('--energy-tolerance', type=float, default=1e-7, help='Hartree')
    parser.add_argument('--force-tolerance', type=float, default=1e-7, help='Hartree/bohr per component')
    parser.add_argument('--cases', choices=CASES, nargs='+', default=list(CASES))
    parser.add_argument('--backends', choices=BACKENDS, nargs='+', default=list(BACKENDS))
    parser.add_argument('--prepare-only', action='store_true', help='snapshot binaries and write inputs/manifest without launching MPI')
    args = parser.parse_args()
    if args.ranks < 1 or args.gpus < 2 or args.gpus > 8 or args.threads < 1:
        parser.error('Require positive ranks/threads and 2..8 GPUs')
    if not all(math.isfinite(v) and v > 0 for v in (args.timeout, args.energy_tolerance, args.force_tolerance)):
        parser.error('Timeout and tolerances must be finite and positive')
    args.output, args.datapath = args.output.resolve(), args.datapath.resolve()
    args.baseline, args.binary, args.template = (path.resolve() for path in (args.baseline, args.binary, args.template))
    for path in (args.baseline, args.binary):
        if not path.is_file() or not os.access(path, os.X_OK):
            parser.error(f'Not an executable file: {path}')
    if not (args.datapath / 'PAO/Pt7.0.pao').is_file() or not (args.datapath / 'VPS/Pt_PBE19.vps').is_file():
        parser.error(f'Pt basis/pseudopotential missing from {args.datapath}')
    if re.search(r'\s', str(args.datapath)):
        parser.error('OpenMX DATAPATH must not contain whitespace')
    template = args.template.read_text()
    template_digest = digest(args.template)
    prepared = {(case, backend): make_input(template, case, backend, args.datapath, args.gpus)
                for case in dict.fromkeys(args.cases) for backend in ('off', *dict.fromkeys(args.backends))}
    args.output.mkdir(parents=True, exist_ok=False)
    binary_dir = args.output / 'binaries'
    binary_dir.mkdir()
    provenance = {}
    for name, source in (('baseline', args.baseline), ('candidate', args.binary)):
        target = binary_dir / name
        shutil.copy2(source, target)
        provenance[name] = {'source': str(source), 'snapshot': str(target), 'sha256': digest(target)}
    plans = []
    for case in dict.fromkeys(args.cases):
        runs = [('baseline', 'off')] + [('candidate', mode) for mode in dict.fromkeys(args.backends)]
        for kind, backend in runs:
            folder = args.output / case / (kind if kind == 'baseline' else backend)
            folder.mkdir(parents=True)
            (folder / 'input.dat').write_text(prepared[case, backend])
            _, controls = environment(backend, args)
            command = [args.mpirun, '-np', str(args.ranks), '--bind-to', 'core', *args.mpi_arg,
                       provenance[kind]['snapshot'], 'input.dat', '-nt', str(args.threads)]
            plans.append({'case': case, 'kind': kind, 'backend': backend, 'directory': str(folder),
                          'command': command, 'controls': controls,
                          'input_sha256': digest(folder / 'input.dat')})
    manifest = {'template': str(args.template), 'template_sha256': template_digest,
                'binaries': provenance, 'datapath': str(args.datapath), 'ranks': args.ranks,
                'gpus': args.gpus, 'energy_tolerance_hartree': args.energy_tolerance,
                'force_tolerance_hartree_per_bohr': args.force_tolerance,
                'prepared_only': args.prepare_only, 'plans': plans}
    save(args.output / 'command.json', manifest)
    if args.prepare_only:
        print(f'Prepared {len(plans)} private runs: {args.output}/command.json')
        return 0
    results, comparisons, baselines = [], [], {}
    for plan in plans:
        print(f"Running {plan['case']} {plan['kind']}/{plan['backend']}", flush=True)
        result = run_one(plan, args)
        if result['kind'] == 'baseline':
            baselines[result['case']] = result
        else:
            comparison = {'case': result['case'], 'backend': result['backend'], 'passed': False}
            base = baselines.get(result['case'], {})
            if base.get('status') == result['status'] == 'completed':
                try:
                    comparison.update(compare(base, result, args))
                except ValueError as error:
                    comparison['error'] = str(error)
            else:
                comparison['error'] = result.get('error', base.get('error', 'Missing baseline'))
            comparisons.append(comparison)
            print(json.dumps(comparison, allow_nan=False), flush=True)
        results.append(result)
        save(args.output / 'results.json', results)
        save(args.output / 'comparisons.json', comparisons)
    complete = len(results) == len(plans) and all(row['status'] == 'completed' for row in results)
    passed = complete and bool(comparisons) and all(row['passed'] for row in comparisons)
    template_unchanged = digest(args.template) == template_digest
    passed &= template_unchanged
    save(args.output / 'summary.json', {'passed': passed, 'completed': complete,
         'planned_runs': len(plans), 'completed_runs': len(results), 'comparisons': len(comparisons),
         'template_unchanged': template_unchanged})
    print(f"{'PASS' if passed else 'FAIL'}: {len(comparisons)} comparisons; results in {args.output}")
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
