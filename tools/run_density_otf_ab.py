#!/usr/bin/env python3
r"""Eight-step native CUDA / prior OpenACC FP64 density and optional original-default comparison.

Run inside an existing GPU allocation and MPS session.  This script does not
start MPS or submit a job.  Inputs are copies; original large examples remain
unchanged.  The results are a fixed-step regression, not converged observables.
OPENMX_SETDEN_FP64_SPLIT_SPIN=0 deliberately isolates the initial CUDA density
kernel from the later automatic spin-splitting optimization for orbital counts
above 32.  The three bundled inputs use at most 32 orbitals per atom, so this
setting also matches the final automatic policy for these cases.

Example:
  python3 tools/run_density_otf_ab.py --binary source/openmx \
      --baseline /path/to/original/openmx --output work/density_otf_comparison

--prepare-only creates reusable inputs under OUTPUT/inputs.  A later run can
use that prepared directory; completed or partial run outputs are never
replaced.  --collect-only reanalyzes existing results without launching MPI.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
from compare_gpu_suites import read_output
from compare_gpu_phase_times import ROW
from run_gpu_suites import stop_job

CASES = ['Pt63', 'Mn12_148_F', 'GGFF']
COMMON = {
    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'BLIS_NUM_THREADS': '1',
    'OPENMX_SETHAM_GPU_RESIDENT': '0', 'OPENMX_DENSITY_GRID_GPU_LOCAL': '1',
    'OPENMX_SETHAM_ONTHEFLY': '1', 'OPENMX_GRID_PRECISION': 'fp64',
    'OPENMX_ORBS_EVAL_PRECISION': 'fp64', 'OPENMX_GRID_FP32': '0',
    'OPENMX_DENSITY_GRID_GPU': '1', 'OPENMX_ORBS_GRID_GPU': '1',
    'OPENMX_BAND_PROFILE': '1', 'OPENMX_CLUSTER_PROFILE': '1',
    'OPENMX_SETDEN_FP64_SPLIT_SPIN': '0',
}
PROFILE = re.compile(r'SETHOTFDENPROF\s+id=(\d+)\s+batches=(\d+)\s+workspace_mib=([\d.]+)'
                     r'\s+mode=(\d+)\s+pack=([\d.eE+-]+)\s+eval=([\d.eE+-]+)'
                     r'\s+device=([\d.eE+-]+)\s+total=([\d.eE+-]+)')
STEP = re.compile(r'\*+\s+MD=\s*(\d+)\s+SCF=\s*(\d+)\s+\*+')


def prepare(output):
    folder = output / 'inputs'
    folder.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        text = (REPO / 'work/large_example' / (case + '.dat')).read_text()
        text, count = re.subn(r'^\s*scf\.maxIter\s+.*$', 'scf.maxIter                 8', text, flags=re.M | re.I)
        if count != 1:
            raise ValueError('Expected exactly one scf.maxIter in ' + case)
        text, count = re.subn(r'^\s*DATA\.PATH\s+.*$', 'DATA.PATH                  ' + str(REPO / 'DFT_DATA19'),
                             text, flags=re.M | re.I)
        if count == 0:
            text = 'DATA.PATH                  ' + str(REPO / 'DFT_DATA19') + '\n' + text
        elif count != 1:
            raise ValueError('Duplicate DATA.PATH in ' + case)
        text, count = re.subn(r'^\s*level\.of\.stdout\s+.*$', 'level.of.stdout             2', text, flags=re.M | re.I)
        if count != 1:
            raise ValueError('Expected exactly one level.of.stdout in ' + case)
        target = folder / (case + '.dat')
        if target.exists():
            if target.read_text() != text:
                raise ValueError('Refusing to replace a changed input: ' + str(target))
        else:
            target.write_text(text)
    return folder


def analyze(folder, case, ranks):
    text = (folder / 'run.log').read_text(errors='replace')
    output = folder / (case + '.out')
    energy, atoms, forces, grids = read_output(output)
    outtext = output.read_text(errors='replace')
    phases = {row[0].strip(): float(row[4]) for row in ROW.findall(outtext.rsplit('Computational Time (second)', 1)[-1])}
    steps = sorted(set((int(md), int(scf)) for md, scf in STEP.findall(text)))
    announced_precision = sorted(set(re.findall(
        r'GPU on-the-fly orbital tiles:[^\n]*, (FP64|double-float) evaluation', text)))
    profiles, counts = [], {}
    for found in PROFILE.finditer(text):
        rank, batches, workspace, mode, pack, evaluate, device, total = found.groups()
        rank = int(rank)
        counts[rank] = counts.get(rank, 0) + 1
        profiles.append(dict(rank=rank, call_index=counts[rank], batches=int(batches), workspace_mib=float(workspace),
                             mode=int(mode), pack_s=float(pack), eval_s=float(evaluate), device_s=float(device),
                             total_s=float(total)))
    per_call = []
    for call in range(1, max(counts.values(), default=0) + 1):
        rows = [row for row in profiles if row['call_index'] == call]
        per_call.append(dict(call_index=call, rank_count=len(rows),
                             **{key: max(row[key] for row in rows) for key in ('pack_s', 'eval_s', 'device_s', 'total_s')}))
    problems = []
    if 'The calculation was normally finished.' not in text:
        problems.append('No normal completion marker')
    if steps != [(1, step) for step in range(1, 9)]:
        problems.append('Expected exactly MD1 / SCF1..8')
    if set(counts) != set(range(ranks)):
        problems.append('OTF profile missing one or more MPI ranks')
    if any(row['mode'] != 0 for row in profiles):
        problems.append('Density profile used non-FP64 mode')
    if len(set(counts.values())) != 1 or any(count < 8 for count in counts.values()):
        problems.append('Expected at least eight equally counted OTF calls on every rank')
    if 'Set_Density_Grid' not in phases:
        problems.append('Missing Set_Density_Grid timing summary')
    return dict(energy_Ha=energy, atoms=atoms, forces_Ha_per_bohr=forces, integration_grid=grids,
                scf_steps=steps, phase_max_seconds=phases, profile_calls_by_rank=counts,
                orbital_precision_announcements=announced_precision,
                profile_rows=profiles, profile_per_call_rank_max=per_call,
                profile_rank_max_sums={key: sum(row[key] for row in per_call)
                                       for key in ('pack_s', 'eval_s', 'device_s', 'total_s')},
                problems=problems)


def collect(output, ranks, cases, include_baseline=False):
    records, comparisons = {}, []
    variants = ('baseline_auto', 'openacc', 'cuda') if include_baseline else ('openacc', 'cuda')
    for case in cases:
        records[case] = {}
        for variant in variants:
            folder = output / case / variant
            record = json.loads((folder / 'run.json').read_text()) if (folder / 'run.json').exists() else {'status': 'missing'}
            if record.get('status') == 'completed':
                try:
                    record.update(analyze(folder, case, ranks))
                    expected_precision = 'double-float' if variant == 'baseline_auto' else 'FP64'
                    record['orbital_precision_expected'] = expected_precision
                    if record['orbital_precision_announcements'] and record['orbital_precision_announcements'] != [expected_precision]:
                        record['problems'].append('Unexpected runtime orbital precision announcement')
                    if record['problems']:
                        record['status'] = 'invalid'
                except Exception as exc:
                    record.update(status='invalid', error=str(exc))
            records[case][variant] = record
        for before_variant in (('openacc', 'baseline_auto') if include_baseline else ('openacc',)):
            before, after = records[case][before_variant], records[case]['cuda']
            if any(record.get('status') != 'completed' for record in (before, after)):
                comparisons.append(dict(case=case, before_variant=before_variant, after_variant='cuda', passed=False, error='Incomplete or invalid paired run'))
                continue
            same = before['atoms'] == after['atoms'] and before['integration_grid'] == after['integration_grid']
            de = abs(before['energy_Ha'] - after['energy_Ha'])
            df = max(abs(a-b) for a, b in zip(before['forces_Ha_per_bohr'], after['forces_Ha_per_bohr']))
            profile_match = before['profile_calls_by_rank'] == after['profile_calls_by_rank']
            comparisons.append(dict(case=case, before_variant=before_variant, after_variant='cuda', passed=same and profile_match and de <= 1e-8 and df <= 1e-8,
                                    energy_abs_delta_Ha=de, force_max_abs_delta_Ha_per_bohr=df,
                                    scf_steps_before=len(before['scf_steps']), scf_steps_after=len(after['scf_steps']),
                                    density_phase_before_s=before['phase_max_seconds']['Set_Density_Grid'],
                                    density_phase_after_s=after['phase_max_seconds']['Set_Density_Grid'],
                                    profile_calls_match=profile_match,
                                    eval_rank_max_sum_before_s=before['profile_rank_max_sums']['eval_s'],
                                    eval_rank_max_sum_after_s=after['profile_rank_max_sums']['eval_s'],
                                    device_rank_max_sum_before_s=before['profile_rank_max_sums']['device_s'],
                                    device_rank_max_sum_after_s=after['profile_rank_max_sums']['device_s']))
    result = dict(passed=all(row['passed'] for row in comparisons), comparisons=comparisons, runs=records,
                  note='Fixed eight-step regression; no convergence claim. Profile call_index is per-rank chronological '
                       'order, avoiding MPI log interleaving. device_s includes the quadrature kernel, output copy, '
                       'synchronization and arena release. Per-call rank maxima may belong to different ranks; their '
                       'sum is not an end-to-end wall time. Existing profile fields are rounded to milliseconds. '
                       'cuda leaves orbital precision unset to test the actual A100 default; openacc forces FP64. '
                       'Optional baseline_auto uses the original binary with orbital precision unset (original DF default). '
                       'Runtime orbital precision announcements are checked when available; the density-only OTF path '
                       'does not announce its orbital precision, so an empty announcement list is recorded explicitly.')
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    with (output / 'profile_per_call.csv').open('w', newline='') as handle:
        columns = ['case', 'variant', 'call_index', 'rank_count', 'pack_s', 'eval_s', 'device_s', 'total_s']
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for case, variants in records.items():
            for variant, record in variants.items():
                for row in record.get('profile_per_call_rank_max', []):
                    writer.writerow(dict(case=case, variant=variant, **row))
    lines = [result['note'], '', 'Case | Comparison | SCF steps | abs delta E (Ha) | max abs delta F (Ha/bohr) | Density phase before/cuda (s) | Device profile sums before/cuda (s) | Status']
    for row in comparisons:
        if 'error' in row:
            lines.append(row['case'] + ' | ' + row['before_variant'] + '/cuda | ' + row['error'])
        else:
            lines.append('{case} | {before_variant}/cuda | {scf_steps_before}/{scf_steps_after} | {energy_abs_delta_Ha:.3e} | '
                         '{force_max_abs_delta_Ha_per_bohr:.3e} | {density_phase_before_s:.3f}/{density_phase_after_s:.3f} | '
                         '{device_rank_max_sum_before_s:.3f}/{device_rank_max_sum_after_s:.3f} | '.format(**row) +
                         ('PASS' if row['passed'] else 'FAIL'))
    report = '\n'.join(lines) + '\n'
    (output / 'comparison.txt').write_text(report)
    print(report, flush=True)
    return result['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--binary', type=Path, default=REPO / 'source/openmx')
    parser.add_argument('--baseline', type=Path, help='optional original binary for original-auto vs new-auto validation')
    parser.add_argument('--output', type=Path, default=REPO / 'work/density_otf_ab')
    parser.add_argument('--mpirun', default='mpirun')
    parser.add_argument('--ranks', type=int, help='MPI ranks (default: 8, or saved metadata for --collect-only)')
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--cases', nargs='+', choices=CASES, help='default: all three cases, or saved cases for --collect-only')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    if (args.ranks is not None and args.ranks <= 0) or args.timeout <= 0:
        parser.error('ranks and timeout must be positive')
    output = args.output.resolve()
    if args.collect_only:
        metadata_path = output / 'command.json'
        if not metadata_path.is_file():
            parser.error('Missing saved run metadata: ' + str(metadata_path))
        metadata = json.loads(metadata_path.read_text())
        include_baseline = args.baseline is not None or 'baseline_auto' in metadata.get('variants', {})
        ranks = args.ranks if args.ranks is not None else metadata.get('ranks', 8)
        cases = args.cases or metadata.get('cases', CASES)
        return 0 if collect(output, ranks, cases, include_baseline) else 1
    args.ranks = args.ranks if args.ranks is not None else 8
    args.cases = args.cases or CASES
    if args.prepare_only:
        try:
            inputs = prepare(output)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print('Prepared eight-step inputs in ' + str(inputs))
        return 0
    # --prepare-only may have created OUTPUT/inputs.  Reuse only that clean
    # preparation; preserve partial runs, binaries, logs and collected data.
    if output.exists() and (not output.is_dir() or any(child.name != 'inputs' for child in output.iterdir())):
        parser.error('Output already contains run data; choose a new --output or use --collect-only: ' + str(output))
    for source_binary in (args.binary, args.baseline):
        if source_binary is not None and not source_binary.is_file():
            parser.error('Binary does not exist: ' + str(source_binary))
    output.mkdir(parents=True, exist_ok=True)
    try:
        inputs = prepare(output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    binary = output / 'openmx'
    shutil.copy2(args.binary.resolve(), binary)
    binaries = {'candidate': binary}
    if args.baseline is not None:
        baseline_binary = output / 'openmx-baseline'
        shutil.copy2(args.baseline.resolve(), baseline_binary)
        binaries['baseline'] = baseline_binary
    common_env = os.environ.copy()
    common_env.update(COMMON)
    variant_specs = {
        'openacc': dict(binary='candidate', env={'OPENMX_SETDEN_FP64_CUDA': '0', 'OPENMX_ORBS_EVAL_PRECISION': 'fp64'}),
        'cuda': dict(binary='candidate', env={'OPENMX_SETDEN_FP64_CUDA': '1', 'OPENMX_ORBS_EVAL_PRECISION': None}),
    }
    if args.baseline is not None:
        variant_specs = dict(baseline_auto=dict(binary='baseline', env={
            'OPENMX_ORBS_EVAL_PRECISION': None, 'OPENMX_SETDEN_FP64_CUDA': None,
            'OPENMX_SETDEN_FP64_SPLIT_SPIN': None}), **variant_specs)
    for spec in variant_specs.values():
        env = common_env.copy()
        for key, value in spec['env'].items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        spec['process_environment'] = env
    command_prefix = [args.mpirun, '-np', str(args.ranks), '--bind-to', 'core']
    prefixes = ('OPENMX_', 'OMP_', 'CUDA_', 'CUBLAS_', 'NVCOMPILER_', 'GEMMUL8_', 'OPENBLAS_', 'BLIS_')
    metadata = dict(binary_source=str(args.binary.resolve()), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    baseline_source=str(args.baseline.resolve()) if args.baseline else None,
                    baseline_sha256=hashlib.sha256(binaries['baseline'].read_bytes()).hexdigest() if args.baseline else None,
                    ranks=args.ranks, cases=args.cases, scf_max_iter=8, input_directory=str(inputs), timeout=args.timeout,
                    command_prefix=command_prefix,
                    variants={name: dict(binary=str(binaries[spec['binary']]), changes=spec['env'],
                              environment={key: value for key, value in spec['process_environment'].items()
                                           if key.startswith(prefixes)}) for name, spec in variant_specs.items()})
    (output / 'command.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for case in args.cases:
        for variant, spec in variant_specs.items():
            folder = output / case / variant
            folder.mkdir(parents=True)
            shutil.copy2(inputs / (case + '.dat'), folder / 'input.dat')
            env = spec['process_environment']
            command = command_prefix + [str(binaries[spec['binary']]), 'input.dat', '-nt', '1']
            record = dict(case=case, variant=variant, command=command, status='running')
            (folder / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
            print('RUN', case, variant, flush=True)
            start = time.monotonic()
            with (folder / 'run.log').open('w') as log:
                proc = subprocess.Popen(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                try:
                    code = proc.wait(timeout=args.timeout)
                    record.update(status='completed' if code == 0 else 'failed', exit_code=code)
                except subprocess.TimeoutExpired:
                    stop_job(proc)
                    record.update(status='timeout', exit_code=proc.returncode)
                except KeyboardInterrupt:
                    stop_job(proc)
                    record.update(status='interrupted', exit_code=proc.returncode)
                    record['elapsed_wall_s'] = time.monotonic() - start
                    (folder / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
                    raise
            record['elapsed_wall_s'] = time.monotonic() - start
            (folder / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
            print('DONE', case, variant, record['status'], f"{record['elapsed_wall_s']:.3f}s", flush=True)
    return 0 if collect(output, args.ranks, args.cases, args.baseline is not None) else 1


if __name__ == '__main__':
    raise SystemExit(main())
