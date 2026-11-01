#!/usr/bin/env python3
"""Audit fresh GPU results without treating a partial suite as complete."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys

B = Path(__file__).resolve().parent
ROOT = B.parents[1]
OLD = ROOT / 'work/codex_sirius_benchmark_20261031'
PREV = ROOT / 'work/codex_mi300a_ggff_opt_20261101'
sys.path.insert(0, str(ROOT / 'tools'))
from compare_gpu_suites import read_output
from run import verify_affinity

spec = importlib.util.spec_from_file_location('old_collection', OLD / 'collect.py')
collection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collection)
M = json.loads((B / 'manifest.json').read_text())
SUITES = dict(S='input_example', L='large_example', L2='large2_example', L3='large3_example')


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_data(path):
    e, a, f, g = read_output(path)
    return dict(energy_hartree=e, atoms=a, forces_hartree_per_bohr=f, grids=g)


def compare(candidate, baseline, position_tolerance=1e-7):
    a, b = candidate['atoms'], baseline['atoms']
    f, g = candidate['forces_hartree_per_bohr'], baseline['forces_hartree_per_bohr']
    assert len(f) == len(g) == 3 * len(a) == 3 * len(b)
    de = abs(candidate['energy_hartree'] - baseline['energy_hartree'])
    df = max(abs(x-y) for x, y in zip(f, g))
    dp = max(abs(x-y) for p, q in zip(a, b) for x, y in zip(p[2:], q[2:]))
    assert all(math.isfinite(x) for x in [de, df, dp, *f, *g])
    grids = tuple(candidate['grids']) == tuple(baseline['grids'])
    atoms = all(tuple(p[:2]) == tuple(q[:2]) for p, q in zip(a, b))
    return dict(delta_energy_hartree=de, max_delta_force_hartree_per_bohr=df,
                max_delta_position_angstrom=dp, grids_match=grids, atoms_match=atoms,
                passed=de <= 1e-7 and df <= 1e-7 and dp <= position_tolerance and grids and atoms)


def write_rows(name, rows):
    if rows:
        with (B / (name + '.csv')).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


def collect():
    cases, checks, trio, bundled, partial, convergence, timings, audits = [], [], [], [], [], [], [], []
    for name, value in M['binaries']['fork_gpu'].items():
        assert digest(B / 'binaries' / name) == value
    for path, value in M['compiled_source_manifest']['source_files'].items():
        if path.startswith(('source/', 'core/')) and Path(path).suffix in ('.c', '.cc', '.cpp', '.cu', '.h', '.hpp', '.json'):
            assert digest(ROOT / path) == value, path
    assert digest(Path(M['runtime']['blas_library_realpath'])) == M['runtime']['blas_library_sha256']
    for record in M['data']:
        assert digest(OLD / record['path']) == record['sha256']
    for workload, nodes in M['workloads'].items():
        folder = B / 'runs/fork_gpu' / workload
        path = folder.parent / (workload + '.status.json')
        d = load(path) if path.exists() else dict(status='pending')
        timings.append(dict(workload=workload, nodes=nodes, status=d['status'],
                            benchmark_seconds=d.get('benchmark_seconds'), wall_seconds=d.get('wall_seconds'),
                            jobid=d.get('jobid'), error=d.get('error')))
        if d['status'] != 'completed':
            continue
        assert d['binary_sha256'] == M['binaries']['fork_gpu']['openmx']
        assert verify_affinity(folder, nodes) == d['affinity']
        for identity in (load(p) for p in (folder / 'affinity').glob('rank.*.json')):
            assert identity['threads'] == 3
            env = identity['runtime_environment']
            assert env['OMP_NUM_THREADS'] == env['BLIS_NUM_THREADS'] == '3'
            assert env['OPENBLAS_NUM_THREADS'] == '1'
            assert env['HSA_OVERRIDE_CPU_AFFINITY_DEBUG'] == '0'
        probe = d['blas_configuration']
        assert probe['bli_info_get_enable_openmp'] == 1
        assert probe['bli_thread_get_num_threads'] == probe['omp_get_max_threads'] == 3
        if workload in SUITES:
            for env in (load(folder / 'command.json')['environment'], load(folder / 'launch-environment.json')):
                assert env['OMP_NUM_THREADS'] == env['BLIS_NUM_THREADS'] == '3'
        conv = collection.read_convergence(folder, workload, d['cases'])
        convergence.extend(dict(workload=workload, **c) for c in conv)
        records = {r['name']: r for r in M['inputs'] if r['workload'] == workload}
        assert {c['name'] for c in d['cases']} == records.keys()
        native = {r['case']: r['elapsed_seconds'] for r in d.get('native_results', [])}
        for case in d['cases']:
            name = case['name']
            r = records[name]
            raw = OLD / r['raw']
            actual = folder / 'together' / SUITES[workload] / (name + '.dat') if workload in SUITES else folder / 'input.dat'
            assert digest(raw) == r['sha256'] == case['original_input_sha256']
            assert digest(actual) == case['runtime_input_sha256'] == r['routes']['fork_gpu']['sha256']
            assert collection.physical(actual.read_text()) == collection.physical(raw.read_text())
            atom_count = int(re.search(r'^\s*Atoms.Number\s+(\d+)', raw.read_text(), re.M | re.I)[1])
            assert len(case['atoms']) == atom_count
            assert len(case['forces_hartree_per_bohr']) == 3 * atom_count
            is_converged = next(c['converged_all_md'] for c in conv if c['case'] == name)
            cases.append(dict(workload=workload, case=name, nodes=nodes, ranks=nodes*32, threads=3,
                              elapsed_seconds=native.get(name, case['elapsed_seconds']),
                              output_elapsed_seconds=case['elapsed_seconds'], scf_iterations=case['scf_iterations'],
                              converged_all_md=is_converged, energy_hartree=case['energy_hartree'],
                              atoms=atom_count, jobid=d['jobid'], original_input_sha256=r['sha256'],
                              runtime_input_sha256=digest(actual),
                              output_sha256=digest((folder/'together' if workload in SUITES else folder)/(r['system']+'.out'))))
            audits.append(dict(workload=workload, case=name, scientific_controls_preserved=True,
                               finite_output=all(math.isfinite(x) for x in [case['energy_hartree'], *case['forces_hartree_per_bohr']]),
                               affinity_valid=True, openmp_and_blas_threads=3))
            if workload in ('S', 'L', 'sidia333_nc_cluster', 'cghelgeo'):
                references = {}
                for route in ('upstream_cpu', 'fork_cpu', 'fork_gpu'):
                    baseline = next(c for c in load(OLD/'runs'/route/(workload+'.status.json'))['cases'] if c['name'] == name)
                    references[route] = baseline
                    checks.append(dict(workload=workload, case=name, baseline='20261031_'+route, **compare(case, baseline)))
                baseline = next(c for c in load(PREV/'runs/fork_gpu'/(workload+'.status.json'))['cases'] if c['name'] == name)
                checks.append(dict(workload=workload, case=name, baseline='20261101_previous_fork_gpu', **compare(case, baseline)))
                references['fresh_gpu'] = case
                for left, right in [('upstream_cpu','fork_cpu'), ('upstream_cpu','fresh_gpu'), ('fork_cpu','fresh_gpu')]:
                    trio.append(dict(workload=workload, case=name, baseline=left, candidate=right,
                                     **compare(references[right], references[left])))
            else:
                assert digest(raw.with_suffix('.out')) == r['reference_sha256']
                reference = output_data(raw.with_suffix('.out'))
                bundled.append(dict(workload=workload, case=name, baseline='bundled_reference',
                                    **compare(case, reference, position_tolerance=1.000001e-5)))
                # A previously stopped suite still has independent complete C1000 outputs.
                for route in ('upstream_cpu', 'fork_cpu', 'fork_gpu'):
                    oldoutput = OLD/'runs'/route/workload/'together'/(r['system']+'.out')
                    if oldoutput.is_file() and not oldoutput.is_symlink():
                        partial.append(dict(workload=workload, case=name, baseline='20261031_completed_prefix_'+route,
                                            **compare(case, output_data(oldoutput))))
    result = dict(complete=len(cases)==len(M['inputs']), expected_cases=len(M['inputs']), completed_cases=len(cases),
                  cases=cases, timings=timings, input_audits=audits, convergence=convergence,
                  previous_comparisons=checks, current_three_route_comparisons=trio,
                  bundled_reference_comparisons=bundled, previous_prefix_comparisons=partial,
                  strict_energy_tolerance_hartree=1e-7, strict_force_tolerance_hartree_per_bohr=1e-7,
                  data_files_checked=len(M['data']))
    (B/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    for name, rows in [('cases',cases),('timings',timings),('comparisons',checks),
                       ('current-three-route-comparisons',trio),('bundled-reference-comparisons',bundled),
                       ('previous-prefix-comparisons',partial)]:
        write_rows(name, rows)
    print(json.dumps(dict(complete=result['complete'], cases=len(cases),
                          primary_checks=len(trio), primary_mismatches=sum(not r['passed'] for r in trio),
                          previous_checks=len(checks), previous_mismatches=sum(not r['passed'] for r in checks),
                          bundled_checks=len(bundled), bundled_mismatches=sum(not r['passed'] for r in bundled))))


if __name__ == '__main__':
    collect()
