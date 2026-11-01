#!/usr/bin/env python3
"""Audit the additionally authorized, fresh two-node CPU L3 measurements."""
import csv
import json
import math
from pathlib import Path
from collect import B, OLD, M, collection, compare, digest, output_data, verify_affinity


def load(path):
    return json.loads(path.read_text())


def save_rows(name, rows):
    if rows:
        with (B/(name+'.csv')).open('w') as handle:
            writer=csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


def collect():
    cases, timings, convergence, bundled, pairs=[], [], [], [], []
    completed={}
    records={r['name']:r for r in M['inputs'] if r['workload']=='L3'}
    for route in ('fork_cpu','upstream_cpu'):
        folder=B/'runs'/route/'L3'
        status=folder.parent/'L3.status.json'
        d=load(status) if status.exists() else dict(status='pending')
        timings.append(dict(route=route,workload='L3',nodes=2,status=d['status'],
                            benchmark_seconds=d.get('benchmark_seconds'),wall_seconds=d.get('wall_seconds'),
                            jobid=d.get('jobid'),error=d.get('error')))
        if d['status']!='completed':continue
        assert d['binary_sha256']==M['binaries'][route]['openmx']
        assert verify_affinity(folder,2)==d['affinity']
        assert d['environment']['OPENMX_GPU']=='0'
        assert d['blas_configuration']['bli_info_get_enable_openmp']==1
        assert d['blas_configuration']['bli_thread_get_num_threads']==3
        for identity in (load(p) for p in (folder/'affinity').glob('rank.*.json')):
            assert identity['threads']==3
            assert identity['runtime_environment']['OMP_NUM_THREADS']==identity['runtime_environment']['BLIS_NUM_THREADS']=='3'
        command_environment=load(folder/'command.json')['environment']
        assert command_environment['OPENMX_GPU']=='0'
        # The existing MPI-wrapper audit records thread/runtime variables;
        # OPENMX_GPU is captured by command.json and the exact Popen env above.
        for env in (command_environment,load(folder/'launch-environment.json')):
            assert env['OMP_NUM_THREADS']==env['BLIS_NUM_THREADS']=='3'
        log=(folder/'together/run.log').read_text()
        assert 'GPUEIGENPROF' not in log, 'Unexpected GPU eigensolver in CPU measurement'
        if route=='fork_cpu':assert 'CPU OpenMP threads: requested=3 configured=3' in log
        assert {c['name'] for c in d['cases']}==records.keys()
        conv=collection.read_convergence(folder,'L3',d['cases'])
        convergence.extend(dict(route=route,workload='L3',**r) for r in conv)
        native={r['case']:r['elapsed_seconds'] for r in d['native_results']}
        completed[route]={c['name']:c for c in d['cases']}
        for case in d['cases']:
            name=case['name'];r=records[name]
            actual=folder/'together/large3_example'/(name+'.dat')
            raw=OLD/r['raw']
            assert digest(raw)==r['sha256']==case['original_input_sha256']
            assert digest(actual)==r['routes'][route]['sha256']==case['runtime_input_sha256']
            assert collection.physical(actual.read_text())==collection.physical(raw.read_text())
            assert len(case['forces_hartree_per_bohr'])==3*len(case['atoms'])
            assert all(math.isfinite(x) for x in [case['energy_hartree'],*case['forces_hartree_per_bohr']])
            is_converged=next(x['converged_all_md'] for x in conv if x['case']==name)
            cases.append(dict(route=route,workload='L3',case=name,nodes=2,ranks=64,threads=3,
                              elapsed_seconds=native[name],output_elapsed_seconds=case['elapsed_seconds'],
                              scf_iterations=case['scf_iterations'],converged_all_md=is_converged,
                              energy_hartree=case['energy_hartree'],atoms=len(case['atoms']),jobid=d['jobid'],
                              original_input_sha256=r['sha256'],runtime_input_sha256=digest(actual),
                              output_sha256=digest(folder/'together'/(r['system']+'.out'))))
            bundled.append(dict(workload='L3',case=name,candidate=route,baseline='bundled_reference',
                                 **compare(case,output_data(raw.with_suffix('.out')),position_tolerance=1.000001e-5)))
    gpath=B/'runs/fork_gpu/L3.status.json'
    if gpath.exists() and load(gpath)['status']=='completed':
        completed['fork_gpu']={c['name']:c for c in load(gpath)['cases']}
    for left,right in [('upstream_cpu','fork_cpu'),('upstream_cpu','fork_gpu'),('fork_cpu','fork_gpu')]:
        if left not in completed or right not in completed:continue
        for name in records:
            pairs.append(dict(workload='L3',case=name,baseline=left,candidate=right,
                              **compare(completed[right][name],completed[left][name])))
    result=dict(complete=len(cases)==40,cases=cases,timings=timings,convergence=convergence,
                paired_comparisons=pairs,bundled_comparisons=bundled,
                strict_energy_tolerance_hartree=1e-7,strict_force_tolerance_hartree_per_bohr=1e-7)
    (B/'cpu-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in [('cpu-cases',cases),('cpu-timings',timings),('L3-three-route-comparisons',pairs),('cpu-bundled-reference-comparisons',bundled)]:
        save_rows(name,rows)
    print(json.dumps(dict(complete=result['complete'],cases=len(cases),comparisons=len(pairs),
                          mismatches=sum(not r['passed'] for r in pairs))))


if __name__=='__main__':
    collect()
