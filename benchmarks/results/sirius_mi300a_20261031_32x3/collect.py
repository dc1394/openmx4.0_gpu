#!/usr/bin/env python3
"""Collect all 18 measurements and compare every energy/force component."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re

B=Path(__file__).resolve().parent
M=json.loads((B/'manifest.json').read_text())
ROUTES=tuple(M['binaries'])
TOL=1e-7

def physical(text):
    return re.sub(r'^\s*(?:DATA\.PATH|scf\.eigen\.lib|System\.CurrentDirectory|System\.CurrrentDirectory)\s+.*\n?', '', text, flags=re.M|re.I)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_convergence(folder,workload,cases):
    log=folder/'together/run.log' if workload in ('S','L','L2','L3') else folder/'run.log'
    events=re.findall(r'The SCF was (not fully )?achieved at MD=\s*(\d+)',log.read_text(errors='replace'))
    groups=[]
    for failed,md in events:
        if int(md)==1: groups.append([])
        assert groups, 'Missing initial MD convergence event'
        groups[-1].append(dict(md_step=int(md),scf_converged=not bool(failed)))
    assert len(groups)==len(cases), 'Missing per-case SCF convergence report: '+str(folder)
    rows=[]
    for case,events in zip(cases,groups):
        assert len(events)==sum(i==1 for i,norm in case['scf_history']), 'SCF/MD event count mismatch'
        rows.append(dict(case=case['name'],converged_all_md=all(e['scf_converged'] for e in events),md_steps=events))
    return rows

def collect():
    checks=[]
    for x in M['inputs']:
        raw=B/x['raw']
        assert digest(raw)==x['sha256'], 'Changed raw input: '+x['name']
        original=physical(raw.read_text())
        for route in ROUTES:
            record=x['routes'][route]
            path=B/record['path']
            assert digest(path)==record['sha256'], 'Changed prepared input: '+str(path)
            assert physical(path.read_text())==original, 'Changed scientific controls: '+str(path)
        checks.append(dict(workload=x['workload'],case=x['name'],scientific_sha256=hashlib.sha256(original.encode()).hexdigest()))
    for x in M['data']:
        assert digest(B/x['path'])==x['sha256'], 'Changed basis/pseudopotential'
    for route,files in M['binaries'].items():
        for name,value in files.items():
            assert digest(B/'binaries'/route/name)==value, 'Changed frozen binary/library'
    for record in M.get('binary_history',[]):
        for path,value in record['files'].items():
            assert digest(B/path)==value, 'Changed retained build: '+path
    for path,value in M.get('source_artifacts',{}).items():
        assert digest(B/path)==value, 'Changed source/validation provenance: '+path
    for helper in M.get('suite_helpers',{}).values():
        assert digest(B/helper['path'])==helper['sha256'], 'Changed native-suite helper'
    runtime=M.get('runtime',{})
    if 'blas_library_sha256' in runtime:
        assert digest(Path(runtime['blas_library_realpath']))==runtime['blas_library_sha256'], 'Changed AOCL BLIS library'
    results={}
    timings=[]
    for workload,nodes in M['workloads'].items():
        for route in ROUTES:
            path=B/'runs'/route/(workload+'.status.json')
            x=json.loads(path.read_text()) if path.exists() else dict(status='pending')
            if x['status']=='completed':
                expected_binary=M.get('binary_sha256_by_workload',{}).get(route,{}).get(workload,M['binaries'][route]['openmx'])
                assert x['binary_sha256']==expected_binary, 'Unexpected executable for '+route+'/'+workload
                if workload in ('S','L','L2','L3'):
                    folder=B/'runs'/route/workload
                    launch=json.loads((folder/'command.json').read_text())['environment']
                    actual=json.loads((folder/'launch-environment.json').read_text())
                    for env in (launch,actual):
                        assert env.get('OMP_NUM_THREADS')=='3' and env.get('BLIS_NUM_THREADS')=='3', 'Wrong actual MPI/BLAS thread setting'
                    ranks=[json.loads(p.read_text()) for p in (folder/'affinity').glob('rank.*.json')]
                    assert len(ranks)==nodes*32
                    assert all(r.get('runtime_environment',{}).get('BLIS_NUM_THREADS')=='3' for r in ranks), 'MPI rank did not inherit threaded BLAS'
                inputs={r['name']:r for r in M['inputs'] if r['workload']==workload}
                assert len(x['cases'])==len(inputs), 'Incomplete case set'
                for case in x['cases']:
                    record=inputs[case['name']]
                    expected=int(re.search(r'^\s*Atoms.Number\s+(\d+)',(B/record['raw']).read_text(),re.M|re.I)[1])
                    assert len(case['atoms'])==expected and len(case['forces_hartree_per_bohr'])==3*expected, 'Wrong atom/force count'
                    assert math.isfinite(case['energy_hartree']) and all(math.isfinite(v) for v in case['forces_hartree_per_bohr']), 'Non-finite energy/force'
                    assert all(math.isfinite(v) for atom in case['atoms'] for v in atom[2:]), 'Non-finite position'
            results[(workload,route)]=x
            timings.append(dict(workload=workload,nodes=nodes,route=route,status=x['status'],
                                benchmark_seconds=x.get('benchmark_seconds'),wall_seconds=x.get('wall_seconds'),
                                cases=len(x.get('cases',[])),jobid=x.get('jobid'),error=x.get('error'),
                                binary_sha256=x.get('binary_sha256')))
    comparisons=[]
    convergence=[]
    for (workload,route),result in results.items():
        if result['status']=='completed':
            for row in read_convergence(B/'runs'/route/workload,workload,result['cases']):
                convergence.append(dict(route=route,workload=workload,**row))
    for workload in M['workloads']:
        for left,right in [('upstream_cpu','fork_cpu'),('upstream_cpu','fork_gpu'),('fork_cpu','fork_gpu')]:
            a,b=results[(workload,left)],results[(workload,right)]
            if a['status']!='completed' or b['status']!='completed':
                continue
            aa={x['name']:x for x in a['cases']}
            bb={x['name']:x for x in b['cases']}
            expected={x['name'] for x in M['inputs'] if x['workload']==workload}
            assert aa.keys()==bb.keys()==expected, 'Incomplete comparison'
            for name in sorted(expected):
                x,y=aa[name],bb[name]
                assert x['original_input_sha256']==y['original_input_sha256']
                ids_match=[z[:2] for z in x['atoms']]==[z[:2] for z in y['atoms']]
                grids_match=x['grids']==y['grids']
                if len(x['forces_hartree_per_bohr'])!=len(y['forces_hartree_per_bohr']):
                    raise ValueError('Force component count mismatch')
                de=abs(x['energy_hartree']-y['energy_hartree'])
                df=max(abs(u-v) for u,v in zip(x['forces_hartree_per_bohr'],y['forces_hartree_per_bohr']))
                dp=max(abs(u-v) for p,q in zip(x['atoms'],y['atoms']) for u,v in zip(p[2:],q[2:]))
                comparisons.append(dict(workload=workload,case=name,baseline=left,candidate=right,
                    delta_energy_hartree=de,max_delta_force_hartree_per_bohr=df,max_delta_position_angstrom=dp,
                    grids_match=grids_match,atoms_match=ids_match,baseline_scf=x['scf_iterations'],candidate_scf=y['scf_iterations'],
                    status='passed' if de<=TOL and df<=TOL and dp<=1.000001e-5 and ids_match and grids_match else 'mismatch'))
    expected=len(M['workloads'])*len(ROUTES)
    report=dict(scope=M['workloads'],expected_measurements=expected,expected_cases_per_route=len(M['inputs']),
                energy_tolerance_hartree=TOL,force_tolerance_hartree_per_bohr=TOL,
                completed_measurements=sum(x['status']=='completed' for x in timings),
                timings=timings,comparisons=comparisons,convergence=convergence,input_checks=checks,data_files_checked=len(M['data']))
    (B/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,rows in [('timings',timings),('comparisons',comparisons)]:
        if rows:
            with (B/(name+'.csv')).open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
                writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(completed=report['completed_measurements'],expected=expected,
                          failed=[(x['route'],x['workload'],x['error']) for x in timings if x['status']=='failed'],
                          comparisons=len(comparisons),mismatches=sum(x['status']=='mismatch' for x in comparisons))))

if __name__=='__main__':
    collect()
