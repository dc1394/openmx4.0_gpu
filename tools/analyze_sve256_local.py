#!/usr/bin/env python3
"""Validate complete sidia333 runs before computing any speedup."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]/'benchmarks/results/graviton3_sve256_20261105'

def original_bytes(folder, name):
    path=folder/name
    if path.exists():
        return path.read_bytes()
    return gzip.decompress((folder/(name+'.gz')).read_bytes())

def read(name):
    folder=ROOT/name
    result=json.loads((folder/'result.json').read_text())
    command=json.loads((folder/'command.json').read_text())
    assert command.get('purpose','benchmark')=='benchmark', name
    assert result['returncode']==0 and result['normal_finish'], name
    assert command['steps']==25 and result['scf_steps']==25, name
    assert result['rank_audit_passed'], name
    assert command['ranks']*command['threads']==32, name
    if command['threads']==1 and 'actual_blas_threads' in result:
        assert result['actual_blas_threads']==[1], name
    assert len(result['forces'])==648 and len(result['atoms'])==216, name
    assert all(map(math.isfinite,[result['energy'],*result['forces']])), name
    assert len(result['scf_history'])==25, name
    for key,path in [('log_sha256','run.log'),('output_sha256','sidia333_nc_cluster.out')]:
        assert hashlib.sha256(original_bytes(folder,path)).hexdigest()==result[key], name
    assert hashlib.sha256((folder/'input.dat').read_bytes()).hexdigest()==command['effective_sha256'], name
    return dict(name=name,result=result,command=command)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',nargs='+',required=True)
    p.add_argument('--optimized',nargs='+',required=True)
    p.add_argument('--output',default='analysis')
    args=p.parse_args()
    baseline=[read(x) for x in args.baseline]
    optimized=[read(x) for x in args.optimized]
    for group in (baseline,optimized):
        for run in group[1:]:
            for key in ('binary_sha256','libraries','environment','ranks','threads'):
                assert run['command'][key]==group[0]['command'][key], (run['name'],key)
            for key in ('blas_sha256','blas_kind'):
                if key in run['command'] and key in group[0]['command']:
                    assert run['command'][key]==group[0]['command'][key], (run['name'],key)
    reference=baseline[0]
    comparisons=[]
    for run in baseline+optimized:
        b=reference['result'];r=run['result'];c=run['command']
        assert c['effective_sha256']==reference['command']['effective_sha256'], run['name']
        assert r['atoms']==b['atoms'] and r['grids']==b['grids'], run['name']
        assert r['scf_not_converged']==b['scf_not_converged'], run['name']
        de=abs(r['energy']-b['energy'])
        differences=[x-y for x,y in zip(r['forces'],b['forces'])]
        df=max(map(abs,differences))
        due=max(abs(x['uele']-y['uele']) for x,y in zip(r['scf_history'],b['scf_history']))
        assert de<=1e-7 and df<=1e-7, (run['name'],de,df)
        comparisons.append(dict(name=run['name'],wall_seconds=r['wall_seconds'],
            ranks=c['ranks'],threads=c['threads'],energy_difference_Ha=de,
            max_force_difference_Ha_bohr=df,
            rms_force_difference_Ha_bohr=math.sqrt(sum(x*x for x in differences)/648),
            max_scf_uele_difference_Ha=due,scf_not_converged=r['scf_not_converged']))
    def timings(runs):
        times=[x['result']['wall_seconds'] for x in runs]
        return dict(samples=times,median=statistics.median(times),minimum=min(times),maximum=max(times))
    before=timings(baseline);after=timings(optimized)
    phases={k:dict(baseline_median=statistics.median(x['result']['phases'][k] for x in baseline),
                   optimized_median=statistics.median(x['result']['phases'][k] for x in optimized))
            for k in reference['result']['phases']}
    result=dict(baseline=before,optimized=after,speedup=before['median']/after['median'],
                complete_blas_audits=all('actual_blas_threads' in x['result'] and
                    'blas_sha256' in x['command'] for x in baseline+optimized),
                reduction_percent=100*(1-after['median']/before['median']),
                comparisons=comparisons,phases=phases,all_checks_passed=True)
    (ROOT/(args.output+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    report=(f"Baseline: {before['median']:.3f} s, samples {before['samples']}\n"
            f"Optimized: {after['median']:.3f} s, samples {after['samples']}\n"
            f"Speedup: {result['speedup']:.4f}x; reduction {result['reduction_percent']:.2f}%\n"
            f"Maximum energy difference: {max(x['energy_difference_Ha'] for x in comparisons):.6g} Ha\n"
            f"Maximum force-component difference: {max(x['max_force_difference_Ha_bohr'] for x in comparisons):.6g} Ha/bohr\n"
            "216 atoms, 648 force components, identical coordinates/grid/input and 25 SCF steps verified.\n"
            f"SCF not converged: {reference['result']['scf_not_converged']}. This is a fixed-iteration benchmark.\n")
    (ROOT/(args.output+'.txt')).write_text(report)
    print(report)

if __name__=='__main__':
    main()
