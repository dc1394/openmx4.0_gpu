import json,re,sys
from run import A,sha,save
from verify_runtime import cpu_set
from analyze_control import CONTROL,FULL_CONTROL
def main():
    full='--full-fp64' in sys.argv
    variant=FULL_CONTROL if full else CONTROL
    folder=A/'runs'/variant;r=json.loads((folder/'result.json').read_text());log=(folder/'run.log').read_text()
    assert r['status']=='completed' and r['converged'] and r['returncode']==0 and r['criterion']==1e-10
    env=r['environment']
    for key,value in dict(OPENMX_GPU='1',OPENMX_CUSOLVER_EMULATION='0',OPENMX_EIGEN_REFINE='0',
                          OPENMX_GRID_PRECISION='fp64',OPENMX_ORBS_EVAL_PRECISION='fp64').items():assert env[key]==value
    text=(folder/'input.dat').read_text()
    assert re.search(r'^scf\.fp32\.enable\s+'+('off' if full else 'on')+r'\s*$',text,re.M)
    assert re.search(r'^scf\.gemmul8\.enable\s+off\s*$',text,re.M)
    assert ('scf.fp32.enable=off' in log)==full and 'scf.gemmul8.enable=off' in log
    assert '<Restart>  Found restart files' in log and '<Restart>  Could not find restart files' not in log
    assert 'GPU dense memory admission declined' not in log and 'GPU initialization failed' not in log
    seed=A/'runs/gpu_default/cghelgeo_rst'
    for d in r['restart_seed_files']:assert sha(seed/d['path'])==d['sha256']
    ranks=[json.loads(p.read_text()) for p in folder.glob('rank-??.json')]
    assert len(ranks)==18 and sorted(x['rank'] for x in ranks)==list(range(18))
    for rank in ranks:
        expected={rank['rank'],rank['rank']+18};assert set(rank['cpus'])==expected
        threads=json.loads((folder/('rank-%02d-threads.json'%rank['rank'])).read_text())
        assert cpu_set(next(t for t in threads if t['tid']==rank['pid'])['cpus'])<=expected
        assert set().union(*(cpu_set(t['cpus']) for t in threads if cpu_set(t['cpus'])<=expected))==expected
        maps=(folder/('rank-%02d-maps.txt'%rank['rank'])).read_text()
        assert str(A/'lib/libopenmx_core.so') in maps and str(A/'lib/libopenmx_host_simd.so') in maps
        assert 'libnvomp.so' in maps and 'libiomp5' not in maps and 'libgomp' not in maps
        assert '/hpcx-2.50/ompi5/' in maps
    modes=[json.loads(line) for p in folder.glob('rank-??-cusolver.jsonl') for line in p.read_text().splitlines()]
    assert modes and all(m['status']==m['read_status']==0 and m['mode']==m['native_mode'] for m in modes)
    solves=[m for m in modes if m['event'] in ['Xsyevd','Xsyevdx']]
    assert solves and max(m['n'] for m in solves)>1000 and all(m['arithmetic'].endswith('fp64') for m in solves)
    counts=re.findall(r'real (\d+) \+ (\d+) calls,.*?complex (\d+) \+ (\d+) calls',log);assert counts
    emulated=sum(int(a)+int(c) for a,b,c,d in counts);native=sum(int(b)+int(d) for a,b,c,d in counts)
    assert emulated==0 and native>0
    mps=A/'control-mps'/variant
    for name in ['start','server','stop']:assert json.loads((mps/(name+'.json')).read_text())['returncode']==0
    assert re.search(r'\d+',json.loads((mps/'server.json').read_text())['stdout'])
    data=dict(passed=True,variant=variant,all_18_rank_runtime_and_pinned_thread_checks_passed=True,
      same_geometry_default_gpu_restart_verified=True,restart_seed_hashes_verified=True,MPS_observed_and_stopped=True,
      native_FP64_eigensolver_calls=len(solves),largest_eigensystem=max(m['n'] for m in solves),
      solver_arithmetic=sorted({m['arithmetic'] for m in solves}),GEMMul8_calls=emulated,native_cuBLAS_calls=native,
      grid_precision_request=env['OPENMX_GRID_PRECISION'],orbital_precision_request=env['OPENMX_ORBS_EVAL_PRECISION'],
      inherited_FP32_projector_product_rounding_retained=not full)
    save(A/('full-control-runtime-validation.json' if full else 'control-runtime-validation.json'),data);print(json.dumps(data,indent=2))
if __name__=='__main__':main()
