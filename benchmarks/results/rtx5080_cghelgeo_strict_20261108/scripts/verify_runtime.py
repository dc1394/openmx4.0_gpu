import json,re
from pathlib import Path
from run import A,ROOT,VARIANTS,sha,save
def cpu_set(mask):
    result=set()
    for part in mask.split(','):
        if '-' in part:
            lo,hi=map(int,part.split('-'));result.update(range(lo,hi+1))
        else:result.add(int(part))
    return result
def main():
    assert json.loads((A/'progress.json').read_text())==dict(status='completed',runs=4)
    provenance=json.loads((A/'provenance.json').read_text())
    probe=json.loads((A/'preliminary/diagnostic.json').read_text())
    assert not probe['converged'] and probe['SCF_steps']==200 and probe['criterion']==1e-12
    seed=A/'preliminary/upstream_criterion_1e-12/cghelgeo_rst'
    for d in probe['restart_files']:assert sha(seed/d['path'])==d['sha256']
    for d in provenance['datasets']:
        assert sha(ROOT/d['path'])==d['sha256']
        assert sha(A/d['path'])==d['sha256']
    for name,digest in provenance['libraries'].items():assert sha(A/'lib'/name)==digest
    assert sha(A/'cusolver_audit.cpp')==provenance['cuSOLVER_audit']['source_sha256']
    assert sha(A/'cusolver_audit.so')==provenance['cuSOLVER_audit']['library_sha256']
    variants=[]
    for variant in VARIANTS:
        folder=A/'runs'/variant
        r=json.loads((folder/'result.json').read_text());log=(folder/'run.log').read_text()
        assert r['status']=='completed' and r['converged'] and r['returncode']==0
        assert r['criterion']==1e-10
        assert (folder/'input.dat').read_text()==(A/'planned-inputs'/(variant+'.dat')).read_text()
        ranks=[json.loads(p.read_text()) for p in folder.glob('rank-??.json')]
        assert len(ranks)==18 and sorted(x['rank'] for x in ranks)==list(range(18))
        assert all(x['cpus']==[x['rank'],x['rank']+18] for x in ranks)
        for rank in ranks:
            threads=json.loads((folder/('rank-%02d-threads.json'%rank['rank'])).read_text())
            expected=set(rank['cpus'])
            main=next(t for t in threads if t['tid']==rank['pid'])
            assert cpu_set(main['cpus'])<=expected
            pinned=[cpu_set(t['cpus']) for t in threads if cpu_set(t['cpus'])<=expected]
            assert set().union(*pinned)==expected
        maps=list(folder.glob('rank-??-maps.txt'));assert len(maps)==18,(variant,len(maps))
        for p in maps:
            text=p.read_text()
            if variant=='upstream':
                assert '/intel/oneapi/mpi/2021.18/' in text and 'libiomp5.so' in text
                assert 'libnvomp' not in text
            else:
                assert str(A/'lib/libopenmx_core.so') in text and str(A/'lib/libopenmx_host_simd.so') in text
                assert 'libnvomp.so' in text and 'libiomp5' not in text and 'libgomp' not in text
                assert '/hpcx-2.50/ompi5/' in text
        details=dict(variant=variant,all_18_rank_placements_verified=True,all_18_runtime_maps_verified=True,
                     all_18_main_and_pinned_thread_masks_verified=True)
        if variant=='upstream':
            assert '<Restart>  Found restart files' in log and '<Restart>  Could not find restart files' not in log
            details['same_geometry_restart_verified']=True
        if variant=='cpu_avx512':
            assert re.search(r'host SIMD\s*=\s*AVX-512',log),variant
            assert 'GPU-accelerated' not in log
            details['AVX512_runtime_report_verified']=True
        if variant.startswith('gpu'):
            assert 'GPU initialization failed' not in log
            assert 'GPU-accelerated' in log
            assert 'GPU dense memory admission declined' not in log
            modes=[json.loads(line) for p in folder.glob('rank-??-cusolver.jsonl') for line in p.read_text().splitlines()]
            assert modes and all(m['status']==m['read_status']==0 for m in modes)
            solves=[m for m in modes if m['event'] in ['Xsyevd','Xsyevdx']]
            assert solves and max(m['n'] for m in solves)>1000
            emulated_fp64=[m for m in solves if m['mode']==m['emulated_mode'] and m['arithmetic'].endswith('fp64')]
            switched=[m for m in modes if m['event']=='set' and m['mode']==m['emulated_mode']]
            if variant=='gpu_native_fp64':
                assert 'scf.fp32.enable=off' in log and 'scf.gemmul8.enable=off' in log
                assert not switched and all(m['mode']==m['native_mode'] for m in modes)
                assert all(m['arithmetic'].endswith('fp64') for m in solves)
            else:
                assert switched and 'scf.fp32.enable=off' not in log and 'scf.gemmul8.enable=off' not in log
                assert emulated_fp64
            details.update(cusolver_records=len(modes),emulated_mode_successes=len(switched),
                           cusolver_solves=len(solves),emulated_fp64_solves=len(emulated_fp64),
                           largest_eigensystem=max(m['n'] for m in solves),
                           solver_arithmetic=sorted({m['arithmetic'] for m in solves}),
                           native_modes=sorted({m['native_mode'] for m in modes}),emulated_modes=sorted({m['emulated_mode'] for m in modes}))
            gemms=re.findall(r'real (\d+) \+ (\d+) calls,.*?complex (\d+) \+ (\d+) calls',log)
            assert gemms,variant
            emulated_calls=sum(int(a)+int(c) for a,b,c,d in gemms)
            native_calls=sum(int(b)+int(d) for a,b,c,d in gemms)
            assert (emulated_calls==0 and native_calls>0) if variant=='gpu_native_fp64' else emulated_calls>0
            details.update(gemmul8_calls=emulated_calls,native_cublas_calls=native_calls)
        variants.append(details)
    mps=json.loads((A/'mps-server-list.json').read_text())
    assert mps['returncode']==0 and re.search(r'\d+',mps['stdout'])
    assert json.loads((A/'mps-stop.json').read_text())['returncode']==0
    data=dict(passed=True,variants=variants,MPS_server_observed=True,MPS_cleaned_up=True,
      source_and_snapshot_data_hashes_verified=True,frozen_libraries_verified=True,
      comparison_criterion=1e-10,preliminary_1e_minus12_probe_converged=False,
      restart_seed_hashes_verified=True)
    save(A/'runtime-validation.json',data);print(json.dumps(data,indent=2))
if __name__=='__main__':main()
