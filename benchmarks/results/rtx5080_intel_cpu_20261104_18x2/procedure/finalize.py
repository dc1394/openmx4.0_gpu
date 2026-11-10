import csv,json,re,subprocess,sys
from pathlib import Path
from runtime import *
def main():
    assert json.loads((A/'progress.json').read_text())==dict(status='completed',runs=4)
    with (A/'analysis-output.log').open('w') as log:
        subprocess.run([sys.executable,str(A/'analyze.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    data=json.loads((A/'analysis.json').read_text())
    assert data['completed_runs']==4 and len(data['comparison'])==2
    runs=json.loads((A/'results.json').read_text())
    assert len(runs)==4 and all(r['status']=='completed' and r['returncode']==0 for r in runs)
    inputs=json.loads((A/'input-data-audit.json').read_text())
    for r in inputs['inputs']:assert digest(Path(r['source']))==r['sha256']
    for r in inputs['datasets']:assert digest(Path(r['source']))==digest(Path(r['snapshot']))==r['sha256']
    start=json.loads((A/'provenance-start.json').read_text())
    assert digest(ROOT/'work/openmx')==start['existing_work_binary_sha256']
    assert digest(Path(start['upstream_source']))==digest(binary('upstream'))==start['upstream_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==start['source_commit']
    assert not subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=no'],cwd=ROOT,text=True)
    frozen=json.loads((A/'measured-binaries.json').read_text())
    assert all(digest(binary(v))==sha for v,sha in frozen.items())
    links=[]
    for r in runs:
        maps=(A/'runs'/r['case']/r['variant']/'rank-maps.txt').read_text()
        assert str(MPI/'lib/libmpi.so') in maps and 'libiomp5.so' in maps
        assert 'libgomp' not in maps and 'libnvomp' not in maps
        if r['variant']=='intel':
            for name in ['libopenmx_core.so','libopenmx_host_simd.so']:
                assert str(A/'install/lib'/name) in maps
            assert 'libmkl_intel_thread.so' in maps and 'libmkl_scalapack_lp64.so' in maps
        links.append(dict(case=r['case'],variant=r['variant'],actual_rank_runtime_verified=True))
    elpa=A/'build/openmx_cpu_elpa_build-prefix/src/openmx_cpu_elpa_build-build/config.log'
    text=elpa.read_text()
    omp={k:re.search(r'^'+k+r"='([^']*)'",text,re.M)[1] for k in ['OPENMP_CFLAGS','OPENMP_FCFLAGS']}
    assert all(omp.values())
    manifest=dict(source=start,build=json.loads((A/'build-state.json').read_text()),
      build_commands=json.loads((A/'build-commands.json').read_text()),versions=json.loads((A/'versions.json').read_text()),
      preflight=json.loads((A/'preflight-state.json').read_text()),ELPA_OpenMP_flags=omp,
      binary_sha256=frozen,installed_library_sha256={p.name:digest(p) for p in (A/'install/lib').glob('libopenmx*.so')},
      ELPA_archive_sha256=digest(A/'build/cpu-elpa/lib/libelpa_openmp.a'),
      runtime_maps_verified=links,protocol=data['protocol'],comparison=data['comparison'],
      start_time=min(r['start_time'] for r in runs),end_time=max(r['end_time'] for r in runs),
      FFTW_provider='oneMKL FFTW-compatible API in libmkl_intel_lp64.so, using MKL FFTW headers',
      core_backend='SERIAL Kokkos 5.2.2 with Intel-compiled host SIMD',
      upstream='Existing original OpenMX executable, copied unchanged; no upstream rebuild',
      limits=['One measurement per input and configuration; no repeated-run averages.',
              'Both sidia333 cases retain the 25-SCF limit and are unconverged.',
              'New ELPA 2026.02.002 and upstream ELPA 2018, C/C++ implementation, link type and inherited precision settings differ.',
              'Native maximum-rank computational time is the primary metric; MPI wall time is reported separately.'],
      original_GPU_work_binary_preserved=True,tracked_worktree_clean=True)
    files=['comparison.csv','analysis.json','results.json','input-data-audit.json','affinity-verified.json',
           'mkl-threading-verified.json','native-ELPA-threading.json','versions.json','build-commands.json']
    manifest['files']={n:dict(sha256=digest(A/n),bytes=(A/n).stat().st_size) for n in files}
    save(A/'manifest.json',manifest)
    print(json.dumps(dict(comparison=data['comparison'],accuracy_failures=data['accuracy_failures'],manifest=str(A/'manifest.json')),indent=2))
if __name__=='__main__':main()
