#!/usr/bin/env python3
import json,os,re,shlex,shutil,subprocess,sys,time
from pathlib import Path
from runtime import *
sys.path.insert(0,str(ROOT/'tools'))
from run_gpu_suites import stop_job
def run(cmd,name,env=None,cwd=None,timeout=300):
    with (A/(name+'.log')).open('w') as log:
        p=subprocess.Popen(list(map(str,cmd)),env=env or environment(),cwd=cwd or A,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:stop_job(p);raise
    if code:raise RuntimeError(name+' failed; see '+str(A/(name+'.log')))
    return (A/(name+'.log')).read_text()
def main():
    save(A/'preflight-state.json',dict(status='waiting_for_build'))
    while True:
        try:state=json.loads((A/'build-state.json').read_text())
        except (OSError,json.JSONDecodeError):time.sleep(2);continue
        if state['status']=='failed':raise RuntimeError('Build failed')
        if state['status']=='completed':break
        time.sleep(2)
    save(A/'preflight-state.json',dict(status='checking_linkage'))
    ldd=run(['ldd',binary('intel')],'intel-ldd')
    assert 'not found' not in ldd and 'libiomp5.so' in ldd and 'libmkl_intel_thread' in ldd
    assert all(name not in ldd for name in ['libgomp','libnvomp','libcudart','libcusolver','libamdhip64'])
    assert str(MPI/'lib') in ldd and str(MKL/'lib') in ldd
    run(['ldd',binary('upstream')],'upstream-ldd',env=environment('upstream'))
    commands=json.loads((A/'build-commands.json').read_text())['commands']
    cmake=Path(commands['configure'][0]);ctest=cmake.with_name('ctest')
    run([ctest,'--test-dir',A/'build','--output-on-failure','-R','^cpu-numerical-toolchain$'],'cpu-numerics-ctest')
    # Exercise the same analytic real/complex residual and orthogonality
    # checks with MPI_THREAD_SERIALIZED and two OpenMP threads.
    source=(ROOT/'tests/cpu_numerics.c').read_text()
    assert source.count('MPI_Init(&argc, &argv);')==1
    source=source.replace('MPI_Init(&argc, &argv);','int intel_bench_mpi_thread; MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &intel_bench_mpi_thread); if (intel_bench_mpi_thread < MPI_THREAD_SERIALIZED) MPI_Abort(MPI_COMM_WORLD, 99);')
    hybrid=A/'cpu_numerics_hybrid.c';hybrid.write_text(source)
    entries=json.loads((A/'build/compile_commands.json').read_text())
    entry=next(r for r in entries if r['file']==str(ROOT/'tests/cpu_numerics.c'))
    args=entry.get('arguments') or shlex.split(entry['command'])
    filtered=[];skip=False
    for arg in args:
        if skip:skip=False;continue
        if arg in ['-MT','-MF']:skip=True;continue
        if arg in ['-MD','-MP']:continue
        filtered.append(arg)
    filtered[filtered.index('-o')+1]=str(A/'cpu_numerics_hybrid.o')
    filtered[filtered.index('-c')+1]=str(hybrid)
    run(filtered,'hybrid-numerics-compile',cwd=Path(entry['directory']))
    link=shlex.split((A/'build/CMakeFiles/openmx_cpu_numerics.dir/link.txt').read_text())
    index=next(i for i,arg in enumerate(link) if arg.endswith('/tests/cpu_numerics.c.o'))
    link[index]=str(A/'cpu_numerics_hybrid.o');link[link.index('-o')+1]=str(A/'cpu_numerics_hybrid')
    run(link,'hybrid-numerics-link',cwd=A/'build')
    env=environment();env['I_MPI_PIN_DOMAIN']='['+','.join(MASKS[:2])+']'
    run([MPI/'bin/mpiexec','-np','2',A/'cpu_numerics_hybrid'],'hybrid-numerics',env=env)
    save(A/'preflight-state.json',dict(status='checking_affinity_and_MKL'))
    compiler=A/'build/cpu-mpi/C'
    run([compiler,'-O2','-qopenmp',A/'affinity.c','-o',A/'affinity'],'affinity-compile')
    log=run([MPI/'bin/mpiexec','-np','18',A/'affinity'],'affinity')
    rows=[json.loads(s) for s in log.splitlines() if s.startswith('{"rank"')]
    assert len(rows)==36
    cpus=[]
    for rank in range(18):
        rs=[r for r in rows if r['rank']==rank]
        assert len(rs)==2 and {r['thread'] for r in rs}=={0,1} and all(r['team']==2 for r in rs)
        assert {r['cpu'] for r in rs}=={rank,rank+18}
        assert all(r['allowed']==[r['cpu']] for r in rs)
        cpus.extend(r['cpu'] for r in rs)
    assert sorted(cpus)==list(range(36))
    save(A/'affinity-verified.json',dict(ranks=18,threads=2,all_36_logical_CPUs_used_once=True,mapping='rank i uses SMT siblings i and i+18',observations=rows))
    run([compiler,'-O2','-qopenmp','-I'+str(MKL/'include'),A/'mkl_probe.c','-L'+str(MKL/'lib'),
         '-Wl,--start-group','-lmkl_intel_lp64','-lmkl_intel_thread','-lmkl_core','-Wl,--end-group','-o',A/'mkl_probe'],'mkl-probe-compile')
    env=environment();env['MKL_VERBOSE']='1'
    log=run([MPI/'bin/mpiexec','-np','18',A/'mkl_probe'],'mkl-probe',env=env)
    probes=[json.loads(s[len('PROBE '):]) for s in log.splitlines() if s.startswith('PROBE ')]
    assert len(probes)==18 and all(r['omp_team']==2 and r['omp_limit']==2 and r['mkl_limit']==2 and r['mpi_thread']>=2 and r['bad']==0 for r in probes)
    assert len(re.findall(r'NThr:\s*2\b',log))>=18
    save(A/'mkl-threading-verified.json',dict(ranks=18,OMP_threads=2,MKL_threads=2,verbose_NThr2_reports=len(re.findall(r'NThr:\s*2\b',log)),observations=probes))
    # Untimed native application check confirms the actual CPU ELPA handle.
    folder=A/'native-thread-check';folder.mkdir()
    text=adapted((ROOT/'work/input_example/C60.dat').read_text())
    for key,value in [('scf.maxIter','1'),('MD.Type','Nomd')]:
        text,count=re.subn(r'^(\s*'+re.escape(key)+r'\s+)\S+',lambda m:m[1]+value,text,flags=re.M|re.I)
        if not count:text+='\n'+key+' '+value+'\n'
    (folder/'input.dat').write_text(text)
    env=environment();env['OPENMX_GPU_VERBOSE']='1'
    log=run(launch('intel',['input.dat']),'native-thread-check',env=env,cwd=folder)
    assert 'The calculation was normally finished.' in log
    reports=re.findall(r'<ELPA> CPU OpenMP threads: requested=(\d+) configured=(\d+) MPI_thread=(\d+)',log)
    assert reports and all(r[0:2]==('2','2') and int(r[2])>=2 for r in reports),reports
    save(A/'native-ELPA-threading.json',dict(input='C60, one SCF, untimed',reports=reports))
    versions={}
    for name,cmd in [('icx',[COMP/'bin/icx','--version']),('icpx',[COMP/'bin/icpx','--version']),('ifx',[COMP/'bin/ifx','--version']),('Intel_MPI',[MPI/'bin/mpiexec','--version']),('cmake',[cmake,'--version'])]:
        p=subprocess.run(list(map(str,cmd)),env=environment(),capture_output=True,text=True)
        versions[name]=dict(command=list(map(str,cmd)),returncode=p.returncode,output=p.stdout+p.stderr)
        assert p.returncode==0
    save(A/'versions.json',versions)
    save(A/'preflight-state.json',dict(status='completed',cpu_numerical_test=True,hybrid_numerical_test_2x2=True,affinity_18x2=True,MKL_NThr2_all_18_ranks=True,native_ELPA_threads2=True))
    print('Intel CPU preflight completed',flush=True)
if __name__=='__main__':main()
