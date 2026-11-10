#!/usr/bin/env python3
import json,re,subprocess
from runtime import *
from preflight import run
assert '100% tests passed' in (A/'cpu-numerics-ctest.log').read_text()
assert 'numerical libraries: PASS' in (A/'hybrid-numerics.log').read_text()
assert json.loads((A/'affinity-verified.json').read_text())['all_36_logical_CPUs_used_once']
assert json.loads((A/'mkl-threading-verified.json').read_text())['verbose_NThr2_reports']>=18
path=A/'native-thread-check/input.dat';text=path.read_text()
for key,value in [('scf.EigenvalueSolver','Cluster'),('level.of.stdout','3')]:
    text,count=re.subn(r'^(\s*'+re.escape(key)+r'\s+)\S+',lambda m:m[1]+value,text,flags=re.M|re.I)
    if not count:text+='\n'+key+' '+value+'\n'
path.write_text(text)
env=environment();env['OPENMX_GPU_VERBOSE']='1'
log=run(launch('intel',['input.dat']),'native-thread-check-cluster',env=env,cwd=A/'native-thread-check')
assert 'The calculation was normally finished.' in log
reports=re.findall(r'<ELPA> CPU OpenMP threads: requested=(\d+) configured=(\d+) MPI_thread=(\d+)',log)
assert reports and all(r[0:2]==('2','2') and int(r[2])>=2 for r in reports)
save(A/'native-ELPA-threading.json',dict(input='C60, Cluster solver, one SCF, stdout 3, untimed',reports=reports,log='native-thread-check-cluster.log'))
commands=json.loads((A/'build-commands.json').read_text())['commands'];cmake=Path(commands['configure'][0]);versions={}
for name,cmd in [('icx',[COMP/'bin/icx','--version']),('icpx',[COMP/'bin/icpx','--version']),('ifx',[COMP/'bin/ifx','--version']),('Intel_MPI',[MPI/'bin/mpiexec','--version']),('cmake',[cmake,'--version'])]:
    p=subprocess.run(list(map(str,cmd)),env=environment(),capture_output=True,text=True)
    versions[name]=dict(command=list(map(str,cmd)),returncode=p.returncode,output=p.stdout+p.stderr);assert p.returncode==0
save(A/'versions.json',versions)
save(A/'preflight-state.json',dict(status='completed',cpu_numerical_test=True,hybrid_numerical_test_2x2=True,
    affinity_18x2=True,MKL_NThr2_all_18_ranks=True,native_ELPA_threads2=True,
    initial_C60_attempt='Added C5.0/C_CA19 missing from the benchmark-only data snapshot; C60 originally selects DC, so the untimed thread check explicitly selects Cluster to exercise ELPA.'))
print('All Intel CPU preflight checks completed',flush=True)
