#!/usr/bin/env python3
"""Execute frozen native suites or named decks inside a full-node PBS job."""
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from compare_gpu_suites import read_output

B = Path(__file__).resolve().parent
ROOT = B.parents[1]
MANIFEST = json.loads((B/'manifest.json').read_text())
SUITES = dict(S='input_example', L='large_example', L2='large2_example', L3='large3_example')

def save(path, result):
    path.write_text(json.dumps(result, indent=2)+'\n')

def inspect_output(path):
    energy, atoms, forces, grids = read_output(path)
    text = path.read_text()
    elapsed = re.search(r'^\s*Elapsed.Time\.\s+(\S+)', text, re.M)
    scf = re.findall(r'SCF=\s*(\d+)\s+NormRD=\s*(\S+)', text)
    return dict(energy_hartree=energy, atoms=atoms, forces_hartree_per_bohr=forces,
                grids=grids, elapsed_seconds=float(elapsed[1]) if elapsed else None,
                scf_iterations=len(scf), scf_history=[(int(n),float(v)) for n,v in scf])

def verify_affinity(folder, nodes):
    rows = [json.loads(p.read_text()) for p in sorted((folder/'affinity').glob('rank.*.json'))]
    if len(rows)!=nodes*32 or len({x['rank'] for x in rows})!=nodes*32:
        raise ValueError('Missing or duplicate MPI rank identity')
    hosts = collections.Counter(x['hostname'] for x in rows)
    if len(hosts)!=nodes or set(hosts.values())!={32}:
        raise ValueError('Wrong physical node layout: '+str(hosts))
    for host in hosts:
        cpus = [c for x in rows if x['hostname']==host for c in x['inherited_cpus']]
        if len(cpus)!=96 or len(set(cpus))!=96 or any(len(x['inherited_cpus'])!=3 for x in rows if x['hostname']==host):
            raise ValueError('Wrong or overlapping CPU binding on '+host)
    return dict(nodes=dict(hosts), ranks=len(rows), threads_per_rank=3, unique_cores_per_node=96)

def run(route, workload):
    nodes = MANIFEST['workloads'][workload]
    frozen = B/'binaries'/route
    folder = B/'runs'/route/workload
    records = [x for x in MANIFEST['inputs'] if x['workload']==workload]
    env = {k:v for k,v in os.environ.items() if not k.startswith(('OPENMX_', 'GEMMUL8_'))}
    env.pop('LD_PRELOAD', None)
    env.update(OPENMX_GPU='1' if route=='fork_gpu' else '0', OPENMX_GPU_VERBOSE='1',
               OPENMX_AFFINITY_DIR=str(folder/'affinity'), OMP_NUM_THREADS='3',
               OMP_MAX_ACTIVE_LEVELS='1', OMP_DYNAMIC='FALSE', BLIS_NUM_THREADS='3',
               OPENBLAS_NUM_THREADS='1', MAGMA_NUM_THREADS='1', HSA_XNACK='1', HSA_ENABLE_SDMA='1', HSA_NO_SCRATCH_RECLAIM='1',
               HSA_OVERRIDE_CPU_AFFINITY_DEBUG='0', LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD='0',
               UCX_RCACHE_ENABLE='n', UCX_TLS='self,cma,rc')
    for key in ('OMPX_APU_MAPS', 'OMPX_DISABLE_USM_MAPS', 'OMP_TARGET_OFFLOAD', 'UCX_MEMTYPE_CACHE'):
        env.pop(key, None)
    env['LD_PRELOAD']='/system/apps/rhel/aocl/5.1.0/aocc/lib/libblis-mt.so.5'
    if route!='upstream_cpu':
        env['LD_PRELOAD'] += ':' + ':'.join(str(frozen/name) for name in ('libopenmx_core.so','libopenmx_host_simd.so'))
    probe=subprocess.run([sys.executable,str(B/'blas_probe.py')],env=env,
                         capture_output=True,text=True,check=True)
    blas_configuration=json.loads(probe.stdout)
    result = dict(status='running', route=route, workload=workload, nodes=nodes,
                  ranks_per_node=32, threads=3, jobid=os.environ['PBS_JOBID'],
                  environment={k:v for k,v in env.items() if k.startswith(('OMP','OPENMX','UCX','HSA','BLIS','OPENBLAS','MAGMA','LIBOMPTARGET','LD_'))},
                  binary_sha256=hashlib.sha256((frozen/'openmx').read_bytes()).hexdigest(),
                  blas_configuration=blas_configuration)
    if workload in SUITES:
        command=[sys.executable, str(B/('harness_'+route)/'tools/run_gpu_suites.py'), '--binary', str(frozen/'openmx'),
                 '--output',str(folder),'--suite',workload,'--ranks',str(nodes*32),'--threads','3',
                 '--env','BLIS_NUM_THREADS=3','--env','OPENBLAS_NUM_THREADS=1','--mpirun',str(B/'suite-mpirun'),'--map-by','ppr:32:node:PE=3','--together','--timeout','72000']
        folder.parent.mkdir(parents=True,exist_ok=True)
        logpath=folder.parent/(workload+'.driver.log')
    else:
        folder.mkdir(parents=True,exist_ok=False)
        shutil.copy2(B/records[0]['routes'][route]['path'], folder/'input.dat')
        command=['mpirun','-np','32','--bind-to','core','--map-by','ppr:32:node:PE=3',sys.executable,
                 str(B/'rank-launcher.py'),str(frozen/'openmx'),'input.dat','-nt','3']
        logpath=folder/'run.log'
    result['command']=command
    statuspath=folder.parent/(workload+'.status.json')
    save(statuspath,result)
    print('START',route,workload,flush=True)
    start=time.monotonic()
    try:
        with logpath.open('w') as log:
            proc=subprocess.run(command,cwd=folder if workload not in SUITES else B,env=env,stdout=log,stderr=subprocess.STDOUT)
        result.update(returncode=proc.returncode,wall_seconds=time.monotonic()-start)
        if proc.returncode:
            raise ValueError('OpenMX/suite driver exit '+str(proc.returncode))
        result['affinity']=verify_affinity(folder,nodes)
        if workload in SUITES:
            rows=json.loads((folder/'results.json').read_text())
            if len(rows)!=len(records) or any(x['status']!='passed' and x['status']!='completed' for x in rows):
                raise ValueError('Incomplete native suite')
            outputdir=folder/'together'
            result['benchmark_seconds']=sum(x['elapsed_seconds'] for x in rows)
            result['native_results']=rows
        else:
            outputdir=folder
        result['cases']=[]
        for record in records:
            row=dict(name=record['name'],original_input_sha256=record['sha256'],
                     runtime_input_sha256=record['routes'][route]['sha256'],
                     **inspect_output(outputdir/(record['system']+'.out')))
            if len(row['forces_hartree_per_bohr'])!=3*len(row['atoms']):
                raise ValueError('Incomplete forces')
            result['cases'].append(row)
        if workload not in SUITES:
            result['benchmark_seconds']=result['cases'][0]['elapsed_seconds']
        if result['benchmark_seconds'] is None or not math.isfinite(result['benchmark_seconds']):
            raise ValueError('Missing finite elapsed time')
        result['status']='completed'
    except Exception as error:
        result.update(status='failed',error=str(error),wall_seconds=time.monotonic()-start)
    save(statuspath,result)
    print('END',route,workload,result['status'],round(result['wall_seconds'],3),result.get('error',''),flush=True)
    return result['status']=='completed'

if __name__=='__main__':
    if (B/'budget-stop.json').exists():
        raise SystemExit('Benchmark stopped by the 130-point project-budget policy; automatic restart is disabled. See budget-stop.json.')
    if not os.environ.get('PBS_JOBID'):
        raise SystemExit('Run inside PBS')
    from budget_guard import check_cached_budget
    check_cached_budget()
    route,*workloads=sys.argv[1:]
    if route not in MANIFEST['binaries'] or not workloads or set(workloads)-MANIFEST['workloads'].keys():
        raise SystemExit('Invalid route/workload')
    ok=True
    for workload in workloads:
        ok=run(route,workload) and ok
    raise SystemExit(0 if ok else 1)
