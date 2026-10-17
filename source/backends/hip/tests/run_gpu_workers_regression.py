#!/usr/bin/env python3
"""Compare all SCF/force outputs while validating non-dense GPU ownership.

Run inside a single-node GPU allocation. The caller manages MPI. Each
output directory is new, and both binaries are frozen before any run starts.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from types import SimpleNamespace
from run_mgpu_sidia_benchmark import CASES, silicon_input
from run_mgpu_cluster_regression import REPO, compare, digest, keyword, read_output, save, stop_job


def run(args, case, candidate):
    folder=args.output/case/('workers' if candidate else 'baseline')
    folder.mkdir(parents=True)
    text,atoms=silicon_input(case,SimpleNamespace(cells=args.cells,datapath=args.datapath,
        scf_iterations=args.scf_iterations,gpus=args.gpus))
    for key,value in [('scf.eigen.lib','gpusolver'),('scf.mgpu.enable','on'),
                      ('scf.fp32.enable',args.fp32),('scf.Mixing.Type',args.mixing)]:
        text=keyword(text,key,value)
    for item in args.input:
        key,sep,value=item.partition('=')
        if not sep: raise ValueError('--input needs KEYWORD=value')
        text=keyword(text,key,value)
    if args.spin_imbalance:
        text,count=re.subn(r'^(\s*\d+\s+Si\s+\S+\s+\S+\s+\S+)\s+2\.0\s+2\.0(.*)$',
            lambda m: f'{m[1]}  {2+args.spin_imbalance/2:.8f} {2-args.spin_imbalance/2:.8f}{m[2]}',text,flags=re.M)
        if count!=atoms: raise ValueError('Incomplete initial spin population override')
    (folder/'input.dat').write_text(text)
    controls={
        'OMP_NUM_THREADS':str(args.threads),'OPENBLAS_NUM_THREADS':str(args.threads),
        'MAGMA_NUM_THREADS':str(args.threads),'OPENMX_EIGEN_REFINE':'0',
        'OPENMX_MGPU_SOLVER':'single','OPENMX_MGPU_GEMM':'0','OPENMX_MGPU_VERBOSE':'1',
        'OPENMX_GPU_WORKERS':args.workers_mode if candidate else '0',
        'OPENMX_CLUSTER_GPU_SWITCH_NUM':'1','OPENMX_CLUSTER_NONCOL_GPU_SWITCH_NUM':'1',
        'OPENMX_BAND_GPU_SWITCH_NUM':'1','OPENMX_BAND_NONCOL_GPU_SWITCH_NUM':'1'}
    for item in args.env+(args.candidate_env if candidate else []):
        key,sep,value=item.partition('=')
        if not sep: raise ValueError('--env needs NAME=value')
        controls[key]=value
    env={key:value for key,value in os.environ.items() if not key.startswith(('OPENMX_','GEMMUL8_'))}
    env.update(controls)
    binary=args.output/('openmx-workers' if candidate else 'openmx-baseline')
    command=[args.mpirun,'-np',str(args.ranks),'--map-by',f'slot:PE={args.threads}',
        '--bind-to','core',str(binary),'input.dat','-nt',str(args.threads)]
    result=dict(case=case,candidate=candidate,controls=controls,command=command,
        binary_sha256=digest(binary),input_sha256=digest(folder/'input.dat'),status='failed')
    started=time.monotonic()
    try:
        with (folder/'openmx.log').open('w') as log:
            process=subprocess.Popen(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try: process.wait(timeout=args.timeout)
            finally:
                if process.poll() is None: stop_job(process)
        result['returncode']=process.returncode
        log=(folder/'openmx.log').read_text(errors='replace')
        if process.returncode or 'normally finished' not in log: raise ValueError('OpenMX did not finish normally')
        if re.search(r'falling back to (?:the )?(?:ScaLAPACK|ELPA)|GPU initialization failed',log):
            raise ValueError('Unexpected CPU eigensolver fallback')
        steps=sorted(set(map(int,re.findall(r'\bMD=\s*1\s+SCF=\s*(\d+)',log))))
        if steps!=list(range(1,args.scf_iterations+1)): raise ValueError(f'Unexpected SCF steps {steps}')
        energy,coordinates,forces,grids=read_output(folder/'sidia.out')
        components={key:float(value.replace('D','E')) for key,value in re.findall(
            r'^\s*((?:U\w+|Enpy))\.\s+(\S+)',(folder/'sidia.out').read_text(),re.M)}
        if len(coordinates)!=atoms or len(forces)!=3*atoms or not components: raise ValueError('Incomplete numerical output')
        if not all(math.isfinite(x) for x in [energy]+list(forces)+list(components.values())): raise ValueError('Nonfinite output')
        work=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in log.splitlines() if '<OpenMX_MGPU> work name=' in line]
        for row in work:
            if row.get('tier')=='remote':
                if int(row.get('nodes','0'))<1: raise ValueError(f'Invalid remote placement: {row}')
                continue
            devices=list(map(int,row['devices'].split(',')))
            if len(set(devices))!=int(row['workers']) or any(d<0 or d>=args.gpus for d in devices):
                raise ValueError(f'Worker GPU limit/ownership violation: {row}')
            if row['completed'] not in ('0','1'): raise ValueError(f'Invalid worker completion: {row}')
        eligible=args.gpus>=2 and (not case.startswith('band_') or args.gpus//2>=2)
        if candidate and args.workers_mode=='1' and eligible and args.ranks<args.gpus and not any(int(r.get('workers','1'))>1 and r['completed']=='1' for r in work):
            raise ValueError('No successful multi-GPU non-diagonalization work')
        if not candidate and work: raise ValueError('Worker-disabled baseline unexpectedly distributed work')
        if candidate:
            names={r['name'] for r in work if r['completed']=='1'}
            missing=set(args.require_work)-names
            if missing: raise ValueError(f'Required GPU operations not exercised: {sorted(missing)}')
        result.update(energy_hartree=energy,energy_components=components,atoms=coordinates,
            forces_hartree_per_bohr=forces,grids=grids,scf_steps=steps,worker_records=work,status='completed')
    except (ValueError,OSError,subprocess.TimeoutExpired) as exc: result['error']=str(exc)
    result['elapsed_seconds']=time.monotonic()-started
    save(folder/'result.json',result)
    print(json.dumps({k:result[k] for k in ('case','candidate','status','elapsed_seconds')}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--datapath',type=Path,default=REPO/'DFT_DATA19')
    p.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES));p.add_argument('--cells',type=int,default=2)
    p.add_argument('--gpus',type=int,default=4);p.add_argument('--ranks',type=int,default=1);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--scf-iterations',type=int,default=2);p.add_argument('--fp32',choices=('on','off'),default='off')
    p.add_argument('--mixing',default='rmm-diisk');p.add_argument('--env',action='append',default=[])
    p.add_argument('--candidate-env',action='append',default=[],help='Candidate-only capacity/fault controls')
    p.add_argument('--spin-imbalance',type=float,default=0,help='Initial Si up-minus-down population; use with spin on/NC')
    p.add_argument('--input',action='append',default=[],help='Override an input keyword for both binaries')
    p.add_argument('--workers-mode',choices=('1','auto','off'),default='1',help='Candidate auxiliary GPU capacity/performance policy')
    p.add_argument('--require-work',action='append',default=[],help='Require a successful named distributed phase')
    p.add_argument('--mpirun',default='mpirun');p.add_argument('--timeout',type=float,default=300)
    p.add_argument('--energy-tolerance',type=float,default=1e-7);p.add_argument('--force-tolerance',type=float,default=1e-7)
    args=p.parse_args();args.output=args.output.resolve();args.datapath=args.datapath.resolve()
    if not math.isfinite(args.spin_imbalance) or abs(args.spin_imbalance)>4:p.error('Si spin imbalance must be within [-4,4]')
    args.output.mkdir(parents=True,exist_ok=False)
    for source,name in ((args.baseline,'openmx-baseline'),(args.binary,'openmx-workers')):shutil.copy2(source,args.output/name)
    results=[];comparisons=[];passed=True
    for case in args.cases:
        pair=[run(args,case,candidate) for candidate in (False,True)];results.extend(pair)
        if all(r['status']=='completed' for r in pair):
            comparison=dict(case=case,**compare(pair[0],pair[1],args))
            comparison['passed'] &= max(comparison['energy_component_differences_hartree'].values())<=args.energy_tolerance
            comparisons.append(comparison);passed &= comparison['passed']
        else:passed=False
        save(args.output/'results.json',results);save(args.output/'comparisons.json',comparisons)
    save(args.output/'summary.json',dict(passed=passed,runs=len(results),comparisons=comparisons))
    return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
