#!/usr/bin/env python3
"""Wait for the serial campaign, retry the documented upstream stack failure, audit."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import ast, csv, fcntl, gzip, itertools, json, math, os, re, resource, shutil, signal, subprocess, sys, time
R=Path('/home/kawai/openmx4.0_gpu');W=R/'work/codex_benchmark_20261121_8x4';O=R/'benchmarks/results/radeon_rx9060xt_20261121_8x4'
sys.path.insert(0,str(R/'tools'))
from compare_gpu_suites import read_output
from run_gpu_suites import available_bytes,group_rss,stop_job
# Reuse the original pure parsing/auditing functions, without its launch side effects.
tree=ast.parse((W/'benchmark.py').read_text())
keep={'now','sha','save','physical','compare','audit_ranks'}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in keep],type_ignores=[]),str(W/'benchmark.py'),'exec'))
proc=None;cancelled=False
def stop(signum,frame):
    global cancelled
    cancelled=True
    if proc is None:raise SystemExit(0)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
print(now(),'Waiting for primary campaign lock; no overlapping calculations.',flush=True)
lock=(W/'benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
meta=json.loads((W/'queue.json').read_text())
if meta.get('stopped_by_user'):
    print(now(),'Primary campaign stopped by user; no retry launched.',flush=True);raise SystemExit(0)
jobs=meta['jobs'];other=[j for j in jobs if j['status']!='completed' and not(j['test']=='cghelgeo' and j['variant']=='upstream-cpu')]
assert not other,other
job=next(j for j in jobs if j['test']=='cghelgeo' and j['variant']=='upstream-cpu')
if job['status']=='invalid_run':
    assert job['returncode']==255 and job['normal_finish_markers']==0
    folder=Path(job['directory']);assert 'Segmentation fault' in (folder/'run.log').read_text(errors='replace')
    previous=json.loads(json.dumps(job));archive=W/'archives/cghelgeo-upstream-cpu-default-stack';archive.parent.mkdir(exist_ok=True);folder.rename(archive)
    previous['directory']=str(archive);save(O/'attempt-history.json',[previous])
    folder.mkdir();shutil.copy2(archive/'input.dat',folder/'input.dat');command=json.loads((archive/'command.json').read_text());controls=command['controls']
    controls.update(OMP_STACKSIZE='256M');command['controls']=controls;command['process_stack_limit_bytes']='unlimited';save(folder/'command.json',command)
    resource.setrlimit(resource.RLIMIT_STACK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY))
    clean={k:v for k,v in os.environ.items() if not k.startswith(('OPENMX_','OMP_','OMPI_','I_MPI_','MKL_','BLIS_','OPENBLAS_','HIP_','ROCM_','ROCR_','HSA_','LIBOMPTARGET_','MAGMA_','KMP_')) and k not in ('LD_LIBRARY_PATH','LD_PRELOAD')}
    meta['primary_controller_finished']=meta.pop('controller_finished');meta['completion_controller_pid']=os.getpid()
    job.update(status='running',started=now(),previous_attempt=previous,retry_reason='Documented upstream 8 MiB stack SIGSEGV; unlimited process stack plus OMP_STACKSIZE=256M. Same executable and scientific input.')
    save(W/'queue.json',meta);print(now(),'START cghelgeo upstream-cpu stack-corrected retry',flush=True)
    start=time.monotonic();peak=0;minimum=available_bytes();audited=False;last=0;reason=None
    try:
        with (folder/'run.log').open('w') as log:
            proc=subprocess.Popen(command['command'],cwd=folder,env={**clean,**controls},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(folder/'process.json',{'pid':proc.pid,'started':now(),'command':command['command']})
            while proc.poll() is None:
                free=available_bytes();minimum=min(minimum,free);peak=max(peak,group_rss(proc.pid))
                if not audited:audited=audit_ranks(proc.pid,folder/'rank-environment.json','intel')
                if cancelled:reason='interrupted'
                elif free<32*1024**3:reason='host_memory_guard'
                elif time.monotonic()-start>4*3600:reason='timeout_4_hours'
                elif not audited and time.monotonic()-start>60:reason='rank_affinity_audit_failed'
                if reason:stop_job(proc);break
                if time.monotonic()-last>=10:
                    with (folder/'run.log').open('rb') as f:
                        f.seek(0,2);f.seek(max(0,f.tell()-65536));tail=f.read().decode(errors='replace')
                    scf=re.findall(r'MD=\s*(\d+)\s+SCF=\s*(\d+)',tail)
                    sample={'time':now(),'test':'cghelgeo','variant':'upstream-cpu','case':'cghelgeo','completed_cases':0,'total_cases':1,
                      'MD':int(scf[-1][0]) if scf else None,'SCF':int(scf[-1][1]) if scf else None,'wall_s':time.monotonic()-start,
                      'available_gib':free/1024**3,'peak_rss_gib':peak/1024**3,'load_average':os.getloadavg(),'rank_environment_verified':audited,'attempt':'stack-corrected retry'}
                    save(W/'current.json',sample)
                    with (W/'cpu-load.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
                    last=time.monotonic()
                time.sleep(1)
        log=(folder/'run.log').read_text(errors='replace');rc=proc.returncode;proc=None
    finally:
        if proc is not None and proc.poll() is None:stop_job(proc)
    times=[float(x) for x in re.findall(r'Total Computational Time\s*=\s*\d+\s+[\d.]+\s+\d+\s+([\d.]+)',log)]
    valid=(not reason and rc==0 and len(times)==1 and audited and log.count('The calculation was normally finished.')==1)
    job.update(status=reason or ('completed' if valid else 'invalid_run'),ended=now(),returncode=rc,wall_seconds=time.monotonic()-start,
      peak_rss_gib=peak/1024**3,min_available_gib=minimum/1024**3,completed=int(valid),expected=1,normal_finish_markers=log.count('The calculation was normally finished.'),
      rank_environment_verified=audited,seconds=times[-1] if times else 0,gpu_accelerated_markers=0,dense_cpu_fallback_lines=[])
    save(folder/'process.json',{'finished':now(),'returncode':rc,'status':job['status']})
    meta['controller_finished']=now();meta['stopped_by_user']=cancelled;save(W/'queue.json',meta)
    assert valid,job
    result=command['inputs'][0];result.update(status='completed',**physical(folder/'cghelgeo.out'),seconds=times[-1],total_computational_seconds=times[-1],
      scf_steps=len(re.findall(r'MD=\s*\d+\s+SCF=',log)),scf_limit_message_count=0)
    save(folder/'results.json',[result]);runs=json.loads((O/'runs.json').read_text())
    runs=[x for x in runs if not(x['test']=='cghelgeo' and x['variant']=='upstream-cpu')]+[{'test':'cghelgeo','variant':'upstream-cpu',**result}]
    save(O/'runs.json',runs);save(O/'run-results.json',jobs)
    print(now(),'END cghelgeo upstream-cpu stack-corrected retry',job['seconds'],flush=True)
else:
    assert job['status']=='completed',job
    runs=json.loads((O/'runs.json').read_text())
# Normalize tuple/list representation before comparing newly parsed and saved outputs.
runs=json.loads((O/'runs.json').read_text())
comparisons=[];names=['upstream-cpu','fork-cpu','fork-gpu']
for test in ['S','L','cghelgeo','sidia333_nc_cluster']:
    grouped={name:{x['case']:x for x in runs if x['test']==test and x['variant']==name and x['status']=='completed'} for name in names}
    for a,b in itertools.combinations(names,2):
        for case in sorted(grouped[a].keys()&grouped[b].keys()):comparisons.append({'test':test,'case':case,'left':a,'right':b,**compare(grouped[a][case],grouped[b][case])})
save(O/'comparisons.json',comparisons)
for filename,rows,fields in [
 ('case-results.csv',runs,['test','case','variant','status','seconds','total_computational_seconds','scf_steps','energy_Ha','scf_criterion','scf_maxIter']),
 ('run-results.csv',jobs,['test','variant','status','seconds','wall_seconds','completed','expected','peak_rss_gib','min_available_gib','rank_environment_verified']),
 ('comparisons.csv',comparisons,['test','case','left','right','pass','delta_energy_Ha','max_delta_force_Ha_bohr','same_geometry_and_grids'])]:
    with (O/filename).open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
validation=json.loads((O/'validation.json').read_text());validation.update(all_jobs_completed=all(j['status']=='completed' for j in jobs),completed_outputs=sum(x['status']=='completed' for x in runs),
    comparison_passes=sum(x['pass'] for x in comparisons),comparisons=len(comparisons),failed_comparisons=[x for x in comparisons if not x['pass']],finished=now());save(O/'validation.json',validation)
shutil.copy2(W/'stack-limit-adjustment.json',O/'stack-limit-adjustment.json')
if (O/'attempt-history.json').exists():
    for attempt in json.loads((O/'attempt-history.json').read_text()):
        d=O/'failed-attempts'/Path(attempt['directory']).name;d.mkdir(parents=True,exist_ok=True)
        for n in ['run.log','command.json','process.json','results.json','rank-environment.json']:
            p=Path(attempt['directory'])/n
            if p.exists():(d/(n+'.gz')).write_bytes(gzip.compress(p.read_bytes(),mtime=0))
audit=['python3','/tmp/openmx-audit-benchmark-20261121-8x4.py']
subprocess.run(audit,check=True)
p=O/'README.md';text=p.read_text();text+='\n## Stack-limit workaround and failed attempt\n\nThe first upstream cghelgeo attempt used the default 8 MiB process stack and crashed in its first eigensolve (SIGSEGV, no completed output). The same failure and workaround were already documented in `docs/CGHELGEO_STRICT_SCF.md`. It was rerun with **unlimited process stack and `OMP_STACKSIZE=256M`**, retaining the exact binary, scientific input and 8 MPI × 4-thread binding. Failed logs and attempt history are preserved; only the completed retry contributes a benchmark time. Remaining primary launches after the recorded stack adjustment inherited unlimited process stack. Earlier successful jobs are retained. Stack limits do not change arithmetic or SCF conditions. See `attempt-history.json`, `failed-attempts/` and `stack-limit-adjustment.json`.\n';p.write_text(text)
shutil.copy2(__file__,O/'complete.py');(W/'final-audit.done').write_text(now()+'\n');print(now(),'ALL BENCHMARKS AND FINAL AUDIT COMPLETE',flush=True)
