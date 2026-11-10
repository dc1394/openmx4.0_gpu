#!/usr/bin/env python3
import itertools,json,os,re,signal,subprocess,sys,time
from pathlib import Path
from runtime import *
sys.path.insert(0,str(ROOT/'tools'))
from compare_gpu_suites import read_output
from run_gpu_suites import available_bytes,group_rss,stop_job
STOP=False
def interrupt(sig,frame):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
def sample_maps(pid,folder):
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:
            stat=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(stat[3])!=pid:continue
            cmd=(p/'cmdline').read_bytes().split(b'\0')
            if Path(os.fsdecode(cmd[0])).name!='openmx':continue
            (folder/'rank-maps.txt').write_text((p/'maps').read_text());return True
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return False
def run_one(variant,case):
    folder=A/'runs'/case/variant;folder.mkdir(parents=True)
    source=(A/'inputs'/(case+'.dat')).read_text();text=adapted(source)
    assert physical(source)==physical(text)
    (folder/'input.dat').write_text(text)
    env=environment(variant);cmd=launch(variant,['input.dat'])
    assert env['OMP_NUM_THREADS']==env['MKL_NUM_THREADS']=='2' and 'MKL_VERBOSE' not in env
    controls={k:v for k,v in env.items() if k.startswith(('OMP_','MKL_','I_MPI_','OPENMX_','BLIS_','OPENBLAS_','CUDA_')) or k=='LD_LIBRARY_PATH'}
    result=dict(variant=variant,case=case,command=cmd,controls=controls,input_sha256=digest(folder/'input.dat'),
                executable_sha256=digest(binary(variant)),start_time=time.strftime('%Y-%m-%d %H:%M:%S %z'))
    save(folder/'result.json',dict(result,status='running'))
    print('START',variant,case,flush=True)
    samples=[];start=time.monotonic();next_sample=start;peak=0.;minimum=available_bytes();mapped=False
    with (folder/'run.log').open('w') as log:
        p=subprocess.Popen(cmd,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(A/'active-process.json',dict(pid=p.pid,variant=variant,case=case))
        while p.poll() is None:
            elapsed=time.monotonic()-start;memory=available_bytes();rss=group_rss(p.pid)
            minimum=min(minimum,memory);peak=max(peak,rss)
            if not mapped and elapsed>2:mapped=sample_maps(p.pid,folder)
            save(A/'progress.json',dict(status='running',variant=variant,case=case,pid=p.pid,elapsed_seconds=round(elapsed,1),
                                       rss_peak_gib=round(peak/2**30,2),available_gib=round(memory/2**30,2)))
            if elapsed>=next_sample:
                samples.append(dict(elapsed_seconds=round(elapsed,1),rss_gib=round(rss/2**30,3),available_gib=round(memory/2**30,3)))
                save(folder/'resources.json',samples);next_sample=elapsed+20
            if STOP or (A/'STOP').exists() or memory<4*2**30 or elapsed>10800:
                stop_job(p);raise RuntimeError('Benchmark interrupted or resource/time limit reached')
            time.sleep(2)
        code=p.wait()
    wall=time.monotonic()-start;log=(folder/'run.log').read_text();out=(folder/(case+'.out')).read_text()
    assert code==0 and log.count('The calculation was normally finished.')==1
    assert re.search(r'number of threads.*?is 2',log,re.I)
    phases=re.findall(r'^\s*([^=\n]+?)\s*=\s+\d+\s+([\d.]+)\s+\d+\s+([\d.]+)\s*$',log,re.M)
    times={name.strip():dict(min=float(lo),max=float(hi)) for name,lo,hi in phases}
    energy,atoms,forces,grids=read_output(folder/(case+'.out'))
    scf=re.findall(r'^\s*SCF=\s*(\d+)\s+NormRD=',out,re.M)
    result.update(status='completed',returncode=code,wall_seconds=wall,total_seconds=times['Total Computational Time']['max'],
                  phases_seconds=times,energy_hartree=energy,atoms=atoms,forces=forces,grids=grids,scf_steps=len(scf),
                  scf_converged_md=len(re.findall(r'The SCF was achieved at MD=',log)),
                  scf_unconverged_md=len(re.findall(r'The SCF was not fully achieved at MD=',log)),
                  peak_RSS_gib=peak/2**30,min_available_gib=minimum/2**30,output_sha256=digest(folder/(case+'.out')),
                  end_time=time.strftime('%Y-%m-%d %H:%M:%S %z'))
    save(folder/'result.json',result)
    print('DONE',variant,case,result['total_seconds'],flush=True)
    return result
def main():
    save(A/'progress.json',dict(status='waiting_for_preflight'))
    while True:
        try:state=json.loads((A/'preflight-state.json').read_text())
        except (OSError,json.JSONDecodeError):time.sleep(2);continue
        if state.get('status')=='completed':break
        if STOP or (A/'STOP').exists():return
        if (A/'build-state.json').exists() and json.loads((A/'build-state.json').read_text()).get('status')=='failed':raise RuntimeError('Build failed')
        time.sleep(2)
    frozen=dict(intel=digest(binary('intel')),upstream=digest(binary('upstream')))
    save(A/'measured-binaries.json',frozen)
    results=[]
    for case in ['sidia333_nc_cluster','cghelgeo']:
        for variant in ['intel','upstream']:
            assert digest(binary(variant))==frozen[variant]
            results.append(run_one(variant,case));save(A/'results.json',results)
    save(A/'progress.json',dict(status='completed',runs=4))
if __name__=='__main__':main()
