"""Isolate inherited FP32 projector-product rounding without changing the tolerance.

Run only after the four-mode campaign and its private MPS service have ended.
Dense eigensolvers/GEMMs and grid/orbital arithmetic use native FP64. By
default scf.fp32.enable remains on to preserve legacy projector products;
--full-fp64 changes it to off for the matched-seed paired control.
Both controls use the same converged default-GPU density and input geometry.
"""
import json,re,shutil,subprocess,sys,time
from pathlib import Path
import run as base
A=base.A
FULL='--full-fp64' in sys.argv
NAME='gpu_full_fp64_same_seed' if FULL else 'gpu_native_legacy_products'
FOLDER=A/'runs'/NAME
MPS_ROOT=A/'control-mps'/NAME
MPS_PIPE=A/('mps-full-pipe' if FULL else 'mps-legacy-pipe')
def set_input(text,key,value):
    pattern=r'(?im)^\s*'+re.escape(key)+r'\s+[^\n]*$'
    return re.sub(pattern,key+' '+value,text) if re.search(pattern,text) else text+'\n'+key+' '+value+'\n'
def main():
    assert json.loads((A/'progress.json').read_text())==dict(status='completed',runs=4)
    assert json.loads((A/'mps-stop.json').read_text())['returncode']==0
    seed_result=json.loads((A/'runs/gpu_default/result.json').read_text())
    assert seed_result['status']=='completed' and seed_result['converged']
    assert not FOLDER.exists() and not MPS_ROOT.exists() and not MPS_PIPE.exists()
    FOLDER.mkdir();MPS_ROOT.mkdir(parents=True)
    seed=A/'runs/gpu_default/cghelgeo_rst'
    seed_files=[dict(path=str(p.relative_to(seed)),sha256=base.sha(p),bytes=p.stat().st_size)
                for p in sorted(seed.rglob('*')) if p.is_file()]
    shutil.copytree(seed,FOLDER/'cghelgeo_rst')
    text=base.input_text('gpu_default')
    for key,value in [('scf.restart','on'),('scf.fp32.enable','off' if FULL else 'on'),('scf.gemmul8.enable','off')]:
        text=set_input(text,key,value)
    (FOLDER/'input.dat').write_text(text)
    env=base.environment('gpu_native_fp64')
    env.update(OPENMX_EIGEN_REFINE='0',OPENMX_GRID_PRECISION='fp64',OPENMX_ORBS_EVAL_PRECISION='fp64',
      CUDA_MPS_PIPE_DIRECTORY=str(MPS_PIPE),CUDA_MPS_LOG_DIRECTORY=str(MPS_ROOT/'log'))
    assert len(str(MPS_PIPE/'control'))<108
    MPS_PIPE.mkdir();(MPS_ROOT/'log').mkdir()
    def control(command):
        return subprocess.run(['nvidia-cuda-mps-control'],input=command+'\n',capture_output=True,text=True,env=env,timeout=20)
    command=[str(base.FORK_MPI),'--bind-to','none','-np','18',sys.executable,str(A/'rank.py'),str(A/'bin/openmx'),'input.dat','-nt','2']
    record=dict(variant=NAME,criterion=1e-10,status='running',command=command,
      input_sha256=base.sha(FOLDER/'input.dat'),binary_sha256=base.sha(A/'bin/openmx'),
      environment={k:v for k,v in env.items() if k.startswith(('OPENMX_','OMX_','OMP_','MKL_','OPENBLAS_','BLIS_','CUDA_'))},
      initialization='same-geometry converged default-GPU density/Hamiltonian; scf.restart on',
      seed_variant='gpu_default',seed_SCF_steps=seed_result['scf_steps'],restart_seed_files=seed_files,
      purpose=('Matched-seed full-FP64 arithmetic: promote inherited projector products as well.' if FULL else
               'Native FP64 dense/grid/orbital arithmetic with inherited FP32 projector-product rounding retained.'),
      start_time=time.strftime('%Y-%m-%d %H:%M:%S %z'))
    base.save(FOLDER/'result.json',record)
    r=subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,capture_output=True,text=True,timeout=20)
    base.save(MPS_ROOT/'start.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));r.check_returncode()
    start=time.monotonic();peak=0;minimum=base.available_bytes();samples=[];next_snapshot=0;p=None
    try:
        with (FOLDER/'run.log').open('w') as log:
            p=subprocess.Popen(command,cwd=FOLDER,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while p.poll() is None:
                elapsed=time.monotonic()-start;rss=base.group_rss(p.pid);available=base.available_bytes()
                peak=max(peak,rss);minimum=min(minimum,available)
                base.save(A/'control-progress.json',dict(status='running',variant=NAME,pid=p.pid,
                  elapsed_seconds=round(elapsed,1),rss_peak_gib=round(peak/2**30,3),available_gib=round(available/2**30,3)))
                if elapsed>=next_snapshot:
                    base.snapshot_ranks(FOLDER);samples.append(dict(seconds=elapsed,rss_gib=rss/2**30,available_gib=available/2**30))
                    base.save(FOLDER/'resources.json',samples);next_snapshot=elapsed+30
                    m=control('get_server_list');base.save(MPS_ROOT/'server.json',dict(returncode=m.returncode,stdout=m.stdout,stderr=m.stderr))
                if base.STOP or (A/'STOP').exists() or available<8*2**30 or elapsed>14400:
                    base.stop_job(p);raise RuntimeError('Control stopped or resource/time limit reached')
                time.sleep(2)
            code=p.wait()
        log=(FOLDER/'run.log').read_text()
        assert code==0 and log.count('The calculation was normally finished.')==1
        assert log.count('The SCF was achieved at MD=')==1 and 'The SCF was not fully achieved' not in log
        assert '<Restart>  Found restart files' in log and '<Restart>  Could not find restart files' not in log
        energy,atoms,forces,grid=base.read_output(FOLDER/'cghelgeo.out')
        assert len(atoms)==650 and len(forces)==1950
        phases={name.strip():dict(min=float(lo),max=float(hi)) for name,lo,hi in re.findall(
          r'^\s*([^=\n]+?)\s*=\s+\d+\s+([\d.]+)\s+\d+\s+([\d.]+)\s*$',log,re.M)}
        record.update(status='completed',converged=True,returncode=code,end_time=time.strftime('%Y-%m-%d %H:%M:%S %z'),
          wall_seconds=time.monotonic()-start,total_seconds=phases['Total Computational Time']['max'],phases_seconds=phases,
          scf_steps=len(re.findall(r'MD=\s*1\s+SCF=\s*\d+',log)),energy_hartree=energy,atoms=atoms,
          forces_hartree_per_bohr=forces,FFT_grid=grid,peak_RSS_gib=peak/2**30,min_available_gib=minimum/2**30,
          output_sha256=base.sha(FOLDER/'cghelgeo.out'))
        base.save(FOLDER/'result.json',record);base.save(A/'control-progress.json',dict(status='completed',variant=NAME))
        print(json.dumps({k:record[k] for k in ['variant','status','scf_steps','energy_hartree','converged']},indent=2))
    except BaseException as exc:
        base.save(A/'control-failure.json',dict(error=str(exc)));raise
    finally:
        if p is not None and p.poll() is None:base.stop_job(p)
        r=control('quit');base.save(MPS_ROOT/'stop.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
if __name__=='__main__':main()
