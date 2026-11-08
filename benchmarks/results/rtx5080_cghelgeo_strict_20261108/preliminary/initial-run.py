"""Four complete fresh cghelgeo calculations; strict convergence, no restarts.

This script accompanies the raw evidence. It uses the frozen executables,
libraries and data prepared in this directory. Timings are diagnostic only.
The source input, MPI launchers and library providers are recorded explicitly.
"""
import ctypes,hashlib,itertools,json,os,re,signal,subprocess,sys,time
from pathlib import Path
A=Path(__file__).resolve().parent
ROOT=Path('/home/dc1394/openmx4.0_gpu')
sys.path.insert(0,str(ROOT/'tools'))
from run_gpu_suites import available_bytes,group_rss,stop_job
from compare_gpu_suites import read_output
INTEL=Path('/home/dc1394/intel/oneapi')
MPI=INTEL/'mpi/2021.18';COMP=INTEL/'compiler/2026.0';MKL=INTEL/'mkl/2026.0'
FORK_MPI=Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.9/comm_libs/mpi/bin/mpirun')
VARIANTS=['upstream','cpu_avx512','gpu_native_fp64','gpu_default']
STOP=False;MPS=False
def save(path,value):
    q=path.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stop(sig,frame):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def clean_environment():
    prefixes=('OPENMX_','OMX_','GEMMUL8_','OMP_','KMP_','MKL_','I_MPI_','CUDA_','CUBLAS_','CUSOLVER_',
              'OMPI_','PMIX_','PRTE_','NVCOMPILER_','OPENBLAS_','BLIS_','ACC_')
    env={k:v for k,v in os.environ.items() if not k.startswith(prefixes) and k not in
         ['LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH']}
    env.update(OMP_NUM_THREADS='2',OMP_DYNAMIC='FALSE',OMP_MAX_ACTIVE_LEVELS='1',OMP_PLACES='threads',
      OMP_PROC_BIND='close',OMP_STACKSIZE='64M',OPENBLAS_NUM_THREADS='2',BLIS_NUM_THREADS='2',
      MKL_NUM_THREADS='2',MKL_DYNAMIC='FALSE')
    return env
def mps_environment():
    env=clean_environment();env.update(CUDA_MPS_PIPE_DIRECTORY=str(A/'mps-pipe'),CUDA_MPS_LOG_DIRECTORY=str(A/'mps-log'),CUDA_VISIBLE_DEVICES='0');return env
def mps_control(command):
    return subprocess.run(['nvidia-cuda-mps-control'],input=command+'\n',text=True,capture_output=True,env=mps_environment(),timeout=20)
def start_mps():
    global MPS
    if MPS:return
    for d in ['mps-pipe','mps-log']:(A/d).mkdir()
    r=subprocess.run(['nvidia-cuda-mps-control','-d'],env=mps_environment(),capture_output=True,text=True,timeout=20)
    save(A/'mps-start.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));r.check_returncode();MPS=True
def environment(variant):
    env=clean_environment()
    if variant=='upstream':
        masks=[format((1<<i)|(1<<(i+18)),'x') for i in range(18)]
        env.update(LD_LIBRARY_PATH=':'.join(map(str,[COMP/'lib',MPI/'lib',MKL/'lib'])),
          I_MPI_PIN='1',I_MPI_PIN_DOMAIN='['+','.join(masks)+']',I_MPI_DEBUG='4',MKL_THREADING_LAYER='INTEL',
          MKL_INTERFACE_LAYER='LP64',OPENMX_GPU='0',CUDA_VISIBLE_DEVICES='')
    elif variant=='cpu_avx512':env.update(OPENMX_GPU='0',OPENMX_HOST_SIMD='avx512')
    else:
        env.update(OPENMX_GPU='1',CUDA_VISIBLE_DEVICES='0',CUDA_MPS_PIPE_DIRECTORY=str(A/'mps-pipe'),
          CUDA_MPS_LOG_DIRECTORY=str(A/'mps-log'),OPENMX_CUSOLVER_VERBOSE='1',OPENMX_GEMMUL8_TIMING='1',
          OMX_PRECISION_AUDIT_LIBRARY=str(A/'cusolver_audit.so'))
        if variant=='gpu_native_fp64':env['OPENMX_CUSOLVER_EMULATION']='0'
    return env
def input_text(variant):
    changes={'DATA.PATH':str(A/'DFT_DATA19'),'scf.criterion':'1e-12','level.of.stdout':'2'}
    if variant in ['upstream','cpu_avx512']:changes['scf.eigen.lib']='elpa2'
    if variant=='gpu_native_fp64':changes.update({'scf.fp32.enable':'off','scf.gemmul8.enable':'off'})
    text=(A/'original-cghelgeo.dat').read_text()
    for key,value in changes.items():
        pattern=r'(?im)^\s*'+re.escape(key)+r'\s+[^\n]*$'
        if re.search(pattern,text):text=re.sub(pattern,key+' '+value,text)
        else:text+='\n'+key+' '+value+'\n'
    return text
def snapshot_ranks(folder):
    for p in folder.glob('rank-??.json'):
        d=json.loads(p.read_text());base=Path('/proc')/str(d['pid'])
        try:
            if Path(os.fsdecode((base/'cmdline').read_bytes().split(b'\0')[0])).name!='openmx':continue
            maps=(base/'maps').read_text();(folder/(p.stem+'-maps.txt')).write_text(maps)
            tasks=[]
            for task in (base/'task').iterdir():
                try:
                    s=(task/'status').read_text();tasks.append(dict(tid=int(task.name),cpus=re.search(r'^Cpus_allowed_list:\s*(.*)',s,re.M)[1]))
                except (OSError,TypeError):pass
            save(folder/(p.stem+'-threads.json'),tasks)
        except (OSError,TypeError,IndexError):pass
def run_one(variant):
    if variant.startswith('gpu'):start_mps()
    folder=A/'runs'/variant;folder.mkdir()
    text=input_text(variant);(folder/'input.dat').write_text(text)
    upstream=variant=='upstream';binary=A/('reference/openmx' if upstream else 'bin/openmx')
    env=environment(variant)
    command=([str(MPI/'bin/mpiexec'),'-np','18'] if upstream else [str(FORK_MPI),'--bind-to','none','-np','18'])
    command += [sys.executable,str(A/'rank.py'),str(binary),'input.dat','-nt','2']
    record=dict(variant=variant,command=command,input_sha256=sha(folder/'input.dat'),binary_sha256=sha(binary),
      environment={k:v for k,v in env.items() if k.startswith(('OPENMX_','OMX_','OMP_','MKL_','I_MPI_','OPENBLAS_','BLIS_','CUDA_'))},
      start_time=time.strftime('%Y-%m-%d %H:%M:%S %z'),status='running')
    save(folder/'result.json',record);print('START',variant,flush=True)
    start=time.monotonic();peak=0;next_snapshot=0;minimum=available_bytes();samples=[]
    with (folder/'run.log').open('w') as log:
        p=subprocess.Popen(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(A/'active-process.json',dict(pid=p.pid,variant=variant))
        while p.poll() is None:
            elapsed=time.monotonic()-start;available=available_bytes();rss=group_rss(p.pid)
            peak=max(peak,rss);minimum=min(minimum,available)
            save(A/'progress.json',dict(status='running',variant=variant,pid=p.pid,elapsed_seconds=round(elapsed,1),
              rss_peak_gib=round(peak/2**30,3),available_gib=round(available/2**30,3)))
            if elapsed>=next_snapshot:
                snapshot_ranks(folder);samples.append(dict(seconds=elapsed,rss_gib=rss/2**30,available_gib=available/2**30))
                save(folder/'resources.json',samples);next_snapshot=elapsed+30
                if MPS:
                    r=mps_control('get_server_list');save(A/'mps-server-list.json',dict(stdout=r.stdout,stderr=r.stderr,returncode=r.returncode))
            if STOP or (A/'STOP').exists() or available<8*2**30 or elapsed>14400:
                stop_job(p);raise RuntimeError('Run stopped or resource/time limit reached')
            time.sleep(2)
        code=p.wait()
    log=(folder/'run.log').read_text();wall=time.monotonic()-start
    assert code==0 and log.count('The calculation was normally finished.')==1,(variant,code)
    assert log.count('The SCF was achieved at MD=')==1 and 'The SCF was not fully achieved' not in log,variant
    energy,atoms,forces,grids=read_output(folder/'cghelgeo.out')
    assert len(atoms)==650 and len(forces)==1950
    ranks=[json.loads(p.read_text()) for p in folder.glob('rank-??.json')]
    assert len(ranks)==18 and all(d['cpus']==[d['rank'],d['rank']+18] for d in ranks)
    phases={n.strip():dict(min=float(lo),max=float(hi)) for n,lo,hi in re.findall(
      r'^\s*([^=\n]+?)\s*=\s+\d+\s+([\d.]+)\s+\d+\s+([\d.]+)\s*$',log,re.M)}
    record.update(status='completed',returncode=code,end_time=time.strftime('%Y-%m-%d %H:%M:%S %z'),
      wall_seconds=wall,total_seconds=phases['Total Computational Time']['max'],phases_seconds=phases,
      scf_steps=len(re.findall(r'MD=\s*1\s+SCF=\s*\d+',log)),converged=True,energy_hartree=energy,
      atoms=atoms,forces_hartree_per_bohr=forces,FFT_grid=grids,peak_RSS_gib=peak/2**30,
      min_available_gib=minimum/2**30,output_sha256=sha(folder/'cghelgeo.out'))
    save(folder/'result.json',record);print('DONE',variant,record['scf_steps'],energy,flush=True);return record
def main():
    assert json.loads((A/'build-status.json').read_text())['returncode']==0
    provenance=json.loads((A/'provenance.json').read_text())
    core=ctypes.CDLL(str(A/'lib/libopenmx_core.so'))
    core.openmx_core_host_simd_mode.argtypes=[ctypes.c_char_p];core.openmx_core_host_simd_mode.restype=ctypes.c_int
    core.openmx_core_host_simd_name.argtypes=[ctypes.c_int];core.openmx_core_host_simd_name.restype=ctypes.c_char_p
    name=core.openmx_core_host_simd_name(core.openmx_core_host_simd_mode(b'avx512')).decode()
    assert name.lower().replace('-','')=='avx512',name
    save(A/'preflight.json',dict(AVX512_available=True,selected_name=name,criterion=1e-12,MPI_ranks=18,OMP_threads=2))
    records=[]
    try:
        for variant in VARIANTS:
            expected=provenance['binaries']['upstream_sha256' if variant=='upstream' else 'fork_snapshot_sha256']
            assert sha(A/('reference/openmx' if variant=='upstream' else 'bin/openmx'))==expected
            records.append(run_one(variant));save(A/'results.json',records)
        save(A/'progress.json',dict(status='completed',runs=4))
    except BaseException as exc:
        save(A/'failure.json',dict(error=str(exc),completed=len(records)));raise
    finally:
        if MPS:
            r=mps_control('quit');save(A/'mps-stop.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
if __name__=='__main__':main()
