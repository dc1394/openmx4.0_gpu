#!/usr/bin/env python3
"""One complete warmup, then one timed sidia333 run, as requested."""
import ctypes
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path('/home/ec2-user/openmx-build')
PREFIX=Path('/home/ec2-user/.local/opt/openmx-4.0.1-sve256')
BINARY=PREFIX/'bin/openmx'
BLAS=Path('/home/ec2-user/.local/opt/openblas-0.3.34-sve256/lib/libopenblas.so')
SERIES=ROOT/'benchmarks/sidia333_nc_cluster_32mpi_1thread_20261110'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

def rank():
 affinity=sorted(os.sched_getaffinity(0))
 vl=ctypes.CDLL(None).prctl(51,0,0,0,0)
 lib=ctypes.CDLL(str(BLAS))
 lib.openblas_get_config.restype=ctypes.c_char_p
 lib.openblas_get_corename.restype=ctypes.c_char_p
 class DlInfo(ctypes.Structure):
  _fields_=[('filename',ctypes.c_char_p),('base',ctypes.c_void_p),
   ('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
 owner=DlInfo()
 ctypes.CDLL(None).dladdr(ctypes.cast(lib.zgemm_,ctypes.c_void_p),ctypes.byref(owner))
 record=dict(rank=int(os.environ['OMPI_COMM_WORLD_RANK']),pid=os.getpid(),
  affinity=affinity,sve_bits=(vl&65535)*8,blas_config=lib.openblas_get_config().decode(),
  blas_core=lib.openblas_get_corename().decode(),blas_threads=lib.openblas_get_num_threads(),
  blas_parallel=lib.openblas_get_parallel(),zgemm_library=str(Path(owner.filename.decode()).resolve()))
 if not (vl>=0 and record['sve_bits']==256 and len(affinity)==1 and
  record['blas_core']=='NEOVERSEV1' and record['blas_threads']==1 and
  record['zgemm_library']==str(BLAS.resolve())):
  raise RuntimeError(record)
 os.sched_setaffinity(0,affinity)
 record['affinity_before_exec']=sorted(os.sched_getaffinity(0))
 Path(f'rank-{record["rank"]:02}.json').write_text(json.dumps(record,indent=2)+'\n')
 os.execv(str(BINARY),[str(BINARY),'input.dat','-nt','1'])

def run(name,purpose):
 folder=SERIES/name
 folder.mkdir(exist_ok=False)
 original=ROOT/'reference/sidia333_nc_cluster.dat'
 text=original.read_text()
 overrides={'DATA.PATH':str(PREFIX/'share/openmx/DFT_DATA19'),
  'level.of.fileout':'1','scf.eigen.lib':'elpa2','scf.restart':'off'}
 for key,value in overrides.items():
  pattern=r'^'+re.escape(key)+r'\s+.*$'
  if re.search(pattern,text,re.M|re.I):
   text=re.sub(pattern,key+' '+value,text,flags=re.M|re.I)
  else:text+='\n'+key+' '+value+'\n'
 (folder/'input.dat').write_text(text)
 env={k:v for k,v in os.environ.items() if not k.startswith(('OPENMX_','OMP_','OPENBLAS_','GOTO_','ELPA_','ARMPL_','LD_'))}
 env.update(OMP_NUM_THREADS='1',OMP_DYNAMIC='FALSE',OMP_MAX_ACTIVE_LEVELS='1',
  OMP_PROC_BIND='close',OMP_PLACES='cores',OPENBLAS_NUM_THREADS='1',GOTO_NUM_THREADS='1')
 command=['/usr/lib64/openmpi/bin/mpirun','-np','32','--bind-to','core',
  '--map-by','slot:PE=1','/usr/bin/python3',str(Path(__file__).resolve()),'--rank']
 linked=subprocess.check_output(['ldd',str(BINARY)],env=env,text=True)
 (folder/'linked-libraries.txt').write_text(linked)
 metadata=dict(command=command,purpose=purpose,started_utc=now(),
  environment={k:v for k,v in env.items() if k.startswith(('OMP_','OPENBLAS_','GOTO_'))},
  input_original_sha256=sha(original),input_effective_sha256=sha(folder/'input.dat'),
  overrides=overrides,ranks=32,openmp_threads=1,scf_steps=25,
  binary=str(BINARY),binary_sha256=sha(BINARY),blas_library=str(BLAS.resolve()),blas_sha256=sha(BLAS),
  build=json.loads((ROOT/'build-info.json').read_text()),
  timer='time.monotonic; mpirun start through exit; includes rank audit, MPI startup and output')
 (folder/'command.json').write_text(json.dumps(metadata,indent=2)+'\n')
 print(f'{now()} START {purpose}: {folder}',flush=True)
 start=time.monotonic()
 with (folder/'run.log').open('w') as log:
  process=subprocess.Popen(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
  (folder/'launcher.json').write_text(json.dumps(dict(pid=process.pid,started_utc=now()))+'\n')
  code=process.wait()
 wall=time.monotonic()-start
 log=(folder/'run.log').read_text(errors='replace')
 rank_data=[json.loads(p.read_text()) for p in sorted(folder.glob('rank-*.json'))]
 cores=[c for r in rank_data for c in r['affinity']]
 tail=log.rsplit('Computational Time (second)',1)[-1]
 pattern=r'^\s*(.+?)\s+=\s+(\d+)\s+([\d.eE+-]+)\s+(\d+)\s+([\d.eE+-]+)\s*$'
 phases={m[0].strip():float(m[4]) for m in re.findall(pattern,tail,re.M)}
 result=dict(purpose=purpose,returncode=code,wall_seconds=wall,finished_utc=now(),
  normal_finish='The calculation was normally finished.' in log,
  scf_steps=len(set(re.findall(r'\*+ MD=\s*(\d+)\s+SCF=\s*(\d+) \*+',log))),
  scf_not_converged='The SCF was not fully achieved' in log,
  rank_audit_passed=len(rank_data)==32 and len(set(cores))==32 and
   all(r['sve_bits']==256 and r['blas_threads']==1 and len(r['affinity'])==1 for r in rank_data),
  phases_rank_max_seconds=phases,log_sha256=sha(folder/'run.log'))
 result['completed']=code==0 and result['normal_finish'] and result['scf_steps']==25 and result['rank_audit_passed']
 (folder/'rank-audit.json').write_text(json.dumps(rank_data,indent=2)+'\n')
 (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2),flush=True)
 if not result['completed']:raise RuntimeError(f'{purpose} did not complete: {result}')
 return result

def main():
 SERIES.mkdir(parents=True,exist_ok=False)
 original=ROOT/'reference/sidia333_nc_cluster.dat'
 reference=json.loads((ROOT/'reference/own-final-01_command.json').read_text())
 assert sha(original)==reference['original_sha256']
 # Both are full, fresh calculations. The stopped validation is excluded.
 warmup=run('warmup-01','warmup')
 measured=run('measurement-01','measurement')
 a=(SERIES/'warmup-01/command.json').read_text()
 b=(SERIES/'measurement-01/command.json').read_text()
 wa=json.loads(a);me=json.loads(b)
 assert wa['binary_sha256']==me['binary_sha256'] and wa['input_effective_sha256']==me['input_effective_sha256']
 summary=dict(input='sidia333_nc_cluster.dat',ranks=32,openmp_threads=1,
  warmup_runs=1,measurement_runs=1,scf_steps_per_run=25,
  warmup_seconds=warmup['wall_seconds'],measured_seconds=measured['wall_seconds'],
  phases_rank_max_seconds=measured['phases_rank_max_seconds'],
  scope='Official OpenMX 4.0 + patch4.0.1; reference compiler/numerical-library conditions.',
  reference='https://github.com/dc1394/openmx4.0_gpu/tree/main/benchmarks/results/graviton3_sve256_20261105',
  stopped_validation_excluded=True,finished_utc=now())
 (SERIES/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines=['OpenMX 4.0 + patch4.0.1: sidia333_nc_cluster.dat',
  '32 MPI x 1 OpenMP; one full warmup, one full measurement; 25 SCF iterations each.',
  f'Warmup: {warmup["wall_seconds"]:.3f} seconds',f'Measurement: {measured["wall_seconds"]:.3f} seconds',
  'Elapsed time includes MPI startup and output. Each run used a fresh directory, restart off.',
  'Fixed 25-step SCF, criterion 1e-13, Si 216 atoms, NC PBE, 200 Ry, 140 x 140 x 140 grid.',
  'ELPA2; each rank pinned to a separate core; SVE256 and one-thread NEOVERSEV1 BLAS audited.',
  'The interrupted validation is excluded. No numerical regression tests were run in this series.',
  'The reference repository contains additional application SIMD implementations absent from official OpenMX.',
  '', 'Measured phase times (maximum over ranks, seconds; not additive):']
 lines.extend(f'{k}: {v:.3f}' for k,v in measured['phases_rank_max_seconds'].items())
 (SERIES/'summary.txt').write_text('\n'.join(lines)+'\n')
 print('BENCHMARK SERIES COMPLETE',flush=True)
 print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
 if '--rank' in sys.argv:rank()
 else:main()
