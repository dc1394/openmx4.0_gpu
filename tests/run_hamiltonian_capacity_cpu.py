#!/usr/bin/env python3
"""Check base assembly and sparse Hamiltonian quadrature across all tiers."""
import ast
import argparse
import os
from pathlib import Path
from task_worker_test_bridge import compile_task_test
import shlex
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--core-library',type=Path,default=os.environ.get('OPENMX_CORE_LIBRARY',root/'build/cuda/core/libopenmx_core.so'))
parser.add_argument('--backend',choices=['CUDA','HIP'],default='CUDA')
args=parser.parse_args()
library=args.core_library.resolve(strict=True)
def literal(path,name):
 tree=ast.parse((root/path).read_text())
 return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')==name for t in n.targets))
prefix=literal('tests/run_gpu_vector_cpu.py','prefix')
body=(root/'source/openmx_gpu_task.c').read_text().replace('#include <cuda_runtime.h>','').replace('#include "set_cuda_default_device_from_local_rank.h"','')
source=(root/'source/Set_Hamiltonian.c').read_text()
base=source[source.index('typedef struct {int spins,nc,hub,hch,vna;'):source.index('static void Set_Hamiltonian_Base_OpenACC(',source.index('typedef struct {int spins,nc,hub,hch,vna;'))]
if args.backend=='HIP':source=(root/'source/backends/hip/src/Set_Hamiltonian.c').read_text()
pair=source[source.index('typedef struct {int n0,n1,points;}'):source.index('/* Enter once per quadrature phase')]
# openmx_host_simd.c is not linked: a kernel that takes its packed host mode
# from the host SIMD policy runs once per grid SIMD mode this processor
# executes (the policy's choice when OPENMX_SETHAM_SIMD is set).
policy_kernel='OpenMX_HostSimdGridMode' in pair
policy=r'''
static const int harness_policy_kernel=%d;
static int harness_grid_mode=OPENMX_GRID_SCALAR;
static inline int OpenMX_HostSimdGridMode(int env_id){
  const char *value=openmx_env_at(env_id);
  return value!=NULL && value[0]!='\0' ? openmx_core_grid_simd(value) : harness_grid_mode;
}
''' % int(policy_kernel)
main=r'''
static int host_packed_calls;
int __real_openmx_core_grid_hamiltonian_packed(const OpenMXGridPackedHamiltonian*,int);
int __wrap_openmx_core_grid_hamiltonian_packed(const OpenMXGridPackedHamiltonian *p,int mode){
  host_packed_calls++;
  if(harness_policy_kernel && openmx_env_at(OPENMX_ENV_SETHAM_SIMD)==NULL)assert(mode==harness_grid_mode);
  return __real_openmx_core_grid_hamiltonian_packed(p,mode);
}
int main(void){
 enum{N=7,MAX=3};double a[N],b[N*4],c[N],d[N*4],e[N*4],out[N*4+2];int cases=0;
 for(int i=0;i<N;i++){a[i]=.17*i;c[i]=.21*(i+1);for(int s=0;s<4;s++){b[i*4+s]=.13*(s-i);d[i*4+s]=.02*(1+i+s);e[i*4+s]=.37*(1-i-s);}}
 for(int spins=1;spins<=4;spins*=2)for(int flags=0;flags<8;flags++)for(int tier=0;tier<4;tier++)for(workers=1;workers<=3;workers++){
  SetHBaseCapacityParameters p={spins,spins==4,flags&1,(flags>>1)&1,(flags>>2)&1,.7,1.3,-.5};
  local_fail=tier>0;remote_fail=tier==2;fail_alloc=tier==3;
  OpenMXGpuTask task={0};task.count=N;task.inputs=5;task.outputs=1;task.kernel=SetH_base_capacity_kernel;task.parameters=&p;task.parameter_bytes=sizeof(p);
  task.view[0]=(OpenMXGpuTaskView){a,0,8};task.view[1]=(OpenMXGpuTaskView){b,0,(size_t)spins*8};
  task.view[2]=(OpenMXGpuTaskView){c,8,p.vna?8:0};task.view[3]=(OpenMXGpuTaskView){d,8,p.hub?(size_t)spins*8:0};task.view[4]=(OpenMXGpuTaskView){e,8,p.hch?(size_t)spins*8:0};
  task.view[5]=(OpenMXGpuTaskView){out+1,0,(size_t)spins*8};out[0]=out[N*spins+1]=999;
  openmx_gpu_task_execute(0,&task,1,"base");
  for(int i=0;i<N;i++)for(int s=0;s<spins;s++){
   int at=i*spins+s;double expected=0;
   if(spins<4||s<2)expected=.7*a[i]+(p.vna?-.5*c[i]:0)+1.3*b[at];else if(s==2)expected=1.3*b[at];
   if(spins<4||s<3){if(p.hub)expected+=d[at];if(p.hch)expected+=e[at];}
   assert(fabs(out[1+at]-expected)<1e-13);
  }
  assert(out[0]==999 && out[N*spins+1]==999);cases++;
 }
 SetHCapacityPair pairs[N];size_t pa[N+1]={0},pb[N+1]={0},pw[N+1]={0};
 double left[200],right[200],weight[300],matrix[N*4*MAX*MAX+2];
 for(int i=0;i<200;i++){left[i]=.13*(i%11-4);right[i]=.07*(i%13-5);}for(int i=0;i<300;i++)weight[i]=.1*(i%19-8);
 int modes=0;
 for(int mode=OPENMX_GRID_SCALAR;mode<=(harness_policy_kernel?OPENMX_GRID_NEON:OPENMX_GRID_SCALAR);mode++){
 if(harness_policy_kernel && !openmx_core_grid_simd_width(mode))continue;
 harness_grid_mode=mode;modes++;
 for(int spins=1;spins<=4;spins*=2)for(int tier=0;tier<4;tier++)for(workers=1;workers<=4;workers++){
  for(int i=0;i<N;i++){pairs[i]=(SetHCapacityPair){1+i%3,1+(i+1)%3,i%4};pa[i+1]=pa[i]+pairs[i].points*pairs[i].n0;pb[i+1]=pb[i]+pairs[i].points*pairs[i].n1;pw[i+1]=pw[i]+pairs[i].points*spins;}
  local_fail=tier>0;remote_fail=tier==2;fail_alloc=tier==3;SetHCapacityParameters p={spins,MAX,.23};
  OpenMXGpuTask t={0};t.count=N;t.inputs=4;t.outputs=1;t.parameters=&p;t.parameter_bytes=sizeof(p);t.kernel=SetH_capacity_kernel;
  t.view[0]=(OpenMXGpuTaskView){pairs,0,sizeof(pairs[0])};t.view[1]=(OpenMXGpuTaskView){left,0,0,pa,8};t.view[2]=(OpenMXGpuTaskView){right,0,0,pb,8};t.view[3]=(OpenMXGpuTaskView){weight,0,0,pw,8};t.view[4]=(OpenMXGpuTaskView){matrix+1,0,(size_t)spins*MAX*MAX*8};
  matrix[0]=matrix[N*spins*MAX*MAX+1]=999;openmx_gpu_task_execute(0,&t,1,"quadrature");
  for(int pair=0;pair<N;pair++)for(int s=0;s<spins;s++)for(int i=0;i<MAX;i++)for(int j=0;j<MAX;j++){
   double expected=0;
   if(i<pairs[pair].n0 && j<pairs[pair].n1)for(int k=0;k<pairs[pair].points;k++)expected+=(.23*weight[pw[pair]+k*spins+s]*left[pa[pair]+k*pairs[pair].n0+i])*right[pb[pair]+k*pairs[pair].n1+j];
   assert(fabs(matrix[1+((pair*spins+s)*MAX+i)*MAX+j]-expected)<1e-13);
  }
  assert(matrix[0]==999 && matrix[N*spins*MAX*MAX+1]==999);cases++;
 }
 }
 assert(host_packed_calls>0 && modes>0);
 printf("PASS: %d Hamiltonian cases (spin, VNA/Hubbard/core-hole, unequal/empty pairs, local/remote/CPU); %d host SIMD calls in %d grid SIMD mode(s)\n",cases,host_packed_calls,modes);
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-hamiltonian-') as folder:
 p=Path(folder);(p/'mpi.h').write_text('typedef int MPI_Comm;\n#define MPI_COMM_NULL (-1)\n')
 (p/"cuda_runtime.h").write_text("/* Runtime mock is provided by the fixture prefix. */\n")
 (p/"openmx_gpu_device.h").write_text("/* Rank availability mock is provided by the fixture prefix. */\n")
 (p/'check.c').write_text('#include "openmx_core_grid.h"\n'+prefix+body+base+policy+pair+main)
 compile_task_test(shlex.split(os.environ.get('CC','cc'))+['-std=c11','-O1','-g','-Wno-unknown-pragmas','-fsanitize=address,undefined','-fno-sanitize-recover=all','-I'+str(p),'-I'+str(root/'source'),str(p/'check.c'),str(library),'-Wl,-rpath,'+str(library.parent),'-Wl,--wrap=openmx_core_grid_hamiltonian_packed','-lm','-o',str(p/'check')],check=True)
 subprocess.run([str(p/'check')],check=True,timeout=60)
