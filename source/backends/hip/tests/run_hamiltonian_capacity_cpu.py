#!/usr/bin/env python3
"""Check base assembly and sparse Hamiltonian quadrature across all tiers."""
import ast
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
def literal(path,name):
 tree=ast.parse((root/path).read_text())
 return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')==name for t in n.targets))
prefix=literal('tests/run_gpu_vector_cpu.py','prefix')
body=(root/'source/openmx_gpu_task.c').read_text().replace('#include "hip_runtime_compat.h"','').replace('#include "set_hip_default_device_from_local_rank.h"','')
source=(root/'source/Set_Hamiltonian.c').read_text()
base=source[source.index('typedef struct {int spins,nc,hub,hch,vna;'):source.index('static void Set_Hamiltonian_Base_OpenMP(int SCF_iter, double *****H0, double *****HNL, double *****H)\n{')]
pair=source[source.index('typedef struct {int n0,n1,points;}'):source.index('/* Enter once per quadrature phase')]
main=r'''
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
 printf("PASS: %d Hamiltonian cases (spin, VNA/Hubbard/core-hole, unequal/empty pairs, local/remote/CPU)\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-hamiltonian-') as folder:
 p=Path(folder);(p/'mpi.h').write_text('typedef int MPI_Comm;\n#define MPI_COMM_NULL (-1)\n')
 (p/'check.c').write_text(prefix+body+base+pair+main)
 subprocess.run(shlex.split(os.environ.get('CC','cc'))+['-std=c11','-O1','-g','-Wno-unknown-pragmas','-fsanitize=address,undefined','-fno-sanitize-recover=all','-I'+str(p),'-I'+str(root/'source'),str(p/'check.c'),'-lm','-o',str(p/'check')],check=True)
 subprocess.run([str(p/'check')],check=True,timeout=60)
