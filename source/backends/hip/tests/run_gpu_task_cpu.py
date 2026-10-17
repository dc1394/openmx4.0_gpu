#!/usr/bin/env python3
"""Test the range-task allocator, slicing, packets and rollback with sanitizers."""
import ast
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

root=Path(__file__).resolve().parents[1]
# Use the same HIP worker simulator as the contraction test. Its MPI
# transport invokes production pack/run/commit, including failed pieces.
tree=ast.parse((root/"tests/run_gpu_vector_cpu.py").read_text())
prefix=next(ast.literal_eval(n.value) for n in tree.body
            if isinstance(n,ast.Assign) and any(getattr(t,"id","")=="prefix" for t in n.targets))
body=(root/"source/openmx_gpu_task.c").read_text().replace('#include "hip_runtime_compat.h"', '').replace('#include "set_hip_default_device_from_local_rank.h"','')
force=(root/"source/Force.c").read_text()
force=force[force.index("static void Force_Reduction_Kernel("):force.index("static size_t Force2_Kinetic_OpenACC(")]
body += "\nstatic MPI_Comm mpi_comm_level1=0;\n"+force
main=r'''
typedef struct {double factor;int step;} Params;
static void kernel(size_t first,size_t n,const void *params,void *const v[],int gpu){
 (void)gpu;const Params *p=params;const double *a=v[0];const float *b=v[1];
 const int *lookup=v[2];double *out=v[3],*sums=v[4],*scratch=v[5];sums[0]=sums[1]=0;
 for(size_t i=0;i<n;i++){
  scratch[i]=p->factor*a[i];
  out[2*i]=scratch[i]+lookup[(first+i)%p->step];
  out[2*i+1]=b[i]-2.0*a[i];sums[0]+=out[2*i];sums[1]+=out[2*i+1];
 }
}
int main(void){
 enum {N=19};double a[N],out[2*N+2],sums[2];float b[N];int lookup[3]={7,-1,5};
 for(int i=0;i<N;i++){a[i]=i*.25;b[i]=i*.5f+2;}
 Params params={1.25,3};int cases=0;
 for(int n=0;n<=N;n+=n?18:1)for(workers=1;workers<=4;workers++)for(int tier=0;tier<4;tier++){
  local_fail=tier>0;remote_fail=tier==2;fail_alloc=tier==3;
  for(int i=0;i<2*N+2;i++)out[i]=111;
  OpenMXGpuTask t={0};t.first=5;t.count=n;t.parameters=&params;t.parameter_bytes=sizeof(params);
  t.inputs=3;t.outputs=2;t.scratch=1;t.kernel=kernel;
  t.view[0]=(OpenMXGpuTaskView){a,0,sizeof(double)};
  t.view[1]=(OpenMXGpuTaskView){b,0,sizeof(float)};
  t.view[2]=(OpenMXGpuTaskView){lookup,sizeof(lookup),0};
  t.view[3]=(OpenMXGpuTaskView){out+1,0,2*sizeof(double)};
  t.view[4]=(OpenMXGpuTaskView){sums,sizeof(sums),0};
  t.view[5]=(OpenMXGpuTaskView){NULL,0,sizeof(double)};
  openmx_gpu_task_execute(0,&t,1,"test");
  double want[2]={0};
  for(int i=0;i<n;i++){
   double x=1.25*a[i]+lookup[(5+i)%3],y=b[i]-2*a[i];
   assert(out[1+2*i]==x && out[2+2*i]==y);want[0]+=x;want[1]+=y;
  }
  assert(out[0]==111 && out[2*n+1]==111 && sums[0]==want[0] && sums[1]==want[1]);cases++;
 }
 size_t offsets[10]={0,0,1,8,8,8,17,18,20,27};
 double weights[27],x[27],y[27],z[27],fx[12],fy[12],fz[12];
 for(int i=0;i<27;i++){weights[i]=.13*(i%5-2);x[i]=.2*i;y[i]=.1-i;z[i]=3+i*.7;}
 for(int atoms=0;atoms<=9;atoms+=atoms?8:1)for(workers=1;workers<=4;workers++)for(int tier=0;tier<4;tier++){
  local_fail=tier>0;remote_fail=tier==2;fail_alloc=tier==3;
  for(int i=0;i<12;i++)fx[i]=fy[i]=fz[i]=111;
  Force_openmp_reduce_xyz(atoms,offsets,offsets[atoms],weights,x,y,z,fx,fy,fz);
  assert(fx[0]==0 && fy[0]==0 && fz[0]==0);
  for(int a=0;a<atoms;a++){
   double sx=0,sy=0,sz=0;
   for(size_t k=offsets[a];k<offsets[a+1];k++){sx+=weights[k]*x[k];sy+=weights[k]*y[k];sz+=weights[k]*z[k];}
   assert(fabs(fx[a+1]-sx)<1e-12 && fabs(fy[a+1]-sy)<1e-12 && fabs(fz[a+1]-sz)<1e-12);
  }
  assert(fx[atoms+1]==111 && fy[atoms+1]==111 && fz[atoms+1]==111);cases++;
 }
 assert(task_add(SIZE_MAX-4,8)==SIZE_MAX);
 TaskPacket bad={0};bad.count=UINT64_MAX;bad.inputs=1;bad.outputs=1;bad.view[0].item_bytes=16;
 OpenMXGpuTask t={0};t.kernel=kernel;void *result=NULL;size_t bytes=0;
 assert(!task_remote(&t,&bad,sizeof(bad),&result,&bytes));
 OpenMXGpuTask retry={0};retry.first=5;retry.count=N;retry.parameters=&params;retry.parameter_bytes=sizeof(params);
 retry.inputs=3;retry.outputs=2;retry.scratch=1;retry.kernel=kernel;
 retry.view[0]=(OpenMXGpuTaskView){a,0,sizeof(double)};
 retry.view[1]=(OpenMXGpuTaskView){b,0,sizeof(float)};
 retry.view[2]=(OpenMXGpuTaskView){lookup,sizeof(lookup),0};
 retry.view[3]=(OpenMXGpuTaskView){out+1,0,2*sizeof(double)};
 retry.view[4]=(OpenMXGpuTaskView){sums,sizeof(sums),0};
 retry.view[5]=(OpenMXGpuTaskView){NULL,0,sizeof(double)};
 workers=4;local_fail=remote_fail=fail_alloc=remote_calls=0;fail_once=1;
 assert(openmx_gpu_task_execute(0,&retry,1,"late allocation"));
 assert(remote_calls==0 && fail_once==0);
 double sum0=0,sum1=0;
 for(int i=0;i<N;i++){double x=1.25*a[i]+lookup[(5+i)%3],y=b[i]-2*a[i];assert(out[1+2*i]==x && out[2+2*i]==y);sum0+=x;sum1+=y;}
 assert(sums[0]==sum0 && sums[1]==sum1);
 printf("PASS: %d range tasks (GPU shards, remote packets, CPU, rollback, reduction and guards)\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-task-") as folder:
    p=Path(folder);(p/"mpi.h").write_text("typedef int MPI_Comm;\n#define MPI_COMM_NULL (-1)\n")
    (p/"check.c").write_text(prefix+body+main)
    subprocess.run(shlex.split(os.environ.get("CC","cc"))+[
        "-std=c11","-O1","-g","-Wall","-Wextra","-Wno-unknown-pragmas",
        "-fsanitize=address,undefined","-fno-sanitize-recover=all","-I"+str(p),
        "-I"+str(root/"source"),str(p/"check.c"),"-lm","-o",str(p/"check")],check=True)
    subprocess.run([str(p/"check")],check=True,timeout=45)
