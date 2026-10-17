#!/usr/bin/env python3
"""Check production host-backed contractions and wire packets under sanitizers."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
code = (root / "source/openmx_gpu_vector.c").read_text()
prefix = r'''
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "openmx_gpu_workers.h"
#include "openmx_gpu_collective.h"
#include "openmx_gpu_vector.h"
enum {hipSuccess, hipMemcpyHostToDevice, hipMemcpyDeviceToHost};
static int gpu_rank_device_usable(void){return 1;}
static int workers=1, fail_alloc=0, local_fail=0, remote_fail=0, remote=0, fail_once=0, remote_calls=0;
static int hipMalloc(void **p,size_t n){if(fail_once){fail_once--;*p=NULL;return 1;}*p=fail_alloc?NULL:malloc(n?n:1);return !*p;}
static int hipFree(void *p){free(p);return 0;}
static int hipMemcpy(void *d,const void *s,size_t n,int kind){(void)kind;memcpy(d,s,n);return 0;}
int openmx_gpu_workers_capacity_enabled(void){return 1;}
int openmx_gpu_workers_capacity(OpenMXGpuWorkers *p,size_t n,int kind,OpenMXGpuWorkerCost cost,void *ctx){
 (void)kind;p->items=n;p->count=n<(size_t)workers?(int)n:workers;
 if(local_fail && !remote)return 0;
 for(;p->count>0;p->count--){
  int fit=1;for(int i=0;i<p->count;i++){size_t nb=cost(ctx,i,p->count);assert(nb>0);if(nb==SIZE_MAX)fit=0;}
  if(fit)return 1;
 }
 return 0;
}
int openmx_gpu_workers_run(const OpenMXGpuWorkers *p,int(*f)(void*,int,int),void *ctx,const char *name){
 (void)name;int ok=1;
 for(int phase=0;phase<4;phase++){
  if(!ok && phase!=OPENMX_GPU_WORKER_RELEASE)continue;
  for(int i=0;i<p->count;i++)if(!f(ctx,i,phase))ok=0;
 }
 return ok;
}
int openmx_gpu_collective_retry(MPI_Comm comm,const OpenMXGpuCollectiveOps *ops,void *ctx,size_t n,int done,int help,const char *label){
 (void)comm;(void)help;(void)label;
 if(done!=0)return done>0;
 remote_calls++;
 void *packet[3]={0},*result[3]={0};size_t nb[3],rb[3];int ok=1;
 remote=1;
 for(int i=0;i<3;i++){
  size_t lo=openmx_gpu_worker_cut(n,i,3),hi=openmx_gpu_worker_cut(n,i+1,3);
  int success=ops->pack(ctx,lo,hi,&packet[i],&nb[i]);
  if(success)success=ops->run(ctx,packet[i],nb[i],&result[i],&rb[i]);
  ok &= success;
 }
 remote=0;if(remote_fail)ok=0;
 if(ok)for(int i=0;i<3;i++)ops->commit(ctx,openmx_gpu_worker_cut(n,i,3),openmx_gpu_worker_cut(n,i+1,3),result[i],rb[i]);
 for(int i=0;i<3;i++){free(packet[i]);free(result[i]);}
 return ok;
}
'''
main = r'''
int main(void){
 enum {N=19,LD=23,C=5,P=15};
 double data[C*LD],weight[N],coeff[C],out[N+2],want[N];int left[P],right[P];
 for(int c=0;c<C;c++){
  coeff[c]=0.1*(c-2);
  for(int i=0;i<LD;i++)data[c*LD+i]=sin(c*2+i*.07);
 }
 for(int i=0;i<N;i++)weight[i]=1+i*.03;
 int p=0;for(int a=0;a<C;a++)for(int b=0;b<=a;b++){left[p]=a;right[p++]=b;}
 int cases=0;
 for(int mode=0;mode<2;mode++)for(int n=0;n<=N;n+=n?18:1){
  OpenMXGpuVector j={0};j.n=n;j.ld=LD;j.columns=C;j.pairs=P;j.mode=mode;
  j.data=data;j.weight=weight;j.coeff=coeff;j.left=left;j.right=right;j.output=out+1;
  int outputs=mode?n:P;
  for(int i=0;i<outputs;i++){
   want[i]=0;
   if(mode)for(int c=0;c<C;c++)want[i]+=coeff[c]*data[c*LD+i];
   else for(int k=0;k<n;k++)want[i]+=weight[k]*data[left[i]*LD+k]*data[right[i]*LD+k];
  }
  for(workers=1;workers<=4;workers++)for(int tier=0;tier<4;tier++){
   local_fail=tier>0;remote_fail=tier==2;fail_alloc=tier==3;
   for(int i=0;i<N+2;i++)out[i]=1234;
   openmx_gpu_vector_execute(0,&j,1,"test");
   for(int i=0;i<outputs;i++)assert(fabs(out[i+1]-want[i])<1e-12);
   assert(out[0]==1234 && out[outputs+1]==1234);cases++;
  }
 }
 VectorPacket bad={UINT64_MAX,0,INT_MAX,INT_MAX};void *result=NULL;size_t bytes;
 assert(!vector_remote(NULL,&bad,sizeof(bad),&result,&bytes));
 /* Planning succeeds, but the first actual allocation fails. The next
    local layout must complete before any remote request is issued. */
 OpenMXGpuVector retry={0};retry.n=N;retry.ld=LD;retry.columns=C;retry.mode=1;
 retry.data=data;retry.coeff=coeff;retry.output=out+1;
 workers=4;local_fail=remote_fail=fail_alloc=remote_calls=0;fail_once=1;
 assert(openmx_gpu_vector_execute(0,&retry,1,"late allocation"));
 assert(remote_calls==0 && fail_once==0);
 for(int i=0;i<N;i++){double v=0;for(int c=0;c<C;c++)v+=coeff[c]*data[c*LD+i];assert(fabs(out[i+1]-v)<1e-12);}
 printf("PASS: %d contraction cases (local, remote, CPU, allocation failure, padding and empty work)\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-vector-") as folder:
    p = Path(folder)
    (p / "mpi.h").write_text("typedef int MPI_Comm;\n")
    body = "\n".join(line for line in code.splitlines()
                     if line not in ('#include "hip_runtime_compat.h"', '#include <omp.h>', '#include "set_hip_default_device_from_local_rank.h"'))
    (p / "check.c").write_text(prefix + body + main)
    subprocess.run(shlex.split(os.environ.get("CC", "cc")) + [
        "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Wno-unknown-pragmas",
        "-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-I"+str(p),
        "-I"+str(root/"source"), str(p/"check.c"), "-lm", "-o", str(p/"check")], check=True)
    subprocess.run([str(p/"check")], check=True, timeout=45)
